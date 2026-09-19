"""Enroll speaker voices: cut ref clips + attach voice_mode (clone | design | fixed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from dubstudio.util.paths import rel_posix


def run_enroll(job_dir: Path, speaker_overrides: dict | None = None, job: dict | None = None) -> dict:
    """Cut a short ref clip per speaker from vocals and intelligently match voices.

    speaker_overrides (from UI / overrides.json):
      { "S00": { "voice_mode": "clone"|"design"|"fixed", "voice_id": "...", "design_prompt": "...", "ref_text": "..." } }
    """
    segs_path = job_dir / "segments" / "segments.json"
    segments = json.loads(segs_path.read_text(encoding="utf-8"))
    vocals = job_dir / "audio" / "vocals.wav"
    if not vocals.exists():
        vocals = job_dir / "audio" / "full.wav"

    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        sid = seg.get("speaker_id") or "S00"
        by_speaker.setdefault(sid, []).append(seg)

    overrides = speaker_overrides or {}
    job_overrides_path = job_dir / "voices" / "overrides.json"
    if job_overrides_path.exists():
        try:
            overrides = {**json.loads(job_overrides_path.read_text(encoding="utf-8")), **overrides}
        except Exception:
            pass

    # 1. Cut reference audio per speaker
    speakers_raw = []
    for sid, segs in by_speaker.items():
        best = max(segs, key=lambda s: float(s["end"]) - float(s["start"]))
        start = float(best["start"])
        end = float(best["end"])
        dur = end - start
        if dur < 4.0:
            end = start + min(8.0, max(dur, 1.0))
        elif dur > 8.0:
            end = start + 8.0

        out_dir = job_dir / "voices" / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        ref = out_dir / "ref.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(vocals),
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-ac", "1", "-ar", "48000", str(ref),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        best_text = (best.get("source_text") or best.get("translated_text") or "").strip()
        speakers_raw.append({
            "speaker_id": sid,
            "ref_start": start,
            "ref_end": end,
            "ref_wav": rel_posix(ref, job_dir),
            "ref_text": best_text,
            "segment_count": len(segs),
            "segs": segs,
        })

    # 2. Determine TTS Engine
    tts_engine = "veena"
    if job and job.get("tts_engine"):
        tts_engine = job["tts_engine"]
    else:
        job_json_path = job_dir / "job.json"
        if job_json_path.exists():
            try:
                meta = json.loads(job_json_path.read_text(encoding="utf-8"))
                tts_engine = meta.get("tts_engine") or "veena"
            except Exception:
                pass

    # 3. Intelligent Voice Matching for each character
    from dubstudio.pipeline.voice_matcher import match_voices_for_job

    matched_voices = match_voices_for_job(
        speakers=[{"speaker_id": s["speaker_id"]} for s in speakers_raw],
        job_dir=job_dir,
        tts_engine=tts_engine,
        overrides=overrides,
    )

    # 4. Finalize speaker assignments and propagate consistently to segments
    final_speakers = []
    for s_info in speakers_raw:
        sid = s_info["speaker_id"]
        segs = s_info["segs"]
        ov = overrides.get(sid) or {}
        m = matched_voices.get(sid) or {}

        voice_id = ov.get("voice_id") or m.get("voice_id") or sid
        voice_mode = ov.get("voice_mode") or m.get("voice_mode") or "clone"
        design_prompt = ov.get("design_prompt")
        ref_text = ov.get("ref_text") or s_info["ref_text"] or None

        label = ov.get("label") or f"Speaker {sid[1:] if sid.startswith('S') else sid}"
        if m.get("persona"):
            label = f"{label} ({m['persona'].replace('_', ' ').title()})"

        spk_entry = {
            "speaker_id": sid,
            "label": label,
            "voice_id": voice_id,
            "voice_mode": voice_mode,
            "gender": m.get("gender"),
            "f0_median": m.get("f0_median"),
            "persona": m.get("persona"),
            "auto_confidence": m.get("confidence", 0.75),
            "auto_suggestion": {
                "voice_id": m.get("voice_id", voice_id),
                "voice_mode": m.get("voice_mode", voice_mode),
                "confidence": m.get("confidence", 0.75),
            },
            "design_prompt": design_prompt,
            "ref_text": ref_text,
            "ref_wav": s_info["ref_wav"],
            "ref_start": s_info["ref_start"],
            "ref_end": s_info["ref_end"],
            "segment_count": s_info["segment_count"],
        }
        final_speakers.append(spk_entry)

        # 100% guarantee that every segment of this speaker uses this exact resolved voice
        for seg in segs:
            seg["voice_id"] = voice_id
            seg["voice_mode"] = voice_mode

    segs_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    map_path = job_dir / "voices" / "speaker_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"speakers": final_speakers}
    map_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload

