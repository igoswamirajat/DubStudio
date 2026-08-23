from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_enroll(job_dir: Path) -> dict:
    segs_path = job_dir / "segments" / "segments.json"
    segments = json.loads(segs_path.read_text(encoding="utf-8"))
    vocals = job_dir / "audio" / "vocals.wav"
    if not vocals.exists():
        vocals = job_dir / "audio" / "full.wav"
    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        sid = seg.get("speaker_id") or "S00"
        by_speaker.setdefault(sid, []).append(seg)
    speakers = []
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
        cmd = ["ffmpeg", "-y", "-i", str(vocals), "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-ac", "1", "-ar", "48000", str(ref)]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        speakers.append({"speaker_id": sid, "label": f"Speaker {sid[1:]}", "voice_id": sid, "ref_wav": str(ref.relative_to(job_dir)), "ref_start": start, "ref_end": end, "segment_count": len(segs)})
        for seg in segs:
            seg["voice_id"] = sid
    segs_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    map_path = job_dir / "voices" / "speaker_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"speakers": speakers}
    map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
