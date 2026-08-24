from __future__ import annotations

import json
from pathlib import Path

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.factory import get_voice_engine


def _speaker_lookup(job_dir: Path) -> dict[str, dict]:
    path = job_dir / "voices" / "speaker_map.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["speaker_id"]: s for s in data.get("speakers", []) if s.get("speaker_id")}


def run_synthesis(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    engine = get_voice_engine(job.get("tts_engine"))
    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    lang = job.get("target_language") or "hi"
    speakers = _speaker_lookup(job_dir)

    for seg in segments:
        out = synth_dir / f"{seg['segment_id']}.wav"
        sid = seg.get("speaker_id") or "S00"
        sp = speakers.get(sid) or {}
        voice_id = seg.get("voice_id") or sp.get("voice_id") or sid
        voice_mode = seg.get("voice_mode") or sp.get("voice_mode") or "clone"
        design_prompt = seg.get("design_prompt") or sp.get("design_prompt")
        ref_text = sp.get("ref_text")  # optional transcript of ref clip

        ref = None
        # Prefer voices/<voice_id>/ref.wav, then voices/<sid>/ref.wav
        for candidate in (
            job_dir / "voices" / voice_id / "ref.wav",
            job_dir / "voices" / sid / "ref.wav",
        ):
            if candidate.exists():
                ref = candidate
                break

        result = engine.generate(
            SynthRequest(
                text=seg.get("translated_text") or seg.get("source_text") or "",
                language=lang,
                voice_id=voice_id,
                ref_wav=ref,
                voice_mode=voice_mode,
                ref_text=ref_text,
                instruct=design_prompt,
            ),
            out,
        )
        seg["generated_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = result.duration_ms
        seg["status"] = "synthesized"
        seg["tts_engine"] = result.engine
        seg["voice_mode"] = voice_mode

    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
