from __future__ import annotations

import json
from pathlib import Path

from dubstudio.engines.base import SynthRequest
from dubstudio.engines.factory import get_voice_engine


def run_synthesis(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8"))
    engine = get_voice_engine(job.get("tts_engine"))
    synth_dir = job_dir / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    lang = job.get("target_language") or "hi"
    for seg in segments:
        out = synth_dir / f"{seg['segment_id']}.wav"
        voice_id = seg.get("voice_id") or seg.get("speaker_id") or "S00"
        ref = job_dir / "voices" / voice_id / "ref.wav"
        result = engine.generate(
            SynthRequest(
                text=seg.get("translated_text") or seg.get("source_text") or "",
                language=lang,
                voice_id=voice_id,
                ref_wav=ref if ref.exists() else None,
            ),
            out,
        )
        seg["generated_wav"] = str(out.relative_to(job_dir))
        seg["generated_duration_ms"] = result.duration_ms
        seg["status"] = "synthesized"
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
