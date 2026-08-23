from __future__ import annotations

import json
from pathlib import Path

from dubstudio.util.ffmpeg import extract_wav, probe


def run_extract(job_dir: Path) -> dict:
    source_dir = job_dir / "source"
    originals = [p for p in source_dir.glob("*") if p.is_file() and p.name != "meta.json"]
    if not originals:
        raise FileNotFoundError("no source media in job")
    src = originals[0]
    meta = probe(src)
    duration = float(meta.get("format", {}).get("duration") or 0)
    (job_dir / "source" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    full = audio_dir / "full.wav"
    extract_wav(src, full, sample_rate=48000)
    return {
        "source_path": str(src.relative_to(job_dir)),
        "full_wav": str(full.relative_to(job_dir)),
        "duration_s": duration,
        "meta": meta,
    }


def run_copy_source_name(job_dir: Path) -> Path:
    source_dir = job_dir / "source"
    for p in source_dir.iterdir():
        if p.is_file() and p.name != "meta.json":
            return p
    raise FileNotFoundError("source missing")
