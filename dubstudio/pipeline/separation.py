from __future__ import annotations

import shutil
from pathlib import Path


def run_separation(job_dir: Path, *, skip: bool = True) -> dict:
    audio = job_dir / "audio"
    full = audio / "full.wav"
    if not full.exists():
        raise FileNotFoundError(full)
    vocals = audio / "vocals.wav"
    bed = audio / "bed.wav"
    shutil.copy2(full, vocals)
    shutil.copy2(full, bed)
    return {"mode": "skip_copy" if skip else "fallback_copy", "vocals": str(vocals), "bed": str(bed)}
