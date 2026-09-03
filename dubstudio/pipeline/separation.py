from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.separation")


def _copy_stub(full: Path, vocals: Path, bed: Path, mode: str) -> dict:
    shutil.copy2(full, vocals)
    shutil.copy2(full, bed)
    return {"mode": mode, "vocals": str(vocals), "bed": str(bed)}


def _demucs_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def run_separation(job_dir: Path, *, skip: bool = False) -> dict:
    """Split full.wav into vocals (dialogue) and bed (music + SFX).

    Uses Demucs two-stem separation so the original music/SFX bed is preserved
    for the final mix. Falls back to copying full.wav on skip or failure.
    """
    audio = job_dir / "audio"
    full = audio / "full.wav"
    if not full.exists():
        raise FileNotFoundError(full)
    vocals = audio / "vocals.wav"
    bed = audio / "bed.wav"

    if skip:
        return _copy_stub(full, vocals, bed, "skip_copy")

    out_root = audio / "demucs"
    model = settings.demucs_model
    device = _demucs_device()
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-d", device,
        "--out", str(out_root),
        str(full),
    ]
    if settings.low_vram:
        # htdemucs is a Transformer model with a max segment of 7.8s; keep under it.
        cmd += ["--segment", "7"]
    try:
        log.info("Running Demucs (%s) on %s", model, device)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        out = getattr(exc, "output", b"")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "ignore")
        log.warning("Demucs failed (%s); falling back to copy. %s", exc, out[-500:] if out else "")
        return _copy_stub(full, vocals, bed, "fallback_copy")

    stem_dir = out_root / model / full.stem
    src_vocals = stem_dir / "vocals.wav"
    src_bed = stem_dir / "no_vocals.wav"
    if not src_vocals.exists() or not src_bed.exists():
        log.warning("Demucs output missing at %s; falling back to copy", stem_dir)
        return _copy_stub(full, vocals, bed, "fallback_copy")

    shutil.move(str(src_vocals), str(vocals))
    shutil.move(str(src_bed), str(bed))
    shutil.rmtree(out_root, ignore_errors=True)
    return {"mode": "demucs", "model": model, "device": device, "vocals": str(vocals), "bed": str(bed)}
