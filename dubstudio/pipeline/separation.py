from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from dubstudio.settings import settings
from dubstudio.util.audio import duration_ms, write_wav

log = logging.getLogger("dubstudio.separation")


def _write_silent_bed(full: Path, bed: Path, sr: int = 48000) -> None:
    """Write a silent bed matching the length of full.wav.

    Copying full.wav into bed.wav puts the ORIGINAL DIALOGUE back underneath the
    dub - the bed is only ducked, never muted - which is one of the two causes
    of the "two voices" artifact. Losing the music on a degraded job is bad;
    shipping the source speaker under the dub is worse. So a degraded job gets a
    silent bed and is flagged instead.
    """
    n = max(1, int(round(duration_ms(full) / 1000.0 * sr)))
    write_wav(bed, np.zeros(n, dtype=np.float32), sr)


def _degraded_stems(full: Path, vocals: Path, bed: Path, mode: str) -> dict:
    """Fallback stems: vocals = source (used only as a timing/emotion reference),
    bed = silence. The bed must never carry the original dialogue."""
    shutil.copy2(full, vocals)
    _write_silent_bed(full, bed)
    log.error(
        "Separation degraded (%s): no music bed available. Using a silent bed so the "
        "original dialogue cannot leak under the dub.",
        mode,
    )
    return {
        "mode": mode,
        "vocals": str(vocals),
        "bed": str(bed),
        "degraded": True,
        "bed_mode": "silent",
    }


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
    for the final mix. On skip or failure the job is marked degraded and the bed
    is silent - it is never a copy of the source mix.
    """
    audio = job_dir / "audio"
    full = audio / "full.wav"
    if not full.exists():
        raise FileNotFoundError(full)
    vocals = audio / "vocals.wav"
    bed = audio / "bed.wav"

    if skip:
        return _degraded_stems(full, vocals, bed, "skip_copy")

    out_root = audio / "demucs"
    model = settings.demucs_model
    device = _demucs_device()

    def _run(dev: str) -> None:
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-n", model,
            "-d", dev,
            "--out", str(out_root),
            str(full),
        ]
        if settings.low_vram:
            # htdemucs is a Transformer model with a max segment of 7.8s; keep under it.
            cmd += ["--segment", "7"]
        log.info("Running Demucs (%s) on %s", model, dev)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    try:
        _run(device)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        out = getattr(exc, "output", b"")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "ignore")
        log.warning("Demucs failed on %s (%s). %s", device, exc, out[-500:] if out else "")
        if device == "cuda":
            # A GPU OOM should not cost the whole music bed; retry once on CPU.
            try:
                log.warning("Retrying Demucs on CPU")
                _run("cpu")
                device = "cpu"
            except (subprocess.CalledProcessError, FileNotFoundError) as exc2:
                log.warning("Demucs CPU retry also failed: %s", exc2)
                return _degraded_stems(full, vocals, bed, "fallback_copy")
        else:
            return _degraded_stems(full, vocals, bed, "fallback_copy")

    stem_dir = out_root / model / full.stem
    src_vocals = stem_dir / "vocals.wav"
    src_bed = stem_dir / "no_vocals.wav"
    if not src_vocals.exists() or not src_bed.exists():
        log.warning("Demucs output missing at %s", stem_dir)
        return _degraded_stems(full, vocals, bed, "fallback_copy")

    shutil.move(str(src_vocals), str(vocals))
    shutil.move(str(src_bed), str(bed))
    shutil.rmtree(out_root, ignore_errors=True)
    return {
        "mode": "demucs",
        "model": model,
        "device": device,
        "vocals": str(vocals),
        "bed": str(bed),
        "degraded": False,
        "bed_mode": "music",
    }
