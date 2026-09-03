"""Source separation contract tests — bed must always exist for BGM/SFX preserve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from dubstudio.pipeline.separation import run_separation


def _write_dummy_wav(path: Path, seconds: float = 1.0, sr: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype(np.float32)
    sf.write(str(path), audio, sr)


def test_skip_copy_produces_vocals_and_bed(tmp_path: Path):
    audio = tmp_path / "audio"
    full = audio / "full.wav"
    _write_dummy_wav(full, 1.0)
    result = run_separation(tmp_path, skip=True)
    assert result["mode"] == "skip_copy"
    assert Path(result["vocals"]).exists()
    assert Path(result["bed"]).exists()
    assert Path(result["vocals"]).stat().st_size == full.stat().st_size
    assert Path(result["bed"]).stat().st_size == full.stat().st_size


def test_no_skip_falls_back_when_demucs_errors(tmp_path: Path, monkeypatch):
    import subprocess as _sp

    import dubstudio.pipeline.separation as sep

    audio = tmp_path / "audio"
    full = audio / "full.wav"
    _write_dummy_wav(full, 0.5)

    def _boom(*a, **k):
        raise _sp.CalledProcessError(1, a[0] if a else "demucs", output=b"forced")

    monkeypatch.setattr(sep.subprocess, "run", _boom)
    result = run_separation(tmp_path, skip=False)
    assert result["mode"] == "fallback_copy"
    assert Path(result["bed"]).exists()
    assert Path(result["vocals"]).exists()


def test_missing_full_raises(tmp_path: Path):
    (tmp_path / "audio").mkdir()
    try:
        run_separation(tmp_path, skip=True)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
