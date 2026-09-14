"""Source separation contract tests.

The bed carries music and SFX only. It is ducked under the dub but never muted,
so if the bed ever contains the source dialogue the viewer hears two voices.
A degraded job therefore gets a SILENT bed, never a copy of full.wav.
"""

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


def _assert_bed_is_silent(bed_path: Path, expected_seconds: float) -> None:
    data, sr = sf.read(str(bed_path))
    assert float(np.max(np.abs(data))) <= 1e-6, "bed must not contain the source dialogue"
    assert abs(len(data) / sr - expected_seconds) < 0.05


def test_skip_produces_reference_vocals_and_silent_bed(tmp_path: Path):
    audio = tmp_path / "audio"
    full = audio / "full.wav"
    _write_dummy_wav(full, 1.0)
    result = run_separation(tmp_path, skip=True)
    assert result["mode"] == "skip_copy"
    assert result["degraded"] is True
    assert result["bed_mode"] == "silent"
    # vocals stay a copy of the source: they are only a timing/emotion reference
    assert Path(result["vocals"]).stat().st_size == full.stat().st_size
    _assert_bed_is_silent(Path(result["bed"]), 1.0)


def test_no_skip_falls_back_when_demucs_errors(tmp_path: Path, monkeypatch):
    import subprocess as _sp

    import dubstudio.pipeline.separation as sep

    audio = tmp_path / "audio"
    full = audio / "full.wav"
    _write_dummy_wav(full, 0.5)

    def _boom(*a, **k):
        raise _sp.CalledProcessError(1, a[0] if a else "demucs", output=b"forced")

    monkeypatch.setattr(sep.subprocess, "run", _boom)
    monkeypatch.setattr(sep, "_demucs_device", lambda: "cpu")
    result = run_separation(tmp_path, skip=False)
    assert result["mode"] == "fallback_copy"
    assert result["degraded"] is True
    assert Path(result["vocals"]).exists()
    _assert_bed_is_silent(Path(result["bed"]), 0.5)


def test_bed_is_never_a_copy_of_the_source_mix(tmp_path: Path, monkeypatch):
    import subprocess as _sp

    import dubstudio.pipeline.separation as sep

    full = tmp_path / "audio" / "full.wav"
    _write_dummy_wav(full, 0.5)

    def _boom(*a, **k):
        raise _sp.CalledProcessError(1, "demucs", output=b"forced")

    monkeypatch.setattr(sep.subprocess, "run", _boom)
    monkeypatch.setattr(sep, "_demucs_device", lambda: "cpu")
    result = run_separation(tmp_path, skip=False)

    source, _ = sf.read(str(full))
    bed, _ = sf.read(str(result["bed"]))
    overlap = min(len(source), len(bed))
    assert not np.allclose(source[:overlap], bed[:overlap]), "bed must not duplicate full.wav"


def test_missing_full_raises(tmp_path: Path):
    (tmp_path / "audio").mkdir()
    try:
        run_separation(tmp_path, skip=True)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
