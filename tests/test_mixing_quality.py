"""Regression tests for the defects behind the "two voices" artifact.

The fixtures are band separated so leakage is measurable rather than subjective:

    original speaker  1200 Hz      dub            400 Hz
    music bed          110 Hz      real SFX      3000 Hz

What is covered:
  * run_mixing runs at all (it used to raise NameError on an unbound `chunks`)
  * the original speaker never returns inside a source speech window - not in a
    mid-line pause, not under a soft line, not under a failed segment
  * genuine non-speech audio outside the speech windows is still preserved
  * the bed is actually ducked behind the dub
  * a failed segment is reported in mix/qc.json instead of being hidden
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dubstudio.pipeline.mixing import run_mixing

SR = 48000
DURATION_S = 10.0
ORIG_F, DUB_F, BED_F, SFX_F = 1200.0, 400.0, 110.0, 3000.0

NORMAL = (0.60, 2.30)   # healthy dubbed line
PAUSE = (3.55, 3.75)    # dub silent here, the source keeps talking
FAILED = (4.70, 6.10)   # TTS produced nothing for this line
SOFT = (6.60, 8.30)     # soft dub, below the old 0.015 amplitude gate
SFX_WIN = (9.05, 9.35)  # real non-speech event, must survive
QUIET = (9.50, 9.90)    # no source speech, no dub: unducked bed reference

MAX_BLEED_DBFS = -45.0
MIN_BLEED_MARGIN_DB = 25.0
MIN_DUCK_DEPTH_DB = 8.0


def _band_rms(sig: np.ndarray, f0: float, t0: float, t1: float, bw: float = 80.0) -> float:
    """RMS of the f0 +- bw band over [t0, t1), via FFT bin masking."""
    x = sig.mean(axis=-1) if sig.ndim > 1 else sig
    x = x[int(t0 * SR):int(t1 * SR)].astype(np.float64)
    if len(x) < 512:
        return 0.0
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    band = np.zeros_like(spec)
    sel = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    band[sel] = spec[sel]
    return float(np.sqrt(np.mean(np.fft.irfft(band, n=len(x)) ** 2)))


def _db(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-9))


def _voiced(dur_s: float, carrier: float, amp: float, mod: float = 5.0,
            pause: tuple[float, float] | None = None) -> np.ndarray:
    t = np.arange(int(dur_s * SR), dtype=np.float32) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * mod * t)
    sig = amp * env * np.sin(2 * np.pi * carrier * t)
    if pause is not None:
        sig[int(pause[0] * SR):int(pause[1] * SR)] = 0.0
    return sig.astype(np.float32)


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    for sub in ("audio", "synth", "segments"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    n = int(DURATION_S * SR)
    t = np.arange(n, dtype=np.float32) / SR
    bed = np.column_stack([
        0.18 * np.sin(2 * np.pi * BED_F * t) + 0.10 * np.sin(2 * np.pi * 660.0 * t),
        0.18 * np.sin(2 * np.pi * BED_F * t + 0.4) + 0.10 * np.sin(2 * np.pi * 660.0 * t + 0.7),
    ]).astype(np.float32)
    sf.write(str(tmp_path / "audio" / "bed.wav"), bed, SR)

    specs = [
        # id, start, end, dub amplitude (0 => synthesis failed), dub-only pause
        ("seg0", 0.50, 2.40, 0.30, None),
        ("seg1", 2.60, 4.30, 0.30, (0.90, 1.20)),
        ("seg2", 4.60, 6.20, 0.00, None),
        ("seg3", 6.50, 8.40, 0.045, None),
    ]

    # The original speaker talks continuously through every window.
    voc = np.zeros(n, dtype=np.float32)
    for _sid, start, end, _amp, _pause in specs:
        voc[int(start * SR):int(end * SR)] = _voiced(end - start, ORIG_F, 0.40)
    chime = 0.25 * np.sin(2 * np.pi * SFX_F * np.arange(int(0.4 * SR), dtype=np.float32) / SR)
    voc[int(9.0 * SR):int(9.0 * SR) + len(chime)] = chime.astype(np.float32)
    sf.write(str(tmp_path / "audio" / "vocals.wav"), voc, SR)

    segments = []
    for sid, start, end, amp, pause in specs:
        record = {
            "segment_id": sid,
            "start": start,
            "end": end,
            "speaker_id": "S00",
            "target_duration_ms": int((end - start) * 1000),
        }
        if amp > 0:
            sf.write(str(tmp_path / "synth" / f"{sid}.wav"),
                     _voiced(end - start, DUB_F, amp, mod=4.0, pause=pause), SR)
            record["generated_wav"] = f"synth/{sid}.wav"
            record["status"] = "synthesized"
        else:
            record["status"] = "failed"
            record["error"] = "simulated TTS timeout"
        segments.append(record)

    (tmp_path / "segments" / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    return tmp_path


@pytest.fixture
def mix(job_dir: Path) -> np.ndarray:
    final_path = run_mixing(job_dir, duration_s=DURATION_S)
    assert final_path.exists()
    audio, sr = sf.read(str(final_path))
    assert sr == SR
    return audio


@pytest.mark.parametrize("label,window", [
    ("normal line", NORMAL),
    ("mid-line pause", PAUSE),
    ("failed segment", FAILED),
    ("soft line", SOFT),
])
def test_original_speaker_never_returns_inside_a_speech_window(mix, label, window):
    """The source voice must not be re-injected anywhere the source was speaking."""
    bleed = _db(_band_rms(mix, ORIG_F, *window))
    dub = _db(_band_rms(mix, DUB_F, *NORMAL))
    assert bleed <= MAX_BLEED_DBFS, f"original voice at {bleed:.1f} dBFS during {label}"
    assert dub - bleed >= MIN_BLEED_MARGIN_DB, (
        f"only {dub - bleed:.1f} dB between dub and original voice during {label}"
    )


def test_failed_segment_is_reported_not_papered_over(job_dir):
    run_mixing(job_dir, duration_s=DURATION_S)
    qc = json.loads((job_dir / "mix" / "qc.json").read_text(encoding="utf-8"))
    assert qc["failed_segments"] == ["seg2"]
    assert qc["segments_mixed"] == 3


def test_real_non_speech_audio_is_preserved(mix):
    """Chimes, laughter and reactions outside the speech windows must survive."""
    assert _db(_band_rms(mix, SFX_F, *SFX_WIN)) > -30.0


def test_bed_is_ducked_behind_the_dub(mix):
    bed_under_speech = _db(_band_rms(mix, BED_F, *NORMAL))
    bed_alone = _db(_band_rms(mix, BED_F, *QUIET))
    dub = _db(_band_rms(mix, DUB_F, *NORMAL))
    assert bed_alone - bed_under_speech >= MIN_DUCK_DEPTH_DB
    assert dub - bed_under_speech >= MIN_DUCK_DEPTH_DB


def test_mix_does_not_clip(mix):
    assert float(np.max(np.abs(mix))) <= 0.995
