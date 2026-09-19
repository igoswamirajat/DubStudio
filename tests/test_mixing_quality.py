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
        # Derive the length from the integer sample bounds: int(end*SR) -
        # int(start*SR) is not always int((end-start)*SR) (1.6*48000 rounds
        # down), and the one-sample mismatch used to raise ValueError here.
        i0, i1 = int(start * SR), int(end * SR)
        voc[i0:i1] = _voiced((i1 - i0) / SR, ORIG_F, 0.40)
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


# --- hybrid bed -------------------------------------------------------------
# Two-stem Demucs files anything voice-like under "vocals", so a chime, a laugh
# or a door slam can be deleted from the bed for good. The original full mix
# still has it, and the hybrid bed is what puts it back bit-exact outside the
# speech windows instead of leaning on the attenuated residual path.

HYBRID_SPEECH = (0.50, 2.40)
HYBRID_SFX = (9.05, 9.35)


@pytest.fixture
def hybrid_job(tmp_path: Path) -> Path:
    """A job whose separator ate a non-speech sound out of the bed."""
    for sub in ("audio", "synth", "segments"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    n = int(DURATION_S * SR)
    t = np.arange(n, dtype=np.float32) / SR
    sp0, sp1 = int(HYBRID_SPEECH[0] * SR), int(HYBRID_SPEECH[1] * SR)
    fx0, fx1 = int(HYBRID_SFX[0] * SR), int(HYBRID_SFX[1] * SR)

    voice = np.zeros(n, dtype=np.float32)
    voice[sp0:sp1] = _voiced((sp1 - sp0) / SR, ORIG_F, 0.30)

    chime = np.zeros(n, dtype=np.float32)
    chime[fx0:fx1] = (0.15 * np.sin(2 * np.pi * SFX_F * (np.arange(fx1 - fx0) / SR))).astype(np.float32)

    bed_l = 0.10 * np.sin(2 * np.pi * BED_F * t)
    bed_r = 0.10 * np.sin(2 * np.pi * BED_F * t + 0.4)

    # What the source actually contains.
    full = np.column_stack([bed_l + voice + chime, bed_r + voice + chime]).astype(np.float32)
    sf.write(str(tmp_path / "audio" / "full.wav"), full, SR)

    # What Demucs handed back: the chime went to "vocals" and is gone from here.
    separated = np.column_stack([bed_l, bed_r]).astype(np.float32)
    sf.write(str(tmp_path / "audio" / "bed.wav"), separated, SR)

    vocals = np.column_stack([voice + chime, voice + chime]).astype(np.float32)
    sf.write(str(tmp_path / "audio" / "vocals.wav"), vocals, SR)

    sf.write(str(tmp_path / "synth" / "seg0.wav"),
             _voiced((sp1 - sp0) / SR, DUB_F, 0.30, mod=4.0), SR)
    segments = [{
        "segment_id": "seg0",
        "start": HYBRID_SPEECH[0],
        "end": HYBRID_SPEECH[1],
        "speaker_id": "S00",
        "target_duration_ms": int((HYBRID_SPEECH[1] - HYBRID_SPEECH[0]) * 1000),
        "generated_wav": "synth/seg0.wav",
        "status": "synthesized",
    }]
    (tmp_path / "segments" / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    return tmp_path


def test_hybrid_bed_is_used_when_the_original_is_available(hybrid_job):
    run_mixing(hybrid_job, duration_s=DURATION_S)
    qc = json.loads((hybrid_job / "mix" / "qc.json").read_text(encoding="utf-8"))
    assert qc["bed_mode"] == "hybrid"


def test_hybrid_bed_restores_a_sound_the_separator_deleted(hybrid_job):
    """The chime the separator ate must come back at unity, not at -4 dB."""
    run_mixing(hybrid_job, duration_s=DURATION_S)
    mix, _ = sf.read(str(hybrid_job / "mix" / "final.wav"))
    full, _ = sf.read(str(hybrid_job / "audio" / "full.wav"))

    reference = _db(_band_rms(full, SFX_F, *HYBRID_SFX))
    got = _db(_band_rms(mix, SFX_F, *HYBRID_SFX))
    assert got > -30.0, f"chime was lost entirely ({got:.1f} dBFS)"
    assert abs(got - reference) <= 0.5, (
        f"chime came through at {got:.2f} dB, source had {reference:.2f} dB"
    )


def test_separated_bed_is_the_fallback_without_the_original(job_dir):
    """No full.wav (older jobs) must keep working on the plain separated bed."""
    run_mixing(job_dir, duration_s=DURATION_S)
    qc = json.loads((job_dir / "mix" / "qc.json").read_text(encoding="utf-8"))
    assert qc["bed_mode"] == "separated"


def test_hybrid_bed_does_not_double_the_non_speech_audio(hybrid_job):
    """Bed + residual would stack the same sound twice outside the speech."""
    run_mixing(hybrid_job, duration_s=DURATION_S)
    mix, _ = sf.read(str(hybrid_job / "mix" / "final.wav"))
    full, _ = sf.read(str(hybrid_job / "audio" / "full.wav"))
    # +6 dB would mean the original got added to itself.
    assert _db(_band_rms(mix, SFX_F, *HYBRID_SFX)) - _db(_band_rms(full, SFX_F, *HYBRID_SFX)) < 3.0
