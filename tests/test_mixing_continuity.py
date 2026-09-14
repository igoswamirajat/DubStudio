"""Continuity regression tests: the dub must sound like one person talking.

Before this, every line was pasted at its ASR timestamp into an all-zero
dialogue bus. Between two lines the voice did not pause, it disappeared, and
the bed released to full gain inside the gap and then ducked again. Measured on
a real 48 s render: median onset step 24.8 dB, +18.2 dB bed swell in a gap,
4 holes totalling 9.4 s (19.3% of the video) shipped without a warning.

Fixtures are band separated so the assertions are objective:

    dub tone 400 Hz     bed 110 Hz     original speaker 1200 Hz

Weld/continuity tests use noise instead of a tone, because an equal-power
crossfade between two correlated sine waves can cancel at the midpoint and
would measure worse than it sounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dubstudio.pipeline import mixing as M
from dubstudio.pipeline.mixing import run_mixing

SR = 48000
DUB_F, BED_F, ORIG_F = 400.0, 110.0, 1200.0


# --- fixtures helpers -------------------------------------------------------
def _tone(dur_s: float, freq: float, amp: float, mod: float = 5.0) -> np.ndarray:
    t = np.arange(int(dur_s * SR), dtype=np.float32) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * mod * t)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(dur_s: float, amp: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(dur_s * SR)).astype(np.float32)
    x = np.convolve(x, np.ones(6, dtype=np.float32) / 6.0, mode="same")
    rms = float(np.sqrt(np.mean(x ** 2))) or 1.0
    return (x / rms * amp).astype(np.float32)


def _build(job_dir: Path, specs, duration_s: float, *, bed: bool = True,
           vocals: bool = True, voice: str = "tone") -> Path:
    """specs: list of (segment_id, start, end, dub_len_s or None).

    dub_len_s None means synthesis produced nothing for that line.
    """
    for sub in ("audio", "synth", "segments"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)
    n = int(duration_s * SR)

    if bed:
        t = np.arange(n, dtype=np.float32) / SR
        stereo = np.column_stack([
            0.18 * np.sin(2 * np.pi * BED_F * t),
            0.18 * np.sin(2 * np.pi * BED_F * t + 0.4),
        ]).astype(np.float32)
        sf.write(str(job_dir / "audio" / "bed.wav"), stereo, SR)

    if vocals:
        voc = np.zeros(n, dtype=np.float32)
        for _sid, start, end, _dub in specs:
            i0, i1 = int(start * SR), min(n, int(end * SR))
            if i1 > i0:
                voc[i0:i1] = _tone((i1 - i0) / SR, ORIG_F, 0.40)[:i1 - i0]
        sf.write(str(job_dir / "audio" / "vocals.wav"), voc, SR)

    rows = []
    for idx, (sid, start, end, dub_len) in enumerate(specs):
        row = {
            "segment_id": sid,
            "start": start,
            "end": end,
            "speaker_id": "S00",
            "target_duration_ms": int((end - start) * 1000),
        }
        if dub_len:
            wav = (_noise(dub_len, 0.20, seed=11 + idx) if voice == "noise"
                   else _tone(dub_len, DUB_F, 0.30, mod=0.0))
            sf.write(str(job_dir / "synth" / (sid + ".wav")), wav, SR)
            row["generated_wav"] = "synth/" + sid + ".wav"
            row["status"] = "synthesized"
        else:
            row["status"] = "failed"
            row["error"] = "simulated TTS failure"
        rows.append(row)

    (job_dir / "segments" / "segments.json").write_text(json.dumps(rows), encoding="utf-8")
    return job_dir


def _qc(job_dir: Path) -> dict:
    return json.loads((job_dir / "mix" / "qc.json").read_text(encoding="utf-8"))


def _read(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path))
    assert sr == SR
    return audio.mean(axis=-1) if audio.ndim > 1 else audio


def _band_rms(sig: np.ndarray, f0: float, t0: float, t1: float, bw: float = 80.0) -> float:
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


def _rms_env(x: np.ndarray, win_s: float = 0.020) -> np.ndarray:
    w = max(1, int(win_s * SR))
    frames = len(x) // w
    if frames == 0:
        return np.zeros(0, dtype=np.float64)
    block = x[:frames * w].reshape(frames, w).astype(np.float64)
    return np.sqrt((block ** 2).mean(axis=1))


def _frame(t: float, win_s: float = 0.020) -> int:
    return int(t / win_s)


# --- welding ----------------------------------------------------------------
def test_short_gap_between_lines_is_welded(tmp_path: Path):
    """A 140 ms ASR gap must become continuous speech, not a dropout."""
    job = _build(tmp_path, [("seg0", 0.50, 2.00, 1.50),
                            ("seg1", 2.14, 3.60, 1.46)],
                 duration_s=5.0, bed=False, vocals=False, voice="noise")
    run_mixing(job, duration_s=5.0)

    qc = _qc(job)
    assert qc["joins_welded"] == 1
    assert qc["max_join_gap_ms"] == 140

    env = _rms_env(_read(job / "mix" / "dialogue.wav"))
    speech = env[_frame(0.70):_frame(1.80)]
    join = env[_frame(1.90):_frame(2.20)]
    assert join.min() > 0.30 * float(np.median(speech)), (
        "dialogue still collapses at the join: %.5f vs %.5f"
        % (join.min(), np.median(speech))
    )


def test_a_line_is_never_pulled_more_than_the_lead_cap(tmp_path: Path):
    """Welding must not drift lip sync: the shift is capped per line."""
    gap_start = 2.29
    job = _build(tmp_path, [("seg0", 0.50, 2.00, 1.50),
                            ("seg1", gap_start, 3.80, 1.40)],
                 duration_s=5.0, bed=False, vocals=False, voice="noise")
    run_mixing(job, duration_s=5.0)

    dia = _read(job / "mix" / "dialogue.wav")
    peak = float(np.max(np.abs(dia)))
    after = dia[int(2.05 * SR):]
    onset_idx = int(np.argmax(np.abs(after) > 0.30 * peak))
    onset = 2.05 + onset_idx / SR

    expected = gap_start - M.MAX_LEAD_S
    assert expected - 0.01 <= onset <= expected + 0.04, (
        "second line landed at %.3f s, expected about %.3f s" % (onset, expected)
    )


def test_welding_does_not_clip(tmp_path: Path):
    specs = [("seg%d" % i, 0.50 + i * 1.05, 0.50 + i * 1.05 + 1.00, 1.05) for i in range(6)]
    job = _build(tmp_path, specs, duration_s=9.0, voice="noise")
    run_mixing(job, duration_s=9.0)
    qc = _qc(job)
    assert qc["joins_welded"] == 5
    assert qc["mix_peak"] <= 0.995
    assert qc["dialogue_peak"] <= 0.995


# --- the bed must not breathe in the holes ----------------------------------
def _hole_job(tmp_path: Path) -> Path:
    return _build(tmp_path, [("seg0", 0.50, 2.40, 1.90),
                             ("seg1", 2.60, 4.40, None),
                             ("seg2", 4.60, 6.40, 1.80)],
                  duration_s=8.0)


def test_bed_does_not_swell_inside_a_dub_hole(tmp_path: Path):
    job = _hole_job(tmp_path)
    mix = _read(run_mixing(job, duration_s=8.0))

    under_speech = _db(_band_rms(mix, BED_F, 1.00, 2.00))
    inside_hole = _db(_band_rms(mix, BED_F, 3.00, 4.00))
    unducked = _db(_band_rms(mix, BED_F, 7.20, 7.90))

    assert abs(inside_hole - under_speech) <= 2.0, (
        "bed jumps %.1f dB inside the hole" % (inside_hole - under_speech)
    )
    assert unducked - under_speech >= 8.0, "ducking no longer releases after speech"


def test_dialogue_bus_keeps_a_room_tone_floor(tmp_path: Path):
    job = _hole_job(tmp_path)
    run_mixing(job, duration_s=8.0)
    qc = _qc(job)
    dia = _read(job / "mix" / "dialogue.wav")

    hole = float(np.sqrt(np.mean(dia[int(3.0 * SR):int(4.0 * SR)] ** 2)))
    speech = float(np.sqrt(np.mean(dia[int(1.0 * SR):int(2.0 * SR)] ** 2)))

    assert qc["room_tone_rms"] > 0.0
    assert hole > 1e-5, "dialogue bus is still digital silence inside the run"
    assert _db(speech) - _db(hole) >= 20.0, "room tone is loud enough to be heard as noise"


def test_head_and_tail_stay_clean(tmp_path: Path):
    """Room tone belongs inside a talking run, not over the whole timeline."""
    job = _hole_job(tmp_path)
    run_mixing(job, duration_s=8.0)
    dia = _read(job / "mix" / "dialogue.wav")
    assert float(np.max(np.abs(dia[:int(0.30 * SR)]))) < 1e-6
    assert float(np.max(np.abs(dia[int(7.20 * SR):int(7.90 * SR)]))) < 1e-6


# --- coverage reporting -----------------------------------------------------
def test_coverage_and_holes_are_reported(tmp_path: Path):
    job = _hole_job(tmp_path)
    run_mixing(job, duration_s=8.0)
    qc = _qc(job)

    assert qc["failed_segments"] == ["seg1"]
    assert 0.60 < qc["coverage"] < 0.72
    assert qc["holes_over_1s"] == 1
    assert 1.7 <= qc["largest_hole_s"] <= 1.9
    assert qc["status"] == "degraded"
    assert qc["uncovered_speech_s"] > 1.5


def test_full_coverage_is_marked_ok(tmp_path: Path):
    job = _build(tmp_path, [("seg0", 0.50, 2.00, 1.50),
                            ("seg1", 2.14, 3.60, 1.70)],
                 duration_s=5.0, voice="noise")
    run_mixing(job, duration_s=5.0)
    qc = _qc(job)

    assert qc["failed_segments"] == []
    assert qc["coverage"] >= 0.99
    assert qc["holes"] == []
    assert qc["holes_over_1s"] == 0
    assert qc["status"] == "ok"


# --- the original speaker stays out unless explicitly asked for -------------
def test_hole_is_silent_by_default(tmp_path: Path):
    job = _hole_job(tmp_path)
    mix = _read(run_mixing(job, duration_s=8.0))
    assert _db(_band_rms(mix, ORIG_F, 3.00, 4.00)) <= -45.0
    assert _qc(job)["original_restored"] is False


def test_hole_can_be_filled_with_the_original_on_request(tmp_path: Path):
    job = _hole_job(tmp_path)
    mix = _read(run_mixing(job, duration_s=8.0, restore_original_on_failure=True))
    assert _db(_band_rms(mix, ORIG_F, 3.00, 4.00)) > -30.0
    assert _qc(job)["original_restored"] is True


# --- duck envelope unit -----------------------------------------------------
def test_duck_holds_across_a_600ms_pause():
    n = int(6.0 * SR)
    env = M._build_duck_envelope([(1.0, 2.0), (2.6, 3.6)], n, sr=SR,
                                 duck_gain=0.32, bridge_gap_s=M.DUCK_HOLD_GAP_S)
    assert round(float(env[int(2.3 * SR)]), 2) == 0.32
    assert round(float(env[int(5.0 * SR)]), 2) == 1.00


def test_interval_helpers():
    assert M._merge([(2.0, 3.0), (0.0, 1.0)]) == [(0.0, 1.0), (2.0, 3.0)]
    assert M._merge([(0.0, 1.0), (1.1, 2.0)], 0.2) == [(0.0, 2.0)]
    assert M._gaps_inside([(0.0, 4.0)], [(0.0, 1.0), (3.0, 4.0)]) == [(1.0, 3.0)]
    assert M._intersect([(0.0, 2.0)], [(1.0, 5.0)]) == [(1.0, 2.0)]
    assert round(M._total([(0.0, 1.5), (2.0, 2.5)]), 3) == 2.0
