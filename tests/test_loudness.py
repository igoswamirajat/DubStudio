"""Loudness, true peak, bandwidth and per-speaker balance."""
from __future__ import annotations

import numpy as np

from dubstudio.pipeline import loudness as L

SR = 48000


def _sine(seconds, freq=1000.0, rms=0.1, phase=0.0):
    t = np.arange(int(SR * seconds)) / float(SR)
    return rms * np.sqrt(2.0) * np.sin(2 * np.pi * freq * t + phase)


def _noise(seconds, amp=0.05, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * seconds)) * amp


def _band_db(x, lo, hi):
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / SR)
    band = (freqs >= lo) & (freqs <= hi)
    return 10.0 * np.log10(max(float(np.sum(power[band])), 1e-20))


# --- measurement ------------------------------------------------------------
def test_a_1k_tone_measures_its_own_level():
    x = _sine(2.0, 1000.0, rms=0.1)
    assert abs(L.lufs(x, SR) - (-20.0)) < 0.5
    assert abs(L.dbfs_rms(x) - (-20.0)) < 0.1


def test_silence_is_floored_not_infinite():
    assert L.lufs(np.zeros(SR), SR) == L.SILENCE_LUFS
    assert L.spectral_top_hz(np.zeros(SR), SR) == 0.0


def test_the_gate_ignores_long_pauses():
    """Ungated loudness drops on every pause, which would push the master up."""
    speech = _sine(2.0, 1000.0, rms=0.1)
    with_gap = np.concatenate([speech, np.zeros(SR * 4), speech])
    assert abs(L.lufs(with_gap, SR) - L.lufs(speech, SR)) < 1.0


def test_true_peak_sees_between_the_samples():
    # a sine at sr/4 with a 45 degree phase never lands on its own crest,
    # so the reconstructed waveform peaks 3 dB above every stored sample
    x = _sine(1.0, SR / 4.0, rms=0.2, phase=np.pi / 4)
    sample_peak_db = 20.0 * np.log10(float(np.max(np.abs(x))))
    assert L.true_peak_dbtp(x, SR) > sample_peak_db + 2.0


# --- mastering --------------------------------------------------------------
def test_master_hits_the_loudness_target():
    out, info = L.master(_sine(4.0, 1000.0, rms=0.05), SR)
    assert info["gain_db"] > 8.0
    assert abs(info["output_lufs"] - L.MASTER_LUFS) < 1.0
    assert abs(L.lufs(out, SR) - L.MASTER_LUFS) < 1.0


def test_master_holds_the_true_peak_ceiling():
    out, info = L.master(_sine(2.0, 1000.0, rms=2.0), SR)
    assert info["limited"] is True
    assert L.true_peak_dbtp(out, SR) <= L.MASTER_TRUE_PEAK_DBTP + 0.3
    assert float(np.max(np.abs(out))) <= 10.0 ** (L.MASTER_TRUE_PEAK_DBTP / 20.0)


def test_master_is_one_constant_gain_so_the_balance_survives():
    x = _noise(2.0, amp=0.02)
    out, info = L.master(x, SR)
    assert info["limited"] is False
    live = np.abs(x) > 1e-4
    ratio = out[live] / x[live]
    assert np.allclose(ratio, ratio[0], rtol=1e-6)


def test_master_leaves_silence_alone():
    out, info = L.master(np.zeros(SR), SR)
    assert info["gain_db"] == 0.0
    assert not np.any(out)


def test_master_gain_is_bounded():
    _, info = L.master(_sine(2.0, 1000.0, rms=0.0005), SR)
    assert info["gain_db"] <= L.MASTER_GAIN_LIMITS[1]


# --- bandwidth --------------------------------------------------------------
def test_spectral_top_follows_a_lowpass():
    x = _noise(2.0)
    assert L.spectral_top_hz(x, SR) > 18000.0
    top = L.spectral_top_hz(L.lowpass_to(x, SR, 8000.0), SR)
    assert 7000.0 < top < 11000.0


def test_the_presence_shelf_lifts_only_the_top():
    x = _noise(2.0)
    y = L.high_shelf(x, SR)
    lift = _band_db(y, 8000, 16000) - _band_db(x, 8000, 16000)
    assert abs(lift - L.SHELF_GAIN_DB) < 0.6
    assert abs(_band_db(y, 200, 1000) - _band_db(x, 200, 1000)) < 0.3


# --- per-speaker balance ----------------------------------------------------
def test_a_single_speaker_is_never_trimmed():
    assert L.speaker_trims({"S00": [_sine(1.0, rms=0.2)]}, SR) == {"S00": 1.0}


def test_a_quiet_speaker_is_nudged_toward_the_median():
    trims = L.speaker_trims({"S00": [_sine(1.0, rms=0.1)], "S01": [_sine(1.0, rms=0.05)]}, SR)
    assert trims["S01"] > 1.0 > trims["S00"]


def test_a_wild_level_difference_is_clamped_not_matched():
    loud = _sine(1.0, rms=0.2)
    trims = L.speaker_trims({"S00": [loud], "S01": [loud * 0.02]}, SR)
    assert abs(20.0 * np.log10(trims["S01"]) - L.SPEAKER_TRIM_LIMIT_DB) < 0.01
    assert abs(20.0 * np.log10(trims["S00"]) + L.SPEAKER_TRIM_LIMIT_DB) < 0.01


def test_small_differences_stay_inside_the_deadband():
    trims = L.speaker_trims({"S00": [_sine(1.0, rms=0.1)], "S01": [_sine(1.0, rms=0.095)]}, SR)
    assert trims == {"S00": 1.0, "S01": 1.0}


def test_a_speaker_with_no_audio_is_left_at_unity():
    trims = L.speaker_trims(
        {"S00": [_sine(1.0, rms=0.1)], "S01": [], "S02": [_sine(1.0, rms=0.05)]}, SR)
    assert trims["S01"] == 1.0
