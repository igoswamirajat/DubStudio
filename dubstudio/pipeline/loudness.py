"""Loudness, true peak and bandwidth measurement, plus the tools to act on them.

Everything here is numpy-only on purpose: the mix bus is already in memory as
float arrays, and adding scipy/pyloudnorm to the render path is not worth it.

Why this module exists:
  * nothing normalised the finished mix, so one job came out at -24 LUFS and
    another clipped between samples;
  * a badly enrolled voice could sit 10 dB under the others for a whole video;
  * a 24 kHz TTS voice under a 48 kHz music bed sounds like a phone call, and
    we never measured it, so we never knew.

Loudness follows ITU-R BS.1770-4: K-weighting, 400 ms blocks with 100 ms hop,
an absolute -70 LUFS gate and a -10 LU relative gate. The gate is the point -
an ungated measurement drops every time the speaker pauses, so pauses would
drag the master gain up.
"""
from __future__ import annotations

import numpy as np

# --- targets ----------------------------------------------------------------
MASTER_LUFS = -16.0             # stereo web/social delivery
MASTER_TRUE_PEAK_DBTP = -1.0
MASTER_GAIN_LIMITS = (-8.0, 12.0)   # never "fix" a broken render with 30 dB
SPEAKER_TRIM_LIMIT_DB = 4.0     # per-voice balance, not per-voice rewrite
SPEAKER_DEADBAND_DB = 1.0
BANDWIDTH_SLACK_HZ = 1500.0     # how far above the dub the bed may reach
SHELF_FREQ_HZ = 6000.0          # presence lift so the dub reads as "close"
SHELF_GAIN_DB = 2.5
SILENCE_LUFS = -120.0

# BS.1770-4 K-weighting, designed at 48 kHz.
_DESIGN_SR = 48000.0
_SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
_RLB_B = (1.0, -2.0, 1.0)
_RLB_A = (1.0, -1.99004745483398, 0.99007225036621)


def _as_channels(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a if a.shape[0] >= a.shape[1] else a.T
    raise ValueError("audio must be 1-D or 2-D")


def _biquad_magnitude(b, a, freqs: np.ndarray) -> np.ndarray:
    """Magnitude response of a 48 kHz biquad at the given frequencies (Hz)."""
    z = np.exp(-1j * 2.0 * np.pi * freqs / _DESIGN_SR)
    num = b[0] + b[1] * z + b[2] * z * z
    den = a[0] + a[1] * z + a[2] * z * z
    return np.abs(num / den)


def _fft_apply(x: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Zero-phase magnitude filtering. Phase does not matter for measurement."""
    if x.size < 8:
        return x.copy()
    return np.fft.irfft(np.fft.rfft(x) * mag, n=x.size)


def k_weight(x, sr: float) -> np.ndarray:
    ch = _as_channels(x)
    freqs = np.fft.rfftfreq(ch.shape[0], d=1.0 / float(sr))
    mag = _biquad_magnitude(_SHELF_B, _SHELF_A, freqs) * _biquad_magnitude(_RLB_B, _RLB_A, freqs)
    return np.stack([_fft_apply(ch[:, i], mag) for i in range(ch.shape[1])], axis=1)


def _blocks_to_lufs(power: np.ndarray) -> float:
    return float(-0.691 + 10.0 * np.log10(max(float(np.mean(power)), 1e-20)))


def lufs(x, sr: float, block_s: float = 0.400, hop_s: float = 0.100) -> float:
    """Gated programme loudness in LUFS. Silence returns SILENCE_LUFS."""
    ch = _as_channels(x)
    if ch.size == 0:
        return SILENCE_LUFS
    weighted = k_weight(ch, sr)
    block = max(1, int(round(block_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    n = weighted.shape[0]
    if n < block:
        power = np.array([float(np.sum(np.mean(weighted ** 2, axis=0)))])
    else:
        power = np.array([
            float(np.sum(np.mean(weighted[s:s + block] ** 2, axis=0)))
            for s in range(0, n - block + 1, hop)
        ])
    loud = -0.691 + 10.0 * np.log10(np.maximum(power, 1e-20))
    keep = loud > -70.0                      # absolute gate
    if not np.any(keep):
        return SILENCE_LUFS
    relative = _blocks_to_lufs(power[keep]) - 10.0   # relative gate
    gated = keep & (loud > relative)
    if not np.any(gated):
        gated = keep
    return max(_blocks_to_lufs(power[gated]), SILENCE_LUFS)


def _oversample(x: np.ndarray, factor: int) -> np.ndarray:
    if x.size == 0 or factor <= 1:
        return x
    spec = np.fft.rfft(x)
    n_out = x.size * factor
    padded = np.zeros(n_out // 2 + 1, dtype=complex)
    padded[:spec.size] = spec
    return np.fft.irfft(padded, n=n_out) * factor


def true_peak_dbtp(x, sr: float, oversample: int = 4) -> float:
    """Inter-sample peak. A sample peak of -0.1 dBFS can be +1 dBTP after DAC."""
    ch = _as_channels(x)
    if ch.size == 0:
        return SILENCE_LUFS
    peak = 0.0
    for i in range(ch.shape[1]):
        up = _oversample(ch[:, i], oversample)
        if up.size:
            peak = max(peak, float(np.max(np.abs(up))))
    if peak <= 1e-12:
        return SILENCE_LUFS
    return float(20.0 * np.log10(peak))


def dbfs_rms(x) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return SILENCE_LUFS
    rms = float(np.sqrt(np.mean(a ** 2)))
    return float(20.0 * np.log10(rms)) if rms > 1e-12 else SILENCE_LUFS


def spectral_top_hz(x, sr: float, drop_db: float = 45.0, smooth_hz: float = 500.0) -> float:
    """Highest frequency still within drop_db of the spectral peak.

    Averaged over the loud frames only, so silence and breaths do not decide
    the answer. This is the number that tells us a TTS voice is 24 kHz native
    while the bed is 48 kHz.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    if a.size == 0 or float(np.max(np.abs(a))) <= 1e-9:
        return 0.0
    win = 4096 if a.size >= 4096 else int(2 ** int(np.floor(np.log2(max(a.size, 8)))))
    hop = win // 2
    frames = [a[s:s + win] for s in range(0, max(1, a.size - win + 1), hop)]
    if not frames:
        frames = [np.pad(a, (0, win - a.size))]
    rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])
    loud = [f for f, r in zip(frames, rms) if r >= 0.1 * float(np.max(rms))]
    window = np.hanning(win)
    acc = np.zeros(win // 2 + 1)
    for f in loud:
        acc += np.abs(np.fft.rfft(f * window[:f.size], n=win)) ** 2
    acc /= max(len(loud), 1)
    freqs = np.fft.rfftfreq(win, d=1.0 / float(sr))
    span = max(1, int(round(smooth_hz / (freqs[1] - freqs[0]))))
    smooth = np.convolve(acc, np.ones(span) / span, mode="same")
    level = 10.0 * np.log10(np.maximum(smooth, 1e-20))
    above = np.nonzero(level >= float(np.max(level)) - drop_db)[0]
    return float(freqs[above[-1]]) if above.size else 0.0


def _taper_mask(freqs: np.ndarray, start_hz: float, end_hz: float,
                start_gain: float, end_gain: float) -> np.ndarray:
    mask = np.full(freqs.shape, end_gain)
    mask[freqs <= start_hz] = start_gain
    band = (freqs > start_hz) & (freqs < end_hz)
    if np.any(band) and end_hz > start_hz:
        t = (freqs[band] - start_hz) / (end_hz - start_hz)
        smooth = 0.5 - 0.5 * np.cos(np.pi * t)
        mask[band] = start_gain + (end_gain - start_gain) * smooth
    return mask


def _apply_mask(x, sr: float, mask_for) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    ch = _as_channels(a)
    freqs = np.fft.rfftfreq(ch.shape[0], d=1.0 / float(sr))
    mask = mask_for(freqs)
    out = np.stack([_fft_apply(ch[:, i], mask) for i in range(ch.shape[1])], axis=1)
    return out[:, 0] if a.ndim == 1 else out


def lowpass_to(x, sr: float, cutoff_hz: float, taper_hz: float = 1000.0) -> np.ndarray:
    """Gentle brick-free lowpass, used to stop the bed out-shining the dub."""
    if cutoff_hz <= 0 or cutoff_hz >= sr / 2.0:
        return np.asarray(x, dtype=np.float64).copy()
    return _apply_mask(x, sr, lambda f: _taper_mask(f, cutoff_hz, cutoff_hz + taper_hz, 1.0, 0.0))


def high_shelf(x, sr: float, freq_hz: float = SHELF_FREQ_HZ,
               gain_db: float = SHELF_GAIN_DB, taper_hz: float = 2000.0) -> np.ndarray:
    """Presence lift above freq_hz. Cheap way to make a dub sit in front."""
    gain = 10.0 ** (gain_db / 20.0)
    start = max(0.0, freq_hz - taper_hz)
    return _apply_mask(x, sr, lambda f: _taper_mask(f, start, freq_hz, 1.0, gain))


def soft_ceiling(x, ceiling: float = 10.0 ** (MASTER_TRUE_PEAK_DBTP / 20.0)) -> np.ndarray:
    """Round off anything near the ceiling; leave everything below untouched."""
    a = np.asarray(x, dtype=np.float64)
    knee = 0.7 * ceiling
    out = a.copy()
    mag = np.abs(a)
    hot = mag > knee
    if np.any(hot):
        head = ceiling - knee
        out[hot] = np.sign(a[hot]) * (knee + head * np.tanh((mag[hot] - knee) / head))
    return out


def master(x, sr: float, target_lufs: float = MASTER_LUFS,
           ceiling_dbtp: float = MASTER_TRUE_PEAK_DBTP,
           gain_limits: tuple = MASTER_GAIN_LIMITS):
    """Normalise a finished mix. One constant gain, so the balance survives."""
    a = np.asarray(x, dtype=np.float64)
    input_lufs = lufs(a, sr)
    info = {
        "input_lufs": round(input_lufs, 2),
        "output_lufs": round(input_lufs, 2),
        "gain_db": 0.0,
        "true_peak_dbtp": round(true_peak_dbtp(a, sr), 2),
        "limited": False,
        "target_lufs": target_lufs,
    }
    if a.size == 0 or input_lufs <= SILENCE_LUFS + 1e-9:
        return a.copy(), info

    gain_db = float(np.clip(target_lufs - input_lufs, gain_limits[0], gain_limits[1]))
    out = a * (10.0 ** (gain_db / 20.0))

    peak = true_peak_dbtp(out, sr)
    limited = False
    if peak > ceiling_dbtp:
        out = out * (10.0 ** ((ceiling_dbtp - peak) / 20.0))
        limited = True
    out = soft_ceiling(out, 10.0 ** (ceiling_dbtp / 20.0))

    info.update({
        "gain_db": round(gain_db, 2),
        "output_lufs": round(lufs(out, sr), 2),
        "true_peak_dbtp": round(true_peak_dbtp(out, sr), 2),
        "limited": limited,
    })
    return out, info


def speaker_trims(groups: dict, sr: float, limit_db: float = SPEAKER_TRIM_LIMIT_DB,
                  deadband_db: float = SPEAKER_DEADBAND_DB) -> dict:
    """Per-speaker gains that pull every voice toward the median voice level.

    Clamped and dead-banded on purpose: a wrongly enrolled voice may dent the
    balance, never take over the mix. A single speaker is always left at 1.0,
    so single-voice jobs come out byte-identical.
    """
    levels: dict = {}
    for speaker, chunks in groups.items():
        arrays = [np.asarray(c, dtype=np.float64).ravel() for c in (chunks or []) if np.size(c)]
        if not arrays:
            levels[speaker] = None
            continue
        level = lufs(np.concatenate(arrays), sr)
        levels[speaker] = None if level <= SILENCE_LUFS + 1e-9 else level

    trims = {speaker: 1.0 for speaker in groups}
    live = [v for v in levels.values() if v is not None]
    if len(live) < 2:
        return trims

    target = float(np.median(live))
    for speaker, level in levels.items():
        if level is None:
            continue
        delta = target - level
        if abs(delta) < deadband_db:
            continue
        trims[speaker] = float(10.0 ** (float(np.clip(delta, -limit_db, limit_db)) / 20.0))
    return trims
