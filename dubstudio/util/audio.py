from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int, axis: int = 0) -> np.ndarray:
    """High-fidelity audio resampling with polyphase anti-aliasing filter."""
    if orig_sr == target_sr or len(data) == 0:
        return data.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = resample_poly(data, up, down, axis=axis)
        return resampled.astype(np.float32)
    except Exception:
        # Fallback linear interpolation
        length = data.shape[axis]
        target_len = int(round(length * (target_sr / orig_sr)))
        x_old = np.linspace(0, 1, num=length, endpoint=False)
        x_new = np.linspace(0, 1, num=target_len, endpoint=False)
        if data.ndim == 1:
            return np.interp(x_new, x_old, data).astype(np.float32)
        out = np.empty((target_len, data.shape[1]), dtype=np.float32)
        for ch in range(data.shape[1]):
            out[:, ch] = np.interp(x_new, x_old, data[:, ch])
        return out


def apply_micro_fades(data: np.ndarray, fade_ms: float = 5.0, sample_rate: int = 48000) -> np.ndarray:
    """Apply raised-cosine micro fades at boundaries to eliminate digital clicks."""
    fade_len = int(round(fade_ms / 1000.0 * sample_rate))
    n = data.shape[0]
    if fade_len <= 0 or n < fade_len * 2:
        return data

    t = np.linspace(0, np.pi / 2, fade_len, dtype=np.float32)
    fade_in = (np.sin(t) ** 2).astype(np.float32)
    fade_out = (np.cos(t) ** 2).astype(np.float32)

    data = data.copy()
    if data.ndim == 1:
        data[:fade_len] *= fade_in
        data[-fade_len:] *= fade_out
    else:
        data[:fade_len, :] *= fade_in[:, None]
        data[-fade_len:, :] *= fade_out[:, None]
    return data


def write_silence(path: Path, duration_s: float, sample_rate: int = 48000, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_s * sample_rate))
    data = np.zeros((n, channels) if channels > 1 else n, dtype=np.float32)
    sf.write(str(path), data, sample_rate)


def write_tone(path: Path, duration_s: float, freq: float = 220.0, sample_rate: int = 48000, amplitude: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    data = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = min(n // 10, sample_rate // 50)
    if fade > 0:
        data[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        data[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    sf.write(str(path), data, sample_rate)


def duration_ms(path: Path) -> int:
    info = sf.info(str(path))
    return int(round(info.duration * 1000))


def read_audio(path: Path, target_sr: int = 48000, mono: bool = False) -> tuple[np.ndarray, int]:
    """Read audio from file, preserving stereo if mono=False, and resample cleanly."""
    data, sr = sf.read(str(path), always_2d=False)
    data = data.astype(np.float32)

    if mono and data.ndim > 1:
        data = data.mean(axis=-1)

    if sr != target_sr:
        data = resample_audio(data, sr, target_sr, axis=0)
        sr = target_sr

    return data, sr


def read_mono(path: Path, target_sr: int = 48000) -> tuple[np.ndarray, int]:
    return read_audio(path, target_sr=target_sr, mono=True)


def write_wav(path: Path, data: np.ndarray, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate)
