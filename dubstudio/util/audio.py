from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


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


def read_mono(path: Path, target_sr: int = 48000) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        duration = len(data) / sr
        n = int(duration * target_sr)
        x_old = np.linspace(0, 1, num=len(data), endpoint=False)
        x_new = np.linspace(0, 1, num=n, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = target_sr
    return data.astype(np.float32), sr


def write_wav(path: Path, data: np.ndarray, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate)
