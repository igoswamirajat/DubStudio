from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger("dubstudio.voice_profile")


@dataclass
class AcousticProfile:
    speaker_id: str
    f0_mean: float
    f0_median: float
    f0_std: float
    gender: str  # "male" | "female"
    spectral_ratio: float  # Low (80-250Hz) / Mid (250-1000Hz) ratio
    rms: float
    duration_s: float
    confidence: float  # 0.0 to 1.0


def extract_acoustic_profile(audio_path: Path | str, speaker_id: str = "S00") -> AcousticProfile:
    """Analyze speaker reference audio to produce a reliable acoustic profile.

    Extracts:
    - Fundamental Frequency (F0) using autocorrelation across speech frames
    - Gender classification (Male: F0 <= 160Hz, Female: F0 > 160Hz)
    - Spectral balance (ratio of low-frequency chest resonance to mid-frequency clarity)
    - RMS energy and speech confidence
    """
    path = Path(audio_path)
    if not path.is_file():
        log.warning("Audio path %s not found; returning default profile", audio_path)
        return AcousticProfile(
            speaker_id=speaker_id,
            f0_mean=130.0,
            f0_median=130.0,
            f0_std=15.0,
            gender="male",
            spectral_ratio=1.2,
            rms=0.08,
            duration_s=0.0,
            confidence=0.1,
        )

    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32)
        total_duration = len(data) / sr

        # Limit analysis to first 6 seconds for speed & consistency
        if len(data) > sr * 6:
            data = data[: sr * 6]

        # Adaptive noise gate (silence removal)
        rms_total = float(np.sqrt(np.mean(data**2)))
        thresh = max(0.01, rms_total * 0.3)
        voiced_samples = data[np.abs(data) > thresh]
        if len(voiced_samples) < sr * 0.5:
            # Fallback if audio is very quiet
            voiced_samples = data

        # 1. Pitch F0 Extraction using short-time autocorrelation
        frame_len = int(sr * 0.05)  # 50ms frames
        hop = int(sr * 0.025)       # 25ms hop
        min_f0 = 70.0               # Deepest male voice
        max_f0 = 350.0              # Highest female / expressive voice
        min_lag = max(1, int(sr / max_f0))
        max_lag = int(sr / min_f0)

        f0_candidates: list[float] = []

        for i in range(0, len(data) - frame_len, hop):
            frame = data[i : i + frame_len]
            frame = frame - np.mean(frame)
            std_val = float(np.std(frame))
            if std_val < thresh:
                continue

            # Normalized autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2 :]

            diff = np.diff(corr)
            valleys = np.where(diff > 0)[0]
            if len(valleys) > 0 and valleys[0] < max_lag:
                search_slice = corr[valleys[0] : max_lag]
                if len(search_slice) > 0:
                    peak_idx = valleys[0] + int(np.argmax(search_slice))
                    if peak_idx > min_lag:
                        f0 = sr / float(peak_idx)
                        if min_f0 <= f0 <= max_f0:
                            f0_candidates.append(f0)

        if f0_candidates:
            f0_mean = float(np.mean(f0_candidates))
            f0_median = float(np.median(f0_candidates))
            f0_std = float(np.std(f0_candidates))
            confidence = min(1.0, len(f0_candidates) / 40.0)
        else:
            f0_mean = 140.0
            f0_median = 140.0
            f0_std = 20.0
            confidence = 0.2

        # 2. Spectral Resonance (Chest resonance vs mid frequencies)
        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1.0 / sr)
        low_energy = float(np.mean(fft_vals[(freqs >= 80) & (freqs < 250)])) if np.any((freqs >= 80) & (freqs < 250)) else 1.0
        mid_energy = float(np.mean(fft_vals[(freqs >= 250) & (freqs < 1000)])) if np.any((freqs >= 250) & (freqs < 1000)) else 1.0
        spectral_ratio = low_energy / max(1e-5, mid_energy)

        # 3. Gender Classification:
        # F0 <= 160Hz indicates adult male; F0 > 160Hz indicates female
        if f0_median <= 155.0:
            gender = "male"
        elif f0_median >= 170.0:
            gender = "female"
        else:
            gender = "male" if spectral_ratio >= 1.05 else "female"

        log.info(
            "Acoustic profile [%s]: gender=%s, f0_med=%.1f Hz, f0_mean=%.1f Hz, spec_ratio=%.2f, conf=%.2f",
            speaker_id,
            gender,
            f0_median,
            f0_mean,
            spectral_ratio,
            confidence,
        )

        return AcousticProfile(
            speaker_id=speaker_id,
            f0_mean=f0_mean,
            f0_median=f0_median,
            f0_std=f0_std,
            gender=gender,
            spectral_ratio=spectral_ratio,
            rms=rms_total,
            duration_s=total_duration,
            confidence=confidence,
        )

    except Exception as exc:
        log.warning("Failed to extract acoustic profile: %s", exc)
        return AcousticProfile(
            speaker_id=speaker_id,
            f0_mean=135.0,
            f0_median=135.0,
            f0_std=20.0,
            gender="male",
            spectral_ratio=1.1,
            rms=0.08,
            duration_s=0.0,
            confidence=0.1,
        )
