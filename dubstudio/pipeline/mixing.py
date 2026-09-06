from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dubstudio.util.audio import read_audio, read_mono, write_wav


def _build_duck_envelope(
    intervals: list[tuple[float, float]],
    n: int,
    sr: int = 48000,
    duck_gain: float = 0.22,
    attack_s: float = 0.15,
    release_s: float = 0.40,
    bridge_gap_s: float = 0.35,
) -> np.ndarray:
    """Build a broadcast-quality smooth ducking envelope.
    
    Bridges pauses shorter than bridge_gap_s to eliminate accordion pumping,
    and uses raised-cosine (S-curve) attack and release transitions.
    """
    envelope = np.ones(n, dtype=np.float32)
    if not intervals:
        return envelope

    # 1. Sort and merge overlapping or closely spaced intervals
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[list[float]] = []
    for start, end in sorted_intervals:
        if end <= start:
            continue
        if not merged:
            merged.append([start, end])
        else:
            prev = merged[-1]
            if start - prev[1] <= bridge_gap_s:
                prev[1] = max(prev[1], end)
            else:
                merged.append([start, end])

    # 2. Render smooth S-curve envelope
    att_samples = int(round(attack_s * sr))
    rel_samples = int(round(release_s * sr))

    for start_s, end_s in merged:
        i0 = int(round(start_s * sr))
        i1 = min(n, int(round(end_s * sr)))
        if i0 >= n or i1 <= 0:
            continue

        # Hold region
        envelope[max(0, i0):i1] = np.minimum(envelope[max(0, i0):i1], duck_gain)

        # Attack transition (ramp down from 1.0 to duck_gain)
        if att_samples > 0 and i0 > 0:
            a0 = max(0, i0 - att_samples)
            count = i0 - a0
            if count > 0:
                t = np.linspace(0, np.pi / 2, count, dtype=np.float32)
                curve = 1.0 - (1.0 - duck_gain) * (np.sin(t) ** 2)
                envelope[a0:i0] = np.minimum(envelope[a0:i0], curve)

        # Release transition (ramp up from duck_gain to 1.0)
        if rel_samples > 0 and i1 < n:
            r1 = min(n, i1 + rel_samples)
            count = r1 - i1
            if count > 0:
                t = np.linspace(0, np.pi / 2, count, dtype=np.float32)
                curve = duck_gain + (1.0 - duck_gain) * (np.sin(t) ** 2)
                envelope[i1:r1] = np.minimum(envelope[i1:r1], curve)

    return envelope


def _soft_limit(mix: np.ndarray, threshold: float = 0.88) -> np.ndarray:
    """Soft-knee peak limiter to prevent digital clipping without squash pumping."""
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak <= threshold:
        return mix

    max_val = 0.995
    headroom = max(1e-4, max_val - threshold)
    abs_mix = np.abs(mix)
    excess = np.maximum(0.0, abs_mix - threshold)
    compressed = threshold + headroom * np.tanh(excess / headroom)
    limited = np.sign(mix) * np.where(abs_mix > threshold, compressed, abs_mix)
    return limited.astype(np.float32)


def run_mixing(job_dir: Path, duration_s: float) -> Path:
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    sr = 48000
    n = max(1, int(duration_s * sr) + sr // 2)

    # 1. Read BGM / Ambient Bed (preserving stereo if present)
    bed_path = job_dir / "audio" / "bed.wav"
    if bed_path.exists():
        bed, _ = read_audio(bed_path, target_sr=sr, mono=False)
        if len(bed) < n:
            pad_shape = ((0, n - len(bed)), (0, 0)) if bed.ndim > 1 else (0, n - len(bed))
            bed = np.pad(bed, pad_shape)[:n]
        else:
            bed = bed[:n]
    else:
        bed = np.zeros(n, dtype=np.float32)

    is_stereo = bed.ndim > 1 and bed.shape[1] >= 2

    # 2. Place Dialogue Segments & Record Timing Intervals
    dialogue = np.zeros(n, dtype=np.float32)
    intervals: list[tuple[float, float]] = []

    for seg in segments:
        wav_rel = seg.get("fitted_wav") or seg.get("generated_wav")
        if not wav_rel:
            continue
        wav_path = job_dir / wav_rel
        if not wav_path.exists():
            continue

        audio, _ = read_mono(wav_path, target_sr=sr)
        start = float(seg.get("start", 0.0))
        i0 = int(round(start * sr))
        i1 = min(n, i0 + len(audio))
        if i0 >= n or i1 <= i0:
            continue

        dialogue[i0:i1] += audio[: i1 - i0]
        intervals.append((start, start + (len(audio) / sr)))

    # 3. Apply Subtle Dialogue High-Pass Filter (> 80 Hz) to eliminate low boominess
    if np.any(dialogue != 0.0):
        try:
            from scipy.signal import butter, lfilter

            b, a = butter(2, 80.0 / (sr / 2), btype="highpass")
            dialogue = lfilter(b, a, dialogue).astype(np.float32)
        except Exception:
            pass

    # 4. Generate Smooth Gap-Bridged Ducking Envelope
    duck = _build_duck_envelope(
        intervals=intervals,
        n=n,
        sr=sr,
        duck_gain=0.22,  # -13 dB ducking
        attack_s=0.15,   # 150ms smooth S-curve attack
        release_s=0.40,  # 400ms smooth S-curve release
        bridge_gap_s=0.35,  # bridge pauses < 350ms
    )

    # 5. Composite Mix
    if is_stereo:
        ducked_bed = bed * duck[:, None]
        stereo_dialogue = np.column_stack([dialogue, dialogue])
        mix = ducked_bed + stereo_dialogue
    else:
        ducked_bed = (bed.squeeze() if bed.ndim > 1 else bed) * duck
        mix = ducked_bed + dialogue

    # 6. Apply Soft-Knee Limiter
    mix = _soft_limit(mix)

    # 7. Write Artifacts
    mix_dir = job_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav(mix_dir / "dialogue.wav", dialogue, sr)
    final_path = mix_dir / "final.wav"
    write_wav(final_path, mix, sr)
    return final_path
