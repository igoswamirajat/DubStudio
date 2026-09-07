from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from dubstudio.util.audio import apply_micro_fades, read_audio, read_mono, write_wav

log = logging.getLogger("dubstudio.mixing")


def _build_duck_envelope(
    intervals: list[tuple[float, float]],
    n: int,
    sr: int = 48000,
    duck_gain: float = 0.78,
    attack_s: float = 0.15,
    hold_s: float = 0.12,
    release_s: float = 0.45,
    bridge_gap_s: float = 0.35,
) -> np.ndarray:
    """Build a broadcast-quality smooth ducking envelope.
    
    Bridges pauses shorter than bridge_gap_s to eliminate accordion pumping,
    uses raised-cosine (S-curve) attack and release transitions, and maintains
    a gentle hold margin (hold_s) after dialogue finishes before ramping back up.
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

    # 2. Render smooth S-curve envelope with hold margin
    att_samples = int(round(attack_s * sr))
    hold_samples = int(round(hold_s * sr))
    rel_samples = int(round(release_s * sr))

    for start_s, end_s in merged:
        i0 = int(round(start_s * sr))
        # Extend hold region past speech end to prevent abrupt music jumps
        i1 = min(n, int(round(end_s * sr)) + hold_samples)
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

        # Release transition (ramp up from duck_gain to 1.0 after hold)
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

    # 2. Read Original Vocals Stem if available (for emotion matching & SFX restoration)
    voc_path = job_dir / "audio" / "vocals.wav"
    voc = None
    voc_mono = None
    if voc_path.exists():
        try:
            voc, _ = read_audio(voc_path, target_sr=sr, mono=False)
            voc_mono = voc.mean(axis=-1) if voc.ndim > 1 else voc
        except Exception as exc:
            log.warning("Could not read vocals stem %s: %s", voc_path, exc)

    # 3. Place Dialogue Segments with Dynamic Emotion & Loudness Matching
    dialogue = np.zeros(n, dtype=np.float32)
    intervals: list[tuple[float, float]] = []

    sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    for idx, seg in enumerate(sorted_segs):
        wav_rel = seg.get("fitted_wav") or seg.get("generated_wav")
        if not wav_rel:
            continue
        wav_path = job_dir / wav_rel
        if not wav_path.exists():
            continue

        audio, _ = read_mono(wav_path, target_sr=sr)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + len(audio) / sr))

        # Anti-collision boundary check: clamp smoothly if overlapping next segment
        if idx + 1 < len(sorted_segs):
            next_start = float(sorted_segs[idx + 1].get("start", float("inf")))
            max_dur = next_start - start
            if max_dur > 0 and (len(audio) / sr) > max_dur:
                max_samples = int(round(max_dur * sr))
                audio = audio[:max_samples]

        audio = apply_micro_fades(audio, fade_ms=5.0, sample_rate=sr)

        # Dynamic Loudness & Emotion Matching:
        # Match the energy contour of the original speaker in this segment
        if voc_mono is not None:
            orig_i0 = int(round(start * sr))
            orig_i1 = min(len(voc_mono), int(round(end * sr)))
            if orig_i1 > orig_i0:
                orig_slice = voc_mono[orig_i0:orig_i1]
                orig_rms = float(np.sqrt(np.mean(orig_slice ** 2)))
                synth_rms = float(np.sqrt(np.mean(audio ** 2)))
                if orig_rms > 1e-4 and synth_rms > 1e-4:
                    # Clamped between 0.60 (-4.4 dB) and 1.65 (+4.3 dB) for safe, natural dynamics
                    gain = float(np.clip(orig_rms / synth_rms, 0.60, 1.65))
                    audio = audio * gain

        i0 = int(round(start * sr))
        i1 = min(n, i0 + len(audio))
        if i0 >= n or i1 <= i0:
            continue

        dialogue[i0:i1] += audio[: i1 - i0]
        intervals.append((start, start + (len(audio) / sr)))

    # 4. Apply Subtle Dialogue High-Pass Filter (> 80 Hz) to eliminate low boominess
    if np.any(dialogue != 0.0):
        try:
            from scipy.signal import butter, lfilter

            b, a = butter(2, 80.0 / (sr / 2), btype="highpass")
            dialogue = lfilter(b, a, dialogue).astype(np.float32)
        except Exception:
            pass

    # 5. Generate Smooth Gap-Bridged Ducking Envelope with Broadcast Hold Margin
    # Transparent -2.2 dB pocket (duck_gain = 0.78) maintains original music/SFX presence!
    duck = _build_duck_envelope(
        intervals=intervals,
        n=n,
        sr=sr,
        duck_gain=0.78,  # -2.2 dB transparent vocal pocket
        attack_s=0.15,   # 150ms smooth S-curve lookahead attack
        hold_s=0.12,     # 120ms broadcast hold after dialogue
        release_s=0.45,  # 450ms smooth S-curve release
        bridge_gap_s=0.35,  # bridge pauses < 350ms
    )

    # 6. Residual SFX & Non-Speech Vocalization Preservation ("Zero Missed Sounds")
    sfx_preserved = np.zeros_like(bed)
    if voc is not None:
        dia_abs = np.abs(dialogue)
        smooth_win = int(sr * 0.05)  # 50ms smoothing window
        if len(dialogue) > smooth_win:
            dia_active = np.convolve(
                (dia_abs > 0.015).astype(np.float32),
                np.ones(smooth_win) / smooth_win,
                mode="same",
            ) > 0.01
        else:
            dia_active = (dia_abs > 0.015).astype(np.float32)

        non_speech_mask = 1.0 - dia_active.astype(np.float32)
        voc_stereo = voc if voc.ndim > 1 else np.column_stack([voc, voc])
        if len(voc_stereo) < n:
            voc_stereo = np.pad(
                voc_stereo,
                ((0, n - len(voc_stereo)), (0, 0)) if voc_stereo.ndim > 1 else (0, n - len(voc_stereo)),
            )[:n]
        else:
            voc_stereo = voc_stereo[:n]

        # In pause intervals, restore original sounds/SFX/reactions (e.g. laughter, breath, chimes)
        if is_stereo:
            sfx_preserved = (voc_stereo[:n, :2] * non_speech_mask[:n, None] * 0.85).astype(np.float32)
        else:
            sfx_preserved = (voc_stereo[:n].mean(axis=-1) * non_speech_mask[:n] * 0.85).astype(np.float32)

    # 7. Composite Mix
    if is_stereo:
        ducked_bed = bed * duck[:, None]
        stereo_dialogue = np.column_stack([dialogue, dialogue])
        mix = ducked_bed + stereo_dialogue + sfx_preserved
    else:
        ducked_bed = (bed.squeeze() if bed.ndim > 1 else bed) * duck
        mix = ducked_bed + dialogue + sfx_preserved

    # 8. Apply Soft-Knee Limiter
    mix = _soft_limit(mix)

    # 9. Write Artifacts
    mix_dir = job_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav(mix_dir / "dialogue.wav", dialogue, sr)
    final_path = mix_dir / "final.wav"
    write_wav(final_path, mix, sr)
    return final_path
