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


def _compute_dynamic_gain(synth: np.ndarray, orig: np.ndarray, sr: int) -> np.ndarray:
    """Compute time‑varying gain curve to match original energy contour.
    
    Returns a gain curve of the same length as `synth`. Uses a 50ms sliding window
    RMS ratio, smoothed with a 100ms moving average, clamped to [0.6, 1.65].
    """
    if len(synth) == 0 or len(orig) == 0:
        return np.ones_like(synth, dtype=np.float32)
    
    # Trim both to the shorter length (should be equal after caller truncates)
    min_len = min(len(synth), len(orig))
    synth = synth[:min_len]
    orig = orig[:min_len]
    
    window = int(0.05 * sr)  # 50ms
    hop = window // 2
    
    # Compute RMS in sliding windows
    def rms_envelope(x, window, hop):
        n = len(x)
        frames = (n - window) // hop + 1
        rms = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            start = i * hop
            end = start + window
            rms[i] = np.sqrt(np.mean(x[start:end]**2))
        return rms
    
    synth_rms = rms_envelope(synth, window, hop)
    orig_rms = rms_envelope(orig, window, hop)
    
    # Ratio, with protection
    ratio = np.divide(orig_rms, synth_rms, out=np.ones_like(orig_rms), where=synth_rms > 1e-6)
    ratio = np.clip(ratio, 0.6, 1.65)
    
    # Smooth with moving average (100ms)
    smooth_window = int(0.1 * sr / hop)  # number of frames
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        ratio = np.convolve(ratio, kernel, mode='same')
    
    # Interpolate back to the length of the original `synth` (before trimming)
    # We'll use the original length passed in, but we have already trimmed.
    # Instead, we'll interpolate to the length of the original synth (len before trim).
    # To avoid loss, we should keep the original synth length.
    # We'll compute ratio envelope, then interpolate to the full synth length.
    # But we already trimmed, so we need to know the full length.
    # Better: don't trim; instead compute on the overlapping part and extrapolate.
    # Simpler: after computing ratio, interpolate to len(synth_full).
    # Since we don't have synth_full, we'll just interpolate to min_len, but then
    # we need to resize to match the caller's `audio` length. So we'll return
    # a curve of length `min_len`, and the caller will interpolate again.
    # Actually, it's cleaner to have the caller ensure lengths match.
    # So we return curve of length `min_len`, and caller must ensure it matches
    # the audio length.
    # But caller currently expects same length as `audio`; we can fix caller to
    # trim audio first.
    # For now, we'll return curve of length `len(synth)` after trimming to min_len,
    # but that may be shorter than audio. So we'll interpolate back to the original
    # synth length (before trimming) by using the original synth length.
    # We'll compute ratio on the overlap, then interpolate to the length of the original synth.
    # That will stretch the gain curve to match the audio length.
    # Let's do that:
    # We have original synth length = len(synth_original) but we overwrote synth.
    # We'll keep the original length as a parameter.
    # So we'll change the function signature to accept orig_len.
    # But to avoid breaking the call, we'll compute orig_len before trimming.
    # I'll restructure: pass synth and orig, compute overlap, then interpolate to len(synth).
    
    # However, the caller now truncates audio to match orig length, so lengths will be equal.
    # So we can just return a curve of length = len(synth) (which equals len(orig) after truncation).
    # So we set x_new = np.linspace(0, 1, len(synth), endpoint=False)
    # That will give curve length = len(synth) which matches.
    x_orig = np.linspace(0, 1, len(ratio), endpoint=False)
    x_new = np.linspace(0, 1, len(synth), endpoint=False)
    gain_curve = np.interp(x_new, x_orig, ratio).astype(np.float32)
    return gain_curve

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

        # Truncate audio to the exact segment duration from the JSON
        expected_len = int(round((end - start) * sr))
        if len(audio) > expected_len:
            audio = audio[:expected_len]
        elif len(audio) < expected_len:
            # Pad with zeros if too short (should not happen, but safe)
            audio = np.pad(audio, (0, expected_len - len(audio)), mode='constant')

        # Anti-collision boundary check: if this segment overlaps the next, truncate
        if idx + 1 < len(sorted_segs):
            next_start = float(sorted_segs[idx + 1].get("start", float("inf")))
            max_dur = next_start - start
            if max_dur > 0 and expected_len > int(round(max_dur * sr)):
                max_samples = int(round(max_dur * sr))
                audio = audio[:max_samples]
                # Also update expected_len for consistency
                expected_len = len(audio)

        audio = apply_micro_fades(audio, fade_ms=25.0, sample_rate=sr)

        # Dynamic Loudness & Emotion Matching:
        # Match the energy contour of the original speaker in this segment
        if voc_mono is not None:
            orig_i0 = int(round(start * sr))
            orig_i1 = min(len(voc_mono), int(round(end * sr)))
            if orig_i1 > orig_i0:
                orig_slice = voc_mono[orig_i0:orig_i1]
                # Ensure orig_slice is the same length as audio after truncation
                if len(orig_slice) > len(audio):
                    orig_slice = orig_slice[:len(audio)]
                elif len(orig_slice) < len(audio):
                    orig_slice = np.pad(orig_slice, (0, len(audio) - len(orig_slice)), mode='constant')
                orig_rms = float(np.sqrt(np.mean(orig_slice ** 2)))
                synth_rms = float(np.sqrt(np.mean(audio ** 2)))
                if orig_rms > 1e-4 and synth_rms > 1e-4:
                    # Compute time‑varying gain curve to match original energy contour
                    gain_curve = _compute_dynamic_gain(audio, orig_slice, sr)
                    audio = audio * gain_curve

        i0 = int(round(start * sr))
        # Use the exact end time for placement, not the audio length
        i1 = min(n, int(round(end * sr)))
        if i0 >= n or i1 <= i0:
            continue

        # Ensure audio length matches the placement length
        placement_len = i1 - i0
        if len(audio) > placement_len:
            audio = audio[:placement_len]
        elif len(audio) < placement_len:
            audio = np.pad(audio, (0, placement_len - len(audio)), mode='constant')

        dialogue[i0:i1] += audio
        intervals.append((start, end))

    # 4. Crossfade adjacent chunks to eliminate level discontinuities
    if chunks:
        # Sort by start time
        chunks.sort(key=lambda x: x[1])
        # Crossfade duration: 20ms (0.02s)
        crossfade_s = 0.020
        crossfade_samples = int(round(crossfade_s * sr))
        
        # Build dialogue with crossfades
        for i, (audio, start, end) in enumerate(chunks):
            i0 = int(round(start * sr))
            i1 = int(round(end * sr))
            if i0 >= n or i1 <= i0:
                continue
            # If this is not the first chunk, check overlap with previous
            if i > 0:
                prev_start, prev_end = chunks[i-1][1], chunks[i-1][2]
                # If current starts before previous ends + crossfade margin, we have overlap or small gap
                if start < prev_end + crossfade_s:
                    overlap_start = max(start, prev_start)
                    overlap_end = min(end, prev_end)
                    # If there is actual overlap
                    if overlap_end > overlap_start:
                        # Compute overlap sample indices
                        ov0 = int(round(overlap_start * sr))
                        ov1 = int(round(overlap_end * sr))
                        if ov1 > ov0:
                            # Get overlapping parts of current and previous
                            # Previous audio overlap: from ov0 - prev_start to ov0 - prev_start + len(ov)
                            prev_audio = chunks[i-1][0]
                            cur_audio = audio
                            # Previous overlap slice
                            prev_ov_start = ov0 - int(round(prev_start * sr))
                            prev_ov_end = prev_ov_start + (ov1 - ov0)
                            if prev_ov_end <= len(prev_audio):
                                prev_ov = prev_audio[prev_ov_start:prev_ov_end]
                            else:
                                prev_ov = np.zeros(ov1 - ov0, dtype=np.float32)
                            # Current overlap slice (starting at 0)
                            cur_ov = cur_audio[:ov1 - ov0]
                            # Ensure same length
                            if len(cur_ov) < ov1 - ov0:
                                cur_ov = np.pad(cur_ov, (0, ov1 - ov0 - len(cur_ov)))
                            # Create raised-cosine crossfade
                            t = np.linspace(0, np.pi, ov1 - ov0, dtype=np.float32)
                            fade_out = (np.cos(t) + 1) / 2  # 1 -> 0
                            fade_in = (np.sin(t) + 1) / 2   # 0 -> 1
                            # Mix overlap
                            mixed_ov = prev_ov * fade_out + cur_ov * fade_in
                            # Place mixed overlap into dialogue
                            dialogue[ov0:ov1] += mixed_ov
                            # For the non-overlapping parts, we'll add later
                            # Mark that we've handled overlap by trimming current audio
                            # We'll add current after the overlap
                            # But simpler: add previous fully, then add current with the overlap part zeroed out
                            # Actually easier: we'll just add the overlap separately and then add the rest.
                            # To avoid double adding, we'll zero out the overlap in the current audio before adding.
                            cur_audio[:ov1 - ov0] = 0.0
            # Add current chunk (with overlap zeroed if any)
            end_idx = i0 + len(audio)
            if end_idx > n:
                audio = audio[:n - i0]
                end_idx = n
            if len(audio) > 0:
                dialogue[i0:end_idx] += audio
    else:
        # Fallback: no chunks (should not happen)
        dialogue = np.zeros(n, dtype=np.float32)
    
    # 5. Apply Subtle Dialogue High-Pass Filter (> 80 Hz) to eliminate low boominess
    if np.any(dialogue != 0.0):
        try:
            from scipy.signal import butter, lfilter

            b, a = butter(2, 80.0 / (sr / 2), btype="highpass")
            dialogue = lfilter(b, a, dialogue).astype(np.float32)
        except Exception:
            pass

    # 6. Generate Smooth Gap-Bridged Ducking Envelope with Broadcast Hold Margin
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

    # 7. Residual SFX & Non-Speech Vocalization Preservation ("Zero Missed Sounds")
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

    # 8. Composite Mix
    if is_stereo:
        ducked_bed = bed * duck[:, None]
        stereo_dialogue = np.column_stack([dialogue, dialogue])
        mix = ducked_bed + stereo_dialogue + sfx_preserved
    else:
        ducked_bed = (bed.squeeze() if bed.ndim > 1 else bed) * duck
        mix = ducked_bed + dialogue + sfx_preserved

    # 9. Apply Soft-Knee Limiter
    mix = _soft_limit(mix)

    # 10. Write Artifacts
    mix_dir = job_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav(mix_dir / "dialogue.wav", dialogue, sr)
    final_path = mix_dir / "final.wav"
    write_wav(final_path, mix, sr)
    return final_path
