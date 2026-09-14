from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from dubstudio.util.audio import apply_micro_fades, read_audio, read_mono, write_wav

log = logging.getLogger("dubstudio.mixing")

# --- mix tuning -------------------------------------------------------------
DUCK_GAIN = 0.32            # -10 dB pocket under dialogue (was 0.78 = -2.2 dB)
DIALOGUE_TARGET_RMS = 0.09  # ~-21 dBFS RMS over speech regions
RESIDUAL_LEVEL = 0.60       # level of preserved original non-speech audio
SPEECH_GUARD_PAD_S = 0.25   # guard band around every SOURCE speech window
CROSSFADE_S = 0.040
BRIDGE_GAP_S = 0.080
GAIN_CLAMP = (0.72, 1.40)   # +-3 dB of emotion tracking (was 0.6 / 1.65)


def _build_duck_envelope(
    intervals: list[tuple[float, float]],
    n: int,
    sr: int = 48000,
    duck_gain: float = DUCK_GAIN,
    attack_s: float = 0.15,
    hold_s: float = 0.12,
    release_s: float = 0.45,
    bridge_gap_s: float = 0.35,
) -> np.ndarray:
    """Smooth ducking envelope with gap bridging and a broadcast hold margin."""
    envelope = np.ones(n, dtype=np.float32)
    if not intervals:
        return envelope

    merged: list[list[float]] = []
    for start, end in sorted(intervals, key=lambda x: x[0]):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= bridge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    att_samples = int(round(attack_s * sr))
    hold_samples = int(round(hold_s * sr))
    rel_samples = int(round(release_s * sr))

    for start_s, end_s in merged:
        i0 = int(round(start_s * sr))
        i1 = min(n, int(round(end_s * sr)) + hold_samples)
        if i0 >= n or i1 <= 0:
            continue

        envelope[max(0, i0):i1] = np.minimum(envelope[max(0, i0):i1], duck_gain)

        if att_samples > 0 and i0 > 0:
            a0 = max(0, i0 - att_samples)
            count = i0 - a0
            if count > 0:
                t = np.linspace(0, np.pi / 2, count, dtype=np.float32)
                curve = 1.0 - (1.0 - duck_gain) * (np.sin(t) ** 2)
                envelope[a0:i0] = np.minimum(envelope[a0:i0], curve)

        if rel_samples > 0 and i1 < n:
            r1 = min(n, i1 + rel_samples)
            count = r1 - i1
            if count > 0:
                t = np.linspace(0, np.pi / 2, count, dtype=np.float32)
                curve = duck_gain + (1.0 - duck_gain) * (np.sin(t) ** 2)
                envelope[i1:r1] = np.minimum(envelope[i1:r1], curve)

    return envelope


def _rms_envelope(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    frames = max(1, (len(x) - window) // hop + 1)
    out = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        chunk = x[i * hop:i * hop + window]
        out[i] = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
    return out


def _compute_dynamic_gain(
    synth: np.ndarray,
    orig: np.ndarray,
    sr: int,
    clamp: tuple[float, float] = GAIN_CLAMP,
) -> np.ndarray:
    """Time-varying gain that follows the original speaker's energy contour.

    Clamped to +-3 dB and smoothed over 250 ms so it shapes delivery instead of
    re-imprinting the source performance (which pumped the dub).
    """
    if len(synth) == 0 or len(orig) == 0:
        return np.ones_like(synth, dtype=np.float32)

    length = min(len(synth), len(orig))
    window = max(1, int(0.05 * sr))
    hop = max(1, window // 2)
    synth_rms = _rms_envelope(synth[:length], window, hop)
    orig_rms = _rms_envelope(orig[:length], window, hop)

    ratio = np.divide(orig_rms, synth_rms, out=np.ones_like(orig_rms), where=synth_rms > 1e-6)
    ratio = np.clip(ratio, clamp[0], clamp[1])

    smooth_frames = max(2, int(0.25 * sr / hop))
    ratio = np.convolve(ratio, np.ones(smooth_frames, dtype=np.float32) / smooth_frames, mode="same")

    x_src = np.linspace(0.0, 1.0, len(ratio), endpoint=False)
    x_dst = np.linspace(0.0, 1.0, len(synth), endpoint=False)
    return np.interp(x_dst, x_src, ratio).astype(np.float32)


def _soft_limit(mix: np.ndarray, threshold: float = 0.88) -> np.ndarray:
    """Soft-knee peak limiter to prevent digital clipping without pumping."""
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


def _speech_mask(
    intervals: list[tuple[float, float]],
    n: int,
    sr: int,
    pad_s: float = SPEECH_GUARD_PAD_S,
    ramp_s: float = 0.05,
) -> np.ndarray:
    """1.0 wherever the SOURCE was speaking (padded), with short ramps.

    The original vocal stem is never allowed back into the mix inside these
    windows, no matter how quiet the dub happens to be there.
    """
    mask = np.zeros(n, dtype=np.float32)
    pad = int(round(pad_s * sr))
    for start, end in intervals:
        i0 = max(0, int(round(start * sr)) - pad)
        i1 = min(n, int(round(end * sr)) + pad)
        if i1 > i0:
            mask[i0:i1] = 1.0
    ramp = int(round(ramp_s * sr))
    if ramp > 1:
        mask = np.convolve(mask, np.ones(ramp, dtype=np.float32) / ramp, mode="same")
        mask = np.clip(mask, 0.0, 1.0)
    return mask


def run_mixing(
    job_dir: Path,
    duration_s: float,
    *,
    duck_gain: float = DUCK_GAIN,
    residual_level: float = RESIDUAL_LEVEL,
    restore_original_on_failure: bool = False,
) -> Path:
    """Mix fitted dialogue over the music/SFX bed.

    Guarantees:
      * the original speaker is never re-injected inside a source speech window
        (mid-line pauses, soft lines and failed segments included);
      * a failed segment leaves a clean hole and is reported in mix/qc.json
        instead of being papered over with the source audio.
    """
    path = job_dir / "segments" / "segments.json"
    segments = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    sr = 48000
    n = max(1, int(duration_s * sr) + sr // 2)

    # 1. Music / ambience bed (stereo preserved)
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

    # 2. Original vocal stem (emotion reference + non-speech preservation)
    voc = None
    voc_mono = None
    voc_path = job_dir / "audio" / "vocals.wav"
    if voc_path.exists():
        try:
            voc, _ = read_audio(voc_path, target_sr=sr, mono=False)
            voc_mono = voc.mean(axis=-1) if voc.ndim > 1 else voc
        except Exception as exc:
            log.warning("Could not read vocals stem %s: %s", voc_path, exc)

    sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0.0)))

    # Every ASR speech window, INCLUDING the ones whose TTS failed. This is the
    # guard region for original-audio restoration.
    source_speech = [
        (float(s.get("start", 0.0)), float(s.get("end", 0.0)))
        for s in sorted_segs
        if float(s.get("end", 0.0)) > float(s.get("start", 0.0))
    ]
    failed_segments = [
        s.get("segment_id")
        for s in sorted_segs
        if not (s.get("fitted_wav") or s.get("generated_wav"))
    ]

    # 3. Build the dialogue chunks
    chunks: list[tuple[np.ndarray, float, float]] = []
    for idx, seg in enumerate(sorted_segs):
        wav_rel = seg.get("fitted_wav") or seg.get("generated_wav")
        if not wav_rel:
            continue
        wav_path = job_dir / wav_rel
        if not wav_path.exists():
            log.warning("Missing audio for segment %s at %s", seg.get("segment_id"), wav_path)
            if seg.get("segment_id") not in failed_segments:
                failed_segments.append(seg.get("segment_id"))
            continue

        audio, _ = read_mono(wav_path, target_sr=sr)
        if audio.size == 0:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + len(audio) / sr))

        # A line may run past its ASR end rather than be cut mid-word, but it
        # must never run into the next line.
        next_start = float(sorted_segs[idx + 1].get("start", float("inf"))) if idx + 1 < len(sorted_segs) else float("inf")
        room_s = max(0.0, min(next_start - start, n / sr - start))
        max_samples = int(round(room_s * sr))
        if max_samples <= 0:
            continue
        if len(audio) > max_samples:
            log.warning(
                "Segment %s overruns the next line by %d ms; trimming",
                seg.get("segment_id"),
                int((len(audio) - max_samples) / sr * 1000),
            )
            audio = audio[:max_samples]

        if voc_mono is not None:
            o0 = int(round(start * sr))
            o1 = min(len(voc_mono), int(round(end * sr)))
            if o1 > o0:
                ref = voc_mono[o0:o1]
                if len(ref) > len(audio):
                    ref = ref[:len(audio)]
                elif len(ref) < len(audio):
                    ref = np.pad(ref, (0, len(audio) - len(ref)))
                if float(np.sqrt(np.mean(ref ** 2))) > 1e-4 and float(np.sqrt(np.mean(audio ** 2))) > 1e-4:
                    audio = audio * _compute_dynamic_gain(audio, ref, sr)

        audio = apply_micro_fades(audio, fade_ms=20.0, sample_rate=sr)
        chunks.append((audio.astype(np.float32), start, start + len(audio) / sr))

    # 4. Render with equal-power crossfades at tight joins
    dialogue = np.zeros(n, dtype=np.float32)
    intervals: list[tuple[float, float]] = []
    prev_end: float | None = None
    for audio, start, _end in chunks:
        i0 = int(round(start * sr))
        if i0 >= n:
            continue
        piece = audio[:max(0, n - i0)]
        if piece.size == 0:
            continue
        if prev_end is not None and 0.0 <= start - prev_end < BRIDGE_GAP_S:
            cf = min(int(round(CROSSFADE_S * sr)), piece.size)
            if cf > 1:
                piece = piece.copy()
                t = np.linspace(0.0, np.pi / 2, cf, dtype=np.float32)
                piece[:cf] *= np.sin(t) ** 2
        dialogue[i0:i0 + piece.size] += piece
        intervals.append((start, start + piece.size / sr))
        prev_end = intervals[-1][1]

    # 5. Normalise dialogue loudness across speech regions only
    if intervals:
        idx_parts = [
            np.arange(max(0, int(a * sr)), min(n, int(b * sr)))
            for a, b in intervals
            if int(b * sr) > int(a * sr)
        ]
        if idx_parts:
            speech_idx = np.concatenate(idx_parts)
            current = float(np.sqrt(np.mean(dialogue[speech_idx] ** 2)))
            if current > 1e-5:
                dialogue = dialogue * float(np.clip(DIALOGUE_TARGET_RMS / current, 0.25, 4.0))

    # 6. Gentle high-pass to remove low boominess
    if np.any(dialogue != 0.0):
        try:
            from scipy.signal import butter, lfilter

            b, a = butter(2, 80.0 / (sr / 2), btype="highpass")
            dialogue = lfilter(b, a, dialogue).astype(np.float32)
        except Exception:
            log.debug("scipy unavailable; skipping dialogue high-pass")

    # 7. Duck the bed under the dub
    duck = _build_duck_envelope(intervals, n, sr=sr, duck_gain=duck_gain, attack_s=0.18,
                                hold_s=0.10, release_s=0.40, bridge_gap_s=0.35)

    # 8. Restore original NON-SPEECH audio only (laughter, chimes, reactions).
    #    Gated by the SOURCE speech windows, never by the dub's amplitude, so a
    #    quiet or missing dub can no longer bring the original speaker back.
    guard = intervals if restore_original_on_failure else source_speech
    residual_gate = 1.0 - _speech_mask(guard, n, sr)
    residual = np.zeros_like(bed)
    if voc is not None:
        voc_stereo = voc if voc.ndim > 1 else np.column_stack([voc, voc])
        if len(voc_stereo) < n:
            voc_stereo = np.pad(voc_stereo, ((0, n - len(voc_stereo)), (0, 0)))[:n]
        else:
            voc_stereo = voc_stereo[:n]
        if is_stereo:
            residual = (voc_stereo[:n, :2] * residual_gate[:n, None] * residual_level).astype(np.float32)
        else:
            residual = (voc_stereo[:n].mean(axis=-1) * residual_gate[:n] * residual_level).astype(np.float32)

    # 9. Composite
    if is_stereo:
        mix = bed * duck[:, None] + np.column_stack([dialogue, dialogue]) + residual
    else:
        mix = (bed.squeeze() if bed.ndim > 1 else bed) * duck + dialogue + residual
    mix = _soft_limit(mix)

    # 10. Artifacts + QC report
    mix_dir = job_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav(mix_dir / "dialogue.wav", dialogue, sr)
    final_path = mix_dir / "final.wav"
    write_wav(final_path, mix, sr)

    qc = {
        "segments_total": len(sorted_segs),
        "segments_mixed": len(chunks),
        "failed_segments": failed_segments,
        "duck_gain": duck_gain,
        "duck_depth_db": round(20.0 * float(np.log10(max(duck_gain, 1e-6))), 2),
        "residual_level": residual_level,
        "dialogue_peak": round(float(np.max(np.abs(dialogue))) if dialogue.size else 0.0, 4),
        "mix_peak": round(float(np.max(np.abs(mix))) if mix.size else 0.0, 4),
    }
    (mix_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if failed_segments:
        log.error("Mixed with %d failed segment(s): %s", len(failed_segments), failed_segments)
    return final_path
