from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from dubstudio.util.audio import apply_micro_fades, read_audio, read_mono, write_wav

log = logging.getLogger("dubstudio.mixing")

# --- mix tuning -------------------------------------------------------------
DUCK_GAIN = 0.32            # -10 dB pocket under dialogue
DIALOGUE_TARGET_RMS = 0.09  # ~-21 dBFS RMS over speech regions
RESIDUAL_LEVEL = 0.60       # level of preserved original non-speech audio
SPEECH_GUARD_PAD_S = 0.25   # guard band around every SOURCE speech window
CROSSFADE_S = 0.040         # legacy name, kept for API compatibility
GAIN_CLAMP = (0.72, 1.40)   # +-3 dB of emotion tracking

# --- continuity (the "every line starts from digital silence" fix) ----------
# A dub used to be pasted at its ASR timestamp into an all-zero bus, so between
# two lines the dialogue did not pause, it *disappeared*: onset steps of 25-44
# dB, and the bed released to full gain inside every gap (+18 dB swells).
JOIN_CROSSFADE_S = 0.030    # real overlapping equal-power crossfade at joins
BRIDGE_GAP_S = 0.30         # joins shorter than this are welded (was 0.080)
MAX_LEAD_S = 0.20           # a line may never be pulled more than this early
DUCK_HOLD_GAP_S = 0.70      # bed stays ducked across pauses shorter than this
CHUNK_FADE_MS = 5.0         # click protection only (was 20 ms -> audible dips)
ROOM_TONE_RATIO = 0.04      # room tone RMS relative to DIALOGUE_TARGET_RMS
ROOM_TONE_EDGE_S = 0.015
MIN_HOLE_S = 0.35           # an uncovered speech window this long is reported
MIN_COVERAGE = 0.95         # below this the mix is flagged "degraded"


# --- interval helpers -------------------------------------------------------
def _merge(intervals, bridge_gap_s: float = 0.0) -> list[tuple[float, float]]:
    """Sort + merge intervals, welding neighbours closer than bridge_gap_s."""
    merged: list[list[float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals):
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= bridge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _total(intervals) -> float:
    return float(sum(b - a for a, b in intervals))


def _intersect(a_list, b_list) -> list[tuple[float, float]]:
    out = []
    for a, b in a_list:
        for c, d in b_list:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                out.append((lo, hi))
    return _merge(out)


def _gaps_inside(runs, covered) -> list[tuple[float, float]]:
    """Stretches inside `runs` that `covered` does not reach."""
    holes = []
    for a, b in runs:
        cursor = a
        for c, d in covered:
            if d <= a or c >= b:
                continue
            c, d = max(c, a), min(d, b)
            if c > cursor:
                holes.append((cursor, c))
            cursor = max(cursor, d)
        if cursor < b:
            holes.append((cursor, b))
    return holes


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

    merged = _merge(intervals, bridge_gap_s)

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


def _highpass(x: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """80 Hz high-pass. Falls back to a moving-average subtraction when scipy
    is missing, so the filter is never a silent no-op."""
    if x.size == 0:
        return x
    try:
        from scipy.signal import butter, lfilter

        b, a = butter(2, cutoff / (sr / 2), btype="highpass")
        return lfilter(b, a, x).astype(np.float32)
    except Exception:
        log.debug("scipy unavailable; using moving-average high-pass")

    win = max(3, int(round(sr / max(cutoff, 1.0))))
    if x.size < win * 2:
        return x
    pad_l = win // 2
    xp = np.pad(x.astype(np.float64), (pad_l, win - pad_l), mode="edge")
    cum = np.concatenate(([0.0], np.cumsum(xp)))
    moving = (cum[win:win + x.size] - cum[:x.size]) / win
    return (x - moving).astype(np.float32)


def _noise_floor(chunks, sr: int) -> float:
    """10th-percentile 20 ms RMS across the rendered dialogue = its own floor."""
    win = max(1, int(0.020 * sr))
    vals = []
    for audio, _s, _e in chunks:
        if audio.size < win:
            continue
        frames = audio.size // win
        block = audio[:frames * win].reshape(frames, win)
        vals.append(np.sqrt((block ** 2).mean(axis=1)))
    if not vals:
        return 0.0
    return float(np.percentile(np.concatenate(vals), 10))


def _room_tone(n: int, sr: int, level: float, seed: int = 20240914) -> np.ndarray:
    """Low-level, softly low-passed presence noise. Deterministic for tests."""
    if n <= 0 or level <= 0.0:
        return np.zeros(max(0, n), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n + 16).astype(np.float32)
    kernel = np.ones(8, dtype=np.float32) / 8.0
    tone = np.convolve(noise, kernel, mode="same")[:n]
    rms = float(np.sqrt(np.mean(tone ** 2)))
    if rms <= 1e-9:
        return np.zeros(n, dtype=np.float32)
    return (tone * (level / rms)).astype(np.float32)


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
        (mid-line pauses, soft lines and failed segments included) unless
        restore_original_on_failure is explicitly turned on;
      * consecutive lines are welded with a real overlapping crossfade instead
        of being pasted into silence, and the dialogue bus keeps a room-tone
        floor for the whole length of a talking run, so a pause sounds like a
        breath rather than like the audio dropping out;
      * the bed stays ducked across pauses shorter than DUCK_HOLD_GAP_S and
        across uncovered speech windows, so it can no longer swell inside a
        hole and then re-duck;
      * coverage is measured and reported in mix/qc.json, so a job that lost
        speech is marked "degraded" instead of shipping silently.
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
    # guard region for original-audio restoration and the ducking reference.
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

        # Click protection only. A 20 ms fade per line is what made every
        # utterance dip to zero at both ends.
        audio = apply_micro_fades(audio, fade_ms=CHUNK_FADE_MS, sample_rate=sr)
        chunks.append((audio.astype(np.float32), start, start + len(audio) / sr))

    # 4. Assemble the dialogue bus with real overlapping crossfades.
    #    A short gap is closed by pulling the incoming line back onto the tail
    #    of the outgoing one (bounded by MAX_LEAD_S against its own timestamp,
    #    so the shift can never accumulate across a run).
    dialogue = np.zeros(n, dtype=np.float32)
    intervals: list[tuple[float, float]] = []
    xfade = max(2, int(round(JOIN_CROSSFADE_S * sr)))
    lead_cap = int(round(MAX_LEAD_S * sr))
    joins_welded = 0
    join_gaps_ms: list[int] = []
    prev_end_i: int | None = None

    for audio, start, _end in chunks:
        desired_i = int(round(start * sr))
        i0 = desired_i
        if prev_end_i is not None:
            gap_s = (desired_i - prev_end_i) / sr
            join_gaps_ms.append(int(round(gap_s * 1000)))
            if gap_s < BRIDGE_GAP_S:
                i0 = max(0, desired_i - lead_cap, min(desired_i, prev_end_i) - xfade)
                joins_welded += 1
        if i0 >= n:
            continue
        piece = audio[:max(0, n - i0)]
        if piece.size == 0:
            continue

        overlap = 0 if prev_end_i is None else max(0, min(prev_end_i - i0, piece.size))
        if overlap > 1:
            t = np.linspace(0.0, np.pi / 2, overlap, dtype=np.float32)
            dialogue[i0:i0 + overlap] *= np.cos(t)   # fade the outgoing line out
            piece = piece.copy()
            piece[:overlap] *= np.sin(t)             # ... and the incoming one in

        dialogue[i0:i0 + piece.size] += piece
        intervals.append((i0 / sr, (i0 + piece.size) / sr))
        prev_end_i = i0 + piece.size

    # 5. Normalise dialogue loudness across speech regions only
    if intervals:
        idx_parts = [
            np.arange(max(0, int(a * sr)), min(n, int(b * sr)))
            for a, b in intervals
            if int(b * sr) > int(a * sr)
        ]
        if idx_parts:
            speech_idx = np.unique(np.concatenate(idx_parts))
            current = float(np.sqrt(np.mean(dialogue[speech_idx] ** 2)))
            if current > 1e-5:
                dialogue = dialogue * float(np.clip(DIALOGUE_TARGET_RMS / current, 0.25, 4.0))

    # 6. Gentle high-pass to remove low boominess
    if np.any(dialogue != 0.0):
        dialogue = _highpass(dialogue, sr, 80.0)

    # 7. Coverage analysis: which source speech did the dub actually replace?
    src_windows = _merge(source_speech)
    source_runs = _merge(source_speech, DUCK_HOLD_GAP_S)
    covered = _merge(intervals)
    speech_total = _total(src_windows)
    covered_total = _total(_intersect(src_windows, covered))
    coverage = 1.0 if speech_total <= 0.0 else covered_total / speech_total
    report_holes = [(a, b) for a, b in _gaps_inside(src_windows, covered) if b - a >= MIN_HOLE_S]
    holes_over_1s = sum(1 for a, b in report_holes if b - a >= 1.0)
    largest_hole = max((b - a for a, b in report_holes), default=0.0)

    # 8. Room tone: inside a talking run the dialogue bus must never sit at
    #    digital zero, otherwise every line starts from -inf dB and each pause
    #    sounds like the track dropped out.
    room_level = 0.0
    fill_gaps = [(a, b) for a, b in _gaps_inside(source_runs, covered) if b - a > 0.005]
    if chunks and fill_gaps:
        cap = DIALOGUE_TARGET_RMS * ROOM_TONE_RATIO
        floor = _noise_floor(chunks, sr)
        room_level = float(min(floor, cap)) if floor > cap * 0.1 else cap
        tone = _room_tone(n, sr, room_level)
        edge = max(1, int(round(ROOM_TONE_EDGE_S * sr)))
        for a, b in fill_gaps:
            i0, i1 = max(0, int(round(a * sr))), min(n, int(round(b * sr)))
            if i1 - i0 <= 2:
                continue
            piece = tone[i0:i1].copy()
            e = min(edge, (i1 - i0) // 2)
            if e > 1:
                t = np.linspace(0.0, np.pi / 2, e, dtype=np.float32)
                piece[:e] *= np.sin(t) ** 2
                piece[-e:] *= np.cos(t) ** 2
            dialogue[i0:i1] += piece

    # 9. Duck the bed under the WHOLE talking run, not just the rendered bits.
    #    Building this from the rendered chunks is what let the bed release to
    #    full gain inside a hole and then duck again (+18 dB swell).
    duck = _build_duck_envelope(
        list(src_windows) + list(covered), n, sr=sr, duck_gain=duck_gain,
        attack_s=0.18, hold_s=0.12, release_s=0.40, bridge_gap_s=DUCK_HOLD_GAP_S,
    )

    # 10. Restore original NON-SPEECH audio only (laughter, chimes, reactions).
    #     Gated by the SOURCE speech windows, never by the dub's amplitude, so a
    #     quiet or missing dub can no longer bring the original speaker back.
    guard = list(covered) if restore_original_on_failure else list(src_windows)
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

    # 11. Composite
    if is_stereo:
        mix = bed * duck[:, None] + np.column_stack([dialogue, dialogue]) + residual
    else:
        mix = (bed.squeeze() if bed.ndim > 1 else bed) * duck + dialogue + residual
    mix = _soft_limit(mix)

    # 12. Artifacts + QC report
    mix_dir = job_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)
    write_wav(mix_dir / "dialogue.wav", dialogue, sr)
    final_path = mix_dir / "final.wav"
    write_wav(final_path, mix, sr)

    degraded = bool(failed_segments) or coverage < MIN_COVERAGE or holes_over_1s > 0
    qc = {
        "segments_total": len(sorted_segs),
        "segments_mixed": len(chunks),
        "failed_segments": failed_segments,
        "duck_gain": duck_gain,
        "duck_depth_db": round(20.0 * float(np.log10(max(duck_gain, 1e-6))), 2),
        "residual_level": residual_level,
        "dialogue_peak": round(float(np.max(np.abs(dialogue))) if dialogue.size else 0.0, 4),
        "mix_peak": round(float(np.max(np.abs(mix))) if mix.size else 0.0, 4),
        # continuity / coverage
        "status": "degraded" if degraded else "ok",
        "coverage": round(float(coverage), 4),
        "source_speech_s": round(speech_total, 2),
        "covered_speech_s": round(covered_total, 2),
        "uncovered_speech_s": round(max(0.0, speech_total - covered_total), 2),
        "holes": [[round(a, 2), round(b, 2)] for a, b in report_holes],
        "largest_hole_s": round(float(largest_hole), 2),
        "holes_over_1s": holes_over_1s,
        "joins_welded": joins_welded,
        "max_join_gap_ms": max(join_gaps_ms) if join_gaps_ms else 0,
        "room_tone_rms": round(room_level, 5),
        "original_restored": bool(restore_original_on_failure),
    }
    (mix_dir / "qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if failed_segments:
        log.error("Mixed with %d failed segment(s): %s", len(failed_segments), failed_segments)
    if coverage < MIN_COVERAGE:
        log.error(
            "Only %.1f%% of the source speech was dubbed; %.2f s left uncovered "
            "(largest hole %.2f s). Mix marked degraded.",
            coverage * 100.0, speech_total - covered_total, largest_hole,
        )
    return final_path
