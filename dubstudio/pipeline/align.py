from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from dubstudio.pipeline.blocks import MAX_SYNTH_PAUSE_S, blocks_path, load_blocks
from dubstudio.pipeline.timing import fit_segment
from dubstudio.util.audio import apply_micro_fades, read_audio, write_wav

log = logging.getLogger("dubstudio.align")

# --- alignment tuning -----------------------------------------------------
# A whole block is fitted with ONE tempo ratio. Per-line stretching was the
# other half of the stutter: neighbouring lines were resampled by wildly
# different factors (up to 33% apart), so the dub changed speaking rate every
# few seconds. Within a block the rate is now constant by construction, and the
# budget is small enough to stay inaudible.
MAX_BLOCK_STRETCH = 0.08   # +-8%; beyond this we report instead of mangling
DEAD_BAND = 0.03           # inside +-3% leave the engine's own rhythm alone
SILENCE_DBFS = -45.0
EDGE_PAD_S = 0.04          # keep a little air around the speech
MIN_SILENCE_S = 0.12       # shorter dips are part of speech, not pauses
FADE_MS = 5.0              # click protection only
OVERFLOW_TOLERANCE_MS = 250


def _atempo_chain(ratio: float) -> list[float]:
    """atempo only accepts 0.5-2.0 per instance; chain them for extremes."""
    factors: list[float] = []
    remaining = ratio
    while remaining < 0.5 or remaining > 2.0:
        if remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        else:
            factors.append(2.0)
            remaining /= 2.0
    factors.append(remaining)
    return factors


def _silence_runs(x: np.ndarray, sr: int, thresh: float, min_len_s: float = MIN_SILENCE_S) -> list[tuple[int, int]]:
    if x.size == 0:
        return []
    frame = max(1, int(round(0.01 * sr)))
    count = x.size // frame
    if count == 0:
        return []
    rms = np.sqrt(np.mean(x[:count * frame].reshape(count, frame).astype(np.float64) ** 2, axis=1))
    quiet = rms < thresh
    runs: list[tuple[int, int]] = []
    i = 0
    while i < count:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < count and quiet[j]:
            j += 1
        a, b = i * frame, min(x.size, j * frame)
        if (b - a) / sr >= min_len_s:
            runs.append((a, b))
        i = j
    return runs


def _trim_edges(x: np.ndarray, sr: int, thresh: float, pad_s: float = EDGE_PAD_S) -> tuple[np.ndarray, int, int]:
    """Drop the engine's own lead-in and tail silence.

    Those two pieces of dead air are why a line seemed to start late and why
    the track went quiet before the next one began.
    """
    loud = np.where(np.abs(x) > thresh)[0]
    if loud.size == 0:
        return x, 0, 0
    pad = int(round(pad_s * sr))
    a = max(0, int(loud[0]) - pad)
    b = min(x.size, int(loud[-1]) + 1 + pad)
    head_ms = int(round(a / sr * 1000))
    tail_ms = int(round((x.size - b) / sr * 1000))
    return x[a:b], head_ms, tail_ms


def _close_pauses(x: np.ndarray, sr: int, max_pause_s: float, thresh: float) -> tuple[np.ndarray, int]:
    """Shorten over-long pauses the engine invented inside a block.

    Only the excess is removed and the kept part is the original low-level
    room noise, so the join stays continuous - there is no digital silence and
    no new edit point in the speech itself.
    """
    keep = int(round(max_pause_s * sr))
    slack = int(round(0.02 * sr))
    runs = [
        (a, b) for a, b in _silence_runs(x, sr, thresh)
        if a > 0 and b < x.size and (b - a) > keep + slack
    ]
    if not runs:
        return x, 0
    pieces: list[np.ndarray] = []
    cursor = 0
    closed = 0
    for a, b in runs:
        pieces.append(x[cursor:a])
        pieces.append(x[a:a + keep])
        closed += (b - a) - keep
        cursor = b
    pieces.append(x[cursor:])
    out = np.concatenate([p for p in pieces if p.size]) if pieces else x
    return out.astype(np.float32), int(round(closed / sr * 1000))


def fit_block(
    src: Path,
    dest: Path,
    target_ms: int,
    *,
    max_stretch: float = MAX_BLOCK_STRETCH,
    max_pause_s: float = MAX_SYNTH_PAUSE_S,
) -> dict[str, Any]:
    """Fit one rendered block into its slot: trim, de-pause, then one atempo."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    data, sr = read_audio(src, target_sr=48000, mono=True)
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    gen_ms = int(round(data.size / sr * 1000))
    target_ms = max(1, int(target_ms))
    if data.size == 0:
        shutil.copy2(src, dest)
        return {
            "stretch_ratio": 1.0, "generated_duration_ms": 0, "fitted_duration_ms": 0,
            "trimmed_ms": 0, "pauses_closed_ms": 0, "overflow_ms": 0,
            "wanted_ratio": 1.0, "status": "empty",
        }

    thresh = 10 ** (SILENCE_DBFS / 20.0)
    work, head_ms, tail_ms = _trim_edges(data, sr, thresh)
    work, closed_ms = _close_pauses(work, sr, max_pause_s, thresh)
    if work.size == 0:
        work = data

    trimmed_ms = int(round(work.size / sr * 1000))
    wanted = trimmed_ms / target_ms
    applied = max(1.0 - max_stretch, min(1.0 + max_stretch, wanted))
    if abs(applied - 1.0) <= DEAD_BAND:
        applied = 1.0

    if applied == 1.0:
        write_wav(dest, apply_micro_fades(work, fade_ms=FADE_MS, sample_rate=sr), sr)
    else:
        pre = dest.parent / f"{dest.stem}.pre.wav"
        write_wav(pre, work, sr)
        filt = ",".join(f"atempo={f:.6f}" for f in _atempo_chain(applied))
        subprocess.check_call(
            ["ffmpeg", "-y", "-i", str(pre), "-filter:a", filt, str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        pre.unlink(missing_ok=True)
        out, out_sr = read_audio(dest, target_sr=48000, mono=True)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        write_wav(dest, apply_micro_fades(out, fade_ms=FADE_MS, sample_rate=out_sr), out_sr)

    fitted, fitted_sr = read_audio(dest, target_sr=48000, mono=True)
    fitted_ms = int(round(np.asarray(fitted).reshape(-1).size / fitted_sr * 1000))
    overflow_ms = max(0, fitted_ms - target_ms)
    status = "overflow" if overflow_ms > OVERFLOW_TOLERANCE_MS else ("stretched" if applied != 1.0 else "ok")
    return {
        "stretch_ratio": round(applied, 4),
        "generated_duration_ms": gen_ms,
        "fitted_duration_ms": fitted_ms,
        "trimmed_ms": head_ms + tail_ms,
        "pauses_closed_ms": closed_ms,
        "overflow_ms": overflow_ms,
        "wanted_ratio": round(wanted, 4),
        "status": status,
    }


def _progress(job: dict | None, i: int, total: int) -> None:
    if not job or not job.get("job_id") or total <= 0:
        return
    try:
        from dubstudio.jobs.store import store

        job["percent"] = min(85 + int((i / total) * 7), 91)
        job["message"] = f"Aligning speech block {i + 1}/{total}"
        store.save(job)
    except Exception:  # noqa: BLE001 - progress reporting is best effort
        pass


def run_align(
    job_dir: Path,
    job: dict | None = None,
    *,
    max_stretch: float = MAX_BLOCK_STRETCH,
    max_pause_s: float = MAX_SYNTH_PAUSE_S,
) -> list[dict[str, Any]]:
    """Fit every rendered block into its slot and publish the result.

    The block's lead cue carries the audio for the whole run; its members are
    marked `in_block` so the mixer lays down one continuous piece instead of
    re-editing the run line by line.
    """
    seg_path = job_dir / "segments" / "segments.json"
    segments = json.loads(seg_path.read_text(encoding="utf-8"))
    if isinstance(segments, dict):
        segments = segments.get("segments", [])

    blocks = load_blocks(job_dir)
    if not blocks:
        log.warning("No blocks.json for this job; falling back to per-cue timing")
        from dubstudio.pipeline.timing import run_timing

        run_timing(job_dir, job)
        return []

    by_id = {str(s.get("segment_id")): s for s in segments}
    out_dir = job_dir / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    overflows: list[str] = []
    total = len(blocks)

    for i, block in enumerate(blocks):
        _progress(job, i, total)
        members = [by_id[str(s)] for s in (block.get("segment_ids") or []) if str(s) in by_id]
        gen_rel = block.get("generated_wav")
        src = (job_dir / gen_rel) if gen_rel else None
        if src is None or not src.exists():
            log.error("Block %s has no rendered audio; its %d cue(s) will not be dubbed",
                      block.get("block_id"), len(members))
            block["status"] = "failed"
            for seg in members:
                seg["fitted_wav"] = None
                seg["status"] = "failed"
                seg.setdefault("error", "block audio missing")
            continue

        dest = out_dir / f"{block['block_id']}.fitted.wav"
        info = fit_block(src, dest, int(block.get("target_duration_ms") or 1000),
                         max_stretch=max_stretch, max_pause_s=max_pause_s)
        rel = str(dest.relative_to(job_dir))
        block.update({
            "fitted_wav": rel,
            "generated_duration_ms": info["generated_duration_ms"],
            "fitted_duration_ms": info["fitted_duration_ms"],
            "stretch_ratio": info["stretch_ratio"],
            "overflow_ms": info["overflow_ms"],
            "pauses_closed_ms": info["pauses_closed_ms"],
            "trimmed_ms": info["trimmed_ms"],
            "status": "fitted",
        })
        (out_dir / f"{block['block_id']}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        if info["status"] == "overflow":
            overflows.append(block["block_id"])
            log.warning("Block %s still runs %d ms past its slot after a %.0f%% fit",
                        block["block_id"], info["overflow_ms"], (info["stretch_ratio"] - 1.0) * 100.0)
        if info["pauses_closed_ms"]:
            log.info("Block %s: closed %d ms of invented pauses", block["block_id"], info["pauses_closed_ms"])

        for k, seg in enumerate(members):
            seg["block_wav"] = rel
            seg["stretch_ratio"] = info["stretch_ratio"]
            if k == 0:
                seg["fitted_wav"] = rel
                seg["generated_duration_ms"] = info["generated_duration_ms"]
                seg["fitted_duration_ms"] = info["fitted_duration_ms"]
                seg["overflow_ms"] = info["overflow_ms"]
                seg["status"] = "ok"
            else:
                seg["fitted_wav"] = None
                seg["fitted_duration_ms"] = None
                seg["overflow_ms"] = 0
                seg["status"] = "in_block"

    # Cues that never made it into a block are still fitted one by one.
    for seg in segments:
        if seg.get("block_id") or not seg.get("generated_wav"):
            continue
        dest = out_dir / f"{seg['segment_id']}.fitted.wav"
        info = fit_segment(job_dir / seg["generated_wav"], dest,
                           int(seg.get("target_duration_ms") or 1000))
        seg["fitted_wav"] = str(dest.relative_to(job_dir))
        seg["generated_duration_ms"] = info["generated_duration_ms"]
        seg["fitted_duration_ms"] = info["fitted_duration_ms"]
        seg["stretch_ratio"] = info["stretch_ratio"]
        seg["overflow_ms"] = info.get("overflow_ms", 0)
        seg["status"] = "ok"

    stretches = [b["stretch_ratio"] for b in blocks if b.get("status") == "fitted"]
    if stretches:
        log.info("Aligned %d block(s); tempo %.3f-%.3f (one ratio per block)",
                 len(stretches), min(stretches), max(stretches))
    if overflows and job is not None:
        job["timing_overflows"] = overflows

    blocks_path(job_dir).write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")
    seg_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return blocks
