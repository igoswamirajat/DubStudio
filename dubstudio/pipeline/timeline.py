"""Global timeline solver: one speaking rate for the whole video.

Before this existed every block was fitted on its own. Each one got the atempo
ratio that made *it* match *its* slot, clamped to +-8%, so two neighbouring
blocks could land at 0.92 and 1.08 and the dub changed speaking rate at every
block boundary. Worse, a block whose rendered audio ran past the *next* block's
start was hard-trimmed by the mixer - its last words were deleted instead of
being paced.

The solver sees every block at once, in timeline order, and decides all the
ratios together:

* a **global tempo** from the total content over the total slot budget (per
  speaker, so a two-voice video does not average a fast and a slow talker into
  one compromise rate);
* each block's ratio is **eased towards that global rate** in proportion to how
  much of the speech it carries. A long block is allowed to keep its own need -
  it dominates the global rate anyway - while a short block follows the flow
  instead of imposing its own;
* a hard **runway check**: the fitted length has to fit between this block's
  start and wherever the next speech begins. If it does not, that block is
  compressed as far as it takes and the collision is reported, so the mixer's
  safety trim can never quietly cut the end off a line again.

Everything here is pure: blocks in, decisions out. `align.run_align` supplies
`content_ms` (the post-trim length of each rendered block) and applies the
result.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("dubstudio.timeline")

# --- solver tuning ---------------------------------------------------------
# How far a single block may be pushed away from the engine's own rhythm. The
# band is shared with align.MAX_BLOCK_STRETCH; it is duplicated here so the
# solver stays a pure function that does not import the aligner (which would be
# a circular import: align imports timeline).
MAX_STRETCH = 0.08

# A block may not be eased further than this from its own honest ratio, even
# towards the global rate. Past this the correction is audible on its own, so
# the block keeps its own ratio and the deviation shows up in the report.
MAX_EASE = 0.05

# A speaking rate this much faster than natural is never asked for, no matter
# what the slot wants (see omnivoice_engine.MIN_NATURAL_WPS for the slow end).
MAX_TEMPO = 2.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _f(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = obj.get(key, default)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def unit_starts(blocks: list[dict[str, Any]], extra_starts: list[float] | None = None) -> list[float]:
    """Every point in time where a piece of speech begins, sorted.

    Blocks are fitted against the *next* thing that will be laid down, which may
    be another block or a cue that stayed on its own (a failed translation, for
    instance). `align.run_align` passes those solo starts in so the runway is
    never over-estimated.
    """
    starts = [ _f(b, "start") for b in (blocks or []) if b.get("block_id") ]
    if extra_starts:
        starts.extend(float(s) for s in extra_starts if s is not None)
    return sorted(set(starts))


def runway_for(start_s: float, starts: list[float], total_duration_s: float | None) -> float:
    """Seconds available before the next speech begins (or the video ends)."""
    nxt = [s for s in starts if s > start_s + 1e-6]
    if nxt:
        end = nxt[0]
        if total_duration_s is not None:
            end = min(end, total_duration_s)
    elif total_duration_s is not None:
        end = total_duration_s
    else:
        # Nothing follows and the length is unknown: assume the block's own slot
        # is the budget rather than pretending the track goes on forever.
        end = start_s
    return max(0.0, end - start_s)


def solve_timeline(
    blocks: list[dict[str, Any]],
    *,
    max_stretch: float = MAX_STRETCH,
    total_duration_s: float | None = None,
    extra_starts: list[float] | None = None,
) -> dict[str, Any]:
    """Decide one tempo ratio per block from the shape of the whole timeline.

    Each block needs `content_ms` (its trimmed, de-paused length) and
    `target_duration_ms` (its slot). Returns a decision per block plus the
    summary numbers the QC report wants.

      ratio = 1.0            -> keep the engine's own rhythm
      ratio > 1.0            -> speed up to fit
      collided = True        -> compressed past the normal band so it fits the
                                runway; the mixer would otherwise cut it
    """
    ordered = sorted(
        [b for b in (blocks or []) if b.get("block_id") and (b.get("content_ms") or 0) > 0],
        key=lambda b: (_f(b, "start"), str(b.get("block_id"))),
    )
    if not ordered:
        return {"global_ratio": 1.0, "per_block": {}, "collisions": [], "rate_spread": 0.0}

    lo, hi = 1.0 - max_stretch, 1.0 + max_stretch
    starts = unit_starts(ordered, extra_starts)

    content = [max(1, int(b.get("content_ms") or 0)) for b in ordered]
    target = [max(1, int(b.get("target_duration_ms") or 1)) for b in ordered]

    total_content = sum(content)
    total_target = sum(target)
    global_ratio = _clamp(total_content / total_target, lo, hi)

    # One rate per speaker: averaging a fast and a slow talker would be wrong
    # for both, and a speaker change is exactly where the rate is allowed to.
    speaker: dict[str, dict[str, float]] = {}
    for b, c, t in zip(ordered, content, target):
        sid = str(b.get("speaker_id") or "S00")
        acc = speaker.setdefault(sid, {"content": 0.0, "target": 0.0, "n": 0})
        acc["content"] += c
        acc["target"] += t
        acc["n"] += 1
    speaker_ratio = {
        sid: _clamp(a["content"] / a["target"], lo, hi) for sid, a in speaker.items()
    }

    per_block: dict[str, dict[str, float]] = {}
    collisions: list[str] = []
    spread: list[float] = []

    for b, c, t in zip(ordered, content, target):
        bid = str(b.get("block_id"))
        base = speaker_ratio.get(str(b.get("speaker_id") or "S00"), global_ratio)
        wanted = c / t

        # Ease towards the speaker's rate, weighted by how much of that
        # speaker's speech this block carries. A block holding half the talk
        # is mostly what set the rate, so it keeps its own; a one-liner should
        # follow the flow. Never ease further than MAX_EASE from the truth.
        share = c / max(1.0, total_content)
        ease = _clamp(share, 0.0, 1.0)
        eased = base + ease * (wanted - base)
        ratio = _clamp(_clamp(eased, wanted - MAX_EASE, wanted + MAX_EASE), lo, hi)

        runway_ms = runway_for(_f(b, "start"), starts, total_duration_s) * 1000.0
        if runway_ms >= 1.0:
            fitted = c / ratio
            if fitted > runway_ms:
                # Compressing past the band is still better than the mixer's
                # hard trim, which deletes words. Report it either way.
                ratio = max(1.0 / MAX_TEMPO, c / runway_ms)
                collisions.append(bid)

        spread.append(ratio)
        per_block[bid] = {
            "ratio": round(ratio, 4),
            "wanted_ratio": round(wanted, 4),
            "speaker_ratio": round(base, 4),
            "runway_ms": int(round(runway_ms)),
            "fitted_ms": int(round(c / ratio)),
            "collided": bid in collisions,
        }

    ratios = [v["ratio"] for v in per_block.values()]
    rate_spread = (max(ratios) - min(ratios)) if len(ratios) > 1 else 0.0
    rate_jump = ((max(ratios) / min(ratios)) - 1.0) * 100.0 if ratios and min(ratios) > 0 else 0.0

    if collisions:
        log.warning(
            "%d block(s) do not fit the runway before the next line and were "
            "compressed past the +-%.0f%% band: %s",
            len(collisions), max_stretch * 100.0, collisions[:8],
        )
    log.info(
        "Timeline solved: global tempo %.3f, %d block(s), worst rate jump "
        "%.1f%%, %d collision(s)",
        global_ratio, len(ordered), rate_jump, len(collisions),
    )
    return {
        "global_ratio": round(global_ratio, 4),
        "speaker_ratio": {k: round(v, 4) for k, v in speaker_ratio.items()},
        "per_block": per_block,
        "collisions": collisions,
        "rate_spread": round(rate_spread, 4),
        "total_content_ms": total_content,
        "total_target_ms": total_target,
    }
