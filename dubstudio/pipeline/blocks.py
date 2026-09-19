from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("dubstudio.blocks")

# --- block tuning ---------------------------------------------------------
# A segment is a subtitle cue. It is not a sensible unit to synthesise: one TTS
# call per cue means the engine gets no context, restarts its prosody on every
# line, adds its own lead-in and tail silence, and the pieces then have to be
# edited back together join by join. That is why the dub stops dead after each
# line instead of flowing like somebody talking.
#
# A *block* is one run of continuous speech by one speaker: consecutive cues
# with no real beat between them. One block = one TTS call = one tempo ratio =
# one contiguous piece of audio in the mix, so segment boundaries inside a
# block stop existing at all.
BLOCK_PAUSE_S = 1.20      # a pause at least this long is a real beat -> new block
MAX_BLOCK_S = 30.0        # keep a request inside what the engines handle well
MAX_BLOCK_CHARS = 700     # ... and inside their text limits
MAX_SYNTH_PAUSE_S = 0.60  # longest pause we ask the engine to hold inside a block
CLAUSE_PAUSE_MS = 250     # a pause this long deserves at least a comma

# --- carry-over prosody ----------------------------------------------------
# A block's slot length sets the pace the engine is asked for, and the slots are
# derived from the ASR timings of whatever the speaker happened to do in each
# run: a slow line followed by a fast one makes the *same* voice drawl and then
# rush. The engine has no way to know it just spoke, so the rate is carried over
# by hand - each speaker gets one smoothed words/sec across the whole video and
# every block's target duration is re-derived from it. OmniVoice then paces
# itself to that single rate instead of restarting its prosody contour at every
# block boundary.
MIN_NATURAL_WPS = 1.8     # never ask for a drawl (mirrors omnivoice_engine)
MAX_NATURAL_WPS = 4.5     # never ask for a sprint
RATE_SMOOTH_ALPHA = 0.5   # half the block's own implied rate, half what came before
MAX_TARGET_SHIFT = 0.15   # a target may only move +-15% towards the smoothed rate
RATE_FIX_THRESHOLD = 0.15 # only touch rates that actually disagree by more than this

_SENT_END = ".?!\u0964\u2026"
_TRAILING = "\"')]\u201d\u2019"
_CLAUSE_END = ",;:\u060c\u3001\u2014-"


def _bare(text: str | None) -> str:
    t = (text or "").strip()
    while t and t[-1] in _TRAILING:
        t = t[:-1]
    return t


def _ends_sentence(text: str | None) -> bool:
    t = _bare(text)
    return bool(t) and t[-1] in _SENT_END


def _ends_clause(text: str | None) -> bool:
    t = _bare(text)
    return bool(t) and t[-1] in _CLAUSE_END


def _line(seg: dict[str, Any]) -> str:
    return (seg.get("translated_text") or "").strip()


def _speaker(seg: dict[str, Any]) -> str:
    return seg.get("speaker_id") or "S00"


def _f(seg: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(seg.get(key, default) if seg.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _dubbable(seg: dict[str, Any]) -> bool:
    """Only lines that actually have target-language text can join a block.

    An untranslated or failed cue must stay on its own so it shows up in QC as
    a hole, instead of being silently swallowed into a neighbour's audio.
    """
    if (seg.get("status") or "") in {"failed", "translation_failed"}:
        return False
    return bool(_line(seg))


def _spoken_words(text: str | None) -> int:
    """Script-agnostic word count (Devanagari combining marks break ``\\w``)."""
    return sum(1 for tok in (text or "").split() if any(ch.isalnum() for ch in tok))


def _implied_wps(block: dict[str, Any]) -> float | None:
    """Words/sec the block's slot would ask the engine for, or None if unknowable."""
    words = _spoken_words(block.get("text"))
    slot_s = (float(block.get("target_duration_ms") or 0)) / 1000.0
    if words <= 0 or slot_s <= 0.05:
        return None
    return words / slot_s


def _runway_s(blocks: list[dict[str, Any]], index: int, total_s: float | None) -> float | None:
    """Seconds from this block's start until the next speech begins.

    `None` when there is nothing after it and the video length is unknown: the
    track simply ends, so there is no runway to respect and no reason to clamp.
    """
    start = _f(blocks[index], "start")
    nxt = [b for b in blocks[index + 1:] if _f(b, "start") > start + 1e-6]
    if nxt:
        end = _f(nxt[0], "start")
    elif total_s is not None:
        end = total_s
    else:
        return None
    return max(0.05, end - start)


def smooth_block_rates(
    blocks: list[dict[str, Any]],
    *,
    alpha: float = RATE_SMOOTH_ALPHA,
    max_shift: float = MAX_TARGET_SHIFT,
    min_wps: float = MIN_NATURAL_WPS,
    max_wps: float = MAX_NATURAL_WPS,
    total_s: float | None = None,
) -> list[dict[str, Any]]:
    """Carry one speaking rate per speaker across the whole video.

    Each block's `target_duration_ms` is re-derived from an exponentially
    smoothed words/sec for its speaker (EMA in timeline order, so the rate
    carries forward rather than averaging a fast start and a slow end into one
    wrong constant). The move is bounded: a target never shifts more than
    `max_shift`, never leaves the natural-rate envelope, and never outgrows the
    runway before the next line starts.

    Jobs whose rates already agree (within `RATE_FIX_THRESHOLD`) are left
    completely untouched - there is nothing to carry over.
    """
    ordered = sorted([b for b in (blocks or []) if b.get("block_id")], key=lambda b: _f(b, "start"))
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for block in ordered:
        by_speaker.setdefault(str(block.get("speaker_id") or "S00"), []).append(block)

    for sid, spk_blocks in by_speaker.items():
        implied = [_implied_wps(b) for b in spk_blocks]
        known = [r for r in implied if r is not None]
        if len(known) < 2:
            continue
        # Already consistent? Leave the job exactly as it was.
        lo, hi = min(known), max(known)
        if lo > 0 and (hi / lo - 1.0) <= RATE_FIX_THRESHOLD:
            continue

        carried = known[0]
        for block, rate in zip(spk_blocks, implied):
            if rate is None:
                continue
            carried = alpha * rate + (1.0 - alpha) * carried
            words = _spoken_words(block.get("text"))
            slot_ms = int(block.get("target_duration_ms") or 0)
            if words <= 0 or slot_ms <= 0:
                continue

            wanted_ms = int(round(words / carried * 1000.0))
            floor_ms = int(round(words / max_wps * 1000.0))
            ceil_ms = int(round(words / min_wps * 1000.0))
            new_ms = int(max(
                max(slot_ms * (1.0 - max_shift), floor_ms),
                min(slot_ms * (1.0 + max_shift), ceil_ms, wanted_ms),
            ))
            # Never ask for more audio than the runway can hold; the timeline
            # solver would only have to compress it straight back out. But a
            # slot that already overruns its runway is the slot's problem, not
            # ours - clamping below the original would speed the block up
            # instead of easing it.
            idx = ordered.index(block)
            runway = _runway_s(ordered, idx, total_s)
            if runway is not None:
                runway_cap = int(runway * 1000.0 * 0.98)
                new_ms = min(new_ms, max(runway_cap, slot_ms))
            new_ms = max(1, new_ms)
            if new_ms == slot_ms:
                continue
            block["target_duration_ms"] = new_ms
            block.setdefault("prosody", {}).update({
                "implied_wps": round(rate, 3),
                "smoothed_wps": round(carried, 3),
                "original_target_ms": slot_ms,
                "target_shift_ms": new_ms - slot_ms,
            })
            log.info(
                "Carry-over prosody %s: %.2f -> %.2f w/s, target %d -> %d ms",
                block["block_id"], rate, carried, slot_ms, new_ms,
            )
    return blocks


def _join_text(parts: list[dict[str, Any]]) -> str:
    """One paragraph for one TTS call.

    Where the speaker took a noticeable pause but the text has no punctuation,
    add a comma: that gives the engine a prosodic cue to breathe there, which
    is what we want instead of splicing in digital silence afterwards.
    """
    out: list[str] = []
    prev_gap = 0
    for part in parts:
        text = (part.get("text") or "").strip()
        if not text:
            continue
        if out and prev_gap >= CLAUSE_PAUSE_MS and not _ends_sentence(out[-1]) and not _ends_clause(out[-1]):
            out[-1] = out[-1] + ","
        out.append(text)
        prev_gap = int(part.get("gap_after_ms") or 0)
    return " ".join(out).strip()


def _build_block(index: int, run: list[dict[str, Any]], *, max_pause_s: float) -> dict[str, Any]:
    cap_ms = int(round(max_pause_s * 1000))
    parts: list[dict[str, Any]] = []
    for i, seg in enumerate(run):
        nxt = run[i + 1] if i + 1 < len(run) else None
        gap_ms = 0 if nxt is None else max(0, int(round((_f(nxt, "start") - _f(seg, "end")) * 1000)))
        parts.append({
            "segment_id": seg.get("segment_id"),
            "text": _line(seg),
            "source_text": seg.get("source_text") or "",
            "start": round(_f(seg, "start"), 3),
            "end": round(_f(seg, "end"), 3),
            "gap_after_ms": gap_ms,
            "pause_ms": min(gap_ms, cap_ms),
        })

    first, last = run[0], run[-1]
    start = _f(first, "start")
    end = _f(last, "end")
    slot_start = _f(first, "slot_start", start)
    slot_end = _f(last, "slot_end", end)
    if slot_end <= slot_start:
        slot_end = max(end, slot_start + 0.001)
    speech_ms = sum(max(0, int(round((_f(s, "end") - _f(s, "start")) * 1000))) for s in run)
    return {
        "block_id": f"blk_{index:04d}",
        "speaker_id": _speaker(first),
        "voice_id": first.get("voice_id") or _speaker(first),
        "voice_mode": first.get("voice_mode"),
        "segment_ids": [s.get("segment_id") for s in run],
        "start": round(start, 3),
        "end": round(end, 3),
        "slot_start": round(slot_start, 3),
        "slot_end": round(slot_end, 3),
        "target_duration_ms": max(1, int(round((slot_end - slot_start) * 1000))),
        "speech_duration_ms": max(1, speech_ms),
        "text": _join_text(parts),
        "parts": parts,
        "internal_pauses_ms": [p["pause_ms"] for p in parts[:-1]],
        "generated_wav": None,
        "fitted_wav": None,
        "generated_duration_ms": None,
        "fitted_duration_ms": None,
        "stretch_ratio": 1.0,
        "status": "pending",
        "notes": "",
    }


def group_blocks(
    segments: list[dict[str, Any]],
    *,
    block_pause_s: float = BLOCK_PAUSE_S,
    max_block_s: float = MAX_BLOCK_S,
    max_block_chars: int = MAX_BLOCK_CHARS,
    max_pause_s: float = MAX_SYNTH_PAUSE_S,
) -> list[dict[str, Any]]:
    """Group translated cues into speech blocks, in timeline order.

    A block ends on a speaker change, on a pause of `block_pause_s` or more, or
    when the block would outgrow what a TTS engine renders reliably.
    """
    ordered = sorted(
        [s for s in (segments or []) if s.get("segment_id")],
        key=lambda s: (_f(s, "start"), str(s.get("segment_id"))),
    )

    runs: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for seg in ordered:
        if not _dubbable(seg):
            if cur:
                runs.append(cur)
                cur = []
            continue
        if not cur:
            cur = [seg]
            continue
        prev = cur[-1]
        gap = _f(seg, "start") - _f(prev, "end")
        span = _f(seg, "end") - _f(cur[0], "start")
        chars = sum(len(_line(s)) + 1 for s in cur) + len(_line(seg))
        if (
            _speaker(seg) != _speaker(prev)
            or gap >= block_pause_s
            or span > max_block_s
            or chars > max_block_chars
        ):
            runs.append(cur)
            cur = [seg]
        else:
            cur.append(seg)
    if cur:
        runs.append(cur)

    return [_build_block(i, run, max_pause_s=max_pause_s) for i, run in enumerate(runs)]


def stamp_segments(segments: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record each cue's place in its block.

    `block_role` is what the rest of the pipeline keys off: the *lead* cue
    carries the block's rendered audio, the *members* are covered by it, and a
    *solo* cue is still rendered on its own.
    """
    place: dict[str, tuple[str, int, int]] = {}
    for block in blocks:
        ids = block.get("segment_ids") or []
        for i, sid in enumerate(ids):
            place[str(sid)] = (block["block_id"], i, len(ids))
    for seg in segments or []:
        found = place.get(str(seg.get("segment_id")))
        if not found:
            seg["block_id"] = None
            seg["block_index"] = 0
            seg["block_size"] = 1
            seg["block_role"] = "solo"
            continue
        block_id, index, size = found
        seg["block_id"] = block_id
        seg["block_index"] = index
        seg["block_size"] = size
        seg["block_role"] = "lead" if index == 0 else "member"
    return segments


def blocks_path(job_dir: Path) -> Path:
    return job_dir / "segments" / "blocks.json"


def load_blocks(job_dir: Path) -> list[dict[str, Any]]:
    path = blocks_path(job_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a bad cache must not kill the job
        log.warning("Could not read %s: %s", path, exc)
        return []
    if isinstance(data, dict):
        data = data.get("blocks", [])
    return [b for b in data if isinstance(b, dict) and b.get("block_id")]


def run_blocks(job_dir: Path, job: dict | None = None, *, smooth_rates: bool = True,
               **kwargs: Any) -> list[dict[str, Any]]:
    """Build `segments/blocks.json` from the translated cues."""
    seg_path = job_dir / "segments" / "segments.json"
    segments = json.loads(seg_path.read_text(encoding="utf-8"))
    if isinstance(segments, dict):
        segments = segments.get("segments", [])

    blocks = group_blocks(segments, **kwargs)
    if smooth_rates:
        total_s = float(job.get("duration_s") or 0.0) if job else 0.0
        smooth_block_rates(blocks, total_s=total_s or None)
    stamp_segments(segments, blocks)

    solo = [s.get("segment_id") for s in segments if s.get("block_role") == "solo"]
    spans = sorted((b["slot_end"] - b["slot_start"]) for b in blocks)
    median = spans[len(spans) // 2] if spans else 0.0
    log.info(
        "Grouped %d cue(s) into %d speech block(s) (median %.1fs, longest %.1fs); "
        "%d cue(s) left on their own",
        len(segments), len(blocks), median, spans[-1] if spans else 0.0, len(solo),
    )
    if solo:
        log.warning("Cue(s) with no usable translation, they will not be dubbed: %s", solo[:10])

    blocks_path(job_dir).parent.mkdir(parents=True, exist_ok=True)
    blocks_path(job_dir).write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")
    seg_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")

    if job is not None and job.get("job_id"):
        job["block_count"] = len(blocks)
        job["unblocked_segments"] = solo
        try:
            from dubstudio.jobs.store import store

            store.save(job)
        except Exception:  # noqa: BLE001 - progress reporting is best effort
            pass
    return blocks
