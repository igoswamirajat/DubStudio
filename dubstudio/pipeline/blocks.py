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


def run_blocks(job_dir: Path, job: dict | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """Build `segments/blocks.json` from the translated cues."""
    seg_path = job_dir / "segments" / "segments.json"
    segments = json.loads(seg_path.read_text(encoding="utf-8"))
    if isinstance(segments, dict):
        segments = segments.get("segments", [])

    blocks = group_blocks(segments, **kwargs)
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
