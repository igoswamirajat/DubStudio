from __future__ import annotations

from typing import Any

# --- segmentation tuning ---------------------------------------------------
# A dub only sounds continuous if each synthesis unit is a whole utterance.
# Splitting on every 0.4s pause turned a 49s clip into 44 fragments (median
# 0.61s, several 0.12-0.20s), and every fragment became its own TTS call, its
# own time-stretch ratio and its own hard-edited join - which is what makes the
# dub stutter after each line.
SPLIT_PAUSE_S = 0.70   # a genuine breath / turn boundary
PUNCT_PAUSE_S = 0.28   # after sentence-final punctuation a short pause is enough
MIN_SEG_S = 1.20       # shorter units are merged into a neighbour
MAX_SEG_S = 12.0
# Gaps at or below this are absorbed into the preceding segment's slot so the
# dialogue bus stays continuous instead of dropping out between lines.
ABSORB_GAP_S = 0.35
TAIL_PAD_S = 0.12      # breathing room before a genuine pause

_SENT_END = ".?!\u0964\u2026"
_TRAILING = "\"')]\u201d\u2019"


def _norm_speaker(raw: str | None) -> str:
    if not raw:
        return "S00"
    s = raw.upper().replace("SPEAKER_", "S")
    if s.startswith("S") and s[1:].isdigit():
        return f"S{int(s[1:]):02d}"
    if s.isdigit():
        return f"S{int(s):02d}"
    return "S00"


def _ends_sentence(word: str | None) -> bool:
    w = (word or "").strip()
    while w and w[-1] in _TRAILING:
        w = w[:-1]
    return bool(w) and w[-1] in _SENT_END


def _dur(unit: list[dict[str, Any]]) -> float:
    return float(unit[-1]["end"]) - float(unit[0]["start"])


def _gap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    return max(0.0, float(b[0]["start"]) - float(a[-1]["end"]))


def _spk(unit: list[dict[str, Any]]) -> str:
    return _norm_speaker(unit[0].get("speaker"))


def _group(words: list[dict[str, Any]], *, max_pause_s: float, max_seg_s: float) -> list[list[dict[str, Any]]]:
    units: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        if not cur:
            cur = [w]
            continue
        prev = cur[-1]
        pause = float(w["start"]) - float(prev["end"])
        speaker_change = _norm_speaker(w.get("speaker")) != _spk(cur)
        too_long = (float(w["end"]) - float(cur[0]["start"])) > max_seg_s
        # Only break mid-utterance for a real breath; a short pause after a full
        # stop is a sentence boundary and is a good place to split.
        sentence_break = _ends_sentence(prev.get("word")) and pause >= PUNCT_PAUSE_S
        if speaker_change or pause >= max_pause_s or sentence_break or too_long:
            units.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        units.append(cur)
    return units


def _merge_short(units: list[list[dict[str, Any]]], *, min_seg_s: float, max_seg_s: float) -> list[list[dict[str, Any]]]:
    """Fold sub-`min_seg_s` units into the adjacent same-speaker unit.

    A 0.15s "segment" cannot be synthesised with usable prosody: the engine has
    no context, adds its own lead-in/tail silence, and the timing stage then
    stretches it by an extreme ratio to fill its slot.
    """
    units = [list(u) for u in units]
    guard = 0
    while len(units) > 1 and guard < 10000:
        guard += 1
        target = None
        for i, u in enumerate(units):
            if _dur(u) >= min_seg_s:
                continue
            options = []
            if i > 0 and _spk(units[i - 1]) == _spk(u) and (
                float(u[-1]["end"]) - float(units[i - 1][0]["start"])
            ) <= max_seg_s:
                options.append((_gap(units[i - 1], u), i - 1))
            if i + 1 < len(units) and _spk(units[i + 1]) == _spk(u) and (
                float(units[i + 1][-1]["end"]) - float(u[0]["start"])
            ) <= max_seg_s:
                options.append((_gap(u, units[i + 1]), i))
            if options:
                options.sort(key=lambda o: o[0])
                target = options[0][1]
                break
        if target is None:
            break
        units[target:target + 2] = [units[target] + units[target + 1]]
    return units


def words_to_segments(
    words: list[dict[str, Any]],
    *,
    job_id: str,
    max_pause_s: float = SPLIT_PAUSE_S,
    max_seg_s: float = MAX_SEG_S,
    min_seg_s: float = MIN_SEG_S,
) -> list[dict[str, Any]]:
    """Turn ASR words into utterance-level dubbing cues.

    Each cue carries both the real speech window (`start`/`end`) and the slot the
    synthesised line may occupy (`slot_start`/`slot_end`). Short inter-utterance
    gaps are folded into the preceding slot, so consecutive lines are contiguous
    and the dub does not go quiet between every line.
    """
    words = [w for w in (words or []) if (w.get("word") or "").strip()]
    if not words:
        return []

    units = _group(words, max_pause_s=max_pause_s, max_seg_s=max_seg_s)
    if min_seg_s > 0:
        units = _merge_short(units, min_seg_s=min_seg_s, max_seg_s=max_seg_s)

    cues: list[dict[str, Any]] = []
    for idx, unit in enumerate(units):
        start = float(unit[0]["start"])
        end = float(unit[-1]["end"])
        text = " ".join(w["word"].strip() for w in unit if w.get("word")).strip()
        if not text:
            continue

        next_start = float(units[idx + 1][0]["start"]) if idx + 1 < len(units) else None
        gap_after = max(0.0, next_start - end) if next_start is not None else None
        if gap_after is not None and gap_after <= ABSORB_GAP_S:
            slot_end = next_start          # contiguous - no micro-pause in the dub
            absorbed = True
        elif gap_after is not None:
            slot_end = end + TAIL_PAD_S    # keep the pause the speaker really took
            absorbed = False
        else:
            slot_end = end
            absorbed = False

        cues.append({
            "job_id": job_id,
            "segment_id": f"seg_{len(cues):04d}",
            "speaker_id": _spk(unit),
            "voice_id": _spk(unit),
            "start": start,
            "end": end,
            "slot_start": start,
            "slot_end": round(slot_end, 3),
            "gap_after_ms": None if gap_after is None else int(round(gap_after * 1000)),
            "absorbed_gap": absorbed,
            "speech_duration_ms": max(1, int(round((end - start) * 1000))),
            "target_duration_ms": max(1, int(round((slot_end - start) * 1000))),
            "source_text": text,
            "translated_text": "",
            "source_words": [
                {"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
                 "score": float(w.get("score", 1.0))}
                for w in unit
            ],
            "generated_wav": None,
            "fitted_wav": None,
            "generated_duration_ms": None,
            "fitted_duration_ms": None,
            "stretch_ratio": 1.0,
            "rewrite_attempts": 0,
            "status": "pending",
            "notes": "",
        })
    return cues
