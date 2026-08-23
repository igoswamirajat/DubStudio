from __future__ import annotations

from typing import Any


def _norm_speaker(raw: str | None) -> str:
    if not raw:
        return "S00"
    s = raw.upper().replace("SPEAKER_", "S")
    if s.startswith("S") and s[1:].isdigit():
        return f"S{int(s[1:]):02d}"
    if s.isdigit():
        return f"S{int(s):02d}"
    return "S00"


def words_to_segments(words: list[dict[str, Any]], *, job_id: str, max_pause_s: float = 0.4, max_seg_s: float = 8.0) -> list[dict[str, Any]]:
    if not words:
        return []
    cues: list[dict[str, Any]] = []
    cur_words: list[dict[str, Any]] = []
    cur_speaker = _norm_speaker(words[0].get("speaker"))

    def flush() -> None:
        nonlocal cur_words, cur_speaker
        if not cur_words:
            return
        start = float(cur_words[0]["start"])
        end = float(cur_words[-1]["end"])
        text = " ".join(w["word"].strip() for w in cur_words if w.get("word")).strip()
        if not text:
            cur_words = []
            return
        idx = len(cues)
        cues.append({
            "job_id": job_id,
            "segment_id": f"seg_{idx:04d}",
            "speaker_id": cur_speaker,
            "voice_id": cur_speaker,
            "start": start,
            "end": end,
            "target_duration_ms": max(1, int(round((end - start) * 1000))),
            "source_text": text,
            "translated_text": "",
            "source_words": [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]), "score": float(w.get("score", 1.0))} for w in cur_words],
            "generated_wav": None,
            "fitted_wav": None,
            "generated_duration_ms": None,
            "fitted_duration_ms": None,
            "stretch_ratio": 1.0,
            "rewrite_attempts": 0,
            "status": "pending",
            "notes": "",
        })
        cur_words = []

    for w in words:
        sp = _norm_speaker(w.get("speaker"))
        if not cur_words:
            cur_speaker = sp
            cur_words = [w]
            continue
        prev_end = float(cur_words[-1]["end"])
        start = float(w["start"])
        pause = start - prev_end
        duration_if = float(w["end"]) - float(cur_words[0]["start"])
        if sp != cur_speaker or pause >= max_pause_s or duration_if > max_seg_s:
            flush()
            cur_speaker = sp
            cur_words = [w]
        else:
            cur_words.append(w)
    flush()
    return cues
