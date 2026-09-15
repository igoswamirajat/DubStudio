from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.diarization")

# --- label smoothing --------------------------------------------------------
# pyannote gives us turns; every ASR word then asks "who was talking here?".
# Words landing in a gap between turns used to be handed to the *nearest* turn,
# which sprays 0.1-0.2 s speaker flickers through a single sentence. Each
# flicker becomes its own speech block, its own TTS call and its own voice - so
# one sentence could come out in two voices. These rules kill flickers while
# leaving genuine turn taking intact.
MIN_TURN_S = 0.60        # nobody takes a turn shorter than this
TINY_RUN_S = 0.35        # ... and a crumb this short is never a turn at all
TINY_RUN_WORDS = 2
SMOOTH_PASSES = 4


def _load_pipeline():
    import torch
    from pyannote.audio import Pipeline

    token = settings.hf_token or None
    try:
        pipeline = Pipeline.from_pretrained(settings.diarization_model, token=token)
    except TypeError:
        # older pyannote uses use_auth_token
        pipeline = Pipeline.from_pretrained(settings.diarization_model, use_auth_token=token)
    if pipeline is None:
        raise RuntimeError(
            "pyannote returned no pipeline - the model may be gated. Accept the terms "
            f"for '{settings.diarization_model}' on HuggingFace and set DUBSTUDIO_HF_TOKEN."
        )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline


def _turns(audio_path: Path) -> list[tuple[float, float, str]]:
    pipeline = _load_pipeline()
    kwargs: dict[str, Any] = {}
    if settings.max_speakers and settings.max_speakers > 0:
        kwargs["max_speakers"] = settings.max_speakers
    diarization = pipeline(str(audio_path), **kwargs)
    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    turns.sort(key=lambda t: t[0])
    return turns


def _speaker_at(turns: list[tuple[float, float, str]], t: float) -> str | None:
    """Kept for compatibility: who was speaking at one instant."""
    best = None
    best_overlap = 0.0
    for start, end, spk in turns:
        if start <= t <= end:
            return spk
        gap = min(abs(t - start), abs(t - end))
        overlap = 1.0 / (1.0 + gap)
        if overlap > best_overlap:
            best_overlap = overlap
            best = spk
    return best


def _speaker_for(turns: list[tuple[float, float, str]], start: float, end: float) -> str | None:
    """Speaker whose turn covers most of the WORD, not the one nearest its mid.

    A word is an interval, so score it as one. Only a word that overlaps nothing
    at all falls back to the nearest turn, and then by real distance rather than
    a 1/(1+gap) score that treats 0.02 s and 2 s as nearly the same thing.
    """
    if not turns:
        return None
    if end < start:
        start, end = end, start

    best: str | None = None
    best_overlap = 0.0
    for t_start, t_end, spk in turns:
        overlap = min(end, t_end) - max(start, t_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = spk
    if best is not None and best_overlap > 0.0:
        return best

    mid = (start + end) / 2.0
    nearest = None
    nearest_gap = float("inf")
    for t_start, t_end, spk in turns:
        gap = 0.0 if t_start <= mid <= t_end else min(abs(mid - t_start), abs(mid - t_end))
        if gap < nearest_gap:
            nearest_gap = gap
            nearest = spk
    return nearest


def _runs(words: list[dict]) -> list[dict]:
    """Consecutive words sharing a label, with their real time span."""
    runs: list[dict] = []
    for idx, word in enumerate(words):
        spk = str(word.get("speaker", "S00"))
        if runs and runs[-1]["speaker"] == spk:
            runs[-1]["end_idx"] = idx
        else:
            runs.append({"speaker": spk, "start_idx": idx, "end_idx": idx})
    for run in runs:
        first = words[run["start_idx"]]
        last = words[run["end_idx"]]
        run["start"] = float(first.get("start", 0.0) or 0.0)
        run["end"] = float(last.get("end", run["start"]) or run["start"])
        run["duration"] = max(0.0, run["end"] - run["start"])
        run["words"] = run["end_idx"] - run["start_idx"] + 1
    return runs


def smooth_word_speakers(
    words: list[dict],
    *,
    min_turn_s: float = MIN_TURN_S,
    tiny_run_s: float = TINY_RUN_S,
    tiny_run_words: int = TINY_RUN_WORDS,
    passes: int = SMOOTH_PASSES,
) -> int:
    """Remove speaker flickers in place. Returns the number of words relabelled.

    Two rules, applied until stable:
      * sandwich - a run shorter than min_turn_s between two runs of the SAME
        other label becomes that label (A B A -> A A A);
      * crumb - a one or two word run under tiny_run_s joins whichever
        neighbour talked longer.
    A real short turn between two DIFFERENT speakers is left alone, so genuine
    back-and-forth dialogue survives.
    """
    if len(words) < 3:
        return 0

    relabelled = 0
    for _ in range(max(1, passes)):
        runs = _runs(words)
        if len(runs) < 2:
            break
        changed = 0
        for i, run in enumerate(runs):
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            target: str | None = None

            if prev and nxt and prev["speaker"] == nxt["speaker"] and run["duration"] < min_turn_s:
                target = prev["speaker"]
            elif run["duration"] < tiny_run_s and run["words"] <= tiny_run_words:
                if prev and nxt:
                    target = prev["speaker"] if prev["duration"] >= nxt["duration"] else nxt["speaker"]
                elif prev or nxt:
                    target = (prev or nxt)["speaker"]

            if target and target != run["speaker"]:
                for word in words[run["start_idx"]:run["end_idx"] + 1]:
                    word["speaker"] = target
                changed += run["words"]

        relabelled += changed
        if changed == 0:
            break
    return relabelled


def run_diarization(job_dir: Path, transcript: dict) -> dict:
    """Assign a real speaker label to each transcript word using pyannote.

    On any failure (disabled, no token, gated model, error) the transcript is
    left single-speaker (S00). Returns a speakers.json payload and writes it.
    """
    asr_dir = job_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    words = transcript.get("words", [])

    audio = job_dir / "audio" / "vocals.wav"
    if not audio.exists():
        audio = job_dir / "audio" / "full.wav"

    label_map: dict[str, str] = {}
    engine = "single"
    if settings.enable_diarization and audio.exists() and words:
        try:
            turns = _turns(audio)
            if turns:
                order: list[str] = []
                for w in words:
                    start = float(w.get("start", 0) or 0)
                    end = float(w.get("end", start) or start)
                    raw = _speaker_for(turns, start, end)
                    if raw is None:
                        raw = turns[0][2]
                    if raw not in label_map:
                        label_map[raw] = f"S{len(order):02d}"
                        order.append(raw)
                    w["speaker"] = label_map[raw]
                relabelled = smooth_word_speakers(words)
                if relabelled:
                    log.info("diarization smoothing relabelled %d of %d words (flicker removal)",
                             relabelled, len(words))
                engine = "pyannote"
                (asr_dir / "diarization.json").write_text(
                    json.dumps(
                        {
                            "model": settings.diarization_model,
                            "turns": turns,
                            "smoothing": {
                                "min_turn_s": MIN_TURN_S,
                                "tiny_run_s": TINY_RUN_S,
                                "relabelled_words": relabelled,
                            },
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            else:
                log.warning("pyannote produced no turns; single speaker")
        except Exception:
            log.exception("diarization failed; single speaker")

    if engine != "pyannote":
        for w in words:
            w.setdefault("speaker", "S00")

    speaker_ids = sorted({str(w.get("speaker", "S00")) for w in words}) or ["S00"]
    speakers = {
        "engine": engine,
        "speakers": [
            {"speaker_id": s, "label": f"Speaker {int(s[1:]) + 1 if s[1:].isdigit() else s}"}
            for s in speaker_ids
        ],
    }
    (asr_dir / "speakers.json").write_text(json.dumps(speakers, indent=2), encoding="utf-8")
    (asr_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    return speakers
