from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.diarization")


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
            "pyannote returned no pipeline — model may be gated. Accept the terms "
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
                    mid = (float(w.get("start", 0)) + float(w.get("end", 0))) / 2.0
                    raw = _speaker_at(turns, mid)
                    if raw is None:
                        raw = turns[0][2]
                    if raw not in label_map:
                        label_map[raw] = f"S{len(order):02d}"
                        order.append(raw)
                    w["speaker"] = label_map[raw]
                engine = "pyannote"
                (asr_dir / "diarization.json").write_text(
                    json.dumps({"model": settings.diarization_model, "turns": turns}, indent=2, default=str),
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
    (asr_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    return speakers
