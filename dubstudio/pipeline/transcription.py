from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.transcription")


def _mock_words(duration_s: float) -> list[dict[str, Any]]:
    lines = [
        (0.40, 2.20, "Hello and welcome to DubStudio."),
        (2.50, 4.80, "This is a short demo clip for testing."),
        (5.10, 7.50, "We will translate and dub this speech."),
    ]
    words: list[dict[str, Any]] = []
    for start, end, text in lines:
        if start >= duration_s:
            break
        end = min(end, duration_s - 0.05)
        tokens = text.split()
        if not tokens:
            continue
        span = max(0.05, end - start)
        step = span / len(tokens)
        for i, tok in enumerate(tokens):
            ws = start + i * step
            we = start + (i + 1) * step
            words.append({"word": tok, "start": round(ws, 3), "end": round(we, 3), "score": 0.99, "speaker": "S00"})
    if not words and duration_s > 0.2:
        words.append({"word": "Hello", "start": 0.1, "end": min(1.0, duration_s - 0.05), "score": 0.9, "speaker": "S00"})
    return words


def _resolve_device_compute() -> tuple[str, str]:
    device = settings.whisper_device
    compute = settings.whisper_compute_type
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _faster_whisper_words(vocals: Path, model_name: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    from faster_whisper import WhisperModel

    device, compute = _resolve_device_compute()
    log.info("Loading faster-whisper %s on %s (%s)", model_name, device, compute)
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, info = model.transcribe(
        str(vocals),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    words: list[dict[str, Any]] = []
    seg_dump: list[dict[str, Any]] = []
    for seg in segments:
        seg_dump.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if seg.words:
            for w in seg.words:
                token = (w.word or "").strip()
                if not token:
                    continue
                words.append({
                    "word": token,
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "score": round(float(w.probability or 1.0), 4),
                    "speaker": "S00",
                })
        elif seg.text.strip():
            toks = seg.text.strip().split()
            s0, s1 = float(seg.start), float(seg.end)
            step = (s1 - s0) / max(1, len(toks))
            for i, tok in enumerate(toks):
                words.append({"word": tok, "start": round(s0 + i * step, 3), "end": round(s0 + (i + 1) * step, 3), "score": 0.9, "speaker": "S00"})
    raw = {
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": seg_dump,
    }
    return info.language or "en", words, raw


def run_transcription(job_dir: Path, *, model: str | None = None) -> dict:
    asr_dir = job_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    vocals = job_dir / "audio" / "vocals.wav"
    if not vocals.exists():
        vocals = job_dir / "audio" / "full.wav"
    if not vocals.exists():
        raise FileNotFoundError("no audio for transcription")

    duration_s = 8.0
    try:
        import soundfile as sf

        duration_s = float(sf.info(str(vocals)).duration)
    except Exception:
        pass

    model_name = model or settings.whisper_model
    engine = "mock"
    language = "en"
    words: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}

    if settings.asr_engine != "mock":
        try:
            language, words, raw = _faster_whisper_words(vocals, model_name)
            engine = "faster-whisper"
            if not words:
                log.warning("faster-whisper returned no words; falling back to mock")
                engine = "mock"
                words = _mock_words(duration_s)
                raw = {"engine": "mock", "reason": "empty_transcription", "duration_s": duration_s}
        except Exception as exc:
            log.exception("faster-whisper failed, using mock words")
            words = _mock_words(duration_s)
            raw = {"engine": "mock", "error": str(exc), "duration_s": duration_s}
    else:
        words = _mock_words(duration_s)
        raw = {"engine": "mock", "duration_s": duration_s}

    transcript = {"engine": engine, "language": language, "duration_s": duration_s, "words": words}
    (asr_dir / "whisper.json").write_text(json.dumps(raw, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (asr_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    return transcript
