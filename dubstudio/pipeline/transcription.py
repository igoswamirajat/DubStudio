from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.transcription")

# Below this, the transcript covers so little of the video that the dub would be
# mostly original audio. Surfaced as a warning on the job instead of shipping it.
MIN_SPEECH_COVERAGE = 0.15


class TranscriptionError(RuntimeError):
    """Raised when ASR is configured but cannot produce a real transcript."""


def _mock_words(duration_s: float) -> list[dict[str, Any]]:
    """Fixed demo transcript. Only reachable when asr_engine == 'mock' (or the
    explicit allow_mock_fallback escape hatch).

    This used to run silently whenever faster-whisper failed, so a 49s video
    became 7s of "Hello and welcome to DubStudio" and 42s of untouched
    original audio.
    """
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
            words.append({"word": tok, "start": round(start + i * step, 3),
                          "end": round(start + (i + 1) * step, 3), "score": 0.99, "speaker": "S00"})
    if not words and duration_s > 0.2:
        words.append({"word": "Hello", "start": 0.1, "end": min(1.0, duration_s - 0.05),
                      "score": 0.9, "speaker": "S00"})
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
                words.append({"word": tok, "start": round(s0 + i * step, 3),
                              "end": round(s0 + (i + 1) * step, 3), "score": 0.9, "speaker": "S00"})
    raw = {
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "duration": getattr(info, "duration", None),
        "segments": seg_dump,
    }
    return info.language or "en", words, raw


def _speech_coverage(words: list[dict[str, Any]], duration_s: float) -> float:
    if not words or duration_s <= 0:
        return 0.0
    spoken = sum(max(0.0, float(w["end"]) - float(w["start"])) for w in words)
    return min(1.0, spoken / duration_s)


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
    allow_mock = bool(getattr(settings, "allow_mock_fallback", False))
    language = "en"
    words: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}

    if settings.asr_engine == "mock":
        engine = "mock"
        words = _mock_words(duration_s)
        raw = {"engine": "mock", "duration_s": duration_s}
    else:
        engine = "faster-whisper"
        try:
            language, words, raw = _faster_whisper_words(vocals, model_name)
        except Exception as exc:
            if not allow_mock:
                # Never silently substitute the demo script for the real video.
                raise TranscriptionError(
                    f"ASR failed with {settings.asr_engine} (model={model_name}): {exc}. "
                    f"Install/repair faster-whisper, or set DUBSTUDIO_ASR_ENGINE=mock to "
                    f"deliberately run the demo transcript."
                ) from exc
            log.exception("faster-whisper failed; allow_mock_fallback is on, using demo words")
            engine = "mock"
            words = _mock_words(duration_s)
            raw = {"engine": "mock", "error": str(exc), "duration_s": duration_s}

        if not words:
            if not allow_mock:
                raise TranscriptionError(
                    f"ASR returned no words for {vocals.name} ({duration_s:.1f}s). "
                    f"The separated vocals track may be silent — check audio/vocals.wav."
                )
            log.warning("faster-whisper returned no words; falling back to demo words")
            engine = "mock"
            words = _mock_words(duration_s)
            raw = {"engine": "mock", "reason": "empty_transcription", "duration_s": duration_s}

    coverage = _speech_coverage(words, duration_s)
    warnings: list[str] = []
    if engine == "mock":
        warnings.append("Transcript is the built-in demo script, not this video.")
    if coverage < MIN_SPEECH_COVERAGE:
        warnings.append(
            f"Transcript covers only {coverage:.0%} of the {duration_s:.0f}s video; "
            f"the rest will keep the original audio."
        )
    for w in warnings:
        log.warning("Transcription QC: %s", w)

    transcript = {
        "engine": engine,
        "language": language,
        "duration_s": duration_s,
        "speech_coverage": round(coverage, 4),
        "warnings": warnings,
        "words": words,
    }
    (asr_dir / "whisper.json").write_text(json.dumps(raw, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (asr_dir / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    return transcript
