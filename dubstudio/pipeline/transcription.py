from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def run_transcription(job_dir: Path, *, model: str = "large-v3-turbo") -> dict:
    asr_dir = job_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    vocals = job_dir / "audio" / "vocals.wav"
    if not vocals.exists():
        vocals = job_dir / "audio" / "full.wav"
    if not vocals.exists():
        raise FileNotFoundError("no audio for transcription")
    duration_s = 0.0
    try:
        import soundfile as sf
        duration_s = float(sf.info(str(vocals)).duration)
    except Exception:
        duration_s = 8.0
    engine = "mock"
    words: list[dict[str, Any]] = []
    raw: dict[str, Any] = {}
    try:
        import whisperx  # type: ignore
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
        audio = whisperx.load_audio(str(vocals))
        model_obj = whisperx.load_model(model, device, compute_type="int8")
        result = model_obj.transcribe(audio, batch_size=8)
        language = result.get("language", "en")
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device)
        except Exception:
            pass
        for seg in result.get("segments", []):
            sp = seg.get("speaker", "S00")
            for w in seg.get("words", []) or []:
                if "word" not in w and "text" in w:
                    w = {**w, "word": w["text"]}
                if not w.get("word"):
                    continue
                words.append({"word": str(w["word"]).strip(), "start": float(w.get("start", seg.get("start", 0))), "end": float(w.get("end", seg.get("end", 0))), "score": float(w.get("score", 1.0) or 1.0), "speaker": sp})
            if not seg.get("words") and seg.get("text"):
                text = seg["text"].strip().split()
                s0, s1 = float(seg["start"]), float(seg["end"])
                step = (s1 - s0) / max(1, len(text))
                for i, tok in enumerate(text):
                    words.append({"word": tok, "start": s0 + i * step, "end": s0 + (i + 1) * step, "score": 0.9, "speaker": sp})
        engine = "whisperx"
        raw = {"language": language, "segments": result.get("segments", [])}
    except Exception as exc:
        words = _mock_words(duration_s)
        raw = {"engine": "mock", "error": str(exc), "duration_s": duration_s}
    language = raw.get("language") or "en"
    transcript = {"engine": engine, "language": language, "duration_s": duration_s, "words": words}
    (asr_dir / "whisperx.json").write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    (asr_dir / "transcript.json").write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return transcript
