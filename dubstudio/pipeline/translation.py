from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.translation")

# A translated line must be at least this fraction of target-script letters,
# otherwise the model echoed the source instead of translating it.
MIN_TARGET_SCRIPT_RATIO = 0.30
# Latin-script targets have no script signal, so fall back to text similarity.
MAX_SOURCE_SIMILARITY = 0.90


class TranslationError(RuntimeError):
    """Raised when the configured translator cannot produce usable output.

    Shipping an un-translated dub is worse than failing: the viewer gets the
    original language back with a synthetic voice on top. Fail loudly instead.
    """


# --------------------------------------------------------------------------
# Demo-only word map. This is NOT a production fallback: it is reachable only
# when settings.translator == "demo". Historically it ran silently whenever the
# LLM was unreachable, which produced output like
#   "So today I am going में show you how में scrape any website के लिए free"
# i.e. three words translated and the rest left in the source language.
# --------------------------------------------------------------------------
_DEMO_HI = {
    "hello": "नमस्ते", "and": "और", "welcome": "स्वागत है",
    "to": "में", "dubstudio": "डबस्टूडियो", "this": "यह", "is": "है",
    "a": "एक", "short": "छोटा", "demo": "डेमो", "clip": "क्लिप",
    "for": "के लिए", "testing": "परीक्षण", "we": "हम", "will": "करेंगे",
    "translate": "अनुवाद", "dub": "डब", "speech": "भाषण",
}


def _simple_map(text: str, target: str) -> str:
    """Demo-mode only. See _DEMO_HI."""
    if target not in {"hi", "hin"}:
        return f"[{target}] {text}"
    out = []
    for tok in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        out.append(_DEMO_HI.get(tok.lower(), tok))
    return " ".join(out)


_LANG_NAMES = {
    "hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi",
    "ur": "Urdu", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
    "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
}

# Expected writing system per target language, used to verify the model really
# switched languages instead of returning the source line.
_TARGET_SCRIPTS: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {
    "hi": ("Devanagari", ((0x0900, 0x097F),)),
    "mr": ("Devanagari", ((0x0900, 0x097F),)),
    "ne": ("Devanagari", ((0x0900, 0x097F),)),
    "bn": ("Bengali", ((0x0980, 0x09FF),)),
    "gu": ("Gujarati", ((0x0A80, 0x0AFF),)),
    "pa": ("Gurmukhi", ((0x0A00, 0x0A7F),)),
    "ta": ("Tamil", ((0x0B80, 0x0BFF),)),
    "te": ("Telugu", ((0x0C00, 0x0C7F),)),
    "kn": ("Kannada", ((0x0C80, 0x0CFF),)),
    "ml": ("Malayalam", ((0x0D00, 0x0D7F),)),
    "or": ("Odia", ((0x0B00, 0x0B7F),)),
    "ur": ("Arabic", ((0x0600, 0x06FF),)),
    "ar": ("Arabic", ((0x0600, 0x06FF),)),
    "fa": ("Arabic", ((0x0600, 0x06FF),)),
    "he": ("Hebrew", ((0x0590, 0x05FF),)),
    "ru": ("Cyrillic", ((0x0400, 0x04FF),)),
    "uk": ("Cyrillic", ((0x0400, 0x04FF),)),
    "el": ("Greek", ((0x0370, 0x03FF),)),
    "th": ("Thai", ((0x0E00, 0x0E7F),)),
    "ko": ("Hangul", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    "ja": ("Japanese", ((0x3040, 0x30FF), (0x4E00, 0x9FFF))),
    "zh": ("Chinese", ((0x4E00, 0x9FFF),)),
}


def _base_lang(code: str) -> str:
    return (code or "").lower().split("-")[0].strip()


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get(_base_lang(code), code or "the target language")


def _script_ratio(text: str, ranges: tuple[tuple[int, int], ...]) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(1 for c in letters if any(lo <= ord(c) <= hi for lo, hi in ranges))
    return hits / len(letters)


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return " ".join(cleaned.split())


def _word_count(text: str) -> int:
    return len(re.findall(r"\w{2,}", text or "", flags=re.UNICODE))


def check_translation(source: str, output: str, target: str) -> str | None:
    """Return a human-readable reason when `output` is not a real translation.

    Returns None when the line looks genuinely translated.
    """
    output = (output or "").strip()
    if not output:
        return "empty output"
    # Very short lines (a brand name, "Okay.", an interjection) can legitimately
    # survive a good translation unchanged, so they are not judged.
    if _word_count(source) < 3:
        return None

    src_norm, out_norm = _normalize(source), _normalize(output)
    if out_norm and out_norm == src_norm:
        return "output is identical to the source line"

    entry = _TARGET_SCRIPTS.get(_base_lang(target))
    if entry:
        # A wrong script is the strongest signal that the model echoed the source.
        name, ranges = entry
        ratio = _script_ratio(output, ranges)
        if ratio < MIN_TARGET_SCRIPT_RATIO:
            return (f"only {ratio:.0%} of letters are {name}; expected at least "
                    f"{MIN_TARGET_SCRIPT_RATIO:.0%} (model likely echoed the source)")
    elif src_norm and out_norm:
        if difflib.SequenceMatcher(None, src_norm, out_norm).ratio() >= MAX_SOURCE_SIMILARITY:
            return "output is near-identical to the source line"
    return None


def _clean_llm_output(text: str) -> str:
    # Strip reasoning blocks some models emit, then surrounding quotes/labels.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^\s*(translation|translated|output)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip("'").strip()


def _build_prompt(text: str, *, source: str, target: str, context: list[str],
                  target_duration_ms: int, strict: bool = False) -> tuple[str, str]:
    src_name = _lang_name(source)
    tgt_name = _lang_name(target)
    target_sec = max(0.5, target_duration_ms / 1000.0)
    # Native conversational tempo: ~2.5 - 3.0 words per second
    max_words = max(3, int(target_sec * 2.8))
    script_name = (_TARGET_SCRIPTS.get(_base_lang(target)) or (None, None))[0]

    system = (
        f"You are a master voice dubbing localizer adapting {src_name} speech into conversational, "
        f"natural {tgt_name} for video dubbing. You are NOT a literal translator.\n"
        "Core Rules:\n"
        f"1. Conversational Brevity & Rhythm: Match the spoken tempo and punchiness of the source line. "
        f"The line must comfortably fit within {target_sec:.1f} seconds. Limit your translation to at most "
        f"{max_words} words so the actor speaks naturally with breath pauses without rushing.\n"
        "2. Meaning over Word-for-Word: Convey the core meaning the way a real speaker talks in casual conversation. "
        "Do NOT use formal, bookish, or elongated sentence structures. Keep it concise, punchy, and modern.\n"
        "3. Technical Terms & Loan Words: Adapt technical terms naturally for Indian speech "
        "(e.g. use standard words like Python, GitHub, scrape, website, free, open source, AI, API).\n"
        f"4. Output ONLY the localized {tgt_name} line — no quotes, no explanations, no notes."
    )
    if strict:
        system += (
            f"\n\nCRITICAL: Your previous reply was not usable. Reply ONLY in {tgt_name}"
            + (f", written in the {script_name} script" if script_name else "")
            + f". Never repeat the {src_name} sentence back. Output one single {tgt_name} line and nothing else."
        )

    ctx = "\n".join(context[-3:])
    user = (
        (f"Context from earlier dialogue:\n{ctx}\n\n" if ctx else "")
        + f"Target duration: {target_sec:.1f}s (max {max_words} words)\n"
        f"Adapt this line into concise conversational {tgt_name}:\n{text}"
    )
    return system, user


def _ollama_translate(text: str, *, source: str, target: str, context: list[str],
                      target_duration_ms: int, strict: bool = False) -> str | None:
    host = settings.ollama_host.rstrip("/")
    system, user = _build_prompt(text, source=source, target=target, context=context,
                                 target_duration_ms=target_duration_ms, strict=strict)
    try:
        r = httpx.post(
            f"{host}/api/chat",
            json={
                "model": settings.ollama_model,
                "stream": False,
                "options": {"temperature": 0.3},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120.0,
        )
        if r.status_code != 200:
            # Previously swallowed in silence, which is how a dead Ollama looked
            # exactly like a successful job.
            log.warning("Ollama translation HTTP %s from %s (model=%s): %s",
                        r.status_code, host, settings.ollama_model, r.text[:200])
            return None
        return _clean_llm_output((r.json().get("message") or {}).get("content") or "") or None
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not raised
        log.warning("Ollama translation call to %s failed: %s", host, exc)
        return None


def _openai_translate(text: str, *, source: str, target: str, context: list[str],
                      target_duration_ms: int, strict: bool = False) -> str | None:
    """Any OpenAI-compatible chat API (OpenAI, Groq, DeepSeek, OpenRouter, Ollama /v1, ...)."""
    base = settings.openai_base_url.rstrip("/")
    system, user = _build_prompt(text, source=source, target=target, context=context,
                                 target_duration_ms=target_duration_ms, strict=strict)
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    try:
        r = httpx.post(
            f"{base}/chat/completions",
            headers=headers,
            json={
                "model": settings.openai_model,
                "temperature": 0.3,
                "max_tokens": 256,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=45.0,
        )
        if r.status_code != 200:
            log.warning("OpenAI translation returned HTTP %s: %s", r.status_code, r.text[:200])
            return None
        choices = r.json().get("choices") or []
        if not choices:
            log.warning("OpenAI translation returned empty choices: %s", r.text[:200])
            return None
        return _clean_llm_output((choices[0].get("message") or {}).get("content") or "") or None
    except Exception as exc:  # noqa: BLE001
        log.warning("OpenAI translation call failed: %s", exc)
        return None


def _translator_fn():
    if settings.translator == "ollama":
        return _ollama_translate, f"ollama:{settings.ollama_model} @ {settings.ollama_host}"
    if settings.translator == "openai":
        return _openai_translate, f"openai:{settings.openai_model} @ {settings.openai_base_url}"
    if settings.translator == "demo":
        return None, "demo"
    raise TranslationError(
        f"Unknown translator {settings.translator!r}. Set DUBSTUDIO_TRANSLATOR to "
        f"'ollama', 'openai' or 'demo'."
    )


def preflight(*, source_language: str, target_language: str) -> str:
    """Verify the translator works BEFORE translating a whole video.

    Returns the engine description. Raises TranslationError with an actionable
    message when the endpoint is down, the model is missing, or the model just
    echoes the source text back.
    """
    fn, desc = _translator_fn()
    if fn is None:
        log.warning("translator=demo — output will be a word-for-word demo map, not a translation")
        return desc

    probe = "Today I will show you how this tool works, step by step."
    reply = fn(probe, source=source_language or "en", target=target_language,
               context=[], target_duration_ms=3000)
    if not reply:
        raise TranslationError(
            f"Translator unreachable: {desc} returned nothing. "
            f"Check the service is running and the model is pulled "
            f"(e.g. `ollama list` / `ollama pull {settings.ollama_model}`), or set "
            f"DUBSTUDIO_TRANSLATOR=openai with DUBSTUDIO_OPENAI_API_KEY."
        )
    reason = check_translation(probe, reply, target_language)
    if reason:
        raise TranslationError(
            f"Translator {desc} is reachable but is not translating into "
            f"{_lang_name(target_language)} ({reason}). Model replied: {reply[:160]!r}. "
            f"Try a stronger model (DUBSTUDIO_OLLAMA_MODEL)."
        )
    log.info("Translation preflight OK via %s", desc)
    return desc


def translate_segments(
    segments: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    job: dict | None = None,
    allow_partial: bool | None = None,
) -> list[dict[str, Any]]:
    if allow_partial is None:
        allow_partial = not bool(getattr(settings, "strict_translation", True))

    fn, desc = _translator_fn()
    if fn is not None:
        preflight(source_language=source_language, target_language=target_language)

    attempts = max(1, int(getattr(settings, "translation_attempts", 2)))
    context: list[str] = []
    failures: list[tuple[str, str]] = []
    total = len(segments)

    for i, seg in enumerate(segments):
        if job and job.get("job_id") and total > 0:
            try:
                from dubstudio.jobs.store import store

                job["percent"] = min(58 + int((i / total) * 7), 64)
                job["message"] = f"Translating segment {i + 1}/{total}"
                store.save(job)
            except Exception:
                pass

        text = seg.get("source_text") or ""
        seg_id = seg.get("segment_id") or f"index_{i}"

        if fn is None:  # demo mode, explicitly selected
            seg["translated_text"] = _simple_map(text, target_language)
            seg["translation_engine"] = "demo"
            seg["status"] = "translated"
            continue

        if not text.strip():
            seg["translated_text"] = ""
            seg["translation_engine"] = desc
            seg["status"] = "translated"
            continue

        translated, reason = None, "no attempt made"
        for attempt in range(1, attempts + 1):
            candidate = fn(text, source=source_language or "en", target=target_language,
                           context=context,
                           target_duration_ms=int(seg.get("target_duration_ms") or 2000),
                           strict=attempt > 1)
            if not candidate:
                reason = "translator returned nothing"
                log.warning("Segment %s attempt %d/%d: %s", seg_id, attempt, attempts, reason)
                continue
            reason = check_translation(text, candidate, target_language) or ""
            if not reason:
                translated = candidate
                break
            log.warning("Segment %s attempt %d/%d rejected: %s", seg_id, attempt, attempts, reason)

        if translated is None:
            seg["translated_text"] = ""
            seg["status"] = "translation_failed"
            seg["error"] = reason or "translation failed"
            seg["translation_engine"] = desc
            failures.append((seg_id, seg["error"]))
            continue

        try:
            from dubstudio.util.phonetics import normalize_hinglish

            translated = normalize_hinglish(translated, target_language)
        except Exception:
            pass

        seg["translated_text"] = translated
        seg["translation_engine"] = desc
        seg["status"] = "translated"
        seg.pop("error", None)
        context.append(f"{text} => {translated}")

    if failures:
        head = "; ".join(f"{sid}: {why}" for sid, why in failures[:5])
        msg = (f"{len(failures)}/{total} segment(s) were not translated via {desc}. "
               f"First failures — {head}")
        if not allow_partial:
            raise TranslationError(msg)
        log.error("%s (continuing because strict_translation is off)", msg)

    return segments


def verify_segments_translated(segments: list[dict[str, Any]], target_language: str) -> list[str]:
    """Return the ids of segments whose stored translation is unusable.

    Used on resume so a cached bad `segments.json` is re-translated instead of
    being replayed into synthesis.
    """
    bad: list[str] = []
    for i, seg in enumerate(segments):
        src = (seg.get("source_text") or "").strip()
        if not src:
            continue
        if check_translation(src, seg.get("translated_text") or "", target_language):
            bad.append(seg.get("segment_id") or f"index_{i}")
    return bad


def run_translation(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get("segments", [])
    try:
        segments = translate_segments(
            segments,
            source_language=job.get("source_language") or "en",
            target_language=job.get("target_language") or "hi",
            job=job,
        )
    finally:
        # Always persist what we have so failures stay inspectable on disk.
        path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
