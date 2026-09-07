from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from dubstudio.settings import settings

_DEMO_HI = {
    "hello": "\u0928\u092e\u0938\u094d\u0924\u0947", "and": "\u0914\u0930", "welcome": "\u0938\u094d\u0935\u093e\u0917\u0924 \u0939\u0948",
    "to": "\u092e\u0947\u0902", "dubstudio": "\u0921\u092c\u0938\u094d\u091f\u0942\u0921\u093f\u092f\u094b", "this": "\u092f\u0939", "is": "\u0939\u0948",
    "a": "\u090f\u0915", "short": "\u091b\u094b\u091f\u093e", "demo": "\u0921\u0947\u092e\u094b", "clip": "\u0915\u094d\u0932\u093f\u092a",
    "for": "\u0915\u0947 \u0932\u093f\u090f", "testing": "\u092a\u0930\u0940\u0915\u094d\u0937\u0923", "we": "\u0939\u092e", "will": "\u0915\u0930\u0947\u0902\u0917\u0947",
    "translate": "\u0905\u0928\u0941\u0935\u093e\u0926", "dub": "\u0921\u092c", "speech": "\u092d\u093e\u0937\u0923",
}


def _simple_map(text: str, target: str) -> str:
    if target not in {"hi", "hin"}:
        return f"[{target}] {text}"
    out = []
    for tok in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        key = tok.lower()
        out.append(_DEMO_HI.get(key, tok))
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


def _lang_name(code: str) -> str:
    return _LANG_NAMES.get((code or "").lower().split("-")[0], code or "the target language")


def _clean_llm_output(text: str) -> str:
    # Strip reasoning blocks some models emit, then surrounding quotes/labels.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    # Drop a leading "Translation:" style prefix if present.
    text = re.sub(r"^\s*(translation|translated|output)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip("'").strip()


def _build_prompt(text: str, *, source: str, target: str, context: list[str], target_duration_ms: int) -> tuple[str, str]:
    src_name = _lang_name(source)
    tgt_name = _lang_name(target)
    target_sec = max(0.5, target_duration_ms / 1000.0)
    # Native conversational tempo: ~2.5 - 3.0 words per second
    max_words = max(3, int(target_sec * 2.8))

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
        "4. Output ONLY the localized line — no quotes, no explanations, no notes."
    )
    ctx = "\n".join(context[-3:])
    user = (
        (f"Context from earlier dialogue:\n{ctx}\n\n" if ctx else "")
        + f"Target duration: {target_sec:.1f}s (max {max_words} words)\n"
        f"Adapt this line into concise conversational {tgt_name}:\n{text}"
    )
    return system, user


def _ollama_translate(text: str, *, source: str, target: str, context: list[str], target_duration_ms: int) -> str | None:
    host = settings.ollama_host.rstrip("/")
    system, user = _build_prompt(text, source=source, target=target, context=context, target_duration_ms=target_duration_ms)
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
            return None
        content = _clean_llm_output((r.json().get("message") or {}).get("content") or "")
        return content or None
    except Exception:
        return None


import logging

log = logging.getLogger(__name__)


def _openai_translate(text: str, *, source: str, target: str, context: list[str], target_duration_ms: int) -> str | None:
    """Any OpenAI-compatible chat API (OpenAI, Groq, DeepSeek, OpenRouter, Ollama /v1, ...)."""
    base = settings.openai_base_url.rstrip("/")
    system, user = _build_prompt(text, source=source, target=target, context=context, target_duration_ms=target_duration_ms)
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
        content = _clean_llm_output((choices[0].get("message") or {}).get("content") or "")
        return content or None
    except Exception as e:
        log.warning("OpenAI translation call failed: %s", e)
        return None


def translate_segments(
    segments: list[dict[str, Any]],
    *,
    source_language: str,
    target_language: str,
    job: dict | None = None,
) -> list[dict[str, Any]]:
    context: list[str] = []
    total = len(segments)
    for i, seg in enumerate(segments):
        if job and job.get("job_id") and total > 0:
            try:
                from dubstudio.jobs.store import store

                pct = 58 + int((i / total) * 7)
                job["percent"] = min(pct, 64)
                job["message"] = f"Translating segment {i + 1}/{total}"
                store.save(job)
            except Exception:
                pass
        text = seg.get("source_text") or ""
        translated = None
        dur = int(seg.get("target_duration_ms") or 2000)
        if settings.translator == "ollama":
            translated = _ollama_translate(text, source=source_language or "en", target=target_language, context=context, target_duration_ms=dur)
        elif settings.translator == "openai":
            translated = _openai_translate(text, source=source_language or "en", target=target_language, context=context, target_duration_ms=dur)
        if not translated:
            log.warning("LLM translation failed for segment %s; using simple fallback map", seg.get("segment_id"))
            translated = _simple_map(text, target_language)
        try:
            from dubstudio.util.phonetics import normalize_hinglish

            translated = normalize_hinglish(translated, target_language)
        except Exception:
            pass
        seg["translated_text"] = translated
        seg["status"] = "translated"
        context.append(f"{text} => {translated}")
    return segments


def run_translation(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get("segments", [])
    segments = translate_segments(
        segments,
        source_language=job.get("source_language") or "en",
        target_language=job.get("target_language") or "hi",
        job=job,
    )
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
