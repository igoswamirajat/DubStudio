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
    # Spoken budget: aim to FILL the original slot so the dub doesn't leave dead
    # air. Target speakers pack ~13-15 chars/sec; use that as a fill target.
    approx_chars = max(8, int(target_duration_ms / 1000 * 15))
    system = (
        f"You are an expert dubbing localizer adapting {src_name} speech into natural, "
        f"conversational {tgt_name} for a voice-over. You are NOT a literal translator.\n"
        "Rules:\n"
        f"1. Convey the MEANING and intent the way a native {tgt_name} speaker would actually say it. "
        "Never translate word-for-word. Rewrite idioms into their natural equivalent "
        "(e.g. \"out of the box\" means ready-to-use, NOT a literal box).\n"
        "2. Keep technical terms, product names, brand names, and common tech words in English "
        "(e.g. Cloud Code, plugin, agent, GitHub, open source, AI, model, API). Do not force-translate them.\n"
        "3. Match the tone/register of the original (casual, hype, formal) and keep it fluent and speakable.\n"
        f"4. IMPORTANT — length: the spoken line must take about the SAME time as the original "
        f"(roughly {approx_chars} characters). Do NOT make it shorter or clipped; a dub that is too "
        "short leaves silent gaps. Naturally expand phrasing to fill the time without padding or repetition.\n"
        f"5. Output ONLY the final {tgt_name} line — no quotes, no notes, no alternatives, no explanation."
    )
    ctx = "\n".join(context[-3:])
    user = (f"Earlier lines already dubbed (context, do not translate again):\n{ctx}\n\n" if ctx else "") + f"Now adapt this line:\n{text}"
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
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=300.0,
        )
        if r.status_code != 200:
            return None
        choices = r.json().get("choices") or []
        if not choices:
            return None
        content = _clean_llm_output((choices[0].get("message") or {}).get("content") or "")
        return content or None
    except Exception:
        return None


def translate_segments(segments: list[dict[str, Any]], *, source_language: str, target_language: str) -> list[dict[str, Any]]:
    context: list[str] = []
    for seg in segments:
        text = seg.get("source_text") or ""
        translated = None
        dur = int(seg.get("target_duration_ms") or 2000)
        if settings.translator == "ollama":
            translated = _ollama_translate(text, source=source_language or "en", target=target_language, context=context, target_duration_ms=dur)
        elif settings.translator == "openai":
            translated = _openai_translate(text, source=source_language or "en", target=target_language, context=context, target_duration_ms=dur)
        if not translated:
            translated = _simple_map(text, target_language)
        seg["translated_text"] = translated
        seg["status"] = "translated"
        context.append(f"{text} => {translated}")
    return segments


def run_translation(job_dir: Path, job: dict) -> list[dict]:
    path = job_dir / "segments" / "segments.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get("segments", [])
    segments = translate_segments(segments, source_language=job.get("source_language") or "en", target_language=job.get("target_language") or "hi")
    path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    return segments
