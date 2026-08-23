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


def _ollama_translate(text: str, *, source: str, target: str, context: list[str], target_duration_ms: int) -> str | None:
    host = settings.ollama_host.rstrip("/")
    system = (
        "You are a professional dubbing translator. Return ONLY the translated line. "
        f"Speakable in about {target_duration_ms} ms. Keep meaning and names."
    )
    user = f"source_lang={source} target_lang={target}\ncontext={json.dumps(context[-3:], ensure_ascii=False)}\ntext={text}"
    try:
        r = httpx.post(
            f"{host}/api/chat",
            json={"model": settings.ollama_model, "stream": False, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=60.0,
        )
        if r.status_code != 200:
            return None
        content = ((r.json().get("message") or {}).get("content") or "").strip().strip('"')
        return content or None
    except Exception:
        return None


def translate_segments(segments: list[dict[str, Any]], *, source_language: str, target_language: str) -> list[dict[str, Any]]:
    context: list[str] = []
    for seg in segments:
        text = seg.get("source_text") or ""
        translated = None
        if settings.translator == "ollama":
            translated = _ollama_translate(text, source=source_language or "en", target=target_language, context=context, target_duration_ms=int(seg.get("target_duration_ms") or 2000))
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
