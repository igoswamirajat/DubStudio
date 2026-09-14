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

# --------------------------------------------------------------------------
# Length budget
#
# The old prompt only handed the model a ceiling ("at most N words"), so a four
# second slot regularly came back with five words in it. Nothing downstream can
# repair that: timing.py stretches the short line to fill the slot, which is
# exactly the slow, dragging delivery with dead air between lines. A dub line
# needs a BAND - long enough to fill the slot, short enough not to overrun it.
# --------------------------------------------------------------------------
_WPS = {
    "hi": 2.6, "mr": 2.5, "bn": 2.6, "gu": 2.6, "pa": 2.7, "ur": 2.6,
    "ta": 2.3, "te": 2.4, "kn": 2.4, "ml": 2.3, "or": 2.4,
    "en": 2.9, "es": 2.9, "pt": 2.9, "it": 2.8, "fr": 2.8, "de": 2.4,
    "nl": 2.6, "pl": 2.4, "ru": 2.3, "tr": 2.3, "ar": 2.4,
    "id": 2.7, "vi": 2.9, "th": 2.7, "ja": 2.6, "ko": 2.5, "zh": 2.6,
}
_WPS_DEFAULT = 2.7
_FILL_MIN = 0.75          # under this share of the slot the line drags
_FILL_MAX = 1.15          # over it the actor has to rush
_SOURCE_EXPANSION = 1.6   # a dub is never three times longer than what was said
_UNDERFILL_SLACK = 0.65   # only spend a retry when the line is badly short
_OVERFILL_SLACK = 1.25    # ... or badly long

# Markers smuggled through the existing `context` list so the translator can see
# what comes next without changing the translator function signature.
NEXT_MARK = "[NEXT] "
FLOW_MARK = "[FLOW] "
FLOW_CONTINUES = "continues"
FLOW_NEW = "new-sentence"
# A source line that ends without terminal punctuation is a sentence in flight.
_SENT_END = ".?!\u0964\u2026"
_TRAILING = "\"')]\u201d\u2019"


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

# Per-language nudges that actually change how a dub sounds.
_STYLE_NOTES = {
    "hi": ("Use natural spoken Hindi, the way a YouTuber talks - not news-anchor "
           "Hindi and not Sanskritised textbook Hindi. Address the viewer as आप and "
           "never switch to तुम halfway through the video."),
    "mr": "Use natural spoken Marathi and keep one consistent address form.",
    "bn": "Use natural spoken Bengali and keep one consistent address form.",
    "es": "Use neutral Latin-American Spanish and keep one consistent address form.",
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
    """Count spoken words, script-agnostically.

    The previous rule was ``re.findall(r"\\w{2,}", ...)``. Devanagari vowel
    signs are combining marks and do not match ``\\w``, so that rule scored
    मैं as zero words and a full twelve word Hindi line as five. Good enough
    for a crude "is this line tiny?" guard, useless as a length budget - it
    would report every Hindi line as far too short and burn a retry on each.
    """
    return sum(1 for tok in (text or "").split() if any(ch.isalnum() for ch in tok))


def _speech_rate(target: str) -> float:
    return _WPS.get(_base_lang(target), _WPS_DEFAULT)


def _word_budget(source: str, target_sec: float, target: str) -> tuple[int, int]:
    """Return the (min, max) word count that fills `target_sec` naturally.

    A band, not a ceiling. The lower bound is the half of this that the old
    prompt was missing entirely.
    """
    ideal = max(0.5, float(target_sec)) * _speech_rate(target)
    lo = max(2, round(ideal * _FILL_MIN))
    hi = max(lo + 1, round(ideal * _FILL_MAX))
    src_words = _word_count(source)
    if src_words:
        # A slot can be long because the speaker paused, not because they said
        # a lot. Never ask the model to pad far past the original content - that
        # is how you get invented filler in a dub.
        cap = max(3, round(src_words * _SOURCE_EXPANSION))
        hi = min(hi, cap)
        lo = min(lo, max(2, cap - 1))
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _budget_distance(source: str, output: str, target: str,
                     target_duration_ms: int | None) -> int:
    if not target_duration_ms:
        return 0
    lo, hi = _word_budget(source, target_duration_ms / 1000.0, target)
    n = _word_count(output)
    if n < lo:
        return lo - n
    if n > hi:
        return n - hi
    return 0


def _length_issue(source: str, output: str, target: str,
                  target_duration_ms: int | None) -> str | None:
    """Describe a badly mis-sized line. Never a correctness failure.

    A wrong-length line is still a real translation, so it is kept rather than
    thrown away; it only earns one more attempt at a better-sized rewrite.
    """
    if not target_duration_ms or _word_count(source) < 3:
        return None
    target_sec = max(0.5, target_duration_ms / 1000.0)
    lo, hi = _word_budget(source, target_sec, target)
    n = _word_count(output)
    if n < max(1, round(lo * _UNDERFILL_SLACK)):
        return (f"only {n} words for a {target_sec:.1f}s slot (needs {lo}-{hi}); "
                f"the line will be stretched and drag")
    if n > round(hi * _OVERFILL_SLACK):
        return (f"{n} words will not fit a {target_sec:.1f}s slot (needs {lo}-{hi}); "
                f"the line will be rushed or clipped")
    return None


def check_translation(source: str, output: str, target: str) -> str | None:
    """Return a human-readable reason when `output` is not a real translation.

    Returns None when the line looks genuinely translated. Length is decided by
    `_length_issue`, not here: a correct translation of the wrong length is
    still a translation.
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


_QUOTE_CHARS = "\"'\u00ab\u00bb\u201c\u201d\u2018\u2019\u300c\u300d"
_LABEL_RE = re.compile(
    r"^\s*(?:option\s*\d*|translation|translated(?:\s+line)?|output|answer|dub|line|"
    r"hindi|marathi|bengali|spanish|french|german)\s*\d*\s*[:\-\u2013\u2014]\s*",
    flags=re.IGNORECASE,
)
_NOTE_RE = re.compile(
    r"\s*[\(\[](?:note|literal|lit\.|meaning|translation|english)[^\)\]]*[\)\]]\s*$",
    flags=re.IGNORECASE,
)


def _clean_llm_output(text: str) -> str:
    """Reduce a chatty model reply to the single spoken line.

    Anything left in here is read out loud by the TTS, so a stray "Translation:"
    or "(literal: ...)" becomes audible garbage in the dub.
    """
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```[a-zA-Z]*", " ", text).replace("```", " ")
    text = text.replace("**", "").replace("`", "").strip()
    # Models like to offer a list or add a footnote; the first real line wins.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[0]
    text = _LABEL_RE.sub("", text)
    text = re.sub(r"^\s*[-*\u2022]\s+", "", text)
    text = _NOTE_RE.sub("", text)
    text = text.strip().strip(_QUOTE_CHARS).strip()
    return re.sub(r"[ \t]+", " ", text).strip()


def _split_context(context: list[str] | None) -> tuple[list[str], str | None, str | None]:
    """Pull the [NEXT]/[FLOW] hints back out of the plain context list.

    They ride inside `context` on purpose: every translator callable (and every
    test fake) has the same fixed signature, so continuity cannot be added as a
    new keyword argument without breaking them.
    """
    pairs: list[str] = []
    nxt: str | None = None
    flow: str | None = None
    for entry in context or []:
        item = (entry or "").strip()
        if not item:
            continue
        if item.startswith(NEXT_MARK):
            nxt = item[len(NEXT_MARK):].strip() or None
        elif item.startswith(FLOW_MARK):
            flow = item[len(FLOW_MARK):].strip() or None
        else:
            pairs.append(item)
    return pairs[-3:], nxt, flow


def _continues(text: str) -> bool:
    """True when a source line stops mid-thought (no terminal punctuation)."""
    stripped = (text or "").rstrip().rstrip(_TRAILING).rstrip()
    return bool(stripped) and stripped[-1] not in _SENT_END


def _flow_hint(prev_seg: dict | None, gap_ms: float | None = None) -> str:
    if prev_seg is None:
        return FLOW_NEW
    if _continues(prev_seg.get("source_text") or ""):
        return FLOW_CONTINUES
    gap = prev_seg.get("gap_after_ms") if gap_ms is None else gap_ms
    try:
        if gap is not None and float(gap) < 300.0:
            return FLOW_CONTINUES
    except (TypeError, ValueError):
        pass
    return FLOW_NEW


def _build_prompt(text: str, *, source: str, target: str, context: list[str],
                  target_duration_ms: int, strict: bool = False) -> tuple[str, str]:
    src_name = _lang_name(source)
    tgt_name = _lang_name(target)
    target_sec = max(0.5, target_duration_ms / 1000.0)
    lo, hi = _word_budget(text, target_sec, target)
    script_name = (_TARGET_SCRIPTS.get(_base_lang(target)) or (None, None))[0]
    style = _STYLE_NOTES.get(
        _base_lang(target),
        f"Use natural spoken {tgt_name}, the way a person actually talks on camera - "
        f"not written or formal {tgt_name}. Keep one consistent level of politeness "
        f"across the whole video.",
    )
    if script_name:
        borrowed = (f"but write them in the {script_name} script so the voice "
                    f"pronounces them correctly")
    else:
        borrowed = f"spelled the way {tgt_name} normally spells them"
    pairs, next_line, flow = _split_context(context)

    system = (
        f"You write dubbing lines. You adapt {src_name} speech into spoken {tgt_name} "
        f"for a voice actor who has to land each line inside a fixed slot in the video. "
        f"You are not a literal translator and not a subtitle writer.\n\n"
        "RULES\n"
        f"1. LENGTH. Write {lo} to {hi} words. The actor has about {target_sec:.1f}s. "
        f"Fewer than {lo} words leaves dead air and the line gets stretched into a slow, "
        f"dragging delivery; more than {hi} words gets rushed or clipped. Aim for the middle "
        f"of that range - after meaning, this is the most important rule.\n"
        "2. FLOW. This is one line inside a continuous monologue, not a standalone sentence. "
        "When the thought carries on, end on a comma or on no punctuation at all so the voice "
        "stays up instead of falling into a full stop. Use a full stop only when the idea "
        "really ends.\n"
        "3. NO RESTARTS, NO REPEATS. Do not open every line with the same connector. Do not "
        "say again what the previous lines already said - use pronouns and short references "
        "the way a real speaker does.\n"
        f"4. SPOKEN REGISTER. {style}\n"
        f"5. BORROWED WORDS. Everyday English tech words are part of normal speech, so keep "
        f"them, {borrowed}. Leave short acronyms (API, HTML, AI, URL) in Latin capitals.\n"
        "6. NUMBERS AND SYMBOLS. Write digits, units and symbols the way they are said out "
        "loud, never as figures.\n"
        f"7. OUTPUT. Exactly one line of {tgt_name}. No quotes, no label, no romanisation, "
        "no alternatives, no notes, no source text.\n\n"
        "Never reply like any of these:\n"
        "  Translation: ...\n"
        "  Option 1: ...  Option 2: ...\n"
        "  ... (literal: ...)\n"
        "  the source sentence handed back unchanged"
    )
    if strict:
        system += (
            f"\n\nCRITICAL: your previous reply was rejected. Reply ONLY in {tgt_name}"
            + (f", written in the {script_name} script" if script_name else "")
            + f". Never hand the {src_name} sentence back. Stay inside {lo} to {hi} words. "
            f"Output one single {tgt_name} line and nothing else."
        )

    parts: list[str] = []
    if pairs:
        parts.append("Lines already dubbed (source => dub):\n" + "\n".join(pairs))
    if flow == FLOW_CONTINUES:
        parts.append("The previous line stopped mid-sentence. Carry that thought straight on - "
                     "do not open this line like a brand new sentence.")
    elif flow == FLOW_NEW and pairs:
        parts.append("The previous sentence finished. Start a fresh thought, same speaker, "
                     "same register.")
    if next_line:
        parts.append("The line that comes right after (context only - do NOT translate it and "
                     "do NOT include it):\n" + next_line)
    parts.append(f"Slot: {target_sec:.1f}s -> write {lo} to {hi} words.")
    parts.append(f"Adapt this {src_name} line into spoken {tgt_name}:\n{text}")
    return system, "\n\n".join(parts)


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
        log.warning("translator=demo - output will be a word-for-word demo map, not a translation")
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
    mis_sized = 0
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

        # Continuity: the model sees what was just said, whether the previous
        # line finished its sentence, and what is coming next. Without this it
        # writes every line as its own closed sentence, which is what makes the
        # dub sound like a list of disconnected statements.
        next_text = ""
        if i + 1 < total:
            next_text = (segments[i + 1].get("source_text") or "").strip()
        call_ctx = list(context[-3:])
        if next_text:
            call_ctx.append(f"{NEXT_MARK}{next_text}")
        call_ctx.append(f"{FLOW_MARK}{_flow_hint(segments[i - 1] if i else None)}")
        slot_ms = int(seg.get("target_duration_ms") or 2000)

        translated, reason = None, "no attempt made"
        best: str | None = None
        best_issue: str | None = None
        for attempt in range(1, attempts + 1):
            candidate = fn(text, source=source_language or "en", target=target_language,
                           context=call_ctx, target_duration_ms=slot_ms,
                           strict=attempt > 1)
            if not candidate:
                reason = "translator returned nothing"
                log.warning("Segment %s attempt %d/%d: %s", seg_id, attempt, attempts, reason)
                continue
            reason = check_translation(text, candidate, target_language) or ""
            if reason:
                log.warning("Segment %s attempt %d/%d rejected: %s",
                            seg_id, attempt, attempts, reason)
                continue
            issue = _length_issue(text, candidate, target_language, slot_ms)
            if not issue:
                translated = candidate
                break
            # A mis-sized line is still a real translation, so it is never
            # thrown away - it just earns one more try at a better fit.
            if best is None or _budget_distance(text, candidate, target_language, slot_ms) < \
                    _budget_distance(text, best, target_language, slot_ms):
                best, best_issue = candidate, issue
            log.info("Segment %s attempt %d/%d: %s", seg_id, attempt, attempts, issue)

        note: str | None = None
        if translated is None and best is not None:
            translated = best
            note = best_issue
            mis_sized += 1
            log.warning("Segment %s kept a mis-sized line: %s", seg_id, best_issue)

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
        seg["translation_words"] = _word_count(translated)
        if note:
            seg["translation_note"] = note
        else:
            seg.pop("translation_note", None)
        seg.pop("error", None)
        context.append(f"{text} => {translated}")

    if mis_sized:
        log.warning("%d/%d line(s) could not be written to fit their slot; "
                    "expect stretching on those lines", mis_sized, total)

    if failures:
        head = "; ".join(f"{sid}: {why}" for sid, why in failures[:5])
        msg = (f"{len(failures)}/{total} segment(s) were not translated via {desc}. "
               f"First failures - {head}")
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
