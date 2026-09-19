"""Regression tests for the "dub says the same words as the original" bug.

The pipeline used to fall back to a 19-word demo dictionary whenever the LLM
was unreachable or unhappy, and still mark the segment as "translated". The
result was a dub in the source language with a handful of words swapped.
"""
from __future__ import annotations

import json

import pytest

from dubstudio.pipeline import translation as T
from dubstudio.settings import settings

EN = "So today I am going to show you how to scrape any website for free."
HI = "आज मैं आपको दिखाता हूँ कि कोई भी वेबसाइट मुफ़्त में कैसे स्क्रैप करें।"
# What the old fallback actually produced for EN: 3 of 16 tokens replaced.
WORD_MAPPED = "So today I am going में show you how में scrape any website के लिए free ."
PROBE_PREFIX = "Today I will show you"  # preflight probe sentence


@pytest.fixture(autouse=True)
def _restore():
    keys = ("translator", "strict_translation", "translation_attempts")
    saved = {k: getattr(settings, k) for k in keys}
    saved_fn = T._ollama_translate
    yield
    for k, v in saved.items():
        setattr(settings, k, v)
    T._ollama_translate = saved_fn


def _seg(i: int = 0, text: str = EN, ms: int = 4000) -> dict:
    return {"segment_id": f"seg_{i:04d}", "source_text": text,
            "target_duration_ms": ms, "translated_text": "", "status": "pending"}


def _fake(reply):
    """Build a stand-in for _ollama_translate. `reply` is a value or callable."""

    def fn(text, *, source, target, context, target_duration_ms, strict=False):
        return reply(text) if callable(reply) else reply

    return fn


# --------------------------------------------------------------------------
# check_translation: the detector
# --------------------------------------------------------------------------
def test_source_echo_is_rejected():
    assert T.check_translation(EN, EN, "hi") is not None


def test_real_hindi_is_accepted():
    assert T.check_translation(EN, HI, "hi") is None


def test_word_mapped_output_is_rejected():
    reason = T.check_translation(EN, WORD_MAPPED, "hi")
    assert reason is not None
    assert "Devanagari" in reason


def test_empty_output_is_rejected():
    assert T.check_translation(EN, "", "hi") is not None


def test_short_line_is_not_a_false_positive():
    # Brand names and interjections can legitimately survive unchanged.
    assert T.check_translation("GitHub.", "GitHub.", "hi") is None


def test_latin_script_target_uses_similarity():
    src = "This tool is completely free"
    assert T.check_translation(src, src, "es") is not None
    assert T.check_translation(src, "Esta herramienta es totalmente gratuita", "es") is None


def test_wrong_script_for_target_is_rejected():
    assert T.check_translation(EN, HI, "ta") is not None


# --------------------------------------------------------------------------
# translate_segments: no silent fallback
# --------------------------------------------------------------------------
def test_dead_translator_raises_instead_of_word_mapping():
    settings.translator = "ollama"
    T._ollama_translate = _fake(None)
    segs = [_seg()]
    with pytest.raises(T.TranslationError) as exc:
        T.translate_segments(segs, source_language="en", target_language="hi")
    assert "unreachable" in str(exc.value).lower()
    assert not segs[0]["translated_text"]


def test_echoing_model_fails_preflight():
    settings.translator = "ollama"
    T._ollama_translate = _fake(lambda text: text)
    with pytest.raises(T.TranslationError) as exc:
        T.translate_segments([_seg()], source_language="en", target_language="hi")
    assert "not translating" in str(exc.value).lower()


def test_unknown_translator_raises():
    settings.translator = "nope"
    with pytest.raises(T.TranslationError):
        T.translate_segments([_seg()], source_language="en", target_language="hi")


def test_happy_path_records_engine():
    settings.translator = "ollama"
    T._ollama_translate = _fake(HI)
    out = T.translate_segments([_seg()], source_language="en", target_language="hi")
    assert out[0]["status"] == "translated"
    assert out[0]["translation_engine"].startswith("ollama:")


def test_bad_first_attempt_is_retried_strictly():
    settings.translator = "ollama"
    settings.translation_attempts = 2
    calls = {"strict": 0}

    def fn(text, *, source, target, context, target_duration_ms, strict=False):
        if text.startswith(PROBE_PREFIX):
            return HI
        if strict:
            calls["strict"] += 1
            return HI
        return text

    T._ollama_translate = fn
    out = T.translate_segments([_seg()], source_language="en", target_language="hi")
    assert out[0]["status"] == "translated"
    assert calls["strict"] == 1


def test_persistent_echo_fails_without_leaking_english():
    settings.translator = "ollama"
    settings.strict_translation = True
    T._ollama_translate = _fake(lambda t: HI if t.startswith(PROBE_PREFIX) else t)
    segs = [_seg()]
    with pytest.raises(T.TranslationError):
        T.translate_segments(segs, source_language="en", target_language="hi")
    assert segs[0]["status"] == "translation_failed"
    assert segs[0]["translated_text"] == ""


def test_demo_mode_only_runs_when_selected():
    settings.translator = "demo"
    out = T.translate_segments([_seg(text="Hello and welcome to DubStudio.")],
                               source_language="en", target_language="hi")
    assert out[0]["translation_engine"] == "demo"


# --------------------------------------------------------------------------
# filler: padding a line out to reach the word count
# --------------------------------------------------------------------------
def test_a_line_that_repeats_itself_is_flagged():
    out = "यह एक बात है यह एक बात है और बस"
    reason = T._filler_issue(EN, out, "hi", 4000)
    assert reason is not None
    assert "repeat" in reason


def test_an_empty_connector_on_an_over_budget_line_is_flagged():
    src = "one two three four five six"
    out = "जब आप request भेजते हैं तो यह काम करता है और बहुत अच्छा है दोस्तों"
    reason = T._filler_issue(src, out, "hi", 2000)
    assert reason is not None
    assert "filler" in reason


def test_a_clean_line_is_not_called_filler():
    # Long, but every word is doing work - over budget is not the same as padded.
    assert T._filler_issue(EN, HI, "hi", 4000) is None


def test_a_padded_line_earns_one_rewrite_and_is_kept():
    calls = {"n": 0}

    def fn(text, *, source, target, context, target_duration_ms, strict=False):
        if text.startswith(PROBE_PREFIX):
            return HI
        calls["n"] += 1
        return "यह एक बात है यह एक बात है और बस"

    settings.translator = "ollama"
    settings.translation_attempts = 2
    T._ollama_translate = fn
    out = T.translate_segments([_seg()], source_language="en", target_language="hi")

    assert calls["n"] == 2, "the padded line should be sent back once, strictly"
    assert out[0]["status"] == "translated"
    assert "repeat" in out[0]["translation_note"]


# --------------------------------------------------------------------------
# resume guard
# --------------------------------------------------------------------------
def test_verify_segments_translated_flags_cached_bad_output():
    cached = [
        {"segment_id": "seg_0000", "source_text": EN, "translated_text": WORD_MAPPED},
        {"segment_id": "seg_0001", "source_text": "This tool is open source.", "translated_text": HI},
    ]
    assert T.verify_segments_translated(cached, "hi") == ["seg_0000"]


def test_run_translation_persists_failed_segments(tmp_path):
    settings.translator = "ollama"
    settings.strict_translation = True
    T._ollama_translate = _fake(lambda t: HI if t.startswith(PROBE_PREFIX) else t)
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True)
    (seg_dir / "segments.json").write_text(json.dumps([_seg()]), encoding="utf-8")

    with pytest.raises(T.TranslationError):
        T.run_translation(tmp_path, {"source_language": "en", "target_language": "hi"})

    saved = json.loads((seg_dir / "segments.json").read_text(encoding="utf-8"))
    assert saved[0]["status"] == "translation_failed"
