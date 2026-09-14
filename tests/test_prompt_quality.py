"""The dub reads badly because of what we ASK the model for, not just the model.

Two concrete defects are pinned here:

1. The old prompt only ever gave a ceiling ("at most N words"). A four second
   slot regularly came back with five words, timing.py stretched them across
   the slot, and that is the slow, dragging delivery with dead air in it.
   A dub line needs a band with a floor.
2. Every line was translated in isolation, so the model closed every line with
   a full stop and re-introduced context the previous line had already given.
   Spoken monologue does not work like that.
"""
from __future__ import annotations

import pytest

from dubstudio.pipeline import translation as T
from dubstudio.settings import settings

EN = "So today I am going to show you how to scrape any website for free."
PROBE_PREFIX = "Today I will show you"  # preflight probe sentence
# 12 words - a real, slot-filling Hindi line.
HI_FIT = "आज मैं आपको दिखाता हूँ कि कोई भी वेबसाइट कैसे स्क्रैप करें"
# 2 words - correct Hindi, but nowhere near enough to fill a four second slot.
HI_SHORT = "ठीक है"


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


# --------------------------------------------------------------------------
# the word budget
# --------------------------------------------------------------------------
def test_word_count_is_script_agnostic():
    # The old `\w{2,}` rule scored this 12 word Hindi line at 5, because
    # Devanagari vowel signs are combining marks. A budget built on that
    # number calls every Hindi line far too short.
    assert T._word_count(HI_FIT) == 12
    assert T._word_count(HI_SHORT) == 2
    assert T._word_count(EN) == 15
    assert T._word_count("") == 0


def test_budget_is_a_band_not_just_a_ceiling():
    lo, hi = T._word_budget(EN, 4.0, "hi")
    assert 2 <= lo < hi
    # Four seconds of Hindi is roughly ten words; a floor of three would let
    # the model hand back a line that has to be stretched to twice its length.
    assert lo >= 7


def test_budget_scales_with_the_slot():
    lo_short, hi_short = T._word_budget(EN, 2.0, "hi")
    lo_long, hi_long = T._word_budget(EN, 6.0, "hi")
    assert lo_short < lo_long
    assert hi_short < hi_long


def test_budget_is_capped_by_what_was_actually_said():
    # A long slot can mean the speaker paused, not that they said a lot.
    lo, hi = T._word_budget("Okay, let's go.", 10.0, "hi")
    assert hi <= 6, "do not ask the model to invent filler"
    assert lo < hi


def test_a_short_line_is_flagged_and_a_fitting_line_is_not():
    assert T._length_issue(EN, HI_SHORT, "hi", 4000) is not None
    assert T._length_issue(EN, HI_FIT, "hi", 4000) is None


def test_length_is_not_judged_without_a_slot_or_on_tiny_lines():
    assert T._length_issue(EN, HI_SHORT, "hi", None) is None
    assert T._length_issue("GitHub.", HI_SHORT, "hi", 4000) is None


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------
def test_prompt_states_a_floor_and_a_ceiling():
    lo, hi = T._word_budget(EN, 4.0, "hi")
    system, user = T._build_prompt(EN, source="en", target="hi", context=[],
                                   target_duration_ms=4000)
    assert f"{lo} to {hi} words" in system
    assert f"{lo} to {hi} words" in user


def test_prompt_asks_for_one_bare_line():
    system, _ = T._build_prompt(EN, source="en", target="hi", context=[],
                                target_duration_ms=4000)
    assert "Exactly one line" in system
    for banned in ("No quotes", "no notes", "no source text"):
        assert banned in system


def test_prompt_covers_flow_and_repetition():
    system, _ = T._build_prompt(EN, source="en", target="hi", context=[],
                                target_duration_ms=4000)
    assert "full stop" in system, "every line ending in a full stop is the flat-reading bug"
    assert "NO RESTARTS" in system


def test_prompt_names_the_target_script_for_borrowed_words():
    system, _ = T._build_prompt(EN, source="en", target="hi", context=[],
                                target_duration_ms=4000)
    assert "Devanagari script" in system
    # Latin-script targets must not get a script instruction they cannot follow.
    system_es, _ = T._build_prompt(EN, source="en", target="es", context=[],
                                   target_duration_ms=4000)
    assert "Devanagari" not in system_es
    assert "Spanish" in system_es


def test_prompt_carries_the_previous_lines():
    ctx = ["Hello there => नमस्ते", "This is a test => यह एक टेस्ट है"]
    _, user = T._build_prompt(EN, source="en", target="hi", context=ctx,
                              target_duration_ms=4000)
    for entry in ctx:
        assert entry in user


def test_continuing_sentence_tells_the_model_not_to_restart():
    ctx = ["Hello there => नमस्ते", T.FLOW_MARK + T.FLOW_CONTINUES]
    _, user = T._build_prompt(EN, source="en", target="hi", context=ctx,
                              target_duration_ms=4000)
    assert "mid-sentence" in user
    assert "brand new sentence" in user


def test_finished_sentence_does_not_get_the_continuation_note():
    ctx = ["Hello there => नमस्ते", T.FLOW_MARK + T.FLOW_NEW]
    _, user = T._build_prompt(EN, source="en", target="hi", context=ctx,
                              target_duration_ms=4000)
    assert "mid-sentence" not in user
    assert "fresh thought" in user


def test_next_line_is_given_as_context_only():
    nxt = "And it works on absolutely any website."
    _, user = T._build_prompt(EN, source="en", target="hi",
                              context=[T.NEXT_MARK + nxt], target_duration_ms=4000)
    assert nxt in user
    assert "do NOT translate it" in user


def test_internal_markers_never_reach_the_model():
    ctx = ["Hello there => नमस्ते",
           T.NEXT_MARK + "And that is it.",
           T.FLOW_MARK + T.FLOW_CONTINUES]
    system, user = T._build_prompt(EN, source="en", target="hi", context=ctx,
                                   target_duration_ms=4000)
    for marker in ("[NEXT]", "[FLOW]"):
        assert marker not in system
        assert marker not in user


def test_strict_mode_repeats_the_length_and_script_rules():
    lo, hi = T._word_budget(EN, 4.0, "hi")
    system, _ = T._build_prompt(EN, source="en", target="hi", context=[],
                                target_duration_ms=4000, strict=True)
    assert "CRITICAL" in system
    assert "Devanagari script" in system
    assert f"{lo} to {hi} words" in system


# --------------------------------------------------------------------------
# reply cleanup - anything left in here is spoken out loud by the TTS
# --------------------------------------------------------------------------
def test_chatty_replies_are_reduced_to_the_spoken_line():
    want = "नमस्ते दुनिया"
    cases = [
        "Translation: नमस्ते दुनिया",
        '"नमस्ते दुनिया"',
        "“नमस्ते दुनिया”",
        "**नमस्ते दुनिया**",
        "- नमस्ते दुनिया",
        "नमस्ते दुनिया (literal: hello world)",
        "<think>the user wants Hindi</think>\nनमस्ते दुनिया",
        "Option 1: नमस्ते दुनिया\nOption 2: हैलो दुनिया",
        "```\nनमस्ते दुनिया\n```",
    ]
    for raw in cases:
        assert T._clean_llm_output(raw) == want, raw


def test_a_clean_line_survives_untouched():
    assert T._clean_llm_output(HI_FIT) == HI_FIT


# --------------------------------------------------------------------------
# continuity wiring through translate_segments
# --------------------------------------------------------------------------
def test_flow_hint_detects_an_unfinished_sentence():
    assert T._flow_hint(None) == T.FLOW_NEW
    assert T._flow_hint({"source_text": "and the best part is"}) == T.FLOW_CONTINUES
    assert T._flow_hint({"source_text": "That is all."}) == T.FLOW_NEW
    # A sentence can end on paper and still run straight on in speech.
    assert T._flow_hint({"source_text": "That is all.",
                         "gap_after_ms": 120}) == T.FLOW_CONTINUES


def test_each_line_is_translated_with_its_neighbours():
    src = [
        EN,
        "And the best part is that this method",
        "works on almost every single page out there.",
    ]
    seen: dict[str, list[str]] = {}

    def fn(text, *, source, target, context, target_duration_ms, strict=False):
        seen[text] = list(context)
        return HI_FIT

    settings.translator = "ollama"
    T._ollama_translate = fn
    segs = [_seg(i, t, 4000 if i == 0 else 4500) for i, t in enumerate(src)]
    T.translate_segments(segs, source_language="en", target_language="hi")

    first, middle, last = (seen[t] for t in src)
    # the line after is handed over as context, and the last line has none
    assert T.NEXT_MARK + src[1] in first
    assert not [c for c in last if c.startswith(T.NEXT_MARK)]
    # what was already dubbed travels forward
    assert f"{src[0]} => {HI_FIT}" in middle
    # src[1] stops mid-thought, so src[2] must be told to carry it on
    assert T.FLOW_MARK + T.FLOW_CONTINUES in last
    assert T.FLOW_MARK + T.FLOW_NEW in first


def test_a_short_line_earns_one_strict_rewrite():
    calls = {"strict": 0}

    def fn(text, *, source, target, context, target_duration_ms, strict=False):
        if text.startswith(PROBE_PREFIX):
            return HI_FIT
        if strict:
            calls["strict"] += 1
            return HI_FIT
        return HI_SHORT

    settings.translator = "ollama"
    settings.translation_attempts = 2
    T._ollama_translate = fn
    out = T.translate_segments([_seg()], source_language="en", target_language="hi")

    assert calls["strict"] == 1
    assert out[0]["translated_text"] == HI_FIT
    assert out[0]["status"] == "translated"
    assert "translation_note" not in out[0]


def test_a_wrong_length_line_is_kept_not_failed():
    """A mis-sized line is still a real translation. Dropping it would punch a
    hole in the dub, which is strictly worse than a line that needs stretching.
    """
    def fn(text, *, source, target, context, target_duration_ms, strict=False):
        return HI_FIT if text.startswith(PROBE_PREFIX) else HI_SHORT

    settings.translator = "ollama"
    settings.translation_attempts = 2
    T._ollama_translate = fn
    out = T.translate_segments([_seg()], source_language="en", target_language="hi")

    assert out[0]["status"] == "translated"
    assert out[0]["translated_text"] == HI_SHORT
    assert out[0]["translation_words"] == 2
    assert "words" in out[0]["translation_note"]
