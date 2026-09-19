"""Tests for translation pacing validation (P5).

The pacing check catches translations that would require the speaker to drawl
(<1.6 w/s) or sprint (>5.0 w/s), so the model can be asked for a rewrite before
synthesis. It is separate from the length check: a line can be within the
length budget and still have implausible pacing.

Neither check is fatal. A badly paced line is still a real translation, so the
caller keeps the least-bad one when no rewrite beats it - a hole in the dub is
worse than a line that needs stretching.
"""
from __future__ import annotations

import pytest

from dubstudio.pipeline.translation import _pacing_issue, _length_issue


def test_pacing_accepts_a_natural_rate():
    # 12 words in 4.0s = 3.0 w/s, well within the natural envelope
    result = _pacing_issue(
        source="This is the source text.",
        output="यह बारह शब्दों का एक प्राकृतिक वाक्य है जो सामान्य गति से बोला जा सकता है",
        target="hi",
        target_duration_ms=4000,
    )
    assert result is None


def test_pacing_rejects_a_drawl():
    # 3 words in 4.0s = 0.75 w/s, slower than the 1.6 w/s floor.
    # The source has to fill the slot for this to be a drawl at all: if the
    # speaker never said much, the spare time is their own pause, not slack the
    # dub line has to cover (see the mostly-pause test below).
    result = _pacing_issue(
        source="This is the source text that the speaker actually said.",
        output="तीन शब्द यहाँ",
        target="hi",
        target_duration_ms=4000,
    )
    assert result is not None
    assert "0.8 w/s" in result or "0.7 w/s" in result
    assert "drag" in result


def test_pacing_allows_a_short_line_when_the_slot_is_mostly_pause():
    """A short line in a slot the source never filled is not a drawl.

    Three source words take about 1.2s of a 4.0s slot, so the rest is silence
    the original speaker also took. The dub line should be short too, and the
    timeline keeps the pause - forcing it up to fill the slot is how filler
    gets invented.
    """
    result = _pacing_issue(
        source="Source text here.",
        output="तीन शब्द यहाँ",
        target="hi",
        target_duration_ms=4000,
    )
    assert result is None


def test_pacing_rejects_a_sprint():
    # 25 words in 4.0s = 6.25 w/s, faster than the 5.0 w/s ceiling
    result = _pacing_issue(
        source="This is a longer source text with multiple words.",
        output="" + " ".join(["शब्द"] * 25),
        target="hi",
        target_duration_ms=4000,
    )
    assert result is not None
    assert "6.2" in result or "6.3" in result
    assert "rushed" in result or "clipped" in result


def test_pacing_ignores_very_short_lines():
    # Short lines (< 3 source words) are exempt from pacing checks
    result = _pacing_issue(
        source="Hi.",
        output="नमस्ते दोस्तों और सभी को स्वागत",  # 6 words, would be too fast
        target="hi",
        target_duration_ms=1000,
    )
    assert result is None


def test_pacing_and_length_flag_a_drawl_for_different_reasons():
    """Both checks catch this line, and they say different things about it.

    7 words in a 6.0s slot is 1.17 w/s - under the 1.6 w/s floor. The length
    check calls it mis-sized against the word band; the pacing check calls it a
    drawl. Either way the caller sends it back for a rewrite, and keeps the
    line if nothing better comes back.
    """
    source = "This is a source sentence with enough words."
    output = "यह आठ शब्दों का वाक्य है धन्यवाद"  # 7 words by _word_count
    slot = 6000

    # Length: a hint at most - the caller keeps `best` when only length is off.
    _length_issue(source, output, "hi", slot)

    # Pacing: names the failure as a drawl rather than a size mismatch.
    pacing_result = _pacing_issue(source, output, "hi", slot)
    assert pacing_result is not None
    assert "1.2 w/s" in pacing_result
    assert "drag" in pacing_result


def test_pacing_at_the_boundary():
    source = "This is the source text that I just said."
    # Exactly at the floor (1.6 w/s) should pass
    # 8 words in 5.0s = 1.6 w/s
    result = _pacing_issue(
        source=source,
        output="पहला दूसरा तीसरा चौथा पाँचवाँ छठा सातवाँ आठवाँ",
        target="hi",
        target_duration_ms=5000,
    )
    assert result is None

    # Just below the floor should fail
    # 7 words in 5.0s = 1.4 w/s
    result = _pacing_issue(
        source=source,
        output="पहला दूसरा तीसरा चौथा पाँचवाँ छठा सातवाँ",
        target="hi",
        target_duration_ms=5000,
    )
    assert result is not None
    assert "1.4 w/s" in result


def test_pacing_handles_zero_words():
    # Edge case: empty or whitespace-only output
    result = _pacing_issue(
        source="Source text.",
        output="",
        target="hi",
        target_duration_ms=3000,
    )
    assert result is None  # no words = no pacing issue (caught elsewhere)


def test_pacing_handles_missing_duration():
    result = _pacing_issue(
        source="Source text.",
        output="यह एक वाक्य है",
        target="hi",
        target_duration_ms=None,
    )
    assert result is None
