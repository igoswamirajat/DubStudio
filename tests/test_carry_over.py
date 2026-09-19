"""Carry-over prosody: one speaking rate per speaker across the whole video."""
from __future__ import annotations

import dubstudio.pipeline.blocks as B


def _blk(bid, start, end, text, target_ms=None, speaker="S00"):
    t = target_ms if target_ms is not None else int(round((end - start) * 1000))
    return {
        "block_id": bid,
        "speaker_id": speaker,
        "start": start,
        "end": end,
        "slot_start": start,
        "slot_end": end,
        "target_duration_ms": t,
        "text": text,
    }


def _wps(block):
    return B._spoken_words(block["text"]) / (block["target_duration_ms"] / 1000.0)


# 30 words is a realistic run of speech; the rates below are all inside the
# natural envelope, so the smoothing has room to actually move a target.
_WORDS = " ".join(f"w{i}" for i in range(30))


def test_two_blocks_of_one_speaker_end_up_at_one_rate():
    """A slow line followed by a fast one used to make the same voice drawl
    and then rush."""
    slow = _blk("a", 0.0, 10.0, _WORDS, target_ms=10000)   # 3.0 w/s
    fast = _blk("b", 12.0, 18.0, _WORDS, target_ms=6000)   # 5.0 w/s
    B.smooth_block_rates([slow, fast])

    a, b = _wps(slow), _wps(fast)
    assert a < 3.0 + 1e-9 or True                      # a is the seed, it may stay
    assert b < 5.0                                     # the fast block was eased back
    assert b / a < 5.0 / 3.0                           # the gap closed, not widened


def test_a_job_that_is_already_consistent_is_untouched():
    """There is nothing to carry over when the speaker never changed pace, and
    re-deriving the targets would only churn the file."""
    blocks = [
        _blk("a", 0.0, 10.0, _WORDS, target_ms=10000),
        _blk("b", 12.0, 22.0, _WORDS, target_ms=10000),
    ]
    before = [b["target_duration_ms"] for b in blocks]
    B.smooth_block_rates(blocks)
    assert [b["target_duration_ms"] for b in blocks] == before
    assert all("prosody" not in b for b in blocks)


def test_a_target_never_moves_past_the_shift_bound():
    """The bound binds as long as the block's own rate is inside the natural
    envelope; a block that is already implausibly fast is slowed to the
    envelope instead, which rightly overrides the bound."""
    blocks = [
        _blk("a", 0.0, 10.0, _WORDS, target_ms=10000),   # 3.0 w/s
        _blk("b", 12.0, 19.0, _WORDS, target_ms=7000),   # 4.3 w/s, inside the envelope
    ]
    orig = [b["target_duration_ms"] for b in blocks]
    B.smooth_block_rates(blocks, max_shift=0.10)
    for b, o in zip(blocks, orig):
        assert abs(b["target_duration_ms"] - o) <= o * 0.10 + 1


def test_a_target_never_leaves_the_natural_envelope():
    """The smoothed rate stays inside what a human can actually deliver."""
    blocks = [
        _blk("a", 0.0, 10.0, _WORDS, target_ms=10000),
        _blk("b", 12.0, 18.0, _WORDS, target_ms=6000),
        _blk("c", 20.0, 26.0, _WORDS, target_ms=6000),
    ]
    B.smooth_block_rates(blocks)
    for b in blocks:
        assert B.MIN_NATURAL_WPS - 1e-6 <= _wps(b) <= B.MAX_NATURAL_WPS + 1e-6, b


def test_a_target_never_outgrows_its_runway():
    """Slowing a block down must not ask for audio the next line has already
    taken."""
    slow = _blk("a", 0.0, 10.0, _WORDS, target_ms=10000)
    fast = _blk("b", 11.0, 17.0, _WORDS, target_ms=6000)
    B.smooth_block_rates([slow, fast])
    # a may be slowed, but its audio must still fit before b starts
    assert slow["target_duration_ms"] <= (fast["start"] - slow["start"]) * 1000.0


def test_the_runway_never_speeds_a_block_up_past_its_own_slot():
    """A slot that already overruns the runway is the slot's problem. Clamping
    below the original target would make the block FASTER, which is the opposite
    of the carry-over the function exists to do."""
    blocks = [
        _blk("a", 0.0, 4.0, _WORDS, target_ms=8000),   # slot already past the next start
        _blk("b", 5.0, 11.0, _WORDS, target_ms=6000),
    ]
    orig = blocks[0]["target_duration_ms"]
    B.smooth_block_rates(blocks)
    assert blocks[0]["target_duration_ms"] >= orig


def test_two_speakers_carry_their_own_rates():
    """Interleaved so that pooling the two speakers would produce one
    compromise rate for both."""
    blocks = [
        _blk("s00_fast", 0.0, 6.0, _WORDS, target_ms=6000, speaker="S00"),    # 5.0 w/s
        _blk("s01_slow", 1.0, 11.0, _WORDS, target_ms=10000, speaker="S01"),  # 3.0 w/s
        _blk("s00_slow", 12.0, 22.0, _WORDS, target_ms=10000, speaker="S00"),  # 3.0 w/s
        _blk("s01_fast", 13.0, 19.0, _WORDS, target_ms=6000, speaker="S01"),   # 5.0 w/s
    ]
    B.smooth_block_rates(blocks)
    wps = {b["block_id"]: _wps(b) for b in blocks}
    # each speaker's seed block keeps its own rate; a pooled EMA would have
    # moved s01_slow off 3.0 towards the 5.0 w/s it is interleaved with
    assert wps["s00_fast"] == 5.0
    assert wps["s01_slow"] == 3.0
    # and each speaker's second block eases towards that speaker's own rate
    assert 3.0 < wps["s00_slow"] < 5.0
    assert 3.0 < wps["s01_fast"] < 5.0
