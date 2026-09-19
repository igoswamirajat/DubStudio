"""The global timeline solver: one speaking rate for the whole video."""
from __future__ import annotations

from dubstudio.pipeline.timeline import (
    MAX_EASE,
    MAX_STRETCH,
    runway_for,
    solve_timeline,
    unit_starts,
)


def _blk(bid, start, content_ms, target_ms, speaker="S00", slot_end=None):
    return {
        "block_id": bid,
        "speaker_id": speaker,
        "start": start,
        "slot_start": start,
        "slot_end": slot_end if slot_end is not None else start + target_ms / 1000.0,
        "target_duration_ms": target_ms,
        "content_ms": content_ms,
    }


# --- runway ----------------------------------------------------------------

def test_the_runway_stops_at_the_next_speech():
    starts = unit_starts([_blk("a", 0.0, 1000, 1000), _blk("b", 5.0, 1000, 1000)])
    assert runway_for(0.0, starts, 60.0) == 5.0
    assert runway_for(5.0, starts, 60.0) == 55.0


def test_solo_cues_count_as_the_next_speech():
    """A cue that stayed on its own is still laid down, so it bounds the runway."""
    starts = unit_starts([_blk("a", 0.0, 1000, 1000)], extra_starts=[3.0])
    assert runway_for(0.0, starts, 60.0) == 3.0


def test_the_video_end_is_the_last_resort():
    starts = unit_starts([_blk("a", 10.0, 1000, 1000)])
    assert runway_for(10.0, starts, 42.0) == 32.0


# --- the global rate --------------------------------------------------------

def test_two_blocks_share_one_rate_instead_of_diverging():
    """0.92 and 1.08 on their own must meet in the middle.

    That lurch from one block to the next is the whole reason the solver exists.
    """
    blocks = [_blk("a", 0.0, 920, 1000), _blk("b", 5.0, 1080, 1000)]
    sol = solve_timeline(blocks)
    ratios = [sol["per_block"][b["block_id"]]["ratio"] for b in blocks]
    assert max(ratios) - min(ratios) < 0.16, ratios
    # and both stay inside the legal band
    assert all(1 - MAX_STRETCH - 1e-9 <= r <= 1 + MAX_STRETCH + 1e-9 for r in ratios)


def test_a_block_dominating_the_video_keeps_its_own_rate():
    """A long block IS the global rate; a one-liner next to it should follow it,
    not impose its own."""
    big = _blk("big", 0.0, 10000, 10800)     # ~8% slow
    small = _blk("small", 20.0, 1000, 800)   # 25% fast on its own
    sol = solve_timeline([big, small])
    assert abs(sol["per_block"]["big"]["ratio"] - sol["per_block"]["big"]["wanted_ratio"]) < 0.05
    # the small block is eased towards the big one's pace, not left at its own
    assert sol["per_block"]["small"]["ratio"] < sol["per_block"]["small"]["wanted_ratio"]


def test_two_speakers_get_two_rates():
    """Averaging a fast and a slow talker would be wrong for both."""
    blocks = [
        _blk("a1", 0.0, 900, 1000, speaker="S00"),
        _blk("b1", 5.0, 1100, 1000, speaker="S01"),
    ]
    sol = solve_timeline(blocks)
    assert sol["speaker_ratio"]["S00"] < 1.0
    assert sol["speaker_ratio"]["S01"] > 1.0


def test_a_perfectly_paced_job_is_left_alone():
    blocks = [_blk("a", 0.0, 1000, 1000), _blk("b", 5.0, 2000, 2000)]
    sol = solve_timeline(blocks)
    assert sol["global_ratio"] == 1.0
    assert all(v["ratio"] == 1.0 for v in sol["per_block"].values())
    assert sol["collisions"] == []


def test_no_blocks_is_not_an_error():
    sol = solve_timeline([])
    assert sol["per_block"] == {} and sol["collisions"] == []


# --- collisions -------------------------------------------------------------

def test_a_block_that_would_be_trimmed_is_compressed_and_reported():
    """Content longer than the runway must be squeezed in, not cut.

    Without this the mixer's hard trim silently deletes the last words of the
    block.
    """
    blocks = [
        _blk("a", 0.0, 3000, 1000),   # 3s of speech but only 1.5s before the next
        _blk("b", 1.5, 1000, 1000),
    ]
    sol = solve_timeline(blocks)
    hit = sol["per_block"]["a"]
    assert hit["collided"] is True
    assert "a" in sol["collisions"]
    assert hit["fitted_ms"] <= hit["runway_ms"], "must fit the runway after compression"


def test_a_block_with_room_to_spill_is_not_flagged():
    """Overrunning your own slot is fine when nothing starts for a while."""
    blocks = [
        _blk("a", 0.0, 1500, 1000, slot_end=1.0),
        _blk("b", 10.0, 1000, 1000),
    ]
    sol = solve_timeline(blocks)
    assert sol["collisions"] == []
    assert sol["per_block"]["a"]["collided"] is False


def test_easing_never_moves_a_block_past_the_truth():
    """MAX_EASE is the guard against smoothing a block into an audible mangling.

    It can only be honoured for a block whose honest ratio is itself inside the
    legal band; a block that is 25% off cannot be fitted at all, and the band
    clamp rightly overrides the easing.
    """
    blocks = [_blk("a", 0.0, 1000, 1000), _blk("b", 5.0, 1000, 1250)]
    sol = solve_timeline(blocks)
    for bid, hit in sol["per_block"].items():
        if abs(hit["wanted_ratio"] - 1.0) <= MAX_STRETCH:
            assert abs(hit["ratio"] - hit["wanted_ratio"]) <= MAX_EASE + 1e-9, (bid, hit)
