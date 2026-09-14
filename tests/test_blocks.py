"""How cues are grouped into speech blocks.

A block is the unit that gets ONE TTS call, so these rules decide whether the
dub flows like a person talking or restarts every couple of seconds.
"""
from __future__ import annotations

import json

import dubstudio.pipeline.blocks as B


def _seg(i, start, end, text="one two three", speaker="S00", slot_end=None, **extra):
    seg = {
        "segment_id": f"seg_{i:04d}",
        "speaker_id": speaker,
        "voice_id": speaker,
        "start": start,
        "end": end,
        "slot_start": start,
        "slot_end": end if slot_end is None else slot_end,
        "source_text": "source line",
        "translated_text": text,
        "target_duration_ms": int(round((end - start) * 1000)),
        "status": "translated",
    }
    seg.update(extra)
    return seg


def test_a_continuous_run_becomes_one_block():
    segs = [
        _seg(0, 0.0, 2.0, "one two"),
        _seg(1, 2.2, 4.0, "three four"),
        _seg(2, 4.15, 6.0, "five six"),
    ]
    blocks = B.group_blocks(segs)
    assert len(blocks) == 1
    assert blocks[0]["segment_ids"] == ["seg_0000", "seg_0001", "seg_0002"]
    assert blocks[0]["text"] == "one two three four five six"
    assert blocks[0]["start"] == 0.0 and blocks[0]["end"] == 6.0


def test_a_real_beat_starts_a_new_block():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 3.6, 5.0)]   # 1.6s of silence between
    blocks = B.group_blocks(segs)
    assert len(blocks) == 2


def test_a_short_breath_does_not_start_a_new_block():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 3.0, 4.5)]   # 1.0s: a breath, not a beat
    assert len(B.group_blocks(segs)) == 1


def test_a_speaker_change_always_starts_a_new_block():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 2.05, 4.0, speaker="S01")]
    blocks = B.group_blocks(segs)
    assert len(blocks) == 2
    assert [b["speaker_id"] for b in blocks] == ["S00", "S01"]


def test_a_block_stops_growing_at_the_time_cap():
    segs = [_seg(i, i * 2.1, i * 2.1 + 2.0, "aa bb") for i in range(20)]
    blocks = B.group_blocks(segs)
    assert len(blocks) > 1
    assert all((b["end"] - b["start"]) <= B.MAX_BLOCK_S for b in blocks)


def test_a_block_stops_growing_at_the_text_cap():
    segs = [_seg(i, i * 2.1, i * 2.1 + 2.0, "word word word word") for i in range(6)]
    blocks = B.group_blocks(segs, max_block_chars=40)
    assert len(blocks) > 1
    assert all(len(b["text"]) <= 40 for b in blocks)


def test_an_untranslated_cue_is_left_out_and_breaks_the_run():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 2.1, 3.0, ""), _seg(2, 3.1, 5.0)]
    blocks = B.group_blocks(segs)
    assert len(blocks) == 2
    assert all("seg_0001" not in b["segment_ids"] for b in blocks)


def test_a_failed_cue_is_left_out_even_with_text():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 2.1, 3.0, status="failed"), _seg(2, 3.1, 5.0)]
    blocks = B.group_blocks(segs)
    assert len(blocks) == 2


def test_the_slot_spans_the_whole_run():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 2.2, 4.0, slot_end=6.5)]
    block = B.group_blocks(segs)[0]
    assert block["slot_start"] == 0.0 and block["slot_end"] == 6.5
    assert block["target_duration_ms"] == 6500


def test_a_pause_inside_a_block_is_capped():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 3.0, 4.5)]   # 1.0s pause, under the beat
    block = B.group_blocks(segs)[0]
    assert block["parts"][0]["gap_after_ms"] == 1000
    assert block["internal_pauses_ms"] == [int(B.MAX_SYNTH_PAUSE_S * 1000)]


def test_a_noticeable_pause_without_punctuation_becomes_a_comma():
    segs = [_seg(0, 0.0, 2.0, "one two"), _seg(1, 2.5, 4.0, "three four")]
    assert B.group_blocks(segs)[0]["text"] == "one two, three four"


def test_existing_punctuation_is_never_doubled():
    segs = [_seg(0, 0.0, 2.0, "one two."), _seg(1, 2.5, 4.0, "three four")]
    assert B.group_blocks(segs)[0]["text"] == "one two. three four"
    segs = [_seg(0, 0.0, 2.0, "one two,"), _seg(1, 2.5, 4.0, "three four")]
    assert B.group_blocks(segs)[0]["text"] == "one two, three four"


def test_a_tight_join_gets_no_comma():
    segs = [_seg(0, 0.0, 2.0, "one two"), _seg(1, 2.1, 4.0, "three four")]
    assert B.group_blocks(segs)[0]["text"] == "one two three four"


def test_stamping_marks_lead_member_and_solo():
    segs = [_seg(0, 0.0, 2.0), _seg(1, 2.1, 4.0), _seg(2, 6.0, 7.0, "")]
    blocks = B.group_blocks(segs)
    B.stamp_segments(segs, blocks)
    assert [s["block_role"] for s in segs] == ["lead", "member", "solo"]
    assert segs[0]["block_id"] == segs[1]["block_id"] == "blk_0000"
    assert segs[2]["block_id"] is None
    assert segs[1]["block_index"] == 1 and segs[1]["block_size"] == 2


def test_run_blocks_publishes_both_files(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / "segments").mkdir(parents=True)
    segs = [_seg(0, 0.0, 2.0), _seg(1, 2.2, 4.0), _seg(2, 9.0, 11.0)]
    (job_dir / "segments" / "segments.json").write_text(json.dumps(segs), encoding="utf-8")

    blocks = B.run_blocks(job_dir)
    assert len(blocks) == 2
    assert B.load_blocks(job_dir) == blocks

    saved = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    assert [s["block_role"] for s in saved] == ["lead", "member", "lead"]
    assert saved[2]["block_id"] == "blk_0001"


def test_cues_are_grouped_in_timeline_order_not_file_order(tmp_path):
    segs = [_seg(1, 2.2, 4.0, "second"), _seg(0, 0.0, 2.0, "first")]
    block = B.group_blocks(segs)[0]
    assert block["text"] == "first second"
    assert block["segment_ids"] == ["seg_0000", "seg_0001"]
