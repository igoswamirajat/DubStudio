from dubstudio.pipeline.segments import SPLIT_PAUSE_S, words_to_segments


def test_merge_same_speaker_small_pause():
    """A 0.6s hesitation is NOT a turn boundary; SPLIT_PAUSE_S is 0.70s.

    This used to split, which is what made the dub restart its prosody after
    every line. Both units here are also long enough to survive the min-length
    merge, so the split really is decided by the pause threshold.
    """
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"},
        {"word": "world", "start": 0.7, "end": 1.4, "speaker": "SPEAKER_00"},
        {"word": "again", "start": 2.0, "end": 3.0, "speaker": "SPEAKER_00"},
    ]
    segs = words_to_segments(words, job_id="job_x")
    assert len(segs) == 1
    assert segs[0]["source_text"] == "Hello world again"
    assert segs[0]["speaker_id"] == "S00"


def test_a_real_breath_does_split():
    """At SPLIT_PAUSE_S or beyond the utterance breaks, as it should.

    Both sides clear MIN_SEG_S, otherwise the min-length merge folds the short
    one back in and the split disappears. The gap is nudged past the threshold
    because 1.4 + 0.70 evaluates to 0.6999999... in binary floating point.
    """
    breath_at = round(1.4 + SPLIT_PAUSE_S + 0.05, 3)
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"},
        {"word": "world", "start": 0.7, "end": 1.4, "speaker": "SPEAKER_00"},
        {"word": "again", "start": breath_at, "end": breath_at + 1.6, "speaker": "SPEAKER_00"},
    ]
    segs = words_to_segments(words, job_id="job_x")
    assert len(segs) == 2
    assert segs[0]["source_text"] == "Hello world"
    assert segs[1]["source_text"] == "again"


def test_split_on_speaker_change():
    words = [
        {"word": "A", "start": 0.0, "end": 0.3, "speaker": "S00"},
        {"word": "B", "start": 0.35, "end": 0.6, "speaker": "S01"},
    ]
    segs = words_to_segments(words, job_id="job_x")
    assert len(segs) == 2
    assert segs[0]["speaker_id"] == "S00"
    assert segs[1]["speaker_id"] == "S01"
