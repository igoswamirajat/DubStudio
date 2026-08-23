from dubstudio.pipeline.segments import words_to_segments


def test_merge_same_speaker_small_pause():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
        {"word": "world", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
        {"word": "again", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},
    ]
    segs = words_to_segments(words, job_id="job_x")
    assert len(segs) == 2
    assert segs[0]["source_text"] == "Hello world"
    assert segs[0]["speaker_id"] == "S00"


def test_split_on_speaker_change():
    words = [
        {"word": "A", "start": 0.0, "end": 0.3, "speaker": "S00"},
        {"word": "B", "start": 0.35, "end": 0.6, "speaker": "S01"},
    ]
    segs = words_to_segments(words, job_id="job_x")
    assert len(segs) == 2
    assert segs[0]["speaker_id"] == "S00"
    assert segs[1]["speaker_id"] == "S01"
