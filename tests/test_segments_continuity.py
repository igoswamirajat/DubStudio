"""Continuity tests for the segmenter.

The dub used to stutter after every line because words_to_segments split on any
pause >= 0.4s with no minimum unit length, so a single sentence became several
independent TTS calls with their own prosody, loudness, stretch ratio and hard
joins. These tests pin utterance-level grouping and contiguous slots.
"""
from __future__ import annotations

from dubstudio.pipeline.segments import ABSORB_GAP_S, MAX_SEG_S, words_to_segments

LEGACY_KEYS = (
    "job_id", "segment_id", "speaker_id", "voice_id", "start", "end",
    "target_duration_ms", "source_text", "translated_text", "source_words",
    "generated_wav", "fitted_wav", "generated_duration_ms", "fitted_duration_ms",
    "stretch_ratio", "rewrite_attempts", "status", "notes",
)


def w(word: str, start: float, end: float, spk: str = "S00") -> dict:
    return {"word": word, "start": round(start, 3), "end": round(end, 3),
            "score": 0.9, "speaker": spk}


def hesitant_stream(n: int = 24, word_s: float = 0.30, pause_s: float = 0.45) -> list[dict]:
    """Words separated by hesitation-length pauses, as real ASR timings look."""
    out, t = [], 0.0
    for i in range(n):
        out.append(w(f"f{i}", t, t + word_s))
        t += word_s + pause_s
    return out


# --- splitting rules -------------------------------------------------------
def test_intra_sentence_pause_does_not_split():
    words = [w("one", 0.0, 0.4), w("small", 0.55, 0.95), w("pause", 1.10, 1.60)]
    assert len(words_to_segments(words, job_id="j", min_seg_s=0.0)) == 1


def test_real_breath_splits():
    words = [w("first", 0.0, 1.40), w("second", 2.40, 3.80)]
    assert len(words_to_segments(words, job_id="j", min_seg_s=0.0)) == 2


def test_full_stop_plus_short_pause_splits():
    words = [w("done.", 0.0, 1.40), w("next", 1.75, 3.20)]
    assert len(words_to_segments(words, job_id="j", min_seg_s=0.0)) == 2


def test_same_pause_without_punctuation_does_not_split():
    words = [w("going", 0.0, 1.40), w("on", 1.75, 3.20)]
    assert len(words_to_segments(words, job_id="j", min_seg_s=0.0)) == 1


def test_speaker_change_always_splits_and_normalises_ids():
    words = [w("hi", 0.0, 1.30, "S00"), w("hello", 1.35, 2.70, "SPEAKER_01")]
    cues = words_to_segments(words, job_id="j")
    assert [c["speaker_id"] for c in cues] == ["S00", "S01"]


def test_max_seg_s_caps_utterance_length():
    words = [w(f"t{i}", i * 0.30, i * 0.30 + 0.28) for i in range(80)]
    cues = words_to_segments(words, job_id="j")
    assert cues
    assert all((c["end"] - c["start"]) <= MAX_SEG_S + 1e-3 for c in cues)


# --- the fragmentation regression -----------------------------------------
def test_hesitations_no_longer_fragment_a_line():
    words = hesitant_stream()
    old = words_to_segments(words, job_id="j", max_pause_s=0.4, min_seg_s=0.0, max_seg_s=8.0)
    new = words_to_segments(words, job_id="j")
    assert len(old) >= 20      # one fragment per word under the old rule
    assert len(new) <= 3


def test_merge_pass_removes_micro_segments():
    new = words_to_segments(hesitant_stream(), job_id="j")
    assert all((c["end"] - c["start"]) >= 0.5 for c in new)


def test_old_behaviour_is_still_reachable():
    words = hesitant_stream(n=6)
    assert len(words_to_segments(words, job_id="j", max_pause_s=0.4, min_seg_s=0.0)) == 6


# --- slot / continuity contract -------------------------------------------
def _clip_like_stream() -> list[dict]:
    """Three utterances: two joined by a 0.2s gap, then a 0.9s breath."""
    return [
        w("aa", 0.00, 0.60), w("bb", 0.70, 1.40),
        w("cc", 1.60, 2.20), w("dd", 2.30, 3.00),
        w("ee", 3.90, 4.60), w("ff", 4.70, 5.40),
    ]


def _sentence_split_stream() -> list[dict]:
    """Two full utterances whose only split is a short punctuation break.

    SPLIT_PAUSE_S is 0.70, so a sub-threshold pause normally stays inside one
    utterance. After sentence-final punctuation PUNCT_PAUSE_S (0.28s) is enough
    to split - and that is the case where the leftover 0.30s gap has to be
    absorbed so the two slots stay contiguous. Each side is over MIN_SEG_S so
    the min-length merge cannot undo the split.
    """
    return [
        w("aa", 0.00, 0.70), w("bb.", 0.80, 1.50),
        w("cc", 1.80, 2.60), w("dd", 2.70, 3.40),
    ]


def test_absorbed_gap_makes_slots_contiguous():
    cues = words_to_segments(_sentence_split_stream(), job_id="j", min_seg_s=0.0)
    assert len(cues) == 2, "a sentence break should split the two utterances"
    absorbed = [i for i in range(len(cues) - 1) if cues[i]["absorbed_gap"]]
    assert absorbed, "expected the sub-0.35s gap to be absorbed"
    for i in absorbed:
        assert abs(cues[i]["slot_end"] - cues[i + 1]["start"]) < 1e-6


def test_a_gap_over_the_absorb_limit_is_not_absorbed():
    """Only sub-ABSORB_GAP_S gaps get welded; a real pause must survive."""
    cues = words_to_segments(_clip_like_stream(), job_id="j", min_seg_s=0.0)
    assert len(cues) == 2
    assert not any(c["absorbed_gap"] for c in cues)


def test_genuine_pause_is_preserved():
    cues = words_to_segments(_clip_like_stream(), job_id="j", min_seg_s=0.0)
    long_gaps = [c for c in cues if c["gap_after_ms"] and c["gap_after_ms"] > ABSORB_GAP_S * 1000]
    assert long_gaps
    for c in long_gaps:
        assert not c["absorbed_gap"]
        assert c["slot_end"] < c["end"] + 0.2


def test_slot_never_overruns_the_next_line():
    cues = words_to_segments(hesitant_stream(n=40), job_id="j")
    for a, b in zip(cues, cues[1:]):
        assert a["slot_end"] <= b["start"] + 1e-6


def test_slot_is_at_least_the_speech_window():
    for c in words_to_segments(hesitant_stream(n=40), job_id="j"):
        assert c["target_duration_ms"] >= c["speech_duration_ms"]


# --- schema compatibility -------------------------------------------------
def test_legacy_keys_are_preserved():
    for c in words_to_segments(hesitant_stream(n=8), job_id="j"):
        for key in LEGACY_KEYS:
            assert key in c


def test_segment_ids_are_dense_and_ordered():
    cues = words_to_segments(hesitant_stream(n=40), job_id="j")
    assert [c["segment_id"] for c in cues] == [f"seg_{i:04d}" for i in range(len(cues))]


def test_empty_input_is_safe():
    assert words_to_segments([], job_id="j") == []


def test_blank_words_are_dropped():
    cues = words_to_segments([w("  ", 0.0, 0.3), w("real", 0.4, 1.9)], job_id="j")
    assert cues[0]["source_text"] == "real"
