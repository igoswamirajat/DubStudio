"""Speaker labels: kill the flicker, keep the conversation.

Words used to be handed to the *nearest* pyannote turn, which sprayed 0.12-0.21 s
speaker changes through single sentences. Every flicker became its own speech
block, its own TTS call and its own voice, so one sentence could come out in two
voices. These tests pin both halves of the fix: flickers die, real turn taking
survives.
"""
from __future__ import annotations

from dubstudio.pipeline import diarization as D

STEP = 0.40          # one word every 400 ms
SPOKEN = 0.32        # of which 320 ms is voiced


def _words(spec, step: float = STEP):
    """spec: [(speaker, word_count), ...] laid end to end on the timeline."""
    words = []
    t = 0.0
    for speaker, count in spec:
        for _ in range(count):
            words.append({
                "word": "w{}".format(len(words)),
                "start": round(t, 3),
                "end": round(t + SPOKEN, 3),
                "speaker": speaker,
            })
            t += step
    return words


def _labels(words):
    return [w["speaker"] for w in words]


# --- flickers must die ------------------------------------------------------
def test_a_single_flicker_inside_a_sentence_is_removed():
    words = _words([("S00", 4), ("S01", 1), ("S00", 4)])
    assert D.smooth_word_speakers(words) == 1
    assert set(_labels(words)) == {"S00"}


def test_a_storm_of_flickers_collapses_to_one_voice():
    words = _words([("S00", 3), ("S01", 1), ("S00", 1), ("S01", 1),
                    ("S00", 1), ("S01", 1), ("S00", 3)])
    D.smooth_word_speakers(words)
    assert set(_labels(words)) == {"S00"}


def test_a_crumb_at_the_start_joins_the_sentence():
    words = _words([("S01", 1), ("S00", 6)])
    assert D.smooth_word_speakers(words) == 1
    assert set(_labels(words)) == {"S00"}


def test_a_crumb_at_the_end_joins_the_sentence():
    words = _words([("S00", 6), ("S01", 1)])
    assert D.smooth_word_speakers(words) == 1
    assert set(_labels(words)) == {"S00"}


# --- real dialogue must survive --------------------------------------------
def test_real_turn_taking_is_left_alone():
    words = _words([("S00", 5), ("S01", 5), ("S00", 5)])
    assert D.smooth_word_speakers(words) == 0
    assert _labels(words)[5:10] == ["S01"] * 5


def test_a_short_but_real_turn_between_two_speakers_survives():
    # a two word answer between two different voices is an answer, not a glitch
    words = _words([("S00", 3), ("S01", 2), ("S02", 3)])
    assert D.smooth_word_speakers(words) == 0
    assert _labels(words)[3:5] == ["S01", "S01"]


def test_a_single_speaker_transcript_is_untouched():
    words = _words([("S00", 8)])
    assert D.smooth_word_speakers(words) == 0
    assert set(_labels(words)) == {"S00"}


def test_a_two_word_transcript_is_untouched():
    words = _words([("S00", 1), ("S01", 1)])
    assert D.smooth_word_speakers(words) == 0
    assert _labels(words) == ["S00", "S01"]


def test_smoothing_touches_labels_and_nothing_else():
    words = _words([("S00", 3), ("S01", 1), ("S00", 3)])
    before = [(w["word"], w["start"], w["end"]) for w in words]
    D.smooth_word_speakers(words)
    assert [(w["word"], w["start"], w["end"]) for w in words] == before


# --- who owns a word -------------------------------------------------------
def test_a_word_belongs_to_the_turn_it_overlaps_most():
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    assert D._speaker_for(turns, 1.8, 2.1) == "SPEAKER_00"
    assert D._speaker_for(turns, 1.9, 2.6) == "SPEAKER_01"


def test_a_word_in_a_gap_goes_to_the_nearest_turn():
    turns = [(0.0, 1.0, "SPEAKER_00"), (5.0, 6.0, "SPEAKER_01")]
    assert D._speaker_for(turns, 1.2, 1.4) == "SPEAKER_00"
    assert D._speaker_for(turns, 4.5, 4.7) == "SPEAKER_01"


def test_no_turns_means_no_guess():
    assert D._speaker_for([], 0.0, 1.0) is None
