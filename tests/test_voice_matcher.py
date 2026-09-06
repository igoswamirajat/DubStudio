"""Unit tests for Voice Matcher, Acoustic Profiling, and Phonetics Number Normalization."""

from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from dubstudio.pipeline.voice_profile import AcousticProfile, extract_acoustic_profile
from dubstudio.pipeline.voice_matcher import match_voices_for_job, score_voice_match, VEENA_VOICES
from dubstudio.util.phonetics import expand_hindi_numbers, normalize_hinglish


def test_expand_hindi_numbers():
    assert "तीस हज़ार" in expand_hindi_numbers("30,000")
    assert "सौ" in expand_hindi_numbers("100")
    assert "पांच" in expand_hindi_numbers("5")
    assert "प्रतिशत" in expand_hindi_numbers("50%")
    assert "डॉलर" in expand_hindi_numbers("$100")


def test_normalize_hinglish_integration():
    raw = "GitHub पर 30,000 से ज़्यादा stars मिले हैं और $50 का फ़ायदा है।"
    norm = normalize_hinglish(raw, "hi")
    assert "गिटहब" in norm
    assert "तीस हज़ार" in norm
    assert "डॉलर" in norm


def test_extract_acoustic_profile_male(tmp_path: Path):
    # Generate 2 seconds of 120Hz male tone with low harmonics
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.sin(2 * np.pi * 240 * t)
    wav_path = tmp_path / "male_ref.wav"
    sf.write(str(wav_path), wave.astype(np.float32), sr)

    prof = extract_acoustic_profile(wav_path, speaker_id="S00")
    assert prof.gender == "male"
    assert 100.0 <= prof.f0_median <= 140.0


def test_extract_acoustic_profile_female(tmp_path: Path):
    # Generate 2 seconds of 220Hz female tone
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * 220 * t)
    wav_path = tmp_path / "female_ref.wav"
    sf.write(str(wav_path), wave.astype(np.float32), sr)

    prof = extract_acoustic_profile(wav_path, speaker_id="S01")
    assert prof.gender == "female"
    assert 200.0 <= prof.f0_median <= 240.0


def test_single_speaker_voice_matching_locks_voice(tmp_path: Path):
    # Setup single speaker S00 with male ref
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * 115 * t)
    v_dir = tmp_path / "voices" / "S00"
    v_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(v_dir / "ref.wav"), wave.astype(np.float32), sr)

    speakers = [{"speaker_id": "S00"}]
    matched = match_voices_for_job(speakers, tmp_path, tts_engine="veena")

    assert "S00" in matched
    assert matched["S00"]["gender"] == "male"
    assert matched["S00"]["voice_id"] in {"agastya", "vinaya"}


def test_multi_speaker_voice_matching_avoids_collision(tmp_path: Path):
    sr = 24000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)

    # Speaker S00: Deep male (95 Hz)
    v0 = tmp_path / "voices" / "S00"
    v0.mkdir(parents=True, exist_ok=True)
    sf.write(str(v0 / "ref.wav"), (0.3 * np.sin(2 * np.pi * 95 * t)).astype(np.float32), sr)

    # Speaker S01: Mid male (135 Hz)
    v1 = tmp_path / "voices" / "S01"
    v1.mkdir(parents=True, exist_ok=True)
    sf.write(str(v1 / "ref.wav"), (0.3 * np.sin(2 * np.pi * 135 * t)).astype(np.float32), sr)

    # Speaker S02: Female (220 Hz)
    v2 = tmp_path / "voices" / "S02"
    v2.mkdir(parents=True, exist_ok=True)
    sf.write(str(v2 / "ref.wav"), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)

    speakers = [{"speaker_id": "S00"}, {"speaker_id": "S01"}, {"speaker_id": "S02"}]
    matched = match_voices_for_job(speakers, tmp_path, tts_engine="veena")

    # Distinct characters must get distinct voices
    assigned = [matched[s["speaker_id"]]["voice_id"] for s in speakers]
    assert len(assigned) == len(set(assigned)), f"Voice collision detected: {assigned}"
    assert matched["S02"]["gender"] == "female"
    assert matched["S00"]["gender"] == "male"
    assert matched["S01"]["gender"] == "male"
