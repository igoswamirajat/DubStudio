from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dubstudio.pipeline.mixing import _build_duck_envelope, _soft_limit, run_mixing
from dubstudio.util.audio import apply_micro_fades, resample_audio, read_audio


def test_resample_audio_exact_shape():
    sr_orig = 24000
    sr_target = 48000
    data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr_orig, endpoint=False)).astype(np.float32)
    resampled = resample_audio(data, sr_orig, sr_target)
    assert len(resampled) == 48000
    # Peak amplitude should be preserved accurately without clipping
    assert abs(np.max(np.abs(resampled)) - 1.0) < 0.05


def test_apply_micro_fades():
    data = np.ones(4800, dtype=np.float32)
    faded = apply_micro_fades(data, fade_ms=5.0, sample_rate=48000)
    # Start and end should smoothly ramp to 0
    assert faded[0] < 0.05
    assert faded[-1] < 0.05
    # Middle should remain 1.0
    assert faded[len(faded) // 2] == 1.0


def test_duck_envelope_bridges_short_pauses():
    sr = 48000
    duration_s = 6.0
    n = int(duration_s * sr)
    # Two dialogue segments: 1.0s to 2.5s, and 2.7s to 4.0s (pause of 200ms < 350ms bridge)
    intervals = [(1.0, 2.5), (2.7, 4.0)]
    env = _build_duck_envelope(intervals, n, sr=sr, duck_gain=0.22, bridge_gap_s=0.35)

    assert env.shape == (n,)
    # Outside dialogue should be unattenuated (1.0)
    assert env[int(0.5 * sr)] == 1.0
    assert env[int(5.5 * sr)] == 1.0
    # Inside first segment
    assert round(float(env[int(1.5 * sr)]), 2) == 0.22
    # In the 200ms gap between segments: must remain ducked, NOT pump up
    assert round(float(env[int(2.6 * sr)]), 2) == 0.22
    # Inside second segment
    assert round(float(env[int(3.5 * sr)]), 2) == 0.22


def test_soft_limiter_preserves_dynamics():
    # Signal with peaks at 2.0
    mix = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    limited = _soft_limit(mix, threshold=0.88)
    assert np.all(np.abs(limited) <= 0.995)
    # Below threshold values should remain unchanged
    assert limited[3] == 0.0
    assert abs(limited[2] - (-0.5)) < 1e-5
    assert abs(limited[4] - 0.5) < 1e-5


def test_run_mixing_stereo_bed_preservation(tmp_path: Path):
    sr = 48000
    duration_s = 2.0
    n = int(duration_s * sr)

    # Create stereo bed.wav
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    stereo_bed = np.zeros((n, 2), dtype=np.float32)
    stereo_bed[:, 0] = 0.2  # Left channel
    stereo_bed[:, 1] = 0.4  # Right channel
    sf.write(str(audio_dir / "bed.wav"), stereo_bed, sr)

    # Create dummy segment
    synth_dir = tmp_path / "synth"
    synth_dir.mkdir(parents=True)
    mono_speech = np.ones(int(0.5 * sr), dtype=np.float32) * 0.3
    sf.write(str(synth_dir / "seg1.wav"), mono_speech, sr)

    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True)
    (seg_dir / "segments.json").write_text(
        json.dumps([
            {
                "segment_id": "seg1",
                "start": 0.5,
                "end": 1.0,
                "generated_wav": "synth/seg1.wav",
            }
        ]),
        encoding="utf-8",
    )

    final_path = run_mixing(tmp_path, duration_s=duration_s)
    assert final_path.exists()

    final_audio, final_sr = sf.read(str(final_path))
    assert final_sr == sr
    # Output should preserve stereo 2 channels!
    assert final_audio.ndim == 2
    assert final_audio.shape[1] == 2


def test_duck_envelope_hold_margin():
    sr = 48000
    n = int(3.0 * sr)
    # Dialogue from 1.0s to 1.5s with 120ms hold margin
    intervals = [(1.0, 1.5)]
    env = _build_duck_envelope(intervals, n, sr=sr, duck_gain=0.22, hold_s=0.12, release_s=0.40)

    # During dialogue
    assert round(float(env[int(1.2 * sr)]), 2) == 0.22
    # During the 120ms hold margin immediately after dialogue (at 1.56s)
    assert round(float(env[int(1.56 * sr)]), 2) == 0.22
    # Well after release (at 2.5s), should be fully restored to 1.0
    assert round(float(env[int(2.5 * sr)]), 2) == 1.0


def test_dialogue_anti_collision_clamping(tmp_path: Path):
    """Verify that segment 0 is clamped before segment 1 starts, preventing collision."""
    sr = 48000
    duration_s = 5.0
    synth_dir = tmp_path / "synth"
    synth_dir.mkdir(parents=True)

    # Segment 0 starts at 0.0s. Its audio is 3.0s long!
    # But Segment 1 starts at 2.0s. Without clamping, they would collide for 1.0s.
    seg0_audio = np.ones(int(3.0 * sr), dtype=np.float32) * 0.4
    seg1_audio = np.ones(int(1.0 * sr), dtype=np.float32) * 0.4
    sf.write(str(synth_dir / "seg0.wav"), seg0_audio, sr)
    sf.write(str(synth_dir / "seg1.wav"), seg1_audio, sr)

    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True)
    (seg_dir / "segments.json").write_text(
        json.dumps([
            {"segment_id": "seg0", "start": 0.0, "end": 2.0, "generated_wav": "synth/seg0.wav"},
            {"segment_id": "seg1", "start": 2.0, "end": 3.0, "generated_wav": "synth/seg1.wav"},
        ]),
        encoding="utf-8",
    )

    final_path = run_mixing(tmp_path, duration_s=duration_s)
    assert final_path.exists()

    diag_path = tmp_path / "mix" / "dialogue.wav"
    diag, _ = sf.read(str(diag_path))

    # At 1.99s (right before seg1 starts at 2.0s), seg0 must be fading/clamped
    # There should be no additive constructive interference spike (e.g. 0.4 + 0.4 = 0.8)
    assert np.max(np.abs(diag)) <= 0.85
    # The dialogue between 1.98s and 2.00s should be at a safe transition level
    val_at_boundary = np.abs(diag[int(1.99 * sr)])
    assert val_at_boundary < 0.50


def test_dynamic_emotion_and_loudness_matching(tmp_path: Path):
    """Verify that synthesized dialogue dynamically scales to match the original speaker's emotional volume."""
    sr = 48000
    duration_s = 4.0
    n = int(duration_s * sr)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    t_full = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    carrier = np.sin(2 * np.pi * 220 * t_full).astype(np.float32)

    # Original vocals: Segment 0 (0-1s) is loud (0.50), Segment 1 (2-3s) is quiet whisper (0.10)
    voc = np.zeros(n, dtype=np.float32)
    voc[int(0.0 * sr) : int(1.0 * sr)] = carrier[int(0.0 * sr) : int(1.0 * sr)] * 0.50
    voc[int(2.0 * sr) : int(3.0 * sr)] = carrier[int(2.0 * sr) : int(3.0 * sr)] * 0.10
    sf.write(str(audio_dir / "vocals.wav"), voc, sr)

    synth_dir = tmp_path / "synth"
    synth_dir.mkdir(parents=True)
    # Synthesized speech is flat at 0.25 for both segments
    t_seg = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False, dtype=np.float32)
    synth_audio = (np.sin(2 * np.pi * 220 * t_seg) * 0.25).astype(np.float32)
    sf.write(str(synth_dir / "seg0.wav"), synth_audio, sr)
    sf.write(str(synth_dir / "seg1.wav"), synth_audio, sr)

    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True)
    (seg_dir / "segments.json").write_text(
        json.dumps([
            {"segment_id": "seg0", "start": 0.0, "end": 1.0, "generated_wav": "synth/seg0.wav"},
            {"segment_id": "seg1", "start": 2.0, "end": 3.0, "generated_wav": "synth/seg1.wav"},
        ]),
        encoding="utf-8",
    )

    run_mixing(tmp_path, duration_s=duration_s)
    diag, _ = sf.read(str(tmp_path / "mix" / "dialogue.wav"))

    rms_seg0 = float(np.sqrt(np.mean(diag[int(0.2 * sr) : int(0.8 * sr)] ** 2)))
    rms_seg1 = float(np.sqrt(np.mean(diag[int(2.2 * sr) : int(2.8 * sr)] ** 2)))

    # Segment 0 must be significantly louder than Segment 1, reflecting the original vocal emotion!
    assert rms_seg0 > rms_seg1 * 1.8


def test_non_speech_sfx_preservation(tmp_path: Path):
    """Verify that non-speech sounds (SFX, laughter, chimes) during pauses are preserved in the final mix."""
    sr = 48000
    duration_s = 3.0
    n = int(duration_s * sr)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    # Original vocals has a distinct SFX/chime at 0.2s - 0.5s
    voc = np.zeros(n, dtype=np.float32)
    voc[int(0.2 * sr) : int(0.5 * sr)] = 0.45
    sf.write(str(audio_dir / "vocals.wav"), voc, sr)

    synth_dir = tmp_path / "synth"
    synth_dir.mkdir(parents=True)
    # Dialogue only begins at 1.5s
    dia_audio = np.ones(int(1.0 * sr), dtype=np.float32) * 0.30
    sf.write(str(synth_dir / "seg0.wav"), dia_audio, sr)

    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True)
    (seg_dir / "segments.json").write_text(
        json.dumps([
            {"segment_id": "seg0", "start": 1.5, "end": 2.5, "generated_wav": "synth/seg0.wav"},
        ]),
        encoding="utf-8",
    )

    final_path = run_mixing(tmp_path, duration_s=duration_s)
    mix, _ = sf.read(str(final_path))

    # At 0.3s (during the original SFX, when dialogue is 0), audio must be present!
    sfx_energy = float(np.sqrt(np.mean(mix[int(0.25 * sr) : int(0.45 * sr)] ** 2)))
    assert sfx_energy > 0.20
