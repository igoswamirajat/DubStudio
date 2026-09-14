"""Block-level alignment: trim the engine's dead air, close the pauses it
invented, then fit the whole run with ONE tempo ratio."""
from __future__ import annotations

import json

import numpy as np

import dubstudio.pipeline.align as A
from dubstudio.util.audio import read_audio, write_wav

SR = 48000


def _tone(dur_s, freq=220.0, amp=0.3):
    t = np.arange(int(round(dur_s * SR)), dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _sil(dur_s):
    return np.zeros(int(round(dur_s * SR)), dtype=np.float32)


def _write(path, *pieces):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, np.concatenate(pieces).astype(np.float32), SR)
    return path


def _read(path):
    data, sr = read_audio(path, target_sr=48000, mono=True)
    return np.asarray(data, dtype=np.float32).reshape(-1), sr


def _ms(path):
    data, sr = _read(path)
    return int(round(data.size / sr * 1000))


def test_the_engines_lead_in_and_tail_are_trimmed(tmp_path):
    src = _write(tmp_path / "in.wav", _sil(0.50), _tone(1.0), _sil(0.60))
    info = A.fit_block(src, tmp_path / "out.wav", 1000)
    assert info["generated_duration_ms"] == 2100
    assert info["trimmed_ms"] >= 900          # ~0.46s head + ~0.56s tail
    assert abs(_ms(tmp_path / "out.wav") - 1000) <= 80


def test_a_pause_the_engine_invented_is_shortened(tmp_path):
    src = _write(tmp_path / "in.wav", _tone(1.0), _sil(1.5), _tone(1.0))
    info = A.fit_block(src, tmp_path / "out.wav", 3500)
    assert 850 <= info["pauses_closed_ms"] <= 950
    out, sr = _read(tmp_path / "out.wav")
    longest = max((b - a) / sr for a, b in A._silence_runs(out, sr, 10 ** (-45.0 / 20.0)))
    assert longest <= 0.75     # 0.60s cap, plus the tempo fit


def test_a_natural_pause_is_left_alone(tmp_path):
    src = _write(tmp_path / "in.wav", _tone(1.0), _sil(0.4), _tone(1.0))
    info = A.fit_block(src, tmp_path / "out.wav", 2400)
    assert info["pauses_closed_ms"] == 0


def test_one_ratio_for_the_whole_block_and_it_is_bounded(tmp_path):
    src = _write(tmp_path / "in.wav", _tone(4.0))
    info = A.fit_block(src, tmp_path / "out.wav", 2000)
    assert info["wanted_ratio"] > 1.9
    assert info["stretch_ratio"] == 1.0 + A.MAX_BLOCK_STRETCH
    assert info["status"] == "overflow" and info["overflow_ms"] > 1000


def test_a_small_mismatch_keeps_the_engines_own_rhythm(tmp_path):
    src = _write(tmp_path / "in.wav", _tone(2.04))
    info = A.fit_block(src, tmp_path / "out.wav", 2000)
    assert info["stretch_ratio"] == 1.0
    assert info["status"] == "ok"


def test_a_block_that_barely_overruns_is_squeezed_in(tmp_path):
    src = _write(tmp_path / "in.wav", _tone(2.12))
    info = A.fit_block(src, tmp_path / "out.wav", 2000)
    assert 1.05 <= info["stretch_ratio"] <= 1.07
    assert info["overflow_ms"] <= 120


# --- run_align ------------------------------------------------------------

def _seg(i, start, end, block="blk_0000", role="lead", **extra):
    seg = {
        "segment_id": f"seg_{i:04d}",
        "speaker_id": "S00",
        "start": start,
        "end": end,
        "slot_start": start,
        "slot_end": end,
        "translated_text": "one two three",
        "target_duration_ms": int(round((end - start) * 1000)),
        "block_id": block,
        "block_index": 0 if role != "member" else 1,
        "block_size": 2,
        "block_role": role,
        "generated_wav": None,
        "fitted_wav": None,
        "status": "synthesized",
    }
    seg.update(extra)
    return seg


def _job(tmp_path, segs, blocks):
    job_dir = tmp_path / "job"
    (job_dir / "segments").mkdir(parents=True, exist_ok=True)
    (job_dir / "segments" / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
    (job_dir / "segments" / "blocks.json").write_text(json.dumps(blocks), encoding="utf-8")
    return job_dir


def _block(target_ms, generated="synth/blk_0000.wav", ids=("seg_0000", "seg_0001")):
    return {
        "block_id": "blk_0000",
        "speaker_id": "S00",
        "segment_ids": list(ids),
        "start": 0.5,
        "end": 4.4,
        "slot_start": 0.5,
        "slot_end": 0.5 + target_ms / 1000.0,
        "target_duration_ms": target_ms,
        "text": "one two three four",
        "parts": [],
        "generated_wav": generated,
        "status": "synthesized",
    }


def test_the_block_audio_goes_on_its_lead_cue(tmp_path):
    segs = [_seg(0, 0.5, 2.4), _seg(1, 2.6, 4.4, role="member")]
    job_dir = _job(tmp_path, segs, [_block(3900)])
    _write(job_dir / "synth" / "blk_0000.wav", _tone(3.9))

    A.run_align(job_dir)
    saved = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    blocks = json.loads((job_dir / "segments" / "blocks.json").read_text(encoding="utf-8"))

    assert saved[0]["fitted_wav"] == "timing/blk_0000.fitted.wav"
    assert saved[0]["status"] == "ok"
    assert saved[1]["fitted_wav"] is None
    assert saved[1]["status"] == "in_block"
    assert saved[1]["block_wav"] == "timing/blk_0000.fitted.wav"
    assert saved[0]["stretch_ratio"] == saved[1]["stretch_ratio"]
    assert blocks[0]["status"] == "fitted"
    assert (job_dir / "timing" / "blk_0000.fitted.wav").exists()
    assert (job_dir / "timing" / "blk_0000.json").exists()


def test_a_block_with_no_audio_fails_all_of_its_cues(tmp_path):
    segs = [_seg(0, 0.5, 2.4), _seg(1, 2.6, 4.4, role="member")]
    job_dir = _job(tmp_path, segs, [_block(3900, generated=None)])

    A.run_align(job_dir)
    saved = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in saved] == ["failed", "failed"]
    assert all(s["fitted_wav"] is None for s in saved)


def test_a_cue_outside_any_block_is_still_fitted_on_its_own(tmp_path):
    segs = [
        _seg(0, 0.5, 2.4),
        _seg(1, 2.6, 4.4, role="member"),
        _seg(2, 6.0, 7.0, block=None, role="solo", generated_wav="synth/seg_0002.wav"),
    ]
    job_dir = _job(tmp_path, segs, [_block(3900)])
    _write(job_dir / "synth" / "blk_0000.wav", _tone(3.9))
    _write(job_dir / "synth" / "seg_0002.wav", _tone(1.0))

    A.run_align(job_dir)
    saved = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    assert saved[2]["fitted_wav"] == "timing/seg_0002.fitted.wav"
    assert (job_dir / "timing" / "seg_0002.fitted.wav").exists()


def test_without_blocks_it_falls_back_to_per_cue_timing(tmp_path):
    segs = [_seg(0, 0.5, 2.4, block=None, role="solo", generated_wav="synth/seg_0000.wav")]
    job_dir = tmp_path / "job"
    (job_dir / "segments").mkdir(parents=True)
    (job_dir / "segments" / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
    _write(job_dir / "synth" / "seg_0000.wav", _tone(1.9))

    assert A.run_align(job_dir) == []
    saved = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    assert saved[0]["fitted_wav"] == "timing/seg_0000.fitted.wav"
