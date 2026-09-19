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


def test_a_breath_inside_a_long_pause_is_kept(tmp_path):
    """An invented pause usually holds the breath, not nothing.

    It is the loudest thing in the pause, and the old code kept the first
    `keep` ms of the run - which dropped the breath when it sat late, and left
    a clipped-off line instead of a breathed one.

    The breath sits below the -45 dBFS pause gate (that is what makes it a
    pause) but well above the -70 dBFS edge floor, so it is real signal the
    picker can find.
    """
    breath = _tone(0.10, freq=500.0, amp=10.0 ** (-50.0 / 20.0))
    # the breath is in the middle of a 2.5 s pause, well past a first-0.6 s keep
    src = _write(tmp_path / "in.wav", _tone(1.0), _sil(1.2), breath, _sil(1.2), _tone(1.0))
    info = A.fit_block(src, tmp_path / "out.wav", 3200)
    assert info["pauses_closed_ms"] > 0

    out, sr = _read(tmp_path / "out.wav")
    # the kept window is centred on the breath, so it lands between the two
    # tones; a breath that was dropped leaves the interior at digital zero
    interior = out[int(1.0 * sr):int(2.2 * sr)]
    peak_db = 20.0 * np.log10(max(float(np.max(np.abs(interior))), 1e-12))
    assert peak_db > -55.0, f"the breath was dropped (interior peak {peak_db:.1f} dBFS)"


def test_the_loudest_window_is_what_survives_a_trimmed_pause():
    """Unit test for the picker: pure silence plus one burst, off-centre."""
    x = np.concatenate([_sil(0.8), _tone(0.10, amp=0.2), _sil(0.8)])
    keep = int(0.6 * SR)
    a, b = A._loudest_window(x, keep)
    assert b - a == keep
    assert float(np.max(np.abs(x[a:b]))) > 0.1, "the burst must be inside the kept span"


def test_one_ratio_for_the_whole_block_and_it_is_bounded(tmp_path):
    src = _write(tmp_path / "in.wav", _tone(4.0))
    info = A.fit_block(src, tmp_path / "out.wav", 2000)
    assert info["wanted_ratio"] > 1.9
    assert info["stretch_ratio"] == 1.0 + A.MAX_BLOCK_STRETCH
    assert info["status"] == "overflow" and info["overflow_ms"] > 1000


def test_a_soft_tail_is_not_mistaken_for_silence(tmp_path):
    """-45 dBFS is not silence: a word's decay and a breath live below it.

    Trimming at the pause gate removed 160 ms of real -49 dBFS decay from the
    tail of every block in job 8, which is what made lines end abruptly - and
    the lost duration was exactly what left a hole at the end of the video.
    """
    soft_tail = _tone(0.30, amp=10.0 ** (-55.0 / 20.0))
    src = _write(tmp_path / "in.wav", _tone(1.0), soft_tail, _sil(0.30))
    info = A.fit_block(src, tmp_path / "out.wav", 1600)

    # Only the true silence goes: 0.30 s minus the keep-pad, not 0.30 + 0.30.
    assert info["trimmed_ms"] <= 320, info

    out, sr = _read(tmp_path / "out.wav")
    tail = out[-int(0.12 * sr):]
    level = 20.0 * np.log10(max(float(np.sqrt(np.mean(tail ** 2))), 1e-12))
    assert level > -70.0, f"the soft tail was trimmed away (tail at {level:.1f} dBFS)"


def test_an_underfill_is_closed_but_a_small_overrun_is_left_alone(tmp_path):
    """The dead band is asymmetric on purpose.

    A small overrun is harmless - the mixer trims it against the next unit - so
    the engine's own rhythm is worth more than the precision. A small underfill
    is dead air at the end of the line, so it gets closed.
    """
    short = _write(tmp_path / "short.wav", _tone(0.985))
    short_info = A.fit_block(short, tmp_path / "short_out.wav", 1000)
    assert short_info["stretch_ratio"] < 1.0
    assert abs(_ms(tmp_path / "short_out.wav") - 1000) <= 30

    long = _write(tmp_path / "long.wav", _tone(1.015))
    long_info = A.fit_block(long, tmp_path / "long_out.wav", 1000)
    assert long_info["stretch_ratio"] == 1.0


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


def test_the_timeline_solver_picks_one_rate_for_every_block(tmp_path):
    """Two blocks that would have been fitted at divergent ratios on their own
    now meet on a single global tempo."""
    segs = [_seg(0, 0.5, 2.4), _seg(1, 2.6, 4.4, role="member")]
    job_dir = _job(tmp_path, segs, [
        _block(1000, generated="synth/blk_0000.wav"),          # 0.94 s of content
        {
            "block_id": "blk_0001", "speaker_id": "S00",
            "segment_ids": [],
            "start": 6.0, "end": 8.0, "slot_start": 6.0, "slot_end": 8.0,
            "target_duration_ms": 2000, "text": "one two three",
            "parts": [], "generated_wav": "synth/blk_0001.wav", "status": "synthesized",
        },
    ])
    # 0.94 s into a 1.0 s slot (wanted 0.94) and 2.12 s into a 2.0 s slot (1.06)
    _write(job_dir / "synth" / "blk_0000.wav", _tone(0.94))
    _write(job_dir / "synth" / "blk_0001.wav", _tone(2.12))

    A.run_align(job_dir)
    saved = json.loads((job_dir / "segments" / "blocks.json").read_text(encoding="utf-8"))
    ratios = {b["block_id"]: b["stretch_ratio"] for b in saved}
    # fitted on their own these were 0.94 and 1.06, a 0.12 spread; the solver
    # eases both towards the single global rate
    assert max(ratios.values()) - min(ratios.values()) < 0.12, ratios
    sol = json.loads((job_dir / "timing" / "timeline.json").read_text(encoding="utf-8"))
    assert abs(sol["global_ratio"] - 1.0) < 0.03
    assert sol["collisions"] == []


def test_a_sub_dead_band_collision_is_not_snapped_back_to_one(tmp_path):
    """A compression inside the +-3% dead band is normally snapped to 1.0 - the
    engine's own rhythm is worth more than 2% of precision. But when the
    compression exists to keep a block inside its runway, snapping it back hands
    the overflow to the mixer's hard trim, which deletes words. The solver's
    decision has to win."""
    work = _tone(1.02)
    info = A._apply(work, SR, 1.015, tmp_path / "a.wav", 1000, force=True)
    assert info["stretch_ratio"] == 1.015
    assert abs(_ms(tmp_path / "a.wav") - 1005) <= 12

    # and without force the same ratio is snapped, which is the normal path
    info = A._apply(work, SR, 1.015, tmp_path / "b.wav", 1000)
    assert info["stretch_ratio"] == 1.0


def test_a_block_that_would_overflow_the_next_line_is_compressed_not_trimmed(tmp_path):
    """The mixer's hard trim deletes the last words of a block. The solver
    compresses it instead and reports the collision."""
    job_dir = tmp_path / "job"
    (job_dir / "segments").mkdir(parents=True, exist_ok=True)
    segs = [_seg(0, 0.0, 0.5), _seg(1, 3.0, 3.5)]
    (job_dir / "segments" / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
    blocks = [
        {
            "block_id": "blk_0000", "speaker_id": "S00",
            "segment_ids": ["seg_0000"], "start": 0.0, "end": 0.5,
            "slot_start": 0.0, "slot_end": 0.5, "target_duration_ms": 500,
            "text": "one two three", "parts": [],
            "generated_wav": "synth/blk_0000.wav", "status": "synthesized",
        },
        {
            "block_id": "blk_0001", "speaker_id": "S00",
            "segment_ids": ["seg_0001"], "start": 3.0, "end": 3.5,
            "slot_start": 3.0, "slot_end": 3.5, "target_duration_ms": 500,
            "text": "four five six", "parts": [],
            "generated_wav": "synth/blk_0001.wav", "status": "synthesized",
        },
    ]
    (job_dir / "segments" / "blocks.json").write_text(json.dumps(blocks), encoding="utf-8")
    # 4.0 s of content in a 0.5 s slot, with the next line starting at 3.0 s
    _write(job_dir / "synth" / "blk_0000.wav", _tone(4.0))
    _write(job_dir / "synth" / "blk_0001.wav", _tone(0.4))

    A.run_align(job_dir)
    fitted = _read(job_dir / "timing" / "blk_0000.fitted.wav")[0]
    # compressed to fit the 3.0 s runway, not trimmed to the 0.5 s slot
    assert len(fitted) <= 3.05 * SR, len(fitted) / SR
    assert len(fitted) > 1.5 * SR, "the content was cut instead of compressed"
    sol = json.loads((job_dir / "timing" / "timeline.json").read_text(encoding="utf-8"))
    assert "blk_0000" in sol["collisions"]
