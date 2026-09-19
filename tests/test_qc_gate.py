"""The ship / no-ship gate.

mix/qc.json has existed for a while and nothing read it, so a job that lost a
third of its speech still finished as "completed". These tests pin the verdict
for each failure mode we have actually seen in a real dub.
"""
from __future__ import annotations

import json

import numpy as np

from dubstudio.pipeline import qc as Q
from dubstudio.util.audio import write_wav

SR = 48000


def _qc(**over):
    base = {
        "segments_total": 3,
        "segments_mixed": 3,
        "failed_segments": [],
        "coverage": 1.0,
        "largest_hole_s": 0.0,
        "holes_over_1s": 0,
        "holes": [],
        "duck_depth_db": -9.9,
        "master_lufs": -16.0,
        "true_peak_dbtp": -1.4,
        "dialogue_top_hz": 15000.0,
        "bed_top_hz": 16000.0,
    }
    base.update(over)
    return base


def _status(gate, cid):
    return next(c["status"] for c in gate["checks"] if c["id"] == cid)


def _tone(seconds, freq=1000.0, rms=0.158):
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    return (rms * np.sqrt(2.0) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- verdicts ---------------------------------------------------------------
def test_a_clean_mix_passes():
    gate = Q.evaluate(_qc())
    assert gate["status"] == "pass"
    assert gate["failures"] == []
    assert gate["warnings"] == []


def test_missing_speech_fails_the_gate():
    gate = Q.evaluate(_qc(coverage=0.78, uncovered_speech_s=10.6))
    assert gate["status"] == "fail"
    assert "coverage" in gate["failures"]


def test_a_dropout_over_a_second_fails_the_gate():
    gate = Q.evaluate(_qc(holes_over_1s=4, largest_hole_s=3.29))
    assert gate["status"] == "fail"
    assert "holes_over_1s" in gate["failures"]
    assert "largest_hole" in gate["failures"]


def test_a_short_gap_only_warns():
    gate = Q.evaluate(_qc(largest_hole_s=0.44))
    assert gate["status"] == "warn"
    assert _status(gate, "largest_hole") == "warn"
    assert gate["failures"] == []


def test_an_undubbed_line_fails_the_gate():
    gate = Q.evaluate(_qc(failed_segments=["seg_0007"]))
    assert gate["status"] == "fail"
    assert "failed_segments" in gate["failures"]


def test_nothing_mixed_fails_the_gate():
    gate = Q.evaluate(_qc(segments_mixed=0))
    assert gate["status"] == "fail"
    assert "units_mixed" in gate["failures"]


def test_the_wrong_loudness_fails_the_gate():
    assert Q.evaluate(_qc(master_lufs=-23.4))["failures"] == ["loudness"]
    assert Q.evaluate(_qc(master_lufs=-16.9))["status"] == "pass"


def test_clipping_between_samples_fails_the_gate():
    assert "true_peak" in Q.evaluate(_qc(true_peak_dbtp=-0.2))["failures"]
    assert Q.evaluate(_qc(true_peak_dbtp=-0.95))["status"] == "pass"


def test_a_band_limited_voice_warns_but_ships():
    gate = Q.evaluate(_qc(dialogue_top_hz=11000.0, bed_top_hz=16000.0))
    assert gate["status"] == "warn"
    assert "dialogue_bandwidth" in gate["warnings"]
    assert "bandwidth_match" in gate["warnings"]
    assert gate["failures"] == []


def test_a_bed_brighter_than_the_dub_warns():
    gate = Q.evaluate(_qc(dialogue_top_hz=12500.0, bed_top_hz=16000.0))
    assert _status(gate, "dialogue_bandwidth") == "pass"
    assert _status(gate, "bandwidth_match") == "warn"


def test_a_shallow_duck_warns():
    gate = Q.evaluate(_qc(duck_depth_db=-2.16))
    assert _status(gate, "duck_depth") == "warn"


def test_an_unmeasurable_job_is_not_a_passing_job():
    gate = Q.evaluate({})
    assert gate["status"] == "skip"
    assert gate["failures"] == []
    assert len(gate["skipped"]) == len(gate["checks"])


def test_measured_audio_overrides_a_stale_report():
    gate = Q.evaluate(_qc(master_lufs=-30.0), {"master_lufs": -16.1})
    assert _status(gate, "loudness") == "pass"


# --- end to end on real files ----------------------------------------------
def test_run_qc_measures_the_rendered_files_and_writes_a_gate(tmp_path):
    job = tmp_path / "job"
    (job / "mix").mkdir(parents=True)
    (job / "audio").mkdir(parents=True)
    write_wav(job / "mix" / "final.wav", _tone(3.0), SR)
    write_wav(job / "mix" / "dialogue.wav", _tone(3.0, rms=0.09), SR)
    rng = np.random.default_rng(3)
    write_wav(job / "audio" / "bed.wav", (rng.standard_normal(SR * 3) * 0.05).astype(np.float32), SR)
    (job / "mix" / "qc.json").write_text(json.dumps(_qc(master_lufs=-40.0)), encoding="utf-8")

    gate = Q.run_qc(job)
    written = json.loads((job / "mix" / "gate.json").read_text(encoding="utf-8"))
    assert written["status"] == gate["status"]
    # the stale -40 LUFS in qc.json must lose to the real file
    assert abs(gate["measured"]["master_lufs"] - (-16.0)) < 1.5
    assert _status(gate, "loudness") == "pass"
    assert gate["measured"]["bed_top_hz"] > 15000.0
    assert gate["measured"]["dialogue_top_hz"] > 0.0


def test_run_qc_survives_a_job_with_nothing_in_it(tmp_path):
    job = tmp_path / "job"
    job.mkdir(parents=True)
    gate = Q.run_qc(job)
    assert gate["status"] == "skip"
    assert (job / "mix" / "gate.json").exists()


# --- non-speech passthrough (null test) -------------------------------------
# The point of the hybrid bed is that the music/SFX outside the speech are the
# untouched original. This is how that claim gets checked instead of assumed.

SPEECH = (0.5, 1.5)


def _null_job(tmp_path, *, passthrough: bool):
    job = tmp_path / "job"
    (job / "mix").mkdir(parents=True)
    (job / "audio").mkdir(parents=True)
    (job / "segments").mkdir(parents=True)

    rng = np.random.default_rng(11)
    n = SR * 3
    full = (rng.standard_normal(n) * 0.1).astype(np.float32)
    write_wav(job / "audio" / "full.wav", full, SR)

    if passthrough:
        final = full.copy()
    else:
        # a bed that went through a separator: same length, different content
        final = (rng.standard_normal(n) * 0.1).astype(np.float32)
    write_wav(job / "mix" / "final.wav", final, SR)

    segments = [{"segment_id": "s0", "start": SPEECH[0], "end": SPEECH[1]}]
    (job / "segments" / "segments.json").write_text(json.dumps(segments), encoding="utf-8")
    return job


def test_untouched_non_speech_passes_the_null_test(tmp_path):
    job = _null_job(tmp_path, passthrough=True)
    measured = Q.measure(job)
    assert measured["null_test_db"] <= Q.NULL_PASS_DB
    assert _status(Q.evaluate({}, measured), "null_test") == "pass"


def test_a_mangled_bed_fails_the_null_test(tmp_path):
    job = _null_job(tmp_path, passthrough=False)
    measured = Q.measure(job)
    assert measured["null_test_db"] > Q.NULL_WARN_DB
    assert _status(Q.evaluate({}, measured), "null_test") == "fail"


def test_null_test_is_skipped_without_clean_non_speech(tmp_path):
    job = _null_job(tmp_path, passthrough=True)
    # one speech window covering the whole file leaves nowhere to measure
    (job / "segments" / "segments.json").write_text(
        json.dumps([{"segment_id": "s0", "start": 0.0, "end": 3.0}]), encoding="utf-8")
    measured = Q.measure(job)
    assert "null_test_db" not in measured
    assert _status(Q.evaluate({}, measured), "null_test") == "skip"


def test_null_test_skips_without_the_original(tmp_path):
    job = _null_job(tmp_path, passthrough=True)
    (job / "audio" / "full.wav").unlink()
    assert "null_test_db" not in Q.measure(job)


# --- A/V sync ---------------------------------------------------------------
def test_a_late_audio_start_fails_the_av_offset_check():
    gate = Q.evaluate(_qc(), {"av_offset_ms": 180.0, "av_length_delta_ms": 0.0})
    assert "av_offset" in gate["failures"]


def test_a_frame_of_drift_passes():
    gate = Q.evaluate(_qc(), {"av_offset_ms": 12.0, "av_length_delta_ms": 20.0})
    assert _status(gate, "av_offset") == "pass"
    assert _status(gate, "av_length") == "pass"
    assert gate["failures"] == []


def test_a_truncated_mix_fails_the_av_length_check():
    # -shortest used to discard 0.58 s of the mix with no warning
    gate = Q.evaluate(_qc(), {"av_offset_ms": 0.0, "av_length_delta_ms": 580.0})
    assert "av_length" in gate["failures"]


def test_av_checks_skip_when_there_is_no_export(tmp_path):
    job = tmp_path / "job"
    job.mkdir(parents=True)
    measured = Q.measure(job)
    assert "av_offset_ms" not in measured
    gate = Q.evaluate(_qc(), measured)
    assert _status(gate, "av_offset") == "skip"
    assert _status(gate, "av_length") == "skip"
