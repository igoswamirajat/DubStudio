"""Mastering a job on disk: loudness, a reversible original, and a verdict."""
from __future__ import annotations

import json

import numpy as np

from dubstudio.pipeline import loudness as L
from dubstudio.pipeline import mastering as M
from dubstudio.util.audio import read_audio, write_wav

SR = 48000


def _tone(seconds, freq=1000.0, rms=0.06):
    t = np.arange(int(SR * seconds)) / float(SR)
    return (rms * np.sqrt(2.0) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _job(tmp_path, rms=0.06):
    job = tmp_path / "job"
    (job / "mix").mkdir(parents=True)
    write_wav(job / "mix" / "final.wav", _tone(3.0, rms=rms), SR)
    write_wav(job / "mix" / "dialogue.wav", _tone(3.0, rms=0.05), SR)
    (job / "mix" / "qc.json").write_text(json.dumps({
        "coverage": 1.0, "holes_over_1s": 0, "largest_hole_s": 0.0,
        "failed_segments": [], "segments_total": 2, "segments_mixed": 2,
        "duck_depth_db": -9.5,
    }), encoding="utf-8")
    return job


def test_mastering_brings_a_quiet_mix_to_target(tmp_path):
    job = _job(tmp_path)
    result = M.master_job(job)
    assert abs(result["master"]["output_lufs"] - L.MASTER_LUFS) < 1.0
    data, sr = read_audio(job / "mix" / "final.wav", target_sr=SR, mono=True)
    assert abs(L.lufs(data, sr) - L.MASTER_LUFS) < 1.0
    assert L.true_peak_dbtp(data, sr) <= L.MASTER_TRUE_PEAK_DBTP + 0.5


def test_mastering_keeps_the_untouched_render_and_never_compounds(tmp_path):
    job = _job(tmp_path)
    M.master_job(job)
    raw, _ = read_audio(job / "mix" / "final.raw.wav", target_sr=SR, mono=True)
    assert abs(L.dbfs_rms(raw) - (-24.44)) < 1.0
    second = M.master_job(job)
    assert abs(second["master"]["gain_db"]) < 0.5
    raw_again, _ = read_audio(job / "mix" / "final.raw.wav", target_sr=SR, mono=True)
    assert abs(L.dbfs_rms(raw_again) - L.dbfs_rms(raw)) < 0.01


def test_mastering_updates_qc_and_writes_the_gate(tmp_path):
    job = _job(tmp_path)
    result = M.master_job(job)
    qc = json.loads((job / "mix" / "qc.json").read_text(encoding="utf-8"))
    assert "master_lufs" in qc and "true_peak_dbtp" in qc and "master_gain_db" in qc
    gate = json.loads((job / "mix" / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == result["gate"]["status"]
    assert "loudness" not in gate["failures"]
    assert "true_peak" not in gate["failures"]


def test_mastering_a_job_with_no_mix_is_a_no_op(tmp_path):
    job = tmp_path / "empty"
    job.mkdir()
    assert M.master_job(job) == {}


# --- pipeline wiring --------------------------------------------------------
# master_job() existed but nothing in the pipeline called it, so every job
# shipped at whatever level mixing landed on. Job 8 came out at -18.4 LUFS and
# failed the gate on loudness alone.

def test_run_gate_can_be_skipped_for_the_pipeline(tmp_path):
    job = _job(tmp_path)
    result = M.master_job(job, run_gate=False)
    assert result["gate"] == {}
    assert not (job / "mix" / "gate.json").exists()
    # mastering itself still happened
    assert abs(result["master"]["output_lufs"] - L.MASTER_LUFS) < 1.0


def test_the_orchestrator_masters_the_mix(tmp_path, monkeypatch):
    from dubstudio.pipeline import orchestrator as orch

    job = _job(tmp_path)
    orch._master(job)
    data, sr = read_audio(job / "mix" / "final.wav", target_sr=SR, mono=True)
    assert abs(L.lufs(data, sr) - L.MASTER_LUFS) < 1.0
    assert (job / "mix" / "final.raw.wav").exists()


def test_a_mastering_failure_does_not_lose_the_render(tmp_path, monkeypatch):
    """Mastering is advisory: a crash must not throw away a good mix."""
    from dubstudio.pipeline import orchestrator as orch

    job = _job(tmp_path)
    before, _ = read_audio(job / "mix" / "final.wav", target_sr=SR, mono=True)

    def boom(*_a, **_k):
        raise RuntimeError("simulated mastering failure")

    monkeypatch.setattr(orch, "master_job", boom)
    assert orch._master(job) == {}
    after, _ = read_audio(job / "mix" / "final.wav", target_sr=SR, mono=True)
    assert np.array_equal(before, after)
