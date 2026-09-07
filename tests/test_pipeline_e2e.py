from __future__ import annotations

import json
import shutil
import uuid

import dubstudio.jobs.store as store_mod
import dubstudio.pipeline.orchestrator as orch
from dubstudio.settings import settings
from dubstudio.util.ffmpeg import probe


def test_full_pipeline_produces_mp4_and_srt(clip_8s, job_workspace, monkeypatch):
    monkeypatch.setattr(settings, "tts_engine", "dummy")
    monkeypatch.setattr(settings, "translator", "demo")
    monkeypatch.setattr(settings, "skip_separation", True)
    monkeypatch.setattr(settings, "asr_engine", "mock")
    monkeypatch.setattr(settings, "enable_diarization", False)
    job_id = f"job_test_e2e_{uuid.uuid4().hex[:10]}"
    job_dir = settings.jobs_dir / job_id
    src_dir = job_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clip_8s, src_dir / "clip.mp4")
    job = {
        "job_id": job_id, "state": "queued", "source_filename": "clip.mp4",
        "source_language": "en", "target_language": "hi", "skip_separation": True,
        "percent": 1, "message": "Queued", "error": None, "speakers": [], "artifacts": {},
    }
    store_mod.store.create(job)
    result = orch.run_job(job_id)
    assert result["state"] == "awaiting_voice_selection"
    assert result["percent"] == 67

    # Resume synthesis phase (as user would via UI apply button)
    result = orch.run_job_synthesis(job_id)
    assert result["state"] == "completed"
    assert result["percent"] == 100
    out_mp4 = job_dir / result["artifacts"]["output_mp4"]
    out_srt = job_dir / result["artifacts"]["output_srt"]
    assert out_mp4.exists() and out_mp4.stat().st_size > 1000
    assert out_srt.exists() and "-->" in out_srt.read_text(encoding="utf-8")
    segs = json.loads((job_dir / "segments" / "segments.json").read_text(encoding="utf-8"))
    assert len(segs) >= 1
    assert all(s.get("translated_text") for s in segs)
    assert all(s.get("status") == "ok" for s in segs)
    assert float(probe(out_mp4)["format"]["duration"]) >= 5.0
