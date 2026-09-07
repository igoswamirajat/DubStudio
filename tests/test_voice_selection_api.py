from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

import uuid
import dubstudio.api.routes_jobs as rj
import dubstudio.api.routes_voice_selection as r_vs
import dubstudio.jobs.store as store_mod
from dubstudio.jobs.store import JobStore
from dubstudio.main import create_app
from dubstudio.settings import settings


@pytest.fixture
def test_setup(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(settings, "data_dir", data)
    test_store = JobStore(db_path=data / "test.db")
    monkeypatch.setattr(store_mod, "store", test_store, raising=False)
    monkeypatch.setattr(rj, "store", test_store, raising=False)
    monkeypatch.setattr(r_vs, "store", test_store, raising=False)
    app = create_app()
    return TestClient(app), test_store


def test_voice_options_api_flow(test_setup, tmp_path, monkeypatch):
    client, test_store = test_setup
    job_id = f"job_test_voice_{uuid.uuid4().hex[:8]}"
    job_dir = settings.jobs_dir / job_id
    voices_dir = job_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    segs_dir = job_dir / "segments"
    segs_dir.mkdir(parents=True, exist_ok=True)

    # Mock speaker_map.json
    speaker_map = {
        "speakers": [
            {
                "speaker_id": "S00",
                "label": "Speaker 00",
                "voice_id": "agastya",
                "voice_mode": "native",
                "gender": "male",
                "f0_median": 120.0,
                "auto_confidence": 0.85,
                "auto_suggestion": {"voice_id": "agastya", "confidence": 0.85},
                "ref_wav": "voices/S00/ref.wav",
                "segment_count": 3,
            }
        ]
    }
    (voices_dir / "speaker_map.json").write_text(json.dumps(speaker_map), encoding="utf-8")

    # Mock segments.json
    segments = [
        {"segment_id": "seg_001", "speaker_id": "S00", "text": "test", "voice_id": "dummy"}
    ]
    (segs_dir / "segments.json").write_text(json.dumps(segments), encoding="utf-8")

    # Create job in store with state awaiting_voice_selection
    job = {
        "job_id": job_id,
        "state": "awaiting_voice_selection",
        "tts_engine": "veena",
        "percent": 67,
    }
    test_store.create(job)

    # 1. GET /voice-options
    res = client.get(f"/api/v1/jobs/{job_id}/voice-options")
    assert res.status_code == 200
    data = res.json()
    assert data["job_state"] == "awaiting_voice_selection"
    assert len(data["speakers"]) == 1
    assert data["speakers"][0]["speaker_id"] == "S00"
    assert any(v["voice_id"] == "clone_original" for v in data["available_voices"])
    assert any(v["voice_id"] == "agastya" for v in data["available_voices"])
    assert any(v["voice_id"] == "vinaya" for v in data["available_voices"])
    assert any(v["voice_id"] == "kavya" for v in data["available_voices"])
    assert any(v["voice_id"] == "maitri" for v in data["available_voices"])

    # 2. POST /voice-options/apply with custom selection (e.g. Vinaya)
    apply_payload = {
        "selections": {
            "S00": {
                "voice_id": "vinaya",
                "voice_mode": "native",
            }
        }
    }
    # Mock resume_synthesis to avoid running background GPU thread in test
    import dubstudio.api.routes_voice_selection as r_vs

    async def mock_resume(jid):
        pass

    monkeypatch.setattr(r_vs, "resume_synthesis", mock_resume)

    res_apply = client.post(f"/api/v1/jobs/{job_id}/voice-options/apply", json=apply_payload)
    assert res_apply.status_code == 200
    assert res_apply.json()["ok"] is True

    # Check that speaker_map.json got updated
    updated_map = json.loads((voices_dir / "speaker_map.json").read_text(encoding="utf-8"))
    assert updated_map["speakers"][0]["voice_id"] == "vinaya"
