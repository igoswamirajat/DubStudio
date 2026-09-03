"""HTTP-layer tests for the FastAPI app (endpoints, upload guard, CRUD)."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

import dubstudio.api.routes_jobs as rj
import dubstudio.api.routes_segments as rseg
import dubstudio.api.routes_speakers as rspk
import dubstudio.jobs.store as store_mod
from dubstudio.jobs.store import JobStore
from dubstudio.main import create_app
from dubstudio.settings import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(settings, "data_dir", data)
    test_store = JobStore(db_path=data / "http.db")
    # Route modules + main bind `store` by import — repoint them at the temp store.
    monkeypatch.setattr(store_mod, "store", test_store, raising=False)
    for mod in (rj, rseg, rspk):
        monkeypatch.setattr(mod, "store", test_store, raising=False)
    # Do not launch the real pipeline from HTTP tests.
    async def _noop_enqueue(job_id: str):
        return None

    monkeypatch.setattr(rj, "enqueue", _noop_enqueue)
    return TestClient(create_app())


def test_health_shape(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    for key in ("ok", "ffmpeg", "gpu", "asr", "tts", "translator"):
        assert key in body


def test_jobs_empty_and_404s(client):
    assert client.get("/api/v1/jobs").json() == {"jobs": []}
    assert client.get("/api/v1/jobs/nope").status_code == 404
    assert client.delete("/api/v1/jobs/nope").status_code == 404
    # segments of unknown job -> empty list (routes_jobs variant)
    assert client.get("/api/v1/jobs/nope/segments").json() == {"segments": []}


def test_create_list_delete_job(client):
    files = {"file": ("clip.mp4", io.BytesIO(b"\x00" * 2048), "video/mp4")}
    data = {"target_language": "hi", "tts_engine": "dummy", "skip_separation": "true"}
    r = client.post("/api/v1/jobs", files=files, data=data)
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]

    listed = client.get("/api/v1/jobs").json()["jobs"]
    assert any(j["job_id"] == job_id for j in listed)

    got = client.get(f"/api/v1/jobs/{job_id}")
    assert got.status_code == 200
    assert got.json()["target_language"] == "hi"

    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 404


def test_upload_too_large_returns_413(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 0)  # any non-empty upload exceeds 0 MB
    files = {"file": ("big.mp4", io.BytesIO(b"\x01" * (1024 * 1024 + 10)), "video/mp4")}
    r = client.post("/api/v1/jobs", files=files, data={"target_language": "hi"})
    assert r.status_code == 413


def test_settings_get_masks_secrets(client, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-secret")
    monkeypatch.setattr(settings, "hf_token", "hf-secret")
    body = client.get("/api/v1/settings").json()
    assert "openai_api_key" not in body
    assert "hf_token" not in body
    assert body["openai_api_key_set"] is True
    assert body["hf_token_set"] is True
    assert body["translator"] == settings.translator


def test_settings_patch_updates_and_persists(client, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # keep .env writes inside the temp dir
    monkeypatch.setattr(settings, "translator", "ollama")
    monkeypatch.setattr(settings, "openai_model", "gpt-4o-mini")
    r = client.patch(
        "/api/v1/settings",
        json={"translator": "openai", "openai_model": "llama-3.1-70b"},
    )
    assert r.status_code == 200
    assert settings.translator == "openai"
    assert settings.openai_model == "llama-3.1-70b"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "DUBSTUDIO_TRANSLATOR=openai" in env_text
    assert "DUBSTUDIO_OPENAI_MODEL=llama-3.1-70b" in env_text
