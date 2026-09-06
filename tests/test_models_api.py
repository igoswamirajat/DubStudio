from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from dubstudio.main import create_app
from dubstudio.settings import settings


def test_get_storage_info():
    client = TestClient(create_app())
    res = client.get("/api/v1/models/storage")
    assert res.status_code == 200
    data = res.json()
    assert "storage_path" in data
    assert "drive" in data
    assert "total_gb" in data
    assert "free_gb" in data
    assert data["total_gb"] > 0


def test_set_storage_directory(tmp_path: Path):
    client = TestClient(create_app())
    custom_dir = tmp_path / "custom_models"
    res = client.post("/api/v1/models/storage", json={"storage_path": str(custom_dir)})
    assert res.status_code == 200
    data = res.json()
    assert Path(data["storage_path"]).resolve() == custom_dir.resolve()
    assert custom_dir.exists()


def test_list_models_includes_veena():
    client = TestClient(create_app())
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    models = data.get("models", [])
    assert len(models) >= 4
    veena = next((m for m in models if m["id"] == "veena"), None)
    assert veena is not None
    assert veena["recommended_for"] == "hi"
    assert "maya-research/veena" in veena["repo_id"]
    assert "storage" in data


def test_download_model_validates_id():
    client = TestClient(create_app())
    res = client.post("/api/v1/models/download", json={"model_id": "non_existent_xyz"})
    assert res.status_code == 404
