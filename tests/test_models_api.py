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


def test_set_storage_directory(tmp_path: Path, monkeypatch):
    # The suite-wide fixture already redirects the persisted .env; assert that
    # the redirect is honoured rather than writing the real project .env.
    env_file = tmp_path / ".env"
    env_file.write_text("DUBSTUDIO_TTS_ENGINE=omnivoice\n", encoding="utf-8")
    monkeypatch.setenv("DUBSTUDIO_ENV_FILE", str(env_file))

    client = TestClient(create_app())
    custom_dir = tmp_path / "custom_models"
    res = client.post("/api/v1/models/storage", json={"storage_path": str(custom_dir)})
    assert res.status_code == 200
    data = res.json()
    assert Path(data["storage_path"]).resolve() == custom_dir.resolve()
    assert custom_dir.exists()

    written = env_file.read_text(encoding="utf-8")
    assert f"DUBSTUDIO_HF_HOME={custom_dir.resolve()}" in written
    # The unrelated setting must survive the rewrite.
    assert "DUBSTUDIO_TTS_ENGINE=omnivoice" in written


def test_set_storage_directory_never_touches_project_env(tmp_path: Path):
    """Regression: a test run must not rewrite the project's real .env.

    The autouse `_isolate_project_env` fixture is what makes this safe; before
    it existed this exact call left DUBSTUDIO_HF_HOME pointing at a pytest tmp
    directory inside the developer's .env.
    """
    project_env = Path(__file__).resolve().parents[1] / ".env"
    before = project_env.read_text(encoding="utf-8") if project_env.exists() else None

    client = TestClient(create_app())
    res = client.post("/api/v1/models/storage", json={"storage_path": str(tmp_path / "m")})
    assert res.status_code == 200

    after = project_env.read_text(encoding="utf-8") if project_env.exists() else None
    assert after == before, "the storage endpoint rewrote the project's .env"


def test_env_file_path_requires_project_root(tmp_path: Path, monkeypatch):
    """Without a project root there is no .env to write, and none is invented."""
    from dubstudio.api import routes_models

    monkeypatch.delenv("DUBSTUDIO_ENV_FILE", raising=False)
    monkeypatch.setattr(routes_models, "__file__", str(tmp_path / "dubstudio" / "api" / "routes_models.py"))
    assert routes_models._env_file_path() is None

    monkeypatch.setenv("DUBSTUDIO_ENV_FILE", str(tmp_path / "custom.env"))
    assert routes_models._env_file_path() == tmp_path / "custom.env"


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
