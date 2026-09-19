from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dubstudio.settings import settings

FIXTURES = Path(__file__).parent / "fixtures"

PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"


@pytest.fixture(scope="session")
def project_env_target(tmp_path_factory) -> Path:
    """One redirected .env for the whole run.

    Session-scoped on purpose: a function-scoped ``mktemp`` here would create a
    fresh directory for every single test, which is ~220 throwaway dirs per run.
    """
    return tmp_path_factory.mktemp("env") / ".env"


@pytest.fixture(autouse=True)
def _isolate_project_env(project_env_target, monkeypatch):
    """Stop any test from writing the developer's real project .env.

    The models API persists the chosen HF cache into .env so it survives a
    restart. Pointed at the CWD, that meant every test run rewrote the real
    .env with a pytest tmp path - which then silently won over HF_HOME at the
    next app start and sent model lookups to a deleted directory.
    """
    monkeypatch.setenv("DUBSTUDIO_ENV_FILE", str(project_env_target))
    return project_env_target


@pytest.fixture(scope="session")
def project_env_path() -> Path:
    return PROJECT_ENV


@pytest.fixture(scope="session")
def clip_8s(tmp_path_factory) -> Path:
    out = FIXTURES / "clip_8s.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return out
    FIXTURES.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=8",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


@pytest.fixture
def job_workspace(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(settings, "data_dir", data)
    from dubstudio.jobs import store as store_mod
    import dubstudio.pipeline.orchestrator as orch
    import dubstudio.jobs.runner as runner
    store_mod.store = store_mod.JobStore(db_path=data / "test.db")
    orch.store = store_mod.store
    runner.store = store_mod.store
    return data
