from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dubstudio.settings import settings

FIXTURES = Path(__file__).parent / "fixtures"


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
