from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dubstudio.settings import settings

log = logging.getLogger("dubstudio.models")
router = APIRouter()

DOWNLOAD_STATUS: dict[str, dict[str, Any]] = {}

MODELS_CATALOG = [
    {
        "id": "veena",
        "name": "Maya Research Veena",
        "category": "tts",
        "recommended_for": "hi",
        "badge": "SOTA Hindi TTS",
        "repo_id": "maya-research/veena",
        "extra_repos": ["hubertsiuzdak/snac_24khz"],
        "size_mb": 7200,
        "description": "3B transformer with 4 Indian character voices (Kavya, Agastya, Maitri, Vinaya) and SNAC 24kHz neural audio codec.",
    },
    {
        "id": "omnivoice",
        "name": "OmniVoice",
        "category": "tts",
        "recommended_for": "multilingual",
        "badge": "Voice Cloning",
        "repo_id": "k2-fsa/OmniVoice",
        "extra_repos": [],
        "size_mb": 2400,
        "description": "Multilingual neural speech synthesis and few-shot voice cloning from reference audio.",
    },
    {
        "id": "faster-whisper",
        "name": "Faster-Whisper (Large-v3)",
        "category": "asr",
        "recommended_for": "all",
        "badge": "Speech-to-Text",
        "repo_id": "Systran/faster-whisper-large-v3",
        "extra_repos": [],
        "size_mb": 3100,
        "description": "State-of-the-art CTranslate2 speech recognition with word timestamps.",
    },
    {
        "id": "demucs",
        "name": "Demucs (HTDemucs)",
        "category": "separation",
        "recommended_for": "all",
        "badge": "Stem Separation",
        "repo_id": "adefossez/HTDemucs",
        "extra_repos": [],
        "size_mb": 1200,
        "description": "Isolates original voice dialogue from background music and cinematic sound effects.",
    },
    {
        "id": "pyannote",
        "name": "Pyannote Speaker Diarization 3.1",
        "category": "diarization",
        "recommended_for": "multi-speaker",
        "badge": "Diarization",
        "repo_id": "pyannote/speaker-diarization-3.1",
        "extra_repos": [],
        "size_mb": 1500,
        "description": "Detects who spoke when and separates speakers for multi-character dubbing.",
    },
]


def _get_active_storage_dir() -> Path:
    """Return effective HuggingFace / models cache directory."""
    if settings.hf_home:
        return Path(settings.hf_home)
    # Check default Windows D:\hf_cache if it exists
    d_cache = Path("D:/hf_cache")
    if d_cache.is_dir():
        return d_cache
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def _disk_stats(dir_path: Path) -> dict[str, Any]:
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(dir_path))
        drive = dir_path.anchor or str(dir_path)[:3]
        total_gb = round(usage.total / (1024**3), 2)
        free_gb = round(usage.free / (1024**3), 2)
        used_gb = round(usage.used / (1024**3), 2)
        free_pct = round((usage.free / usage.total) * 100, 1) if usage.total else 0.0
        return {
            "storage_path": str(dir_path.resolve()),
            "drive": drive,
            "total_gb": total_gb,
            "free_gb": free_gb,
            "used_gb": used_gb,
            "free_percent": free_pct,
            "is_low_space": free_gb < 5.0,
        }
    except Exception as exc:
        return {
            "storage_path": str(dir_path),
            "drive": "?",
            "total_gb": 0.0,
            "free_gb": 0.0,
            "used_gb": 0.0,
            "free_percent": 0.0,
            "is_low_space": True,
            "error": str(exc),
        }


def _calc_unique_bytes(path: Path) -> int:
    """Calculate unique file bytes on disk, deduplicating hardlinks."""
    if not path.is_dir():
        return 0
    seen_inodes = set()
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                stat = f.stat()
                if stat.st_ino not in seen_inodes:
                    seen_inodes.add(stat.st_ino)
                    total += stat.st_size
            except Exception:
                pass
    return total


def _check_model_downloaded(storage_dir: Path, repo_id: str) -> tuple[bool, float]:
    """Check if model snapshot exists in storage_dir/hub/models--..."""
    hub_dir = storage_dir / "hub"
    folder_name = f"models--{repo_id.replace('/', '--')}"
    target = hub_dir / folder_name
    if not target.is_dir():
        return False, 0.0

    snapshots = target / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return False, 0.0

    # Calculate actual unique MB in directory (avoiding hardlink double-counting)
    total_bytes = _calc_unique_bytes(target)
    return True, round(total_bytes / (1024 * 1024), 1)


@router.get("/models/storage")
def get_storage():
    storage_dir = _get_active_storage_dir()
    return _disk_stats(storage_dir)


class UpdateStoragePayload(BaseModel):
    storage_path: str


@router.post("/models/storage")
def set_storage(payload: UpdateStoragePayload):
    new_path = Path(payload.storage_path.strip())
    try:
        new_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot create or access directory: {exc}")

    # Update runtime settings and environment
    settings.hf_home = str(new_path.resolve())
    os.environ["HF_HOME"] = str(new_path.resolve())

    # Persist to .env
    try:
        env_file = Path(".env")
        lines = []
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if not line.startswith("DUBSTUDIO_HF_HOME="):
                    lines.append(line)
        lines.append(f"DUBSTUDIO_HF_HOME={new_path.resolve()}")
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        log.warning("Could not persist HF_HOME to .env: %s", exc)

    return _disk_stats(new_path)


@router.get("/models")
def list_models():
    storage_dir = _get_active_storage_dir()
    models = []
    for item in MODELS_CATALOG:
        m = dict(item)
        is_dl, size_mb = _check_model_downloaded(storage_dir, m["repo_id"])
        # Check extra repos (like SNAC for Veena)
        if is_dl and m.get("extra_repos"):
            for extra in m["extra_repos"]:
                extra_dl, extra_size = _check_model_downloaded(storage_dir, extra)
                if not extra_dl:
                    is_dl = False
                    break
                size_mb += extra_size

        prog = DOWNLOAD_STATUS.get(m["id"])
        if prog and prog.get("status") == "downloading":
            m["status"] = "downloading"
            # Calculate actual unique downloaded bytes on disk in real time
            hub_dir = storage_dir / "hub"
            folder = hub_dir / f"models--{m['repo_id'].replace('/', '--')}"
            dl_bytes = _calc_unique_bytes(folder)
            for extra in m.get("extra_repos", []):
                extra_folder = hub_dir / f"models--{extra.replace('/', '--')}"
                dl_bytes += _calc_unique_bytes(extra_folder)

            dl_mb = round(dl_bytes / (1024 * 1024), 1)
            target_mb = max(1, m["size_mb"])
            calc_pct = min(98, max(5, int((dl_mb / target_mb) * 100)))
            m["progress"] = calc_pct
            m["progress_message"] = f"Downloading {dl_mb:.0f} MB / ~{target_mb} MB ({calc_pct}%)"
            # Keep global status synchronized
            DOWNLOAD_STATUS[m["id"]]["percent"] = calc_pct
            DOWNLOAD_STATUS[m["id"]]["message"] = m["progress_message"]
        elif is_dl:
            m["status"] = "downloaded"
            m["size_on_disk_mb"] = size_mb
            m["progress"] = 100
        else:
            m["status"] = "not_downloaded"
            m["progress"] = 0

        models.append(m)

    return {"models": models, "storage": _disk_stats(storage_dir)}


class DownloadModelPayload(BaseModel):
    model_id: str


def _background_download(model_id: str, repos: list[str], storage_dir: Path, token: str | None):
    try:
        from huggingface_hub import snapshot_download

        total_repos = len(repos)
        for idx, repo in enumerate(repos):
            DOWNLOAD_STATUS[model_id] = {
                "model_id": model_id,
                "status": "downloading",
                "percent": int((idx / total_repos) * 100) + 5,
                "message": f"Downloading {repo} ({idx + 1}/{total_repos})...",
                "error": None,
            }
            log.info("Downloading %s to %s", repo, storage_dir)
            snapshot_download(
                repo_id=repo,
                token=token or None,
                local_files_only=False,
            )

        DOWNLOAD_STATUS[model_id] = {
            "model_id": model_id,
            "status": "completed",
            "percent": 100,
            "message": "Download complete",
            "error": None,
        }
    except Exception as exc:
        log.exception("Download failed for %s: %s", model_id, exc)
        DOWNLOAD_STATUS[model_id] = {
            "model_id": model_id,
            "status": "error",
            "percent": 0,
            "message": "Download failed",
            "error": str(exc),
        }


@router.post("/models/download")
def download_model(payload: DownloadModelPayload):
    model_id = payload.model_id.strip()
    target = next((m for m in MODELS_CATALOG if m["id"] == model_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Unknown model ID '{model_id}'")

    if DOWNLOAD_STATUS.get(model_id, {}).get("status") == "downloading":
        return {"status": "already_downloading", "model_id": model_id}

    storage_dir = _get_active_storage_dir()
    repos = [target["repo_id"]] + list(target.get("extra_repos") or [])

    # Check available disk space (accounting for already downloaded chunks)
    usage = shutil.disk_usage(str(storage_dir))
    free_mb = usage.free / (1024 * 1024)

    hub_dir = storage_dir / "hub"
    already_bytes = _calc_unique_bytes(hub_dir / f"models--{target['repo_id'].replace('/', '--')}")
    for extra in target.get("extra_repos", []):
        already_bytes += _calc_unique_bytes(hub_dir / f"models--{extra.replace('/', '--')}")
    already_mb = already_bytes / (1024 * 1024)
    needed_mb = max(0.0, target["size_mb"] - already_mb)

    if free_mb < needed_mb and needed_mb > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient disk space on {storage_dir}. Needs {int(needed_mb)} MB more, but only {int(free_mb)} MB is free.",
        )

    token = settings.hf_token or os.environ.get("HF_TOKEN") or None
    thread = threading.Thread(
        target=_background_download,
        args=(model_id, repos, storage_dir, token),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "model_id": model_id, "message": "Background download started"}


@router.get("/models/download/progress")
def get_download_progress():
    return {"downloads": DOWNLOAD_STATUS}
