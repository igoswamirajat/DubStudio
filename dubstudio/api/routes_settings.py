"""Runtime settings panel API (local single-user).

Exposes the editable subset of Settings so the whole studio can be configured
from the UI. Secrets are masked on read and persisted to .env so changes
survive a restart.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from dubstudio.settings import settings

router = APIRouter()

# field name -> (.env variable, type)
_EDITABLE: dict[str, tuple[str, type]] = {
    "translator": ("DUBSTUDIO_TRANSLATOR", str),
    "ollama_host": ("DUBSTUDIO_OLLAMA_HOST", str),
    "ollama_model": ("DUBSTUDIO_OLLAMA_MODEL", str),
    "openai_base_url": ("DUBSTUDIO_OPENAI_BASE_URL", str),
    "openai_model": ("DUBSTUDIO_OPENAI_MODEL", str),
    "openai_api_key": ("DUBSTUDIO_OPENAI_API_KEY", str),
    "tts_engine": ("DUBSTUDIO_TTS_ENGINE", str),
    "asr_engine": ("DUBSTUDIO_ASR_ENGINE", str),
    "whisper_model": ("DUBSTUDIO_WHISPER_MODEL", str),
    "enable_diarization": ("DUBSTUDIO_ENABLE_DIARIZATION", bool),
    "diarization_model": ("DUBSTUDIO_DIARIZATION_MODEL", str),
    "hf_token": ("DUBSTUDIO_HF_TOKEN", str),
    "hf_home": ("DUBSTUDIO_HF_HOME", str),
    "max_speakers": ("DUBSTUDIO_MAX_SPEAKERS", int),
    "skip_separation": ("DUBSTUDIO_SKIP_SEPARATION", bool),
    "demucs_model": ("DUBSTUDIO_DEMUCS_MODEL", str),
    "low_vram": ("DUBSTUDIO_LOW_VRAM", bool),
}
_SECRETS = {"openai_api_key", "hf_token"}


class SettingsPatch(BaseModel):
    translator: str | None = None
    ollama_host: str | None = None
    ollama_model: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None
    openai_api_key: str | None = None
    tts_engine: str | None = None
    asr_engine: str | None = None
    whisper_model: str | None = None
    enable_diarization: bool | None = None
    diarization_model: str | None = None
    hf_token: str | None = None
    hf_home: str | None = None
    max_speakers: int | None = None
    skip_separation: bool | None = None
    demucs_model: str | None = None
    low_vram: bool | None = None


def _public() -> dict:
    out: dict = {}
    for field in _EDITABLE:
        val = getattr(settings, field)
        if field in _SECRETS:
            out[f"{field}_set"] = bool(val)
        else:
            out[field] = val
    return out


def _env_value(typ: type, val) -> str:
    if typ is bool:
        return "1" if val else "0"
    return str(val)


def _persist_env(updates: dict[str, str]) -> None:
    env_path = Path(".env")
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            seen.add(key)
    for key, val in updates.items():
        if key not in seen:
            lines.append(f"{key}={val}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@router.get("/settings")
def get_settings() -> dict:
    return _public()


@router.patch("/settings")
def patch_settings(body: SettingsPatch) -> dict:
    data = body.model_dump(exclude_none=True)
    env_updates: dict[str, str] = {}
    for field, val in data.items():
        env_var, typ = _EDITABLE[field]
        setattr(settings, field, val)
        env_updates[env_var] = _env_value(typ, val)
    if env_updates:
        _persist_env(env_updates)
    return _public()
