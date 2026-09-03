from fastapi import APIRouter

from dubstudio.settings import settings
from dubstudio.util.ffmpeg import ffmpeg_ok

router = APIRouter()


def _tts_status() -> str:
    """Report the configured engine + availability WITHOUT loading the model."""
    import importlib.util

    name = (settings.tts_engine or "dummy").lower()
    if name in {"omnivoice", "omni", "omni_voice"}:
        return "omnivoice" if importlib.util.find_spec("omnivoice") else "omnivoice(missing→dummy)"
    if name in {"chatterbox", "xtts"}:
        return "chatterbox"
    return "dummy"


def _gpu_status() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {"available": True, "device": torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {"available": False, "device": None}


def _asr_status() -> str:
    if settings.asr_engine == "mock":
        return "mock"
    try:
        import faster_whisper  # noqa: F401

        return "faster-whisper"
    except Exception:
        return "not_installed"


def _diarization_status() -> str:
    if not settings.enable_diarization:
        return "disabled"
    try:
        import pyannote.audio  # noqa: F401

        return "installed" if settings.hf_token else "needs_hf_token"
    except Exception:
        return "not_installed"


def _translator_status() -> str:
    t = settings.translator
    if t == "openai":
        return "openai" if settings.openai_api_key else "openai(no_key)"
    if t == "ollama":
        return "ollama"
    return "demo"


@router.get("/health")
def health() -> dict:
    gpu = _gpu_status()
    return {
        "ok": True,
        "ffmpeg": ffmpeg_ok(),
        "gpu": gpu["available"],
        "gpu_device": gpu["device"],
        "asr": _asr_status(),
        "diarization": _diarization_status(),
        "tts": _tts_status(),
        "translator": _translator_status(),
        "separation": "demucs" if not settings.skip_separation else "skip_copy",
    }
