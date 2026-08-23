from fastapi import APIRouter

from dubstudio.engines.factory import get_voice_engine
from dubstudio.settings import settings
from dubstudio.util.ffmpeg import ffmpeg_ok

router = APIRouter()


@router.get("/health")
def health() -> dict:
    tts_name = "unknown"
    try:
        tts_name = get_voice_engine().name
    except Exception:
        tts_name = "error"
    whisper = "not_loaded"
    try:
        import whisperx  # noqa: F401

        whisper = "installed"
    except Exception:
        whisper = "mock"
    return {
        "ok": True,
        "ffmpeg": ffmpeg_ok(),
        "gpu": False,
        "whisper": whisper,
        "tts": tts_name,
        "translator": settings.translator,
        "phase": 1,
    }
