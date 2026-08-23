from fastapi import APIRouter

from dubstudio.util.ffmpeg import ffmpeg_ok

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "ffmpeg": ffmpeg_ok(),
        "gpu": False,
        "whisper": "not_loaded",
        "tts": "not_loaded",
        "phase": 0,
    }
