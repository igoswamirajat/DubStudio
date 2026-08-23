from __future__ import annotations

from dubstudio.engines.base import VoiceEngine
from dubstudio.engines.dummy import DummyEngine
from dubstudio.settings import settings


def get_voice_engine(name: str | None = None) -> VoiceEngine:
    name = (name or settings.tts_engine or "dummy").lower()
    if name in {"chatterbox", "xtts"}:
        try:
            if name == "chatterbox":
                from dubstudio.engines.chatterbox_engine import ChatterboxEngine
                eng = ChatterboxEngine()
                eng.warmup()
                return eng
        except Exception:
            pass
    eng = DummyEngine()
    eng.warmup()
    return eng
