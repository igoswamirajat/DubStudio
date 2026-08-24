from __future__ import annotations

from dubstudio.engines.base import VoiceEngine
from dubstudio.engines.dummy import DummyEngine
from dubstudio.settings import settings


def get_voice_engine(name: str | None = None) -> VoiceEngine:
    """Return the requested voice engine, falling back to DummyEngine.

    Primary: OmniVoice
    Optional: Chatterbox
    Always-safe: DummyEngine
    """
    name = (name or settings.tts_engine or "omnivoice").lower().strip()

    # --- OmniVoice (primary) ---
    if name in {"omnivoice", "omni", "omni_voice"}:
        try:
            from dubstudio.engines.omnivoice_engine import OmniVoiceEngine

            eng = OmniVoiceEngine()
            eng.warmup()
            return eng
        except Exception:
            pass  # fall through to dummy

    # --- Chatterbox (optional) ---
    if name in {"chatterbox", "xtts"}:
        try:
            from dubstudio.engines.chatterbox_engine import ChatterboxEngine

            eng = ChatterboxEngine()
            eng.warmup()
            return eng
        except Exception:
            pass

    # --- Explicit dummy or final safety net ---
    eng = DummyEngine()
    eng.warmup()
    return eng
