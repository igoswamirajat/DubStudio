from __future__ import annotations

from dubstudio.engines.base import VoiceEngine
from dubstudio.engines.dummy import DummyEngine
from dubstudio.settings import settings


def get_voice_engine(name: str | None = None) -> VoiceEngine:
    """Return the requested voice engine, falling back to DummyEngine.

    Primary: OmniVoice
    Optional: Chatterbox
    Always-safe: DummyEngine

    Engines are constructed but NOT warmed up here — model loading happens
    lazily on the first generate() call, so this stays cheap for availability
    checks and callers that only need the engine name.
    """
    name = (name or settings.tts_engine or "omnivoice").lower().strip()

    # --- OmniVoice (primary) ---
    if name in {"omnivoice", "omni", "omni_voice"}:
        try:
            from dubstudio.engines.omnivoice_engine import OmniVoiceEngine

            return OmniVoiceEngine()
        except Exception:
            pass  # fall through to dummy

    # --- Chatterbox (optional) ---
    if name in {"chatterbox", "xtts"}:
        try:
            from dubstudio.engines.chatterbox_engine import ChatterboxEngine

            return ChatterboxEngine()
        except Exception:
            pass

    # --- Explicit dummy or final safety net ---
    return DummyEngine()
