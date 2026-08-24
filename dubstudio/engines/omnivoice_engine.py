"""OmniVoice primary engine adapter.

Primary TTS + zero-shot voice cloning engine for DubStudio.
Falls back cleanly when the package is not installed (factory → DummyEngine).
"""

from __future__ import annotations

from pathlib import Path

from dubstudio.engines.base import SynthRequest, SynthResult, VoiceEngine


class OmniVoiceEngine(VoiceEngine):
    """Adapter over k2-fsa OmniVoice.

    Real generation is implemented when `omnivoice` is installed.
    Until then this class raises on construction so factory falls back.
    """

    name = "omnivoice"

    def __init__(self) -> None:
        try:
            import importlib

            if (
                importlib.util.find_spec("omnivoice") is None
                and importlib.util.find_spec("omni_voice") is None
            ):
                raise ImportError(
                    "omnivoice package not installed. "
                    "pip install omnivoice  (or follow k2-fsa/OmniVoice docs)"
                )
        except Exception as exc:
            raise ImportError(f"OmniVoice unavailable: {exc}") from exc

        self._model = None
        self._device = "cuda"

    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            import importlib

            mod = importlib.import_module("omnivoice")
            self._model = getattr(mod, "OmniVoice", None)
            if self._model is None:
                raise ImportError("OmniVoice class not found in package")
        except Exception as exc:
            raise ImportError(f"OmniVoice load failed: {exc}") from exc

    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        if self._model is None:
            self.warmup()
        raise NotImplementedError(
            "OmniVoice real generate() lands in Phase 2. "
            "Factory already falls back to DummyEngine when this raises/imports fail."
        )

    def unload(self) -> None:
        self._model = None
