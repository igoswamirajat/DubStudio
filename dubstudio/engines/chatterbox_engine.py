from __future__ import annotations

from pathlib import Path

from dubstudio.engines.base import SynthRequest, SynthResult, VoiceEngine


class ChatterboxEngine(VoiceEngine):
    name = "chatterbox"

    def __init__(self) -> None:
        import importlib.util
        if importlib.util.find_spec("chatterbox") is None and importlib.util.find_spec("resemble_chatterbox") is None:
            raise ImportError("chatterbox package not installed")

    def warmup(self) -> None:
        return

    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        raise NotImplementedError("Wire real Chatterbox API when package is installed")
