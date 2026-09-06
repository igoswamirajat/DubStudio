from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SynthRequest:
    text: str
    language: str
    voice_id: str
    ref_wav: Path | None
    emotion: str | None = None
    speed: float = 1.0
    # Phase 2 — OmniVoice modes
    voice_mode: str = "clone"  # clone | design | fixed | auto
    ref_text: str | None = None  # transcript of ref_wav (helps cloning)
    instruct: str | None = None  # e.g. "female, young adult, hindi accent"
    target_duration_ms: int | None = None


@dataclass
class SynthResult:
    wav_path: Path
    duration_ms: int
    sample_rate: int
    engine: str


class VoiceEngine(ABC):
    name: str

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult: ...

    def unload(self) -> None:
        return
