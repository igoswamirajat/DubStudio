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
