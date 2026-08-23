from __future__ import annotations

from pathlib import Path

from dubstudio.engines.base import SynthRequest, SynthResult, VoiceEngine
from dubstudio.util.audio import duration_ms, write_tone


class DummyEngine(VoiceEngine):
    name = "dummy"

    def warmup(self) -> None:
        return

    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        chars = max(1, len(req.text.strip()))
        base_s = chars * 0.055
        base_s = base_s / max(0.5, min(2.0, req.speed))
        base_s = max(0.25, min(base_s, 12.0))
        freq = 180.0 + (sum(ord(c) for c in req.voice_id) % 120)
        write_tone(out_wav, base_s, freq=freq, sample_rate=48000, amplitude=0.18)
        return SynthResult(wav_path=out_wav, duration_ms=duration_ms(out_wav), sample_rate=48000, engine=self.name)
