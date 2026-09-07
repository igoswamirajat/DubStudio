"""OmniVoice primary engine (Phase 2).

API (k2-fsa/OmniVoice):
  model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=..., dtype=...)
  audio = model.generate(text=..., ref_audio=..., ref_text=...)   # clone
  audio = model.generate(text=..., instruct="female, young")     # design
  audio = model.generate(text=...)                               # auto
  # audio[0] is np.ndarray @ 24 kHz

Falls back cleanly when package is not installed (factory → DummyEngine).
"""

from __future__ import annotations

import logging
from pathlib import Path

from dubstudio.engines.base import SynthRequest, SynthResult, VoiceEngine

log = logging.getLogger("dubstudio.omnivoice")
OMNIVOICE_SR = 24000


class OmniVoiceEngine(VoiceEngine):
    name = "omnivoice"

    def __init__(self, model_id: str = "k2-fsa/OmniVoice", device_map: str | None = None) -> None:
        try:
            import importlib

            if importlib.util.find_spec("omnivoice") is None:
                raise ImportError(
                    "omnivoice package not installed. "
                    "pip install omnivoice  (https://github.com/k2-fsa/OmniVoice)"
                )
        except Exception as exc:
            raise ImportError(f"OmniVoice unavailable: {exc}") from exc

        self.model_id = model_id
        self.device_map = device_map
        self._model = None

    def _pick_device(self) -> str:
        if self.device_map:
            return self.device_map
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def warmup(self) -> None:
        if self._model is not None:
            return
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from omnivoice import OmniVoice

        device = self._pick_device()
        dtype = torch.float16 if device.startswith("cuda") or device == "mps" else torch.float32
        log.info("Loading OmniVoice %s on %s (%s)", self.model_id, device, dtype)
        self._model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=device,
            dtype=dtype,
            load_asr=False,
        )
        self.device_map = device

    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        if self._model is None:
            self.warmup()

        import numpy as np
        import soundfile as sf

        text = (req.text or "").strip() or "."
        if not text:
            raise ValueError("Empty text provided to OmniVoice generation")

        kwargs: dict = {"text": text}
        mode = (req.voice_mode or "clone").lower()

        if mode == "design" and req.instruct:
            kwargs["instruct"] = req.instruct
        elif mode in {"clone", "fixed"} and req.ref_wav and Path(req.ref_wav).exists():
            kwargs["ref_audio"] = str(req.ref_wav)
            if req.ref_text:
                kwargs["ref_text"] = req.ref_text
        elif mode == "auto":
            pass
        elif req.instruct:
            kwargs["instruct"] = req.instruct
        elif req.ref_wav and Path(req.ref_wav).exists():
            kwargs["ref_audio"] = str(req.ref_wav)
            if req.ref_text:
                kwargs["ref_text"] = req.ref_text

        if req.language:
            kwargs.setdefault("language", req.language)

        log.debug("OmniVoice generate mode=%s keys=%s", mode, list(kwargs.keys()))
        
        try:
            audio = self._model.generate(**kwargs)
        except Exception as e:
            log.error("OmniVoice generation failed: %s", e)
            raise RuntimeError(f"OmniVoice generation failed: {e}") from e

        if audio is None:
            raise RuntimeError("OmniVoice returned None (generation failed)")

        if isinstance(audio, (list, tuple)):
            wave = audio[0]
        else:
            wave = audio
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim > 1:
            wave = wave.mean(axis=-1)

        if len(wave) == 0:
            raise RuntimeError("OmniVoice generated empty audio")

        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), wave, OMNIVOICE_SR)

        duration_ms = int(round(len(wave) / OMNIVOICE_SR * 1000))
        return SynthResult(
            wav_path=out_wav,
            duration_ms=duration_ms,
            sample_rate=OMNIVOICE_SR,
            engine=self.name,
        )

    def unload(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
