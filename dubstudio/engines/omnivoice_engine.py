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

# --- pacing -----------------------------------------------------------------
# OmniVoice's own pace for Hindi is far quicker than the words-per-second the
# translation budget is written against. Measured on job
# job_01M2JTG79S269HPHSZ5ETB64ED: 60 Hindi words came back in 13.82s, i.e.
# 4.34 w/s, while translation.py sized the line for 2.6 w/s. Every block
# therefore under-filled its slot by 18-33%, the aligner can only absorb +-8%
# without mangling the delivery, and the remainder shipped as dead air (5.6s of
# silence over a talking presenter, QC status "degraded").
#
# The engine accepts a target `duration` and meets it by budgeting its own
# audio tokens, so one pass is enough - no extra generation, and the aligner
# then has nothing left to stretch. Verified: asking for 20.64s returned 20.84s
# where the unpaced call returned 13.82s.
MIN_NATURAL_WPS = 1.8      # conservative floor; never ask the engine to drawl
MIN_TARGET_DURATION_S = 0.5


def _spoken_words(text: str) -> int:
    """Script-agnostic word count (Devanagari combining marks break ``\\w``)."""
    return sum(1 for tok in (text or "").split() if any(ch.isalnum() for ch in tok))


def _pacing_duration_s(req: SynthRequest) -> float | None:
    """Seconds to ask OmniVoice for, or None to leave its own pace alone."""
    if not req.target_duration_ms or req.target_duration_ms <= 0:
        return None
    target = req.target_duration_ms / 1000.0
    if target < MIN_TARGET_DURATION_S:
        return None
    words = _spoken_words(req.text)
    if words:
        # Stretching a handful of words across a long slot only produces a
        # drawl, so cap the request at a plausible natural length. Whatever is
        # still missing stays visible in QC instead of being hidden inside
        # mangled pacing.
        target = min(target, words / MIN_NATURAL_WPS)
    return round(target, 3)


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

        # Ask for the slot length so the engine paces itself to fill it. Without
        # this the engine returns its own (much quicker) pace and the shortfall
        # becomes dead air at mix time.
        pacing = _pacing_duration_s(req)
        if pacing is not None:
            kwargs["duration"] = pacing

        log.debug("OmniVoice generate mode=%s pacing=%ss keys=%s", mode, pacing, list(kwargs.keys()))
        
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
