"""Maya Research Veena TTS Engine.

Veena is a 3B autoregressive Llama-based TTS model developed by Maya Research,
specifically engineered for native Hindi, English, and code-mixed speech with
expressive prosody, low latency, and 24 kHz audio output via the SNAC neural codec.

Voices:
  - kavya: Female, warm & expressive (default female)
  - agastya: Male, clear & articulate (default male)
  - maitri: Female, conversational & expressive
  - vinaya: Male, deep & authoritative
"""

from __future__ import annotations

import logging
from pathlib import Path

from dubstudio.engines.base import SynthRequest, SynthResult, VoiceEngine

log = logging.getLogger("dubstudio.veena")
VEENA_SR = 24000

# Control token IDs (fixed for Veena architecture)
START_OF_SPEECH_TOKEN = 128257
END_OF_SPEECH_TOKEN = 128258
START_OF_HUMAN_TOKEN = 128259
END_OF_HUMAN_TOKEN = 128260
START_OF_AI_TOKEN = 128261
END_OF_AI_TOKEN = 128262
AUDIO_CODE_BASE_OFFSET = 128266

KNOWN_VOICES = {"kavya", "agastya", "maitri", "vinaya"}
VOICE_CYCLE = ["kavya", "agastya", "maitri", "vinaya"]


def _decode_snac_tokens(snac_tokens: list[int], snac_model) -> list[float] | None:
    """De-interleave and decode SNAC tokens to 24 kHz audio waveform."""
    if not snac_tokens or len(snac_tokens) % 7 != 0:
        return None

    import torch

    device = next(snac_model.parameters()).device
    codes_lvl = [[] for _ in range(3)]
    offsets = [AUDIO_CODE_BASE_OFFSET + i * 4096 for i in range(7)]

    for i in range(0, len(snac_tokens), 7):
        # Level 0: Coarse (1 token)
        codes_lvl[0].append(snac_tokens[i] - offsets[0])
        # Level 1: Medium (2 tokens)
        codes_lvl[1].append(snac_tokens[i + 1] - offsets[1])
        codes_lvl[1].append(snac_tokens[i + 4] - offsets[4])
        # Level 2: Fine (4 tokens)
        codes_lvl[2].append(snac_tokens[i + 2] - offsets[2])
        codes_lvl[2].append(snac_tokens[i + 3] - offsets[3])
        codes_lvl[2].append(snac_tokens[i + 5] - offsets[5])
        codes_lvl[2].append(snac_tokens[i + 6] - offsets[6])

    hierarchical_codes = []
    for lvl_codes in codes_lvl:
        tensor = torch.tensor(lvl_codes, dtype=torch.int32, device=device).unsqueeze(0)
        if torch.any((tensor < 0) | (tensor > 4095)):
            log.warning("SNAC token values out of range [0, 4095]")
            return None
        hierarchical_codes.append(tensor)

    with torch.no_grad():
        audio_hat = snac_model.decode(hierarchical_codes)

    # Flatten audio output to 1D numpy array
    return audio_hat.squeeze().clamp(-1, 1).cpu().numpy().tolist()


def _detect_ref_gender(ref_path: Path | str | None) -> str | None:
    """Analyze pitch F0 of reference clip to classify speaker as male or female."""
    if not ref_path:
        return None
    try:
        import numpy as np
        import soundfile as sf

        path = Path(ref_path)
        if not path.is_file():
            return None
        data, sr = sf.read(str(path))
        if len(data.shape) > 1:
            data = data[:, 0]
        data = data[: int(sr * 3.0)]
        data = data - np.mean(data)
        corr = np.correlate(data, data, mode="full")
        corr = corr[len(corr) // 2 :]
        min_lag = int(sr / 350)
        max_lag = int(sr / 75)
        d = np.diff(corr)
        start = np.where(d > 0)[0]
        if len(start) > 0 and start[0] < max_lag:
            peak = start[0] + np.argmax(corr[start[0]:max_lag])
            f0 = sr / peak
            return "male" if f0 < 160 else "female"
    except Exception:
        pass
    return None


class VeenaEngine(VoiceEngine):
    name = "veena"

    def __init__(
        self,
        model_id: str = "maya-research/veena",
        device_map: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._snac_model = None

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

    def _resolve_snapshot_dir(self) -> Path | None:
        """Find local Hugging Face snapshot directory if model was downloaded."""
        try:
            from dubstudio.api.routes_models import _get_active_storage_dir

            base = _get_active_storage_dir() / "hub"
            repo_slug = f"models--{self.model_id.replace('/', '--')}"
            snap_base = base / repo_slug / "snapshots"
            if snap_base.is_dir():
                for sub in snap_base.iterdir():
                    if sub.is_dir():
                        return sub
        except Exception:
            pass
        return None

    def warmup(self) -> None:
        if self._model is not None:
            return

        import gc
        import glob
        import os
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = self._pick_device()
        is_cuda = device.startswith("cuda")

        log.info("Loading Veena (%s) on %s", self.model_id, device)

        snap_dir = self._resolve_snapshot_dir()
        model_path = str(snap_dir) if snap_dir else self.model_id

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if is_cuda and self.load_in_4bit:
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig
                from transformers.integrations.bitsandbytes import replace_with_bnb_linear
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                import safetensors.torch

                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(config)
                replace_with_bnb_linear(model, quantization_config=quant)

                shards = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
                if not shards:
                    raise FileNotFoundError(f"No safetensors shards found in {model_path}")

                from safetensors import safe_open

                for sf_path in shards:
                    log.info("Loading Veena shard: %s", os.path.basename(sf_path))
                    with safe_open(sf_path, framework="pt", device="cpu") as f:
                        for full_name in f.keys():
                            if full_name.startswith("lm_head."):
                                continue
                            tensor = f.get_tensor(full_name)
                            parts = full_name.split(".")
                            parent = model
                            for p in parts[:-1]:
                                parent = getattr(parent, p)
                            attr = parts[-1]

                            if isinstance(parent, bnb.nn.Linear4bit) and attr == "weight":
                                p4 = bnb.nn.Params4bit(tensor, requires_grad=False, quant_type="nf4").to(device)
                                parent.weight = p4
                            else:
                                p_norm = torch.nn.Parameter(
                                    tensor.to(device, dtype=torch.bfloat16), requires_grad=False
                                )
                                setattr(parent, attr, p_norm)
                            del tensor
                    torch.cuda.empty_cache()
                    gc.collect()

                if config.tie_word_embeddings:
                    model.lm_head = torch.nn.Linear(
                        config.hidden_size,
                        config.vocab_size,
                        bias=False,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    model.lm_head.weight = model.model.embed_tokens.weight

                model.model.rotary_emb = LlamaRotaryEmbedding(config=config, device=device)
                model.eval()
                self._model = model
                log.info(
                    "Veena 4-bit loaded successfully on %s! VRAM allocated: %.2f GB",
                    device,
                    torch.cuda.memory_allocated() / (1024**3),
                )
            except Exception as e:
                log.exception("Custom 4-bit load failed: %s", e)
                raise RuntimeError(f"Veena 4-bit CUDA initialization failed: {e}") from e

        if self._model is None:
            torch_dtype = (
                torch.bfloat16
                if is_cuda and torch.cuda.is_bf16_supported()
                else torch.float16
                if is_cuda
                else torch.float32
            )
            model_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
            if is_cuda:
                model_kwargs["device_map"] = device
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            if not is_cuda:
                self._model.to("cpu")
            self._model.eval()

        # Initialize SNAC 24kHz neural audio codec
        try:
            from snac import SNAC

            log.info("Loading SNAC 24kHz codec (hubertsiuzdak/snac_24khz)")
            snac_m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            if is_cuda:
                snac_m = snac_m.to(device)
            self._snac_model = snac_m
        except Exception as exc:
            raise RuntimeError(f"Failed to load SNAC audio codec: {exc}") from exc

        self.device_map = device

    def _anchor_pitch(self, audio: np.ndarray, speaker: str, sr: int = VEENA_SR) -> np.ndarray:
        """Dynamically anchor male voices (agastya/vinaya) into natural male register (110-130 Hz)."""
        if speaker not in ("agastya", "vinaya") or len(audio) < int(sr * 0.3):
            return audio

        try:
            import numpy as np
            import torch
            import torchaudio.functional as F

            frame_len = int(sr * 0.04)
            hop_len = int(sr * 0.02)
            f0s = []
            for i in range(0, len(audio) - frame_len, hop_len):
                chunk = audio[i:i + frame_len]
                if np.max(np.abs(chunk)) < 0.03:
                    continue
                corr = np.correlate(chunk, chunk, mode="full")[frame_len - 1:]
                min_lag = int(sr / 400)
                max_lag = int(sr / 70)
                d = np.diff(corr)
                starts = np.where(d > 0)[0]
                if len(starts) > 0 and starts[0] < max_lag:
                    peak = starts[0] + np.argmax(corr[starts[0]:max_lag])
                    f0 = sr / peak
                    if 70 <= f0 <= 380:
                        f0s.append(f0)

            if f0s:
                median_f0 = float(np.median(f0s))
                target_f0 = 130.0 if speaker == "agastya" else 110.0
                # If synthesized voice drifted into female register (> 160 Hz)
                if median_f0 > 160.0:
                    semitones = int(round(12.0 * np.log2(target_f0 / median_f0)))
                    if semitones <= -2:
                        t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                        orig_rms = float(np.sqrt(np.mean(audio ** 2)))
                        shifted = F.pitch_shift(t, sr, semitones).squeeze(0).numpy()
                        shift_rms = float(np.sqrt(np.mean(shifted ** 2)))
                        if shift_rms > 1e-5 and orig_rms > 1e-5:
                            shifted = shifted * (orig_rms / shift_rms)
                        shifted = np.clip(shifted, -1.0, 1.0).astype(np.float32)
                        log.info(
                            "Anchored %s pitch: %.1f Hz -> target %.1f Hz (%d semitones)",
                            speaker,
                            median_f0,
                            target_f0,
                            semitones,
                        )
                        return shifted
        except Exception as exc:
            log.warning("Pitch anchoring failed for %s: %s", speaker, exc)

        return audio

    def _resolve_voice(self, req: SynthRequest) -> str:
        """Map user request / speaker id / instruct to one of the 4 native voices."""
        # 1. Direct match in voice_id
        vid = (req.voice_id or "").lower().strip()
        if vid in KNOWN_VOICES:
            return vid

        # 2. Check instruct text
        inst = (req.instruct or "").lower()
        for v in KNOWN_VOICES:
            if v in inst:
                return v

        # 3. Check male / female cues in instruct
        if "male" in inst and "female" not in inst:
            return "agastya"
        if "female" in inst:
            return "kavya"

        # 4. Auto-detect gender from reference audio if explicitly available and reliable
        if req.ref_wav:
            detected = _detect_ref_gender(req.ref_wav)
            if detected == "female":
                return "kavya"
            if detected == "male":
                return "agastya"

        # 5. Deterministic speaker ID mapping (S00 -> agastya, S01 -> kavya, etc.)
        if vid.startswith("s") and vid[1:].isdigit():
            idx = int(vid[1:]) % len(VOICE_CYCLE)
            return VOICE_CYCLE[idx]

        return "agastya"

    def _generate_single(self, req: SynthRequest) -> np.ndarray:
        import numpy as np
        import torch

        text = (req.text or "").strip()
        if not text:
            return np.zeros(int(VEENA_SR * 0.1), dtype=np.float32)

        speaker = self._resolve_voice(req)

        prompt = f"<spk_{speaker}> {text}"
        prompt_tokens = self._tokenizer.encode(prompt, add_special_tokens=False)

        input_tokens = [
            START_OF_HUMAN_TOKEN,
            *prompt_tokens,
            END_OF_HUMAN_TOKEN,
            START_OF_AI_TOKEN,
            START_OF_SPEECH_TOKEN,
        ]

        device = next(self._model.parameters()).device
        input_ids = torch.tensor([input_tokens], device=device)

        # SNAC 24kHz generates at 200 tokens/sec. Provide generous token budget so speech
        # NEVER truncates mid-sentence. Model naturally stops on END_OF_SPEECH_TOKEN.
        token_count = max(350, int(len(text) * 4.0) * 7)
        max_tokens = min(2048 - len(input_tokens) - 10, token_count)
        max_tokens = max(140, max_tokens - (max_tokens % 7))

        stop_token_ids = [END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN, 128009, 128001]

        # Natural prosody & emotional cadence (Maya Research official parameters: temp 0.40, top_p 0.90)
        # Avoid forcing low temperatures (<0.30) which causes flat, robotic, monotone cadence.
        temp = 0.40
        top_p = 0.90

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=stop_token_ids,
            )

        generated_ids = output[0][len(input_tokens):].tolist()
        snac_tokens = []
        for tid in generated_ids:
            if tid in stop_token_ids:
                break
            if AUDIO_CODE_BASE_OFFSET <= tid < (AUDIO_CODE_BASE_OFFSET + 7 * 4096):
                snac_tokens.append(tid)

        # Truncate to multiple of 7
        snac_tokens = snac_tokens[: len(snac_tokens) - (len(snac_tokens) % 7)]
        if not snac_tokens:
            log.warning("Veena produced no audio tokens for '%s'; returning silence", text[:30])
            return np.zeros(int(VEENA_SR * 0.2), dtype=np.float32)

        audio_samples = _decode_snac_tokens(snac_tokens, self._snac_model)
        if audio_samples is None or len(audio_samples) == 0:
            return np.zeros(int(VEENA_SR * 0.2), dtype=np.float32)

        raw_wave = np.asarray(audio_samples, dtype=np.float32)
        return raw_wave

    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        if self._model is None or self._snac_model is None or self._tokenizer is None:
            self.warmup()

        import re
        import numpy as np
        import soundfile as sf

        text = (req.text or "").strip()
        if not text:
            text = "..."

        # Normalize numbers, symbols, and technical terms into authentic Hindi phonetics
        try:
            from dubstudio.util.phonetics import normalize_hinglish

            text = normalize_hinglish(text, req.language or "hi")
        except Exception:
            pass

        # Generate each dialogue segment as a single fluid, unbroken utterance
        # to prevent unnatural speech breaks, hiccups, and dead zero silences.
        # Only break if text is exceptionally long (> 140 chars).
        if len(text) > 140 and any(p in text for p in ("।", ".", "!", "?")):
            raw_sentences = [s.strip() for s in re.split(r"[।\.\!\?]+", text) if s.strip()]
            clauses: list[str] = []
            buf = ""
            for s in raw_sentences:
                if buf and (len(buf) + len(s) > 90):
                    clauses.append(buf.strip())
                    buf = s
                else:
                    buf = f"{buf} {s}".strip() if buf else s
            if buf:
                clauses.append(buf.strip())

            waves: list[np.ndarray] = []
            target_total = req.target_duration_ms or 0
            total_len = max(1, sum(len(c) for c in clauses))
            for c in clauses:
                sub_target = int(target_total * (len(c) / total_len)) if target_total else None
                sub_req = SynthRequest(
                    text=c,
                    language=req.language,
                    voice_id=req.voice_id,
                    ref_wav=req.ref_wav,
                    emotion=req.emotion,
                    speed=req.speed,
                    voice_mode=req.voice_mode,
                    ref_text=req.ref_text,
                    instruct=req.instruct,
                    target_duration_ms=sub_target,
                )
                w = self._generate_single(sub_req)
                waves.append(w)

            # Stitch with a smooth 15ms raised-cosine crossfade (no digital zero dead silence)
            xfade_samples = int(VEENA_SR * 0.015)
            full_wave: list[float] = []
            for idx, w in enumerate(waves):
                if idx == 0:
                    full_wave.extend(w.tolist())
                else:
                    # Apply crossfade over the junction
                    t = np.linspace(0, 1, xfade_samples, dtype=np.float32)
                    fade_in = 0.5 * (1 - np.cos(np.pi * t))
                    fade_out = 1.0 - fade_in
                    tail = np.array(full_wave[-xfade_samples:], dtype=np.float32)
                    head = w[:xfade_samples]
                    blended = tail * fade_out + head * fade_in
                    full_wave[-xfade_samples:] = blended.tolist()
                    full_wave.extend(w[xfade_samples:].tolist())
            wave = np.asarray(full_wave, dtype=np.float32)
        else:
            single_req = SynthRequest(
                text=text,
                language=req.language,
                voice_id=req.voice_id,
                ref_wav=req.ref_wav,
                emotion=req.emotion,
                speed=req.speed,
                voice_mode=req.voice_mode,
                ref_text=req.ref_text,
                instruct=req.instruct,
                target_duration_ms=req.target_duration_ms,
            )
            wave = self._generate_single(single_req)

        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), wave, VEENA_SR)

        duration_ms = int(round(len(wave) / VEENA_SR * 1000))
        return SynthResult(
            wav_path=out_wav,
            duration_ms=duration_ms,
            sample_rate=VEENA_SR,
            engine=self.name,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._snac_model = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
