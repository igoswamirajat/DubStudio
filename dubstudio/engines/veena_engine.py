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
        # Level 0 (1 token)
        codes_lvl[0].append(snac_tokens[i] - offsets[0])
        # Level 1 (2 tokens)
        codes_lvl[1].append(snac_tokens[i + 1] - offsets[1])
        codes_lvl[1].append(snac_tokens[i + 2] - offsets[2])
        # Level 2 (4 tokens)
        for j in range(4):
            codes_lvl[2].append(snac_tokens[i + 3 + j] - offsets[3 + j])

    # Format tensors with batch dimension
    codes = [
        torch.tensor([lvl], dtype=torch.long, device=device)
        for lvl in codes_lvl
    ]

    with torch.no_grad():
        audio_hat = snac_model.decode(codes)

    # Flatten audio output to 1D numpy array
    return audio_hat[0, 0].cpu().numpy().tolist()


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

                # lm_head stays standard Linear sharing weights with embed_tokens
                model.lm_head = torch.nn.Linear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    device=device,
                    dtype=torch.bfloat16,
                )

                shards = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
                if not shards:
                    raise FileNotFoundError(f"No safetensors shards found in {model_path}")

                for sf_path in shards:
                    log.info("Loading Veena shard: %s", os.path.basename(sf_path))
                    tensors = safetensors.torch.load_file(sf_path, device="cpu")
                    for full_name, tensor in tensors.items():
                        if full_name.startswith("lm_head."):
                            continue
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
                    del tensors

                if config.tie_word_embeddings:
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
                log.exception("Custom 4-bit load failed (%s), falling back to standard from_pretrained", e)
                self._model = None

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

        # 4. Deterministic speaker ID mapping (S00 -> kavya, S01 -> agastya, etc.)
        if vid.startswith("s") and vid[1:].isdigit():
            idx = int(vid[1:]) % len(VOICE_CYCLE)
            return VOICE_CYCLE[idx]

        return "kavya"

    def generate(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        if self._model is None or self._snac_model is None or self._tokenizer is None:
            self.warmup()

        import numpy as np
        import soundfile as sf
        import torch

        text = (req.text or "").strip()
        if not text:
            text = "..."

        speaker = self._resolve_voice(req)
        log.info("Veena synthesis: speaker=%s text='%s'", speaker, text[:40])

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

        # Calculate reasonable max tokens based on text length (7 tokens per audio frame)
        max_tokens = min(int(len(text) * 1.5) * 7 + 28, 800)

        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=[END_OF_SPEECH_TOKEN, END_OF_AI_TOKEN],
            )

        generated_ids = output[0][len(input_tokens):].tolist()
        snac_tokens = [
            tid
            for tid in generated_ids
            if AUDIO_CODE_BASE_OFFSET <= tid < (AUDIO_CODE_BASE_OFFSET + 7 * 4096)
        ]

        # Truncate to multiple of 7
        snac_tokens = snac_tokens[: len(snac_tokens) - (len(snac_tokens) % 7)]

        if not snac_tokens:
            raise ValueError(f"Veena generated no audio tokens for: '{text}'")

        audio_samples = _decode_snac_tokens(snac_tokens, self._snac_model)
        if audio_samples is None or len(audio_samples) == 0:
            raise ValueError("SNAC audio decoding produced no samples")

        wave = np.asarray(audio_samples, dtype=np.float32)

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
