from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Fields are read from the environment with the DUBSTUDIO_ prefix (e.g.
    # DUBSTUDIO_TTS_ENGINE -> tts_engine). A few well-known names (HF_TOKEN,
    # OLLAMA_HOST/MODEL) are also accepted unprefixed for convenience.
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DUBSTUDIO_", extra="ignore")

    data_dir: Path = Path("./data")
    max_upload_mb: int = 500

    # Local-first default: bind loopback. Docker/compose overrides to 0.0.0.0.
    host: str = "127.0.0.1"
    port: int = 8080

    # --- ASR (speech-to-text) ---
    asr_engine: str = "faster-whisper"  # faster-whisper | mock
    whisper_model: str = "large-v3"
    whisper_device: str = "auto"  # auto | cuda | cpu
    whisper_compute_type: str = "auto"  # auto | float16 | int8_float16 | int8

    # --- TTS ---
    tts_engine: str = "omnivoice"  # omnivoice | chatterbox | dummy

    # --- Translation ---
    translator: str = "ollama"  # ollama | openai | demo
    ollama_host: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("DUBSTUDIO_OLLAMA_HOST", "OLLAMA_HOST"),
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        validation_alias=AliasChoices("DUBSTUDIO_OLLAMA_MODEL", "OLLAMA_MODEL"),
    )
    # OpenAI-compatible API (OpenAI, Groq, DeepSeek, OpenRouter, Gemini compat, Ollama /v1 ...)
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("DUBSTUDIO_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DUBSTUDIO_OPENAI_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("DUBSTUDIO_OPENAI_MODEL", "OPENAI_MODEL"),
    )

    # --- Separation / diarization ---
    skip_separation: bool = False
    demucs_model: str = "htdemucs"
    enable_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    max_speakers: int = 0  # 0 = auto

    # --- Models / auth ---
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices("DUBSTUDIO_HF_TOKEN", "HF_TOKEN"),
    )
    # Where model weights are cached (Whisper / OmniVoice / Demucs / pyannote).
    # Point this at a drive with space; exported to HF_HOME at startup.
    hf_home: str = Field(
        default="",
        validation_alias=AliasChoices("DUBSTUDIO_HF_HOME", "HF_HOME"),
    )

    # --- Runtime ---
    low_vram: bool = False
    max_duration_s: int = 600
    chunk_s: int = 30
    enable_rewrite_loop: bool = True
    enable_resume: bool = True

    def model_post_init(self, __context: Any) -> None:
        import os

        if "openrouter" in self.openai_base_url.lower() or self.openai_api_key in {"local-bypass", "dummy", ""}:
            router_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if router_key:
                self.openai_api_key = router_key

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "dubstudio.db"


settings = Settings()
