from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    data_dir: Path = Path("./data")
    max_upload_mb: int = 500
    host: str = "0.0.0.0"
    port: int = 8080
    whisper_model: str = "base"
    tts_engine: str = "dummy"
    translator: str = "demo"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:14b"
    hf_token: str = ""
    low_vram: bool = False
    skip_separation: bool = True
    max_duration_s: int = 600
    chunk_s: int = 30
    enable_rewrite_loop: bool = True
    enable_resume: bool = True

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "dubstudio.db"


settings = Settings()
