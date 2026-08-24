from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("./data")
    max_upload_mb: int = 500
    host: str = "127.0.0.1"
    port: int = 8080
    whisper_model: str = "large-v3-turbo"
    tts_engine: str = "omnivoice"
    translator: str = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:14b"
    hf_token: str = ""
    low_vram: bool = False
    skip_separation: bool = False

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "dubstudio.db"


settings = Settings()
