# app/config.py

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = "Persian Whisper STT"
    app_version: str = "1.0.0"

    host: str = "0.0.0.0"
    port: int = 8000

    # ---------------------------------------------------------
    # Security
    # ---------------------------------------------------------

    api_key: str = Field(default="change-me")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model_path: Path = Path(r"D:\ambaqinejad\code\python\openai-compatible-whisper-based-persian-stt\model\snapshots\b84fc89f5d8c6a08acbd0930c74010f8bb555253")

    # ---------------------------------------------------------
    # Audio
    # ---------------------------------------------------------

    sample_rate: int = 16000
    channels: int = 1

    # Whisper works internally around 30 sec windows.
    chunk_seconds: float = 30.0

    # Requested overlap
    overlap_seconds: float = 10.0

    # ---------------------------------------------------------
    # Upload
    # ---------------------------------------------------------

    max_upload_size_mb: int = 4096

    upload_dir: Path = Path("/tmp/stt")

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    device: str = "cuda"

    # float16 is a good choice on RTX GPUs
    torch_dtype: str = "float16"

    language: str = "fa"

    task: str = "transcribe"

    # ---------------------------------------------------------
    # Concurrency
    # ---------------------------------------------------------

    # One Large-v3 inference at a time on a single GPU.
    max_concurrent_transcriptions: int = 1

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------

    num_beams: int = 1

    temperature: float = 0.0

    condition_on_prev_tokens: bool = True

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    if settings.overlap_seconds >= settings.chunk_seconds:
        raise ValueError(
            "overlap_seconds must be smaller than chunk_seconds"
        )

    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Local model directory does not exist: "
            f"{settings.model_path}"
        )

    return settings