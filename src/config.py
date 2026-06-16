"""Application configuration and runtime paths."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables or .env.

    Defaults are chosen so the app can run locally with Ollama without
    requiring cloud credentials.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama3.2:latest", alias="OLLAMA_MODEL")

    chroma_db_dir: Path = Field(
        default=BASE_DIR / "data" / "chroma_db",
        alias="CHROMA_DB_DIR",
    )
    upload_dir: Path = Field(
        default=BASE_DIR / "data" / "uploads",
        alias="UPLOAD_DIR",
    )
    template_dir: Path = Field(
        default=BASE_DIR / "data" / "templates",
        alias="TEMPLATE_DIR",
    )
    output_dir: Path = Field(
        default=BASE_DIR / "data" / "outputs",
        alias="OUTPUT_DIR",
    )

    @property
    def normalized_llm_provider(self) -> str:
        """Return lower-cased LLM provider name."""
        return self.llm_provider.lower().strip()

    @property
    def normalized_ollama_base_url(self) -> str:
        """Return Ollama base URL without trailing slash."""
        return self.ollama_base_url.rstrip("/")


settings = AppSettings()

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = settings.upload_dir
CHROMA_DB_DIR = settings.chroma_db_dir
TEMPLATE_DIR = settings.template_dir
OUTPUT_DIR = settings.output_dir

LLM_PROVIDER = settings.normalized_llm_provider
OPENAI_API_KEY = settings.openai_api_key
OPENAI_MODEL = settings.openai_model
OLLAMA_BASE_URL = settings.normalized_ollama_base_url
OLLAMA_MODEL = settings.ollama_model


def ensure_directories() -> None:
    """Create required runtime directories if they do not exist."""
    for directory in [DATA_DIR, UPLOAD_DIR, CHROMA_DB_DIR, TEMPLATE_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
