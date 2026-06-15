"""Application configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
TEMPLATE_DIR = DATA_DIR / "templates"
OUTPUT_DIR = DATA_DIR / "outputs"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")


def ensure_directories() -> None:
    """Create required runtime directories if they do not exist."""
    for directory in [DATA_DIR, UPLOAD_DIR, CHROMA_DB_DIR, TEMPLATE_DIR, OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
