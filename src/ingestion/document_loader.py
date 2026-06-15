"""Document loading utilities for PDF and text files."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import BinaryIO

import fitz

from src.models import DocumentPage, ExtractionResult


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def clean_text(text: str) -> str:
    """Normalize extracted text by removing excessive whitespace."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_uploaded_file(uploaded_file: BinaryIO, upload_dir: Path) -> Path:
    """
    Save a Streamlit uploaded file to disk using a unique safe file name.

    Args:
        uploaded_file: File-like object uploaded through Streamlit.
        upload_dir: Directory where uploaded files should be saved.

    Returns:
        Path to the saved file.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(getattr(uploaded_file, "name", "uploaded_file")).name
    unique_name = f"{uuid.uuid4().hex[:8]}_{original_name}"
    output_path = upload_dir / unique_name

    with output_path.open("wb") as file:
        file.write(uploaded_file.getbuffer())

    return output_path


def extract_pdf_pages(file_path: Path) -> list[DocumentPage]:
    """
    Extract text page by page from a PDF file.

    Args:
        file_path: Path to a PDF file.

    Returns:
        List of extracted pages with metadata.
    """
    pages: list[DocumentPage] = []

    with fitz.open(file_path) as document:
        for index, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))

            if text:
                pages.append(
                    DocumentPage(
                        source_file=file_path.name,
                        file_path=str(file_path),
                        page_number=index,
                        text=text,
                    )
                )

    return pages


def extract_text_file(file_path: Path) -> list[DocumentPage]:
    """
    Extract text from a TXT or Markdown file.

    Args:
        file_path: Path to a text-like file.

    Returns:
        A single-page DocumentPage list.
    """
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = file_path.read_text(encoding="latin-1")

    text = clean_text(raw_text)

    if not text:
        return []

    return [
        DocumentPage(
            source_file=file_path.name,
            file_path=str(file_path),
            page_number=1,
            text=text,
        )
    ]


def load_document(file_path: str | Path) -> ExtractionResult:
    """
    Load and extract text from a supported document.

    Args:
        file_path: Path to PDF, TXT, or Markdown file.

    Returns:
        ExtractionResult containing extracted pages and metadata.

    Raises:
        ValueError: If file type is unsupported or no extractable text is found.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        pages = extract_pdf_pages(path)
    else:
        pages = extract_text_file(path)

    if not pages:
        raise ValueError(f"No extractable text found in file: {path.name}")

    total_characters = sum(len(page.text) for page in pages)

    return ExtractionResult(
        source_file=path.name,
        file_path=str(path),
        pages=pages,
        total_characters=total_characters,
    )
