"""Text splitting utilities for creating RAG-ready chunks."""

from __future__ import annotations

import re
import textwrap

from src.models import DocumentChunk, DocumentPage


def make_safe_id(text: str) -> str:
    """
    Convert a file name or title into a safe identifier.

    Args:
        text: Input text.

    Returns:
        Safe lowercase identifier.
    """
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return safe or "document"


def split_long_unit_by_words(text: str, max_size: int) -> list[str]:
    """
    Split a very long text unit without breaking words.

    Args:
        text: Text unit that may be longer than the chunk size.
        max_size: Maximum size for each wrapped segment.

    Returns:
        List of word-safe text segments.
    """
    return [
        segment.strip()
        for segment in textwrap.wrap(
            text,
            width=max_size,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if segment.strip()
    ]


def split_text_into_units(text: str, chunk_size: int) -> list[str]:
    """
    Split text into logical units before creating chunks.

    This prefers non-empty lines first, then sentences, then word-safe wrapping.

    Args:
        text: Input text.
        chunk_size: Maximum chunk size.

    Returns:
        List of text units.
    """
    cleaned_text = text.strip()

    if not cleaned_text:
        return []

    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]

    if len(lines) > 1:
        units: list[str] = []
        for line in lines:
            if len(line) <= chunk_size:
                units.append(line)
            else:
                units.extend(split_long_unit_by_words(line, chunk_size))
        return units

    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if len(sentences) > 1:
        units = []
        for sentence in sentences:
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(split_long_unit_by_words(sentence, chunk_size))
        return units

    return split_long_unit_by_words(cleaned_text, chunk_size)


def estimate_units_length(units: list[str]) -> int:
    """
    Estimate the character length of joined text units.

    Args:
        units: Text units.

    Returns:
        Estimated joined character count.
    """
    return len("\n".join(units))


def get_overlap_units(units: list[str], chunk_overlap: int) -> list[str]:
    """
    Keep the last few units for overlap without cutting words.

    Args:
        units: Current chunk units.
        chunk_overlap: Target overlap size in characters.

    Returns:
        Units to carry into the next chunk.
    """
    if chunk_overlap <= 0 or not units:
        return []

    overlap_units: list[str] = []
    total_length = 0

    for unit in reversed(units):
        unit_length = len(unit)

        if overlap_units and total_length + unit_length > chunk_overlap:
            break

        overlap_units.insert(0, unit)
        total_length += unit_length

        if total_length >= chunk_overlap:
            break

    return overlap_units


def split_text_by_characters(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks without cutting words or lines mid-way.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Approximate overlap size in characters.

    Returns:
        List of clean text chunks.

    Raises:
        ValueError: If chunk settings are invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    units = split_text_into_units(text, chunk_size)

    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        candidate_units = current_units + [unit]

        if estimate_units_length(candidate_units) <= chunk_size:
            current_units = candidate_units
            continue

        if current_units:
            chunk_text = "\n".join(current_units).strip()
            if chunk_text:
                chunks.append(chunk_text)

            current_units = get_overlap_units(current_units, chunk_overlap)

        current_units.append(unit)

    if current_units:
        chunk_text = "\n".join(current_units).strip()
        if chunk_text and chunk_text not in chunks:
            chunks.append(chunk_text)

    return chunks


def chunk_page(
    page: DocumentPage,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Create chunks from a single extracted document page.

    Args:
        page: Extracted document page.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Approximate overlap size between chunks.

    Returns:
        List of DocumentChunk objects.
    """
    raw_chunks = split_text_by_characters(
        text=page.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    safe_source = make_safe_id(page.source_file)
    document_chunks: list[DocumentChunk] = []

    for index, chunk_text in enumerate(raw_chunks, start=1):
        chunk_id = f"{safe_source}_p{page.page_number}_c{index}"

        document_chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                source_file=page.source_file,
                file_path=page.file_path,
                page_number=page.page_number,
                chunk_index=index,
                text=chunk_text,
                character_count=len(chunk_text),
            )
        )

    return document_chunks
