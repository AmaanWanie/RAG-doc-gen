"""Citation formatting utilities for generated document sections."""

from __future__ import annotations

import re

from src.models import RetrievedChunk


def clean_display_filename(source_file: str) -> str:
    """
    Remove internal random upload prefix from source file names.

    Example:
        72e4b736_report.pdf -> report.pdf
    """
    return re.sub(r"^[0-9a-fA-F]{8}_", "", source_file)


def format_sources_markdown(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks as clean Markdown citations.

    Args:
        chunks: Retrieved chunks used for generation.

    Returns:
        Markdown sources list.
    """
    if not chunks:
        return "**Sources used:**\n- No source chunks retrieved."

    lines = ["**Sources used:**"]
    seen: set[str] = set()

    for chunk in chunks:
        display_file = clean_display_filename(chunk.source_file)
        key = f"{display_file}|{chunk.page_number}|{chunk.chunk_index}"

        if key in seen:
            continue

        seen.add(key)

        lines.append(
            f"- `{display_file}` — Page {chunk.page_number}, "
            f"Chunk {chunk.chunk_index}"
        )

    return "\n".join(lines)
