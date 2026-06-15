"""Ingestion pipeline combining extraction results and chunking."""

from __future__ import annotations

from src.ingestion.text_splitter import chunk_page
from src.models import ChunkingResult, DocumentChunk, ExtractionResult


def create_chunks_from_extraction(
    extraction: ExtractionResult,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ChunkingResult:
    """
    Create searchable chunks from an extraction result.

    Args:
        extraction: Extracted document result.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        ChunkingResult containing all chunks for the document.
    """
    all_chunks: list[DocumentChunk] = []

    for page in extraction.pages:
        page_chunks = chunk_page(
            page=page,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    return ChunkingResult(
        source_file=extraction.source_file,
        file_path=extraction.file_path,
        chunks=all_chunks,
        total_chunks=len(all_chunks),
    )
