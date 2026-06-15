"""Retrieval utilities for searching relevant document chunks."""

from __future__ import annotations

from typing import Literal

from src.models import RetrievedChunk
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.embeddings import LocalEmbeddingModel


DistanceFilter = float | Literal["auto"] | None


def auto_filter_chunks(
    chunks: list[RetrievedChunk],
    margin: float = 0.20,
    absolute_cap: float = 0.85,
) -> list[RetrievedChunk]:
    """
    Automatically filter retrieved chunks based on their distance scores.

    Lower distance means a stronger semantic match.

    Args:
        chunks: Retrieved chunks from vector search.
        margin: Allowed distance gap from the best chunk.
        absolute_cap: Maximum allowed distance for keeping weak chunks.

    Returns:
        Filtered chunks. Always keeps at least the best chunk if chunks exist.
    """
    if not chunks:
        return []

    chunks_with_distance = [
        chunk for chunk in chunks if chunk.distance is not None
    ]

    if not chunks_with_distance:
        return chunks[:1]

    best_distance = min(chunk.distance for chunk in chunks_with_distance if chunk.distance is not None)
    dynamic_threshold = min(best_distance + margin, absolute_cap)

    filtered = [
        chunk
        for chunk in chunks
        if chunk.distance is not None and chunk.distance <= dynamic_threshold
    ]

    if filtered:
        return filtered

    return [chunks[0]]


class Retriever:
    """Retriever that searches ChromaDB using embedded queries."""

    def __init__(
        self,
        embedding_model: LocalEmbeddingModel,
        vector_store: ChromaVectorStore,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            embedding_model: Embedding model used to embed search queries.
            vector_store: ChromaDB vector store.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_distance: DistanceFilter = "auto",
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks relevant to a query.

        Args:
            query: Search query.
            top_k: Number of chunks to retrieve before filtering.
            max_distance:
                - "auto" or None: automatically filter weak chunks.
                - float: keep chunks with distance <= max_distance.

        Returns:
            Relevant chunks after distance filtering.
        """
        if not query.strip():
            return []

        query_embedding = self.embedding_model.embed_query(query)
        retrieved_chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        if max_distance == "auto" or max_distance is None:
            return auto_filter_chunks(retrieved_chunks)

        return [
            chunk
            for chunk in retrieved_chunks
            if chunk.distance is not None and chunk.distance <= float(max_distance)
        ]
