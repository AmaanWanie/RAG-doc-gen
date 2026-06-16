"""Retrieval utilities for stronger semantic search with lightweight reranking."""

from __future__ import annotations

import re
from typing import Literal

from src.models import RetrievedChunk
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.embeddings import LocalEmbeddingModel

DistanceFilter = float | Literal["auto"] | None

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized keyword tokens.

    Args:
        text: Input text.

    Returns:
        Normalized tokens.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]


def keyword_overlap_score(query: str, text: str) -> float:
    """
    Calculate a lightweight keyword overlap score.

    This helps exact terms like ChromaDB, Ollama, latency, citation,
    architecture, and evaluation influence ranking.

    Args:
        query: Retrieval query.
        text: Candidate chunk text.

    Returns:
        Score between 0 and 1.
    """
    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    query_terms = set(query_tokens)
    text_terms = set(tokenize(text))

    if not text_terms:
        return 0.0

    matched_terms = query_terms.intersection(text_terms)
    overlap = len(matched_terms) / len(query_terms)

    normalized_query_phrase = " ".join(query_tokens)
    normalized_text = " ".join(tokenize(text))

    phrase_boost = 0.15 if normalized_query_phrase in normalized_text else 0.0

    return min(1.0, overlap + phrase_boost)


def semantic_score_from_distance(distance: float | None) -> float:
    """
    Convert cosine distance into an approximate semantic score.

    ChromaDB cosine distance is lower for more similar chunks.

    Args:
        distance: ChromaDB distance.

    Returns:
        Score between 0 and 1.
    """
    if distance is None:
        return 0.0

    return max(0.0, min(1.0, 1.0 - distance))


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    semantic_weight: float = 0.75,
    keyword_weight: float = 0.25,
) -> list[RetrievedChunk]:
    """
    Rerank candidate chunks using semantic score and keyword overlap.

    Args:
        query: Retrieval query.
        chunks: Candidate chunks from vector search.
        top_k: Final number of chunks to return.
        semantic_weight: Weight for semantic vector similarity.
        keyword_weight: Weight for keyword overlap.

    Returns:
        Reranked final chunks.
    """
    scored_chunks: list[tuple[float, RetrievedChunk]] = []

    for chunk in chunks:
        semantic_score = semantic_score_from_distance(chunk.distance)
        lexical_score = keyword_overlap_score(query, chunk.text)

        final_score = (
            semantic_weight * semantic_score
            + keyword_weight * lexical_score
        )

        scored_chunks.append((final_score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    return [chunk for _, chunk in scored_chunks[:top_k]]


def auto_filter_chunks(
    chunks: list[RetrievedChunk],
    margin: float = 0.25,
    absolute_cap: float = 0.90,
) -> list[RetrievedChunk]:
    """
    Filter candidate chunks using a dynamic semantic distance threshold.

    Args:
        chunks: Retrieved chunks.
        margin: Allowed distance margin above the best chunk.
        absolute_cap: Maximum allowed semantic distance.

    Returns:
        Filtered chunks.
    """
    if not chunks:
        return []

    chunks_with_distance = [chunk for chunk in chunks if chunk.distance is not None]

    if not chunks_with_distance:
        return chunks

    best_distance = min(
        chunk.distance
        for chunk in chunks_with_distance
        if chunk.distance is not None
    )

    dynamic_threshold = min(best_distance + margin, absolute_cap)

    filtered = [
        chunk
        for chunk in chunks
        if chunk.distance is not None and chunk.distance <= dynamic_threshold
    ]

    return filtered or [chunks[0]]


class Retriever:
    """
    Retriever for RAG context search.

    The retriever first asks ChromaDB for a larger candidate set, then reranks
    those candidates using semantic similarity and keyword overlap. This improves
    retrieval quality without adding a separate visible hybrid-search UI.
    """

    def __init__(
        self,
        embedding_model: LocalEmbeddingModel,
        vector_store: ChromaVectorStore,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            embedding_model: Embedding model used to vectorize queries.
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
        Retrieve and rerank relevant chunks.

        Args:
            query: Retrieval query.
            top_k: Final number of chunks to return.
            max_distance: Automatic, manual, or disabled distance filtering.

        Returns:
            Final reranked chunks.
        """
        if not query.strip():
            return []

        query_embedding = self.embedding_model.embed_query(query)

        candidate_top_k = max(top_k * 5, 15)

        candidate_chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=candidate_top_k,
        )

        if not candidate_chunks:
            return []

        if isinstance(max_distance, int | float):
            candidate_chunks = [
                chunk
                for chunk in candidate_chunks
                if chunk.distance is not None and chunk.distance <= float(max_distance)
            ]
        elif max_distance == "auto" or max_distance is None:
            candidate_chunks = auto_filter_chunks(candidate_chunks)

        if not candidate_chunks:
            return []

        return rerank_chunks(
            query=query,
            chunks=candidate_chunks,
            top_k=top_k,
        )
