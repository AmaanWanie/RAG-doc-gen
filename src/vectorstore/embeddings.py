"""Embedding utilities for converting text into vectors."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddingModel:
    """Local sentence-transformers embedding model wrapper."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """
        Initialize the local embedding model.

        Args:
            model_name: Hugging Face sentence-transformers model name.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed one search query.

        Args:
            query: Search query text.

        Returns:
            Query embedding vector.
        """
        return self.embed_texts([query])[0]
