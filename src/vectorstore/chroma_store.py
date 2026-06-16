"""ChromaDB vector store utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.config import CHROMA_DB_DIR
from src.models import DocumentChunk, RetrievedChunk


class ChromaVectorStore:
    """Persistent ChromaDB wrapper for storing and retrieving document chunks."""

    def __init__(
        self,
        persist_directory: Path = CHROMA_DB_DIR,
        collection_name: str = "rag_documents",
    ) -> None:
        """
        Initialize the ChromaDB persistent client.

        Args:
            persist_directory: Directory where ChromaDB data is stored.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """
        Get or create the ChromaDB collection.

        Returns:
            ChromaDB collection.
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self._get_or_create_collection()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Add document chunks and their embeddings to ChromaDB.

        Args:
            chunks: Document chunks to store.
            embeddings: Embedding vectors matching the chunks.

        Returns:
            Number of chunks added.

        Raises:
            ValueError: If chunks and embeddings lengths do not match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "source_file": chunk.source_file,
                "file_path": chunk.file_path,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "character_count": chunk.character_count,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Query ChromaDB using a query embedding.

        Args:
            query_embedding: Embedded user query.
            top_k: Number of chunks to retrieve.

        Returns:
            List of retrieved chunks.
        """
        n_results = min(top_k, self.count())

        if n_results <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(document),
                    source_file=str(metadata.get("source_file", "")),
                    file_path=str(metadata.get("file_path", "")),
                    page_number=int(metadata.get("page_number", 0)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=float(distance) if distance is not None else None,
                )
            )

        return retrieved

    def count(self) -> int:
        """
        Count stored chunks.

        Returns:
            Number of chunks in the collection.
        """
        return int(self.collection.count())
