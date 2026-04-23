"""
vector_store.py
---------------
Module 4: Vector Storage
Manages ChromaDB — creation, persistence, and retrieval of embeddings.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import config


class VectorStore:
    """
    Wraps ChromaDB to provide:
      - Creation and population of a vector store from document chunks
      - Loading of an existing persisted store
      - A LangChain-compatible retriever interface

    Why ChromaDB?
      - Runs fully locally with no server process required
      - Persists to disk between sessions
      - Native LangChain integration
      - Efficient ANN search via HNSW index (cosine similarity)
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.persist_dir = config.CHROMA_PERSIST_DIR
        self.collection_name = config.CHROMA_COLLECTION_NAME
        self._store: Chroma | None = None

    def store_exists(self) -> bool:
        """Check whether a persisted ChromaDB collection already exists."""
        chroma_dir = os.path.join(self.persist_dir, self.collection_name)
        return os.path.isdir(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0

    def create_store(self, chunks: List[Document]) -> Chroma:
        """
        Embed all chunks and store them in a new (or replaced) ChromaDB collection.

        Args:
            chunks: Chunked Document objects from the Chunker.

        Returns:
            A populated Chroma vector store instance.
        """
        if not chunks:
            raise ValueError("Cannot create a vector store from an empty chunk list.")

        print(f"[VectorStore] Embedding and storing {len(chunks)} chunks into ChromaDB...")

        self._store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )

        print(f"[VectorStore] Store created and persisted at '{self.persist_dir}'")
        return self._store

    def load_store(self) -> Chroma:
        """
        Load an existing persisted ChromaDB collection from disk.

        Returns:
            A loaded Chroma vector store instance.

        Raises:
            RuntimeError: If no persisted store is found.
        """
        if not self.store_exists():
            raise RuntimeError(
                f"No persisted ChromaDB store found at '{self.persist_dir}'.\n"
                f"Run ingestion first by providing a PDF file path."
            )

        print(f"[VectorStore] Loading existing store from '{self.persist_dir}'...")

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir,
        )

        count = self._store._collection.count()
        print(f"[VectorStore] Loaded store with {count} chunks")
        return self._store

    def get_retriever(self, k: int = config.RETRIEVAL_K):
        """
        Return a LangChain retriever that fetches the top-k similar chunks
        with relevance scores.

        Args:
            k: Number of chunks to retrieve per query.

        Returns:
            A LangChain VectorStoreRetriever configured for similarity search.
        """
        if self._store is None:
            raise RuntimeError("Vector store not initialised. Call create_store() or load_store() first.")

        return self._store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": config.MIN_SIMILARITY_SCORE,
            },
        )

    def similarity_search_with_scores(self, query: str, k: int = config.RETRIEVAL_K):
        """
        Direct similarity search returning (Document, score) tuples.
        Used by the retrieval node to compute confidence.

        Args:
            query: User's natural-language query string.
            k: Number of results to return.

        Returns:
            List of (Document, similarity_score) tuples, sorted by score descending.
        """
        if self._store is None:
            raise RuntimeError("Vector store not initialised.")

        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        return results
