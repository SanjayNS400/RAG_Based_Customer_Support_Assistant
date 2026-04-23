"""
chunker.py
----------
Module 2: Chunking
Splits Document objects into overlapping text chunks for embedding.
"""

from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import config


class Chunker:
    """
    Splits raw Document pages into smaller, overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    - Tries to split on paragraph boundaries first (\n\n),
      then sentence boundaries (\n), then word boundaries (" ").
    - This preserves semantic coherence better than fixed character splits.
    - The overlap ensures context spanning chunk boundaries is not lost.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Document pages into overlapping chunks.

        Each chunk inherits the parent document's metadata and gets
        an additional 'chunk_id' and 'chunk_index' field.

        Args:
            documents: List of full-page Document objects from DocumentProcessor.

        Returns:
            List of chunk-level Document objects.

        Raises:
            ValueError: If documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot chunk an empty document list.")

        chunks = self.splitter.split_documents(documents)

        # Annotate each chunk with a unique ID and sequential index
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source_file", "doc")
            page = chunk.metadata.get("page", 0)
            chunk.metadata["chunk_id"] = f"{source}_p{page}_c{i}"
            chunk.metadata["chunk_index"] = i

        print(
            f"[Chunker] Split {len(documents)} pages → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
