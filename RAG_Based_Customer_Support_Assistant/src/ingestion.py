"""
ingestion.py
------------
Ingestion Pipeline Runner
Orchestrates: PDF Load → Chunk → Embed → Store in ChromaDB

Run this once (or whenever the knowledge base PDF is updated).
The resulting ChromaDB store is persisted to disk and reused by the query pipeline.
"""

from src.document_processor import DocumentProcessor
from src.chunker import Chunker
from src.embedder import get_embedding_model
from src.vector_store import VectorStore
from src.config import config


def run_ingestion(pdf_path: str) -> VectorStore:
    """
    Full ingestion pipeline: PDF → ChromaDB.

    Args:
        pdf_path: Path to the PDF knowledge base file.

    Returns:
        An initialised VectorStore instance ready for querying.
    """
    print("\n" + "=" * 60)
    print("  RAG Ingestion Pipeline Starting")
    print("=" * 60)

    # Step 1: Load PDF
    processor = DocumentProcessor()
    documents = processor.load_pdf(pdf_path)

    # Step 2: Chunk documents
    chunker = Chunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(documents)

    # Step 3: Load embedding model
    embedding_model = get_embedding_model()

    # Step 4: Store in ChromaDB
    vs = VectorStore(embedding_model)
    vs.create_store(chunks)

    print("=" * 60)
    print("  Ingestion Complete ✓")
    print(f"  {len(chunks)} chunks stored in ChromaDB at '{config.CHROMA_PERSIST_DIR}'")
    print("=" * 60 + "\n")

    return vs


def load_existing_store() -> VectorStore:
    """
    Load an already-ingested ChromaDB store without re-processing the PDF.

    Returns:
        An initialised VectorStore instance.

    Raises:
        RuntimeError: If no persisted store is found.
    """
    embedding_model = get_embedding_model()
    vs = VectorStore(embedding_model)
    vs.load_store()
    return vs
