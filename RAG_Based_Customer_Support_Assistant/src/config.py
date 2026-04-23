"""
config.py
---------
Centralised configuration loader for the RAG Customer Support Assistant.
Reads from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LLM ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # If no OpenAI key is set, the system falls back to local Ollama
    USE_OPENAI: bool = bool(os.getenv("OPENAI_API_KEY", "").strip())
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"          # cost-efficient GPT-4 class model
    OLLAMA_LLM_MODEL: str = "mistral"               # local fallback

    # ── Embeddings ─────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    # "openai" or "huggingface"
    OPENAI_EMBED_MODEL: str = "text-embedding-ada-002"
    HF_EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Chunking ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # ── Retrieval ──────────────────────────────────────────────────────────
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", 4))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))
    MIN_SIMILARITY_SCORE: float = 0.40   # below this, chunk is ignored entirely

    # ── ChromaDB ───────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "support_kb")

    # ── HITL ───────────────────────────────────────────────────────────────
    HITL_MODE: str = os.getenv("HITL_MODE", "sync")   # "sync" or "async"

    # ── Complexity keywords that always trigger HITL ────────────────────────
    COMPLEXITY_KEYWORDS: list = [
        "legal", "lawsuit", "sue", "court", "attorney",
        "complaint", "refund dispute", "chargeback",
        "escalate", "manager", "supervisor", "fraud",
        "stolen", "police", "harassment",
    ]

    # ── LLM uncertainty phrases that trigger HITL ──────────────────────────
    UNCERTAINTY_PHRASES: list = [
        "i cannot find",
        "i'm not sure",
        "i am not sure",
        "insufficient information",
        "not enough information",
        "i don't have information",
        "i do not have information",
        "cannot answer",
        "outside my knowledge",
    ]


config = Config()
