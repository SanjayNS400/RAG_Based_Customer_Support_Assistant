"""
embedder.py
-----------
Module 3: Embedding
Provides a unified embedding interface supporting both
OpenAI (text-embedding-ada-002) and HuggingFace (sentence-transformers).
"""

from src.config import config


def get_embedding_model():
    """
    Factory function: returns the appropriate LangChain Embeddings object
    based on the EMBEDDING_PROVIDER config setting.

    Returns:
        A LangChain-compatible Embeddings object.

    Notes:
        - Both query and document chunks MUST use the same embedding model.
        - Mixing models produces incomparable vectors and breaks retrieval.
    """
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise EnvironmentError(
                "EMBEDDING_PROVIDER is set to 'openai' but OPENAI_API_KEY is not set.\n"
                "Either set OPENAI_API_KEY in your .env file or change "
                "EMBEDDING_PROVIDER to 'huggingface'."
            )
        from langchain_openai import OpenAIEmbeddings
        model = OpenAIEmbeddings(
            model=config.OPENAI_EMBED_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
        print(f"[Embedder] Using OpenAI embeddings: {config.OPENAI_EMBED_MODEL}")
        return model

    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = HuggingFaceEmbeddings(
            model_name=config.HF_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"[Embedder] Using HuggingFace embeddings: {config.HF_EMBED_MODEL}")
        return model

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
            f"Choose 'openai' or 'huggingface'."
        )
