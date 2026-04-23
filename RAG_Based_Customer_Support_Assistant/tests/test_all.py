"""
tests/test_all.py
-----------------
Comprehensive unit and integration tests for the RAG pipeline.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.state import GraphState
from src.nodes import (
    input_node, retrieval_node, llm_node, router_node,
    output_node, hitl_node, error_node, set_dependencies,
)
from src.chunker import Chunker
from src.config import config


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_state(**kwargs) -> GraphState:
    """Create a default GraphState with optional overrides."""
    base: GraphState = {
        "query": "How do I reset my password?",
        "retrieved_docs": [],
        "chunks_text": [],
        "sources": [],
        "similarity_scores": [],
        "confidence": 0.0,
        "has_context": False,
        "llm_answer": None,
        "final_answer": None,
        "escalated": False,
        "error": None,
    }
    base.update(kwargs)
    return base


def make_doc(text="Sample text", page=1, source="kb.pdf") -> Document:
    return Document(
        page_content=text,
        metadata={"page": page, "source_file": source},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DocumentProcessor
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentProcessor:
    def test_validate_nonexistent_file(self):
        from src.document_processor import DocumentProcessor
        dp = DocumentProcessor()
        with pytest.raises(FileNotFoundError):
            dp.validate_file("/nonexistent/path/file.pdf")

    def test_validate_wrong_extension(self, tmp_path):
        from src.document_processor import DocumentProcessor
        f = tmp_path / "document.txt"
        f.write_text("hello")
        dp = DocumentProcessor()
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            dp.validate_file(str(f))

    def test_validate_empty_file(self, tmp_path):
        from src.document_processor import DocumentProcessor
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        dp = DocumentProcessor()
        with pytest.raises(ValueError, match="empty"):
            dp.validate_file(str(f))


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Chunker
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        """Chunks are created from documents."""
        docs = [make_doc("word " * 200)]
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        """Each chunk retains source metadata."""
        docs = [make_doc("word " * 200, page=3, source="faq.pdf")]
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(docs)
        for chunk in chunks:
            assert chunk.metadata.get("source_file") == "faq.pdf"
            assert "chunk_id" in chunk.metadata

    def test_chunk_id_unique(self):
        """Every chunk gets a unique chunk_id."""
        docs = [make_doc("word " * 500)]
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_documents(docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_documents_raises(self):
        chunker = Chunker()
        with pytest.raises(ValueError):
            chunker.chunk_documents([])

    def test_overlap_less_than_size(self):
        """chunk_overlap must be less than chunk_size to be meaningful."""
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_overlap < chunker.chunk_size


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Input Node
# ─────────────────────────────────────────────────────────────────────────────

class TestInputNode:
    def test_valid_query_passes_through(self):
        state = make_state(query="What is your refund policy?")
        result = input_node(state)
        assert result["query"] == "What is your refund policy?"
        assert result["error"] is None
        assert result["escalated"] is False

    def test_empty_query_sets_error(self):
        state = make_state(query="")
        result = input_node(state)
        assert result["error"] is not None
        assert "empty" in result["error"].lower()

    def test_whitespace_only_query_sets_error(self):
        state = make_state(query="   ")
        result = input_node(state)
        assert result["error"] is not None

    def test_initialises_all_fields(self):
        state = make_state(query="Hello?")
        result = input_node(state)
        assert result["retrieved_docs"] == []
        assert result["chunks_text"] == []
        assert result["confidence"] == 0.0
        assert result["has_context"] is False
        assert result["escalated"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Retrieval Node
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalNode:
    def setup_method(self):
        """Set up a mock vector store."""
        self.mock_vs = MagicMock()
        self.mock_llm = MagicMock()
        set_dependencies(self.mock_vs, self.mock_llm)

    def test_successful_retrieval(self):
        doc = make_doc("To reset your password, click Forgot Password.")
        self.mock_vs.similarity_search_with_scores.return_value = [(doc, 0.88)]
        state = make_state(query="How to reset password?")
        result = retrieval_node(state)
        assert result["has_context"] is True
        assert result["confidence"] > 0
        assert len(result["retrieved_docs"]) == 1
        assert len(result["sources"]) == 1

    def test_no_results_above_threshold(self):
        self.mock_vs.similarity_search_with_scores.return_value = [
            (make_doc("Unrelated text"), 0.20)   # below MIN_SIMILARITY_SCORE=0.40
        ]
        state = make_state(query="unknown query")
        result = retrieval_node(state)
        assert result["has_context"] is False
        assert result["confidence"] == 0.0
        assert result["retrieved_docs"] == []

    def test_skips_on_upstream_error(self):
        state = make_state(query="hello", error="upstream error")
        result = retrieval_node(state)
        assert result["error"] == "upstream error"   # unchanged

    def test_confidence_computed_from_top3(self):
        docs_scores = [
            (make_doc("A"), 0.90),
            (make_doc("B"), 0.80),
            (make_doc("C"), 0.70),
            (make_doc("D"), 0.60),
        ]
        self.mock_vs.similarity_search_with_scores.return_value = docs_scores
        state = make_state(query="test")
        result = retrieval_node(state)
        expected = (0.90 + 0.80 + 0.70) / 3
        assert abs(result["confidence"] - expected) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Router
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_routes_to_output_on_high_confidence(self):
        state = make_state(
            confidence=0.85, has_context=True,
            llm_answer="Here is your answer.",
            query="normal question"
        )
        assert router_node(state) == "output"

    def test_routes_to_hitl_on_low_confidence(self):
        state = make_state(
            confidence=0.50, has_context=True,
            llm_answer="Here is your answer.",
            query="normal question"
        )
        assert router_node(state) == "hitl"

    def test_routes_to_hitl_on_no_context(self):
        state = make_state(
            confidence=0.85, has_context=False,
            llm_answer="I cannot find this.",
            query="normal question"
        )
        assert router_node(state) == "hitl"

    def test_routes_to_hitl_on_complex_keyword(self):
        state = make_state(
            confidence=0.90, has_context=True,
            llm_answer="Here is your answer.",
            query="I want to file a legal complaint"
        )
        assert router_node(state) == "hitl"

    def test_routes_to_hitl_on_llm_uncertainty(self):
        state = make_state(
            confidence=0.85, has_context=True,
            llm_answer="I cannot find sufficient information to answer this.",
            query="normal question"
        )
        assert router_node(state) == "hitl"

    def test_routes_to_error_on_hard_error(self):
        state = make_state(
            confidence=0.0, has_context=False,
            llm_answer=None,
            error="ChromaDB connection failed",
            query="test"
        )
        assert router_node(state) == "error_node"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: LLM Node
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMNode:
    def setup_method(self):
        self.mock_vs = MagicMock()
        self.mock_llm = MagicMock()
        set_dependencies(self.mock_vs, self.mock_llm)

    def test_generates_answer_when_context_present(self):
        self.mock_llm.generate_answer.return_value = "Your password can be reset via..."
        state = make_state(
            query="reset password",
            has_context=True,
            retrieved_docs=[make_doc("Reset instructions here.")],
        )
        result = llm_node(state)
        assert result["llm_answer"] == "Your password can be reset via..."

    def test_skips_llm_when_no_context(self):
        state = make_state(has_context=False, retrieved_docs=[])
        result = llm_node(state)
        assert "cannot find" in result["llm_answer"].lower()
        self.mock_llm.generate_answer.assert_not_called()

    def test_handles_llm_failure_gracefully(self):
        self.mock_llm.generate_answer.side_effect = RuntimeError("API timeout")
        state = make_state(has_context=True, retrieved_docs=[make_doc("text")])
        result = llm_node(state)
        assert result["llm_answer"] is not None
        assert result["error"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Output Node
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputNode:
    def test_promotes_llm_answer_to_final(self):
        state = make_state(llm_answer="Your answer is X.", final_answer=None)
        result = output_node(state)
        assert result["final_answer"] == "Your answer is X."

    def test_keeps_hitl_final_answer(self):
        state = make_state(
            llm_answer="LLM draft",
            final_answer="Human approved answer",
            escalated=True
        )
        result = output_node(state)
        assert result["final_answer"] == "Human approved answer"

    def test_handles_hard_error(self):
        state = make_state(error="DB crash", llm_answer=None, final_answer=None)
        result = output_node(state)
        assert "error occurred" in result["final_answer"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_confidence_threshold_in_range(self):
        assert 0.0 <= config.CONFIDENCE_THRESHOLD <= 1.0

    def test_chunk_overlap_less_than_chunk_size(self):
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_retrieval_k_positive(self):
        assert config.RETRIEVAL_K > 0

    def test_complexity_keywords_is_list(self):
        assert isinstance(config.COMPLEXITY_KEYWORDS, list)
        assert len(config.COMPLEXITY_KEYWORDS) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
