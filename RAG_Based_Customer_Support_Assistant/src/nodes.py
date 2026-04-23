"""
nodes.py
--------
Module 7 & 8: Graph Execution + HITL
All LangGraph node functions.
Each function receives the current GraphState and returns an updated GraphState dict.
"""

from typing import List
from langchain_core.documents import Document

from src.state import GraphState
from src.config import config


# ─────────────────────────────────────────────────────────────────────────────
# These are injected at graph build time to avoid circular imports
# ─────────────────────────────────────────────────────────────────────────────
_vector_store = None
_llm_handler = None


def set_dependencies(vector_store, llm_handler):
    """Inject shared instances into this module."""
    global _vector_store, _llm_handler
    _vector_store = vector_store
    _llm_handler = llm_handler


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: Input Node
# ─────────────────────────────────────────────────────────────────────────────
def input_node(state: GraphState) -> dict:
    """
    Validates the user query and initialises all state fields to defaults.
    Entry point of the LangGraph workflow.
    """
    query = state.get("query", "").strip()

    if not query:
        return {
            **state,
            "error": "Query cannot be empty. Please enter a valid question.",
            "has_context": False,
            "confidence": 0.0,
            "escalated": False,
            "retrieved_docs": [],
            "chunks_text": [],
            "sources": [],
            "similarity_scores": [],
            "llm_answer": None,
            "final_answer": None,
        }

    print(f"\n[InputNode] Processing query: '{query}'")

    return {
        **state,
        "query": query,
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


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: Retrieval Node
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_node(state: GraphState) -> dict:
    """
    Performs semantic similarity search in ChromaDB.
    Computes a confidence score from the retrieved similarity scores.
    Sets has_context based on whether meaningful results were found.
    """
    # Early exit if upstream error
    if state.get("error"):
        return state

    query = state["query"]
    print(f"[RetrievalNode] Searching ChromaDB for: '{query}'")

    try:
        results = _vector_store.similarity_search_with_scores(query, k=config.RETRIEVAL_K)
    except Exception as e:
        return {
            **state,
            "error": f"Retrieval failed: {e}",
            "has_context": False,
            "confidence": 0.0,
        }

    # Filter results below minimum similarity threshold
    filtered = [(doc, score) for doc, score in results if score >= config.MIN_SIMILARITY_SCORE]

    if not filtered:
        print(f"[RetrievalNode] No chunks above threshold ({config.MIN_SIMILARITY_SCORE})")
        return {
            **state,
            "retrieved_docs": [],
            "chunks_text": [],
            "sources": [],
            "similarity_scores": [],
            "confidence": 0.0,
            "has_context": False,
        }

    docs = [doc for doc, _ in filtered]
    scores = [score for _, score in filtered]

    # Confidence = mean of top-3 scores (or all if fewer than 3)
    top_scores = sorted(scores, reverse=True)[:3]
    confidence = sum(top_scores) / len(top_scores)

    # Source citations
    sources = []
    for doc in docs:
        source = doc.metadata.get("source_file", "document")
        page = doc.metadata.get("page", 0)
        citation = f"{source} (page {page + 1})"
        if citation not in sources:
            sources.append(citation)

    print(
        f"[RetrievalNode] Retrieved {len(docs)} chunks | "
        f"confidence={confidence:.2f} | sources={sources}"
    )

    return {
        **state,
        "retrieved_docs": docs,
        "chunks_text": [doc.page_content for doc in docs],
        "sources": sources,
        "similarity_scores": scores,
        "confidence": round(confidence, 4),
        "has_context": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: LLM Node
# ─────────────────────────────────────────────────────────────────────────────
def llm_node(state: GraphState) -> dict:
    """
    Calls the LLM with the user query and retrieved chunks.
    Stores the generated answer in state.llm_answer.
    If no context was found, returns an uncertainty signal instead of calling LLM.
    """
    if state.get("error"):
        return state

    # Skip LLM call if no context was retrieved — save API costs
    if not state["has_context"]:
        print("[LLMNode] No context retrieved — skipping LLM call")
        return {
            **state,
            "llm_answer": "I cannot find sufficient information in our knowledge base to answer this question.",
        }

    query = state["query"]
    docs = state["retrieved_docs"]

    print(f"[LLMNode] Generating answer with {len(docs)} context chunks...")

    try:
        answer = _llm_handler.generate_answer(query, docs)
        print(f"[LLMNode] Answer generated ({len(answer)} chars)")
        return {**state, "llm_answer": answer, "error": None}
    except RuntimeError as e:
        print(f"[LLMNode] LLM failed: {e}")
        return {
            **state,
            "llm_answer": "I cannot find sufficient information in our knowledge base to answer this question.",
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Edge: Router
# ─────────────────────────────────────────────────────────────────────────────
def router_node(state: GraphState) -> str:
    """
    Conditional edge function — determines the next node.

    Routing Logic:
        → "output" : High confidence + context found + no complexity signals
        → "hitl"   : Low confidence OR missing context OR complex query OR LLM uncertain
        → "error"  : A hard error occurred upstream

    Returns:
        String name of the next node.
    """
    if state.get("error") and not state.get("llm_answer"):
        return "error_node"

    confidence = state.get("confidence", 0.0)
    has_context = state.get("has_context", False)
    llm_answer = (state.get("llm_answer") or "").lower()
    query_lower = state.get("query", "").lower()

    # Check for complexity keywords
    is_complex = any(kw in query_lower for kw in config.COMPLEXITY_KEYWORDS)

    # Check for LLM uncertainty signals
    is_uncertain = any(phrase in llm_answer for phrase in config.UNCERTAINTY_PHRASES)

    # Routing decision
    low_confidence = confidence < config.CONFIDENCE_THRESHOLD
    no_context = not has_context

    if low_confidence or no_context or is_complex or is_uncertain:
        reason = []
        if low_confidence:  reason.append(f"low confidence ({confidence:.2f})")
        if no_context:      reason.append("no relevant context")
        if is_complex:      reason.append("complex/sensitive keywords detected")
        if is_uncertain:    reason.append("LLM expressed uncertainty")
        print(f"[Router] → HITL | Reasons: {', '.join(reason)}")
        return "hitl"
    else:
        print(f"[Router] → Output | confidence={confidence:.2f}, context=True")
        return "output"


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: HITL Node
# ─────────────────────────────────────────────────────────────────────────────
def hitl_node(state: GraphState) -> dict:
    """
    Human-in-the-Loop escalation node.
    Presents the query, context, and LLM draft to a human agent.
    Collects and injects the human-approved answer back into state.
    """
    print("\n" + "=" * 65)
    print("  ⚠  ESCALATION: Human Review Required")
    print("=" * 65)
    print(f"  Customer Query  : {state['query']}")
    print(f"  Confidence Score: {state['confidence']:.2f} "
          f"(threshold: {config.CONFIDENCE_THRESHOLD})")
    print(f"  Sources Found   : {', '.join(state['sources']) or 'None'}")
    print(f"\n  LLM Draft Answer:\n  {state.get('llm_answer', 'N/A')}")
    print("-" * 65)
    print("  Please review the above and provide the correct response.")
    print("  (Press Enter without typing to accept the LLM draft)")
    print("-" * 65)

    try:
        human_answer = input("  Your Response: ").strip()
    except (EOFError, KeyboardInterrupt):
        # Non-interactive environment — use LLM draft as fallback
        human_answer = ""
        print("\n  [HITL] Non-interactive mode: using LLM draft as fallback")

    if not human_answer:
        human_answer = state.get("llm_answer") or "I'm sorry, I was unable to find an answer. Please contact support."
        print(f"  [HITL] Using LLM draft answer.")
    else:
        print(f"  [HITL] Human-approved answer recorded.")

    print("=" * 65 + "\n")

    return {
        **state,
        "final_answer": human_answer,
        "escalated": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: Output Node
# ─────────────────────────────────────────────────────────────────────────────
def output_node(state: GraphState) -> dict:
    """
    Finalises the answer by copying llm_answer → final_answer (if not already set by HITL).
    The main.py display logic reads state.final_answer for the CLI output.
    """
    if state.get("error") and not state.get("llm_answer"):
        final = f"An error occurred: {state['error']}"
    elif not state.get("final_answer"):
        # Not escalated — promote LLM answer to final
        final = state.get("llm_answer") or "I'm sorry, I was unable to generate a response."
    else:
        # Escalated — final_answer already set by hitl_node
        final = state["final_answer"]

    return {**state, "final_answer": final}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: Error Node
# ─────────────────────────────────────────────────────────────────────────────
def error_node(state: GraphState) -> dict:
    """
    Handles hard errors (e.g., LLM API failure, ChromaDB crash).
    Returns a graceful error message to the user.
    """
    error_msg = state.get("error", "An unknown error occurred.")
    print(f"[ErrorNode] Hard error: {error_msg}")
    return {
        **state,
        "final_answer": (
            "I'm sorry, the system encountered a technical issue and could not process your request. "
            "Please try again or contact support directly."
        ),
        "escalated": True,
    }
