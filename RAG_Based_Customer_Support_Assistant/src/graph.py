"""
graph.py
--------
Module 7: Graph Execution
Assembles and compiles the LangGraph StateGraph.

Graph Structure:
    START
      │
      ▼
  input_node
      │
      ▼
  retrieval_node
      │
      ▼
  llm_node
      │
      ▼  (conditional edge)
  router_node ──────────────────┐
      │ "output"                │ "hitl"
      ▼                         ▼
  output_node             hitl_node
      │                         │
      ▼                         ▼
     END                   output_node
                                │
                                ▼
                               END
"""

from langgraph.graph import StateGraph, END

from src.state import GraphState
from src.nodes import (
    input_node,
    retrieval_node,
    llm_node,
    router_node,
    hitl_node,
    output_node,
    error_node,
    set_dependencies,
)


def build_graph(vector_store, llm_handler):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        vector_store : Initialised VectorStore instance.
        llm_handler  : Initialised LLMHandler instance.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    # Inject dependencies into the nodes module
    set_dependencies(vector_store, llm_handler)

    # ── Create the StateGraph ────────────────────────────────────────────────
    graph = StateGraph(GraphState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("input",      input_node)
    graph.add_node("retrieval",  retrieval_node)
    graph.add_node("llm",        llm_node)
    graph.add_node("output",     output_node)
    graph.add_node("hitl",       hitl_node)
    graph.add_node("error_node", error_node)

    # ── Set entry point ──────────────────────────────────────────────────────
    graph.set_entry_point("input")

    # ── Fixed edges ──────────────────────────────────────────────────────────
    graph.add_edge("input",     "retrieval")
    graph.add_edge("retrieval", "llm")

    # ── Conditional edge (router) ─────────────────────────────────────────────
    graph.add_conditional_edges(
        "llm",
        router_node,
        {
            "output":     "output",
            "hitl":       "hitl",
            "error_node": "error_node",
        },
    )

    # ── After HITL → always go to output ─────────────────────────────────────
    graph.add_edge("hitl",       "output")
    graph.add_edge("error_node", "output")

    # ── Terminal edge ─────────────────────────────────────────────────────────
    graph.add_edge("output", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    compiled = graph.compile()
    print("[Graph] LangGraph compiled successfully")
    return compiled
