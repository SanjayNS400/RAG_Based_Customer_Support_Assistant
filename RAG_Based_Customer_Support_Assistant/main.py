"""
main.py
-------
RAG-Based Customer Support Assistant
Entry point for the CLI application.

Usage:
    # First run (with a PDF to ingest):
    python main.py --pdf ./data/knowledge_base.pdf

    # Subsequent runs (reuse existing ChromaDB store):
    python main.py

    # Force re-ingestion even if store exists:
    python main.py --pdf ./data/knowledge_base.pdf --reingest
"""

import argparse
import sys
import os
import time

# ── Rich for beautiful CLI output ─────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_sys_path():
    """Add project root to Python path."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

setup_sys_path()

from src.config import config
from src.ingestion import run_ingestion, load_existing_store
from src.llm_handler import LLMHandler
from src.graph import build_graph
from src.state import GraphState


# ─────────────────────────────────────────────────────────────────────────────
# CLI Display Helpers
# ─────────────────────────────────────────────────────────────────────────────

console = Console() if RICH_AVAILABLE else None


def print_banner():
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]RAG Customer Support Assistant[/bold blue]\n"
            "[dim]Powered by LangGraph + ChromaDB + LLM[/dim]\n"
            "[dim]Type 'quit' or 'exit' to stop | 'help' for commands[/dim]",
            border_style="blue",
        ))
    else:
        print("\n" + "=" * 60)
        print("  RAG Customer Support Assistant")
        print("  Powered by LangGraph + ChromaDB + LLM")
        print("  Type 'quit' or 'exit' to stop")
        print("=" * 60)


def print_answer(state: GraphState):
    """Display the final answer with metadata."""
    answer = state.get("final_answer", "No answer generated.")
    sources = state.get("sources", [])
    confidence = state.get("confidence", 0.0)
    escalated = state.get("escalated", False)
    elapsed = state.get("_elapsed", 0)

    if RICH_AVAILABLE:
        tag = "[yellow]🔁 Escalated to Human[/yellow]" if escalated else "[green]🤖 AI Answer[/green]"
        console.print(f"\n{tag}")
        console.print(Panel(
            answer,
            title="Response",
            border_style="green" if not escalated else "yellow",
        ))
        if sources:
            console.print(f"[dim]📄 Sources: {' | '.join(sources)}[/dim]")
        console.print(
            f"[dim]Confidence: {confidence:.0%} | "
            f"Time: {elapsed:.1f}s[/dim]\n"
        )
    else:
        tag = "[ESCALATED]" if escalated else "[AI ANSWER]"
        print(f"\n{tag}")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        if sources:
            print(f"Sources: {' | '.join(sources)}")
        print(f"Confidence: {confidence:.0%} | Time: {elapsed:.1f}s\n")


def print_error(message: str):
    if RICH_AVAILABLE:
        console.print(f"[bold red]Error:[/bold red] {message}")
    else:
        print(f"ERROR: {message}")


def print_info(message: str):
    if RICH_AVAILABLE:
        console.print(f"[dim]{message}[/dim]")
    else:
        print(message)


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG Customer Support Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pdf ./data/support_manual.pdf
  python main.py                             # reuse existing ChromaDB
  python main.py --pdf ./data/faq.pdf --reingest
        """
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the PDF knowledge base file (required on first run).",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Force re-ingestion even if a ChromaDB store already exists.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query in non-interactive mode and exit.",
    )
    return parser.parse_args()


def run_query(graph, query: str) -> GraphState:
    """Execute a single query through the LangGraph pipeline."""
    initial_state: GraphState = {
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

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start
    final_state["_elapsed"] = elapsed

    return final_state


def interactive_loop(graph):
    """Main interactive query loop."""
    print_banner()

    HELP_TEXT = (
        "\nAvailable Commands:\n"
        "  help     - Show this help message\n"
        "  clear    - Clear the screen\n"
        "  quit     - Exit the application\n"
    )

    while True:
        try:
            if RICH_AVAILABLE:
                query = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            else:
                query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! Thank you for using the RAG Support Assistant.")
            break

        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        if query.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        # Process query through LangGraph
        print_info("Thinking...")
        try:
            final_state = run_query(graph, query)
            print_answer(final_state)
        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")


def main():
    args = parse_args()

    # ── Step 1: Initialise or load Vector Store ──────────────────────────────
    vs = None
    store_exists = True

    from src.vector_store import VectorStore
    from src.embedder import get_embedding_model
    _vs_check = VectorStore(None)
    if not _vs_check.store_exists():
        store_exists = False

    if args.reingest and args.pdf:
        print_info("Re-ingesting PDF (--reingest flag set)...")
        vs = run_ingestion(args.pdf)
    elif args.pdf and (not store_exists or args.reingest):
        vs = run_ingestion(args.pdf)
    elif store_exists:
        print_info("Loading existing ChromaDB store...")
        try:
            vs = load_existing_store()
        except RuntimeError as e:
            print_error(str(e))
            if args.pdf:
                print_info("Falling back to ingestion...")
                vs = run_ingestion(args.pdf)
            else:
                print_error("No PDF provided and no existing store found. Use --pdf to specify a PDF.")
                sys.exit(1)
    elif args.pdf:
        vs = run_ingestion(args.pdf)
    else:
        print_error(
            "No ChromaDB store found and no PDF provided.\n"
            "Run: python main.py --pdf ./path/to/your/document.pdf"
        )
        sys.exit(1)

    # ── Step 2: Initialise LLM ───────────────────────────────────────────────
    try:
        llm_handler = LLMHandler()
    except Exception as e:
        print_error(f"Failed to initialise LLM: {e}")
        sys.exit(1)

    # ── Step 3: Build LangGraph ──────────────────────────────────────────────
    try:
        graph = build_graph(vs, llm_handler)
    except Exception as e:
        print_error(f"Failed to build graph: {e}")
        sys.exit(1)

    # ── Step 4: Run ──────────────────────────────────────────────────────────
    if args.query:
        # Single-shot non-interactive mode
        print_info(f"Running single query: '{args.query}'")
        final_state = run_query(graph, args.query)
        print_answer(final_state)
    else:
        # Interactive CLI loop
        interactive_loop(graph)


if __name__ == "__main__":
    main()
