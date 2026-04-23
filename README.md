# RAG-Based Customer Support Assistant

**Innomatics Research Labs — Internship Project**

A production-style Retrieval-Augmented Generation (RAG) system with LangGraph orchestration and Human-in-the-Loop (HITL) escalation.

---

## Project Structure

```
rag_project/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
│
├── src/
│   ├── config.py            # Centralised configuration
│   ├── document_processor.py# Module 1: PDF loading
│   ├── chunker.py           # Module 2: Text chunking
│   ├── embedder.py          # Module 3: Embedding model factory
│   ├── vector_store.py      # Module 4: ChromaDB management
│   ├── llm_handler.py       # Module 5: LLM + prompt management
│   ├── state.py             # LangGraph GraphState definition
│   ├── nodes.py             # All LangGraph node functions
│   ├── graph.py             # LangGraph graph assembly
│   └── ingestion.py         # Ingestion pipeline runner
│
├── tests/
│   └── test_all.py          # Full test suite (pytest)
│
├── data/                    # Put your PDF knowledge base here
└── chroma_db/               # Auto-created: persisted vector store
```

---

## Setup

### 1. Get into the folder

```bash
cd rag_project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

**Minimum required `.env` for a free, fully local setup:**

```env
EMBEDDING_PROVIDER=huggingface
# No API key needed — uses sentence-transformers locally
# Make sure Ollama is running for the LLM:
#   brew install ollama && ollama serve && ollama pull mistral
```

### 5. Install Ollama for Local LLM

```bash
# macOS
brew install ollama
ollama serve
ollama pull mistral

# Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

---

## Running the Application

### First Run (with PDF ingestion)

```bash
python main.py --pdf ./data/your_knowledge_base.pdf
```

### Subsequent Runs (reuse existing ChromaDB)

```bash
python main.py
```

### Force Re-ingestion

```bash
python main.py --pdf ./data/updated_kb.pdf --reingest
```

---

## Example Session

```
╭─────────────────────────────────────────────────────╮
│  RAG Customer Support Assistant                     │
│  Powered by LangGraph + ChromaDB + LLM              │
│  Type 'quit' or 'exit' to stop | 'help' for commands│
╰─────────────────────────────────────────────────────╯

You: How do I track my order?

Thinking...
[RetrievalNode] Retrieved 4 chunks | confidence=0.87
[Router] → Output | confidence=0.87, context=True

🤖 AI Answer
╭─ Response ──────────────────────────────────────────╮
│ You can track your order by logging into your       │
│ account and visiting the 'My Orders' section.       │
│ Enter your order number to see real-time updates.   │
╰─────────────────────────────────────────────────────╯
📄 Sources: knowledge_base.pdf (page 4)
Confidence: 87% | Time: 2.3s

You: I want to file a legal complaint

[Router] → HITL | Reasons: complex/sensitive keywords detected

═══════════════════════════════════════════════════════════
  ⚠  ESCALATION: Human Review Required
═══════════════════════════════════════════════════════════
  Customer Query  : I want to file a legal complaint
  Confidence Score: 0.85 (threshold: 0.70)
  ...
  Your Response: Please email legal@company.com with your case details.
═══════════════════════════════════════════════════════════

🔁 Escalated to Human
╭─ Response ──────────────────────────────────────────╮
│ Please email legal@company.com with your case       │
│ details and we will respond within 2 business days. │
╰─────────────────────────────────────────────────────╯

You: quit
Goodbye!
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## How It Works

### Ingestion Pipeline (runs once)

```
PDF → DocumentProcessor → Chunker → Embedder → ChromaDB
```

### Query Pipeline (LangGraph, runs per query)

```
User Query
    │
    ▼
input_node          ← Validates query, initialises state
    │
    ▼
retrieval_node      ← Embeds query, searches ChromaDB, scores confidence
    │
    ▼
llm_node            ← Builds prompt, calls LLM, stores answer
    │
    ▼
router (edge)       ← Confidence ≥ 0.70 AND no complexity? → output
    │                  Otherwise → hitl
    ├─── output_node ← Formats and returns answer
    │
    └─── hitl_node   ← Human reviews, edits, approves → output_node
```

### Routing Decision Logic

| Condition                                                                    | Route         |
| ---------------------------------------------------------------------------- | ------------- |
| confidence ≥ 0.70 AND has_context AND no complex keywords AND LLM is certain | `output_node` |
| confidence < 0.70                                                            | `hitl_node`   |
| No relevant chunks found                                                     | `hitl_node`   |
| Query contains sensitive keywords (legal, complaint, etc.)                   | `hitl_node`   |
| LLM expresses uncertainty                                                    | `hitl_node`   |

---

## Configuration Reference (`.env`)

| Variable               | Default       | Description                        |
| ---------------------- | ------------- | ---------------------------------- |
| `OPENAI_API_KEY`       | _(blank)_     | OpenAI key; leave blank for Ollama |
| `EMBEDDING_PROVIDER`   | `huggingface` | `openai` or `huggingface`          |
| `CHUNK_SIZE`           | `500`         | Max characters per chunk           |
| `CHUNK_OVERLAP`        | `50`          | Overlap between chunks             |
| `RETRIEVAL_K`          | `4`           | Number of chunks to retrieve       |
| `CONFIDENCE_THRESHOLD` | `0.70`        | Minimum score before HITL          |
| `HITL_MODE`            | `sync`        | `sync` (blocking) or `async`       |
| `CHROMA_PERSIST_DIR`   | `./chroma_db` | ChromaDB storage path              |

---
