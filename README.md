# agentic-rag-for-review

Personal review version of an Agentic RAG system for course and document Q&A. The current codebase focuses on a local Gradio app backed by PostgreSQL + pgvector, LangGraph orchestration, hybrid retrieval, reranking, document ingestion, and evaluation scripts.

## What It Does

- Upload local PDF, Markdown, Word, or PowerPoint files from the Gradio UI.
- Convert documents to Markdown, clean repeated headers/footers, extract structure, and split content into parent and child chunks.
- Store parent chunks, child chunks, metadata, dense vectors, and sparse search fields in PostgreSQL with pgvector.
- Retrieve with dense search, PostgreSQL full-text sparse search, and reciprocal rank fusion.
- Rerank candidates with a local cross-encoder when the model files are available.
- Select answer context adaptively from child chunks, neighboring child chunks, or full parent chunks.
- Run a LangGraph agent flow for intent recognition, query rewriting, clarification, task planning, retrieval, answer evaluation, fallback, and aggregation.
- Scope chat to a course and maintain lightweight session memory.
- Run RAGBench/RAGAS-oriented evaluation scripts under `project/evaluation`.

## Architecture

```text
Gradio UI
-> RagApplication
-> RAGSystem
-> LangGraph agent graph
-> rag_research tool
-> RetrievalPipeline
-> PostgreSQL + pgvector
```

Main code paths:

- `project/app.py` starts the Gradio app.
- `project/application/rag_application.py` wires the RAG system, document manager, and chat interface.
- `project/core/rag_system.py` initializes PostgreSQL storage, the DeepSeek/OpenAI-compatible chat model, LangGraph, and tools.
- `project/ingestion/` handles conversion, cleaning, chunking, integrity checks, manifests, image extraction, and course structure.
- `project/storage/` contains pgvector-backed child chunk search and parent chunk storage.
- `project/retrieval/` contains RRF fusion, reranking, and adaptive context selection.
- `project/rag_agent/` contains graph state, prompts, tools, nodes, and routing logic.
- `project/evaluation/` contains RAGBench import/evaluation runners and reporting utilities.

## Tech Stack

- Python 3.11+
- Gradio
- LangGraph / LangChain
- DeepSeek or another OpenAI-compatible chat API through `ChatOpenAI`
- PostgreSQL 17 with pgvector
- SQLAlchemy and Alembic
- HuggingFace sentence-transformer embeddings
- PostgreSQL full-text search with `jieba` tokenization
- Cross-encoder reranking via `sentence-transformers`
- MarkItDown, PyMuPDF, and related document conversion tools
- Optional Langfuse tracing

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a root `.env` file:

```bash
cp .env.example .env
```

Set at least:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
```

Optional project-level settings for Langfuse and multimodal ingestion are documented in `project/.env.example`.

## Database

Start PostgreSQL with pgvector:

```bash
docker compose up -d postgres
```

Apply the schema migration:

```bash
alembic upgrade head
```

The default database URL is:

```text
postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag
```

You can override it with `DATABASE_URL`.

## Run

```bash
python project/app.py
```

The UI has two main tabs:

- `Documents`: upload files, bind them to courses, inspect indexed documents, rename courses or sections, and clear the knowledge base.
- `Chat`: ask questions against all indexed documents or a selected course scope.

Runtime outputs are written under `runtime/`, including converted Markdown, cleaned Markdown, ingestion logs, image extracts, course structure, session memory, and evaluation reports.

## Retrieval Behavior

The default retrieval mode is `rrf`:

```text
dense vector search + sparse full-text search -> reciprocal rank fusion -> rerank -> context selection
```

Context selection is controlled by:

- `RETRIEVAL_CONTEXT_POLICY`: `adaptive`, `child`, `neighbor`, or `parent`
- `RETRIEVAL_NEIGHBOR_WINDOW`
- `RETRIEVAL_PARENT_EXPAND_MIN_HITS`

In adaptive mode, broad explanatory queries prefer parent context, fact-style queries can stay closer to child chunks, and repeated hits within the same parent can expand to neighboring child chunks.

## Evaluation

Evaluation utilities live under `project/evaluation`. The RAGBench/RAGAS runner can generate answers, score them, and write outputs under `runtime/evaluation_reports`.

Example:

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 50 \
  --output-dir runtime/evaluation_reports/ragbench_covidqa_test_50 \
  --ragas-max-workers 1 \
  --ragas-batch-size 1
```

See `project/evaluation/README.md` for evaluation-specific details.

## Notes

- The repository intentionally ignores local runtime data, local documents, notebooks, internal task notes, and local agent configuration.
- If a path is already tracked by Git, adding it to `.gitignore` does not remove it from GitHub. It must also be removed from the Git index with `git rm --cached`.
