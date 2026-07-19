# Agentic RAG Developer Notes

This project is an Agentic RAG system for local course material and document question answering. The current main application is a FastAPI/Uvicorn service started by `python project/app.py`; it serves the static browser UI, internal `/api/*` routes, and Server-Sent Events chat streaming.

## Overview

Current capabilities:

- Upload PDF, Markdown, Word, and PowerPoint files from the browser UI.
- Convert documents to Markdown, clean repeated headers/footers, split into parent and child chunks, and index them.
- Store parent chunks, child chunks, metadata, dense vectors, and sparse retrieval fields in PostgreSQL + pgvector.
- Retrieve with dense vector search, PostgreSQL full-text search, RRF fusion, and optional cross-encoder reranking.
- Select child, neighboring child, or full parent context according to query shape and retrieval hits.
- Use LangGraph to orchestrate history summarization, intent recognition, query rewriting, clarification, task planning, retrieval, answer evaluation, fallback answers, and aggregation.
- Scope chat to a course, manage courses/sections, and persist lightweight session memory in SQLite.
- Run RAGBench, RAGAS, local retrieval, and chunking ablation evaluation scripts under `project/evaluation`.

## Quick Start

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start PostgreSQL with pgvector:

```bash
docker compose up -d postgres
```

Create configuration:

```bash
cp project/.env.example project/.env
```

Set at least:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
```

Start the application:

```bash
python project/app.py
```

Open:

```text
http://localhost:7860
```

To run the app and database with Docker:

```bash
cp project/.env.example project/.env
# Fill DEEPSEEK_API_KEY in project/.env
# The first run installs Python dependencies; Hugging Face model files are cached in the hf_cache volume.
docker compose up --build app
```

Configuration is loaded from the repository root `.env` first, then from `project/.env` with override behavior. The Docker image excludes `.env` files; `docker-compose.yml` passes `project/.env` at runtime when present and overrides `DATABASE_URL` to use the Compose `postgres` service.

## Architecture

```text
Browser static UI
-> FastAPI /api routes
-> RagApplication
-> RAGSystem
-> LangGraph agent graph
-> rag_research tool
-> RetrievalPipeline
-> PostgreSQL + pgvector
```

Important paths:

- `project/app.py`: main entrypoint; starts Uvicorn on port `7860`.
- `project/server.py`: FastAPI app; serves `/`, `/static`, and `/api`.
- `project/static/`: current browser UI.
- `project/api/`: document, course, session, task, and streaming chat routes.
- `project/application/rag_application.py`: wires the RAG system, document manager, and chat interface.
- `project/core/rag_system.py`: initializes storage, `ChatOpenAI`, LangGraph, and retrieval tools.
- `project/ingestion/`: conversion, Markdown cleaning, chunking, index manifest, file integrity, and course structure.
- `project/storage/`: PostgreSQL connection handling, pgvector child chunk store, and parent chunk store.
- `project/retrieval/`: RRF fusion, reranking, source filtering, and context policy selection.
- `project/rag_agent/`: LangGraph state, nodes, edges, prompts, schemas, and tool factory.
- `project/evaluation/`: datasets, metrics, validation, reports, and evaluation runners.

`project/ui/gradio_app.py` is a legacy or alternate Gradio UI module. The current `python project/app.py` path does not mount it.

## Configuration

The primary runtime settings are in `project/config.py`.

### LLM

The current application uses `langchain_openai.ChatOpenAI` with DeepSeek/OpenAI-compatible settings:

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`project/core/rag_system.py` requires `DEEPSEEK_API_KEY` before the RAG app can start.

### Database

Default database URL:

```text
postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag
```

Override with:

```env
DATABASE_URL=postgresql://user:password@host:5432/database
```

The app initializes the minimal runtime schema, including the `vector` extension, parent chunks, and child chunks.

### Retrieval and Chunking

Current defaults:

```python
DENSE_MODEL = "BAAI/bge-base-zh-v1.5"
DENSE_EMBEDDING_DIMENSION = 768
RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVAL_FUSION_MODE = "rrf"
DENSE_TOP_K = 70
SPARSE_TOP_K = 30
RRF_TOP_K = 20
RRF_K = 60
RERANKER_FINAL_TOP_K = 3
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 60
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
```

Supported document extensions default to:

```text
.pdf,.md,.docx,.pptx
```

Context policy can be controlled with:

```env
RETRIEVAL_CONTEXT_POLICY=adaptive
RETRIEVAL_NEIGHBOR_WINDOW=1
RETRIEVAL_PARENT_EXPAND_MIN_HITS=2
```

Valid context policies are `adaptive`, `child`, `neighbor`, and `parent`.

### Hugging Face Cache

The app sets Hugging Face cache paths under `.cache/huggingface` by default. Useful environment options:

```env
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_OFFLINE=0
DENSE_LOCAL_FILES_ONLY=false
RERANKER_LOCAL_FILES_ONLY=false
```

If the embedding dimension changes, rebuild the PostgreSQL volume and re-index documents.

### Optional Features

Langfuse tracing:

```env
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

Legacy multimodal PDF image settings are present in `project/config.py` and `project/.env.example`, but the default MarkItDown conversion path is the primary document conversion flow.

## Internal HTTP API

The browser UI uses these internal routes. They are documented for local development and are not promised as a stable external API.

Documents and courses:

- `POST /api/documents/upload`
- `GET /api/documents/tasks/{task_id}`
- `GET /api/documents/files`
- `GET /api/documents/courses`
- `POST /api/documents/clear`
- `POST /api/documents/courses/rename`
- `POST /api/documents/sections/rename`

Sessions:

- `GET /api/sessions`
- `POST /api/sessions`
- `DELETE /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/turns`

Chat:

- `POST /api/chat`: streams `text/event-stream` events.
- `POST /api/chat/clear`

Upload tasks are tracked in memory and expire after the task cleanup window. Chat turns and session metadata are persisted in `runtime/session_memory.sqlite3`.

## Runtime Data

Runtime output is written under `runtime/`:

- `markdown_docs`: converted Markdown.
- `markdown_docs_cleaned`: cleaned Markdown.
- `markdown_cleaning_logs` and `markdown_cleaning_diffs`: cleaning audit files.
- `ingestion_logs`: per-document ingestion stage logs.
- `index_state`: index manifest and course structure.
- `document_images`: PDF image extraction output when enabled.
- `session_memory.sqlite3`: chat session memory.
- `evaluation_reports`: evaluation outputs.

## Evaluation

See `project/evaluation/README.md` for detailed evaluation modes and report validity rules.

Oracle-context RAGBench generation example:

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 50 \
  --output-dir runtime/evaluation_reports/ragbench_covidqa_test_50 \
  --ragas-max-workers 1 \
  --ragas-batch-size 1
```

Local retrieval example:

```bash
python project/evaluation/runners/retrieval_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --output-dir runtime/evaluation_reports/local_retrieval
```

Local RAGAS generation example:

```bash
python project/evaluation/runners/ragas_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --output-dir runtime/evaluation_reports/local_ragas
```

Use evaluation numbers only with their dataset, split, limit, evaluation type, and warning summary. An empty or placeholder gold dataset cannot support quality conclusions.

## Docker Notes

`docker-compose.yml` supports both database-only development and full container deployment.

- `docker compose up -d postgres` starts PostgreSQL + pgvector for local Python development.
- `docker compose up --build app` builds `Dockerfile` and starts the FastAPI app with PostgreSQL.
- `docker/init.sql` is still used by the PostgreSQL container to initialize the `vector` extension, so the `docker/` directory should stay while compose mounts that file.
- `rag_runtime`, `hf_cache`, and `pgdata` volumes persist uploaded documents/session state, Hugging Face model files, and PostgreSQL data.

## Verification

Minimum code sanity check:

```bash
python -m py_compile project/app.py project/server.py project/config.py
```

For documentation-only changes, the full test suite is not required unless a referenced command or behavior is changed.
