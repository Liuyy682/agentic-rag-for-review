# agentic-rag-for-review

面向课程资料和本地文档问答的 Agentic RAG 复盘项目。当前主应用是一个 FastAPI/Uvicorn 服务：`project/app.py` 启动后提供静态 Web 页面、内部 `/api/*` 接口和 SSE 流式聊天；后端使用 LangGraph 编排 RAG Agent，使用 PostgreSQL + pgvector 存储知识库。

## 当前能力

- Web UI 上传 PDF、Markdown、Word、PowerPoint 文件，并显示异步摄入进度。
- 文档转换为 Markdown 后进行清洗、父子分块、索引和课程绑定。
- 基于 PostgreSQL + pgvector 保存父块、子块、元数据、稠密向量和稀疏检索字段。
- 默认检索链路为稠密向量检索 + PostgreSQL 全文检索 + RRF 融合 + cross-encoder 重排。
- 根据问题类型和命中情况选择子块、邻近子块或父块作为回答上下文。
- LangGraph 编排会话摘要、意图识别、查询改写、澄清、任务规划、检索、答案评估、降级回答和聚合。
- 支持课程范围问答、课程/章节重命名、会话创建/删除和 SQLite 会话记忆。
- `project/evaluation` 提供 RAGBench、RAGAS、本地检索评测和分块消融脚本。

## 架构

```text
Browser static UI
-> FastAPI /api endpoints
-> RagApplication
-> RAGSystem
-> LangGraph agent graph
-> rag_research tool
-> RetrievalPipeline
-> PostgreSQL + pgvector
```

主要路径：

- `project/app.py`：当前主入口，启动 Uvicorn 服务。
- `project/server.py`：FastAPI 应用，挂载 `/static` 并注册 `/api` 路由。
- `project/static/`：当前浏览器 UI。
- `project/api/`：文档、课程、会话和 SSE 聊天接口。
- `project/application/rag_application.py`：组装 RAG 系统、文档管理器和聊天接口。
- `project/core/rag_system.py`：初始化 DeepSeek/OpenAI-compatible 聊天模型、LangGraph、存储和检索工具。
- `project/ingestion/`：文档转换、清洗、分块、索引 manifest、课程结构和文件完整性检查。
- `project/storage/`：PostgreSQL 连接、pgvector 子块检索和父块存储。
- `project/retrieval/`：RRF 融合、cross-encoder 重排和上下文策略选择。
- `project/rag_agent/`：LangGraph 状态、节点、路由、提示词、schema 和工具。
- `project/evaluation/`：评测数据、指标、报告和 runner。

`project/ui/gradio_app.py` 是旧版/备用 Gradio UI 模块；当前 `python project/app.py` 不会使用它。

## 技术栈

- Python 3.11+
- FastAPI / Uvicorn
- 浏览器静态 UI + Server-Sent Events
- LangGraph / LangChain
- `langchain_openai.ChatOpenAI` 接 DeepSeek 或其他 OpenAI-compatible API
- PostgreSQL 17 + pgvector
- `psycopg2` 直连 PostgreSQL
- HuggingFace sentence-transformer embeddings
- PostgreSQL 全文检索 + `jieba`
- `sentence-transformers` cross-encoder 重排
- MarkItDown、PyMuPDF 等文档转换工具
- SQLite 会话记忆
- 可选 Langfuse 链路追踪

## 安装与运行

创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

启动 PostgreSQL + pgvector：

```bash
docker compose up -d postgres
```

创建环境配置。项目会先读取根目录 `.env`，再读取 `project/.env` 并以 `project/.env` 覆盖同名配置：

```bash
cp project/.env.example project/.env
```

至少需要配置：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
```

启动应用：

```bash
python project/app.py
```

浏览器访问：

```text
http://localhost:7860
```

## 配置要点

当前 LLM 入口在 `project/core/rag_system.py`，使用 `ChatOpenAI` 连接 DeepSeek/OpenAI-compatible API：

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

默认数据库地址：

```text
postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag
```

可以通过 `DATABASE_URL` 覆盖。

默认模型与检索配置来自 `project/config.py`：

```python
DENSE_MODEL = "BAAI/bge-base-zh-v1.5"
DENSE_EMBEDDING_DIMENSION = 768
RERANKER_MODEL = "BAAI/bge-reranker-base"
RETRIEVAL_FUSION_MODE = "rrf"
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 60
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
SUPPORTED_DOCUMENT_EXTENSIONS = [".pdf", ".md", ".docx", ".pptx"]
```

如果 Hugging Face 直连不稳定，可以在启动前设置镜像或离线缓存配置：

```env
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_OFFLINE=0
DENSE_LOCAL_FILES_ONLY=false
RERANKER_LOCAL_FILES_ONLY=false
```

如果修改 `DENSE_EMBEDDING_DIMENSION` 或 embedding 模型维度，需要重建 PostgreSQL 数据卷并重新索引文档。

## Web UI 与内部 API

当前前端使用这些内部 HTTP 接口；它们服务于本仓库 UI，不承诺作为稳定外部 API：

- `POST /api/documents/upload`：上传文档并创建后台摄入任务。
- `GET /api/documents/tasks/{task_id}`：查询摄入进度和结果。
- `GET /api/documents/files`：列出知识库中的 Markdown 文件。
- `GET /api/documents/courses`：列出课程选项和课程结构文本。
- `POST /api/documents/clear`：清空知识库。
- `POST /api/documents/courses/rename`：重命名课程。
- `POST /api/documents/sections/rename`：重命名章节。
- `GET /api/sessions` / `POST /api/sessions`：列出或创建会话。
- `GET /api/sessions/{session_id}/turns`：读取会话历史。
- `DELETE /api/sessions/{session_id}`：删除会话。
- `POST /api/chat`：SSE 流式聊天，响应类型为 `text/event-stream`。
- `POST /api/chat/clear`：清空指定会话。

## 运行时产物

运行时文件默认写入 `runtime/`：

- `runtime/markdown_docs`：转换后的 Markdown。
- `runtime/markdown_docs_cleaned`：清洗后的 Markdown。
- `runtime/markdown_cleaning_logs` / `runtime/markdown_cleaning_diffs`：清洗日志和差异。
- `runtime/ingestion_logs`：文档摄入阶段日志。
- `runtime/index_state`：索引 manifest 和课程结构。
- `runtime/session_memory.sqlite3`：会话记忆。
- `runtime/evaluation_reports`：评测报告。

## 检索行为

默认检索链路：

```text
稠密向量检索 + 稀疏全文检索 -> RRF 融合 -> cross-encoder 重排 -> 上下文选择
```

上下文策略由以下配置控制：

- `RETRIEVAL_CONTEXT_POLICY`：`adaptive`、`child`、`neighbor` 或 `parent`
- `RETRIEVAL_NEIGHBOR_WINDOW`
- `RETRIEVAL_PARENT_EXPAND_MIN_HITS`

`adaptive` 模式下，解释、比较、机制类问题倾向使用父块上下文；事实类问题可以保留子块上下文；同一父块命中较多时会扩展到父块或邻近子块。

## 评测

评测工具位于 `project/evaluation`，详细说明见 `project/evaluation/README.md`。

RAGBench oracle-context 生成评测示例：

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 50 \
  --output-dir runtime/evaluation_reports/ragbench_covidqa_test_50 \
  --ragas-max-workers 1 \
  --ragas-batch-size 1
```

本地检索评测示例：

```bash
python project/evaluation/runners/retrieval_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --output-dir runtime/evaluation_reports/local_retrieval
```

注意：仓库默认的 `project/evaluation/datasets/eval_questions.jsonl` 可能为空或仅用于占位；要得出效果结论，需要使用带真实 `gold_parent_ids` 或 `gold_child_ids` 的数据集，并同时查看 `validity_summary.json` 和 `evaluation_warnings.jsonl`。

## 验证

文档改动后的最小代码验证：

```bash
python -m py_compile project/app.py project/server.py project/config.py
```

Markdown 只描述当前代码路径和配置，不包含未由代码或评测产物支撑的效果指标。
