# Agentic RAG 开发说明

这是一个面向课程资料和本地文档问答的 Agentic RAG 系统。当前主应用由 `python project/app.py` 启动 FastAPI/Uvicorn 服务，提供静态浏览器页面、内部 `/api/*` 路由和基于 Server-Sent Events 的流式聊天。

## 概览

当前能力：

- 在浏览器 UI 中上传 PDF、Markdown、Word、PowerPoint 文件。
- 将文档转换为 Markdown，清理重复页眉页脚，切分为父块和子块，并写入索引。
- 使用 PostgreSQL + pgvector 存储父块、子块、元数据、稠密向量和稀疏检索字段。
- 通过稠密向量检索、PostgreSQL 全文检索、RRF 融合和 cross-encoder 重排召回上下文。
- 根据问题类型和命中情况选择子块、邻近子块或完整父块作为回答上下文。
- 使用 LangGraph 编排会话摘要、意图识别、查询改写、澄清、任务规划、检索、答案评估、降级回答和聚合。
- 支持按课程范围聊天、课程/章节管理，以及 SQLite 轻量会话记忆。
- 在 `project/evaluation` 下提供 RAGBench、RAGAS、本地检索评测和分块消融脚本。

## 快速开始

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

启动 PostgreSQL + pgvector：

```bash
docker compose up -d postgres
```

创建配置文件：

```bash
cp project/.env.example project/.env
```

至少配置：

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

也可以用 Docker 同时启动应用和数据库：

```bash
cp project/.env.example project/.env
# 填写 project/.env 中的 DEEPSEEK_API_KEY
# 首次启动会安装 Python 依赖，Hugging Face 模型文件会缓存在 hf_cache 卷中
docker compose up --build app
```

配置加载顺序是先读取仓库根目录 `.env`，再读取 `project/.env`，并以 `project/.env` 覆盖同名配置。Docker 镜像会排除 `.env` 文件；`docker-compose.yml` 会在运行时传入存在的 `project/.env`，并把 `DATABASE_URL` 覆盖为 Compose 内部的 `postgres` 服务地址。

## 架构

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

主要路径：

- `project/app.py`：当前主入口，在 `7860` 端口启动 Uvicorn。
- `project/server.py`：FastAPI 应用，提供 `/`、`/static` 和 `/api`。
- `project/static/`：当前浏览器 UI。
- `project/api/`：文档、课程、会话、任务和流式聊天接口。
- `project/application/rag_application.py`：组装 RAG 系统、文档管理器和聊天接口。
- `project/core/rag_system.py`：初始化存储、`ChatOpenAI`、LangGraph 和检索工具。
- `project/ingestion/`：转换、Markdown 清洗、分块、索引 manifest、文件完整性检查和课程结构。
- `project/storage/`：PostgreSQL 连接、pgvector 子块存储和父块存储。
- `project/retrieval/`：RRF 融合、重排、来源过滤和上下文策略选择。
- `project/rag_agent/`：LangGraph 状态、节点、边、提示词、schema 和工具工厂。
- `project/evaluation/`：数据集、指标、校验、报告和评测 runner。

`project/ui/gradio_app.py` 是旧版/备用 Gradio UI 模块；当前 `python project/app.py` 路径不会挂载它。

## 配置

主要运行配置在 `project/config.py`。

### LLM

当前应用使用 `langchain_openai.ChatOpenAI` 连接 DeepSeek/OpenAI-compatible API：

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`project/core/rag_system.py` 会在启动 RAG 应用前检查 `DEEPSEEK_API_KEY`。

### 数据库

默认数据库地址：

```text
postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag
```

可通过环境变量覆盖：

```env
DATABASE_URL=postgresql://user:password@host:5432/database
```

应用会初始化最小运行时 schema，包括 `vector` extension、父块表和子块表。

### 检索与切分

当前默认配置：

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

默认支持的文档扩展名：

```text
.pdf,.md,.docx,.pptx
```

上下文策略配置：

```env
RETRIEVAL_CONTEXT_POLICY=adaptive
RETRIEVAL_NEIGHBOR_WINDOW=1
RETRIEVAL_PARENT_EXPAND_MIN_HITS=2
```

可用策略包括 `adaptive`、`child`、`neighbor` 和 `parent`。

### Hugging Face 缓存

应用默认把 Hugging Face 缓存放在 `.cache/huggingface`。常用环境变量：

```env
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_OFFLINE=0
DENSE_LOCAL_FILES_ONLY=false
RERANKER_LOCAL_FILES_ONLY=false
```

如果修改 embedding 模型维度，需要重建 PostgreSQL 数据卷并重新索引文档。

### 可选功能

Langfuse 链路追踪：

```env
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

`project/config.py` 和 `project/.env.example` 中保留了旧版多模态 PDF 图片相关配置，但默认主转换链路是 MarkItDown。

## 内部 HTTP API

当前浏览器 UI 使用以下内部路由。这些接口用于本地开发说明，不承诺作为稳定外部 API。

文档与课程：

- `POST /api/documents/upload`
- `GET /api/documents/tasks/{task_id}`
- `GET /api/documents/files`
- `GET /api/documents/courses`
- `POST /api/documents/clear`
- `POST /api/documents/courses/rename`
- `POST /api/documents/sections/rename`

会话：

- `GET /api/sessions`
- `POST /api/sessions`
- `DELETE /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/turns`

聊天：

- `POST /api/chat`：返回 `text/event-stream` 流式事件。
- `POST /api/chat/clear`

上传任务保存在内存中，并会在清理窗口后过期。聊天轮次和会话元数据保存在 `runtime/session_memory.sqlite3`。

## 运行时数据

运行时产物写入 `runtime/`：

- `markdown_docs`：转换后的 Markdown。
- `markdown_docs_cleaned`：清洗后的 Markdown。
- `markdown_cleaning_logs` 和 `markdown_cleaning_diffs`：清洗审计文件。
- `ingestion_logs`：每个文档的摄入阶段日志。
- `index_state`：索引 manifest 和课程结构。
- `document_images`：启用时保存 PDF 图片提取结果。
- `session_memory.sqlite3`：聊天会话记忆。
- `evaluation_reports`：评测输出。

## 评测

详细评测模式和报告有效性规则见 `project/evaluation/README.md`。

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

本地 RAGAS 生成评测示例：

```bash
python project/evaluation/runners/ragas_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --output-dir runtime/evaluation_reports/local_ragas
```

使用评测数字时必须同时说明 dataset、split、limit、evaluation type 和 warning 摘要。空数据集或占位 gold 数据不能支撑效果结论。

## Docker 说明

`docker-compose.yml` 同时支持只启动数据库和完整容器部署。

- `docker compose up -d postgres`：为本地 Python 开发启动 PostgreSQL + pgvector。
- `docker compose up --build app`：构建 `Dockerfile`，并启动 FastAPI 应用和 PostgreSQL。
- `docker/init.sql` 仍被 PostgreSQL 容器用于初始化 `vector` 扩展，所以 compose 还挂载该文件时应保留 `docker/` 目录。
- `rag_runtime`、`hf_cache`、`pgdata` 三个卷分别持久化上传文档/会话状态、Hugging Face 模型文件和 PostgreSQL 数据。

## 验证

最小代码检查：

```bash
python -m py_compile project/app.py project/server.py project/config.py
```

仅修改文档时，不需要运行完整测试套件；如果 README 引用的命令或行为发生变化，再运行对应最小验证。
