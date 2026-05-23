# agentic-rag-for-review

这是一个面向课程资料和本地文档问答的 Agentic RAG 个人复盘版本。当前代码重点是本地 Gradio 应用、PostgreSQL + pgvector 存储、LangGraph 编排、混合检索、重排序、文档摄入和评测脚本。

## 项目能力

- 在 Gradio UI 中上传本地 PDF、Markdown、Word 或 PowerPoint 文件。
- 将文档转换为 Markdown，清理重复页眉页脚，抽取结构，并切分为父块和子块。
- 使用 PostgreSQL + pgvector 存储父块、子块、元数据、稠密向量和稀疏检索字段。
- 结合稠密向量检索、PostgreSQL 全文检索和 RRF 融合。
- 在本地 cross-encoder 模型可用时，对候选结果进行重排序。
- 根据问题类型自适应选择子块、邻近子块或完整父块作为回答上下文。
- 使用 LangGraph 编排意图识别、查询改写、澄清、任务规划、检索、答案评估、降级回答和聚合。
- 支持按课程范围对话，并维护轻量会话记忆。
- 在 `project/evaluation` 下提供 RAGBench/RAGAS 相关评测脚本。

## 架构

```text
Gradio UI
-> RagApplication
-> RAGSystem
-> LangGraph agent graph
-> rag_research tool
-> RetrievalPipeline
-> PostgreSQL + pgvector
```

主要代码路径：

- `project/app.py` 启动 Gradio 应用。
- `project/application/rag_application.py` 组装 RAG 系统、文档管理器和聊天接口。
- `project/core/rag_system.py` 初始化 PostgreSQL 存储、DeepSeek/OpenAI-compatible 聊天模型、LangGraph 和工具。
- `project/ingestion/` 负责转换、清洗、切分、完整性检查、索引清单、图片提取和课程结构。
- `project/storage/` 提供基于 pgvector 的子块检索和父块存储。
- `project/retrieval/` 提供 RRF 融合、重排序和自适应上下文选择。
- `project/rag_agent/` 包含图状态、提示词、工具、节点和路由逻辑。
- `project/evaluation/` 包含 RAGBench 导入、评测运行器和报告工具。

## 技术栈

- Python 3.11+
- Gradio
- LangGraph / LangChain
- DeepSeek 或其他 OpenAI-compatible 聊天 API，通过 `ChatOpenAI` 接入
- PostgreSQL 17 + pgvector
- SQLAlchemy 和 Alembic
- HuggingFace sentence-transformer embeddings
- PostgreSQL 全文检索，结合 `jieba` 分词
- 基于 `sentence-transformers` 的 cross-encoder 重排序
- MarkItDown、PyMuPDF 等文档转换工具
- 可选 Langfuse 链路追踪

## 安装

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

创建根目录 `.env`：

```bash
cp .env.example .env
```

至少配置：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
```

Langfuse 和多模态摄入的可选配置见 `project/.env.example`。

## 数据库

启动带 pgvector 的 PostgreSQL：

```bash
docker compose up -d postgres
```

执行数据库迁移：

```bash
alembic upgrade head
```

默认数据库地址为：

```text
postgresql://agentic_rag:dev_only@localhost:5432/agentic_rag
```

也可以通过 `DATABASE_URL` 覆盖。

## 运行

```bash
python project/app.py
```

UI 主要包含两个页面：

- `Documents`：上传文件、绑定课程、查看已索引文档、重命名课程或章节、清空知识库。
- `Chat`：基于全部已索引文档或指定课程范围提问。

运行时产物会写入 `runtime/`，包括转换后的 Markdown、清洗后的 Markdown、摄入日志、图片提取结果、课程结构、会话记忆和评测报告。

## 检索行为

默认检索模式是 `rrf`：

```text
稠密向量检索 + 稀疏全文检索 -> RRF 融合 -> 重排序 -> 上下文选择
```

上下文选择由以下配置控制：

- `RETRIEVAL_CONTEXT_POLICY`：`adaptive`、`child`、`neighbor` 或 `parent`
- `RETRIEVAL_NEIGHBOR_WINDOW`
- `RETRIEVAL_PARENT_EXPAND_MIN_HITS`

在自适应模式下，解释型问题倾向使用父块上下文，事实型问题可以保留更小的子块上下文，同一父块命中较多时会扩展到邻近子块。

## 评测

评测工具位于 `project/evaluation`。RAGBench/RAGAS runner 可以生成回答、评分，并把结果写入 `runtime/evaluation_reports`。

示例：

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 50 \
  --output-dir runtime/evaluation_reports/ragbench_covidqa_test_50 \
  --ragas-max-workers 1 \
  --ragas-batch-size 1
```

评测细节见 `project/evaluation/README.md`。

## 说明

- 仓库会忽略本地运行数据、本地文档、notebook、内部任务文档和本地 agent 配置。
- 如果某个路径已经被 Git 跟踪，仅加入 `.gitignore` 不会让它从 GitHub 消失；还需要用 `git rm --cached` 从 Git 索引移除。
