# Agentic RAG 系统文档

这是一个基于 **LangGraph** 构建的 **Agentic Retrieval-Augmented Generation（RAG）** 系统，具备 **父子分块（parent-child chunking）**、**稠密 + 稀疏混合检索（hybrid dense + sparse retrieval）** 和 **多 LLM 提供商支持**。


## 目录

[快速开始](#快速开始) | [架构概览](#架构概览) | [项目结构](#项目结构) | [配置指南](#配置指南) | [常见定制](#常见定制) | [可观测性](#可观测性) | [高级主题](#高级主题) | [故障排查](#故障排查)

---

## 快速开始

### 安装

安装所有必需依赖：

```bash
pip install -r requirements.txt
```

### 运行应用

在本地启动 Gradio 界面：

```bash
python project/app.py
```

应用将在 `http://localhost:7860` 可用（Gradio 默认端口）。

### 前置条件

- Python 3.11+
- Ollama（本地）或 OpenAI、Anthropic、Google 的 API Key

---

## 架构概览

该系统实现了高级 RAG 流水线，核心特性如下：

- **父子分块**：文档被拆分为小的子块（便于精准检索），并关联到更大的父块（提供更丰富上下文）
- **混合检索**：结合稠密向量检索与稀疏检索（BM25），获得更优结果
- **LangGraph Agent**：编排查询改写、检索与回答生成
- **多提供商支持**：可在 Ollama、OpenAI GPT、Google Gemini、Anthropic Claude 之间无缝切换
- **向量存储**：使用 Qdrant 进行高效相似度搜索

### 数据流

```
PDF → Markdown 转换 → 父/子分块 → 向量索引 → Agent 检索 → LLM 回答
```

---

## 项目结构

### 入口与配置

| 文件 | 作用 |
|------|------|
| `project/app.py` | 应用入口，启动 Gradio UI |
| `project/config.py` | **中心配置枢纽** - 模型/提供商/分块策略等优先在这里修改 |
| `project/utils.py` | PDF 转 Markdown 以及上下文 token 估算 |
| `project/document_chunker.py` | 父子分块逻辑，包含清洗与合并规则 |
| `project/Dockerfile` | 集成 Ollama 的 Dockerfile（本地部署） |

### 核心系统

| 文件 | 作用 |
|------|------|
| `project/core/rag_system.py` | 系统引导：创建管理器并编译 LangGraph Agent |
| `project/core/document_manager.py` | 文档摄取流水线（转换、分块、索引） |
| `project/core/chat_interface.py` | 与 Agent 图交互的轻量封装 |
| `project/core/observability.py` | 可选 Langfuse 追踪：回调处理器生命周期 |

### 数据库层

| 文件 | 作用 |
|------|------|
| `project/db/vector_db_manager.py` | Qdrant 客户端封装，含向量模型初始化 |
| `project/db/parent_store_manager.py` | 基于文件的父块存储 |

### RAG Agent（LangGraph）

| 文件 | 作用 |
|------|------|
| `project/rag_agent/graph.py` | 图构建与编译逻辑 |
| `project/rag_agent/graph_state.py` | 全局/局部图状态定义，以及答案聚合/重置逻辑 |
| `project/rag_agent/nodes.py` | 节点实现（总结、改写、Agent 执行、聚合） |
| `project/rag_agent/edges.py` | 条件边路由逻辑（例如按问题清晰度路由） |
| `project/rag_agent/tools.py` | 检索工具（`search_child_chunks`、`retrieve_parent_chunks`） |
| `project/rag_agent/prompts.py` | Agent 行为的系统提示词 |
| `project/rag_agent/schemas.py` | 结构化输出 Schema（Pydantic 模型） |

### 用户界面

| 文件 | 作用 |
|------|------|
| `project/ui/css.py` | Gradio 界面的自定义 CSS 样式 |
| `project/ui/gradio_app.py` | Gradio UI 实现（文档上传 + 聊天） |

---

## 配置指南

所有主要设置都在 `project/config.py` 中。关键参数如下：

### 目录配置

```python
MARKDOWN_DIR = "markdown_docs"        # 转换后的 PDF → Markdown 文件存储目录
PARENT_STORE_PATH = "parent_store"    # 基于文件的父块存储目录
QDRANT_DB_PATH = "qdrant_db"          # 本地 Qdrant 向量数据库路径
```

### Qdrant 配置

```python
CHILD_COLLECTION = "document_child_chunks"  # 子块集合名称
SPARSE_VECTOR_NAME = "sparse"               # 命名稀疏向量字段（BM25）
```

### 模型配置

```python
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
LLM_PROVIDER = "ollama"  # "ollama" 或 "openai"
OLLAMA_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
OPENAI_MODEL = "gpt-5.4-mini"
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LLM_TEMPERATURE = 0  # 0 = 更确定，1 = 更有创造性
```

如需使用 OpenAI API，请在启动应用前安装依赖并设置环境变量：

```bash
pip install -r requirements.txt
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-openai-key"
export OPENAI_MODEL="gpt-5.4-mini"
python project/app.py
```

### Agent 配置
```python
# 防止无限循环的硬限制
MAX_TOOL_CALLS = 8       # 每次 Agent 运行的最大工具调用次数
MAX_ITERATIONS = 10      # Agent 循环最大迭代次数
GRAPH_RECURSION_LIMIT = 50 # 触发停止条件前的最大步骤数

# 上下文压缩阈值
BASE_TOKEN_THRESHOLD = 2000     # 初始压缩阈值
TOKEN_GROWTH_FACTOR = 0.9       # 每次压缩后应用的乘数
```

### 文本切分器配置

```python
CHILD_CHUNK_SIZE = 500              # 用于检索的子块大小
CHILD_CHUNK_OVERLAP = 100           # 子块重叠（防止上下文丢失）
MIN_PARENT_SIZE = 2000              # 父块最小大小
MAX_PARENT_SIZE = 4000             # 父块最大大小

# Markdown 标题切分策略
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]
```

### Langfuse 可观测性（可选）

```python
LANGFUSE_ENABLED = False               # 可通过 LANGFUSE_ENABLED 环境变量设为 True
LANGFUSE_PUBLIC_KEY = ""               # 来自你的 Langfuse 项目设置
LANGFUSE_SECRET_KEY = ""               # 来自你的 Langfuse 项目设置
LANGFUSE_BASE_URL = "http://localhost:3000"  # Langfuse Cloud 或自托管 URL
```

---

## 常见定制

### 1. 切换 LLM 提供商（单提供商）

> **性能说明：** 参数规模在 7B 以上的 LLM，通常在推理能力、上下文理解和回答质量上优于更小模型。无论是闭源还是开源模型，这一规律通常都成立，前提是模型 **支持原生 tool/function calling**，这是 agentic RAG 工作流所必需的。

如果你想永久从一个提供商切换到另一个（例如 Ollama → Google Gemini），请按以下步骤操作：

**步骤 1：** 安装对应提供商 SDK

```bash
pip install langchain-google-genai
```

**步骤 2：** 设置环境变量

```bash
export GOOGLE_API_KEY="your-google-key"
```

**步骤 3：** 更新 `project/config.py`

```python
LLM_MODEL = "gemini-2.5-pro"
LLM_TEMPERATURE = 0
```

**步骤 4：** 修改 `project/core/rag_system.py`

将：

```python
llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
```

替换为：

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
```

### 2. 多提供商配置

这种方式允许你维护多套提供商配置，并轻松切换。

**步骤 1：** 安装所需 SDK

```bash
pip install langchain-openai langchain-anthropic langchain-google-genai
```

**步骤 2：** 设置环境变量

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"
```

**步骤 3：** 在 `project/config.py` 中添加多提供商配置

```python
# --- Multi-Provider LLM Configuration ---
LLM_CONFIGS = {
    "ollama": {
        "model": "ministral-3:14b-instruct-2512-q4_K_M",
        "url":"http://localhost:11434",
        "temperature": 0
    },
    "openai": {
        "model": "gpt-5.2",
        "temperature": 0
    },
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "temperature": 0
    },
    "google": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    }
}

# 仅修改这一行即可切换提供商
ACTIVE_LLM_CONFIG = "ollama"
```

**步骤 4：** 修改 `project/core/rag_system.py` 中的 `initialize()` 方法

用以下代码替换现有 LLM 初始化逻辑：

```python
def initialize(self):
    self.vector_db.create_collection(self.collection_name)
    collection = self.vector_db.get_collection(self.collection_name)
    
    # 读取当前激活配置
    active_config = config.LLM_CONFIGS[config.ACTIVE_LLM_CONFIG]
    model = active_config["model"]
    temperature = active_config["temperature"]
    
    if config.ACTIVE_LLM_CONFIG == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=temperature, base_url=active_config["url"])
        
    elif config.ACTIVE_LLM_CONFIG == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temperature)
        
    elif config.ACTIVE_LLM_CONFIG == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=temperature)
        
    elif config.ACTIVE_LLM_CONFIG == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {config.ACTIVE_LLM_CONFIG}")
    
    # 继续初始化工具和图
    tools = ToolFactory(collection).create_tools()
    self.agent_graph = create_agent_graph(llm, tools)
```

**切换提供商：** 只需在 `config.py` 里修改 `ACTIVE_LLM_CONFIG`：

```python
ACTIVE_LLM_CONFIG = "google"  # 切到 Gemini Pro
# ACTIVE_LLM_CONFIG = "anthropic"  # 或 Claude Sonnet
# ACTIVE_LLM_CONFIG = "openai"  # 或 GPT-4o
```

---

**提供商参考表：**

| 提供商 | 环境变量 | 导入语句 | 示例模型 |
|----------|---------------------|------------------|----------------|
| OpenAI | `OPENAI_API_KEY` | `from langchain_openai import ChatOpenAI` | `gpt-5.2`, `ggpt-5-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `from langchain_anthropic import ChatAnthropic` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| Google | `GOOGLE_API_KEY` | `from langchain_google_genai import ChatGoogleGenerativeAI` | `gemini-2.5-pro`, `gemini-2.5-flash` |
| Ollama | 无（本地） | `from langchain_ollama import ChatOllama` | `qwen3:4b-instruct-2507-q4_K_M`, `ministral-3:8b-instruct-2512-q4_K_M`, `llama3.1:8b-instruct-q6_K` |

---

### 3. 更换 Embedding 模型

**为什么要换？** 在速度、成本、质量之间做权衡。

**步骤 1：** 更新 `project/config.py`

```python
# 示例：更快、更小的模型
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 示例：质量更高、速度更慢的模型
# DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"

# 示例：Gemma embedding（Google 开源模型）
# DENSE_MODEL = "google/embeddinggemma-300m"

# 示例：Qwen embedding（阿里多语言模型）
# DENSE_MODEL = "Qwen/Qwen3-Embedding-8B"

SPARSE_MODEL = "Qdrant/bm25"  # 通常无需修改
```

**步骤 2：** 重新索引文档

⚠️ **重要：** 修改 embedding 后，必须通过 Gradio UI 对所有文档重新索引。

**实现细节**（位于 `project/db/vector_db_manager.py`）：

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse
import config

self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)
```

**常见 Embedding 模型：**

| 模型 | 上下文长度 | 向量维度 | 速度 | 质量 | 适用场景 |
|-------|--------------|------------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 256 tokens | 384 | 快 | 良好 | 通用场景，快速语义相似度 |
| all-mpnet-base-v2 | 512 tokens | 768 | 中 | 优秀 | 高精度语义检索 |
| bge-large-en-v1.5 | 512 tokens | 1024 | 慢 | 最佳 | 生产级 GPU 检索 |
| google/embeddinggemma-300m | 2048 tokens | 768 | 快 | 很好 | 轻量高效的多语言检索 |
| Qwen/Qwen3-Embedding-8B | 32768 tokens | 4096 | 慢 | 优秀 / SOTA | 大规模多语言 embedding，长上下文 RAG |

---

### 4. 调整分块策略

**为什么要调？** 在检索精度与上下文丰富度之间平衡。

> **💡 验证工具：** 为避免反复试错，你可以使用 🐿️[**Chunky**](https://github.com/GiovanniPasq/chunky) 可视化检查不同策略对文档的影响。

**步骤 1：** 在 `project/config.py` 中调整分块大小

```python
# 适用于短事实型查询（例如技术文档）
CHILD_CHUNK_SIZE = 300
CHILD_CHUNK_OVERLAP = 60
MIN_PARENT_SIZE = 1500
MAX_PARENT_SIZE = 8000

# 适用于叙述型/强上下文查询（例如法律文档）
# CHILD_CHUNK_SIZE = 800
# CHILD_CHUNK_OVERLAP = 150
# MIN_PARENT_SIZE = 3000
# MAX_PARENT_SIZE = 15000
```

**步骤 2（可选）：** 替换 `project/document_chunker.py` 中的切分器

**默认（基于字符）：**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

self.__child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHILD_CHUNK_OVERLAP
)
```

**替代（句子感知）：**
```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

self.__child_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=config.CHILD_CHUNK_SIZE,
    chunk_overlap=config.CHILD_CHUNK_OVERLAP
)
```

**步骤 3：** 重新运行文档摄取流程

在 Gradio 界面重新上传文档，以应用新的分块策略。

**分块建议：**

> ⚠️ **说明：** 以下为经验值。最佳尺寸依赖于：
> - **子块（Child chunk）** → embedding 模型上下文窗口（例如 all-MiniLM-L6-v2 为 256 tokens，bge-large-en-v1.5 为 512）：子块大小不应超过该窗口
> - **父块（Parent chunk）** → 生成模型上下文窗口（例如 8K、32K、128K tokens）：父块必须能与查询一起放入传给 LLM 的上下文中
>
> 请务必在你的语料上做实测验证。

| 文档类型 | 子块大小 | 父块大小 | 原因 |
|---------------|-----------|-------------|-----------|
| 技术文档 | 300-500 | 2000-4000 | 精确定位、代码片段场景 |
| 法律合同 | 600-1000 | 5000-15000 | 强上下文、定义密集 |
| 研究论文 | 400-600 | 3000-8000 | 精度与上下文平衡 |
| FAQ / 知识库 | 200-400 | 1500-4000 | 短而聚焦的回答 |

---

### 5. Agent 配置

在 `project/config.py` 中调节 Agent 行为：
```python
# 防止无限循环的硬限制
MAX_TOOL_CALLS = 8       # 每次 Agent 运行最大工具调用数
MAX_ITERATIONS = 10      # Agent 推理循环最大迭代次数
GRAPH_RECURSION_LIMIT = 50 # 触发停止条件前最大步骤数

# 上下文压缩阈值
BASE_TOKEN_THRESHOLD = 2000     # 初始压缩阈值
TOKEN_GROWTH_FACTOR = 0.9       # 每次压缩后的乘数
```

| 参数 | 影响 |
|-----------|--------|
| `MAX_TOOL_CALLS` | 复杂问题可适当增大；简单问题可减小以提速 |
| `MAX_ITERATIONS` | 控制 Agent 可进行多少轮推理循环 |
| `GRAPH_RECURSION_LIMIT` | 面对复杂 [graphs](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT) 可适当提高 |
| `BASE_TOKEN_THRESHOLD` | 增大可延后压缩触发 |
| `TOKEN_GROWTH_FACTOR` | 值越低，压缩越激进 |

---

## 可观测性

通过 [Langfuse](https://langfuse.com) 的可选追踪功能，你可以捕获每次 LLM 调用、工具调用和图状态切换。这对于调试 Agent 行为、跟踪成本和评估检索质量非常有用。

### 启用 Langfuse

1. 在 [Langfuse Cloud](https://cloud.langfuse.com/) 注册，创建组织与项目，然后在项目设置中生成 API Key。
2. 设置环境变量（或复制 `.env.example` 为 `.env`）：

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # 或你的自托管地址
```

4. 正常运行应用后，可在 [Langfuse dashboard](https://cloud.langfuse.com/) 查看追踪。

如需禁用追踪，将 `LANGFUSE_ENABLED=false` 或不设置相关变量即可。无论是否启用追踪，应用行为保持一致。

关于 Langfuse 与 LangChain/LangGraph 集成的更多信息，请参考官方[文档](https://docs.langchain.com/oss/python/integrations/providers/langfuse)。

### 会被追踪的内容

| 组件 | 追踪操作 |
|-----------|-------------------|
| 图节点 | `summarize_history`、`rewrite_query`、`orchestrator`、`compress_context`、`fallback_response`、`aggregate_answers` |
| 工具 | `search_child_chunks`、`retrieve_parent_chunks`（参数 + 结果） |
| 结构化输出 | 改写步骤中的 `QueryAnalysis` 解析 |
| 子图并行分发 | 通过 `Send()` 发起的并行 Agent 调用 |

### 部署方式

- **Langfuse Cloud**：在 [cloud.langfuse.com](https://cloud.langfuse.com) 注册，免费额度为每月 50K observations。
- **自托管**：MIT 协议，可使用 Docker Compose 部署。参见官方[自托管文档](https://langfuse.com/self-hosting)。

关于不同可观测性平台（LangSmith、Arize Phoenix、AgentOps、Braintrust、Helicone）的详细对比以及完整自托管方案，可参考 [`Observability_Guide.ipynb`](../Observability_Guide.ipynb)。

---

## 高级主题

### 定制 RAG Agent

**位置：** `project/rag_agent/`

**增加/删除节点：** 编辑 `graph.py` 与 `nodes.py`

示例：添加事实核查节点
```python
# In nodes.py
def fact_check_node(state):
    # Your fact-checking logic
    return {"fact_checked": True}

# In graph.py
builder.add_node("fact_check", fact_check_node)
builder.add_edge("retrieve", "fact_check")
```

**修改条件路由：** 编辑 `edges.py` 以调整图流转逻辑

系统中的示例 - 基于问题清晰度进行路由：
```python
def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    """Routes to human input if question unclear, otherwise processes all rewritten queries"""
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        # Fan-out: send each rewritten question to parallel processing
        return [
            Send("agent", {"question": query, "question_index": idx, "messages": []})
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]
```

这种模式允许 Agent 在“向用户请求澄清”和“并行处理多个改写问题”之间进行选择。

**修改提示词：** 编辑 `prompts.py` 以改变 Agent 行为和回答风格

**添加自定义工具：** 扩展 `tools.py`，加入新的检索策略或外部集成

### 替换存储后端

**向量数据库：**
- 默认：本地 Qdrant
- 可选：远程 Qdrant Cloud、Pinecone、Weaviate
- 修改文件：`project/db/vector_db_manager.py`

**父块存储：**
- 默认：JSON 文件
- 可选：PostgreSQL、MongoDB、S3
- 修改文件：`project/db/parent_store_manager.py`

### 扩展 UI

**位置：** `project/ui/gradio_app.py`

你可以增加运行时设置、管理面板或分析功能：
```python
with gr.Accordion("Advanced Settings", open=False):
    provider_dropdown = gr.Dropdown(
        choices=["openai", "anthropic", "google", "ollama"],
        label="LLM Provider"
    )
```

### Docker 部署

> ⚠️ **系统要求**：Docker 至少分配 8GB 内存。默认 Ollama 模型约需 3.3GB 才能运行。

#### 构建并运行
```bash
# 构建镜像
docker build -t agentic-rag -f project/Dockerfile .

# 运行容器
docker run --name rag-assistant -p 7860:7860 agentic-rag
```

**可选：GPU 加速**（仅 NVIDIA）：
```bash
docker run --gpus all --name rag-assistant -p 7860:7860 agentic-rag
```

**常用命令：**
```bash
docker stop rag-assistant      # 停止
docker start rag-assistant     # 重启
docker logs -f rag-assistant   # 查看日志
docker rm -f rag-assistant     # 删除
```

> ⚠️ **性能说明**：在 Windows/Mac 上，Docker 通过 Linux VM 运行，可能会降低文档索引等 I/O 操作性能；LLM 推理速度通常影响不大。在 Linux 上，性能与本地运行基本相当。

启动后，访问 `http://localhost:7860`。

---

## 故障排查

| 问题 | 原因 | 解决方案 |
|-------|-------|----------|
| "Model not found" 错误 | 提供商对应的模型名错误 | 校验 `LLM_MODEL` 是否符合提供商 API（例如 `gpt-4o-mini` 而不是 `gpt4-mini`） |
| 检索质量差 | embedding 模型或分块配置不佳 | 使用更好的 embedding（如 all-mpnet-base-v2）并重新索引，或调整分块大小 |
| 响应慢 | embedding 模型过大或 `top_k` 过高 | 使用更小 embedding（如 all-MiniLM-L6-v2）或下调检索工具中的 `top_k` |
| API 速率限制超限 | 向外部提供商请求过多 | 增加指数退避重试逻辑，或改用本地 Ollama 模型 |
| 内存不足（OOM） | 文档规模过大或 embedding 过重 | 使用更小 embedding、减小 batch size，或启用 GPU 加速 |
| 检索结果为空 | 集合未索引或集合名不匹配 | 检查文档是否已上传，并确认配置中的 `CHILD_COLLECTION` 名称一致 |
| 切换提供商后导入报错 | 缺少 SDK 依赖 | 安装对应包：`pip install langchain-{provider}` |
| 多次运行回答不一致 | 温度参数过高 | 在配置中设定 `LLM_TEMPERATURE = 0` 获取稳定结果 |
