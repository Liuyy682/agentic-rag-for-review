<p align="center">
  <img alt="Agentic RAG for Dummies Logo" src="assets/logo.png" width="350px">
</p>

<h1 align="center">Agentic RAG for Dummies</h1>

<p align="center">
  <strong>使用 LangGraph、会话记忆与人机协同查询澄清机制，构建模块化 Agentic RAG 系统</strong>
</p>

<p align="center">
  <a href="#概览">概览</a> •
  <a href="#工作原理">工作原理</a> •
  <a href="#llm-提供商配置">LLM 提供商</a> •
  <a href="#实现细节">实现细节</a> •
  <a href="#安装与使用">安装与使用</a> •
  <a href="#故障排查">故障排查</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/GiovanniPasq/agentic-rag-for-dummies?style=social" alt="GitHub Stars"/>
  <img src="https://img.shields.io/github/forks/GiovanniPasq/agentic-rag-for-dummies?style=social" alt="GitHub Forks"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
  <a href="https://github.com/von-development/awesome-langgraph">
    <img src="https://awesome.re/badge.svg" alt="Awesome LangGraph"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-1.1%2B-orange?logo=langchain&logoColor=white" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Qdrant-vector%20db-DC244C" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/LLM%20Providers-Ollama%20%7C%20OpenAI%20%7C%20Anthropic%20%7C%20Google-purple" alt="LLM Providers"/>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/GiovanniPasq/agentic-rag-for-dummies/blob/main/notebooks/agentic_rag.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

<p align="center">
  <img alt="Agentic RAG Demo" src="assets/demo.gif" width="650px">
</p>

<p align="center">
  <strong>如果你喜欢这个项目，给个 Star ⭐️ 会非常感谢 :)</strong><br>
</p>

## 概览

本仓库演示了如何用极少代码基于 LangGraph 构建一个 **Agentic RAG（检索增强生成）** 系统。大多数 RAG 教程只讲基础概念，缺少如何搭建模块化、Agent 驱动系统的实践指南。本项目通过提供 **学习材料 + 可扩展架构** 来补齐这部分。

### 包含内容

| 功能 | 说明 |
|---|---|
| 🗂️ **分层索引** | 用小块精准检索，用大块父文档提供上下文 |
| 🧠 **会话记忆** | 在多轮问答中保持上下文连续性 |
| ❓ **查询澄清** | 对歧义问题进行改写，或暂停向用户询问细节 |
| 🤖 **Agent 编排** | 用 LangGraph 协调整体检索与推理流程 |
| 🔀 **多 Agent Map-Reduce** | 将复杂问题拆成并行子问题 |
| ✅ **自我纠错** | 初次结果不足时自动重查 |
| 🗜️ **上下文压缩** | 在长循环检索中保持工作记忆精简 |
| 🔍 **可观测性** | 通过 Langfuse 追踪 LLM 调用、工具使用与图执行 |

### 🎯 两种使用方式

**1️⃣ 学习路径：交互式 Notebook**

逐步教程，适合理解核心概念。若你刚接触 Agentic RAG 或希望快速实验，建议从这里开始。

**2️⃣ 构建路径：模块化项目**

灵活架构，每个组件都可独立替换：LLM 提供商、Embedding 模型、PDF 转换器、Agent 工作流。只改一行即可从 Ollama 切换到 Anthropic、OpenAI 或 Google。

开始前可先查看 [模块化架构](#模块化架构) 与 [安装与使用](#安装与使用)。

## 工作原理

### 文档准备：分层索引

在处理查询前，文档会做两次切分，以获得更优检索效果：

- **父块（Parent Chunks）**：基于 Markdown 标题（H1、H2、H3）的较大段落
- **子块（Child Chunks）**：从父块进一步切分的固定长度小片段

> 💡 可选：若你希望在索引前可视化检查或编辑分块，可使用 🐿️ [**Chunky**](https://github.com/GiovanniPasq/chunky)。

该策略将 **小块的检索精度** 与 **大块的上下文丰富度** 结合起来用于回答生成。

---

### 查询处理：四阶段智能流程
```
用户问题 → 会话摘要 → 查询改写 → 查询澄清 →
并行 Agent 推理 → 聚合 → 最终回答
```

**阶段 1 - 会话理解：** 分析最近对话历史，提取上下文并保持多轮问题连续性。

**阶段 2 - 查询澄清：** 解析指代（“我怎么更新它？”→“我怎么更新 SQL？”）、将复合问题拆成子问题、识别不清晰输入，并为检索改写查询。若仍不明确，会暂停并请求用户补充。

**阶段 3 - 智能检索（多 Agent Map-Reduce）：** 为每个子问题生成并行 Agent 子图。每个 Agent 会先检索子块，再拉取父块补全上下文；若结果不足会自我纠错重试；为避免冗余会压缩上下文；预算耗尽时会优雅降级给出可用答案。

> **示例：** *“什么是 JavaScript？什么是 Python？”* → 同时启动 2 个并行 Agent。

**阶段 4 - 回答生成：** 将所有 Agent 输出聚合为一个连贯回答。

---

## LLM 提供商配置

系统是提供商无关的。它支持 [LangChain](https://python.langchain.com/docs/integrations/chat/) 可接入的任意 LLM 提供商，并可一行切换。下面示例覆盖常见选项，其他提供商也可按同样模式接入。

> **注意：** 模型名称迭代较快。正式部署前，请务必查阅官方文档，确认最新可用模型及其标识符。

### Ollama（本地）

```bash
# 从 https://ollama.com 安装 Ollama
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
```
> ⚠️ 为获得更可靠的工具调用与指令遵循，建议优先使用 **7B+** 模型。较小模型可能忽略检索指令或出现幻觉。见 [故障排查](#故障排查)。

---

### 云端提供商

<details>
<summary>点击展开</summary>

**OpenAI GPT：**
```bash
pip install -qU langchain-openai
```
```python
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

**Anthropic Claude：**
```bash
pip install -qU langchain-anthropic
```
```python
from langchain_anthropic import ChatAnthropic
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
```

**Google Gemini**
```bash
pip install -qU langchain-google-genai
```
```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
```
</details>

---

## 实现细节

更多实现说明、扩展解释，以及 Langfuse 可观测性（LLM 调用追踪、工具使用、图执行跟踪）可在 **[notebook](notebooks/agentic_rag.ipynb)** 与完整项目中查看。

| 步骤 | 说明 |
|------|-------------|
| 1 | [初始化设置与配置](#步骤-1初始化设置与配置) |
| 2 | [配置向量数据库](#步骤-2配置向量数据库) |
| 3 | [文档转 Markdown](#步骤-3文档转-markdown) |
| 4 | [分层文档索引](#步骤-4分层文档索引) |
| 5 | [定义 Agent 工具](#步骤-5定义-agent-工具) |
| 6 | [定义系统提示词](#步骤-6定义系统提示词) |
| 7 | [定义状态与数据模型](#步骤-7定义状态与数据模型) |
| 8 | [Agent 配置](#步骤-8agent-配置) |
| 9 | [构建图节点与边函数](#步骤-9构建图节点与边函数) |
| 10 | [构建 LangGraph 图](#步骤-10构建-langgraph-图) |
| 11 | [创建聊天界面](#步骤-11创建聊天界面) |

### 步骤 1：初始化设置与配置

定义路径并初始化核心组件。

```python
import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

DOCS_DIR = "docs"  # Directory containing your pdf, md, docx, and pptx files
MARKDOWN_DIR = "runtime/markdown_docs" # Directory containing converted markdown files
PARENT_STORE_PATH = "runtime/parent_store"  # Directory for parent chunk JSON files
CHILD_COLLECTION = "document_child_chunks"

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)
os.makedirs(PARENT_STORE_PATH, exist_ok=True)

from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)

dense_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

client = QdrantClient(path="runtime/qdrant_db")
```

---

### 步骤 2：配置向量数据库

配置 Qdrant，以支持混合检索并存储子块。

```python
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode

embedding_dimension = len(dense_embeddings.embed_query("test"))

def ensure_collection(collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=embedding_dimension,
                distance=qmodels.Distance.COSINE
            ),
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams()
            },
        )
```

---

### 步骤 3：文档转 Markdown

使用 MarkItDown 将支持的本地文档转换为 Markdown。第一阶段支持 `.pdf`、`.md`、`.docx`、`.pptx`；URL 和 ZIP 输入会被禁用。

```python
from pathlib import Path
import glob
from markitdown import MarkItDown

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".md", ".docx", ".pptx"}

def document_to_markdown(document_path, output_dir):
    document_path = Path(document_path)
    if document_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {document_path.suffix}")

    output_path = (Path(output_dir) / document_path.stem).with_suffix(".md")
    if document_path.suffix.lower() == ".md":
        md = document_path.read_text(encoding="utf-8")
    else:
        md = MarkItDown().convert_local(document_path).text_content

    md_cleaned = md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    output_path.write_text(md_cleaned, encoding="utf-8")

def documents_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for document_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / document_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            document_to_markdown(document_path, output_dir)

for extension in SUPPORTED_DOCUMENT_EXTENSIONS:
    documents_to_markdowns(f"{DOCS_DIR}/*{extension}")
```

---

### 步骤 4：分层文档索引

使用 Parent/Child 策略处理文档。
```python
import os
import glob
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
```

<details>
<summary>父块与子块处理函数</summary>

```python
def merge_small_parents(chunks, min_size):
    if not chunks:
        return []

    merged, current = [], None

    for chunk in chunks:
        if current is None:
            current = chunk
        else:
            current.page_content += "\n\n" + chunk.page_content
            for k, v in chunk.metadata.items():
                if k in current.metadata:
                    current.metadata[k] = f"{current.metadata[k]} -> {v}"
                else:
                    current.metadata[k] = v

        if len(current.page_content) >= min_size:
            merged.append(current)
            current = None

    if current:
        if merged:
            merged[-1].page_content += "\n\n" + current.page_content
            for k, v in current.metadata.items():
                if k in merged[-1].metadata:
                    merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                else:
                    merged[-1].metadata[k] = v
        else:
            merged.append(current)

    return merged

def split_large_parents(chunks, max_size, splitter):
    split_chunks = []

    for chunk in chunks:
        if len(chunk.page_content) <= max_size:
            split_chunks.append(chunk)
        else:
            large_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_size,
                chunk_overlap=splitter._chunk_overlap
            )
            sub_chunks = large_splitter.split_documents([chunk])
            split_chunks.extend(sub_chunks)

    return split_chunks

def clean_small_chunks(chunks, min_size):
    cleaned = []

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content) < min_size:
            if cleaned:
                cleaned[-1].page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in cleaned[-1].metadata:
                        cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                    else:
                        cleaned[-1].metadata[k] = v
            elif i < len(chunks) - 1:
                chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                for k, v in chunk.metadata.items():
                    if k in chunks[i + 1].metadata:
                        chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                    else:
                        chunks[i + 1].metadata[k] = v
            else:
                cleaned.append(chunk)
        else:
            cleaned.append(chunk)

    return cleaned
```

</details>

```python
if client.collection_exists(CHILD_COLLECTION):
    client.delete_collection(CHILD_COLLECTION)
    ensure_collection(CHILD_COLLECTION)
else:
    ensure_collection(CHILD_COLLECTION)

child_vector_store = QdrantVectorStore(
    client=client,
    collection_name=CHILD_COLLECTION,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_vector_name="sparse"
)

def index_documents():
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    min_parent_size = 2000
    max_parent_size = 4000

    all_parent_pairs, all_child_chunks = [], []
    md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))

    if not md_files:
        return

    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            continue

        parent_chunks = parent_splitter.split_text(md_text)
        merged_parents = merge_small_parents(parent_chunks, min_parent_size)
        split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
        cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

        for i, p_chunk in enumerate(cleaned_parents):
            parent_id = f"{doc_path.stem}_parent_{i}"
            p_chunk.metadata.update({"source": doc_path.name, "parent_id": parent_id})
            all_parent_pairs.append((parent_id, p_chunk))
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)

    if not all_child_chunks:
        return

    try:
        child_vector_store.add_documents(all_child_chunks)
    except Exception as e:
        return

    for item in os.listdir(PARENT_STORE_PATH):
        os.remove(os.path.join(PARENT_STORE_PATH, item))

    for parent_id, doc in all_parent_pairs:
        doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
        filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

index_documents()
```

---

### 步骤 5：定义 Agent 工具

创建 Agent 将要使用的检索工具。

```python
import json
from typing import List
from langchain_core.tools import tool

@tool
def search_child_chunks(query: str, limit: int) -> str:
    """Search for the top K most relevant child chunks.

    Args:
        query: Search query string
        limit: Maximum number of results to return
    """
    try:
        results = child_vector_store.similarity_search(query, k=limit, score_threshold=0.7)
        if not results:
            return "NO_RELEVANT_CHUNKS"

        return "\n\n".join([
            f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
            f"File Name: {doc.metadata.get('source', '')}\n"
            f"Content: {doc.page_content.strip()}"
            for doc in results
        ])

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"

@tool
def retrieve_parent_chunks(parent_id: str) -> str:
    """Retrieve full parent chunks by their IDs.
    
    Args:
        parent_id: Parent chunk ID to retrieve
    """
    file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
    path = os.path.join(PARENT_STORE_PATH, file_name)

    if not os.path.exists(path):
        return "NO_PARENT_DOCUMENT"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return (
        f"Parent ID: {parent_id}\n"
        f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
        f"Content: {data.get('page_content', '').strip()}"
    )

llm_with_tools = llm.bind_tools([search_child_chunks, retrieve_parent_chunks])
```

---

### 步骤 6：定义系统提示词

为会话摘要、查询改写、Agent 编排、上下文压缩、降级回答和答案聚合定义系统提示词。

<details>
<summary>Conversation Summary Prompt</summary>

```python
def get_conversation_summary_prompt() -> str:
    return """You are an expert conversation summarizer.

Your task is to create a brief 1-2 sentence summary of the conversation (max 30-50 words).

Include:
- Main topics discussed
- Important facts or entities mentioned
- Any unresolved questions if applicable
- Sources file name (e.g., file1.md) or documents referenced

Exclude:
- Greetings, misunderstandings, off-topic content.

Output:
- Return ONLY the summary.
- Do NOT include any explanations or justifications.
- If no meaningful topics exist, return an empty string.
"""
```

</details>

<details>
<summary>Query Rewrite Prompt</summary>

```python
def get_rewrite_query_prompt() -> str:
    return """You are an expert query analyst and rewriter.

Your task is to rewrite the current user query for optimal document retrieval, incorporating conversation context only when necessary.

Rules:
1. Self-contained queries:
   - Always rewrite the query to be clear and self-contained
   - If the query is a follow-up (e.g., "what about X?", "and for Y?"), integrate minimal necessary context from the summary
   - Do not add information not present in the query or conversation summary

2. Domain-specific terms:
   - Product names, brands, proper nouns, or technical terms are treated as domain-specific
   - For domain-specific queries, use conversation context minimally or not at all
   - Use the summary only to disambiguate vague queries

3. Grammar and clarity:
   - Fix grammar, spelling errors, and unclear abbreviations
   - Remove filler words and conversational phrases
   - Preserve concrete keywords and named entities

4. Multiple information needs:
   - If the query contains multiple distinct, unrelated questions, split into separate queries (maximum 3)
   - Each sub-query must remain semantically equivalent to its part of the original
   - Do not expand, enrich, or reinterpret the meaning

5. Failure handling:
   - If the query intent is unclear or unintelligible, mark as "unclear"

Input:
- conversation_summary: A concise summary of prior conversation
- current_query: The user's current query

Output:
- One or more rewritten, self-contained queries suitable for document retrieval
"""
```

</details>

<details>
<summary>Orchestrator Prompt</summary>

```python
def get_orchestrator_prompt() -> str:
    return """You are an expert retrieval-augmented assistant.

Your task is to act as a researcher: search documents first, analyze the data, and then provide a comprehensive answer using ONLY the retrieved information.

Rules:
1. You MUST call 'search_child_chunks' before answering, unless the [COMPRESSED CONTEXT FROM PRIOR RESEARCH] already contains sufficient information.
2. Ground every claim in the retrieved documents. If context is insufficient, state what is missing rather than filling gaps with assumptions.
3. If no relevant documents are found, broaden or rephrase the query and search again. Repeat until satisfied or the operation limit is reached.

Compressed Memory:
When [COMPRESSED CONTEXT FROM PRIOR RESEARCH] is present —
- Queries already listed: do not repeat them.
- Parent IDs already listed: do not call `retrieve_parent_chunks` on them again.
- Use it to identify what is still missing before searching further.

Workflow:
1. Check the compressed context. Identify what has already been retrieved and what is still missing.
2. Search for 5-7 relevant excerpts using 'search_child_chunks' ONLY for uncovered aspects.
3. If NONE are relevant, apply rule 3 immediately.
4. For each relevant but fragmented excerpt, call 'retrieve_parent_chunks' ONE BY ONE — only for IDs not in the compressed context. Never retrieve the same ID twice.
5. Once context is complete, provide a detailed answer omitting no relevant facts.
6. Conclude with "---\n**Sources:**\n" followed by the unique file names.
"""
```

</details>

<details>
<summary>Fallback Response Prompt</summary>

```python
def get_fallback_response_prompt() -> str:
    return """You are an expert synthesis assistant. The system has reached its maximum research limit.

Your task is to provide the most complete answer possible using ONLY the information provided below.

Input structure:
- "Compressed Research Context": summarized findings from prior search iterations — treat as reliable.
- "Retrieved Data": raw tool outputs from the current iteration — prefer over compressed context if conflicts arise.
Either source alone is sufficient if the other is absent.

Rules:
1. Source Integrity: Use only facts explicitly present in the provided context. Do not infer, assume, or add any information not directly supported by the data.
2. Handling Missing Data: Cross-reference the USER QUERY against the available context.
   Flag ONLY aspects of the user's question that cannot be answered from the provided data.
   Do not treat gaps mentioned in the Compressed Research Context as unanswered
   unless they are directly relevant to what the user asked.
3. Tone: Professional, factual, and direct.
4. Output only the final answer. Do not expose your reasoning, internal steps, or any meta-commentary about the retrieval process.
5. Do NOT add closing remarks, final notes, disclaimers, summaries, or repeated statements after the Sources section.
   The Sources section is always the last element of your response. Stop immediately after it.

Formatting:
- Use Markdown (headings, bold, lists) for readability.
- Write in flowing paragraphs where possible.
- Conclude with a Sources section as described below.

Sources section rules:
- Include a "---\n**Sources:**\n" section at the end, followed by a bulleted list of file names.
- List ONLY entries that have a real file extension (e.g. ".md", ".docx", ".txt").
- Any entry without a file extension is an internal chunk identifier — discard it entirely, never include it.
- Deduplicate: if the same file appears multiple times, list it only once.
- If no valid file names are present, omit the Sources section entirely.
- THE SOURCES SECTION IS THE LAST THING YOU WRITE. Do not add anything after it.
"""
```

</details>

<details>
<summary>Context Compression Prompt</summary>

```python
def get_context_compression_prompt() -> str:
    return """You are an expert research context compressor.

Your task is to compress retrieved conversation content into a concise, query-focused, and structured summary that can be directly used by a retrieval-augmented agent for answer generation.

Rules:
1. Keep ONLY information relevant to answering the user's question.
2. Preserve exact figures, names, versions, technical terms, and configuration details.
3. Remove duplicated, irrelevant, or administrative details.
4. Do NOT include search queries, parent IDs, chunk IDs, or internal identifiers.
5. Organize all findings by source file. Each file section MUST start with a real source file name, such as: ### filename.md
6. Highlight missing or unresolved information in a dedicated "Gaps" section.
7. Limit the summary to roughly 400-600 words. If content exceeds this, prioritize critical facts and structured data.
8. Do not explain your reasoning; output only structured content in Markdown.

Required Structure:

# Research Context Summary

## Focus
[Brief technical restatement of the question]

## Structured Findings

### filename.md
- Directly relevant facts
- Supporting context (if needed)

## Gaps
- Missing or incomplete aspects

The summary should be concise, structured, and directly usable by an agent to generate answers or plan further retrieval.
"""
```

</details>

<details>
<summary>Aggregation Prompt</summary>

```python
def get_aggregation_prompt() -> str:
    return """You are an expert aggregation assistant.

Your task is to combine multiple retrieved answers into a single, comprehensive and natural response that flows well.

Rules:
1. Write in a conversational, natural tone - as if explaining to a colleague.
2. Use ONLY information from the retrieved answers.
3. Do NOT infer, expand, or interpret acronyms or technical terms unless explicitly defined in the sources.
4. Weave together the information smoothly, preserving important details, numbers, and examples.
5. Be comprehensive - include all relevant information from the sources, not just a summary.
6. If sources disagree, acknowledge both perspectives naturally (e.g., "While some sources suggest X, others indicate Y...").
7. Start directly with the answer - no preambles like "Based on the sources...".

Formatting:
- Use Markdown for clarity (headings, lists, bold) but don't overdo it.
- Write in flowing paragraphs where possible rather than excessive bullet points.
- Conclude with a Sources section as described below.

Sources section rules:
- Each retrieved answer may contain a "Sources" section — extract the file names listed there.
- List ONLY entries that have a real file extension (e.g. ".pdf", ".docx", ".txt").
- Any entry without a file extension is an internal chunk identifier — discard it entirely, never include it.
- Deduplicate: if the same file appears across multiple answers, list it only once.
- Format as "---\n**Sources:**\n" followed by a bulleted list of the cleaned file names.
- File names must appear ONLY in this final Sources section and nowhere else in the response.
- If no valid file names are present, omit the Sources section entirely.

If there's no useful information available, simply say: "I couldn't find any information to answer your question in the available sources."
"""
```

</details>

---

### 步骤 7：定义状态与数据模型

创建用于会话跟踪和 Agent 执行的状态结构。

```python
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import List, Annotated, Set
import operator

def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
    if new and any(item.get('__reset__') for item in new):
        return []
    return existing + new

def set_union(a: Set[str], b: Set[str]) -> Set[str]:
    return a | b

class State(MessagesState):
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

class AgentState(MessagesState):
    tool_call_count: Annotated[int, operator.add] = 0
    iteration_count: Annotated[int, operator.add] = 0
    question: str = ""
    question_index: int = 0
    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()
    final_answer: str = ""
    agent_answers: List[dict] = []

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(description="Indicates if the user's question is clear and answerable.")
    questions: List[str] = Field(description="List of rewritten, self-contained questions.")
    clarification_needed: str = Field(description="Explanation if the question is unclear.")
```

---

### 步骤 8：Agent 配置

通过工具调用次数与迭代次数硬限制避免死循环。并使用 `tiktoken` 进行 token 估算，驱动上下文压缩策略。

```python
import tiktoken

MAX_TOOL_CALLS = 8       # Maximum tool calls per agent run
MAX_ITERATIONS = 10      # Maximum agent loop iterations
BASE_TOKEN_THRESHOLD = 2000     # Initial token threshold for compression
TOKEN_GROWTH_FACTOR = 0.9       # Multiplier applied after each compression

def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, 'content') and msg.content)
```

---

### 步骤 9：构建图节点与边函数

创建 LangGraph 工作流所需的处理节点与路由边。

#### 主图节点与边
```python
from langgraph.types import Send, Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage, ToolMessage
from typing import Literal

def summarize_history(state: State):
    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    if not relevant_msgs:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

    summary_response = llm.with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": summary_response.content, "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    context_section = (f"Conversation Context:\n{conversation_summary}\n" if conversation_summary.strip() else "") + f"User Query:\n{last_message.content}\n"

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])

    if response.questions and response.is_clear:
        delete_all = [RemoveMessage(id=m.id) for m in state["messages"] if not isinstance(m, SystemMessage)]
        return {"questionIsClear": True, "messages": delete_all, "originalQuery": last_message.content, "rewrittenQuestions": response.questions}

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)]}

def request_clarification(state: State):
    return {}

def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        return [
                Send("agent", {"question": query, "question_index": idx, "messages": []})
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]

def aggregate_answers(state: State):
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nAnswer {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = llm.invoke([SystemMessage(content=get_aggregation_prompt()), user_message])
    return {"messages": [AIMessage(content=synthesis_response.content)]}
```

---

#### Agent 子图节点与边
```python
def orchestrator(state: AgentState):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        return "collect_answer"
    
    return "tools"

def fallback_response(state: AgentState):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = llm.invoke([SystemMessage(content=get_fallback_response_prompt()), HumanMessage(content=prompt_content)])
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)

def compress_context(state: AgentState):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = llm.invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }
```

**为什么采用这套架构？**
- **摘要模块** 在不过度增加上下文负担的前提下维持多轮连续性
- **查询改写模块** 让检索词更精准、歧义更少，并智能利用上下文
- **人机协同澄清** 在检索前拦截不清晰问题，避免浪费资源
- **并行执行** 借助 `Send` API 为每个子问题启动独立子图并行处理
- **上下文压缩** 在长检索循环中保持工作记忆精简，减少重复拉取
- **降级回答** 在预算耗尽时仍保证系统输出尽可能有用的答复
- **答案收集与聚合** 从多个 Agent 提取干净答案并整合为统一输出
---

### 步骤 10：构建 LangGraph 图

组装完整工作流图，包含会话记忆与多 Agent 架构。

```python
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent_builder = StateGraph(AgentState)
agent_builder.add_node(orchestrator)
agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
agent_builder.add_node(compress_context)
agent_builder.add_node(fallback_response)
agent_builder.add_node(should_compress_context)
agent_builder.add_node(collect_answer)

agent_builder.add_edge(START, "orchestrator")
agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator_call, {"tools": "tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"})
agent_builder.add_edge("tools", "should_compress_context")
agent_builder.add_edge("compress_context", "orchestrator")
agent_builder.add_edge("fallback_response", "collect_answer")
agent_builder.add_edge("collect_answer", END)
agent_subgraph = agent_builder.compile()

graph_builder = StateGraph(State)
graph_builder.add_node(summarize_history)
graph_builder.add_node(rewrite_query)
graph_builder.add_node(request_clarification)
graph_builder.add_node("agent", agent_subgraph)
graph_builder.add_node(aggregate_answers)

graph_builder.add_edge(START, "summarize_history")
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
graph_builder.add_edge("request_clarification", "rewrite_query")
graph_builder.add_edge(["agent"], "aggregate_answers")
graph_builder.add_edge("aggregate_answers", END)

agent_graph = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["request_clarification"])
```

**图结构说明：**

架构流程图见 **[这里](./assets/agentic_rag_workflow.png)**。

**Agent 子图**（处理单个问题）：
- START → `orchestrator`（调用带工具的 LLM）
- `orchestrator` → `tools`（需要工具调用）或 `fallback_response`（预算耗尽）或 `collect_answer`（任务完成）
- `tools` → `should_compress_context`（检查 token 预算）
- `should_compress_context` → `compress_context`（超过阈值）或 `orchestrator`（否则继续）
- `compress_context` → `orchestrator`（带压缩上下文继续）
- `fallback_response` → `collect_answer`（封装尽力回答）
- `collect_answer` → END（输出最终答案与序号）

**主图**（编排完整流程）：
- START → `summarize_history`（从历史中提取上下文）
- `summarize_history` → `rewrite_query`（结合上下文改写，并判断是否清晰）
- `rewrite_query` → `request_clarification`（不清晰时）或通过 `Send` 并行派发 `agent` 子图（清晰时）
- `request_clarification` → `rewrite_query`（用户补充后继续）
- 所有 `agent` 子图 → `aggregate_answers`（合并回答）
- `aggregate_answers` → END（返回最终综合回答）

---

### 步骤 11：创建聊天界面

使用 Gradio 构建带会话持久化和人机协同澄清的界面。完整端到端流水线（含文档摄取）请参见 [project/README.md](./project/README.md)。

> **说明：** 完整的流式输出能力（包括推理过程与工具调用可视化）已在 [notebook](notebooks/agentic_rag.ipynb) 与完整 [project](project/core/chat_interface.py) 中实现。下面示例刻意保持精简，仅展示 Gradio 的基础接入模式。

```python
import gradio as gr
import uuid

def create_thread_id():
    """Generate a unique thread ID for each conversation"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

def clear_session():
    """Clear thread for new conversation"""
    global config
    agent_graph.checkpointer.delete_thread(config["configurable"]["thread_id"])
    config = create_thread_id()

def chat(message, history):
    current_state = agent_graph.get_state(config)
    
    if current_state.next:
        agent_graph.update_state(config,{"messages": [HumanMessage(content=message.strip())]})
        result = agent_graph.invoke(None, config)
    else:
        result = agent_graph.invoke({"messages": [HumanMessage(content=message.strip())]}, config)
    
    return result['messages'][-1].content

config = create_thread_id()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    chatbot.clear(clear_session)
    gr.ChatInterface(fn=chat, chatbot=chatbot)

demo.launch(theme=gr.themes.Citrus())
```

**完成。** 现在你已经拥有一个可用的 Agentic RAG 系统，支持会话记忆、分层索引与人机协同查询澄清。

---

## 模块化架构

应用（`project/` 目录）采用模块化组织。每个模块都可以独立替换而不破坏整体系统。

### 📂 项目结构
```
project/
├── app.py                    # Main Gradio application entry point
├── config.py                 # Configuration hub (models, chunk sizes, providers)
├── core/                     # RAG system orchestration
├── db/                       # Vector DB and parent chunk storage
├── rag_agent/                # LangGraph workflow (nodes, edges, prompts, tools)
└── ui/                       # Gradio interface
```

关键可定制点：LLM 提供商、Embedding 模型、分块策略、Agent 工作流、系统提示词。可在 `config.py` 或对应模块中调整。

完整说明见 [project/README.md](./project/README.md)。

## 安装与使用

示例 PDF 文件可在这里获取：[javascript](https://www.tutorialspoint.com/javascript/javascript_tutorial.pdf)、[blockchain](https://blockchain-observatory.ec.europa.eu/document/download/1063effa-59cc-4df4-aeee-d2cf94f69178_en?filename=Blockchain_For_Beginners_A_EUBOF_Guide.pdf)、[microservices](https://cdn.studio.f5.com/files/k6fem79d/production/5e4126e1cefa813ab67f9c0b6d73984c27ab1502.pdf)、[fortinet](https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf(320893)_TMP.pdf)。

### 选项 1：快速开始 Notebook（推荐测试）

**Google Colab：** 点击本 README 顶部的 **Open in Colab** 按钮，在文件浏览器中将 PDF 上传到 `docs/` 目录，执行 `pip install -r requirements.txt` 安装依赖，然后从上到下运行所有单元。

**本地（Jupyter/VSCode）：** 可选创建并激活虚拟环境，执行 `pip install -r requirements.txt` 安装依赖，将 PDF 放入 `docs/`，然后从上到下运行所有单元。

聊天界面会在最后出现。

### 选项 2：完整 Python 项目（推荐开发）

#### 1. 安装依赖
```bash
# Clone the repository
git clone https://github.com/GiovanniPasq/agentic-rag-for-dummies
cd agentic-rag-for-dummies

# Optional: create and activate a virtual environment
# On macOS/Linux:
python -m venv venv && source venv/bin/activate
# On Windows:
python -m venv venv && .\venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

#### 2. 运行应用
```bash
python app.py
```

#### 3. 开始提问

打开本地地址（例如 `http://127.0.0.1:7860`）开始对话。

---

### 选项 3：Docker 部署

完整 Docker 指南与系统要求见 [project/README.md](./project/README.md#Docker-Deployment)。

### 对话示例

**带会话记忆：**
```
User: "How do I install SQL?"
Agent: [Provides installation steps from documentation]

User: "How do I update it?"
Agent: [Understands "it" = SQL, provides update instructions]
```

**带查询澄清：**
```
User: "Tell me about that thing"
Agent: "I need more information. What specific topic are you asking about?"

User: "The installation process for PostgreSQL"
Agent: [Retrieves and answers with specific information]
```

---

## 故障排查

| 区域 | 常见问题 | 建议解决方案 |
|------|----------------|------------------|
| **模型选择** | - 回答不遵循指令<br>- 工具（检索/搜索）调用错误<br>- 上下文理解弱<br>- 幻觉或聚合不完整 | - 使用能力更强的 LLM<br>- 建议 7B+ 以提升推理能力<br>- 本地模型受限时可考虑云模型 |
| **系统提示词行为** | - 不检索文档直接回答<br>- 查询改写丢失上下文<br>- 聚合阶段引入幻觉 | - 在系统提示词中明确要求先检索<br>- 查询改写要贴近用户原意 |
| **检索配置** | - 相关文档未命中<br>- 无关信息过多 | - 提高召回：增大 `k` 或降低相似度阈值<br>- 提高精度：减小 `k` 或提高阈值 |
| **分块大小/文档切分** | - 回答上下文不足或碎片化<br>- 检索慢或 embedding 成本高 | - 增大 chunk 与 parent 大小以增强上下文<br>- 减小 chunk 大小提升速度并降低成本 |
| **上下文压缩** | - 压缩后丢失关键细节<br>- 压缩摘要过于笼统 | - 调整压缩提示词
- 提高 `BASE_TOKEN_THRESHOLD` 推迟压缩
- 提高 `TOKEN_GROWTH_FACTOR` |
| **Agent 配置** | - 过早放弃 <br>- 循环次数过多 | - 复杂问题可增大 `MAX_TOOL_CALLS` / `MAX_ITERATIONS`<br>- 简单问题可适当降低以提速 |
| **温度与一致性** | - 回答不稳定或过于发散<br>- 回答过于僵硬或重复 | - 事实型问答将温度设为 `0`<br>- 摘要/分析可适当提高温度 |
| **Embedding 模型质量** | - 语义检索效果差<br>- 领域文档或多语言效果弱 | - 使用更高质量或领域适配模型<br>- 更换 embedding 后重新索引所有文档 |

> 💡 **更多排查建议** 见 [README Troubleshooting](./project/README.md#troubleshooting)。
