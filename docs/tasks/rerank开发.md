# Cross-Encoder Reranker 开发文档

> 面向 Codex 的开发任务说明。  
> **开发前必须先阅读项目中的 skill / README / 架构说明文件，确认当前目录结构、运行方式、依赖管理方式和已有 RAG 流程。不要直接凭经验改代码。**

---

## 1. 背景与目标

当前项目已经具备基础 RAG 能力：

- PDF → Markdown 转换
- 父子分块 parent-child chunking
- Qdrant 向量数据库
- Dense embedding 检索
- Sparse BM25 检索
- Hybrid retrieval / RRF 融合
- LangGraph Agent 编排
- `search_child_chunks` / `retrieve_parent_chunks` 检索工具

目前加入 RRF 后，检索效果提升不明显。原因是 RRF 主要解决“多路召回结果如何融合”，但不会真正判断每个 chunk 是否能回答用户问题。

因此需要新增 **Cross-Encoder Reranker 精排模块**，在初步召回后，对候选 child chunks 进行重新排序，提高最终进入 LLM 上下文的相关性和纯度。

---

## 2. 总体设计原则

### 2.1 不调用外部 API

本次实现要求：

- 从 Hugging Face 下载 cross-encoder reranker 模型
- 模型在本地运行
- 不调用 Cohere、Jina、OpenAI、Voyage 等远程 rerank API
- 不需要配置 API Key

推荐第一版模型：

```python
RERANKER_MODEL = "BAAI/bge-reranker-base"
```

可选更高质量模型：

```python
RERANKER_MODEL = "BAAI/bge-reranker-large"
```

第一版优先使用 `BAAI/bge-reranker-base`，因为它更轻，适合验证 rerank 对当前项目的收益。

---

### 2.2 rerank 只处理召回后的候选，不处理全库

不要对 Qdrant 全量文档进行 rerank。

推荐流程：

```text
User Query
→ Query Rewrite
→ Dense Recall + Sparse Recall
→ RRF / Hybrid Fusion
→ Candidate Child Chunks top N
→ Cross-Encoder Rerank
→ Reranked Child Chunks top M
→ Retrieve Parent Chunks
→ Context Assembly
→ LLM Answer Generation
```

推荐参数：

```python
RETRIEVAL_CANDIDATE_K = 30      # 初始召回候选数量
RERANK_TOP_N = 20               # 送入 reranker 的候选数量
RERANK_FINAL_TOP_K = 5          # rerank 后返回给 Agent 的 child chunk 数量
RERANK_SCORE_THRESHOLD = None   # 第一版不强制阈值，先只按 top_k 截断
```

后续可根据评测结果调整。

---

### 2.3 rerank child chunks，不 rerank parent chunks

当前项目使用 parent-child chunking：

- child chunk：短文本，用于精准定位
- parent chunk：长文本，用于补充完整上下文

reranker 应该作用在 child chunks 上，而不是 parent chunks 上。

原因：

- child chunk 更短，cross-encoder 判断更准确
- parent chunk 较长，相关性容易被无关段落稀释
- 先 rerank child，再通过 `parent_id` 拉取 parent，符合当前架构

---

## 3. 需要新增或修改的文件

具体文件名以项目实际结构为准，开发前先阅读项目目录和已有实现。

建议新增：

```text
project/rag_agent/reranker.py
```

建议修改：

```text
project/config.py
project/rag_agent/tools.py
requirements.txt
```

如果当前项目已经有 retrieval service、vector db manager 或 tool factory，也可以将 reranker 接入到已有检索服务层，不强制必须放在 `tools.py` 中。

---

## 4. 配置项设计

在 `project/config.py` 中新增 reranker 配置。

```python
# --- Cross-Encoder Reranker Configuration ---
RERANKER_ENABLED = True
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_DEVICE = "auto"  # "auto", "cpu", "cuda", "mps"
RERANKER_BATCH_SIZE = 8
RERANKER_TOP_N = 20
RERANKER_FINAL_TOP_K = 5
RERANKER_MAX_LENGTH = 512
RERANKER_SCORE_THRESHOLD = None
```

说明：

- `RERANKER_ENABLED`：允许一键关闭 rerank，方便 A/B 测试。
- `RERANKER_MODEL`：Hugging Face 模型名。
- `RERANKER_DEVICE`：自动检测或显式指定设备。
- `RERANKER_BATCH_SIZE`：控制推理批大小，避免内存不足。
- `RERANKER_TOP_N`：从召回结果中取前 N 个送入 reranker。
- `RERANKER_FINAL_TOP_K`：rerank 后最终返回给 Agent 的 child chunk 数量。
- `RERANKER_MAX_LENGTH`：cross-encoder 输入最大长度。
- `RERANKER_SCORE_THRESHOLD`：第一版可以为 `None`，后续根据评测再启用。

---

## 5. 依赖设计

在 `requirements.txt` 中确认或新增：

```text
sentence-transformers
transformers
torch
```

如果项目已经依赖 `langchain-huggingface`，仍然需要确认是否已包含 cross-encoder 所需依赖。不要假设已有依赖一定完整。

注意：

- Hugging Face 模型会自动下载到本地缓存。
- 不需要 Hugging Face API Key，除非使用私有模型。
- Windows / macOS / Linux 路径不要写死。
- Docker 环境下建议保留 Hugging Face cache，避免每次重建重复下载模型。

---

## 6. Reranker 模块接口设计

新增 `project/rag_agent/reranker.py`。

推荐实现一个独立类：

```python
class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ):
        ...

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list:
        ...
```

其中 `documents` 建议是 LangChain `Document` 列表，或者项目已有的 chunk 数据结构。

输入示例：

```python
query = "如何修改 embedding 模型？"

documents = [
    Document(
        page_content="更换 Embedding 模型的步骤...",
        metadata={"parent_id": "xxx", "source": "file.pdf"}
    ),
    Document(
        page_content="LLM Provider 配置...",
        metadata={"parent_id": "yyy", "source": "file.pdf"}
    ),
]
```

输出要求：

- 保持原有 Document 结构
- 在 `metadata` 中新增：

```python
metadata["rerank_score"] = float(score)
metadata["rerank_rank"] = int(rank)
```

可选保留原始检索分数：

```python
metadata["retrieval_score"] = original_score
metadata["retrieval_rank"] = original_rank
```

如果原始检索结果没有分数，也不要强行伪造。

---

## 7. 推荐实现方式

### 7.1 使用 sentence-transformers CrossEncoder

推荐第一版使用：

```python
from sentence_transformers import CrossEncoder
```

初始化：

```python
self.model = CrossEncoder(
    model_name,
    device=resolved_device,
    max_length=max_length,
)
```

推理：

```python
pairs = [(query, doc.page_content) for doc in documents]
scores = self.model.predict(pairs, batch_size=batch_size)
```

排序：

```python
ranked = sorted(
    zip(documents, scores),
    key=lambda item: float(item[1]),
    reverse=True,
)
```

返回 top_k。

---

### 7.2 设备选择

实现一个设备解析函数：

```python
def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

注意：

- CPU 可以运行，但会慢。
- CUDA 优先。
- macOS 可尝试 MPS，但要允许失败回退 CPU。

---

## 8. 与检索工具的接入位置

当前项目通常有类似工具：

```python
@tool
def search_child_chunks(query: str, limit: int) -> str:
    results = child_vector_store.similarity_search(query, k=limit, score_threshold=0.7)
    ...
```

需要改成：

```text
1. 先从 Qdrant / hybrid retrieval 获取较多候选
2. 如果 RERANKER_ENABLED=True，则对候选 child chunks 做 rerank
3. 返回 rerank 后的 top_k child chunks
4. 输出中包含 rerank_score，便于调试和观测
```

伪代码：

```python
@tool
def search_child_chunks(query: str, limit: int) -> str:
    try:
        candidate_k = max(limit, config.RERANKER_TOP_N)

        results = child_vector_store.similarity_search(
            query,
            k=candidate_k,
            score_threshold=0.7,
        )

        if not results:
            return "NO_RELEVANT_CHUNKS"

        if config.RERANKER_ENABLED:
            results = reranker.rerank(
                query=query,
                documents=results[:config.RERANKER_TOP_N],
                top_k=min(limit, config.RERANKER_FINAL_TOP_K),
                score_threshold=config.RERANKER_SCORE_THRESHOLD,
            )
        else:
            results = results[:limit]

        return format_child_chunk_results(results)

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"
```

注意：

- 不要破坏现有工具返回格式。
- Agent prompt 中可能依赖 `Parent ID`、`File Name`、`Content`，这些字段必须保留。
- 可以新增 `Rerank Score` 字段，但不要删除原有字段。

推荐输出格式：

```text
Parent ID: xxx
File Name: file.pdf
Rerank Score: 0.8734
Content: ...
```

如果未启用 rerank，则不输出 `Rerank Score` 或输出 `Rerank Score: N/A` 均可，但要保持格式稳定。

---

## 9. Lazy Loading 设计

不要在模块 import 时立即加载 reranker 模型。

原因：

- 模型加载较慢
- 单元测试可能不需要加载模型
- CLI / UI 启动时不应被非必要模型阻塞

推荐使用懒加载：

```python
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model_name=config.RERANKER_MODEL,
            device=config.RERANKER_DEVICE,
            batch_size=config.RERANKER_BATCH_SIZE,
            max_length=config.RERANKER_MAX_LENGTH,
        )
    return _reranker
```

在第一次真正执行 rerank 时再加载模型。

---

## 10. 错误处理与降级策略

reranker 失败时，不应该导致整个 RAG 系统不可用。

必须实现降级：

```text
如果 reranker 加载失败 / 推理失败：
→ 记录错误日志
→ 返回原始检索结果 top_k
→ 不中断问答流程
```

伪代码：

```python
try:
    results = get_reranker().rerank(...)
except Exception as e:
    logger.exception("Rerank failed, fallback to original retrieval results")
    results = original_results[:limit]
```

不要把完整 traceback 暴露给最终用户。

---

## 11. 日志与可观测性

需要增加基础日志，便于分析 rerank 是否有效。

建议记录：

```text
query
retrieval candidate count
rerank input count
rerank output count
device
model name
top rerank scores
rerank latency_ms
fallback 是否发生
```

如果项目已经接入 Langfuse 或其他 observability，可以后续把 rerank 作为 span / observation 记录。第一版只要求 Python logger 即可。

示例日志：

```text
Reranker enabled: model=BAAI/bge-reranker-base device=cuda input=20 output=5 latency_ms=183.4
Top rerank scores: [0.91, 0.86, 0.73, 0.68, 0.61]
```

---

## 12. 测试要求

### 12.1 单元测试

需要覆盖：

1. `CrossEncoderReranker.rerank()` 能返回按分数降序排列的结果
2. 输出 Document metadata 中包含 `rerank_score` 和 `rerank_rank`
3. `top_k` 生效
4. `score_threshold` 生效，如果启用
5. reranker 抛异常时，检索工具能回退到原始结果
6. `RERANKER_ENABLED=False` 时不调用 reranker

为了避免测试下载真实模型，可以 mock `CrossEncoder.predict()`。

---

### 12.2 集成测试

使用少量测试文档验证完整链路：

```text
上传 / 索引文档
→ 提问
→ search_child_chunks 返回包含 Rerank Score 的结果
→ Agent 能继续 retrieve_parent_chunks
→ 最终回答正常生成
```

重点确认：

- 工具输出格式没有破坏
- parent_id 没丢失
- source 文件名没丢失
- rerank 后仍能正确拉取 parent chunk

---

### 12.3 A/B 测试

保留配置开关：

```python
RERANKER_ENABLED = False
RERANKER_ENABLED = True
```

分别运行同一批问题，对比：

- Context Precision
- MRR
- NDCG
- Answer Correctness
- Faithfulness
- 最终答案是否更少答偏

注意：

- rerank 通常不会提高 Recall@K，因为它不负责扩大召回范围。
- rerank 更可能提升排序质量、上下文精度和最终答案质量。

---

## 13. 性能注意事项

Cross-Encoder 是 query-document pair 级别推理，不能像 embedding 那样提前预计算全库向量。

每次查询的计算量大约是：

```text
rerank_input_count × cross_encoder_forward
```

因此必须控制 `RERANKER_TOP_N`。

建议第一版：

```python
RERANKER_TOP_N = 20
RERANKER_FINAL_TOP_K = 5
RERANKER_BATCH_SIZE = 8
```

如果 CPU 很慢，可以降低：

```python
RERANKER_TOP_N = 10
RERANKER_FINAL_TOP_K = 5
RERANKER_BATCH_SIZE = 4
```

如果 GPU 可用，可以提高：

```python
RERANKER_TOP_N = 30
RERANKER_FINAL_TOP_K = 8
RERANKER_BATCH_SIZE = 16
```

---

## 14. 与 RRF 的关系

不要用 rerank 替代 RRF。

推荐关系：

```text
Dense Retrieval + Sparse Retrieval
→ RRF Fusion
→ Cross-Encoder Rerank
→ Final Top K
```

解释：

- Dense / Sparse：提供不同角度的召回
- RRF：融合多个召回列表
- Rerank：对融合后的候选进行深度相关性判断

如果当前项目的 Qdrant hybrid retrieval 已经内部做融合，也可以先在现有 `similarity_search` 结果后接 rerank。后续再把显式 RRF 与 rerank 结合。

---

## 15. 不要做的事情

本次任务不要做：

- 不要调用远程 rerank API
- 不要把 reranker 接到全库扫描
- 不要 rerank parent chunks 作为第一版方案
- 不要删除现有 hybrid retrieval / RRF 逻辑
- 不要破坏 `search_child_chunks` 和 `retrieve_parent_chunks` 的工具契约
- 不要让 reranker 失败导致整个 Agent 失败
- 不要在 import 阶段直接下载 / 加载模型
- 不要把 rerank 分数当作最终答案置信度

---

## 16. 验收标准

完成后应满足：

1. `project/config.py` 中可以配置 reranker 开关和模型名。
2. 默认使用 Hugging Face 本地模型 `BAAI/bge-reranker-base`。
3. 首次使用时模型自动下载到本地 Hugging Face cache。
4. 不需要任何 rerank API Key。
5. `search_child_chunks` 返回结果经过 rerank 排序。
6. 工具输出保留 `Parent ID`、`File Name`、`Content`。
7. 启用 rerank 时输出中可看到 `Rerank Score`。
8. reranker 异常时自动回退到原始检索结果。
9. 可以通过 `RERANKER_ENABLED=False` 关闭 rerank 做对照实验。
10. 单元测试和基础集成测试通过。

---

## 17. 推荐实现顺序

1. 阅读项目 skill、README、`project/config.py`、`project/rag_agent/tools.py`、`project/db/vector_db_manager.py`。
2. 确认当前 `search_child_chunks` 的输入输出格式。
3. 新增 reranker 配置项。
4. 新增 `project/rag_agent/reranker.py`。
5. 实现 `CrossEncoderReranker` 与 lazy loading。
6. 在 `search_child_chunks` 中接入 rerank。
7. 增加异常回退逻辑。
8. 增加日志。
9. 修改 `requirements.txt`。
10. 编写 mock 单元测试。
11. 做少量真实文档集成测试。
12. 用同一批问题对比 rerank 开关前后的效果。

---

## 18. 给 Codex 的最终任务描述

请在当前 Agentic RAG 项目中新增一个本地 Hugging Face Cross-Encoder Reranker 模块。实现时必须先阅读项目 skill / README / 现有架构，确认当前检索工具和配置结构。reranker 使用 `sentence-transformers` 从 Hugging Face 加载 `BAAI/bge-reranker-base`，不调用任何外部 rerank API。将 reranker 接入 child chunk 检索结果之后、parent chunk 拉取之前。保留现有 `Parent ID`、`File Name`、`Content` 输出格式，并新增可观测的 `Rerank Score`。支持配置开关、lazy loading、异常回退、基础日志和 mock 单元测试。最终确保 reranker 失败时系统仍能使用原始检索结果正常回答。
