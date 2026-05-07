# Knowledge Fallback 降级策略任务文档

## 背景

当前 RAG Agent 在用户问题被识别为 `rag_qa` 后，会进入查询改写、任务规划、`rag_research` 检索、单任务回答、答案评估和最终聚合流程。

当知识库中没有可用知识时，当前系统倾向于要求模型只基于检索证据回答，并在证据不足时说明缺失信息。这样可以保证 RAG 答案不幻觉，但用户体验上会出现一个问题：

- 用户提出的是一个可以由通用知识回答的问题。
- 知识库没有命中或命中内容质量很弱。
- 系统只返回“知识库中没有找到相关信息”，没有继续给出可参考回答。

本任务目标是在 RAG 证据不可用时增加一个明确的 `knowledge_fallback` 降级策略：先说明知识库未找到可用知识，再允许模型基于通用知识回答用户问题。

## 已确认约束

1. 触发降级的条件包括：
   - rerank 分数低于阈值。
   - LLM 判断当前检索证据不足。
   - 多次重试后仍无法基于知识库回答。

2. 降级类型命名为 `knowledge_fallback`。

3. 降级回答允许模型基于通用知识回答。

4. 降级回答不输出 `Sources`。

5. 多任务场景下，每个子任务独立判断是否降级。

6. 检索失败也进入降级流程。

7. 不需要新增配置开关。

8. 不立即降级，应先按现有机制重试；重试后仍不可回答再降级。

## 总体目标

新增 `knowledge_fallback` 策略，使单个 RAG 子任务在知识库证据不可用时，能够返回更有用的回答。

核心目标：

- 保留现有 `rag_qa` 意图识别和任务规划流程。
- 保留现有 RAG 检索优先策略。
- 在证据不足、低分或检索失败时，先执行有限重试。
- 重试后仍不可回答时，进入 `knowledge_fallback`。
- 降级回答明确声明知识库未找到可用信息。
- 降级回答基于模型通用知识生成。
- 降级回答不输出知识库 `Sources`。
- 多子任务独立降级，不影响其他子任务使用正常 RAG 答案。

## 非目标

本任务不处理以下内容：

- 不把用户原始意图改成 `chitchat`。
- 不修改闲聊分支的语义。
- 不新增全局配置开关。
- 不改变文档入库、chunking、向量化流程。
- 不引入外部联网搜索。
- 不要求降级回答提供知识库引用。
- 不把模型通用知识混入 RAG 证据上下文。

## 命名和语义

### 为什么不复用 chitchat

`chitchat` 表示用户意图本身是闲聊、打招呼或普通交流，不需要检索。

`knowledge_fallback` 表示用户意图仍然是 `rag_qa`，只是 RAG 证据不可用，因此答案来源从知识库降级为模型通用知识。

两者语义不同，不应共用同一个分支或 prompt。

建议保留如下语义：

```text
rag_qa
  -> 优先使用知识库回答
  -> 知识库证据不可用
  -> knowledge_fallback
  -> 说明未找到知识库信息
  -> 基于通用知识回答

chitchat
  -> 不检索
  -> 直接普通对话回复
```

## 推荐处理流程

单任务执行流程建议调整为：

```text
task_executor
  -> rag_research
  -> evaluate_retrieval_evidence
      -> evidence_sufficient: answer_from_rag
      -> evidence_insufficient: retry_rag_research
      -> retrieval_error: retry_or_fallback
  -> evaluate_answer
      -> satisfactory: collect_answer
      -> unsatisfactory and retry_budget_available: task_executor
      -> unsatisfactory and retry_budget_exhausted: knowledge_fallback
  -> collect_answer
```

主图流程不需要把整体意图改成 `chitchat`：

```text
recognize_intent
  -> rag_qa
  -> rewrite_query
  -> plan_rag_tasks
  -> task_executor subgraph
      -> 每个子任务独立决定 rag_answer 或 knowledge_fallback
  -> aggregate_answers
```

## 降级触发条件

建议将降级触发分成三类。

### 1. Rerank 低分

当 rerank 开启时，应对最终 rerank 结果设置最低可用证据阈值。

建议判断：

```text
reranked_contexts == 0
or max(rerank_score) < evidence_score_threshold
```

注意：

- 当前 `RERANKER_SCORE_THRESHOLD` 默认为 `None`，实现时需要明确一个代码内阈值或复用现有配置项。
- 由于用户确认“不需要新增配置开关”，阈值可以作为模块内常量或直接复用现有 `RERANKER_SCORE_THRESHOLD`。
- 如果继续使用 `RERANKER_SCORE_THRESHOLD`，应定义 `None` 时的默认行为，避免低质量召回永远被当成可用证据。

### 2. LLM 判断证据不足

即使检索返回了上下文，也可能与用户问题无关。

应让任务执行 LLM 或答案评估 LLM 明确判断：

```text
retrieved contexts cannot answer the question
```

如果 LLM 判断证据不足，应优先触发聚焦重试，而不是立即降级。

### 3. 重试后仍无法回答

当达到现有重试上限后仍无法生成满意 RAG 答案，应进入 `knowledge_fallback`。

建议复用现有限制：

```text
MAX_ANSWER_EVALUATION_RETRIES
MAX_ITERATIONS
MAX_TOOL_CALLS
```

只要重试预算耗尽且答案仍不可基于知识库完成，就进入降级，而不是返回纯粹的无结果回答。

## 检索失败处理

检索失败包括但不限于：

- `rag_research` 返回 `RAG_RESEARCH_ERROR`。
- vector db 查询异常。
- parent store 读取异常。
- reranker 异常且无法产生可用上下文。
- tool 调用异常。

处理策略：

```text
检索失败
  -> 如果还有重试预算，允许重试
  -> 如果重试后仍失败，进入 knowledge_fallback
```

降级回答中可以说明：

```text
知识库检索未能返回可用信息，我先基于通用知识回答。
```

不建议向用户暴露内部错误栈、工具名、数据库名或节点名。

## 降级回答格式

降级回答应满足：

- 开头明确说明知识库没有找到可用信息。
- 后续基于通用知识直接回答问题。
- 不输出 `Sources`。
- 不声称答案来自知识库。
- 不暴露内部节点、工具调用、rerank 分数或 retry 细节。

推荐格式：

```text
我没有在知识库中找到可用于回答这个问题的信息。以下是基于通用知识的回答：

[模型基于通用知识给出的回答]
```

如果是检索失败导致降级：

```text
知识库检索没有返回可用信息。以下是基于通用知识的回答：

[模型基于通用知识给出的回答]
```

## 多任务聚合策略

多任务场景下，每个子任务独立决定是否降级。

示例：

```text
task_1: 找到知识库证据 -> RAG 答案，允许 Sources
task_2: 未找到可用证据 -> knowledge_fallback 答案，不输出 Sources
task_3: 检索失败 -> knowledge_fallback 答案，不输出 Sources
```

聚合阶段需要保留每个子任务答案的来源类型，避免把通用知识答案误标成知识库来源。

建议每个 `TaskResult` 增加或规范化以下字段：

```text
answer_mode: rag_qa | knowledge_fallback
used_knowledge_base: bool
fallback_reason: str
sources: list[str]
```

字段语义：

- `answer_mode="rag_qa"`：答案基于知识库证据。
- `answer_mode="knowledge_fallback"`：答案基于模型通用知识。
- `used_knowledge_base=false`：聚合时不得输出该子任务的 Sources。
- `fallback_reason`：仅用于诊断或日志，不直接暴露给用户。
- `sources`：仅包含真实来源文件名。

聚合 prompt 需要明确：

- RAG 子答案可以保留 Sources。
- `knowledge_fallback` 子答案没有 Sources。
- 不要为降级答案编造 Sources。
- 如果最终答案混合了知识库内容和通用知识，应在自然语言中区分。

## 状态和诊断设计

建议在子图状态或任务结果中记录降级信息，便于 UI、日志和评测使用。

建议字段：

```text
fallback_triggered: bool
fallback_reason: str
answer_mode: str
retrieval_evidence_status: sufficient | insufficient | low_score | error | exhausted
best_rerank_score: float | None
retry_count_before_fallback: int
```

这些字段主要用于 diagnostics，不需要展示给最终用户。

## Prompt 调整建议

### Task Executor Prompt

需要明确：

- 默认必须优先调用 `rag_research`。
- 证据不足时先重试。
- 重试预算耗尽后，不要继续假装使用知识库回答。
- 如果进入 `knowledge_fallback`，可以基于通用知识回答，但必须声明知识库未找到可用信息。

### Answer Evaluation Prompt

需要区分两类合格答案：

```text
RAG answer:
  - 必须完全基于检索证据
  - 可以包含 Sources

knowledge_fallback answer:
  - 必须明确说明知识库没有可用信息
  - 可以基于通用知识
  - 不得包含 Sources
```

### Aggregation Prompt

需要允许混合答案：

- 知识库答案按原规则聚合。
- 降级答案保留“基于通用知识”的说明。
- 最终 Sources 只收集真实知识库来源。

## 文件影响范围建议

预计后续实现会影响以下文件：

```text
project/rag_agent/prompts.py
project/rag_agent/nodes/execution.py
project/rag_agent/nodes/evaluation.py
project/rag_agent/nodes/aggregation.py
project/rag_agent/edges.py
project/rag_agent/graph_state.py
project/retrieval/pipeline.py
project/config.py
tests/
```

说明：

- 如果复用现有 `RERANKER_SCORE_THRESHOLD`，`config.py` 可以不新增配置项。
- 如果需要默认阈值但不希望暴露配置开关，可以把阈值常量放在 retrieval 或 evaluation 相关模块中。
- 如果只通过 LLM 评估触发降级，`pipeline.py` 可以少改；如果要严格使用 rerank 阈值，`pipeline.py` 需要返回更明确的证据状态。

## 测试建议

### 单元测试

应覆盖：

1. rerank 分数低于阈值时，任务结果标记为 `knowledge_fallback`。

2. `rag_research` 返回 `contexts=[]` 且包含 gaps 时，先触发重试；重试耗尽后进入 `knowledge_fallback`。

3. `rag_research` 返回 `RAG_RESEARCH_ERROR` 时，重试耗尽后进入 `knowledge_fallback`。

4. `knowledge_fallback` 答案不包含 `Sources`。

5. `knowledge_fallback` 答案包含“知识库未找到可用信息”的说明。

6. 多任务中一个任务正常 RAG、另一个任务 fallback 时，两个任务的 `answer_mode` 独立正确。

7. 聚合阶段不会为 fallback 子答案生成 Sources。

### 集成测试

建议构造三类问题：

```text
知识库能回答的问题
知识库不能回答但通用知识能回答的问题
触发检索异常的问题
```

验证：

- 第一类仍走正常 RAG。
- 第二类先检索和重试，再降级。
- 第三类不暴露内部错误，最终给出通用知识回答。

### 回归测试

应确认以下行为不变：

- `chitchat` 闲聊不触发检索。
- `clarification` 仍进入澄清流程。
- `follow_up` 仍能基于会话摘要还原问题。
- 正常 RAG 答案仍输出真实 Sources。
- 无真实文件来源时不输出 Sources。

## 验收标准

实现完成后，应满足：

1. `rag_qa` 问题仍优先使用知识库回答。

2. 知识库证据不足时，系统先按现有重试机制尝试补充检索。

3. 重试后仍不可回答时，单任务进入 `knowledge_fallback`。

4. `knowledge_fallback` 回答明确说明知识库未找到可用信息。

5. `knowledge_fallback` 回答基于模型通用知识生成。

6. `knowledge_fallback` 回答不输出 `Sources`。

7. 检索失败不会直接暴露内部错误，重试后进入降级回答。

8. 多任务中每个任务独立降级，互不影响。

9. 聚合答案不会把通用知识回答伪装成知识库来源。

10. 原有 `chitchat`、`clarification`、正常 RAG 流程不被破坏。

## 建议实施阶段

### 阶段 1：补充状态和结果结构

- 为任务结果增加 `answer_mode`、`used_knowledge_base`、`fallback_reason` 等字段。
- 保证聚合阶段可以识别每个子任务答案来源。

### 阶段 2：定义证据不足判断

- 在 rerank 结果中保留最高分。
- 明确低分阈值判断。
- 将 `contexts=[]`、低分、检索错误统一映射成证据状态。

### 阶段 3：接入重试后降级

- 保留现有重试机制。
- 当答案评估仍不满意且重试预算耗尽时，进入 `knowledge_fallback`。
- 不把降级挂到 `chitchat_response`。

### 阶段 4：调整 prompts

- 新增或修改 fallback prompt。
- 明确降级回答允许使用通用知识。
- 明确不输出 Sources。

### 阶段 5：补充测试

- 覆盖低分、无上下文、检索失败、多任务混合和 Sources 规则。
- 跑现有 RAG、rerank、intent 相关测试，确认回归稳定。
