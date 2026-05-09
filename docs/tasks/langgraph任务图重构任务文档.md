# LangGraph 任务图重构任务文档

## 背景

当前 `project/rag_agent` 中的 LangGraph 主图和子图职责边界不清晰：

- 主图同时承担会话摘要、查询改写、澄清、任务派发和答案聚合。
- 子图既叫 agent/orchestrator，又负责工具调用、检索循环、rerank、上下文压缩、fallback、答案评价和重试。
- `State` 与 `AgentState` 通过 `agent_answers` 等字段隐式传递结果，字段语义偏松散。
- UI 流式展示依赖节点名，图结构调整时需要兼容。

本次重构目标是把系统拆成清晰的两层：

- 主图只负责意图识别和任务编排。
- 子图只负责执行单个任务。
- RAG 检索流程封装为确定性 tool，由子图中的 LLM 按需调用。

## 用户确认的设计约束

1. 意图类型支持：
   - `rag_qa`
   - `clarification`
   - `chitchat`
   - `follow_up`

2. 去掉 `unsupported` 意图。

3. 主图允许直接处理闲聊回复。

4. 主图允许做最终答案聚合。

5. 子图中的 RAG 流程作为 LLM tool 暴露。

6. RAG tool 内部使用确定性流水线。

7. 子图中的 LLM 可以根据 tool 返回结果判断是否重新检索，也可以保留部分结果、针对缺口重新检索。

8. 需要兼容当前 Gradio 流式展示。

## 目标架构

### 主图职责

主图只处理对话级控制流，不直接执行检索细节。

主图负责：

- 会话历史摘要。
- 当前用户输入的意图识别。
- 多轮追问还原。
- 澄清问题生成。
- 闲聊直接回复。
- RAG 任务规划。
- 并发派发任务子图。
- 聚合多个任务结果。

主图不负责：

- child chunk 搜索。
- parent chunk 获取。
- rerank。
- 检索上下文组装。
- RAG 重试细节。
- 单任务答案质量判断。

### 子图职责

子图只执行一个任务。

子图负责：

- 接收主图规划出的单个任务。
- 调用确定性 RAG tool 获取证据。
- 判断证据是否足够。
- 必要时基于缺口重新调用 RAG tool。
- 生成单任务答案。
- 返回结构化任务结果。

子图不负责：

- 判断用户整体意图。
- 拆分多个任务。
- 聚合多个任务结果。
- 处理闲聊或澄清。

## 目标图结构

### 主图

```text
START
  -> summarize_history
  -> recognize_intent
  -> route_after_intent
      -> clarification_response
      -> chitchat_response
      -> plan_rag_tasks
  -> dispatch_task_executors
  -> aggregate_answers
  -> END
```

说明：

- `follow_up` 不作为最终执行分支，而是在 `recognize_intent` 或 `plan_rag_tasks` 中结合 `conversation_summary` 还原为完整任务。
- `chitchat` 可以由主图直接回复，不进入子图。
- `clarification` 继续保留 interrupt 机制，兼容当前人机澄清流程。
- `rag_qa` 进入任务规划和子图执行。

### 子图

```text
START
  -> task_executor
  -> route_after_task_executor
      -> rag_research_tool
      -> collect_task_answer
  -> assess_or_continue
      -> task_executor
      -> collect_task_answer
  -> END
```

说明：

- 子图中的 LLM 节点可以调用 `rag_research` tool。
- `rag_research` tool 内部是确定性流水线，不由 LLM 决定具体检索步骤。
- LLM 只决定是否需要检索、是否需要针对缺口再次检索、最终如何基于证据回答。

## 意图识别设计

新增结构化输出模型，替代当前过载的 `QueryAnalysis`。

建议字段：

```text
IntentAnalysis
  intent_type: rag_qa | clarification | chitchat | follow_up
  is_clear: bool
  original_query: str
  normalized_query: str
  clarification_needed: str
  follow_up_context: str
  tasks: list[TaskSpec]
```

### 各意图处理规则

#### rag_qa

用户提出明确的信息查询，需要从文档中检索回答。

处理方式：

- 进入 `plan_rag_tasks`。
- 生成一个或多个 `TaskSpec`。
- 通过 `Send` 并发派发到任务子图。

#### clarification

用户输入不足以判断具体问题。

处理方式：

- 主图生成澄清问题。
- 保留当前 interrupt 行为。
- 用户补充后继续主图流程。

#### chitchat

用户进行闲聊、打招呼、简单交流，不需要 RAG 检索。

处理方式：

- 主图直接生成自然回复。
- 不进入任务子图。
- 不调用 RAG tool。

#### follow_up

用户问题依赖历史上下文，例如“那它的优缺点呢？”。

处理方式：

- 使用 `conversation_summary` 还原指代对象。
- 如果还原后是知识查询，转换为 `rag_qa` 任务。
- 如果仍不清楚，转换为 `clarification`。
- 如果是闲聊延续，转换为 `chitchat`。

## 任务规划设计

新增 `TaskSpec`，作为主图与子图之间的唯一任务契约。

建议字段：

```text
TaskSpec
  task_id: str
  task_type: rag_qa
  query: str
  original_query: str
  context: str
  constraints: dict
```

任务规划规则：

- 单一问题生成一个任务。
- 多个独立信息需求拆成多个任务。
- 最多任务数建议保持当前逻辑的上限，例如 3 个。
- 每个任务必须是自包含查询。
- 任务中可以携带必要的多轮上下文，但不能引入用户没有表达过的新事实。

## RAG Tool 设计

新增高层 tool：`rag_research`。

该 tool 替代子图中直接暴露 `search_child_chunks` 和 `retrieve_parent_chunks` 的默认路径。

### Tool 输入

建议输入：

```text
rag_research
  query: str
  focus: str | None
  keep_parent_ids: list[str]
  exclude_parent_ids: list[str]
  retry_reason: str | None
```

字段说明：

- `query`：当前检索问题。
- `focus`：本次检索重点，用于补齐某个缺口。
- `keep_parent_ids`：此前结果中仍然保留的 parent ids。
- `exclude_parent_ids`：本轮需要排除的 parent ids，避免重复检索。
- `retry_reason`：重新检索原因，便于 diagnostics 和调试。

### Tool 内部确定性流水线

```text
normalize query
  -> search child chunks
  -> rerank child chunks
  -> select top parent ids
  -> retrieve parent chunks
  -> assemble evidence
  -> derive gaps
  -> return structured result
```

说明：

- LLM 不直接决定 child search 和 parent retrieve 的调用顺序。
- Rerank 继续复用当前 reranker 配置。
- 返回结果应该是结构化对象，而不是主要依赖字符串解析。
- 可以继续保留旧工具作为内部实现或调试入口，但子图默认只暴露 `rag_research`。

### Tool 输出

建议输出：

```text
RagResearchResult
  query: str
  focus: str
  contexts: list[RetrievedContext]
  sources: list[str]
  parent_ids: list[str]
  gaps: list[str]
  diagnostics: dict
```

其中 `RetrievedContext` 建议包含：

```text
RetrievedContext
  parent_id: str
  source: str
  content: str
  score: float | None
```

## 子图答案质量控制

答案质量控制保留在子图内部，作为执行任务的一部分。

建议流程：

1. 子图 LLM 调用 `rag_research`。
2. LLM 检查结果是否足够回答当前任务。
3. 如果不足：
   - 明确缺口。
   - 决定是否保留已有 parent ids。
   - 决定是否排除无用 parent ids。
   - 使用更聚焦的 `focus` 再次调用 `rag_research`。
4. 达到检索上限或证据足够后生成最终任务答案。

这部分不放到主图，避免主图理解 RAG 执行细节。

## State 重构设计

### 主图状态

建议将当前 `State` 重构为更明确的 `MainState`。

```text
MainState
  messages
  conversation_summary
  original_query
  intent_type
  normalized_query
  clarification_needed
  task_plan
  task_results
```

其中：

- `task_plan` 只由主图写入。
- `task_results` 只接收子图返回。
- `messages` 保留，用于 LangGraph checkpoint 和 UI streaming 兼容。

### 子图状态

建议将当前 `AgentState` 重构为 `TaskState`。

```text
TaskState
  messages
  task_id
  task_type
  query
  original_query
  task_context
  research_results
  kept_parent_ids
  excluded_parent_ids
  final_answer
  diagnostics
```

其中：

- `research_results` 记录每次 `rag_research` tool 的结构化结果。
- `diagnostics` 汇总 search、rerank、parent retrieve、retry 次数等调试信息。

## 文件级改造计划

### `project/rag_agent/schemas.py`

新增或调整：

- `IntentAnalysis`
- `TaskSpec`
- `TaskResult`
- `RagResearchResult`
- `RetrievedContext`

保留：

- `AnswerEvaluation` 可以保留，但语义改为子图内部质量判断。

### `project/rag_agent/graph_state.py`

调整：

- `State` 字段语义清理，或重命名为 `MainState` 后通过导入兼容旧名称。
- `AgentState` 字段语义清理，或重命名为 `TaskState` 后通过导入兼容旧名称。
- 避免 `agent_answers` 这种宽泛命名，改为 `task_results`。

兼容策略：

- 如果外部代码依赖 `State` / `AgentState` 名称，第一阶段可以保留别名。

### `project/rag_agent/nodes.py`

主图节点建议：

- `summarize_history`
- `recognize_intent`
- `clarification_response`
- `chitchat_response`
- `plan_rag_tasks`
- `aggregate_answers`

子图节点建议：

- `task_executor`
- `collect_task_answer`

可以考虑拆文件：

- `main_nodes.py`
- `task_nodes.py`
- `retrieval_pipeline.py`

第一阶段也可以先不拆文件，只先改清楚节点名和职责边界。

### `project/rag_agent/edges.py`

新增或调整：

- `route_after_intent`
- `route_after_task_executor`
- `route_after_task_assessment`

移除或弱化：

- `route_after_rewrite`
- `route_after_orchestrator_call`

### `project/rag_agent/tools.py`

新增：

- `rag_research`

内部复用：

- `_search_child_chunks`
- `_retrieve_parent_chunks`
- reranker 相关逻辑

默认暴露给子图 LLM 的工具：

- `rag_research`

旧工具处理策略：

- 第一阶段保留旧工具，避免破坏测试和调试。
- 第二阶段根据使用情况决定是否仅作为内部函数保留。

### `project/rag_agent/graph.py`

调整：

- `create_agent_subgraph` 改为 `create_task_executor_subgraph`。
- `create_agent_graph` 内部创建主图和任务子图。
- 子图节点名从 `orchestrator` 改为 `task_executor`。

兼容策略：

- 外部仍可保留 `create_agent_graph(llm, tools_list)` 函数签名，减少 UI 和初始化代码改动。

### `project/core/chat_interface.py`

更新 UI streaming 兼容配置：

- 新增系统节点名：
  - `recognize_intent`
  - `plan_rag_tasks`
  - `clarification_response`
  - `chitchat_response`

保留或映射旧标题：

- `recognize_intent` 可以显示为 `Query Analysis & Intent Recognition`。
- `plan_rag_tasks` 可以显示为 `Task Planning`。

注意：

- tool 展示逻辑仍兼容，因为 `rag_research` 也是标准 LangChain tool call。
- 如果保留旧节点名别名，UI 改动可以更小。

## 分阶段实施计划

### 阶段 1：契约和命名重构

目标：

- 明确主图/子图职责。
- 新增 schema。
- 新增 state 字段。
- 保留现有行为尽量不变。

任务：

- 新增 `IntentAnalysis`、`TaskSpec`、`TaskResult`。
- 将 `rewrite_query` 语义改为 `recognize_intent`。
- 将子图 `orchestrator` 语义改为 `task_executor`。
- 保留旧函数名兼容外部调用。

验收：

- 原有 RAG 问答仍能跑通。
- 澄清流程仍能 interrupt。
- UI 能显示新节点或兼容旧节点。

### 阶段 2：引入确定性 RAG tool

目标：

- 把搜索、rerank、parent retrieve 封装成一个 `rag_research` tool。
- 子图 LLM 默认只调用该高层 tool。

任务：

- 在 `tools.py` 中实现 `rag_research`。
- 将当前 rerank 逻辑迁入 tool 内部或公共 retrieval pipeline。
- 子图工具列表默认替换为 `rag_research`。
- 保留旧工具作为内部实现或 debug 工具。

验收：

- 子图不再直接暴露 `search_child_chunks` / `retrieve_parent_chunks` 给 LLM。
- tool 返回结构化检索结果。
- RAG 问答质量不低于当前基线。

### 阶段 3：子图重试和结果保留策略

目标：

- 子图 LLM 能基于 `rag_research` 返回的 gaps 和 evidence 决定是否重试。
- 支持保留部分 parent ids，并针对缺口继续检索。

任务：

- 设计子图 prompt。
- 限制最大 tool 调用次数和最大循环次数。
- 记录 `kept_parent_ids`、`excluded_parent_ids`、retry diagnostics。

验收：

- 子图能在证据不足时重新检索。
- 子图不会无限重复同一 query 或 parent id。
- diagnostics 能反映检索和重试过程。

### 阶段 4：主图意图和多轮追问完善

目标：

- 支持 `rag_qa`、`clarification`、`chitchat`、`follow_up`。
- 多轮追问能还原上下文。

任务：

- 更新 intent prompt。
- 增加 follow-up rewrite 逻辑。
- 增加 chitchat 直接回复节点。
- 更新 UI 展示。

验收：

- 明确 RAG 问题进入任务子图。
- 模糊问题触发澄清。
- 闲聊不触发检索。
- 多轮追问能结合历史摘要变成自包含任务。

### 阶段 5：测试和回归

目标：

- 确认重构没有破坏现有 RAG、rerank 和 evaluation 流程。

任务：

- 增加 intent 路由测试。
- 增加 task planning 测试。
- 增加 `rag_research` tool 单元测试。
- 增加子图 retry 行为测试。
- 跑现有 rerank / retrieval 测试。

验收：

- 现有测试通过。
- 新增测试覆盖核心路由和 tool contract。
- Gradio 基本流式展示可用。

## 兼容性要求

### 外部 API

尽量保留：

- `create_agent_graph(llm, tools_list)`
- 当前 RAG system 初始化方式
- 当前 Gradio chat 调用方式

内部可以改名，但对外保持兼容。

### UI streaming

需要兼容：

- 系统节点消息展示。
- tool call 展示。
- tool result preview。
- 最终 assistant token streaming。

节点名变化时需要同步更新：

- `SILENT_NODES`
- `SYSTEM_NODES`
- `SYSTEM_NODE_CONFIG`

### 配置

继续复用：

- `MAX_TOOL_CALLS`
- `MAX_ITERATIONS`
- reranker 相关配置
- retrieval fusion 相关配置

必要时新增：

- `MAX_RAG_RESEARCH_RETRIES`
- `MAX_TASKS_PER_QUERY`

## 验收标准

### 架构验收

- 主图代码中不直接出现 child search、parent retrieve、rerank 的执行细节。
- 子图不做用户整体意图识别。
- RAG 执行逻辑集中在 `rag_research` tool 或 retrieval pipeline。
- 主图和子图通过 `TaskSpec` / `TaskResult` 交换数据。

### 行为验收

- 普通 RAG 问题能正常回答。
- 多问题输入能拆成多个任务并聚合。
- 模糊输入能触发澄清。
- 闲聊能直接回复且不触发 RAG tool。
- 多轮追问能结合历史上下文。
- 检索结果不足时，子图可以重新检索。

### UI 验收

- Gradio 中仍能看到系统分析节点。
- tool call 能展示为折叠消息。
- `rag_research` tool result 能正常 preview。
- 最终答案能正常流式输出或完整输出。

### 测试验收

- 现有 rerank 和 retrieval 测试通过。
- 新增 intent routing 测试通过。
- 新增 RAG tool contract 测试通过。
- 新增子图 retry 测试通过。

## 风险和处理方式

### 风险 1：结构化 tool result 与 LangChain tool 输出兼容问题

处理：

- 初期可以让 tool 返回 JSON 字符串。
- 子图 LLM 按 JSON 内容判断。
- 后续再优化为更强的结构化 artifact。

### 风险 2：节点改名影响 UI

处理：

- 第一阶段保留旧节点名或在 UI 中增加映射。
- UI 修改与图修改同阶段完成。

### 风险 3：确定性 RAG tool 降低灵活性

处理：

- tool 输入保留 `focus`、`keep_parent_ids`、`exclude_parent_ids`。
- 灵活性放在“是否重试、针对什么重试”，不放在底层检索步骤。

### 风险 4：多轮追问误还原

处理：

- intent prompt 要明确不能引入历史中不存在的信息。
- 如果指代对象不明确，转为 `clarification`。

## 非目标

本次重构不做：

- 更换 LLM provider。
- 更换 vector database。
- 重写 reranker 模型。
- 改动文档切分策略。
- 改动 evaluation 数据集格式。
- 大规模 UI redesign。

## 建议优先级

优先级从高到低：

1. 明确主图/子图 state 和 schema 契约。
2. 新增 `rag_research` 确定性 tool。
3. 重构子图为 task executor。
4. 重构主图 intent 和 task planning。
5. 更新 UI streaming 兼容。
6. 补测试和回归 evaluation。

