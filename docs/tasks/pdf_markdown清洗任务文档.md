# PDF 转 Markdown 后处理清洗任务文档

## 背景

当前项目中 PDF 会先通过 `pymupdf4llm` 转换为 Markdown，再进入 RAG 文档处理、分块、向量化和检索流程。

在 PPT 导出的 PDF 场景下，转换后的 Markdown 往往会混入页眉、页脚、页码、版权声明、保密标识等噪音信息。这些内容如果直接进入 RAG，会带来几个问题：

- 重复页眉页脚被大量入库，污染向量索引。
- 页码、版权、公司名等低价值文本影响 chunk 语义纯度。
- 检索结果可能召回无意义的重复块。
- LLM 上下文被噪音占用，降低回答质量。
- 引用溯源时页面信息不稳定，难以定位原始内容。

本任务目标是在 PDF 转 Markdown 之后、RAG 入库之前，新增一个保守型 Markdown 清洗阶段，尽量去除高置信噪音，同时最大限度避免误删正文。

## 已确认约束

1. PDF 转 Markdown 工具使用 `pymupdf4llm`。

2. Markdown 中存在分页标记，格式类似：

```text
--- end of page.page_number=1 ---
```

3. 文档语言为中英混合。

4. 输入 PDF 主要来自 PPT 导出。

5. 清洗策略优先级：

```text
尽量不误删正文 > 尽量删干净噪音
```

6. 清洗后的 Markdown 用于 RAG 入库。

7. 转换后的 Markdown 已经包含标题层级，例如 `#`、`##`。

8. RAG metadata 建议保留：

```text
source_file
page_number
slide_title
chunk_index
```

## 总体目标

新增 Markdown 清洗能力，输入为 `pymupdf4llm` 生成的 Markdown，输出为更适合 RAG 的 Markdown 和结构化清洗日志。

核心目标：

- 按分页标记切分页面。
- 删除高置信页码。
- 删除高置信重复页眉、页脚、版权和保密信息。
- 保留 PPT 页面标题、项目符号、表格和正文短句。
- 保留页面级 metadata，支持后续 chunk 溯源。
- 输出清洗日志，便于排查误删和规则效果。
- 在 RAG 入库前支持 chunk 级低价值过滤和去重。

## 非目标

本任务第一阶段不处理以下内容：

- 不重做 PDF 到 Markdown 的转换工具选型。
- 不引入 OCR 或视觉模型识别。
- 不对 PPT 页面进行复杂版面还原。
- 不强制重排 Markdown 的标题层级。
- 不做激进的短文本删除。
- 不直接修改检索、rerank、Agent 编排逻辑。

## 推荐处理流程

整体流程：

```text
PDF
  -> pymupdf4llm
  -> raw Markdown
  -> 按 page marker 切页
  -> 页码清洗
  -> 重复页眉页脚识别
  -> 高置信噪音删除
  -> 页面 metadata 提取
  -> cleaned Markdown
  -> RAG chunking
  -> chunk 级去重和低价值过滤
```

## 页面切分设计

清洗模块应先基于分页标记切分页面，而不是直接对整篇 Markdown 做全局正则替换。

每个页面建议抽象为一个 page block：

```text
PageBlock
  source_file: str
  page_number: int
  raw_text: str
  raw_lines: list[str]
  cleaned_text: str
  cleaned_lines: list[str]
  slide_title: str | None
  removed_items: list[CleanEvent]
  candidates: list[CleanCandidate]
```

分页标记本身不应进入最终 RAG 正文，但页码应进入 metadata。

## 页码清洗规则

页码属于高置信可删除对象，但必须结合当前页码判断。

可删除形式包括：

```text
1
- 1 -
Page 1
page 1
第 1 页
1 / 20
1 of 20
```

建议删除条件：

- 独占一行。
- 行文本较短。
- 数字与当前 `page_number` 匹配。
- 位于页面开头或结尾附近，优先判断后 1-3 行。

禁止仅因一行包含数字就删除。以下内容应保留：

```text
方案 1
步骤 1：准备数据
Module 1
Q1 Revenue
2024 年模型效果对比
```

## 页眉页脚识别规则

PPT 导出的 PDF 中，页面顶部第一行很可能是 slide title，因此不能简单删除每页前几行。

页眉页脚识别应使用跨页重复统计：

1. 提取每页前 1-3 行和后 1-3 行作为候选区域。
2. 对候选行做文本规范化。
3. 统计规范化文本跨页出现频率。
4. 满足高重复、高位置一致性、低正文价值时才删除。

规范化建议：

- 去除首尾空格。
- 合并连续空格。
- 英文大小写归一。
- 去掉明显页码数字。
- 去掉 Markdown 粗体等轻量格式符号。

建议删除条件：

- 至少出现在 3 页以上。
- 覆盖比例超过总页数的 30%-50%。
- 主要出现在页面前 3 行或后 3 行。
- 文本较短，且不像正文句子。
- 不属于页面唯一标题。

常见可删除文本：

```text
Company Confidential
Internal Use Only
Confidential
All rights reserved
版权所有
仅供内部使用
© 2024 xxx
```

## 正文保护规则

为降低误删风险，应对以下内容做保护：

- Markdown 标题，例如 `#`、`##`、`###`。
- 页面顶部不重复或低重复的 slide title。
- 项目符号行，例如 `-`、`*`、`+` 开头的行。
- 编号列表，例如 `1.`、`2.` 开头的行。
- 表格行，例如包含多个 `|` 的行。
- 代码块内容。
- 含冒号说明的正文短句。
- 中英混合技术名词短句。

特别注意：PPT 页面经常由短标题和短 bullet 组成，不能使用“短行删除”作为主规则。

## Slide Title 提取

由于 Markdown 已经包含标题层级，优先使用页面内第一个 Markdown 标题作为 `slide_title`。

建议规则：

1. 每页内寻找第一个 `#`、`##` 或 `###` 标题。
2. 若找不到 Markdown 标题，可使用页面前几行中第一个非噪音、非页码、非空行作为候选标题。
3. 如果候选标题被重复页眉规则命中，则不作为 `slide_title`。

`slide_title` 应进入 chunk metadata，但不应重复拼接到每个 chunk 正文中，除非现有 chunker 需要标题上下文增强。

## 清洗日志设计

清洗过程必须可审计。每条删除或保留决策建议记录为结构化事件。

删除事件示例：

```text
CleanEvent
  source_file: "example.pdf"
  page_number: 12
  text: "Company Confidential"
  action: "removed"
  reason: "repeated_footer"
  confidence: 0.96
```

低置信候选但保留的事件示例：

```text
CleanCandidate
  source_file: "example.pdf"
  page_number: 8
  text: "Agentic RAG"
  action: "kept"
  reason: "possible_repeated_top_line_but_title_like"
  confidence: 0.62
```

日志用途：

- 排查误删。
- 统计规则命中率。
- 对比清洗前后 RAG 召回质量。
- 后续调整阈值。

## RAG Metadata 设计

清洗后进入 RAG 的每个 chunk 至少应保留：

```text
source_file: 原始文件名
page_number: 来源页码
slide_title: 页面标题
chunk_index: 当前文档内 chunk 序号
```

如果一个 chunk 跨页，应优先避免跨页分块；如果无法避免，metadata 应支持页码列表：

```text
page_numbers: [3, 4]
```

第一版建议按页处理后再分块，降低跨页 chunk 的复杂度。

## Chunk 级后处理

Markdown 清洗后，RAG 入库前建议再做一层 chunk 级过滤。

建议过滤对象：

- 完全重复 chunk。
- 只包含页码、版权、保密标识的 chunk。
- 字符数很短且不包含标题、项目符号或有效技术词的 chunk。
- 在大量页面中重复出现的低信息 chunk。

该阶段仍应保持保守。无法高置信判定为噪音的 chunk 可以保留，但可在 metadata 中标记低价值，用于后续降权或调试。

## 验收标准

### 功能验收

- 能正确按 `--- end of page.page_number=N ---` 切分页面。
- 能删除与当前页码匹配的独占页码行。
- 能识别并删除跨页重复的高置信页眉页脚。
- 能保留 Markdown 标题、slide title、bullet、编号列表和表格。
- 能输出清洗后的 Markdown。
- 能输出结构化清洗日志。
- 能为后续 RAG chunk 提供 `source_file`、`page_number`、`slide_title`、`chunk_index` metadata。

### 质量验收

- 对 PPT 导出的中英混合 Markdown，正文误删率应优先控制。
- 重复页眉页脚在向量库中的残留显著减少。
- 清洗前后抽样对比时，页面标题和关键 bullet 不应丢失。
- 清洗日志能解释每一次删除。
- 低置信内容应保留或标记，不能静默删除。

### 回归验收

- 不影响现有 PDF 转 Markdown notebook 的基本流程。
- 不破坏现有 document chunker 的输入格式。
- 不改变现有 RAG 检索接口。
- 如新增测试，测试应覆盖页码、重复 footer、slide title 保护、bullet 保护和中英混合文本。

## 建议测试样例

建议准备一个小型 fixture Markdown，至少包含：

```text
# Agentic RAG Overview

- Multi-step reasoning
- Tool calling
- State management

Company Confidential
--- end of page.page_number=1 ---

# Retrieval Pipeline

1. Query rewrite
2. Hybrid retrieval
3. Rerank

Company Confidential
--- end of page.page_number=2 ---

# Evaluation

第 1 步：构造评测集
第 2 步：运行检索评测

3 / 3
Company Confidential
--- end of page.page_number=3 ---
```

期望：

- 删除 `Company Confidential`。
- 删除 `3 / 3`。
- 保留三个标题。
- 保留 bullet 和编号列表。
- 保留 `第 1 步`、`第 2 步`。

## 实施建议

第一阶段建议只实现确定性规则，不引入 LLM 判断。

推荐模块边界：

```text
markdown_cleaner
  parse_pages
  detect_page_numbers
  detect_repeated_headers_footers
  protect_content_lines
  clean_pages
  emit_cleaning_log
```

推荐配置项：

```text
PAGE_MARKER_PATTERN
HEADER_FOOTER_SCAN_LINES = 3
MIN_REPEAT_PAGES = 3
MIN_REPEAT_RATIO = 0.3
PAGE_NUMBER_CONFIDENCE_THRESHOLD = 0.9
HEADER_FOOTER_CONFIDENCE_THRESHOLD = 0.85
```

后续可根据真实文档清洗日志调整阈值。

## 开放问题

1. 清洗模块最终放在 `project/document_chunker.py` 附近，还是新增独立预处理模块。
2. 清洗日志输出为 JSONL、CSV，还是跟随项目现有报告目录格式。
3. 是否需要在 UI 中展示清洗摘要。
4. 是否需要保留原始 Markdown 与清洗后 Markdown 的差异对比文件。
5. 是否需要为每个 chunk 附带清洗版本号，便于后续重建索引。
