# RAG Evaluation

评测系统的目标是让分数可解释、可复查、不可误读。每次运行会保留原有输出文件，并额外写出：

```text
evaluation_warnings.jsonl
validity_summary.json
```

报告中的 `evaluation_valid=false` 或 warning 不代表脚本失败，而是说明该次结果不能直接用于结论，必须先看 warning 原因。

## Evaluation Modes

### 1. Oracle-context generation eval

入口：

```bash
.venv/bin/python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 50 \
  --output-dir runtime/evaluation_reports/ragbench_covidqa_test_50 \
  --ragas-max-workers 1 \
  --ragas-batch-size 1 \
  --ragas-timeout 120 \
  --ragas-max-retries 1
```

评测链路：

```text
RAGBench question + RAGBench documents
-> 待测 LLM 生成 answer
-> RAGAS 对 question / documents / answer / reference 评分
```

它证明的是：在 RAGBench 已给定上下文文档时，答案生成和 RAGAS 质量如何。

它不能证明：当前项目的向量库、retriever、reranker、生产分块策略是否有效。报告 metadata 会标记：

```json
{
  "evaluation_type": "oracle_context_generation_eval",
  "uses_project_retriever": false,
  "uses_oracle_contexts": true
}
```

### 2. Local retriever / local RAG eval

入口：

```bash
.venv/bin/python project/evaluation/runners/retrieval_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --output-dir runtime/evaluation_reports/local_retrieval
```

或端到端生成：

```bash
.venv/bin/python project/evaluation/runners/ragas_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --output-dir runtime/evaluation_reports/local_ragas
```

它证明的是：当前项目的数据库检索、RRF/dense/sparse、reranker 和生成链路在本地 gold 数据集上的表现。

注意：

- 当前仓库里的 `project/evaluation/datasets/eval_questions.jsonl` 默认为空，不能直接用于效果结论；`eval_questions.sample.jsonl` 只是中文占位样例。
- `eval_questions.jsonl` 必须包含真实 `gold_parent_ids` 或 `gold_child_ids`，否则该样本不会参与主检索指标均值。
- 如果 reranker 最终返回数小于配置的 `top_k`，报告会写出 `actual_results@k` 和 `insufficient_results_for_k` warning。
- `score_threshold` 在 `rrf`、`dense`、`sparse` 模式下目前不生效；报告会写 `score_threshold_ignored` warning。

### 3. RAGBench local synthetic retrieval eval

入口：

```bash
.venv/bin/python project/evaluation/runners/ragbench_local_rag_runner.py \
  --subset covidqa \
  --split test \
  --limit 50 \
  --top-k 5 \
  --output-dir runtime/evaluation_reports/ragbench_local
```

它把每条 RAGBench document 作为一个合成 chunk 写入本地库，用当前检索策略召回。

它证明的是：检索策略在 RAGBench document-level gold 上的表现。

它不能证明：生产摄取链路的真实 child chunk/parent chunk 质量。metadata 会标记：

```json
{
  "evaluation_type": "synthetic_ragbench_document_retrieval",
  "uses_project_retriever": true,
  "uses_synthetic_document_chunks": true
}
```

### 4. Chunking ablation

入口：

```bash
.venv/bin/python project/evaluation/runners/chunking_ablation.py \
  --source-contexts runtime/evaluation_reports/.../ragbench_covidqa_test_200_source_contexts.jsonl \
  --output-dir runtime/evaluation_reports/chunking_ablation
```

它证明的是：在固定检索策略和固定数据下，不同 chunking/context policy 的相对差异。

它不能证明：线上整体 RAG 质量，也不能把单一 subset 的收益泛化到所有任务。

#### 中文 Hugging Face 检索基准

如果本地 gold 集还没准备好，可以先用 Hugging Face 的中文检索基准 [`C-MTEB/T2Retrieval`](https://huggingface.co/datasets/C-MTEB/T2Retrieval) 和 [`C-MTEB/T2Retrieval-qrels`](https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels) 做分块消融：

```bash
.venv/bin/python project/evaluation/runners/chunking_ablation.py \
  --dataset t2_retrieval \
  --limit 100 \
  --offset 0 \
  --t2-distractor-docs 120 \
  --variants single_300_60,single_500_100,single_800_160,pc_300_60_800_160,pc_500_100_2000_400,pc_800_160_2000_400,pc_adaptive \
  --output-dir runtime/evaluation_reports/chunking_ablation_t2
```

这条链路使用 T2Retrieval 的 qrels 评估检索与分块策略，报告会标记 `dataset_language=zh` 和 `evaluation_type=t2_retrieval_chunking_ablation`。它不评估答案生成质量，也不替代你自己文档上的本地 gold 集。

建议优先看：

- `balanced_score`
- `mrr`
- `hitrate@5`
- `doc_recall@5`
- `sentence_recall@5`
- `context_chars@5`

如果两个方案 `balanced_score` 差距小于 `0.02`，优先选 `context_chars@5` 和 `index_chunks` 更小的方案。

## Validity And Warnings

新增输出：

```text
evaluation_warnings.jsonl
validity_summary.json
```

常见 warning：

| Code | Meaning |
|---|---|
| `empty_dataset` | 没有评测样本，分数不可用于结论。 |
| `missing_primary_gold` | 样本缺少 `gold_child_ids` 和 `gold_parent_ids`，不参与主检索指标均值。 |
| `gold_result_id_no_overlap` | 所有返回结果都无法和 primary gold id 对齐，需检查 gold id 命名或检索质量。 |
| `insufficient_results_for_k` | 实际返回条数小于声明的 `@k`。 |
| `score_threshold_ignored` | 当前检索模式没有应用记录的 score threshold。 |
| `oracle_context_generation_eval` | RAGBench documents 直接作为上下文，不评估项目 retriever。 |
| `synthetic_ragbench_document_retrieval` | RAGBench document 被当成合成 chunk，不代表生产分块。 |
| `ragas_metric_missing` | RAGAS 某条样本缺少必需指标。 |
| `reuse_existing_question_mismatch` | `--reuse-existing` 的旧输出和当前样本不匹配，不能复用。 |

`validity_summary.json` 示例：

```json
{
  "evaluation_type": "local_retriever_eval",
  "rows": 50,
  "warning_count": 2,
  "error_count": 0,
  "evaluation_valid": true,
  "warning_codes": {
    "insufficient_results_for_k": 2
  }
}
```

## Metrics

检索指标：

- `mrr`
- `recall@k`
- `hitrate@k`
- `precision@k`
- `ndcg@k`
- `child_recall@k`
- `parent_recall@k`
- `source_hitrate@k`
- `actual_results@k`

聚合字段：

- `rows`
- `scored_rows`
- `unscored_rows`
- `warning_count`
- `evaluation_valid`

RAGAS 默认指标：

- `faithfulness`
- `context_precision`
- `context_recall`

RAGAS 汇总会记录每个指标的有效样本数：

- `faithfulness_rows`
- `context_precision_rows`
- `context_recall_rows`
- `<metric>_missing_rows`

`answer_relevancy` 在 RAGAS 0.4.x 中存在 embedding API 兼容问题，默认关闭。只有设置 `RAGAS_ENABLE_ANSWER_RELEVANCY=true` 时才会启用。

## Environment

需要 `.venv` 中已安装：

```bash
pyarrow
ragas
datasets
openai
```

默认使用 DeepSeek OpenAI-compatible API：

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-chat"
```

代码默认 base URL 为 `https://api.deepseek.com`。如果使用其他 OpenAI-compatible 服务，可以设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANSWER_LLM_MODEL` 和 `RAGAS_LLM_MODEL` 覆盖默认值。

## RAGBench Subsets

常用 subset：

```text
covidqa
hotpotqa
msmarco
pubmedqa
techqa
finqa
```

如果只做某一领域特化，可以只跑相关 subset；如果要看泛化能力，再跑多个 subset。报告结论必须写明 subset、split、limit、offset、evaluation_type 和 warning 摘要。
