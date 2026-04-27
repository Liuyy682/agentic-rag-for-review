# RAG Evaluation

该目录提供独立于主 RAG 流程的评测系统，用同一套数据集、指标和报告比较不同检索、分块、清洗、rerank 和 prompt 实验。

## Dataset

默认数据集路径：

```bash
project/evaluation/datasets/eval_questions.jsonl
```

每行是一条 JSON：

```json
{
  "question_id": "q_0001",
  "question": "文档中 X 方法的主要限制是什么？",
  "reference_answer": "标准答案",
  "source_file": "paper_a.pdf",
  "gold_parent_ids": ["paper_a_parent_3"],
  "gold_child_ids": ["paper_a_parent_3_child_2"],
  "gold_evidence_text": ["关键证据片段"],
  "question_type": "single-hop",
  "difficulty": "medium",
  "tags": ["definition"]
}
```

必填字段是 `question_id`、`question`、`reference_answer`、`source_file`。第一版可以只标注 `gold_parent_ids`，后续再补充稳定的 `gold_child_ids`。

## Retrieval Evaluation

运行当前 baseline 检索评测：

```bash
python project/evaluation/runners/retrieval_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --score-threshold 0.7 \
  --run-label baseline
```

输出：

```text
project/evaluation/reports/
├── retrieval_results.jsonl
├── retrieval_error_cases.jsonl
├── retrieval_metrics_summary.csv
├── retrieval_report.md
└── eval_runs/<run_id>/
```

已实现指标：

- `recall@k`
- `hitrate@k`
- `precision@k`
- `mrr`
- `ndcg@k`
- `child_recall@k`
- `parent_recall@k`
- `source_hitrate@k`

## RAGAS / Generation Evaluation

先保存完整 RAG 输出但不跑 RAGAS：

```bash
python project/evaluation/runners/ragas_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10 \
  --score-threshold 0.7 \
  --skip-ragas
```

安装可选依赖后运行 RAGAS：

```bash
pip install ragas datasets
python project/evaluation/runners/ragas_eval_runner.py \
  --dataset project/evaluation/datasets/eval_questions.jsonl \
  --top-k 10
```

RAGAS runner 会生成 `rag_outputs.jsonl`，并在可选依赖存在时生成 `ragas_results.jsonl`、`ragas_metrics_summary.csv` 和 `ragas_report.md`。

## Compare Runs

```bash
python project/evaluation/runners/compare_runs.py \
  --baseline project/evaluation/reports/eval_runs/<baseline_run>/retrieval_metrics_summary.csv \
  --current project/evaluation/reports/eval_runs/<current_run>/retrieval_metrics_summary.csv \
  --output project/evaluation/reports/compare_report.md
```

## RAGBench

RAGBench 数据集在 Hugging Face 上以 Parquet 发布。本项目通过 Hugging Face Dataset Server 的 HTTP rows API 拉取数据，因此不强制安装 `datasets/pandas/pyarrow`。

只导入 RAGBench 样本并转换为本项目 JSONL：

```bash
python project/evaluation/runners/ragbench_importer.py \
  --subset covidqa \
  --split test \
  --limit 20
```

输出：

```text
project/evaluation/datasets/ragbench_eval_questions.jsonl
project/evaluation/datasets/ragbench_contexts.jsonl
```

自动生成 RAGBench 官方标注汇总报告：

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 20
```

用当前 RAGAS 版本重新评测 RAGBench 自带回答：

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 20 \
  --ragas
```

如果本地 Ollama 模型可用，也可以用 RAGBench 自带 documents 作为上下文生成答案，并输出 token F1 和失败样本：

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 20 \
  --generate
```

对本地模型生成答案重新跑 RAGAS：

```bash
python project/evaluation/runners/ragbench_eval_runner.py \
  --subset covidqa \
  --split test \
  --limit 20 \
  --generate \
  --ragas
```

`--ragas` 需要安装可选依赖 `ragas` 和 `datasets`，并配置 RAGAS judge 所需的 LLM/API 环境。RAGAS 输入字段使用官方 single-turn schema：`user_input`、`retrieved_contexts`、`response`、`reference`。

注意：RAGBench 每条样本自带候选 documents，并不是当前项目已入库的 PDF 语料。默认 RAGBench runner 不会污染现有 Qdrant collection；如果要测试 retriever，需要先将 `ragbench_contexts.jsonl` 建成单独评测索引。

## Notes

- 评测模块只调用主系统已有接口，不修改主 RAG pipeline。
- 每次运行都会保存 `run_metadata.json`，包含 git commit、数据集版本、模型、分块和 top-k 配置快照。
- `eval_questions.jsonl` 当前为空，请用真实文档标注数据填充；`eval_questions.sample.jsonl` 只作为格式示例。
