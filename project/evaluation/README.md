# RAG Evaluation

当前评测主路径保持简单：

```text
RAGBench question + documents
-> 待测 LLM 生成 answer
-> RAGAS 对 question / documents / answer / reference 评分
```

它不走当前项目的向量库和 retriever，也不使用 RAGBench 自带回答做最终分数。RAGBench 只提供问题、上下文文档和参考答案。

## Environment

需要 `.venv` 中已安装：

```bash
ragas
datasets
openai
```

默认使用 DeepSeek OpenAI-compatible API：

```bash
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-chat"
```

代码默认 base URL 为 `https://api.deepseek.com`。如果仍想使用其他 OpenAI-compatible 服务，可以设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANSWER_LLM_MODEL` 和 `RAGAS_LLM_MODEL` 覆盖默认值。

默认只跑稳定的 RAGAS 指标：

- `faithfulness`
- `context_precision`
- `context_recall`

`answer_relevancy` 在 RAGAS 0.4.x 中存在 embedding API 兼容问题，默认关闭。

## Run

单数据集：

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

多数据集：

```bash
for subset in covidqa hotpotqa msmarco; do
  .venv/bin/python project/evaluation/runners/ragbench_eval_runner.py \
    --subset "$subset" \
    --split test \
    --limit 50 \
    --output-dir "runtime/evaluation_reports/ragbench_${subset}_test_50" \
    --ragas-max-workers 1 \
    --ragas-batch-size 1 \
    --ragas-timeout 120 \
    --ragas-max-retries 1
done
```

`--ragas` 和 `--generate` 仍可传入，但只是兼容旧命令；新 runner 总是生成回答并运行 RAGAS。

## Outputs

每次运行输出：

```text
rag_outputs.jsonl
ragas_results.jsonl
ragas_error_cases.jsonl
ragas_metrics_summary.csv
ragbench_ragas_report.md
```

其中 `rag_outputs.jsonl` 保存每条样本的：

- `question`
- `retrieved_contexts`
- `answer`
- `reference`

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

如果你只做某一领域特化，可以只跑相关 subset；如果要看泛化能力，再跑多个 subset。
