# RAG Evaluation Quick Start

Quick reference guide for running RAG evaluations.

## Setup (One-Time)

```bash
# 1. Set environment variables
export LANGSMITH_API_KEY=your_langsmith_key
export WANDB_API_KEY=your_wandb_key
export OPENAI_API_KEY=your_openai_key  # or ANTHROPIC_API_KEY
export LLM_PROVIDER=openai  # or anthropic, lmstudio

# 2. Create evaluation dataset
python scripts/create_rag_eval_datasets.py
```

## Run Evaluation

```bash
# Baseline (no MCTS)
python scripts/evaluate_rag.py \
  --dataset rag-eval-dataset \
  --limit 50 \
  --mcts-enabled false \
  --output results/baseline_eval.csv

# With MCTS
python scripts/evaluate_rag.py \
  --dataset rag-eval-dataset \
  --limit 50 \
  --mcts-enabled true \
  --output results/mcts_eval.csv
```

## View Results

```bash
# Summary statistics
python -c "import pandas as pd; print(pd.read_csv('results/baseline_eval.csv').describe())"
python -c "import pandas as pd; print(pd.read_csv('results/mcts_eval.csv').describe())"
```

## Expected Metrics

| Metric | Target | Good | Warning | Critical |
|--------|--------|------|---------|----------|
| Faithfulness | ≥ 0.80 | ≥ 0.80 | < 0.70 | < 0.50 |
| Answer Relevancy | ≥ 0.75 | ≥ 0.75 | < 0.70 | < 0.50 |
| Context Precision | ≥ 0.80 | ≥ 0.80 | < 0.70 | < 0.50 |
| Context Recall | ≥ 0.80 | ≥ 0.80 | < 0.70 | < 0.50 |

## Monitoring

- **Grafana**: http://localhost:3000/d/rag-evaluation
- **Prometheus**: http://localhost:9090
- **Weights & Biases**: https://wandb.ai
- **LangSmith**: https://smith.langchain.com

## CI/CD

```bash
# Trigger via GitHub CLI
gh workflow run ci.yml --ref your-branch

# Or via GitHub Actions UI
# Actions → CI Pipeline → Run workflow
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dataset not found | Run `python scripts/create_rag_eval_datasets.py` |
| Missing API key | Set `LANGSMITH_API_KEY`, `WANDB_API_KEY`, etc. |
| Low scores | Check LLM model, retrieval quality, dataset |
| Import error | Run `pip install ragas langsmith wandb` |

## Full Documentation

See [RAG_EVALUATION.md](./RAG_EVALUATION.md) for complete guide.
