# Benchmark Suite - Quick Start Guide

## 30-Second Setup

```bash
# 1. Install dependencies
pip install numpy pandas scipy matplotlib seaborn

# 2. Run test
python training/benchmark_suite.py

# 3. Run benchmarks via CLI
python scripts/run_benchmarks.py --all
```

## 5-Minute Tutorial

### Step 1: Basic Retrieval Benchmark

```python
from training.benchmark_suite import BenchmarkSuite

# Initialize
suite = BenchmarkSuite()

# Define retrieval function
def my_retrieval(query):
    # Your RAG system here
    return {"doc_ids": ["doc1", "doc2"], "scores": [0.9, 0.8]}

# Run benchmark
run = suite.run_retrieval_benchmark(
    dataset_name="custom_mcts",
    retrieval_fn=my_retrieval,
    model_config={"model": "my-model-v1"}
)

# View results
for metric_name, metric in run.metrics.items():
    print(f"{metric_name}: {metric.value:.4f}")
```

### Step 2: Generate Reports

```python
# JSON
suite.generate_report(run, "json", "report.json")

# Markdown
suite.generate_report(run, "markdown", "report.md")

# CSV
suite.export_to_csv(run, "metrics.csv")
```

### Step 3: Visualize

```python
# Generate all plots
plots = suite.visualize_results()
# Creates: metric_comparison.png, radar_plot.png, trend_plot.png
```

## Common Use Cases

### Use Case 1: A/B Testing

```python
# Test two models
run_a = suite.run_retrieval_benchmark("dataset", model_a_fn, {"name": "A"})
run_b = suite.run_retrieval_benchmark("dataset", model_b_fn, {"name": "B"})

# Compare
comparison = suite.compare_runs(run_a.timestamp, [run_b.timestamp])
print(comparison.recommendations)
```

### Use Case 2: CI/CD Quality Gates

```python
# Run benchmark
run = suite.run_retrieval_benchmark(...)

# Check thresholds
THRESHOLDS = {"nDCG@10": 0.70, "Recall@100": 0.85}

for metric, threshold in THRESHOLDS.items():
    if run.metrics[metric].value < threshold:
        print(f"FAIL: {metric} below threshold")
        exit(1)

print("PASS: All quality gates passed")
```

### Use Case 3: Track Improvements

```python
# Run multiple versions
for version in ["v1", "v2", "v3"]:
    run = suite.run_retrieval_benchmark(
        "dataset",
        get_model(version),
        {"version": version}
    )

# Generate trend visualization
suite.visualize_results()  # Shows improvement over time
```

## CLI Commands

```bash
# Run all benchmarks
python scripts/run_benchmarks.py --all

# Run specific benchmark
python scripts/run_benchmarks.py --benchmark rag_retrieval

# With custom config
python scripts/run_benchmarks.py --config my_config.yaml --all

# Compare with baseline
python scripts/run_benchmarks.py --all --compare-with baseline_id

# Generate visualizations
python scripts/run_benchmarks.py --all --visualize

# Verbose output
python scripts/run_benchmarks.py --all --verbose
```

## Key Metrics Explained

| Metric | What It Measures | Good Value | When to Use |
|--------|------------------|------------|-------------|
| nDCG@10 | Ranking quality in top 10 | > 0.70 | Evaluate retrieval ranking |
| Recall@100 | Coverage in top 100 | > 0.85 | Ensure relevant docs found |
| MRR | Position of first relevant | > 0.75 | Quick answer retrieval |
| Precision@5 | Accuracy in top 5 | > 0.80 | High-precision use cases |
| Accuracy | Correct answers | > 0.80 | Reasoning evaluation |
| Pass@1 | Code that passes tests | > 0.60 | Code generation |

## Configuration Template

```yaml
# benchmark_config.yaml
benchmarks:
  rag_retrieval:
    enabled: true
    datasets: ["custom_mcts"]
    k_values: [5, 10, 20, 100]

output:
  base_dir: "./benchmarks"
  generate_visualizations: true

integrations:
  wandb:
    enabled: true
    project: "my-project"
```

## Integration with W&B

```bash
# Set API key
export WANDB_API_KEY="your_key"

# Run with tracking
python scripts/run_benchmarks.py --all
# Results automatically logged to W&B
```

## Integration with LangSmith

```bash
# Set API key
export LANGSMITH_API_KEY="your_key"

# Run with datasets
python scripts/run_benchmarks.py --all
# Uses LangSmith datasets automatically
```

## Troubleshooting

**Problem**: Import errors
```bash
Solution: pip install numpy pandas scipy matplotlib seaborn
```

**Problem**: No datasets found
```bash
Solution: Check LANGSMITH_API_KEY or falls back to mock data
```

**Problem**: Visualization fails
```bash
Solution: Install matplotlib/seaborn or disable in config
```

## Examples

See complete examples:
```bash
# Run all examples
python training/benchmark_examples.py

# Run specific example
python -c "from training.benchmark_examples import example_basic_retrieval_benchmark; example_basic_retrieval_benchmark()"
```

## File Structure

```
training/
├── benchmark_suite.py          # Main benchmark code
├── benchmark_config.yaml       # Configuration
├── benchmark_examples.py       # Usage examples
├── BENCHMARK_SUITE_README.md   # Full documentation
└── BENCHMARK_QUICKSTART.md     # This file

scripts/
└── run_benchmarks.py           # CLI interface

tests/
└── test_benchmark_suite.py     # Unit tests

benchmarks/                     # Output directory
├── report.json
├── report.md
├── metrics.csv
└── plots/
    ├── metric_comparison.png
    ├── radar_plot.png
    └── trend_plot.png
```

## Next Steps

1. **Customize**: Edit `benchmark_config.yaml` for your needs
2. **Integrate**: Connect your RAG/reasoning systems
3. **Baseline**: Establish baseline metrics
4. **Iterate**: Track improvements over time
5. **Deploy**: Use in CI/CD for quality gates

## Resources

- Full docs: `training/BENCHMARK_SUITE_README.md`
- Examples: `training/benchmark_examples.py`
- Tests: `tests/test_benchmark_suite.py`
- API: See inline documentation

## Support

Check examples first, then review test cases for usage patterns.
