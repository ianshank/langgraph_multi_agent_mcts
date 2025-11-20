# Benchmark Suite for RAG System Evaluation

Comprehensive benchmarking framework for evaluating RAG (Retrieval-Augmented Generation) systems, knowledge base quality, reasoning capabilities, and code generation.

## Features

### Benchmark Categories

1. **RAG Retrieval**
   - Datasets: custom_mcts, custom_langgraph, custom_multiagent
   - Metrics: nDCG@k, Recall@k, MRR, Precision@k, MAP

2. **Reasoning**
   - Datasets: gsm8k_subset, math_subset, dabstep_subset
   - Metrics: Accuracy, Reasoning_quality, Step_correctness

3. **Code Generation**
   - Datasets: humaneval_subset, mbpp_subset
   - Metrics: Pass@k, Syntax_correctness, Code_quality

### Core Capabilities

- **5+ Evaluation Metrics**: nDCG@10, Recall@100, MRR, Precision@k, LLM-as-judge
- **Statistical Analysis**: Bootstrap confidence intervals, paired t-tests, effect size computation
- **Baseline Comparison**: Compare multiple runs, track improvements/regressions
- **Automated Testing**: CI/CD integration with quality gates
- **Visualization**: Comparison charts, radar plots, trend graphs
- **Export Formats**: JSON, CSV, Markdown, HTML
- **Integration**: LangSmith datasets, W&B tracking

## Installation

### Prerequisites

```bash
# Install required dependencies
pip install numpy>=1.24.0
pip install pandas>=2.0.0
pip install scipy>=1.11.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0

# Optional integrations
pip install wandb>=0.16.0
pip install langsmith>=0.1.0
```

### Setup

```bash
# Clone the repository
cd langgraph_multi_agent_mcts

# Verify installation
python -c "from training.benchmark_suite import BenchmarkSuite; print('Success!')"
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from training.benchmark_suite import BenchmarkSuite

# Initialize benchmark suite
suite = BenchmarkSuite(output_dir=Path("./benchmarks"))

# Define your retrieval function
def my_retrieval_fn(query: str):
    # Your RAG implementation here
    # Return format: {"doc_ids": [...], "scores": [...]}
    return {"doc_ids": ["doc1", "doc2"], "scores": [0.9, 0.8]}

# Run benchmark
model_config = {"embedding_model": "all-MiniLM-L6-v2"}
run = suite.run_retrieval_benchmark(
    dataset_name="custom_mcts",
    retrieval_fn=my_retrieval_fn,
    model_config=model_config,
    k_values=[5, 10, 20, 100]
)

# View results
for metric_name, metric in run.metrics.items():
    print(f"{metric_name}: {metric.value:.4f}")

# Generate report
suite.generate_report(run, output_format="markdown",
                     output_file=Path("./benchmarks/report.md"))
```

### Command-Line Interface

```bash
# Run all benchmarks
python scripts/run_benchmarks.py --all

# Run specific benchmark
python scripts/run_benchmarks.py --benchmark rag_retrieval

# Run with custom config
python scripts/run_benchmarks.py --config custom_config.yaml --all

# Compare with baseline
python scripts/run_benchmarks.py --all --compare-with baseline_run_id

# Generate visualizations
python scripts/run_benchmarks.py --all --visualize
```

## Configuration

### Benchmark Configuration File

Create `training/benchmark_config.yaml`:

```yaml
benchmarks:
  rag_retrieval:
    enabled: true
    datasets:
      - name: "custom_mcts"
        size: 100
    metrics:
      - "nDCG@10"
      - "Recall@100"
      - "MRR"
    k_values: [5, 10, 20, 100]

statistical:
  confidence_level: 0.95
  bootstrap_samples: 1000
  significance_threshold: 0.05

output:
  base_dir: "./benchmarks"
  generate_visualizations: true
  export_formats: ["json", "csv", "markdown"]

integrations:
  wandb:
    enabled: true
    project: "rag-benchmarks"
  langsmith:
    enabled: true
    project: "rag-evaluation"
```

## Detailed Usage

### 1. RAG Retrieval Benchmark

```python
from training.benchmark_suite import BenchmarkSuite

suite = BenchmarkSuite()

def retrieval_function(query: str):
    """Your retrieval implementation."""
    # Call your RAG system
    docs = your_rag_system.retrieve(query, k=100)
    return {
        "doc_ids": [doc.id for doc in docs],
        "scores": [doc.score for doc in docs]
    }

run = suite.run_retrieval_benchmark(
    dataset_name="custom_mcts",
    retrieval_fn=retrieval_function,
    model_config={"model": "my-retriever-v1"},
    k_values=[5, 10, 20, 100]
)

# Access metrics
ndcg_10 = run.metrics["nDCG@10"].value
recall_100 = run.metrics["Recall@100"].value
mrr = run.metrics["MRR"].value

print(f"nDCG@10: {ndcg_10:.4f}")
print(f"Recall@100: {recall_100:.4f}")
print(f"MRR: {mrr:.4f}")
```

### 2. Reasoning Benchmark

```python
def reasoning_function(problem: str):
    """Your reasoning implementation."""
    # Call your reasoning system
    result = your_reasoning_system.solve(problem)
    return {
        "answer": result.answer,
        "steps": result.reasoning_steps
    }

run = suite.run_reasoning_benchmark(
    dataset_name="gsm8k_subset",
    reasoning_fn=reasoning_function,
    model_config={"model": "gpt-4", "temperature": 0.7},
    use_llm_judge=True  # Enable LLM-as-judge scoring
)

# Access metrics
accuracy = run.metrics["Accuracy"].value
quality = run.metrics["Reasoning_quality"].value
```

### 3. Code Generation Benchmark

```python
def code_generation_function(problem: str):
    """Your code generation implementation."""
    # Call your code generation system
    code = your_codegen_system.generate(problem)
    return code

run = suite.run_code_generation_benchmark(
    dataset_name="humaneval_subset",
    code_gen_fn=code_generation_function,
    model_config={"model": "codex", "max_tokens": 512},
    k_values=[1, 10]
)

# Access metrics
pass_at_1 = run.metrics["Pass@1"].value
syntax_correct = run.metrics["Syntax_correctness"].value
```

### 4. A/B Testing

```python
# Run baseline
baseline_run = suite.run_retrieval_benchmark(
    dataset_name="custom_mcts",
    retrieval_fn=baseline_retrieval_fn,
    model_config={"name": "baseline", "version": "1.0"}
)

# Run experimental
experimental_run = suite.run_retrieval_benchmark(
    dataset_name="custom_mcts",
    retrieval_fn=experimental_retrieval_fn,
    model_config={"name": "experimental", "version": "2.0"}
)

# Compare
comparison = suite.compare_runs(
    baseline_run_id=baseline_run.timestamp,
    comparison_run_ids=[experimental_run.timestamp],
    output_file=Path("./comparison.json")
)

# View recommendations
for rec in comparison.recommendations:
    print(rec)
```

### 5. Visualization

```python
# Run multiple benchmarks
runs = []
for config in model_configs:
    run = suite.run_retrieval_benchmark(
        dataset_name="custom_mcts",
        retrieval_fn=get_retrieval_fn(config),
        model_config=config
    )
    runs.append(run)

# Generate visualizations
plot_files = suite.visualize_results(runs=runs)

# Plots generated:
# - metric_comparison.png: Bar chart comparing metrics
# - radar_plot.png: Multi-metric radar chart
# - trend_plot.png: Historical trend analysis
```

### 6. Export Results

```python
# Export to JSON
suite.generate_report(run, output_format="json",
                     output_file=Path("./report.json"))

# Export to Markdown
suite.generate_report(run, output_format="markdown",
                     output_file=Path("./report.md"))

# Export to HTML
suite.generate_report(run, output_format="html",
                     output_file=Path("./report.html"))

# Export to CSV
suite.export_to_csv(run, output_file=Path("./metrics.csv"))
```

## Integration with LangSmith and W&B

### Setup

```bash
# Set environment variables
export LANGSMITH_API_KEY="your_key_here"
export WANDB_API_KEY="your_key_here"
export WANDB_PROJECT="rag-benchmarks"
```

### Usage

```python
config = {
    "use_wandb": True,
    "wandb_project": "rag-benchmarks",
    "wandb_entity": "your_entity"
}

suite = BenchmarkSuite(config=config)
suite.initialize_integrations()

# Run benchmarks - results automatically logged
run = suite.run_retrieval_benchmark(...)

# Results are now in W&B and LangSmith!
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmark Quality Gates

on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run benchmarks
        run: |
          python scripts/run_benchmarks.py --all

      - name: Check quality gates
        run: |
          python -c "
          import json
          from pathlib import Path

          # Load results
          results = json.load(open('benchmarks/report.json'))

          # Check thresholds
          thresholds = {
              'nDCG@10': 0.70,
              'Recall@100': 0.85,
              'MRR': 0.75
          }

          failed = []
          for metric, threshold in thresholds.items():
              value = results['metrics'][metric]['value']
              if value < threshold:
                  failed.append(f'{metric}: {value:.4f} < {threshold:.4f}')

          if failed:
              print('Quality gates failed:')
              for f in failed:
                  print(f'  - {f}')
              exit(1)
          else:
              print('All quality gates passed!')
          "
```

### Quality Gate Configuration

```yaml
# In benchmark_config.yaml
ci_cd:
  enabled: true
  fail_on_regression: true
  required_metrics:
    - metric: "nDCG@10"
      min_value: 0.70
    - metric: "Accuracy"
      min_value: 0.80
    - metric: "Pass@1"
      min_value: 0.60
```

## Metrics Reference

### Retrieval Metrics

- **nDCG@k**: Normalized Discounted Cumulative Gain at position k
  - Measures ranking quality with position-weighted relevance
  - Range: [0.0, 1.0], higher is better

- **Recall@k**: Fraction of relevant documents in top k
  - Measures coverage of relevant documents
  - Range: [0.0, 1.0], higher is better

- **Precision@k**: Fraction of top k that are relevant
  - Measures accuracy of top results
  - Range: [0.0, 1.0], higher is better

- **MRR**: Mean Reciprocal Rank
  - Measures position of first relevant result
  - Range: [0.0, 1.0], higher is better

- **MAP**: Mean Average Precision
  - Measures precision across all positions
  - Range: [0.0, 1.0], higher is better

### Reasoning Metrics

- **Accuracy**: Fraction of correct answers
  - Range: [0.0, 1.0], higher is better

- **Reasoning Quality**: Quality of reasoning steps
  - Heuristic or LLM-based scoring
  - Range: [0.0, 1.0], higher is better

### Code Generation Metrics

- **Pass@k**: Fraction of problems solved in k attempts
  - Measures functional correctness
  - Range: [0.0, 1.0], higher is better

- **Syntax Correctness**: Fraction of syntactically valid code
  - Range: [0.0, 1.0], higher is better

- **Code Quality**: Heuristic quality score
  - Checks for docs, types, comments, structure
  - Range: [0.0, 1.0], higher is better

## Statistical Analysis

### Confidence Intervals

All metrics include 95% bootstrap confidence intervals:

```python
metric = run.metrics["nDCG@10"]
print(f"Value: {metric.value:.4f}")
print(f"95% CI: [{metric.confidence_interval[0]:.4f}, {metric.confidence_interval[1]:.4f}]")
```

### Statistical Significance

Comparisons include paired t-tests:

```python
comparison = suite.compare_runs(baseline_id, [experimental_id])

for run_name, deltas in comparison.metric_deltas.items():
    for metric_name, delta in deltas.items():
        is_sig = comparison.statistical_significance[run_name][metric_name]
        print(f"{metric_name}: {delta:+.4f} (significant: {is_sig})")
```

## Examples

See `training/benchmark_examples.py` for complete examples:

```bash
# Run all examples
python training/benchmark_examples.py

# Run specific example
python -c "from training.benchmark_examples import example_basic_retrieval_benchmark; example_basic_retrieval_benchmark()"
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'numpy'**
   ```bash
   pip install numpy pandas scipy matplotlib seaborn
   ```

2. **LangSmith dataset not found**
   - Ensure LANGSMITH_API_KEY is set
   - Check dataset name matches LangSmith project
   - Falls back to mock data if unavailable

3. **W&B login required**
   ```bash
   export WANDB_API_KEY="your_key"
   # Or run: wandb login
   ```

4. **Visualization errors**
   - Ensure matplotlib and seaborn are installed
   - Check X11/display settings for remote systems

## Performance Considerations

- **Large Datasets**: Use streaming for datasets > 10k samples
- **Batch Processing**: Process queries in batches for efficiency
- **Caching**: Enable result caching for repeated evaluations
- **Parallel Execution**: Run multiple benchmarks in parallel

## Best Practices

1. **Version Control**: Track model configs and benchmark results
2. **Reproducibility**: Set random seeds, save configurations
3. **Baseline Comparison**: Always compare against established baseline
4. **Multiple Runs**: Run multiple times for statistical confidence
5. **Quality Gates**: Set clear thresholds for production deployment
6. **Documentation**: Document benchmark methodology and results

## API Reference

### Main Classes

- `BenchmarkSuite`: Main benchmark orchestrator
- `RetrievalMetrics`: Retrieval quality metrics
- `ReasoningMetrics`: Reasoning quality metrics
- `CodeMetrics`: Code generation metrics
- `StatisticalAnalysis`: Statistical testing utilities

### Data Classes

- `BenchmarkRun`: Complete benchmark run results
- `MetricResult`: Single metric result with CI
- `ComparisonReport`: Comparison between runs
- `RetrievalResult`: Single retrieval result

See inline documentation for detailed API reference.

## Contributing

To add new benchmarks or metrics:

1. Add metric computation to appropriate class
2. Update BENCHMARKS configuration
3. Add tests and examples
4. Update documentation

## License

This benchmark suite is part of the LangGraph Multi-Agent MCTS project.

## Support

For issues or questions:
- Check examples in `training/benchmark_examples.py`
- Review test cases in `training/benchmark_suite.py`
- Consult inline documentation

## Changelog

### Version 1.0.0
- Initial release
- RAG retrieval benchmarks
- Reasoning benchmarks
- Code generation benchmarks
- LangSmith and W&B integration
- Statistical analysis
- Visualization support
- CI/CD integration
