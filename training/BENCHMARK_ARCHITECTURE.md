# Benchmark Suite Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Benchmark Suite Architecture                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   User Interface    │
├─────────────────────┤
│ • CLI (run_benchmarks.py)
│ • Python API        │
│ • Configuration     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BenchmarkSuite (Core)                         │
├─────────────────────────────────────────────────────────────────┤
│  run_retrieval_benchmark()                                       │
│  run_reasoning_benchmark()                                       │
│  run_code_generation_benchmark()                                 │
│  compare_runs()                                                  │
│  generate_report()                                               │
│  visualize_results()                                             │
└───┬─────────────┬────────────────┬────────────────┬─────────────┘
    │             │                │                │
    ▼             ▼                ▼                ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────┐
│Retrieval│  │Reasoning │  │     Code     │  │Statistical │
│ Metrics │  │ Metrics  │  │   Metrics    │  │  Analysis  │
├─────────┤  ├──────────┤  ├──────────────┤  ├────────────┤
│nDCG@k   │  │Accuracy  │  │Pass@k        │  │Bootstrap CI│
│Recall@k │  │Quality   │  │Syntax        │  │t-tests     │
│MRR      │  │LLM Judge │  │Quality       │  │Effect size │
│Precision│  │          │  │              │  │            │
└─────────┘  └──────────┘  └──────────────┘  └────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Data Management                             │
├─────────────────────────────────────────────────────────────────┤
│  • Dataset Loading (LangSmith / Local)                          │
│  • Result Storage (JSON / CSV / Markdown / HTML)                │
│  • Caching & Optimization                                       │
└───┬─────────────┬────────────────┬─────────────────────────────┘
    │             │                │
    ▼             ▼                ▼
┌──────────┐ ┌──────────┐  ┌─────────────┐
│LangSmith │ │  W&B     │  │Visualization│
│Integration│ │Integration│  │  Engine    │
├──────────┤ ├──────────┤  ├─────────────┤
│Datasets  │ │Logging   │  │Bar Charts   │
│Tracing   │ │Tracking  │  │Radar Plots  │
│          │ │Tables    │  │Trend Graphs │
└──────────┘ └──────────┘  └─────────────┘
```

## Component Details

### 1. BenchmarkSuite (Core Engine)

**Purpose**: Orchestrate benchmark execution and result management

**Key Methods**:
- `run_retrieval_benchmark()`: RAG evaluation
- `run_reasoning_benchmark()`: Reasoning quality
- `run_code_generation_benchmark()`: Code correctness
- `compare_runs()`: A/B testing and comparison
- `generate_report()`: Multi-format reporting
- `visualize_results()`: Plot generation

**State Management**:
- Runs history (list[BenchmarkRun])
- Integration clients (LangSmith, W&B)
- Configuration cache

### 2. Metrics Classes

#### RetrievalMetrics
```python
Input: RetrievalResult
  - query: str
  - retrieved_docs: list[str]
  - ground_truth_relevant: list[str]

Output: float (0.0 to 1.0)
  - nDCG@k
  - Recall@k
  - Precision@k
  - MRR
  - MAP
```

#### ReasoningMetrics
```python
Input: Predictions + Ground Truth
  - predicted_answer: Any
  - predicted_steps: list[str]
  - ground_truth: Any
  - ground_truth_steps: list[str]

Output: float (0.0 to 1.0)
  - Accuracy
  - Reasoning Quality
  - Step Correctness
```

#### CodeMetrics
```python
Input: Code + Tests
  - generated_code: str
  - test_cases: list[tuple]

Output: float (0.0 to 1.0)
  - Pass@k
  - Syntax Correctness
  - Code Quality
```

### 3. Statistical Analysis

**Bootstrap Confidence Intervals**:
- Resampling with replacement
- 1000 bootstrap samples
- 95% confidence level

**Paired T-Tests**:
- Compare baseline vs experimental
- Significance threshold: p < 0.05
- Effect size: Cohen's d

**Quality Gates**:
- Configurable thresholds
- Automatic pass/fail
- CI/CD integration

### 4. Data Flow

```
Input Dataset
     │
     ▼
Load from LangSmith or Local
     │
     ▼
Apply Test Function (retrieval/reasoning/code)
     │
     ▼
Collect Results
     │
     ▼
Compute Metrics
     │
     ▼
Statistical Analysis
     │
     ▼
Generate Reports & Visualizations
     │
     ▼
Export (JSON/CSV/MD/HTML)
     │
     ▼
Log to W&B/LangSmith
```

### 5. Integration Architecture

```
┌─────────────────────────────────────────────┐
│            External Systems                  │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │  LangSmith   │      │  Weights & Biases│ │
│  ├──────────────┤      ├─────────────────┤ │
│  │ • Datasets   │      │ • Experiment    │ │
│  │ • Examples   │      │   Tracking      │ │
│  │ • Tracing    │      │ • Visualizations│ │
│  └──────┬───────┘      └────────┬────────┘ │
│         │                       │          │
└─────────┼───────────────────────┼──────────┘
          │                       │
          └───────┬───────────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ BenchmarkSuite  │
        │   Integration   │
        │     Layer       │
        └─────────────────┘
```

### 6. CLI Architecture

```
scripts/run_benchmarks.py
          │
          ├─> Parse Arguments
          │
          ├─> Load Configuration
          │
          ├─> Initialize BenchmarkSuite
          │
          ├─> Run Selected Benchmarks
          │     ├─> rag_retrieval
          │     ├─> reasoning
          │     └─> code_generation
          │
          ├─> Compare with Baseline (optional)
          │
          ├─> Generate Reports
          │     ├─> JSON
          │     ├─> Markdown
          │     ├─> HTML
          │     └─> CSV
          │
          └─> Generate Visualizations (optional)
                ├─> Metric Comparison
                ├─> Radar Plot
                └─> Trend Analysis
```

## File Organization

```
training/
├── benchmark_suite.py          # Core implementation (1,705 lines)
│   ├── BenchmarkSuite class
│   ├── Metrics classes
│   ├── Statistical analysis
│   └── Visualization
│
├── benchmark_config.yaml       # Configuration
│   ├── Benchmark settings
│   ├── Dataset definitions
│   ├── Integration config
│   └── Quality gates
│
├── benchmark_examples.py       # Usage examples (408 lines)
│   ├── Basic usage
│   ├── A/B testing
│   ├── CI/CD integration
│   └── Custom metrics
│
├── BENCHMARK_SUITE_README.md   # Full documentation
├── BENCHMARK_QUICKSTART.md     # Quick start guide
└── BENCHMARK_ARCHITECTURE.md   # This file

scripts/
└── run_benchmarks.py          # CLI interface (348 lines)

tests/
└── test_benchmark_suite.py    # Unit tests (426 lines)
```

## Execution Flow

### Single Benchmark Run

```
1. Initialize BenchmarkSuite
   └─> Load config
   └─> Setup output directory
   └─> Initialize integrations

2. Load Dataset
   └─> Try LangSmith
   └─> Fall back to mock data

3. Execute Test Function
   └─> For each example in dataset:
       └─> Call user's test function
       └─> Collect results
       └─> Track timing/memory

4. Compute Metrics
   └─> Calculate all requested metrics
   └─> Compute confidence intervals
   └─> Store in MetricResult objects

5. Create BenchmarkRun
   └─> Package all results
   └─> Add metadata
   └─> Store timestamp

6. Generate Outputs
   └─> Log to integrations
   └─> Generate reports
   └─> Create visualizations

7. Return BenchmarkRun
```

### Comparison Flow

```
1. Load Baseline Run
   └─> Find by timestamp

2. Load Comparison Runs
   └─> Find all by timestamps

3. Compute Deltas
   └─> For each metric:
       └─> delta = comparison - baseline

4. Statistical Testing
   └─> Paired t-test for each metric
   └─> Determine significance

5. Generate Recommendations
   └─> Check for improvements
   └─> Check for regressions
   └─> Create actionable items

6. Create ComparisonReport
   └─> Package all comparisons
   └─> Export to JSON

7. Return ComparisonReport
```

## Extension Points

### Adding New Metrics

```python
# 1. Add to appropriate metrics class
class RetrievalMetrics:
    @staticmethod
    def new_metric(result: RetrievalResult) -> float:
        # Implementation
        return score

# 2. Update BENCHMARKS config
BENCHMARKS["rag_retrieval"]["metrics"].append("new_metric")

# 3. Use in benchmark
suite.run_retrieval_benchmark(...)
# Automatically computed!
```

### Adding New Benchmark Category

```python
# 1. Add to BenchmarkSuite
def run_new_benchmark(self, dataset_name, test_fn, model_config):
    # Load dataset
    # Run tests
    # Compute metrics
    # Return BenchmarkRun

# 2. Add to BENCHMARKS config
BENCHMARKS["new_category"] = {
    "datasets": [...],
    "metrics": [...]
}

# 3. Use
suite.run_new_benchmark(...)
```

### Custom Dataset Loader

```python
# Override dataset loading
def load_custom_dataset(dataset_name: str) -> list[dict]:
    # Your implementation
    return dataset

# Inject
suite._load_dataset = load_custom_dataset
```

## Performance Optimization

### Batch Processing
```python
# Process queries in batches
batch_size = 100
for i in range(0, len(queries), batch_size):
    batch = queries[i:i+batch_size]
    results = retrieval_fn.batch_retrieve(batch)
```

### Caching
```python
# Cache retrieval results
cache_file = Path("cache/retrieval_results.json")
if cache_file.exists():
    results = json.load(cache_file.open())
else:
    results = run_retrieval(...)
    json.dump(results, cache_file.open("w"))
```

### Parallel Execution
```python
# Run multiple benchmarks in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(suite.run_retrieval_benchmark, ...)
        for dataset in datasets
    ]
    results = [f.result() for f in futures]
```

## Error Handling

### Graceful Degradation
- Missing dependencies → Use simplified versions
- Dataset not found → Use mock data
- Integration failures → Continue without logging
- Visualization errors → Skip plots

### Logging Strategy
```python
logger.debug()    # Detailed execution flow
logger.info()     # Key milestones
logger.warning()  # Degraded functionality
logger.error()    # Failures (non-fatal)
```

## Testing Strategy

### Unit Tests
- Individual metric computation
- Statistical functions
- Data class behavior

### Integration Tests
- Full benchmark runs
- Report generation
- Comparison logic

### Mock Support
- Mock numpy for lightweight testing
- Mock LangSmith/W&B clients
- Synthetic test data

## Deployment

### Development
```bash
python training/benchmark_suite.py  # Run tests
python training/benchmark_examples.py  # Try examples
```

### Production
```bash
python scripts/run_benchmarks.py --all --config prod_config.yaml
```

### CI/CD
```yaml
# GitHub Actions
- run: python scripts/run_benchmarks.py --all
- run: python -c "check_quality_gates()"
```

## Summary

The benchmark suite provides:

1. **Comprehensive Evaluation**: 5+ metrics across 3 categories
2. **Statistical Rigor**: CI, significance testing, effect size
3. **Production Ready**: Error handling, logging, monitoring
4. **Extensible**: Easy to add metrics/benchmarks
5. **Well Documented**: 3 documentation files + examples
6. **Fully Tested**: Unit tests with mock support
7. **Integrated**: LangSmith, W&B, CI/CD ready

Total: **2,887 lines** of production-ready code.
