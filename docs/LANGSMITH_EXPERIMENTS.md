# LangSmith Experiments Guide

Complete guide to running experiments on HRM, TRM, and MCTS agents using LangSmith datasets and evaluation framework.

## Overview

This framework enables systematic comparison of:
- **Model variations** (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
- **MCTS configurations** (100, 200, 500 iterations)
- **Agent strategies** (HRM-only, TRM-only, HRM+TRM, full-stack with MCTS)
- **Performance metrics** (latency, confidence, decision quality)

## Quick Start

### 1. Prerequisites

```bash
# Set LangSmith credentials
export LANGSMITH_API_KEY="your_api_key_here"
export LANGSMITH_PROJECT="langgraph-multi-agent-mcts"

# Optional: set organization ID for URL generation
export LANGSMITH_ORG_ID="your_org_id"
```

### 2. Create Datasets

```bash
# Create all datasets in LangSmith
python scripts/create_langsmith_datasets.py
```

This creates three datasets:
- **tactical_e2e_scenarios** - Tactical military decision scenarios
- **cybersecurity_e2e_scenarios** - Incident response scenarios
- **mcts_benchmark_scenarios** - MCTS performance benchmarks

### 3. Run Experiments

```bash
# Run all experiments on all datasets
python scripts/run_langsmith_experiments.py

# Run specific experiment
python scripts/run_langsmith_experiments.py --experiment exp_full_stack_mcts_200

# Run on specific dataset
python scripts/run_langsmith_experiments.py --dataset tactical_e2e_scenarios

# List available experiments
python scripts/run_langsmith_experiments.py --list-experiments
```

### 4. View Results

Navigate to your LangSmith project:
```
https://smith.langchain.com/o/YOUR_ORG_ID/projects/p/langgraph-multi-agent-mcts
```

Filter by experiment tags:
- `tags: experiment`
- `tags: exp_hrm_trm_baseline`
- `tags: exp_full_stack_mcts_200`

---

## Available Experiments

### Baseline Experiments

#### `exp_hrm_trm_baseline`
**Description**: Baseline HRM+TRM without MCTS
**Configuration**:
- Model: `gpt-4o`
- MCTS: Disabled
- Strategy: `hrm_trm`

**Use case**: Establish baseline performance for HRM+TRM collaboration

---

### MCTS Iteration Experiments

#### `exp_full_stack_mcts_100`
**Description**: Full stack with 100 MCTS iterations
**Configuration**:
- Model: `gpt-4o`
- MCTS: Enabled (100 iterations)
- Strategy: `full_stack` (HRM+TRM+MCTS)

#### `exp_full_stack_mcts_200`
**Description**: Full stack with 200 MCTS iterations
**Configuration**:
- Model: `gpt-4o`
- MCTS: Enabled (200 iterations)
- Strategy: `full_stack`

#### `exp_full_stack_mcts_500`
**Description**: Full stack with 500 MCTS iterations
**Configuration**:
- Model: `gpt-4o`
- MCTS: Enabled (500 iterations)
- Strategy: `full_stack`

**Use case**: Compare MCTS performance vs iteration count to find optimal trade-off between quality and latency

---

### Model Comparison Experiments

#### `exp_model_gpt4o_mini`
**Description**: Cost-optimized model variant
**Configuration**:
- Model: `gpt-4o-mini`
- MCTS: Disabled
- Strategy: `hrm_trm`

**Use case**: Evaluate if smaller model maintains quality for cost savings

---

## Datasets

### Tactical E2E Scenarios

**Dataset**: `tactical_e2e_scenarios`

**Examples**:
1. **Defensive Strategy** - Limited visibility night defense
2. **Multi-sector Threat** - Priority targeting with limited resources
3. **Terrain Advantage** - Offensive vs defensive posture decision

**Evaluation Metrics**:
- Presence of expected elements (defensive_position, threat_assessment, etc.)
- Confidence thresholds
- Risk level assessment

### Cybersecurity E2E Scenarios

**Dataset**: `cybersecurity_e2e_scenarios`

**Examples**:
1. **APT28 Detection** - Credential harvesting containment
2. **Ransomware Response** - Recovery strategy with compromised backups
3. **C2 Traffic** - Investigation and response workflow

**Evaluation Metrics**:
- Threat actor identification
- Containment action completeness
- Severity assessment
- Response sequence quality

### MCTS Benchmark Scenarios

**Dataset**: `mcts_benchmark_scenarios`

**Examples**:
1. **Neutral Position** - 5 action choices, secure objective
2. **Defensive Position** - Limited resources, minimize casualties

**Evaluation Metrics**:
- Win probability threshold
- Best action selection
- Convergence iteration count
- Tree exploration efficiency

---

## Evaluation Metrics

### Agent-Specific Metrics

#### HRM Metrics
- `hrm_confidence`: Confidence score (0.0-1.0)
- `hierarchical_objectives`: Count of identified objectives
- `decomposition_depth`: Levels in task hierarchy
- `objective_clarity_score`: Quality of objective definitions

#### TRM Metrics
- `trm_confidence`: Confidence score (0.0-1.0)
- `refinement_cycles`: Number of refinement iterations
- `alternatives_evaluated`: Count of alternatives considered
- `convergence_achieved`: Boolean convergence flag
- `improvement`: Quality improvement over iterations

#### MCTS Metrics
- `mcts_win_probability`: Best action win probability
- `mcts_iterations`: Number of simulations run
- `best_action`: Recommended action
- `tree_depth`: Maximum search depth
- `exploration_rate`: % of actions explored

### System Metrics
- `elapsed_ms`: End-to-end latency (milliseconds)
- `consensus_score`: Agreement across agents
- `success`: Boolean success flag
- `error`: Error message (if failed)

---

## Custom Experiment Configuration

### Adding New Experiments

Edit `scripts/run_langsmith_experiments.py`:

```python
EXPERIMENTS["exp_custom_name"] = ExperimentConfig(
    name="exp_custom_name",
    description="Custom experiment description",
    model="gpt-4o",
    use_mcts=True,
    mcts_iterations=300,
    agent_strategy="full_stack",
)
```

### Adding New Datasets

Edit `scripts/create_langsmith_datasets.py`:

```python
def create_custom_dataset() -> str:
    examples = [
        {
            "inputs": {
                "query": "Your scenario description...",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "custom",
            },
            "outputs": {
                "expected_elements": ["element1", "element2"],
                "confidence_threshold": 0.75,
            },
        },
    ]

    dataset_id = create_test_dataset(
        dataset_name="custom_scenarios",
        examples=examples,
        description="Custom scenario set",
    )
    return dataset_id
```

Then call it in `main()`:
```python
custom_id = create_custom_dataset()
```

---

## Filtering Results in LangSmith

### By Experiment
```
tags: experiment AND tags: exp_full_stack_mcts_200
```

### By Agent
```
tags: experiment AND tags: hrm
tags: experiment AND tags: trm
tags: experiment AND tags: mcts
```

### By Dataset
```
metadata.dataset: "tactical_e2e_scenarios"
metadata.dataset: "cybersecurity_e2e_scenarios"
```

### By Performance
```
metadata.elapsed_ms < 1000
metadata.hrm_confidence > 0.85
metadata.mcts_win_probability > 0.75
```

### By Success/Failure
```
metadata.success: true
metadata.success: false
```

---

## Dashboard Recommendations

### Experiment Comparison Dashboard

**Filters**: `tags: experiment`

**Charts**:
1. **Latency by Experiment** - Bar chart of avg_elapsed_ms across experiments
2. **Confidence Trends** - Line chart showing HRM/TRM confidence over time
3. **Success Rate** - Pie chart of successful vs failed runs
4. **MCTS Performance** - Scatter plot of iterations vs win_probability

### Model Comparison Dashboard

**Filters**: `tags: experiment AND (tags: gpt-4o OR tags: gpt-4o-mini)`

**Charts**:
1. **Cost vs Quality** - Scatter: latency (x) vs confidence (y), colored by model
2. **Model Distribution** - Count of runs per model
3. **Error Rate by Model** - Bar chart of failures per model

### Dataset Performance Dashboard

**Filters**: `tags: experiment`

**Group by**: `metadata.dataset`

**Charts**:
1. **Success Rate by Dataset** - Stacked bar showing success/failure per dataset
2. **Avg Latency by Dataset** - Bar chart
3. **Confidence by Dataset** - Box plot showing distribution

---

## CI/CD Integration

### GitHub Actions Example

Add to `.github/workflows/experiments.yml`:

```yaml
name: LangSmith Experiments

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  run-experiments:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run experiments
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          LANGSMITH_PROJECT: langgraph-multi-agent-mcts
        run: |
          python scripts/run_langsmith_experiments.py

      - name: Check for regressions
        run: |
          # Add custom regression detection logic here
          # Fail build if metrics drop below threshold
          python scripts/check_experiment_regressions.py
```

### Regression Detection

Create `scripts/check_experiment_regressions.py`:

```python
from langsmith import Client

client = Client()

# Get recent experiment runs
runs = list(client.list_runs(
    project_name="langgraph-multi-agent-mcts",
    filter='tags: experiment',
    limit=100
))

# Check for regressions
for run in runs:
    metadata = run.extra or {}

    # Check latency threshold
    if metadata.get('elapsed_ms', 0) > 5000:
        print(f"⚠️  Latency regression: {run.name} took {metadata['elapsed_ms']}ms")

    # Check confidence threshold
    if metadata.get('hrm_confidence', 1.0) < 0.70:
        print(f"⚠️  Confidence regression: {run.name} HRM confidence {metadata['hrm_confidence']}")

# Exit with error if regressions found
```

---

## Best Practices

### 1. Experiment Naming
- Use consistent prefixes: `exp_`
- Include configuration in name: `exp_full_stack_mcts_200`
- Add version for iterations: `exp_baseline_v2`

### 2. Dataset Management
- Keep datasets focused (3-5 examples per scenario type)
- Include diverse difficulty levels
- Update expected outputs as system improves
- Version datasets when making breaking changes

### 3. Metrics Tracking
- Always include `elapsed_ms` for performance tracking
- Add `success` boolean for reliability metrics
- Use consistent metadata keys across experiments
- Include model/config in metadata for filtering

### 4. Result Analysis
- Run experiments multiple times for statistical significance
- Compare same dataset across experiments
- Monitor trends over time, not just point-in-time
- Set up alerts for regression detection

### 5. Cost Management
- Start with smaller datasets for expensive models
- Use `gpt-4o-mini` for initial testing
- Limit MCTS iterations during development
- Schedule expensive experiments for off-peak hours

---

## Troubleshooting

### Dataset Not Found
```bash
# List available datasets
python -c "from langsmith import Client; c = Client(); print([d.name for d in c.list_datasets()])"

# Recreate if missing
python scripts/create_langsmith_datasets.py
```

### Experiment Not Running
```bash
# Check environment variables
echo $LANGSMITH_API_KEY
echo $LANGSMITH_PROJECT

# Verify credentials
python -c "from langsmith import Client; c = Client(); print('Connected!')"
```

### Traces Not Appearing
```bash
# Enable tracing explicitly
export LANGSMITH_TRACING=true

# Check project name matches
export LANGSMITH_PROJECT="langgraph-multi-agent-mcts"
```

### Performance Issues
```bash
# Reduce dataset size for testing
python scripts/run_langsmith_experiments.py --dataset tactical_e2e_scenarios

# Run single experiment
python scripts/run_langsmith_experiments.py --experiment exp_hrm_trm_baseline
```

---

## Next Steps

1. **Create initial datasets**: Run `python scripts/create_langsmith_datasets.py`
2. **Run baseline experiment**: `python scripts/run_langsmith_experiments.py --experiment exp_hrm_trm_baseline`
3. **View in LangSmith UI**: Check results and create dashboards
4. **Compare MCTS variants**: Run full-stack experiments with different iteration counts
5. **Set up CI/CD**: Add to GitHub Actions for automated experimentation
6. **Monitor regressions**: Create alerts for performance degradation

---

## Resources

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **Experiments Guide**: https://docs.smith.langchain.com/evaluation
- **Datasets API**: https://docs.smith.langchain.com/evaluation/datasets
- **Main Tracing Guide**: [docs/LANGSMITH_E2E.md](./LANGSMITH_E2E.md)
- **Agent Tracing Guide**: [docs/AGENT_TRACING_GUIDE.md](./AGENT_TRACING_GUIDE.md)

---

**Last Updated**: 2025-01-17
**Status**: ✅ Complete - Ready for experimentation
