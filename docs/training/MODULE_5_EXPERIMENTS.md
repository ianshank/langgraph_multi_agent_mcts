# Module 5: Experiments & Datasets in LangSmith

**Duration:** 10 hours (2 days)
**Format:** Workshop + Experimentation Lab
**Difficulty:** Intermediate to Advanced
**Prerequisites:** Completed Module 4, understanding of experiment design and statistical analysis

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Create LangSmith datasets** for systematic testing and evaluation
2. **Design experiments** to compare models, configurations, and strategies
3. **Run evaluations** across multiple scenarios and analyze results
4. **Interpret metrics** and make data-driven optimization decisions
5. **Automate experimentation** in CI/CD pipelines for continuous improvement

---

## Session 1: Dataset Creation (2.5 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [LANGSMITH_EXPERIMENTS.md](../LANGSMITH_EXPERIMENTS.md) - Complete experiments guide
- [scripts/create_langsmith_datasets.py](../../scripts/create_langsmith_datasets.py) - Dataset creation script
- LangSmith Datasets documentation: https://docs.smith.langchain.com/evaluation/datasets

### Lecture: Dataset Design Principles (60 minutes)

#### What is a LangSmith Dataset?

**Definition:** A dataset is a collection of example inputs and expected outputs used for:
- **Regression testing:** Ensure changes don't break existing functionality
- **Benchmarking:** Compare different models or configurations
- **Quality assurance:** Validate system behavior across scenarios
- **Performance tracking:** Monitor improvements over time

**Dataset Structure:**
```python
{
    "dataset_name": "tactical_e2e_scenarios",
    "description": "Tactical military decision scenarios",
    "examples": [
        {
            "inputs": {
                "query": "What defensive strategy for urban warfare?",
                "use_rag": True,
                "use_mcts": True,
                "scenario_type": "tactical",
            },
            "outputs": {
                "expected_elements": [
                    "defensive_position",
                    "threat_assessment",
                    "force_allocation"
                ],
                "confidence_threshold": 0.75,
                "expected_risk_level": "high",
            },
        },
        # More examples...
    ]
}
```

#### Dataset Design Best Practices

**1. Coverage:**
- **Happy paths:** Common, well-formed queries
- **Edge cases:** Unusual or boundary conditions
- **Error cases:** Invalid or malformed inputs
- **Domain variety:** Different scenario types (tactical, cybersecurity, etc.)

**2. Size:**
- **Development:** 3-5 examples per scenario type (fast iteration)
- **Testing:** 10-20 examples per scenario (good coverage)
- **Production:** 50+ examples (comprehensive validation)

**3. Quality:**
- **Realistic:** Reflect actual user queries
- **Diverse:** Cover different query patterns and complexities
- **Validated:** Expected outputs verified by domain experts
- **Versioned:** Track changes over time

**4. Metadata:**
- **Scenario type:** tactical, cybersecurity, general
- **Difficulty level:** simple, moderate, complex
- **Expected latency:** Performance benchmarks
- **Domain-specific fields:** Custom evaluation criteria

#### Dataset Categories

**1. E2E Scenario Datasets:**
```python
# Tactical scenarios
tactical_examples = [
    {
        "inputs": {
            "query": "Develop defensive strategy for urban environment with limited visibility",
            "use_rag": True,
            "use_mcts": False,
            "scenario_type": "tactical",
        },
        "outputs": {
            "expected_elements": [
                "defensive_position",
                "sensor_deployment",
                "force_allocation",
                "contingency_plans"
            ],
            "confidence_threshold": 0.75,
            "expected_risk_level": "high",
        }
    },
    # More examples...
]
```

**2. Agent-Specific Datasets:**
```python
# HRM decomposition tests
hrm_examples = [
    {
        "inputs": {
            "query": "What's the best tactical approach for urban warfare?",
            "agent": "hrm",
        },
        "outputs": {
            "expected_task_count": (3, 7),  # Between 3 and 7 tasks
            "required_elements": ["terrain_analysis", "force_composition"],
            "min_confidence": 0.7,
        }
    },
]

# TRM refinement tests
trm_examples = [
    {
        "inputs": {
            "task": "Mitigate SQL injection vulnerabilities",
            "initial_solution": "Use parameterized queries",
            "agent": "trm",
        },
        "outputs": {
            "max_iterations": 5,
            "must_converge": True,
            "min_improvement": 0.15,
            "required_topics": ["input_validation", "least_privilege", "monitoring"],
        }
    },
]
```

**3. MCTS Benchmark Datasets:**
```python
# MCTS performance tests
mcts_examples = [
    {
        "inputs": {
            "scenario": "neutral_position",
            "action_choices": 5,
            "objective": "secure_position",
            "iterations": 200,
        },
        "outputs": {
            "min_win_probability": 0.65,
            "expected_best_action": "fortify_center",
            "max_iterations_to_converge": 150,
        }
    },
]
```

### Live Demo: Create First Dataset (30 minutes)

**Instructor Demo:**

```python
from langsmith import Client

client = Client()

# Define examples
examples = [
    {
        "inputs": {
            "query": "What defensive strategy for urban warfare?",
            "use_rag": True,
            "use_mcts": False,
            "scenario_type": "tactical",
        },
        "outputs": {
            "expected_elements": [
                "defensive_position",
                "threat_assessment",
                "force_allocation"
            ],
            "confidence_threshold": 0.75,
        },
    },
    {
        "inputs": {
            "query": "How to respond to APT28 credential harvesting?",
            "use_rag": True,
            "use_mcts": False,
            "scenario_type": "cybersecurity",
        },
        "outputs": {
            "expected_elements": [
                "threat_actor_identification",
                "containment_actions",
                "recovery_steps"
            ],
            "confidence_threshold": 0.8,
        },
    },
]

# Create dataset
dataset = client.create_dataset(
    dataset_name="demo_scenarios",
    description="Demo dataset for training",
)

# Add examples
for example in examples:
    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"],
    )

print(f"Dataset created: {dataset.id}")
print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")
```

### Hands-On Exercise: Create Custom Dataset (30 minutes)

**Exercise 1: Build Domain-Specific Dataset**

**Objective:** Create a dataset for your specific use case.

**Tasks:**
1. Define 5+ examples for a specific scenario type
2. Include diverse query patterns and difficulty levels
3. Specify expected outputs with validation criteria
4. Create dataset using LangSmith API
5. Verify dataset in LangSmith UI

**Template:**
```python
from langsmith import Client

def create_custom_dataset():
    """Create custom dataset for specific domain."""
    client = Client()

    examples = [
        # TODO: Add your examples
        {
            "inputs": {
                "query": "Your query here",
                # Add more input fields
            },
            "outputs": {
                "expected_elements": ["element1", "element2"],
                "confidence_threshold": 0.75,
                # Add more validation criteria
            },
        },
    ]

    # Create dataset
    dataset = client.create_dataset(
        dataset_name="your_dataset_name",
        description="Your dataset description",
    )

    # Add examples
    for example in examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs=example["inputs"],
            outputs=example["outputs"],
        )

    return dataset.id

# Run
dataset_id = create_custom_dataset()
print(f"Dataset created: {dataset_id}")
```

**Deliverable:** Dataset with 5+ examples created in LangSmith

---

## Session 2: Experiment Design (3 hours)

### Pre-Reading (30 minutes)

- [LANGSMITH_EXPERIMENTS.md](../LANGSMITH_EXPERIMENTS.md) - Available experiments
- [scripts/run_langsmith_experiments.py](../../scripts/run_langsmith_experiments.py) - Experiment runner

### Lecture: Experiment Design Principles (60 minutes)

#### What Makes a Good Experiment?

**Key Principles:**
1. **Clear hypothesis:** What are you testing?
2. **Controlled variables:** Change one thing at a time
3. **Measurable outcomes:** Define success metrics
4. **Statistical significance:** Run multiple iterations
5. **Reproducibility:** Document configuration and random seeds

#### Experiment Types

**1. Model Comparison:**
```python
experiments = {
    "baseline_gpt4o": {
        "model": "gpt-4o",
        "temperature": 0.7,
    },
    "cost_optimized_gpt4o_mini": {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
    },
    "creative_gpt4o": {
        "model": "gpt-4o",
        "temperature": 0.9,
    },
}

# Hypothesis: gpt-4o-mini maintains quality at lower cost
# Metrics: quality (confidence), latency, cost per query
```

**2. Configuration Tuning:**
```python
experiments = {
    "mcts_100": {
        "use_mcts": True,
        "mcts_iterations": 100,
    },
    "mcts_200": {
        "use_mcts": True,
        "mcts_iterations": 200,
    },
    "mcts_500": {
        "use_mcts": True,
        "mcts_iterations": 500,
    },
}

# Hypothesis: 200 iterations is optimal tradeoff
# Metrics: win_probability, latency, convergence_rate
```

**3. Agent Strategy Comparison:**
```python
experiments = {
    "hrm_only": {
        "use_hrm": True,
        "use_trm": False,
        "use_mcts": False,
    },
    "hrm_trm": {
        "use_hrm": True,
        "use_trm": True,
        "use_mcts": False,
    },
    "full_stack": {
        "use_hrm": True,
        "use_trm": True,
        "use_mcts": True,
    },
}

# Hypothesis: Full stack improves quality but increases latency
# Metrics: confidence, completeness, latency
```

**4. Prompt Engineering:**
```python
experiments = {
    "baseline_prompt": {
        "prompt_version": "v1",
    },
    "detailed_prompt": {
        "prompt_version": "v2_detailed",
    },
    "cot_prompt": {
        "prompt_version": "v3_chain_of_thought",
    },
}

# Hypothesis: Chain-of-thought improves reasoning quality
# Metrics: confidence, reasoning_quality, task_decomposition_quality
```

#### Experiment Configuration Structure

**ExperimentConfig Class:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    model: str
    temperature: float = 0.7
    use_mcts: bool = False
    mcts_iterations: Optional[int] = None
    agent_strategy: str = "full_stack"  # hrm_only, hrm_trm, full_stack
    prompt_version: str = "v1"
    tags: list[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for tracing metadata."""
        return {
            "experiment_name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "use_mcts": self.use_mcts,
            "mcts_iterations": self.mcts_iterations,
            "agent_strategy": self.agent_strategy,
        }
```

### Lecture: Metrics and Evaluation (60 minutes)

#### Agent-Specific Metrics

**HRM Metrics:**
```python
hrm_metrics = {
    "hrm_confidence": 0.85,           # Decomposition confidence
    "task_count": 5,                  # Number of tasks identified
    "hierarchical_depth": 2,          # Levels in task hierarchy
    "objective_clarity": 0.9,         # Objective definition quality
    "domain_relevance": 0.88,         # Domain-specific accuracy
}
```

**TRM Metrics:**
```python
trm_metrics = {
    "trm_confidence": 0.92,           # Final solution confidence
    "refinement_cycles": 3,           # Number of iterations
    "converged": True,                # Convergence achieved
    "improvement_rate": 0.15,         # Quality improvement
    "alternatives_evaluated": 4,      # Candidate solutions
}
```

**MCTS Metrics:**
```python
mcts_metrics = {
    "win_probability": 0.78,          # Best action win rate
    "mcts_iterations": 200,           # Simulations run
    "best_action": "fortify_center",  # Recommended action
    "tree_depth": 5,                  # Maximum search depth
    "exploration_rate": 0.65,         # % actions explored
    "convergence_iteration": 150,     # When converged
}
```

#### System-Wide Metrics

**Performance Metrics:**
```python
performance_metrics = {
    "elapsed_ms": 2345.67,            # Total latency
    "llm_call_count": 12,             # Number of LLM calls
    "total_tokens": 3500,             # Token usage
    "cost_usd": 0.08,                 # Estimated cost
    "throughput_qps": 0.43,           # Queries per second
}
```

**Quality Metrics:**
```python
quality_metrics = {
    "consensus_score": 0.88,          # Agent agreement
    "completeness": 0.92,             # All required elements present
    "coherence": 0.85,                # Response coherence
    "factual_accuracy": 0.90,         # Factual correctness
    "relevance": 0.87,                # Relevance to query
}
```

**Reliability Metrics:**
```python
reliability_metrics = {
    "success": True,                  # Success flag
    "error_type": None,               # Error type if failed
    "retry_count": 0,                 # Number of retries
    "fallback_used": False,           # Fallback triggered
}
```

#### Custom Evaluators

**Example: Content Quality Evaluator**
```python
def evaluate_content_quality(run, example):
    """
    Custom evaluator for content quality.

    Args:
        run: LangSmith run object with outputs
        example: Dataset example with expected outputs

    Returns:
        dict: Evaluation results with scores and feedback
    """
    output = run.outputs.get("response", "")
    expected = example.outputs.get("expected_elements", [])

    # Check for required elements
    elements_present = sum(
        1 for elem in expected if elem.lower() in output.lower()
    )
    completeness = elements_present / len(expected) if expected else 0

    # Check confidence threshold
    confidence = run.outputs.get("confidence", 0)
    meets_threshold = confidence >= example.outputs.get("confidence_threshold", 0.7)

    # Calculate composite score
    score = (completeness + (1.0 if meets_threshold else 0)) / 2

    return {
        "key": "content_quality",
        "score": score,
        "comment": f"Completeness: {completeness:.2%}, Confidence: {confidence:.2f}",
    }
```

### Hands-On Exercise: Design Experiment (30 minutes)

**Exercise 2: Design MCTS Iteration Experiment**

**Objective:** Design an experiment to find optimal MCTS iteration count.

**Requirements:**
1. Define hypothesis
2. Create 3+ experiment configurations
3. Specify metrics to track
4. Define success criteria
5. Estimate runtime and cost

**Template:**
```python
from dataclasses import dataclass

@dataclass
class MCTSExperiment:
    """MCTS iteration experiment design."""

    # Hypothesis
    hypothesis: str = "200 iterations provides optimal quality/latency tradeoff"

    # Configurations
    configs: dict = None

    def __post_init__(self):
        self.configs = {
            "mcts_100": ExperimentConfig(
                name="mcts_100",
                description="MCTS with 100 iterations",
                model="gpt-4o",
                use_mcts=True,
                mcts_iterations=100,
                tags=["experiment", "mcts", "mcts_100"],
            ),
            # TODO: Add mcts_200 config
            # TODO: Add mcts_500 config
        }

    # Metrics
    primary_metric: str = "win_probability"
    secondary_metrics: list = None

    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = [
                "elapsed_ms",
                "convergence_iteration",
                "exploration_rate",
            ]

    # Success criteria
    def evaluate_success(self, results: dict) -> bool:
        """Determine if experiment validated hypothesis."""
        # TODO: Implement evaluation logic
        pass

# TODO: Complete the experiment design
experiment = MCTSExperiment()
```

**Deliverable:** Complete experiment design document

---

## Session 3: Running Experiments (2.5 hours)

### Pre-Reading (30 minutes)

- [scripts/run_langsmith_experiments.py](../../scripts/run_langsmith_experiments.py) - Experiment runner
- [scripts/create_langsmith_datasets.py](../../scripts/create_langsmith_datasets.py) - Dataset creation

### Lecture: Experiment Execution (45 minutes)

#### Experiment Runner Architecture

**High-Level Flow:**
```python
def run_experiment(experiment_config, dataset_name):
    """
    Run experiment on dataset.

    Flow:
    1. Load dataset from LangSmith
    2. For each example in dataset:
       a. Apply experiment configuration
       b. Execute workflow with tracing
       c. Collect metrics
       d. Evaluate outputs
    3. Aggregate results
    4. Store in LangSmith
    """
    # Load dataset
    dataset = client.read_dataset(dataset_name=dataset_name)

    # Run on each example
    results = []
    for example in dataset.examples:
        # Apply config
        inputs = {**example.inputs, **experiment_config.to_dict()}

        # Execute with tracing
        with tracing_context(
            tags=experiment_config.tags,
            metadata=experiment_config.to_dict()
        ):
            output = workflow.invoke(inputs)

        # Evaluate
        evaluation = evaluate_output(output, example.outputs)
        results.append(evaluation)

    # Aggregate
    summary = aggregate_results(results)
    return summary
```

#### Running Experiments with Scripts

**Using run_langsmith_experiments.py:**
```bash
# Run all experiments on all datasets
python scripts/run_langsmith_experiments.py

# Run specific experiment
python scripts/run_langsmith_experiments.py --experiment exp_mcts_200

# Run on specific dataset
python scripts/run_langsmith_experiments.py --dataset tactical_e2e_scenarios

# List available experiments
python scripts/run_langsmith_experiments.py --list-experiments

# Dry run (no execution)
python scripts/run_langsmith_experiments.py --dry-run
```

**Experiment Script Structure:**
```python
# Define experiments
EXPERIMENTS = {
    "exp_hrm_trm_baseline": ExperimentConfig(
        name="exp_hrm_trm_baseline",
        description="Baseline HRM+TRM without MCTS",
        model="gpt-4o",
        use_mcts=False,
        agent_strategy="hrm_trm",
        tags=["experiment", "baseline", "hrm_trm"],
    ),
    "exp_full_stack_mcts_200": ExperimentConfig(
        name="exp_full_stack_mcts_200",
        description="Full stack with 200 MCTS iterations",
        model="gpt-4o",
        use_mcts=True,
        mcts_iterations=200,
        agent_strategy="full_stack",
        tags=["experiment", "full_stack", "mcts_200"],
    ),
    # More experiments...
}

# Define datasets
DATASETS = [
    "tactical_e2e_scenarios",
    "cybersecurity_e2e_scenarios",
    "mcts_benchmark_scenarios",
]

def main():
    """Run all experiments on all datasets."""
    for dataset_name in DATASETS:
        for exp_name, exp_config in EXPERIMENTS.items():
            print(f"Running {exp_name} on {dataset_name}...")
            run_experiment(exp_config, dataset_name)
```

### Live Demo: Run Experiment (30 minutes)

**Instructor Demo:**

```python
# Create dataset
from scripts.create_langsmith_datasets import create_tactical_dataset
dataset_id = create_tactical_dataset()

# Define experiment
from scripts.run_langsmith_experiments import ExperimentConfig

experiment = ExperimentConfig(
    name="demo_experiment",
    description="Demo experiment for training",
    model="gpt-4o-mini",  # Faster, cheaper
    use_mcts=False,
    tags=["demo", "training"],
)

# Run experiment
from scripts.run_langsmith_experiments import run_experiment

results = run_experiment(
    experiment_config=experiment,
    dataset_name="tactical_e2e_scenarios"
)

# View results
print(f"Total runs: {results['total']}")
print(f"Success rate: {results['success_rate']:.2%}")
print(f"Avg latency: {results['avg_latency_ms']:.2f}ms")
print(f"Avg confidence: {results['avg_confidence']:.2f}")

# View in LangSmith UI
print(f"Filter: tags: experiment AND tags: demo")
```

### Hands-On Exercise: Run Complete Experiment (75 minutes)

**Exercise 3: Run and Analyze MCTS Experiment**

**Objective:** Run your MCTS iteration experiment from Exercise 2 and analyze results.

**Tasks:**

1. **Create dataset:**
```python
# Create MCTS benchmark dataset
def create_mcts_benchmark():
    """Create dataset for MCTS experiments."""
    examples = [
        {
            "inputs": {
                "scenario": "neutral_position",
                "action_choices": 5,
                "objective": "secure_position",
            },
            "outputs": {
                "min_win_probability": 0.65,
                "max_latency_ms": 5000,
            },
        },
        # Add 4+ more examples
    ]

    # TODO: Create dataset in LangSmith
    return dataset_id
```

2. **Run experiments:**
```python
# Run all MCTS configurations
configs = ["mcts_100", "mcts_200", "mcts_500"]

for config_name in configs:
    run_experiment(
        experiment_config=EXPERIMENTS[config_name],
        dataset_name="mcts_benchmark_scenarios"
    )
```

3. **Collect results:**
```python
# Query LangSmith for results
from langsmith import Client

client = Client()

results = {}
for config_name in configs:
    runs = list(client.list_runs(
        project_name="langgraph-multi-agent-mcts",
        filter=f'tags: experiment AND tags: {config_name}',
    ))

    # Aggregate metrics
    results[config_name] = aggregate_metrics(runs)
```

4. **Analyze and visualize:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame
df = pd.DataFrame(results).T

# Plot latency vs quality
plt.figure(figsize=(10, 6))
plt.scatter(df["avg_latency_ms"], df["avg_win_probability"])
for idx, row in df.iterrows():
    plt.annotate(idx, (row["avg_latency_ms"], row["avg_win_probability"]))
plt.xlabel("Average Latency (ms)")
plt.ylabel("Average Win Probability")
plt.title("MCTS: Quality vs Latency Tradeoff")
plt.grid(True)
plt.savefig("mcts_tradeoff.png")
```

5. **Write analysis report:**
```markdown
# MCTS Iteration Experiment Results

## Hypothesis
200 iterations provides optimal quality/latency tradeoff

## Results
| Config | Avg Win Prob | Avg Latency (ms) | Cost per Query |
|--------|-------------|------------------|----------------|
| mcts_100 | 0.72 | 1850 | $0.05 |
| mcts_200 | 0.78 | 2950 | $0.08 |
| mcts_500 | 0.81 | 6200 | $0.15 |

## Analysis
- **Quality improvement:** +8% from 100→200, +4% from 200→500
- **Latency impact:** +59% from 100→200, +110% from 200→500
- **Cost:** Scales linearly with iterations

## Conclusion
**Hypothesis validated:** 200 iterations provides best quality/latency tradeoff.
```

**Deliverable:** Complete experiment run + analysis report

---

## Session 4: Analysis and Insights (2 hours)

### Lecture: Statistical Analysis (45 minutes)

#### Comparing Experiment Results

**Statistical Significance:**
```python
from scipy import stats

def compare_experiments(results_a, results_b, metric="confidence"):
    """
    Compare two experiments using t-test.

    Returns:
        dict: Statistical test results
    """
    values_a = [r[metric] for r in results_a]
    values_b = [r[metric] for r in results_b]

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.var(values_a) + np.var(values_b)) / 2
    )
    cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
    }
```

**Confidence Intervals:**
```python
def calculate_confidence_interval(values, confidence=0.95):
    """Calculate confidence interval for metric."""
    mean = np.mean(values)
    stderr = stats.sem(values)
    interval = stderr * stats.t.ppf((1 + confidence) / 2, len(values) - 1)

    return {
        "mean": mean,
        "ci_lower": mean - interval,
        "ci_upper": mean + interval,
        "margin_of_error": interval,
    }
```

#### Regression Detection

**Automated Regression Checks:**
```python
def check_for_regressions(baseline_results, current_results, thresholds):
    """
    Check if current results show regression vs baseline.

    Args:
        baseline_results: Baseline experiment results
        current_results: Current experiment results
        thresholds: Acceptable regression thresholds

    Returns:
        list: List of detected regressions
    """
    regressions = []

    # Check latency
    if current_results["avg_latency_ms"] > baseline_results["avg_latency_ms"] * (1 + thresholds["latency"]):
        regressions.append({
            "metric": "latency",
            "baseline": baseline_results["avg_latency_ms"],
            "current": current_results["avg_latency_ms"],
            "change_pct": (current_results["avg_latency_ms"] / baseline_results["avg_latency_ms"] - 1) * 100,
        })

    # Check confidence
    if current_results["avg_confidence"] < baseline_results["avg_confidence"] * (1 - thresholds["confidence"]):
        regressions.append({
            "metric": "confidence",
            "baseline": baseline_results["avg_confidence"],
            "current": current_results["avg_confidence"],
            "change_pct": (current_results["avg_confidence"] / baseline_results["avg_confidence"] - 1) * 100,
        })

    return regressions
```

### Hands-On Exercise: Analysis Dashboard (60 minutes)

**Exercise 4: Build Experiment Comparison Dashboard**

**Objective:** Create comprehensive analysis of experiment results.

**Tasks:**

1. **Load experiment results from LangSmith**
2. **Calculate summary statistics**
3. **Perform statistical comparisons**
4. **Generate visualizations**
5. **Create recommendations**

**Deliverable:** Analysis notebook or report with visualizations

### Discussion: Best Practices (15 minutes)

**Key Topics:**
1. **Experiment versioning:** Track changes over time
2. **Cost management:** Optimize experiment runs
3. **Result interpretation:** Avoid common pitfalls
4. **Continuous experimentation:** Integrate into development workflow

---

## Module 5 Assessment

### Practical Assessment

**Task:** Design, run, and analyze a complete experiment

**Scenario:** Optimize prompt engineering for HRM agent

**Requirements:**
1. Create dataset with 10+ diverse queries
2. Design 3+ prompt variations
3. Run experiments with proper tracing
4. Collect and analyze metrics
5. Perform statistical comparison
6. Create visualizations
7. Write recommendations document

**Deliverable:**
- Dataset in LangSmith (20 points)
- Experiment configurations (20 points)
- Complete experiment runs (20 points)
- Statistical analysis (20 points)
- Visualizations and report (20 points)

**Total:** 100 points (passing: 70+)

**Submission:** LangSmith dataset URL + analysis report + visualizations

---

## Assessment Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Dataset Quality** | 20% | Diverse, realistic examples with proper validation |
| **Experiment Design** | 20% | Well-designed experiments with clear hypotheses |
| **Execution** | 20% | Complete runs with proper tracing and metrics |
| **Analysis** | 20% | Rigorous statistical analysis and insights |
| **Communication** | 20% | Clear visualizations and actionable recommendations |

**Minimum Passing:** 70% overall

---

## Additional Resources

### Reading
- [LANGSMITH_EXPERIMENTS.md](../LANGSMITH_EXPERIMENTS.md) - Complete guide
- [scripts/create_langsmith_datasets.py](../../scripts/create_langsmith_datasets.py) - Dataset creation
- [scripts/run_langsmith_experiments.py](../../scripts/run_langsmith_experiments.py) - Experiment runner

### Code Examples
- Dataset creation examples
- Experiment configuration examples
- Analysis notebooks (to be added)

### Office Hours
- When: [Schedule TBD]
- Topics: Experiment design, statistical analysis, optimization strategies

---

## Next Module

Continue to [MODULE_6_PYTHON_PRACTICES.md](MODULE_6_PYTHON_PRACTICES.md) - 2025 Python Coding & Testing Practices

**Prerequisites for Module 6:**
- Completed Module 5 practical assessment
- Basic Python proficiency
- Familiarity with testing concepts
