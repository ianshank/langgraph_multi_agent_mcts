# Training Datasets for LangGraph Multi-Agent MCTS Framework

Comprehensive guide to training datasets for the LangGraph Multi-Agent MCTS framework, covering HRM (Hierarchical Reasoning Model), TRM (Task Refinement Model), MCTS (Monte Carlo Tree Search), and quality engineering applications.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Summary](#dataset-summary)
3. [Core Reasoning Datasets](#core-reasoning-datasets)
4. [Mathematical Reasoning](#mathematical-reasoning)
5. [Code & Quality Engineering](#code--quality-engineering)
6. [Strategic Planning](#strategic-planning)
7. [Installation & Setup](#installation--setup)
8. [Usage Examples](#usage-examples)
9. [Training Pipeline](#training-pipeline)
10. [Configuration](#configuration)
11. [Best Practices](#best-practices)

---

## Overview

The framework supports 8 specialized datasets designed for training different agent types and capabilities:

| Dataset | Purpose | Agent Type | Size | License |
|---------|---------|------------|------|---------|
| **ARC** | Abstract reasoning & pattern recognition | HRM | 1,000 train + 400 eval | Apache 2.0 |
| **GSM8K** | Mathematical multi-step reasoning | TRM | 7,500 train + 1,000 test | MIT |
| **IDoFT** | Flaky test detection & analysis | HRM/TRM | 2,000+ tests | Research |
| **HumanEval** | Code generation evaluation | TRM | 164 problems | MIT |
| **Chess Games** | Strategic planning & MCTS policies | MCTS | 14M games | CC-BY-4.0 |
| **BIG-Bench Hard** | Complex reasoning evaluation | HRM | 23 tasks | Apache 2.0 |
| **DABStep** | Multi-step data analysis | HRM/TRM | 450+ tasks | CC-BY-4.0 |
| **PRIMUS** | Cybersecurity domain knowledge | All | 674K docs | ODC-BY |

---

## Dataset Summary

### Quick Reference: Dataset-to-Agent Mapping

```
HRM (Hierarchical Reasoning Model):
├── ARC (abstract pattern recognition)
├── DABStep (multi-step decomposition)
├── GSM8K (mathematical reasoning)
└── BIG-Bench Hard (complex reasoning)

TRM (Task Refinement Model):
├── GSM8K (iterative solution refinement)
├── DABStep (iterative problem-solving)
├── HumanEval (code refinement)
└── PRIMUS-Instruct (instruction tuning)

MCTS (Monte Carlo Tree Search):
├── Chess Games (policy & value training)
└── Strategic planning tasks

Quality Engineering:
├── IDoFT (flaky test analysis)
└── HumanEval (test generation)
```

---

## Core Reasoning Datasets

### 1. ARC (Abstraction and Reasoning Corpus)

**Purpose**: Train HRM agents on hierarchical pattern recognition and abstract reasoning.

**Key Statistics**:
- Training samples: ~1,000
- Evaluation tasks: 400
- Format: Visual grid pattern puzzles
- Benchmark: HRM achieved 40.3% (vs GPT-4: 21.2%)

**Why It's Ideal**:
- Requires hierarchical decomposition of visual patterns
- Small dataset demonstrates data efficiency (critical for specialized domains)
- Directly measures AGI capabilities
- Perfect for validating HRM architecture

**Usage**:
```python
from src.data.dataset_loader import ARCLoader

# Load ARC dataset
arc_loader = ARCLoader(cache_dir="./cache/arc")
arc_samples = arc_loader.load(split="train")

# Get statistics
stats = arc_loader.get_statistics()
print(f"Loaded {stats.total_samples} ARC samples")

# Access task data
sample = arc_samples[0]
task_data = sample.metadata["task_data"]
input_grids = task_data["input_grids"]
output_grids = task_data["output_grids"]
```

**Training Recommendation**:
- Train HRM for 150 epochs on 1,000 samples
- Use convergence depth of 100 for hierarchical decomposition
- Validate on held-out evaluation set (400 tasks)

**Source**: https://github.com/fchollet/ARC

---

### 2. BIG-Bench Hard (BBH)

**Purpose**: Evaluate complex reasoning capabilities including causal reasoning, counterfactual thinking, and multi-hop inference.

**Key Statistics**:
- Tasks: 23 challenging reasoning categories
- Samples: Varies by task
- Difficulty: Hard (designed to challenge frontier models)

**Task Categories**:
```
Causal & Logical:
- causal_judgement
- logical_deduction
- web_of_lies

Temporal & Spatial:
- date_understanding
- temporal_sequences
- navigate

Linguistic:
- disambiguation_qa
- hyperbaton
- snarks

Quantitative:
- multistep_arithmetic
- object_counting
- penguins_in_a_table
```

**Usage**:
```python
from src.data.dataset_loader import BIGBenchHardLoader

# Load all tasks
bbh_loader = BIGBenchHardLoader(cache_dir="./cache/bigbench_hard")
bbh_samples = bbh_loader.load(split="train")

# Filter specific reasoning tasks
causal_samples = bbh_loader.get_by_task("causal_judgement")
arithmetic_samples = bbh_loader.get_by_task("multistep_arithmetic")

# Load only specific tasks
bbh_samples = bbh_loader.load(
    split="train",
    tasks=["causal_judgement", "logical_deduction", "multistep_arithmetic"]
)
```

**Training Recommendation**:
- Use as evaluation benchmark for HRM hierarchical reasoning
- Test consensus mechanism across multiple reasoning paths
- Compare performance against human baseline

**Source**: https://github.com/suzgunmirac/BIG-Bench-Hard

---

## Mathematical Reasoning

### 3. GSM8K (Grade School Math 8K)

**Purpose**: Train TRM iterative refinement on multi-step mathematical reasoning.

**Key Statistics**:
- Training: 7,500 problems
- Test: 1,000 problems
- Average steps per problem: 3-5
- Difficulty: Elementary school level

**Dataset Features**:
- Natural language word problems
- Step-by-step solutions included
- Diverse problem types (arithmetic, algebra, word problems)
- High-quality human annotations

**Usage**:
```python
from src.data.dataset_loader import GSM8KLoader

# Load training set
gsm8k_loader = GSM8KLoader(cache_dir="./cache/gsm8k")
gsm8k_samples = gsm8k_loader.load(split="train", config="main")

# Get samples with multi-step reasoning
reasoning_samples = gsm8k_loader.get_reasoning_samples()

# Access reasoning steps
sample = reasoning_samples[0]
print(f"Question: {sample.text}")
print(f"Steps: {len(sample.reasoning_steps)}")
for i, step in enumerate(sample.reasoning_steps, 1):
    print(f"  Step {i}: {step}")

# Access ground truth answer
answer = sample.metadata["answer"]
```

**Training Recommendation**:
- Train TRM on iterative solution refinement
- Use initial question as starting point
- Compare refined solutions with ground truth
- Validate convergence after 3-5 refinement iterations

**Evaluation**:
```python
# Use GSM8K test set for validation
test_samples = gsm8k_loader.load(split="test")

# Measure:
# 1. Final answer accuracy
# 2. Reasoning step quality
# 3. Convergence speed
```

**Source**: https://github.com/openai/grade-school-math

---

## Code & Quality Engineering

### 4. IDoFT (Illinois Dataset of Flaky Tests)

**Purpose**: Train quality engineering agents to detect, classify, and fix flaky tests.

**Key Statistics**:
- Flaky tests: 2,000+
- Projects: Multiple real-world Java projects
- Root cause categories: 8 types
- Includes: Failure rates, fix commits

**Flaky Test Categories**:
```python
CATEGORIES = [
    "async_wait",          # Timing/synchronization issues
    "concurrency",         # Race conditions
    "time_dependent",      # Tests relying on system time
    "unordered_collection", # Assumption about order
    "test_order_dependency", # Inter-test dependencies
    "resource_leak",       # Resource cleanup issues
    "network_dependent",   # Network-related flakiness
    "io_dependent",        # File I/O issues
]
```

**Usage**:
```python
from src.data.dataset_loader import IDoFTLoader

# Option 1: Load from HuggingFace (if available)
idoft_loader = IDoFTLoader(cache_dir="./cache/idoft")
try:
    idoft_samples = idoft_loader.load(split="train")
except FileNotFoundError:
    print("IDoFT requires manual download")

# Option 2: Load from local clone
idoft_loader = IDoFTLoader(
    cache_dir="./cache/idoft",
    data_path="/path/to/idoft"  # Local clone
)
idoft_samples = idoft_loader.load()

# Filter by category
async_tests = idoft_loader.get_by_category("async_wait")
concurrency_tests = idoft_loader.get_by_category("concurrency")

# Access flaky test metadata
sample = idoft_samples[0]
print(f"Category: {sample.labels}")
print(f"Project: {sample.metadata['project']}")
print(f"Test: {sample.metadata['test_name']}")
print(f"Failure rate: {sample.metadata.get('failure_rate', 'N/A')}")
```

**Training Recommendation**:
```
Phase 1: Classification (HRM)
- Train HRM to decompose flaky test into root cause categories
- Use hierarchical decision tree for classification
- Achieve category-level understanding

Phase 2: Root Cause Analysis (HRM + TRM)
- HRM: Decompose test code into suspicious components
- TRM: Iteratively refine root cause hypothesis

Phase 3: Fix Generation (TRM + MCTS)
- MCTS: Explore different fix strategies
- TRM: Refine fix implementation
- Validate using test stability metrics
```

**Installation**:
```bash
# Clone IDoFT repository
git clone https://github.com/TestingResearchIllinois/idoft

# Or download from research website
# http://mir.cs.illinois.edu/flakytests/
```

**Source**: http://mir.cs.illinois.edu/flakytests/

---

### 5. HumanEval

**Purpose**: Evaluate code generation and test synthesis capabilities.

**Key Statistics**:
- Problems: 164 hand-crafted
- Language: Python
- Includes: Function signatures, docstrings, unit tests
- Manually designed to avoid training data contamination

**Usage**:
```python
from src.data.dataset_loader import HumanEvalLoader

# Load HumanEval
humaneval_loader = HumanEvalLoader(cache_dir="./cache/humaneval")
humaneval_samples = humaneval_loader.load(split="test")

# Access problem details
sample = humaneval_samples[0]
task_id = sample.metadata["task_id"]
prompt = sample.metadata["prompt"]
canonical_solution = sample.metadata["canonical_solution"]
tests = sample.metadata["test"]
entry_point = sample.metadata["entry_point"]

print(f"Task: {task_id}")
print(f"Function: {entry_point}")
print(f"Prompt:\n{prompt}")
print(f"Tests:\n{tests}")
```

**Training Recommendation**:
```
HRM Application:
- Decompose function requirements into testable units
- Identify edge cases and corner cases
- Generate test case hierarchy

TRM Application:
- Iteratively refine code implementation
- Validate against unit tests after each refinement
- Converge on passing implementation

Combined Approach:
1. HRM: Decompose into subtasks
2. TRM: Implement and refine each subtask
3. Integration: Combine refined subtasks
4. Validation: Run full unit test suite
```

**Evaluation Metrics**:
- Pass@k: Percentage of problems solved in k attempts
- Code quality: Readability, efficiency
- Test coverage: Comprehensive test generation

**Source**: https://github.com/openai/human-eval

---

## Strategic Planning

### 6. Chess Games

**Purpose**: Train MCTS policies and value networks for strategic decision-making.

**Key Statistics**:
- Games: 14 million
- Player ELO: Mean 2388 (high-level players)
- Time range: 1600-2024
- Format: UCI and SAN notation

**Usage**:
```python
from src.data.dataset_loader import ChessGamesLoader

# Load high-level games
chess_loader = ChessGamesLoader(cache_dir="./cache/chess_games")
chess_samples = chess_loader.load(
    split="train",
    min_elo=2200,
    max_samples=10000,
    streaming=True  # Required for large dataset
)

# Filter elite games
elite_games = chess_loader.get_high_level_games(min_elo=2500)

# Access game data
sample = chess_samples[0]
white_elo = sample.metadata["white_elo"]
black_elo = sample.metadata["black_elo"]
moves_uci = sample.metadata["moves_uci"]
moves_san = sample.metadata["moves_san"]
result = sample.metadata["end_type"]

print(f"ELO: {white_elo} vs {black_elo}")
print(f"Moves: {moves_san}")
print(f"Result: {result}")
```

**Training Recommendation**:
```
MCTS Policy Training:
1. Parse moves into board positions
2. Extract policy targets (move probabilities)
3. Train neural network to predict move distribution

MCTS Value Training:
1. Extract position evaluations from game outcomes
2. Train value network to estimate win probability
3. Use Stockfish evaluations for additional supervision

Integration:
- Combine policy and value networks in MCTS
- Use trained models for tree search
- Validate with self-play
```

**ELO Filtering Strategy**:
```python
# Beginners (learning basic patterns)
beginner_games = chess_loader.load(min_elo=1000, max_elo=1500)

# Intermediate (tactical patterns)
intermediate_games = chess_loader.load(min_elo=1500, max_elo=2000)

# Advanced (strategic planning)
advanced_games = chess_loader.load(min_elo=2000, max_elo=2400)

# Elite (master-level play)
elite_games = chess_loader.load(min_elo=2400)
```

**Source**: https://huggingface.co/datasets/angeluriot/chess_games

---

## Installation & Setup

### Prerequisites

```bash
# Install required packages
pip install datasets  # HuggingFace datasets library
pip install torch     # For training
pip install transformers  # For model architectures
```

### Dataset Installation

#### Option 1: Automatic Download (HuggingFace)

Most datasets automatically download on first use:

```python
from src.data.dataset_loader import CombinedDatasetLoader

# Initialize loader
loader = CombinedDatasetLoader(cache_dir="./cache")

# Datasets will auto-download to cache_dir
samples = loader.load_all(
    include_arc=True,
    include_gsm8k=True,
    include_humaneval=True,
    include_chess=True,
    include_bbh=True,
)
```

#### Option 2: Manual Installation (IDoFT)

IDoFT requires manual download:

```bash
# Clone repository
git clone https://github.com/TestingResearchIllinois/idoft

# Use local path
loader = CombinedDatasetLoader(
    cache_dir="./cache",
    idoft_data_path="./idoft"
)
```

### Configuration Setup

Edit `training/config.yaml` to enable datasets:

```yaml
data:
  # Enable ARC for HRM training
  arc:
    enabled: true
    split: "train"
    max_samples: 1000

  # Enable GSM8K for TRM training
  gsm8k:
    enabled: true
    split: "train"
    max_samples: null  # Use all 7,500

  # Enable Chess for MCTS training
  chess_games:
    enabled: true
    min_elo: 2200
    max_samples: 10000
```

---

## Usage Examples

### Example 1: Train HRM on ARC

```python
from src.data.dataset_loader import ARCLoader

# Load ARC dataset
arc_loader = ARCLoader(cache_dir="./cache/arc")
arc_samples = arc_loader.load(split="train")

# Training loop (pseudocode)
for sample in arc_samples:
    task_data = sample.metadata["task_data"]

    # Train HRM to decompose pattern recognition
    hrm_output = hrm_agent.decompose(
        input_grids=task_data["input_grids"],
        output_grids=task_data["output_grids"],
        max_depth=5
    )

    # Validate decomposition quality
    loss = evaluate_decomposition(hrm_output, task_data)
    optimizer.step(loss)
```

### Example 2: Train TRM on GSM8K

```python
from src.data.dataset_loader import GSM8KLoader

# Load GSM8K
gsm8k_loader = GSM8KLoader(cache_dir="./cache/gsm8k")
gsm8k_samples = gsm8k_loader.load(split="train")

# Get reasoning samples
reasoning_samples = gsm8k_loader.get_reasoning_samples()

# Training loop
for sample in reasoning_samples:
    question = sample.text
    ground_truth_steps = sample.reasoning_steps

    # Train TRM iterative refinement
    solution = question
    for iteration in range(max_iterations):
        solution = trm_agent.refine(solution)

        # Check convergence
        if converged(solution, ground_truth_steps):
            break

    # Calculate loss
    loss = evaluate_solution(solution, ground_truth_steps)
    optimizer.step(loss)
```

### Example 3: Train MCTS on Chess

```python
from src.data.dataset_loader import ChessGamesLoader

# Load chess games
chess_loader = ChessGamesLoader(cache_dir="./cache/chess_games")
chess_samples = chess_loader.load(
    min_elo=2400,
    max_samples=100000,
    streaming=True
)

# Training loop
for game in chess_samples:
    moves = game.metadata["moves_uci"].split()

    # Extract positions and targets
    for move_idx, move in enumerate(moves):
        position = get_position_at_move(moves[:move_idx])
        policy_target = move
        value_target = get_game_outcome(game)

        # Train MCTS networks
        policy_loss = mcts.train_policy(position, policy_target)
        value_loss = mcts.train_value(position, value_target)

        total_loss = policy_loss + value_loss
        optimizer.step(total_loss)
```

### Example 4: Combined Dataset Loading

```python
from src.data.dataset_loader import CombinedDatasetLoader

# Initialize combined loader
loader = CombinedDatasetLoader(
    cache_dir="./cache",
    idoft_data_path="./idoft"
)

# Load all datasets
samples = loader.load_all(
    # Original datasets
    dabstep_split="train",
    primus_max_samples=1000,
    include_instruct=True,

    # New datasets
    include_arc=True,
    include_gsm8k=True,
    include_idoft=True,
    include_humaneval=True,
    include_chess=True,
    chess_max_samples=10000,
    include_bbh=True,
)

# Get agent-specific samples
hrm_samples = loader.get_hrm_training_samples()
trm_samples = loader.get_trm_training_samples()
mcts_samples = loader.get_mcts_training_samples()
code_samples = loader.get_code_generation_samples()

# Get summary
summary = loader.get_dataset_summary()
print(f"Total samples: {summary['total_samples']}")
print(f"HRM samples: {summary['hrm_training_samples']}")
print(f"TRM samples: {summary['trm_training_samples']}")
print(f"MCTS samples: {summary['mcts_training_samples']}")

# Export for training
loader.export_for_training("./data/combined.jsonl")
```

---

## Training Pipeline

### Recommended Training Sequence

#### Phase 1: Core Reasoning (Weeks 1-4)

```python
# Week 1-2: HRM on ARC
arc_loader = ARCLoader()
arc_samples = arc_loader.load(split="train")
train_hrm(arc_samples, epochs=150, model_size="27M")

# Week 2-3: TRM on GSM8K
gsm8k_loader = GSM8KLoader()
gsm8k_samples = gsm8k_loader.load(split="train")
train_trm(gsm8k_samples, epochs=10, max_iterations=5)

# Week 4: Validation
validate_on_arc_eval(hrm_agent)
validate_on_gsm8k_test(trm_agent)
```

#### Phase 2: Code Understanding (Weeks 5-8)

```python
# Week 5-6: HumanEval fine-tuning
humaneval_loader = HumanEvalLoader()
humaneval_samples = humaneval_loader.load()
finetune_on_code(hrm_agent, trm_agent, humaneval_samples)

# Week 7-8: IDoFT quality engineering
idoft_loader = IDoFTLoader(data_path="./idoft")
idoft_samples = idoft_loader.load()
train_quality_engineering(idoft_samples)
```

#### Phase 3: Strategic Planning (Weeks 9-12)

```python
# Week 9-11: MCTS on Chess
chess_loader = ChessGamesLoader()
chess_samples = chess_loader.load(
    min_elo=2200,
    max_samples=100000
)
train_mcts_policies(chess_samples, iterations=1000000)

# Week 12: Integration testing
test_multi_agent_coordination()
```

#### Phase 4: Evaluation & Deployment (Weeks 13-16)

```python
# Week 13-14: BIG-Bench Hard evaluation
bbh_loader = BIGBenchHardLoader()
bbh_samples = bbh_loader.load()
evaluate_reasoning(hrm_agent, bbh_samples)

# Week 15: Production validation
validate_on_internal_datasets()

# Week 16: Deployment
deploy_to_production()
```

---

## Configuration

### Full Configuration Example

```yaml
# training/config.yaml

data:
  # ARC: HRM hierarchical reasoning
  arc:
    path: "barc0/abstraction_and_reasoning_corpus"
    cache_dir: "./cache/arc"
    enabled: true
    split: "train"
    max_samples: 1000
    trust_remote_code: true

  # GSM8K: TRM mathematical reasoning
  gsm8k:
    path: "openai/gsm8k"
    cache_dir: "./cache/gsm8k"
    enabled: true
    config: "main"
    split: "train"
    max_samples: null

  # IDoFT: Quality engineering
  idoft:
    enabled: true
    data_path: "./idoft"
    cache_dir: "./cache/idoft"
    categories:
      - "async_wait"
      - "concurrency"
      - "time_dependent"

  # HumanEval: Code generation
  humaneval:
    path: "openai/openai_humaneval"
    cache_dir: "./cache/humaneval"
    enabled: true
    split: "test"

  # Chess: MCTS strategic planning
  chess_games:
    path: "angeluriot/chess_games"
    cache_dir: "./cache/chess_games"
    enabled: true
    min_elo: 2200
    max_samples: 100000
    streaming: true

  # BIG-Bench Hard: Reasoning evaluation
  bigbench_hard:
    path: "maveriq/bigbenchhard"
    cache_dir: "./cache/bigbench_hard"
    enabled: true
    split: "train"
    tasks: null

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 10

  # Agent-specific settings
  agents:
    hrm:
      model_name: "microsoft/deberta-v3-base"
      max_decomposition_depth: 5
      learning_rate: 2e-5

    trm:
      model_name: "microsoft/deberta-v3-base"
      max_refinement_iterations: 5
      convergence_threshold: 0.95

    mcts:
      simulations: 200
      exploration_constant: 1.414
```

---

## Best Practices

### Data Efficiency

```python
# Start with small, high-quality datasets
arc_samples = arc_loader.load(split="train")  # Only 1,000 samples
# HRM achieved 40.3% with just 1,000 examples

# Scale to larger datasets after validation
gsm8k_samples = gsm8k_loader.load()  # 7,500 samples
```

### Dataset Stratification

```python
from src.data.train_test_split import StratifiedSplitter

# Stratify by difficulty
splitter = StratifiedSplitter()
train, val, test = splitter.split_by_difficulty(
    samples=gsm8k_samples,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)

# Stratify by domain
train, val, test = splitter.split_by_domain(
    samples=combined_samples,
    train_ratio=0.8
)
```

### Data Augmentation

```python
from src.data.tactical_augmentation import TacticalAugmenter

# Augment training data
augmenter = TacticalAugmenter()
augmented = augmenter.augment(
    samples=gsm8k_samples,
    num_variations=3,
    paraphrase_probability=0.3
)
```

### Streaming for Large Datasets

```python
# Use streaming for Chess (14M games)
chess_samples = chess_loader.load(
    streaming=True,  # Don't load all into memory
    max_samples=100000  # Limit samples
)

# Process in batches
for batch in chess_loader.iterate_samples(batch_size=32):
    train_mcts(batch)
```

### Evaluation Strategy

```python
# Hold out test sets
arc_train = arc_loader.load(split="train")  # 1,000
arc_eval = arc_loader.load(split="evaluation")  # 400

gsm8k_train = gsm8k_loader.load(split="train")  # 7,500
gsm8k_test = gsm8k_loader.load(split="test")  # 1,000

# Never train on test sets
# Use for final evaluation only
```

### Monitoring & Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Track dataset statistics
stats = loader.get_dataset_summary()
logger.info(f"Dataset summary: {stats}")

# Monitor training progress
for epoch in range(epochs):
    train_loss = train_epoch(hrm_agent, arc_samples)
    val_loss = validate_epoch(hrm_agent, arc_eval)

    logger.info(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")
```

### Production Deployment

```python
# Export trained models
torch.save(hrm_agent.state_dict(), "models/hrm_arc_trained.pt")
torch.save(trm_agent.state_dict(), "models/trm_gsm8k_trained.pt")

# Version control datasets
loader.export_for_training(
    output_path="./data/v1.0/combined_training.jsonl",
    format="jsonl"
)

# Document dataset versions
with open("./data/v1.0/manifest.json", "w") as f:
    json.dump({
        "version": "1.0",
        "datasets": {
            "arc": {"split": "train", "samples": 1000},
            "gsm8k": {"split": "train", "samples": 7500},
            "chess": {"min_elo": 2200, "samples": 100000},
        },
        "date": "2025-01-21"
    }, f)
```

---

## License Attribution

When using these datasets, ensure proper attribution:

```python
# In your research paper or documentation
"""
Training Data Attribution:
- ARC: Apache 2.0 License (Chollet, 2019)
- GSM8K: MIT License (Cobbe et al., 2021)
- HumanEval: MIT License (Chen et al., 2021)
- Chess Games: CC-BY-4.0 (Uriot, 2024)
- BIG-Bench Hard: Apache 2.0 (Suzgun et al., 2022)
- IDoFT: Research/Academic Use (Lam et al., 2019)
- DABStep: CC-BY-4.0 (Adyen, 2024)
- PRIMUS: ODC-BY (Trend Micro, 2024)
"""
```

---

## Troubleshooting

### Common Issues

#### Dataset Not Found
```python
# Error: Dataset not found on HuggingFace
# Solution: Check dataset name and availability
try:
    samples = loader.load()
except Exception as e:
    logger.error(f"Failed to load: {e}")
    # Try alternative dataset source or manual download
```

#### Out of Memory
```python
# Error: OOM when loading large datasets
# Solution: Use streaming mode
chess_samples = chess_loader.load(
    streaming=True,  # Enable streaming
    max_samples=10000  # Limit samples
)
```

#### Slow Downloads
```python
# Issue: HuggingFace downloads are slow
# Solution: Use local cache
loader = CombinedDatasetLoader(
    cache_dir="/fast/ssd/cache"  # Fast storage
)
```

---

## Additional Resources

- **Examples**: See `examples/dataset_training_examples.py`
- **Configuration**: See `training/config.yaml`
- **Loaders**: See `src/data/dataset_loader.py`
- **Documentation**: See official dataset repositories

---

## Support

For issues or questions:
1. Check GitHub issues: https://github.com/ianshank/langgraph_multi_agent_mcts/issues
2. Review dataset documentation links above
3. Consult training examples in `examples/` directory

---

**Last Updated**: 2025-01-21
**Framework Version**: 1.0.0
**Maintained by**: LangGraph Multi-Agent MCTS Team
