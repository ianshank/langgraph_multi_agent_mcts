# AlphaZero-Style Self-Play Training Pipeline

## Overview

This module implements a comprehensive AlphaZero-inspired self-play training pipeline for continuous system improvement in the LangGraph Multi-Agent MCTS framework. The pipeline enables autonomous learning through iterative self-play, episode generation, and training data extraction.

## Key Features

### 1. **Self-Play Episode Generation**
- Runs agents on diverse problem tasks
- Records complete execution traces
- Captures MCTS search trees at each decision point
- Logs decision points and outcomes
- Stores successful solution paths

### 2. **Comprehensive Task Generators**
- **Math Problems**: Arithmetic, algebra, quadratic equations, systems of equations
- **Code Generation**: Functions, algorithms, class designs across multiple languages
- **Multi-Step Reasoning**: Logical deduction, causal reasoning, planning
- **MCTS Search Tasks**: Game states, optimization problems, path finding

### 3. **Training Data Extraction**
Extracts four types of training examples:
- **(state, optimal_action)** pairs for policy learning
- **(state, value)** pairs for value estimation
- **(query, reasoning_chain)** for LLM fine-tuning
- **Negative examples** from failed attempts

### 4. **AlphaZero-Style Iteration Loop**
```
1. Generate N episodes with current model
2. Extract training examples from episodes
3. Train improved model on examples
4. Evaluate new model vs old model
5. Update if better, repeat
```

### 5. **Production Features**
- Async parallel episode generation
- Checkpointing and resumability
- Integration with HRM, TRM, and MCTS agents
- Resource monitoring (CPU/GPU usage)
- Quality metrics tracking

## Architecture

### Core Components

```
training/self_play_generator.py
├── Episode Data Structures
│   ├── Action
│   ├── State
│   ├── MCTSTrace
│   └── SelfPlayEpisode
│
├── Task Generators
│   ├── MathProblemGenerator
│   ├── CodeGenerationTaskGenerator
│   ├── MultiStepReasoningGenerator
│   └── MCTSSearchTaskGenerator
│
├── Episode Generation
│   └── SelfPlayEpisodeGenerator
│       ├── generate_episode()
│       ├── _run_mcts_simulation()
│       └── _execute_action()
│
├── Training Data Extraction
│   └── TrainingDataExtractor
│       ├── extract_examples()
│       ├── _extract_policy_examples()
│       ├── _extract_value_examples()
│       └── _extract_negative_examples()
│
└── Training Loop
    └── SelfPlayTrainer
        ├── iteration()
        ├── run_self_play()
        ├── train()
        └── evaluate_models()
```

## Usage

### Basic Episode Generation

```python
import asyncio
from training.self_play_generator import (
    SelfPlayEpisodeGenerator,
    MathProblemGenerator,
)

async def generate_episode():
    # Create generator
    generator = SelfPlayEpisodeGenerator(device="cpu")

    # Generate a task
    task_gen = MathProblemGenerator(seed=42)
    tasks = task_gen.generate(1)

    # Generate episode
    episode = await generator.generate_episode(
        task=tasks[0],
        max_steps=20,
        timeout=60.0
    )

    print(f"Outcome: {episode.outcome}")
    print(f"Reward: {episode.total_reward}")
    print(f"MCTS traces: {len(episode.mcts_traces)}")

asyncio.run(generate_episode())
```

### With Trained Agents

```python
from src.agents.hrm_agent import create_hrm_agent
from src.agents.trm_agent import create_trm_agent
from src.training.system_config import HRMConfig, TRMConfig
from training.self_play_generator import SelfPlayEpisodeGenerator

# Create agents
hrm_config = HRMConfig(h_dim=512, l_dim=256)
trm_config = TRMConfig(latent_dim=256)

hrm_agent = create_hrm_agent(hrm_config, device="cuda")
trm_agent = create_trm_agent(trm_config, device="cuda")

# Create generator with agents
generator = SelfPlayEpisodeGenerator(
    hrm_agent=hrm_agent,
    trm_agent=trm_agent,
    mcts_config={
        "num_simulations": 1600,
        "c_puct": 1.25,
    },
    device="cuda"
)
```

### Complete Training Iteration

```python
import asyncio
from training.self_play_generator import SelfPlayTrainer

async def train():
    config = {
        "games_per_iteration": 1000,
        "batch_size": 64,
        "parallel_batch_size": 32,
        "max_buffer_size": 10000,
        "mcts": {
            "num_simulations": 800,
            "c_puct": 1.25,
        }
    }

    trainer = SelfPlayTrainer(
        hrm_agent=hrm_agent,
        trm_agent=trm_agent,
        config=config,
        device="cuda",
        checkpoint_dir="./checkpoints/self_play"
    )

    # Run multiple iterations
    for i in range(100):
        metrics = await trainer.iteration(i)

        print(f"Iteration {i}:")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Best model: {metrics['best_model_metric']:.2%}")

asyncio.run(train())
```

### Training Data Extraction

```python
from training.self_play_generator import TrainingDataExtractor

extractor = TrainingDataExtractor(
    success_weight=1.0,
    failure_weight=0.3
)

# Extract from episodes
training_data = extractor.extract_examples(episodes)

# Access different example types
policy_examples = training_data["policy"]
value_examples = training_data["value"]
reasoning_examples = training_data["reasoning"]
negative_examples = training_data["negative"]

# Create PyTorch datasets
from training.self_play_generator import SelfPlayDataset
from torch.utils.data import DataLoader

policy_dataset = SelfPlayDataset(policy_examples, "policy")
policy_loader = DataLoader(policy_dataset, batch_size=64, shuffle=True)
```

## Data Structures

### SelfPlayEpisode

```python
@dataclass
class SelfPlayEpisode:
    task_id: str
    initial_state: Any
    actions: List[Action]
    states: List[State]
    rewards: List[float]
    mcts_traces: List[MCTSTrace]
    outcome: str  # "success", "failure", "timeout", "error"
    metadata: dict

    # Computed fields
    total_reward: float
    episode_length: int
    search_efficiency: float
    solution_path: List[str]
```

### MCTSTrace

Captures complete MCTS search information:

```python
@dataclass
class MCTSTrace:
    root_state_id: str
    num_simulations: int
    visit_counts: dict[str, int]      # action_id -> count
    q_values: dict[str, float]        # action_id -> Q-value
    prior_probs: dict[str, float]     # action_id -> prior
    selected_action: str
    tree_depth: int
    search_time: float
    value_estimates: dict[str, float]  # state_id -> value
```

### TrainingExample

```python
@dataclass
class TrainingExample:
    example_id: str
    example_type: str  # "policy", "value", "reasoning", "negative"
    state: torch.Tensor
    target: Any  # Type depends on example_type
    weight: float
    metadata: dict
```

## Task Generators

### MathProblemGenerator

Generates mathematical reasoning problems:
- Arithmetic: Basic operations
- Algebra: Linear equations
- Quadratic: Quadratic equations
- Systems: Systems of equations

```python
gen = MathProblemGenerator(
    difficulty_range=(0.1, 1.0),
    seed=42
)
tasks = gen.generate(100)
```

### CodeGenerationTaskGenerator

Generates programming tasks:
- Simple functions (sum_list, reverse_string)
- Algorithms (binary_search, merge_sort)
- Class designs (DataProcessor, etc.)

Languages: Python, JavaScript, Java

### MultiStepReasoningGenerator

Generates reasoning problems:
- Logical deduction
- Causal reasoning
- Planning tasks
- Constraint satisfaction

### MCTSSearchTaskGenerator

Generates search-friendly tasks:
- Game states (tic-tac-toe, connect-four)
- Optimization problems
- Path finding problems

## Quality Metrics

The trainer tracks comprehensive quality metrics:

```python
metrics = trainer.get_quality_metrics()

# Available metrics:
# - avg_success_rate: Average success rate across iterations
# - success_rate_std: Standard deviation of success rate
# - success_rate_trend: Improvement trend
# - avg_episode_length: Average episode length
# - total_episodes_generated: Total episodes generated
# - best_success_rate: Best achieved success rate
```

## Resource Monitoring

Monitor CPU, memory, and GPU usage:

```python
resources = trainer.get_resource_usage()

# Available metrics (if psutil installed):
# - cpu_percent: CPU usage percentage
# - memory_used_gb: Memory usage in GB
# - memory_percent: Memory usage percentage
# - gpu_memory_allocated_gb: GPU memory allocated
# - gpu_memory_reserved_gb: GPU memory reserved
```

## Checkpointing

### Save Checkpoint

```python
# Automatically saved during training
# Manual save:
trainer._save_checkpoint(iteration_num, best=True)
```

### Load Checkpoint

```python
trainer = SelfPlayTrainer(config=config, device="cuda")
trainer.load_checkpoint("./checkpoints/self_play/best_model.pt")

# Resume from iteration
current_iter = trainer.current_iteration
best_metric = trainer.best_model_metric
```

## Integration with Existing Agents

### With HRM Agent

```python
from src.agents.hrm_agent import create_hrm_agent
from src.training.system_config import HRMConfig

hrm_config = HRMConfig(
    h_dim=512,
    l_dim=256,
    num_h_layers=2,
    num_l_layers=4
)

hrm_agent = create_hrm_agent(hrm_config, device="cuda")

# Use in episode generator
generator = SelfPlayEpisodeGenerator(
    hrm_agent=hrm_agent,
    device="cuda"
)
```

### With TRM Agent

```python
from src.agents.trm_agent import create_trm_agent
from src.training.system_config import TRMConfig

trm_config = TRMConfig(
    latent_dim=256,
    num_recursions=16,
    deep_supervision=True
)

trm_agent = create_trm_agent(trm_config, device="cuda")

# Use for value estimation
generator = SelfPlayEpisodeGenerator(
    trm_agent=trm_agent,
    device="cuda"
)
```

### With Both Agents

```python
generator = SelfPlayEpisodeGenerator(
    hrm_agent=hrm_agent,  # For policy
    trm_agent=trm_agent,  # For value
    mcts_config={
        "num_simulations": 1600,
        "c_puct": 1.25,
    },
    device="cuda"
)
```

## Advanced Usage

### Custom Task Generator

```python
from training.self_play_generator import BaseTaskGenerator

class CustomTaskGenerator(BaseTaskGenerator):
    def generate(self, num_tasks: int) -> list[dict[str, Any]]:
        tasks = []
        for i in range(num_tasks):
            difficulty = self.rng.uniform(*self.difficulty_range)

            task = {
                "task_id": f"custom_{i}",
                "type": "custom",
                "difficulty": difficulty,
                "problem": "Your problem description",
                "answer": "Expected answer",
            }
            tasks.append(task)

        return tasks

    def _difficulty_to_params(self, difficulty: float) -> dict:
        return {"complexity": int(1 + difficulty * 10)}

# Use in trainer
trainer = SelfPlayTrainer(config=config)
trainer.task_generators["custom"] = CustomTaskGenerator()
```

### Parallel Episode Generation

```python
config = {
    "games_per_iteration": 10000,
    "parallel_batch_size": 128,  # Process 128 episodes in parallel
}

trainer = SelfPlayTrainer(config=config, device="cuda")

# Generates 10,000 episodes in batches of 128
episodes = await trainer.run_self_play(num_games=10000)
```

### Custom Training Loop

```python
async def custom_training_loop():
    trainer = SelfPlayTrainer(config=config)

    for iteration in range(100):
        # Generate episodes
        episodes = await trainer.run_self_play(num_games=1000)

        # Extract training data
        training_data = trainer.data_extractor.extract_examples(episodes)

        # Custom training logic
        # ... your training code here ...

        # Evaluate
        eval_metrics = await trainer.evaluate_models(num_eval_games=100)

        # Custom checkpoint logic
        if eval_metrics["eval_success_rate"] > threshold:
            trainer._save_checkpoint(iteration, best=True)
```

## Performance Optimization

### GPU Acceleration

```python
trainer = SelfPlayTrainer(
    hrm_agent=hrm_agent.cuda(),
    trm_agent=trm_agent.cuda(),
    config=config,
    device="cuda"
)
```

### Batch Size Tuning

```python
config = {
    "batch_size": 256,           # Training batch size
    "parallel_batch_size": 64,    # Parallel episode generation
}
```

### Episode Buffer

```python
config = {
    "max_buffer_size": 50000,  # Keep last 50k episodes
}

# Buffer automatically manages memory
trainer = SelfPlayTrainer(config=config)
```

## Troubleshooting

### Out of Memory

Reduce batch sizes:
```python
config = {
    "parallel_batch_size": 16,  # Reduce from 32
    "batch_size": 32,           # Reduce from 64
}
```

### Slow Episode Generation

Reduce MCTS simulations:
```python
config = {
    "mcts": {
        "num_simulations": 200,  # Reduce from 1600
    }
}
```

### Low Success Rate

1. Increase episode timeout
2. Adjust task difficulty range
3. Increase MCTS exploration
4. Check agent training

## Examples

See `training/examples/self_play_example.py` for comprehensive examples:

1. Basic episode generation
2. Using with HRM/TRM agents
3. Complete training iteration
4. Multi-iteration training
5. Task diversity
6. Data extraction details
7. Checkpoint management

Run examples:
```bash
python training/examples/self_play_example.py
```

## Testing

Run tests:
```bash
pytest training/tests/test_self_play_generator.py -v
```

## Dependencies

Required:
- torch
- numpy

Optional (for enhanced features):
- transformers (for tokenization)
- psutil (for resource monitoring)

## Future Enhancements

- [ ] Distributed episode generation across multiple GPUs
- [ ] Advanced curriculum learning
- [ ] Automatic task difficulty adjustment
- [ ] Integration with LangSmith for tracking
- [ ] Multi-modal task support (vision + language)
- [ ] Real-time training visualization
- [ ] Automated hyperparameter tuning

## References

- AlphaGo Zero: Mastering the game of Go without human knowledge
- AlphaZero: A general reinforcement learning algorithm
- MCTS: Monte Carlo Tree Search for decision making
- HRM: Hierarchical Reasoning Models
- TRM: Task Refinement Models

## License

Part of the LangGraph Multi-Agent MCTS Framework.
