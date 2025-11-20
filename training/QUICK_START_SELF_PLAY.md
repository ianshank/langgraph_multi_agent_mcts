# Self-Play Training Quick Start

## 5-Minute Quick Start

### 1. Generate Your First Episode

```python
import asyncio
from training.self_play_generator import (
    SelfPlayEpisodeGenerator,
    MathProblemGenerator
)

async def quick_start():
    # Create generators
    generator = SelfPlayEpisodeGenerator(device="cpu")
    task_gen = MathProblemGenerator(seed=42)

    # Generate task and episode
    task = task_gen.generate(1)[0]
    episode = await generator.generate_episode(task, max_steps=10)

    print(f"Outcome: {episode.outcome}")
    print(f"Steps: {episode.episode_length}")
    print(f"Reward: {episode.total_reward:.2f}")

asyncio.run(quick_start())
```

### 2. Extract Training Data

```python
from training.self_play_generator import TrainingDataExtractor

extractor = TrainingDataExtractor()
training_data = extractor.extract_examples([episode])

print(f"Policy examples: {len(training_data['policy'])}")
print(f"Value examples: {len(training_data['value'])}")
```

### 3. Run Complete Training Iteration

```python
from training.self_play_generator import SelfPlayTrainer

async def train_iteration():
    trainer = SelfPlayTrainer(
        config={"games_per_iteration": 100},
        device="cpu"
    )

    metrics = await trainer.iteration(0)
    print(f"Success rate: {metrics['success_rate']:.2%}")

asyncio.run(train_iteration())
```

## Common Commands

### Generate Episodes

```python
# Single episode
episode = await generator.generate_episode(task, max_steps=20)

# Multiple episodes in parallel
episodes = await trainer.run_self_play(num_games=1000)
```

### Extract Training Data

```python
# From episodes
training_data = extractor.extract_examples(episodes)

# Create PyTorch dataset
from training.self_play_generator import SelfPlayDataset
policy_dataset = SelfPlayDataset(training_data["policy"], "policy")
```

### Training Loop

```python
# Single iteration
metrics = await trainer.iteration(iteration_num)

# Multiple iterations
for i in range(10):
    metrics = await trainer.iteration(i)
    print(f"Iteration {i}: {metrics['success_rate']:.2%}")
```

### Checkpointing

```python
# Save
trainer._save_checkpoint(iteration, best=True)

# Load
trainer.load_checkpoint("./checkpoints/self_play/best_model.pt")
```

## Task Generators

```python
from training.self_play_generator import (
    MathProblemGenerator,
    CodeGenerationTaskGenerator,
    MultiStepReasoningGenerator,
    MCTSSearchTaskGenerator
)

# Math problems
math_gen = MathProblemGenerator(difficulty_range=(0.1, 1.0))
math_tasks = math_gen.generate(100)

# Code tasks
code_gen = CodeGenerationTaskGenerator()
code_tasks = code_gen.generate(50)

# Reasoning tasks
reasoning_gen = MultiStepReasoningGenerator()
reasoning_tasks = reasoning_gen.generate(30)

# MCTS search tasks
mcts_gen = MCTSSearchTaskGenerator()
mcts_tasks = mcts_gen.generate(20)
```

## Integration with Agents

```python
from src.agents.hrm_agent import create_hrm_agent
from src.agents.trm_agent import create_trm_agent
from src.training.system_config import HRMConfig, TRMConfig

# Create agents
hrm_agent = create_hrm_agent(HRMConfig(), device="cuda")
trm_agent = create_trm_agent(TRMConfig(), device="cuda")

# Use with self-play
trainer = SelfPlayTrainer(
    hrm_agent=hrm_agent,
    trm_agent=trm_agent,
    config={"games_per_iteration": 1000},
    device="cuda"
)
```

## Key Configuration Options

```python
config = {
    # Episode generation
    "games_per_iteration": 1000,      # Episodes per iteration
    "parallel_batch_size": 32,        # Parallel episodes

    # Training
    "batch_size": 64,                 # Training batch size
    "max_buffer_size": 10000,         # Episode buffer size

    # MCTS
    "mcts": {
        "num_simulations": 800,       # Simulations per decision
        "c_puct": 1.25,               # Exploration constant
    }
}
```

## Metrics and Monitoring

```python
# Quality metrics
quality = trainer.get_quality_metrics()
print(f"Avg success: {quality['avg_success_rate']:.2%}")
print(f"Best success: {quality['best_success_rate']:.2%}")

# Resource usage
resources = trainer.get_resource_usage()
print(f"CPU: {resources.get('cpu_percent', 0):.1f}%")
print(f"Memory: {resources.get('memory_used_gb', 0):.1f} GB")
```

## File Structure

```
training/
├── self_play_generator.py           # Main implementation (1,568 lines)
│   ├── 14 classes
│   ├── 9 async functions
│   └── 45 total functions
│
├── tests/
│   └── test_self_play_generator.py  # Comprehensive tests
│
├── examples/
│   └── self_play_example.py         # 7 usage examples
│
├── SELF_PLAY_README.md              # Full documentation
├── INTEGRATION_GUIDE.md             # Integration guide
└── QUICK_START_SELF_PLAY.md         # This file
```

## Production Checklist

- [ ] Configure task generators for your domain
- [ ] Tune MCTS simulations (start with 200, scale to 1600)
- [ ] Set appropriate batch sizes for your hardware
- [ ] Enable checkpointing
- [ ] Monitor resource usage
- [ ] Track success rate trends
- [ ] Integrate with existing training pipeline

## Common Patterns

### Pattern 1: Curriculum Learning

```python
# Start easy, increase difficulty
for iteration in range(100):
    difficulty = 0.1 + (iteration / 100) * 0.9

    gen = MathProblemGenerator(difficulty_range=(difficulty, difficulty + 0.1))
    tasks = gen.generate(1000)
    # ... train on tasks
```

### Pattern 2: Mixed Task Distribution

```python
# Combine different task types
all_tasks = (
    math_gen.generate(250) +
    code_gen.generate(250) +
    reasoning_gen.generate(250) +
    mcts_gen.generate(250)
)
```

### Pattern 3: Periodic Evaluation

```python
for iteration in range(100):
    # Train
    metrics = await trainer.iteration(iteration)

    # Evaluate every 10 iterations
    if iteration % 10 == 0:
        eval_metrics = await trainer.evaluate_models(num_eval_games=100)
        print(f"Eval: {eval_metrics['eval_success_rate']:.2%}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `parallel_batch_size` to 8-16 |
| Slow episodes | Reduce `num_simulations` to 100-200 |
| Low success rate | Decrease task difficulty or increase timeout |
| No improvement | Increase `c_puct` for more exploration |

## Next Steps

1. **Read**: `training/SELF_PLAY_README.md` for full documentation
2. **Try**: `training/examples/self_play_example.py` for examples
3. **Test**: `pytest training/tests/test_self_play_generator.py`
4. **Integrate**: Follow `training/INTEGRATION_GUIDE.md`

## Performance Tips

### CPU Optimization
```python
config = {
    "parallel_batch_size": 16,  # Lower for CPU
    "mcts": {"num_simulations": 200}
}
```

### GPU Optimization
```python
config = {
    "parallel_batch_size": 64,  # Higher for GPU
    "batch_size": 256,
    "mcts": {"num_simulations": 1600}
}
trainer = SelfPlayTrainer(config=config, device="cuda")
```

### Memory Optimization
```python
config = {
    "max_buffer_size": 5000,    # Smaller buffer
    "parallel_batch_size": 16,
}
```

## Resources

- Main implementation: `training/self_play_generator.py`
- Full docs: `training/SELF_PLAY_README.md`
- Integration: `training/INTEGRATION_GUIDE.md`
- Examples: `training/examples/self_play_example.py`
- Tests: `training/tests/test_self_play_generator.py`

---

**Need help?** Check the full documentation or run the examples!
