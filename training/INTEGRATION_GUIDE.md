# Self-Play Training Integration Guide

## Overview

This guide demonstrates how to integrate the AlphaZero-style self-play training pipeline with the existing agent training infrastructure.

## Integration Points

### 1. Agent Trainer Integration

The self-play pipeline integrates seamlessly with `training/agent_trainer.py`:

```python
from training.agent_trainer import HRMTrainer, TRMTrainer, AgentTrainingOrchestrator
from training.self_play_generator import SelfPlayTrainer, TrainingDataExtractor

# Initialize existing trainers
orchestrator = AgentTrainingOrchestrator("training/config.yaml")
orchestrator.initialize_trainers()

# Get trained agents
hrm_agent = orchestrator.hrm_trainer.model
trm_agent = orchestrator.trm_trainer.model

# Create self-play trainer with existing agents
self_play_trainer = SelfPlayTrainer(
    hrm_agent=hrm_agent,
    trm_agent=trm_agent,
    config={
        "games_per_iteration": 1000,
        "batch_size": 64,
    },
    device="cuda"
)
```

### 2. Combined Training Loop

Alternate between supervised training and self-play:

```python
import asyncio

async def combined_training_loop():
    # Phase 1: Supervised pre-training
    print("Phase 1: Supervised Pre-training")
    orchestrator.train_phase(
        phase_name="supervised",
        hrm_dataloader=supervised_hrm_data,
        trm_dataloader=supervised_trm_data,
    )

    # Phase 2: Self-play fine-tuning
    print("Phase 2: Self-Play Fine-tuning")
    for iteration in range(50):
        # Generate self-play data
        metrics = await self_play_trainer.iteration(iteration)

        print(f"Iteration {iteration}:")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        print(f"  Best model: {metrics['best_model_metric']:.2%}")

        # Optional: Supervised fine-tuning every N iterations
        if iteration % 10 == 0:
            orchestrator.train_phase(
                phase_name=f"supervised_refinement_{iteration}",
                hrm_dataloader=supervised_hrm_data,
                trm_dataloader=supervised_trm_data,
            )

asyncio.run(combined_training_loop())
```

### 3. Data Pipeline Integration

Use self-play data alongside existing datasets:

```python
from training.data_pipeline import DABStepLoader, HRMSample, TRMSample
from training.self_play_generator import SelfPlayDataset
from torch.utils.data import ConcatDataset

# Load existing datasets
dabstep_loader = DABStepLoader(config)
supervised_data = dabstep_loader.load()

# Generate self-play data
episodes = await self_play_trainer.run_self_play(num_games=1000)
extractor = TrainingDataExtractor()
training_data = extractor.extract_examples(episodes)

# Combine datasets
policy_dataset_selfplay = SelfPlayDataset(training_data["policy"], "policy")
combined_dataset = ConcatDataset([supervised_data, policy_dataset_selfplay])
```

### 4. Evaluation Integration

Use existing evaluation framework:

```python
from training.evaluation import EvaluationFramework

# Create evaluation framework
evaluator = EvaluationFramework(config)

# Evaluate self-play trained models
eval_results = evaluator.evaluate_all_agents(
    hrm_agent=self_play_trainer.hrm_agent,
    trm_agent=self_play_trainer.trm_agent,
)

print("Evaluation Results:")
print(f"  HRM Accuracy: {eval_results['hrm']['accuracy']:.2%}")
print(f"  TRM Convergence Rate: {eval_results['trm']['convergence_rate']:.2%}")
```

### 5. Monitoring Integration

Integrate with existing monitoring:

```python
from training.monitoring import SystemMonitor

monitor = SystemMonitor(config)

# Track self-play metrics
for iteration in range(100):
    metrics = await self_play_trainer.iteration(iteration)

    # Log to monitoring system
    monitor.log_metrics({
        "self_play/success_rate": metrics["success_rate"],
        "self_play/eval_success_rate": metrics["eval_success_rate"],
        "self_play/best_model_metric": metrics["best_model_metric"],
        "iteration": iteration,
    })

    # Track resources
    resources = self_play_trainer.get_resource_usage()
    monitor.log_metrics(resources)
```

## Complete Integration Example

### Full Training Pipeline

```python
"""
Complete integration example combining all training components.
"""

import asyncio
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

from training.agent_trainer import AgentTrainingOrchestrator
from training.continual_learning import ContinualLearningManager
from training.data_pipeline import DABStepLoader
from training.evaluation import EvaluationFramework
from training.monitoring import SystemMonitor
from training.self_play_generator import SelfPlayTrainer, TrainingDataExtractor


class IntegratedTrainingPipeline:
    """Integrated training pipeline with self-play."""

    def __init__(self, config_path: str = "training/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.orchestrator = AgentTrainingOrchestrator(config_path)
        self.orchestrator.initialize_trainers()

        self.monitor = SystemMonitor(self.config)
        self.evaluator = EvaluationFramework(self.config)

        # Initialize self-play trainer
        self.self_play_trainer = None

        # Data loaders
        self.supervised_loaders = {}

    def setup_supervised_data(self):
        """Setup supervised learning datasets."""
        dabstep_loader = DABStepLoader(self.config["data"]["dabstep"])
        dabstep_data = dabstep_loader.load()

        # Create data loaders
        self.supervised_loaders["hrm"] = DataLoader(
            dabstep_data["train"],
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )

        self.supervised_loaders["trm"] = DataLoader(
            dabstep_data["train"],
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )

    def setup_self_play(self):
        """Setup self-play trainer."""
        self.self_play_trainer = SelfPlayTrainer(
            hrm_agent=self.orchestrator.hrm_trainer.model,
            trm_agent=self.orchestrator.trm_trainer.model,
            config={
                "games_per_iteration": 1000,
                "batch_size": 64,
                "parallel_batch_size": 32,
                "max_buffer_size": 10000,
                "mcts": {
                    "num_simulations": 800,
                    "c_puct": 1.25,
                },
            },
            device=self.orchestrator.device,
            checkpoint_dir="./checkpoints/integrated_self_play",
        )

    async def run_training(self, num_iterations: int = 100):
        """Run complete training pipeline."""

        print("=" * 60)
        print("INTEGRATED TRAINING PIPELINE")
        print("=" * 60)

        # Phase 1: Supervised Pre-training
        print("\n[Phase 1] Supervised Pre-training")
        print("-" * 60)

        supervised_results = self.orchestrator.train_phase(
            phase_name="supervised_pretraining",
            hrm_dataloader=self.supervised_loaders.get("hrm"),
            trm_dataloader=self.supervised_loaders.get("trm"),
            val_dataloaders=self.supervised_loaders,
        )

        print(f"Supervised training completed")
        print(f"  HRM final loss: {supervised_results['hrm_metrics'][-1]['loss']:.4f}")
        print(f"  TRM final loss: {supervised_results['trm_metrics'][-1]['loss']:.4f}")

        # Phase 2: Self-Play Training
        print("\n[Phase 2] Self-Play Training")
        print("-" * 60)

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Self-play iteration
            metrics = await self.self_play_trainer.iteration(iteration)

            # Log metrics
            self.monitor.log_metrics(
                {
                    "iteration": iteration,
                    "self_play/success_rate": metrics["success_rate"],
                    "self_play/eval_success_rate": metrics["eval_success_rate"],
                    "self_play/num_episodes": metrics["num_episodes"],
                    "self_play/best_model": metrics["best_model_metric"],
                }
            )

            print(f"  Success rate: {metrics['success_rate']:.2%}")
            print(f"  Eval success: {metrics['eval_success_rate']:.2%}")
            print(f"  Best model: {metrics['best_model_metric']:.2%}")

            # Periodic supervised refinement
            if iteration % 10 == 0 and iteration > 0:
                print("  Running supervised refinement...")

                refinement_results = self.orchestrator.train_phase(
                    phase_name=f"refinement_{iteration}",
                    hrm_dataloader=self.supervised_loaders.get("hrm"),
                    trm_dataloader=self.supervised_loaders.get("trm"),
                )

                print(f"  Refinement complete")

            # Comprehensive evaluation every 20 iterations
            if iteration % 20 == 0:
                print("  Running comprehensive evaluation...")

                eval_results = self.evaluator.evaluate_all_agents(
                    hrm_agent=self.self_play_trainer.hrm_agent,
                    trm_agent=self.self_play_trainer.trm_agent,
                )

                self.monitor.log_metrics(
                    {
                        "iteration": iteration,
                        "eval/hrm_accuracy": eval_results["hrm"]["accuracy"],
                        "eval/trm_convergence": eval_results["trm"]["convergence_rate"],
                    }
                )

                print(f"  HRM Accuracy: {eval_results['hrm']['accuracy']:.2%}")
                print(f"  TRM Convergence: {eval_results['trm']['convergence_rate']:.2%}")

        # Final evaluation
        print("\n[Phase 3] Final Evaluation")
        print("-" * 60)

        final_eval = self.evaluator.evaluate_all_agents(
            hrm_agent=self.self_play_trainer.hrm_agent,
            trm_agent=self.self_play_trainer.trm_agent,
        )

        print("\nFinal Results:")
        print(f"  HRM Accuracy: {final_eval['hrm']['accuracy']:.2%}")
        print(f"  TRM Convergence Rate: {final_eval['trm']['convergence_rate']:.2%}")

        # Quality metrics
        quality = self.self_play_trainer.get_quality_metrics()
        print(f"\nSelf-Play Quality:")
        print(f"  Avg Success Rate: {quality['avg_success_rate']:.2%}")
        print(f"  Best Success Rate: {quality['best_success_rate']:.2%}")
        print(f"  Total Episodes: {quality['total_episodes_generated']}")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        return {
            "supervised_results": supervised_results,
            "final_eval": final_eval,
            "quality_metrics": quality,
        }


async def main():
    """Main entry point."""
    # Create pipeline
    pipeline = IntegratedTrainingPipeline()

    # Setup data
    print("Setting up supervised data...")
    pipeline.setup_supervised_data()

    # Setup self-play
    print("Setting up self-play trainer...")
    pipeline.setup_self_play()

    # Run training
    results = await pipeline.run_training(num_iterations=50)

    print("\nTraining pipeline completed successfully!")
    return results


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Integration

### Update config.yaml

Add self-play configuration to existing `training/config.yaml`:

```yaml
# Existing configuration
training:
  epochs: 10
  batch_size: 64
  learning_rate: 0.001
  # ... existing settings ...

# Add self-play configuration
self_play:
  enabled: true
  games_per_iteration: 1000
  parallel_batch_size: 32
  max_buffer_size: 10000

  mcts:
    num_simulations: 800
    c_puct: 1.25
    temperature_init: 1.0
    temperature_final: 0.1

  task_generators:
    math:
      enabled: true
      difficulty_range: [0.1, 1.0]
      num_per_iteration: 250

    code:
      enabled: true
      difficulty_range: [0.2, 0.9]
      num_per_iteration: 250

    reasoning:
      enabled: true
      difficulty_range: [0.3, 1.0]
      num_per_iteration: 250

    mcts:
      enabled: true
      difficulty_range: [0.1, 0.8]
      num_per_iteration: 250

  training:
    success_weight: 1.0
    failure_weight: 0.3
    alternating_interval: 10  # Supervised training every N iterations

  checkpointing:
    enabled: true
    save_every: 5
    keep_best: 5
```

## CLI Integration

Add self-play commands to `training/cli.py`:

```python
import asyncio
import click
from training.self_play_generator import SelfPlayTrainer

@click.group()
def cli():
    """LangGraph Multi-Agent MCTS Training CLI"""
    pass

@cli.command()
@click.option("--iterations", default=10, help="Number of training iterations")
@click.option("--games", default=1000, help="Games per iteration")
@click.option("--device", default="cuda", help="Device to use")
@click.option("--checkpoint-dir", default="./checkpoints/self_play")
def self_play(iterations, games, device, checkpoint_dir):
    """Run self-play training."""

    async def run():
        config = {
            "games_per_iteration": games,
            "batch_size": 64,
            "parallel_batch_size": 32,
        }

        trainer = SelfPlayTrainer(
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        for i in range(iterations):
            metrics = await trainer.iteration(i)
            click.echo(f"Iteration {i}: {metrics['success_rate']:.2%} success")

    asyncio.run(run())

@cli.command()
@click.option("--config", default="training/config.yaml")
@click.option("--iterations", default=50)
def integrated_training(config, iterations):
    """Run integrated training pipeline."""
    from training.integration_example import IntegratedTrainingPipeline

    async def run():
        pipeline = IntegratedTrainingPipeline(config)
        pipeline.setup_supervised_data()
        pipeline.setup_self_play()
        results = await pipeline.run_training(num_iterations=iterations)
        click.echo("Training complete!")
        click.echo(f"Best success rate: {results['quality_metrics']['best_success_rate']:.2%}")

    asyncio.run(run())

if __name__ == "__main__":
    cli()
```

## Usage Examples

### Run Self-Play Training

```bash
# Basic self-play
python -m training.cli self-play --iterations 10 --games 1000

# With custom device
python -m training.cli self-play --iterations 50 --device cuda:0

# Integrated training
python -m training.cli integrated-training --iterations 100
```

### Programmatic Usage

```python
# Option 1: Standalone self-play
from training.self_play_generator import SelfPlayTrainer

trainer = SelfPlayTrainer(config=config, device="cuda")
asyncio.run(trainer.iteration(0))

# Option 2: With agent trainer
from training.agent_trainer import HRMTrainer, TRMTrainer

hrm_trainer = HRMTrainer(config)
trm_trainer = TRMTrainer(config)

self_play_trainer = SelfPlayTrainer(
    hrm_agent=hrm_trainer.model,
    trm_agent=trm_trainer.model,
    config=config,
)

# Option 3: Full integration
from training.integration_example import IntegratedTrainingPipeline

pipeline = IntegratedTrainingPipeline()
asyncio.run(pipeline.run_training(num_iterations=100))
```

## Best Practices

### 1. Training Schedule

```python
# Week 1: Supervised pre-training
supervised_training(epochs=20)

# Week 2-4: Self-play with periodic supervised refinement
for iteration in range(100):
    self_play_iteration()

    if iteration % 10 == 0:
        supervised_refinement(epochs=5)

# Week 5+: Pure self-play
for iteration in range(100, 200):
    self_play_iteration()
```

### 2. Hyperparameter Tuning

```python
# Start conservative
config = {
    "games_per_iteration": 500,
    "mcts": {"num_simulations": 200},
}

# Scale up gradually
config = {
    "games_per_iteration": 1000,
    "mcts": {"num_simulations": 800},
}

# Production settings
config = {
    "games_per_iteration": 5000,
    "mcts": {"num_simulations": 1600},
}
```

### 3. Monitoring

Always monitor:
- Success rate trends
- Episode length
- Resource usage
- Model checkpoints

### 4. Checkpointing Strategy

```python
# Save every iteration
if iteration % 1 == 0:
    trainer._save_checkpoint(iteration)

# Keep only best 5
if len(checkpoints) > 5:
    remove_worst_checkpoint()

# Always save best model
if metrics["eval_success_rate"] > best_metric:
    trainer._save_checkpoint(iteration, best=True)
```

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce `parallel_batch_size`
2. **Slow Training**: Reduce `num_simulations`
3. **Low Success Rate**: Adjust task difficulty
4. **No Improvement**: Increase exploration (c_puct)

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

trainer = SelfPlayTrainer(config=config)
# Will show detailed logs of episode generation
```

## Next Steps

1. Start with supervised pre-training
2. Run small-scale self-play (100 episodes)
3. Evaluate and tune hyperparameters
4. Scale up to production (1000+ episodes)
5. Monitor and iterate

For more details, see:
- `training/SELF_PLAY_README.md` - Detailed documentation
- `training/examples/self_play_example.py` - Usage examples
- `training/tests/test_self_play_generator.py` - Test cases
