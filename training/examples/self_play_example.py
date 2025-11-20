"""
Example: AlphaZero-Style Self-Play Training Pipeline

This example demonstrates how to use the self-play training pipeline
to continuously improve your multi-agent MCTS system.

Usage:
    python training/examples/self_play_example.py
"""

import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage: Generate episodes and extract training data."""
    from training.self_play_generator import (
        MathProblemGenerator,
        SelfPlayEpisodeGenerator,
        TrainingDataExtractor,
    )

    logger.info("Example 1: Basic Self-Play Episode Generation")
    logger.info("=" * 60)

    # Create episode generator
    generator = SelfPlayEpisodeGenerator(device="cpu")

    # Generate a task
    task_gen = MathProblemGenerator(seed=42)
    tasks = task_gen.generate(1)
    task = tasks[0]

    logger.info(f"Task: {task['problem']}")

    # Generate episode
    episode = await generator.generate_episode(task, max_steps=10, timeout=30.0)

    logger.info(f"Episode outcome: {episode.outcome}")
    logger.info(f"Episode length: {episode.episode_length} steps")
    logger.info(f"Total reward: {episode.total_reward:.2f}")
    logger.info(f"MCTS traces captured: {len(episode.mcts_traces)}")

    # Extract training data
    extractor = TrainingDataExtractor()
    training_data = extractor.extract_examples([episode])

    logger.info("\nTraining data extracted:")
    logger.info(f"  Policy examples: {len(training_data['policy'])}")
    logger.info(f"  Value examples: {len(training_data['value'])}")
    logger.info(f"  Reasoning examples: {len(training_data['reasoning'])}")
    logger.info(f"  Negative examples: {len(training_data['negative'])}")

    return episode, training_data


async def example_with_agents():
    """Example with HRM and TRM agents."""
    from src.agents.hrm_agent import create_hrm_agent
    from src.agents.trm_agent import create_trm_agent
    from src.training.system_config import HRMConfig, TRMConfig
    from training.self_play_generator import SelfPlayEpisodeGenerator

    logger.info("\nExample 2: Self-Play with HRM and TRM Agents")
    logger.info("=" * 60)

    # Create agents
    hrm_config = HRMConfig(h_dim=256, l_dim=128)
    trm_config = TRMConfig(latent_dim=256)

    hrm_agent = create_hrm_agent(hrm_config, device="cpu")
    trm_agent = create_trm_agent(trm_config, device="cpu")

    logger.info(f"HRM parameters: {hrm_agent.get_parameter_count():,}")
    logger.info(f"TRM parameters: {trm_agent.get_parameter_count():,}")

    # Create generator with agents
    generator = SelfPlayEpisodeGenerator(
        hrm_agent=hrm_agent,
        trm_agent=trm_agent,
        mcts_config={"num_simulations": 50, "c_puct": 1.25},
        device="cpu",
    )

    # Generate episodes
    from training.self_play_generator import CodeGenerationTaskGenerator

    task_gen = CodeGenerationTaskGenerator(seed=42)
    tasks = task_gen.generate(3)

    logger.info(f"Generating {len(tasks)} episodes...")

    episodes = []
    for i, task in enumerate(tasks):
        episode = await generator.generate_episode(task, max_steps=8, timeout=20.0)
        episodes.append(episode)
        logger.info(
            f"Episode {i+1}: {episode.outcome}, "
            f"length={episode.episode_length}, "
            f"reward={episode.total_reward:.2f}"
        )

    success_count = sum(1 for ep in episodes if ep.outcome == "success")
    logger.info(f"\nSuccess rate: {success_count}/{len(episodes)}")

    return episodes


async def example_full_iteration():
    """Example: Complete training iteration."""
    from training.self_play_generator import SelfPlayTrainer

    logger.info("\nExample 3: Complete Training Iteration")
    logger.info("=" * 60)

    # Configuration
    config = {
        "games_per_iteration": 50,  # Generate 50 episodes per iteration
        "batch_size": 16,
        "parallel_batch_size": 10,  # Process 10 episodes in parallel
        "max_buffer_size": 1000,
        "mcts": {
            "num_simulations": 50,
            "c_puct": 1.25,
        },
    }

    # Create trainer
    trainer = SelfPlayTrainer(
        hrm_agent=None,  # Can add real agents here
        trm_agent=None,
        config=config,
        device="cpu",
        checkpoint_dir="./checkpoints/self_play_example",
    )

    logger.info("Running training iteration...")

    # Run one iteration
    metrics = await trainer.iteration(iteration_num=0)

    logger.info("\nIteration Results:")
    logger.info(f"  Episodes generated: {metrics['num_episodes']}")
    logger.info(f"  Success rate: {metrics['success_rate']:.2%}")
    logger.info(f"  Policy examples: {metrics['num_policy_examples']}")
    logger.info(f"  Value examples: {metrics['num_value_examples']}")
    logger.info(f"  Eval success rate: {metrics['eval_success_rate']:.2%}")
    logger.info(f"  Elapsed time: {metrics['elapsed_time']:.2f}s")

    # Get quality metrics
    quality = trainer.get_quality_metrics()
    logger.info("\nQuality Metrics:")
    for key, value in quality.items():
        logger.info(f"  {key}: {value:.4f}")

    # Get resource usage
    resources = trainer.get_resource_usage()
    if resources:
        logger.info("\nResource Usage:")
        for key, value in resources.items():
            logger.info(f"  {key}: {value:.2f}")

    return trainer


async def example_multi_iteration():
    """Example: Multiple training iterations with improvement tracking."""
    from training.self_play_generator import SelfPlayTrainer

    logger.info("\nExample 4: Multi-Iteration Training")
    logger.info("=" * 60)

    config = {
        "games_per_iteration": 30,
        "batch_size": 8,
        "parallel_batch_size": 5,
        "mcts": {"num_simulations": 30},
    }

    trainer = SelfPlayTrainer(config=config, device="cpu")

    num_iterations = 5
    logger.info(f"Running {num_iterations} training iterations...\n")

    for i in range(num_iterations):
        logger.info(f"Iteration {i+1}/{num_iterations}")
        logger.info("-" * 40)

        metrics = await trainer.iteration(i)

        logger.info(f"  Success rate: {metrics['success_rate']:.2%}")
        logger.info(f"  Eval success rate: {metrics['eval_success_rate']:.2%}")
        logger.info(f"  Best model metric: {metrics['best_model_metric']:.2%}")
        logger.info(f"  Time: {metrics['elapsed_time']:.2f}s\n")

    # Final summary
    logger.info("\nTraining Summary")
    logger.info("=" * 60)

    final_quality = trainer.get_quality_metrics()
    logger.info(f"Average success rate: {final_quality['avg_success_rate']:.2%}")
    logger.info(f"Success rate std: {final_quality['success_rate_std']:.2%}")
    logger.info(f"Success rate trend: {final_quality['success_rate_trend']:.4f}")
    logger.info(f"Best success rate: {final_quality['best_success_rate']:.2%}")
    logger.info(f"Total episodes: {final_quality['total_episodes_generated']}")

    return trainer


async def example_task_diversity():
    """Example: Diverse task generation."""
    from training.self_play_generator import (
        CodeGenerationTaskGenerator,
        MathProblemGenerator,
        MCTSSearchTaskGenerator,
        MultiStepReasoningGenerator,
    )

    logger.info("\nExample 5: Diverse Task Generation")
    logger.info("=" * 60)

    # Create different task generators
    generators = {
        "Math": MathProblemGenerator(difficulty_range=(0.2, 0.8), seed=42),
        "Code": CodeGenerationTaskGenerator(difficulty_range=(0.3, 0.9), seed=42),
        "Reasoning": MultiStepReasoningGenerator(difficulty_range=(0.4, 1.0), seed=42),
        "MCTS": MCTSSearchTaskGenerator(difficulty_range=(0.1, 0.7), seed=42),
    }

    for name, generator in generators.items():
        logger.info(f"\n{name} Tasks:")
        tasks = generator.generate(3)

        for i, task in enumerate(tasks, 1):
            logger.info(f"  {i}. {task['problem'][:60]}...")
            logger.info(f"     Difficulty: {task['difficulty']:.2f}")
            logger.info(f"     Type: {task['type']}")


async def example_data_extraction_details():
    """Example: Detailed training data extraction."""
    from training.self_play_generator import (
        SelfPlayTrainer,
        TrainingDataExtractor,
    )

    logger.info("\nExample 6: Detailed Training Data Extraction")
    logger.info("=" * 60)

    # Generate some episodes
    trainer = SelfPlayTrainer(
        config={
            "games_per_iteration": 10,
            "parallel_batch_size": 5,
        },
        device="cpu",
    )

    episodes = await trainer.run_self_play(num_games=10)

    # Extract data
    extractor = TrainingDataExtractor(
        success_weight=1.0,  # Weight for successful episodes
        failure_weight=0.3,  # Lower weight for failures
    )

    training_data = extractor.extract_examples(episodes)

    # Analyze policy examples
    logger.info("\nPolicy Examples Analysis:")
    policy_examples = training_data["policy"]
    if policy_examples:
        logger.info(f"  Total: {len(policy_examples)}")
        logger.info(f"  Avg weight: {sum(e.weight for e in policy_examples) / len(policy_examples):.2f}")

        # Sample example
        example = policy_examples[0]
        logger.info(f"  Sample state shape: {example.state.shape}")
        logger.info(f"  Sample target: {list(example.target.keys())[:3]}...")

    # Analyze value examples
    logger.info("\nValue Examples Analysis:")
    value_examples = training_data["value"]
    if value_examples:
        logger.info(f"  Total: {len(value_examples)}")
        logger.info(f"  Target range: [{min(e.target for e in value_examples):.2f}, "
                    f"{max(e.target for e in value_examples):.2f}]")

    # Analyze negative examples
    logger.info("\nNegative Examples Analysis:")
    negative_examples = training_data["negative"]
    logger.info(f"  Total: {len(negative_examples)}")

    return training_data


async def example_checkpoint_management():
    """Example: Checkpoint saving and loading."""
    from training.self_play_generator import SelfPlayTrainer

    logger.info("\nExample 7: Checkpoint Management")
    logger.info("=" * 60)

    checkpoint_dir = "./checkpoints/self_play_checkpoint_example"

    # Create trainer and run iteration
    trainer = SelfPlayTrainer(
        config={"games_per_iteration": 10, "parallel_batch_size": 5},
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("Running iteration and saving checkpoint...")
    await trainer.iteration(0)

    logger.info(f"Best model metric: {trainer.best_model_metric:.2%}")

    # Load checkpoint
    logger.info("\nLoading checkpoint...")
    trainer2 = SelfPlayTrainer(
        config={"games_per_iteration": 10},
        device="cpu",
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_path = Path(checkpoint_dir) / "iteration_0.pt"
    if checkpoint_path.exists():
        trainer2.load_checkpoint(str(checkpoint_path))
        logger.info(f"Loaded checkpoint from iteration {trainer2.current_iteration}")
        logger.info(f"Best metric: {trainer2.best_model_metric:.2%}")
    else:
        logger.info("No checkpoint found")


async def main():
    """Run all examples."""
    logger.info("=" * 60)
    logger.info("SELF-PLAY TRAINING PIPELINE EXAMPLES")
    logger.info("=" * 60)

    try:
        # Example 1: Basic usage
        await example_basic_usage()

        # Example 2: With agents (requires agent modules)
        try:
            await example_with_agents()
        except ImportError as e:
            logger.info(f"\nSkipping agent example (missing dependencies): {e}")

        # Example 3: Full iteration
        await example_full_iteration()

        # Example 4: Multi-iteration
        await example_multi_iteration()

        # Example 5: Task diversity
        await example_task_diversity()

        # Example 6: Data extraction details
        await example_data_extraction_details()

        # Example 7: Checkpoint management
        await example_checkpoint_management()

        logger.info("\n" + "=" * 60)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
