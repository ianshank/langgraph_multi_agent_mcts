"""
Training Pipeline Integration Examples for Extended Datasets.

This module demonstrates how to use the new dataset loaders with the
LangGraph Multi-Agent MCTS framework for training different components.

Examples include:
- HRM training on ARC dataset
- TRM training on GSM8K
- MCTS training on Chess games
- Quality engineering with IDoFT
- Code generation with HumanEval
- Reasoning evaluation with BIG-Bench Hard
"""

import asyncio
import logging
from pathlib import Path

from src.data.dataset_loader import (
    ARCLoader,
    GSM8KLoader,
    IDoFTLoader,
    HumanEvalLoader,
    ChessGamesLoader,
    BIGBenchHardLoader,
    CombinedDatasetLoader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Training HRM on ARC Dataset
# =============================================================================


def example_1_hrm_training_on_arc():
    """
    Train HRM (Hierarchical Reasoning Model) on ARC dataset.

    ARC provides ~1,000 training examples of abstract pattern recognition
    tasks that require hierarchical decomposition—perfect for HRM training.
    """
    logger.info("=" * 80)
    logger.info("Example 1: HRM Training on ARC Dataset")
    logger.info("=" * 80)

    # Load ARC dataset
    arc_loader = ARCLoader(cache_dir="./cache/arc")
    arc_samples = arc_loader.load(split="train")

    logger.info(f"Loaded {len(arc_samples)} ARC training samples")

    # Get statistics
    stats = arc_loader.get_statistics()
    logger.info(f"Dataset statistics: {stats}")

    # Example: Process first sample for HRM training
    if arc_samples:
        sample = arc_samples[0]
        logger.info(f"\nSample ID: {sample.id}")
        logger.info(f"Domain: {sample.domain}")
        logger.info(f"Difficulty: {sample.difficulty}")
        logger.info(f"Task Data: {sample.metadata.get('task_data', {}).keys()}")

        # In practice, you would:
        # 1. Extract input-output pairs from task_data
        # 2. Train HRM to decompose the pattern recognition task
        # 3. Validate hierarchical decomposition structure

    logger.info("\nUse case: Train HRM agent to hierarchically decompose")
    logger.info("visual pattern recognition tasks into subtasks.")


# =============================================================================
# Example 2: Training TRM on GSM8K Dataset
# =============================================================================


def example_2_trm_training_on_gsm8k():
    """
    Train TRM (Task Refinement Model) on GSM8K dataset.

    GSM8K contains 7,500 training math problems requiring multi-step reasoning.
    Perfect for training iterative refinement capabilities.
    """
    logger.info("=" * 80)
    logger.info("Example 2: TRM Training on GSM8K Dataset")
    logger.info("=" * 80)

    # Load GSM8K dataset
    gsm8k_loader = GSM8KLoader(cache_dir="./cache/gsm8k")
    gsm8k_samples = gsm8k_loader.load(split="train", config="main")

    logger.info(f"Loaded {len(gsm8k_samples)} GSM8K training samples")

    # Get reasoning samples
    reasoning_samples = gsm8k_loader.get_reasoning_samples()
    logger.info(f"Samples with multi-step reasoning: {len(reasoning_samples)}")

    # Example: Process sample for TRM training
    if reasoning_samples:
        sample = reasoning_samples[0]
        logger.info(f"\nSample ID: {sample.id}")
        logger.info(f"Question: {sample.text[:200]}...")
        logger.info(f"Number of reasoning steps: {len(sample.reasoning_steps)}")

        # Show reasoning steps
        logger.info("\nReasoning steps:")
        for i, step in enumerate(sample.reasoning_steps[:3], 1):
            logger.info(f"  {i}. {step[:100]}...")

        # In practice, you would:
        # 1. Use initial question as starting point
        # 2. Train TRM to iteratively refine solution
        # 3. Compare refined solution with ground truth answer

    logger.info("\nUse case: Train TRM agent to iteratively refine")
    logger.info("mathematical solutions through multiple refinement steps.")


# =============================================================================
# Example 3: Training MCTS on Chess Games
# =============================================================================


def example_3_mcts_training_on_chess():
    """
    Train MCTS policies on high-level chess games.

    Chess dataset contains 14M games from players with mean ELO 2388.
    Perfect for training MCTS tree search policies.
    """
    logger.info("=" * 80)
    logger.info("Example 3: MCTS Training on Chess Games")
    logger.info("=" * 80)

    # Load Chess dataset (limited samples for demo)
    chess_loader = ChessGamesLoader(cache_dir="./cache/chess_games")
    chess_samples = chess_loader.load(
        split="train", min_elo=2200, max_samples=100, streaming=True
    )

    logger.info(f"Loaded {len(chess_samples)} high-level chess games")

    # Get high-level games
    elite_games = chess_loader.get_high_level_games(min_elo=2500)
    logger.info(f"Elite games (ELO >= 2500): {len(elite_games)}")

    # Example: Process game for MCTS training
    if chess_samples:
        sample = chess_samples[0]
        logger.info(f"\nGame ID: {sample.id}")
        logger.info(f"White ELO: {sample.metadata.get('white_elo')}")
        logger.info(f"Black ELO: {sample.metadata.get('black_elo')}")
        logger.info(f"Result: {sample.metadata.get('end_type')}")
        logger.info(f"Number of moves: {sample.metadata.get('num_moves')}")

        # Show first moves
        moves_san = sample.metadata.get("moves_san", "").split()[:10]
        logger.info(f"First 10 moves: {' '.join(moves_san)}")

        # In practice, you would:
        # 1. Parse game moves into board positions
        # 2. Train MCTS value network on position evaluations
        # 3. Train MCTS policy network on move selections

    logger.info("\nUse case: Train MCTS engine to evaluate positions and")
    logger.info("select optimal moves through tree search.")


# =============================================================================
# Example 4: Quality Engineering with IDoFT
# =============================================================================


def example_4_quality_engineering_with_idoft():
    """
    Train quality engineering agents on IDoFT flaky test dataset.

    IDoFT contains 2,000+ flaky tests with root cause classifications.
    Perfect for training flaky test detection and fix generation.
    """
    logger.info("=" * 80)
    logger.info("Example 4: Quality Engineering with IDoFT")
    logger.info("=" * 80)

    # Note: IDoFT may require local data or manual download
    # For this example, we show the intended usage
    idoft_loader = IDoFTLoader(
        cache_dir="./cache/idoft",
        data_path=None,  # Set to local IDoFT path if available
    )

    try:
        # Attempt to load IDoFT data
        idoft_samples = idoft_loader.load(split="train")
        logger.info(f"Loaded {len(idoft_samples)} IDoFT samples")

        # Get samples by category
        async_samples = idoft_loader.get_by_category("async_wait")
        concurrency_samples = idoft_loader.get_by_category("concurrency")

        logger.info(f"Async wait flaky tests: {len(async_samples)}")
        logger.info(f"Concurrency flaky tests: {len(concurrency_samples)}")

        # Example: Process flaky test sample
        if idoft_samples:
            sample = idoft_samples[0]
            logger.info(f"\nSample ID: {sample.id}")
            logger.info(f"Category: {sample.labels}")
            logger.info(f"Project: {sample.metadata.get('project')}")
            logger.info(f"Test name: {sample.metadata.get('test_name')}")

            # In practice, you would:
            # 1. Use HRM to decompose flaky test into root cause analysis
            # 2. Use TRM to iteratively refine fix strategies
            # 3. Use MCTS to explore different fix approaches

    except FileNotFoundError:
        logger.warning("IDoFT dataset not found locally.")
        logger.info("\nTo use IDoFT dataset:")
        logger.info("1. Clone: git clone https://github.com/TestingResearchIllinois/idoft")
        logger.info("2. Or download from: http://mir.cs.illinois.edu/flakytests/")
        logger.info("3. Set data_path parameter to local IDoFT directory")

    logger.info("\nUse case: Train agents to detect flaky tests,")
    logger.info("classify root causes, and generate fixes.")


# =============================================================================
# Example 5: Code Generation with HumanEval
# =============================================================================


def example_5_code_generation_with_humaneval():
    """
    Evaluate code generation capabilities using HumanEval.

    HumanEval contains 164 hand-crafted programming problems with unit tests.
    Perfect for evaluating code generation and test synthesis.
    """
    logger.info("=" * 80)
    logger.info("Example 5: Code Generation with HumanEval")
    logger.info("=" * 80)

    # Load HumanEval dataset
    humaneval_loader = HumanEvalLoader(cache_dir="./cache/humaneval")
    humaneval_samples = humaneval_loader.load(split="test")

    logger.info(f"Loaded {len(humaneval_samples)} HumanEval problems")

    # Example: Process programming problem
    if humaneval_samples:
        sample = humaneval_samples[0]
        logger.info(f"\nTask ID: {sample.metadata.get('task_id')}")
        logger.info(f"Entry Point: {sample.metadata.get('entry_point')}")
        logger.info(f"\nPrompt:\n{sample.metadata.get('prompt', '')[:300]}...")

        # In practice, you would:
        # 1. Use HRM to decompose function into testable units
        # 2. Use TRM to iteratively refine implementation
        # 3. Validate against provided unit tests

    logger.info("\nUse case: Generate code implementations and comprehensive")
    logger.info("unit tests using multi-agent decomposition and refinement.")


# =============================================================================
# Example 6: Reasoning Evaluation with BIG-Bench Hard
# =============================================================================


def example_6_reasoning_evaluation_with_bbh():
    """
    Evaluate complex reasoning capabilities using BIG-Bench Hard.

    BBH contains 23 challenging tasks requiring causal reasoning,
    counterfactual thinking, and multi-hop inference.
    """
    logger.info("=" * 80)
    logger.info("Example 6: Reasoning Evaluation with BIG-Bench Hard")
    logger.info("=" * 80)

    # Load BIG-Bench Hard dataset
    bbh_loader = BIGBenchHardLoader(cache_dir="./cache/bigbench_hard")
    bbh_samples = bbh_loader.load(split="train")

    logger.info(f"Loaded {len(bbh_samples)} BIG-Bench Hard samples")

    # Get statistics
    stats = bbh_loader.get_statistics()
    logger.info(f"Task distribution: {stats.difficulty_distribution}")

    # Example: Process reasoning task
    if bbh_samples:
        sample = bbh_samples[0]
        logger.info(f"\nTask: {sample.labels}")
        logger.info(f"Question: {sample.text[:200]}...")
        logger.info(f"Target: {sample.metadata.get('target', '')[:100]}")

        # Get samples for specific task
        causal_samples = bbh_loader.get_by_task("causal_judgement")
        logger.info(f"\nCausal judgement samples: {len(causal_samples)}")

        # In practice, you would:
        # 1. Use HRM to decompose complex reasoning task
        # 2. Evaluate consensus mechanism on reasoning paths
        # 3. Benchmark against human performance

    logger.info("\nUse case: Evaluate HRM hierarchical reasoning and")
    logger.info("consensus mechanisms on challenging reasoning tasks.")


# =============================================================================
# Example 7: Combined Dataset Loading
# =============================================================================


def example_7_combined_dataset_loading():
    """
    Load and analyze multiple datasets together.

    Demonstrates using CombinedDatasetLoader to manage all datasets
    and get training samples for different agent types.
    """
    logger.info("=" * 80)
    logger.info("Example 7: Combined Dataset Loading")
    logger.info("=" * 80)

    # Initialize combined loader with all datasets
    combined_loader = CombinedDatasetLoader(cache_dir="./cache")

    # Load datasets selectively
    logger.info("Loading multiple datasets...")
    combined_samples = combined_loader.load_all(
        # Original datasets
        dabstep_split="train",
        primus_max_samples=1000,
        include_instruct=True,
        # New datasets (set to True to load)
        include_arc=False,  # Set True to load ARC
        include_gsm8k=False,  # Set True to load GSM8K
        include_idoft=False,  # Set True to load IDoFT
        include_humaneval=False,  # Set True to load HumanEval
        include_chess=False,  # Set True to load Chess
        chess_max_samples=1000,
        include_bbh=False,  # Set True to load BIG-Bench Hard
    )

    logger.info(f"Total samples loaded: {len(combined_samples)}")

    # Get dataset summary
    summary = combined_loader.get_dataset_summary()
    logger.info(f"\nDataset Summary:")
    logger.info(f"  Total samples: {summary['total_samples']}")
    logger.info(f"  Domains: {summary['domains']}")
    logger.info(f"  Sources: {summary['sources']}")

    # Get training samples for each agent type
    hrm_samples = combined_loader.get_hrm_training_samples()
    trm_samples = combined_loader.get_trm_training_samples()
    mcts_samples = combined_loader.get_mcts_training_samples()
    code_samples = combined_loader.get_code_generation_samples()

    logger.info(f"\nAgent-specific training data:")
    logger.info(f"  HRM training samples: {len(hrm_samples)}")
    logger.info(f"  TRM training samples: {len(trm_samples)}")
    logger.info(f"  MCTS training samples: {len(mcts_samples)}")
    logger.info(f"  Code generation samples: {len(code_samples)}")

    # Export combined dataset
    output_path = "./data/combined_training_data.jsonl"
    combined_loader.export_for_training(output_path, format="jsonl")
    logger.info(f"\nExported combined dataset to: {output_path}")


# =============================================================================
# Example 8: Training Pipeline with Configuration
# =============================================================================


def example_8_training_pipeline_with_config():
    """
    Demonstrate loading datasets using configuration file.

    Shows how to integrate with training/config.yaml for production use.
    """
    logger.info("=" * 80)
    logger.info("Example 8: Training Pipeline with Configuration")
    logger.info("=" * 80)

    import yaml

    # Load configuration
    config_path = Path(__file__).parent.parent / "training" / "config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info("Loaded training configuration")

        # Display dataset configurations
        data_config = config.get("data", {})
        logger.info("\nConfigured datasets:")

        for dataset_name, dataset_config in data_config.items():
            if isinstance(dataset_config, dict) and dataset_config.get("path"):
                enabled = dataset_config.get("enabled", True)
                status = "ENABLED" if enabled else "DISABLED"
                logger.info(f"  {dataset_name}: {status}")
                logger.info(f"    Path: {dataset_config.get('path')}")
                logger.info(f"    Cache: {dataset_config.get('cache_dir')}")

        # Example: Enable specific datasets in config
        logger.info("\nTo enable datasets, edit training/config.yaml:")
        logger.info("  arc.enabled: true     # For HRM training")
        logger.info("  gsm8k.enabled: true   # For TRM training")
        logger.info("  chess_games.enabled: true  # For MCTS training")

    else:
        logger.warning(f"Configuration file not found: {config_path}")


# =============================================================================
# Main: Run All Examples
# =============================================================================


def main():
    """Run all dataset training examples."""
    print("\n" + "=" * 80)
    print("LANGGRAPH MULTI-AGENT MCTS: DATASET TRAINING EXAMPLES")
    print("=" * 80 + "\n")

    # Run examples
    example_1_hrm_training_on_arc()
    print("\n")

    example_2_trm_training_on_gsm8k()
    print("\n")

    example_3_mcts_training_on_chess()
    print("\n")

    example_4_quality_engineering_with_idoft()
    print("\n")

    example_5_code_generation_with_humaneval()
    print("\n")

    example_6_reasoning_evaluation_with_bbh()
    print("\n")

    example_7_combined_dataset_loading()
    print("\n")

    example_8_training_pipeline_with_config()
    print("\n")

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
