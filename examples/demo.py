#!/usr/bin/env python3
"""
Interactive Demo for LangGraph Multi-Agent MCTS.

This demo showcases:
1. Basic MCTS search with configurable presets
2. Neural MCTS integration (if available)
3. Multi-agent pipeline simulation
4. Configuration customization
5. Expert Iteration concepts

Run with: python examples/demo.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from src.framework.actions import (
    ActionType,
    AgentType,
    GraphConfig,
    ConfidenceConfig,
    RolloutWeights,
    create_research_config,
    create_coding_config,
    create_creative_config,
)
from src.framework.mcts.config import (
    MCTSConfig,
    ConfigPreset,
    create_preset_config,
)
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import RandomRolloutPolicy

# Optional imports
try:
    from src.framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        NeuralRolloutPolicy,
        create_neural_mcts_adapter,
        get_balanced_neural_config,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    from src.training.expert_iteration import (
        ExpertIterationConfig,
        ReplayBuffer,
        Trajectory,
        TrajectoryStep,
    )
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}--- {text} ---{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_metric(name: str, value: Any):
    """Print a metric."""
    print(f"  {Colors.CYAN}{name}:{Colors.ENDC} {value}")


async def demo_basic_mcts():
    """Demo 1: Basic MCTS Search."""
    print_header("Demo 1: Basic MCTS Search")

    print_info("Creating MCTS engine with balanced configuration...")

    # Create configuration
    config = create_preset_config(ConfigPreset.BALANCED)
    print_metric("Iterations", config.num_iterations)
    print_metric("Exploration Weight", config.exploration_weight)
    print_metric("Progressive Widening K", config.progressive_widening_k)

    # Create engine and rollout policy
    engine = MCTSEngine(
        seed=42,
        exploration_weight=config.exploration_weight,
        progressive_widening_k=config.progressive_widening_k,
    )
    rollout_policy = RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    # Create root state
    root_state = MCTSState(
        state_id="root",
        features={"query": "Solve a complex reasoning problem", "depth": 0},
    )
    root = MCTSNode(state=root_state, rng=engine.rng)

    print_section("Defining Action Space")

    # Define actions
    root_actions = ActionType.root_actions()
    cont_actions = ActionType.continuation_actions()

    print_info(f"Root actions: {root_actions}")
    print_info(f"Continuation actions: {cont_actions}")

    def action_generator(state: MCTSState) -> list[str]:
        depth = len(state.state_id.split("_")) - 1
        if depth == 0:
            return root_actions
        elif depth < 5:
            return cont_actions
        return []

    def state_transition(state: MCTSState, action: str) -> MCTSState:
        return MCTSState(
            state_id=f"{state.state_id}_{action}",
            features={**state.features, "depth": state.features.get("depth", 0) + 1},
        )

    print_section("Running MCTS Search")

    start_time = time.perf_counter()

    best_action, stats = await engine.search(
        root=root,
        num_iterations=config.num_iterations,
        action_generator=action_generator,
        state_transition=state_transition,
        rollout_policy=rollout_policy,
    )

    elapsed = time.perf_counter() - start_time

    print_success(f"Search completed in {elapsed:.3f}s")
    print_metric("Best Action", best_action)
    print_metric("Iterations Completed", stats["iterations"])
    print_metric("Best Action Visits", stats["best_action_visits"])
    print_metric("Best Action Value", f"{stats['best_action_value']:.4f}")
    print_metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.2%}")

    return stats


async def demo_configuration_presets():
    """Demo 2: Configuration Presets."""
    print_header("Demo 2: Configuration Presets")

    print_section("MCTS Presets")

    for preset in ConfigPreset:
        config = create_preset_config(preset)
        print(f"\n{Colors.BOLD}{preset.value.upper()}{Colors.ENDC}")
        print_metric("Iterations", config.num_iterations)
        print_metric("Exploration", config.exploration_weight)
        print_metric("Rollout Depth", config.max_rollout_depth)

    print_section("Domain Presets")

    presets = {
        "Research": create_research_config(),
        "Coding": create_coding_config(),
        "Creative": create_creative_config(),
    }

    for name, config in presets.items():
        print(f"\n{Colors.BOLD}{name}{Colors.ENDC}")
        print_metric("Max Iterations", config.max_iterations)
        print_metric("Parallel Agents", config.enable_parallel_agents)
        print_metric("Consensus Threshold", config.confidence.consensus_threshold)
        print_metric("Temperature", config.synthesis.temperature)
        print_metric("Heuristic Weight", config.rollout_weights.heuristic_weight)


async def demo_neural_mcts():
    """Demo 3: Neural MCTS Integration."""
    print_header("Demo 3: Neural MCTS Integration")

    if not NEURAL_AVAILABLE:
        print_warning("Neural MCTS not available. Install PyTorch to enable.")
        return None

    print_info("Creating Neural MCTS adapter...")

    # Create neural config
    config = get_balanced_neural_config()
    print_metric("Simulations", config.num_simulations)
    print_metric("C_PUCT", config.c_puct)
    print_metric("Dirichlet Epsilon", config.dirichlet_epsilon)
    print_metric("Temperature Init", config.temperature_init)

    # Create adapter (without actual network for demo)
    adapter = create_neural_mcts_adapter(network=None, config=config)

    print_section("Testing Neural Rollout Policy")

    import numpy as np

    policy = adapter.rollout_policy
    state = MCTSState(
        state_id="test",
        features={"confidence": 0.8, "quality": 0.7},
    )

    rng = np.random.default_rng(42)
    values = []

    for _ in range(10):
        value = await policy.evaluate(state, rng)
        values.append(value)

    print_success("Neural rollout policy working")
    print_metric("Mean Value", f"{np.mean(values):.4f}")
    print_metric("Std Value", f"{np.std(values):.4f}")

    # Check cache stats
    cache_stats = policy.get_cache_stats()
    print_metric("Cache Size", cache_stats["cache_size"])
    print_metric("Cache Hit Rate", f"{cache_stats['hit_rate']:.2%}")

    return values


async def demo_multi_agent_simulation():
    """Demo 4: Multi-Agent Pipeline Simulation."""
    print_header("Demo 4: Multi-Agent Pipeline Simulation")

    print_info("Simulating multi-agent reasoning pipeline...")

    # Configure agents
    config = GraphConfig(
        max_iterations=3,
        enable_parallel_agents=True,
        confidence=ConfidenceConfig(consensus_threshold=0.75),
    )

    print_metric("Max Iterations", config.max_iterations)
    print_metric("Parallel Agents", config.enable_parallel_agents)
    print_metric("Consensus Threshold", config.confidence.consensus_threshold)

    print_section("Agent Execution")

    agents = [
        ("HRM", "Hierarchical Reasoning Model", 0.85),
        ("TRM", "Task Refinement Model", 0.82),
        ("MCTS", "Monte Carlo Tree Search", 0.88),
    ]

    total_confidence = 0.0

    for agent_id, agent_name, confidence in agents:
        print(f"\n{Colors.BOLD}{agent_id}{Colors.ENDC}: {agent_name}")
        print_metric("Confidence", f"{confidence:.2%}")

        # Simulate processing
        await asyncio.sleep(0.1)
        total_confidence += confidence

        print_success("Agent completed")

    print_section("Consensus Evaluation")

    consensus_score = total_confidence / len(agents)
    threshold = config.confidence.consensus_threshold

    print_metric("Consensus Score", f"{consensus_score:.2%}")
    print_metric("Threshold", f"{threshold:.2%}")

    if consensus_score >= threshold:
        print_success("Consensus reached!")
    else:
        print_warning("Consensus not reached, would iterate")

    return consensus_score


async def demo_expert_iteration():
    """Demo 5: Expert Iteration Concepts."""
    print_header("Demo 5: Expert Iteration Concepts")

    if not TRAINING_AVAILABLE:
        print_warning("Training module not available. Install PyTorch to enable.")
        return None

    print_info("Demonstrating Expert Iteration self-improvement loop...")

    print_section("1. Configure Training")

    config = ExpertIterationConfig(
        num_episodes_per_iteration=10,
        mcts_simulations=50,
        batch_size=32,
        buffer_size=1000,
    )

    print_metric("Episodes per Iteration", config.num_episodes_per_iteration)
    print_metric("MCTS Simulations", config.mcts_simulations)
    print_metric("Batch Size", config.batch_size)
    print_metric("Buffer Size", config.buffer_size)

    print_section("2. Initialize Replay Buffer")

    buffer = ReplayBuffer(max_size=config.buffer_size, seed=42)

    # Generate sample data
    for i in range(100):
        state = MCTSState(
            state_id=f"state_{i}",
            features={"step": i},
        )
        policy = {
            "explore": 0.4,
            "exploit": 0.35,
            "delegate": 0.25,
        }
        value = 0.5 + 0.3 * (i / 100)  # Simulated improving value

        buffer.add(state, policy, value)

    print_success(f"Buffer populated with {len(buffer)} entries")

    print_section("3. Generate Trajectories")

    trajectories = []
    for i in range(5):
        steps = []
        for j in range(8):
            step = TrajectoryStep(
                state=MCTSState(state_id=f"traj_{i}_step_{j}", features={}),
                action=["explore", "exploit", "delegate"][j % 3],
                mcts_policy={"explore": 0.4, "exploit": 0.35, "delegate": 0.25},
                value=0.5 + 0.05 * j,
            )
            steps.append(step)

        outcome = 0.7 + 0.1 * i  # Improving outcomes
        trajectories.append(Trajectory(
            steps=steps,
            outcome=outcome,
            metadata={"trajectory_id": i},
        ))

    print_success(f"Generated {len(trajectories)} trajectories")
    print_metric("Avg Trajectory Length", sum(t.length for t in trajectories) / len(trajectories))
    print_metric("Avg Outcome", sum(t.outcome for t in trajectories) / len(trajectories))

    print_section("4. Training Loop (Conceptual)")

    print_info("In a full implementation, the training loop would:")
    print("  1. Sample batches from replay buffer")
    print("  2. Compute policy and value losses")
    print("  3. Update neural network weights")
    print("  4. Generate new trajectories with improved network")
    print("  5. Repeat for self-improvement")

    return buffer


async def demo_interactive_session():
    """Demo 6: Interactive Session."""
    print_header("Demo 6: Interactive Session")

    print_info("Simulating an interactive user session...")

    queries = [
        "Explain quantum computing fundamentals",
        "How does MCTS improve decision making?",
        "Compare supervised vs unsupervised learning",
    ]

    results = []

    for i, query in enumerate(queries, 1):
        print_section(f"Query {i}")
        print(f"{Colors.YELLOW}Query:{Colors.ENDC} {query}")

        # Process with MCTS
        engine = MCTSEngine(seed=i)
        rollout_policy = RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)
        root = MCTSNode(
            state=MCTSState(
                state_id="root",
                features={"query": query},
            ),
            rng=engine.rng,
        )

        def action_gen(s):
            return ["analyze", "synthesize", "elaborate"] if len(s.state_id) < 15 else []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        best_action, stats = await engine.search(
            root=root,
            num_iterations=25,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        result = {
            "query": query,
            "best_action": best_action,
            "confidence": stats["best_action_value"],
        }
        results.append(result)

        print_metric("Best Action", best_action)
        print_metric("Confidence", f"{stats['best_action_value']:.2%}")
        print_success("Query processed")

    print_section("Session Summary")

    print_metric("Total Queries", len(results))
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    print_metric("Avg Confidence", f"{avg_confidence:.2%}")

    return results


async def main():
    """Run all demos."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     LangGraph Multi-Agent MCTS - Interactive Demo         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")

    print_info("This demo showcases the key features of the framework.")
    print_info(f"Neural MCTS Available: {NEURAL_AVAILABLE}")
    print_info(f"Training Available: {TRAINING_AVAILABLE}")

    demos = [
        ("Basic MCTS Search", demo_basic_mcts),
        ("Configuration Presets", demo_configuration_presets),
        ("Neural MCTS Integration", demo_neural_mcts),
        ("Multi-Agent Simulation", demo_multi_agent_simulation),
        ("Expert Iteration Concepts", demo_expert_iteration),
        ("Interactive Session", demo_interactive_session),
    ]

    results = {}

    for name, demo_fn in demos:
        try:
            result = await demo_fn()
            results[name] = {"status": "success", "result": result}
        except Exception as e:
            print(f"{Colors.RED}Error in {name}: {e}{Colors.ENDC}")
            results[name] = {"status": "error", "error": str(e)}

    # Summary
    print_header("Demo Summary")

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    print_metric("Demos Completed", f"{success_count}/{len(demos)}")

    for name, result in results.items():
        status = "✓" if result["status"] == "success" else "✗"
        color = Colors.GREEN if result["status"] == "success" else Colors.RED
        print(f"{color}{status} {name}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Demo complete!{Colors.ENDC}")
    print_info("Run 'streamlit run src/ui/app.py' for the full UI experience.")
    print_info("Run 'uvicorn src.api.server:app --reload' for the API server.")


if __name__ == "__main__":
    asyncio.run(main())
