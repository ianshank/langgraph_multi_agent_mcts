"""
Hybrid Agent Demo.

Demonstrates the hybrid LLM-neural agent with cost-performance analysis.
"""

import asyncio
import time

import torch

from src.agents.hybrid_agent import DecisionSource, HybridAgent, HybridConfig
from src.models.policy_network import create_policy_network
from src.models.value_network import create_value_network


class MockLLMClient:
    """Mock LLM client for demonstration."""

    def __init__(self, latency_ms=200, cost_per_call=0.05):
        self.latency_ms = latency_ms
        self.cost_per_call = cost_per_call

    async def generate(self, _prompt):
        """Mock LLM generation."""
        # Simulate LLM latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Simple mock response
        return {"text": "3", "tokens": 100}


async def demo_hybrid_modes():
    """Demonstrate different hybrid agent modes."""
    print("\n" + "=" * 60)
    print("HYBRID AGENT MODES DEMONSTRATION")
    print("=" * 60)

    # Create neural networks
    policy_net = create_policy_network(state_dim=9, action_dim=9, config={"hidden_dims": [64, 32]})

    value_net = create_value_network(state_dim=9, config={"hidden_dims": [64, 32]})

    # Create mock LLM client
    llm_client = MockLLMClient()

    # Test different modes
    modes = ["neural_only", "llm_only", "auto"]

    test_state = torch.randn(9)

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        config = HybridConfig(mode=mode, policy_confidence_threshold=0.8)

        agent = HybridAgent(policy_net=policy_net, value_net=value_net, llm_client=llm_client, config=config)

        # Select action
        start_time = time.time()
        action, metadata = await agent.select_action(test_state)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"  Action: {action}")
        print(f"  Source: {metadata.source.value}")
        print(f"  Confidence: {metadata.confidence}")
        print(f"  Cost: ${metadata.cost:.6f}")
        print(f"  Latency: {elapsed_ms:.2f}ms")


async def demo_confidence_thresholds():
    """Demonstrate effect of confidence thresholds."""
    print("\n" + "=" * 60)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 60)

    # Create networks
    policy_net = create_policy_network(state_dim=9, action_dim=9, config={"hidden_dims": [64, 32]})

    value_net = create_value_network(state_dim=9, config={"hidden_dims": [64, 32]})

    llm_client = MockLLMClient()

    # Test different thresholds
    thresholds = [0.5, 0.7, 0.9]

    test_states = [torch.randn(9) for _ in range(10)]

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")

        config = HybridConfig(mode="auto", policy_confidence_threshold=threshold)

        agent = HybridAgent(policy_net=policy_net, value_net=value_net, llm_client=llm_client, config=config)

        neural_count = 0
        llm_count = 0

        for state in test_states:
            action, metadata = await agent.select_action(state)

            if metadata.source == DecisionSource.POLICY_NETWORK:
                neural_count += 1
            else:
                llm_count += 1

        # Get statistics
        stats = agent.get_statistics()
        savings = agent.get_cost_savings()

        print(f"  Neural usage: {neural_count}/10 ({neural_count * 10}%)")
        print(f"  LLM fallback: {llm_count}/10 ({llm_count * 10}%)")
        print(f"  Total cost: ${stats['costs']['total']:.6f}")
        print(f"  Cost savings: {savings.savings_percentage:.1f}%")


async def demo_cost_performance_analysis():
    """Demonstrate cost-performance tradeoff analysis."""
    print("\n" + "=" * 60)
    print("COST-PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Create networks
    policy_net = create_policy_network(state_dim=9, action_dim=9, config={"hidden_dims": [64, 32]})

    value_net = create_value_network(state_dim=9, config={"hidden_dims": [64, 32]})

    llm_client = MockLLMClient()

    # Run tests with different configurations
    results = []

    configurations = [
        {"name": "Pure LLM", "mode": "llm_only", "threshold": 0.0},
        {"name": "Hybrid (Low)", "mode": "auto", "threshold": 0.5},
        {"name": "Hybrid (Medium)", "mode": "auto", "threshold": 0.7},
        {"name": "Hybrid (High)", "mode": "auto", "threshold": 0.9},
        {"name": "Pure Neural", "mode": "neural_only", "threshold": 1.0},
    ]

    test_states = [torch.randn(9) for _ in range(50)]

    for cfg in configurations:
        print(f"\nTesting: {cfg['name']}")

        config = HybridConfig(mode=cfg["mode"], policy_confidence_threshold=cfg["threshold"])

        agent = HybridAgent(policy_net=policy_net, value_net=value_net, llm_client=llm_client, config=config)

        start_time = time.time()

        for state in test_states:
            await agent.select_action(state)

        elapsed_time = time.time() - start_time

        # Get final statistics
        stats = agent.get_statistics()
        savings = agent.get_cost_savings()

        result = {
            "name": cfg["name"],
            "total_cost": stats["costs"]["total"],
            "neural_pct": stats["usage"]["neural_percentage"],
            "avg_latency": elapsed_time / len(test_states) * 1000,
            "savings_pct": savings.savings_percentage,
        }

        results.append(result)

        print(f"  Total cost: ${result['total_cost']:.6f}")
        print(f"  Neural usage: {result['neural_pct']:.1f}%")
        print(f"  Avg latency: {result['avg_latency']:.2f}ms")
        print(f"  Cost savings: {result['savings_pct']:.1f}%")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Cost':>12} {'Neural%':>10} {'Latency':>12} {'Savings':>10}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['name']:<20} ${r['total_cost']:>11.6f} {r['neural_pct']:>9.1f}% "
            f"{r['avg_latency']:>11.2f}ms {r['savings_pct']:>9.1f}%"
        )


async def demo_adaptive_thresholds():
    """Demonstrate adaptive threshold adjustment."""
    print("\n" + "=" * 60)
    print("ADAPTIVE THRESHOLD DEMONSTRATION")
    print("=" * 60)

    # Create networks
    policy_net = create_policy_network(state_dim=9, action_dim=9, config={"hidden_dims": [64, 32]})

    value_net = create_value_network(state_dim=9, config={"hidden_dims": [64, 32]})

    llm_client = MockLLMClient()

    config = HybridConfig(
        mode="adaptive",
        policy_confidence_threshold=0.7,
        adaptive_threshold_window=20,
        adaptive_min_threshold=0.5,
        adaptive_max_threshold=0.95,
    )

    agent = HybridAgent(policy_net=policy_net, value_net=value_net, llm_client=llm_client, config=config)

    print("\nAdaptive threshold will adjust based on recent confidence scores...")
    print("\nProcessing 100 states...")

    test_states = [torch.randn(9) for _ in range(100)]

    for i, state in enumerate(test_states):
        action, metadata = await agent.select_action(state)

        if (i + 1) % 20 == 0:
            threshold = agent._get_confidence_threshold()
            stats = agent.get_statistics()
            print(f"\n  After {i + 1} calls:")
            print(f"    Current threshold: {threshold:.3f}")
            print(f"    Neural usage: {stats['usage']['neural_percentage']:.1f}%")
            print(f"    Recent confidences: {len(agent.recent_confidences)}")


async def main():
    """Run all hybrid agent demos."""
    print("\n" + "=" * 60)
    print("HYBRID AGENT DEMONSTRATION")
    print("Combining LLM Reasoning with Neural Efficiency")
    print("=" * 60)

    # Run demos
    await demo_hybrid_modes()
    await demo_confidence_thresholds()
    await demo_cost_performance_analysis()
    await demo_adaptive_thresholds()

    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
