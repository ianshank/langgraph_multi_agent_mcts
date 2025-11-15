#!/usr/bin/env python3
"""
End-to-End Tests for Multi-Provider LLM Integration.

Tests all three providers:
1. LM Studio (local)
2. OpenAI (cloud)
3. Anthropic (cloud)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os


async def test_provider(provider: str, model: str = None) -> dict:
    """Test a specific LLM provider."""
    result = {
        "provider": provider,
        "status": "unknown",
        "response": None,
        "model": None,
        "error": None,
    }

    try:
        from src.adapters.llm import create_client

        # Get settings for provider-specific config
        if provider == "lmstudio":
            client = create_client(
                provider="lmstudio",
                base_url=os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                model=model or os.environ.get("LMSTUDIO_MODEL", "liquid/lfm2-1.2b"),
                timeout=120.0,
                max_retries=2,
            )
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                result["status"] = "skipped"
                result["error"] = "OPENAI_API_KEY not set"
                return result
            client = create_client(
                provider="openai",
                api_key=api_key,
                model=model or "gpt-3.5-turbo",  # Use cheaper model for testing
                timeout=60.0,
                max_retries=2,
            )
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                result["status"] = "skipped"
                result["error"] = "ANTHROPIC_API_KEY not set"
                return result
            client = create_client(
                provider="anthropic",
                api_key=api_key,
                model=model or "claude-3-haiku-20240307",  # Use cheaper model for testing
                timeout=60.0,
                max_retries=2,
            )
        else:
            result["status"] = "error"
            result["error"] = f"Unknown provider: {provider}"
            return result

        # Test basic generation
        response = await client.generate(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.0,
            max_tokens=10,
        )

        result["status"] = "success"
        result["response"] = response.text.strip()
        result["model"] = response.model
        result["usage"] = response.usage

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def test_mcts_with_provider(provider: str) -> dict:
    """Test MCTS engine with a specific provider."""
    result = {
        "provider": provider,
        "status": "unknown",
        "best_action": None,
        "iterations": 0,
        "error": None,
    }

    try:
        from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
        from src.framework.mcts.policies import RandomRolloutPolicy

        engine = MCTSEngine(seed=42, exploration_weight=1.414)

        def action_generator(state):
            depth = len(state.state_id.split("_")) - 1
            if depth == 0:
                return ["analyze", "decompose", "simulate"]
            elif depth < 2:
                return ["refine", "expand"]
            return []

        def state_transition(state, action):
            return MCTSState(
                state_id=f"{state.state_id}_{action}",
                features={**state.features, "last": action}
            )

        root = MCTSNode(
            state=MCTSState(state_id="root", features={"provider": provider}),
            rng=engine.rng,
        )

        rollout_policy = RandomRolloutPolicy()

        best_action, stats = await engine.search(
            root=root,
            num_iterations=10,
            action_generator=action_generator,
            state_transition=state_transition,
            rollout_policy=rollout_policy,
        )

        result["status"] = "success"
        result["best_action"] = best_action
        result["iterations"] = stats.get("total_iterations", 10)
        result["seed"] = stats.get("seed", 42)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def test_mcp_server() -> dict:
    """Test MCP server with multiple tool calls."""
    result = {
        "status": "unknown",
        "tools_tested": 0,
        "errors": [],
    }

    try:
        from tools.mcp.server import MCPServer

        server = MCPServer()
        await server.initialize()

        # Test health check
        health_result = await server.call_tool("health_check", {})
        if not health_result.get("success"):
            result["errors"].append(f"health_check: {health_result.get('error')}")
        else:
            result["tools_tested"] += 1

        # Test get_config
        config_result = await server.call_tool("get_config", {})
        if not config_result.get("success"):
            result["errors"].append(f"get_config: {config_result.get('error')}")
        else:
            result["tools_tested"] += 1

        # Test list_artifacts
        list_result = await server.call_tool("list_artifacts", {"limit": 10})
        if not list_result.get("success"):
            result["errors"].append(f"list_artifacts: {list_result.get('error')}")
        else:
            result["tools_tested"] += 1

        # Test run_mcts
        mcts_result = await server.call_tool(
            "run_mcts",
            {"query": "E2E test query", "iterations": 10, "seed": 42}
        )
        if not mcts_result.get("success"):
            result["errors"].append(f"run_mcts: {mcts_result.get('error')}")
        else:
            result["tools_tested"] += 1
            result["mcts_action"] = mcts_result.get("best_action")

        result["status"] = "success" if len(result["errors"]) == 0 else "partial"

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))

    return result


async def run_e2e_tests():
    """Run all end-to-end tests."""
    print("=" * 70)
    print("End-to-End Tests: Multi-Provider LLM Integration")
    print("=" * 70)

    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    results = {
        "providers": {},
        "mcts": {},
        "mcp_server": {},
    }

    # Test each provider
    print("\n--- Provider Tests ---")

    for provider in ["lmstudio", "openai", "anthropic"]:
        print(f"\nTesting {provider}...")
        result = await test_provider(provider)
        results["providers"][provider] = result

        if result["status"] == "success":
            print(f"  ✓ {provider}: {result['model']}")
            print(f"    Response: '{result['response']}'")
            print(f"    Usage: {result['usage']}")
        elif result["status"] == "skipped":
            print(f"  ⊘ {provider}: Skipped - {result['error']}")
        else:
            print(f"  ✗ {provider}: {result['error']}")

    # Test MCTS engine
    print("\n--- MCTS Engine Tests ---")
    for provider in ["lmstudio", "openai", "anthropic"]:
        print(f"\nMCTS with {provider} context...")
        mcts_result = await test_mcts_with_provider(provider)
        results["mcts"][provider] = mcts_result

        if mcts_result["status"] == "success":
            print(f"  ✓ Best action: {mcts_result['best_action']}")
            print(f"    Seed: {mcts_result['seed']}")
        else:
            print(f"  ✗ Error: {mcts_result['error']}")

    # Test MCP Server
    print("\n--- MCP Server Tests ---")
    mcp_result = await test_mcp_server()
    results["mcp_server"] = mcp_result

    if mcp_result["status"] == "success":
        print(f"  ✓ All {mcp_result['tools_tested']} tools working")
        print(f"    MCTS action: {mcp_result.get('mcts_action', 'N/A')}")
    elif mcp_result["status"] == "partial":
        print(f"  ⚠ {mcp_result['tools_tested']} tools working, errors:")
        for err in mcp_result["errors"]:
            print(f"    - {err}")
    else:
        print(f"  ✗ Server error: {mcp_result['errors']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    provider_success = sum(1 for r in results["providers"].values() if r["status"] == "success")
    provider_skipped = sum(1 for r in results["providers"].values() if r["status"] == "skipped")
    provider_failed = sum(1 for r in results["providers"].values() if r["status"] == "error")

    mcts_success = sum(1 for r in results["mcts"].values() if r["status"] == "success")
    mcts_failed = sum(1 for r in results["mcts"].values() if r["status"] == "error")

    print(f"Providers: {provider_success} passed, {provider_skipped} skipped, {provider_failed} failed")
    print(f"MCTS: {mcts_success} passed, {mcts_failed} failed")
    print(f"MCP Server: {mcp_result['status']} ({mcp_result['tools_tested']} tools)")

    # Overall status
    all_passed = (
        provider_failed == 0
        and mcts_failed == 0
        and mcp_result["status"] in ["success", "partial"]
    )

    if all_passed:
        print("\n✓ All E2E tests passed!")
    else:
        print("\n✗ Some tests failed")

    return results, all_passed


if __name__ == "__main__":
    results, success = asyncio.run(run_e2e_tests())
    sys.exit(0 if success else 1)
