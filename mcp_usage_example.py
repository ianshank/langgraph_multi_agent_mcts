#!/usr/bin/env python3
"""
MCP Server Usage Example

This script demonstrates how to interact with the MCP server
using the standard MCP protocol over stdio.
"""

import asyncio
import json
from typing import Any


class MCPClient:
    """Simple MCP client for demonstration purposes."""

    def __init__(self):
        self.request_id = 0

    def _next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id

    def create_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create an MCP request."""
        return {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or {}}

    def format_response(self, response: dict[str, Any]) -> str:
        """Format response for display."""
        if "error" in response:
            return f"Error: {response['error']['message']}"
        elif "result" in response:
            return f"Success: {json.dumps(response['result'], indent=2)}"
        else:
            return f"Unknown response: {json.dumps(response, indent=2)}"


async def demo_mcp_tools():
    """Demonstrate MCP server tools."""
    client = MCPClient()

    print("MCP Server Usage Examples")
    print("=" * 60)

    # Example 1: List available tools
    print("\n1. Listing Available Tools:")
    request = client.create_request("tools/list")
    print(f"Request: {json.dumps(request, indent=2)}")
    print("Expected Response: List of available tools with their schemas")

    # Example 2: Run MCTS Search
    print("\n2. Running MCTS Search:")
    request = client.create_request(
        "tools/call",
        {
            "name": "run_mcts",
            "arguments": {
                "query": "How can I optimize a Python web application for better performance?",
                "iterations": 50,
                "exploration_weight": 1.414,
                "seed": 42,
                "use_rag": False,
            },
        },
    )
    print(f"Request: {json.dumps(request, indent=2)}")
    print("Expected Response: Best action and statistics from MCTS search")

    # Example 3: Query an Agent
    print("\n3. Querying HRM Agent:")
    request = client.create_request(
        "tools/call",
        {
            "name": "query_agent",
            "arguments": {
                "agent_type": "hrm",
                "query": "Break down the problem of implementing a caching system",
                "temperature": 0.7,
                "max_tokens": 500,
            },
        },
    )
    print(f"Request: {json.dumps(request, indent=2)}")
    print("Expected Response: HRM's hierarchical decomposition of the problem")

    # Example 4: Health Check
    print("\n4. Performing Health Check:")
    request = client.create_request("tools/call", {"name": "health_check", "arguments": {}})
    print(f"Request: {json.dumps(request, indent=2)}")
    print("Expected Response: Health status of framework and LLM provider")

    # Example 5: Get Configuration
    print("\n5. Getting Current Configuration:")
    request = client.create_request("tools/call", {"name": "get_config", "arguments": {}})
    print(f"Request: {json.dumps(request, indent=2)}")
    print("Expected Response: Current framework configuration settings")

    # Example 6: List Artifacts
    print("\n6. Listing Stored Artifacts:")
    request = client.create_request(
        "tools/call", {"name": "list_artifacts", "arguments": {"artifact_type": "mcts_stats", "limit": 10}}
    )
    print(f"Request: {json.dumps(request, indent=2)}")
    print("Expected Response: List of stored MCTS run artifacts")

    print("\n" + "=" * 60)
    print("Usage Notes:")
    print("- The MCP server communicates via JSON-RPC over stdio")
    print("- All tool calls use the 'tools/call' method with tool name and arguments")
    print("- Responses include either 'result' (success) or 'error' (failure)")
    print("- The server maintains state and stores artifacts from MCTS runs")
    print("\nTo use in your application:")
    print("1. Start the MCP server: python tools/mcp/server.py")
    print("2. Send JSON-RPC requests to stdin")
    print("3. Read JSON-RPC responses from stdout")
    print("4. Handle async operations appropriately")


def main():
    """Main entry point."""
    print("\nMCP Server Configuration:")
    print("- Provider: lmstudio")
    print("- Model: llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b")
    print("- Base URL: http://localhost:1234/v1")
    print("- MCTS Iterations: 100")
    print("- Exploration Weight: 1.414")

    # Run the demo
    asyncio.run(demo_mcp_tools())

    print("\nFor a working implementation, see:")
    print("- Server: tools/mcp/server.py")
    print("- Config: mcp_config.json")
    print("- LLM Adapter: src/adapters/llm.py")


if __name__ == "__main__":
    main()
