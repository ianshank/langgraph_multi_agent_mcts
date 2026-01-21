import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def reproduce():
    try:
        from tools.mcp.server import MCPServer

        server = MCPServer()
        await server.initialize()

        # This should fail if seed is None and MCTSEngine expects int
        print("Calling run_mcts without seed...")
        result = await server.call_tool("run_mcts", {"query": "test query", "iterations": 1})
        print(f"Result: {result}")

    except Exception as e:
        print(f"Caught expected exception: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(reproduce())
