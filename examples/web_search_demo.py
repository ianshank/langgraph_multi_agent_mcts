"""
Web Search Agent Demo - Example Usage

This script demonstrates how to use the Web Search Agent with different providers.

Usage:
    # With Tavily (recommended)
    export WEB_SEARCH_PROVIDER=tavily
    export TAVILY_API_KEY=your_key
    python examples/web_search_demo.py

    # With SerpAPI
    export WEB_SEARCH_PROVIDER=serpapi
    export SERPAPI_API_KEY=your_key
    python examples/web_search_demo.py

    # With DuckDuckGo (no API key required)
    export WEB_SEARCH_PROVIDER=duckduckgo
    python examples/web_search_demo.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.web_search_agent import WebSearchAgent, WebSearchAgentConfig
from src.adapters.web_search import create_web_search_client
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_basic_search():
    """Demo 1: Basic web search."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Web Search")
    print("=" * 60)

    settings = get_settings()
    settings.WEB_SEARCH_ENABLED = True

    # Create search client
    search_client = create_web_search_client(settings=settings)

    # Create agent (without LLM for simplicity)
    agent = WebSearchAgent(
        model_adapter=None,
        search_client=search_client,
        config=WebSearchAgentConfig(
            max_results=3,
            enable_synthesis=False,  # Disable LLM synthesis for demo
            enable_vector_storage=False,
        ),
    )

    # Perform search
    query = "LangGraph multi-agent systems 2024"
    print(f"\nQuery: {query}")
    print("-" * 60)

    result = await agent.process(query)

    # Display results
    print(f"\nProvider: {result['metadata']['provider']}")
    print(f"Search time: {result['metadata']['search_time_ms']:.0f}ms")
    print(f"Results found: {len(result['sources'])}")
    print("\nSources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['title']}")
        print(f"   URL: {source['url']}")
        print(f"   Relevance: {source['relevance_score']:.2f}")

    print("\n" + "=" * 60)

    await agent.close()


async def demo_multi_query():
    """Demo 2: Multi-query concurrent search."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Query Concurrent Search")
    print("=" * 60)

    settings = get_settings()
    settings.WEB_SEARCH_ENABLED = True

    search_client = create_web_search_client(settings=settings)

    agent = WebSearchAgent(
        model_adapter=None,
        search_client=search_client,
        config=WebSearchAgentConfig(
            max_results=2,
            enable_synthesis=False,
            enable_vector_storage=False,
            max_concurrent_searches=3,
        ),
    )

    # Multiple related queries
    queries = [
        "LangGraph architecture overview",
        "Multi-agent systems best practices",
        "MCTS algorithms in AI",
    ]

    print(f"\nExecuting {len(queries)} queries concurrently:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print("\n" + "-" * 60)

    result = await agent.multi_query_search(queries)

    print(f"\nQueries executed: {result['metadata']['query_count']}")
    print(f"Successful: {result['metadata']['successful_queries']}")
    print(f"Unique sources: {result['metadata']['total_sources']}")

    print("\nAll sources:")
    for i, source in enumerate(list(result['sources'])[:10], 1):
        print(f"{i}. {source['title'][:60]}... ({source['url']})")

    print("\n" + "=" * 60)

    await agent.close()


async def demo_with_caching():
    """Demo 3: Demonstrate caching."""
    print("\n" + "=" * 60)
    print("Demo 3: Search Result Caching")
    print("=" * 60)

    settings = get_settings()
    settings.WEB_SEARCH_ENABLED = True

    search_client = create_web_search_client(settings=settings)

    agent = WebSearchAgent(
        model_adapter=None,
        search_client=search_client,
        config=WebSearchAgentConfig(max_results=3, enable_synthesis=False),
    )

    query = "Python async programming guide"

    # First search (cache miss)
    print(f"\nFirst search: {query}")
    result1 = await agent.process(query)
    print(f"Cached: {result1['metadata']['cached']}")
    print(f"Search time: {result1['metadata']['search_time_ms']:.0f}ms")

    # Second search (cache hit)
    print(f"\nSecond search (same query): {query}")
    result2 = await agent.process(query)
    print(f"Cached: {result2['metadata']['cached']}")
    print(f"Search time: {result2['metadata']['search_time_ms']:.0f}ms")

    # Display cache stats
    if hasattr(search_client, "stats"):
        stats = search_client.stats
        if "cache" in stats:
            cache_stats = stats["cache"]
            print(f"\nCache statistics:")
            print(f"  Hits: {cache_stats['hits']}")
            print(f"  Misses: {cache_stats['misses']}")
            print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")

    print("\n" + "=" * 60)

    await agent.close()


async def demo_statistics():
    """Demo 4: Agent statistics."""
    print("\n" + "=" * 60)
    print("Demo 4: Agent Statistics")
    print("=" * 60)

    settings = get_settings()
    settings.WEB_SEARCH_ENABLED = True

    search_client = create_web_search_client(settings=settings)

    agent = WebSearchAgent(
        model_adapter=None,
        search_client=search_client,
        config=WebSearchAgentConfig(max_results=3, enable_synthesis=False),
    )

    # Perform multiple searches
    queries = [
        "Machine learning fundamentals",
        "Deep learning tutorials",
        "Neural networks explained",
    ]

    for query in queries:
        await agent.process(query)

    # Display statistics
    stats = agent.stats
    print("\nAgent Statistics:")
    print(f"  Total searches: {stats['search_count']}")
    print(f"  Total results: {stats['total_results_retrieved']}")
    print(f"  Avg results per search: {stats['avg_results_per_search']:.1f}")
    print(f"  Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")

    if "search_client" in stats:
        client_stats = stats["search_client"]
        print(f"\nSearch Client Statistics:")
        print(f"  Requests: {client_stats.get('request_count', 0)}")
        print(f"  Failed: {client_stats.get('failed_requests', 0)}")

    print("\n" + "=" * 60)

    await agent.close()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("WEB SEARCH AGENT DEMONSTRATIONS")
    print("=" * 60)

    try:
        settings = get_settings()
        print(f"\nProvider: {settings.WEB_SEARCH_PROVIDER}")
        print(f"Enabled: {settings.WEB_SEARCH_ENABLED}")

        # Run demos
        await demo_basic_search()
        await demo_multi_query()
        await demo_with_caching()
        await demo_statistics()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Set WEB_SEARCH_ENABLED=True")
        print("2. Set WEB_SEARCH_PROVIDER to tavily, serpapi, or duckduckgo")
        print("3. Set the appropriate API key if required")


if __name__ == "__main__":
    asyncio.run(main())
