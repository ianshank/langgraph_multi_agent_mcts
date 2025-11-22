
# Web Search Integration

## Overview

The LangGraph Multi-Agent MCTS framework now includes comprehensive web search capabilities, allowing agents to augment their reasoning with real-time information from the internet.

## Features

- **Multiple Search Providers**: Tavily AI, SerpAPI, and DuckDuckGo
- **RAG Integration**: Automatic storage of search results in vector databases
- **Result Synthesis**: LLM-powered synthesis of search results
- **Intelligent Caching**: Built-in caching with configurable TTL
- **Rate Limiting**: Automatic rate limiting to respect API quotas
- **Comprehensive Logging**: Detailed logging and debugging support
- **Provider-Agnostic**: Easy switching between search providers

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Configuration

Set environment variables for your chosen provider:

**Tavily (Recommended for RAG applications):**
```bash
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=your_tavily_api_key
```

**SerpAPI (Google Search):**
```bash
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=serpapi
export SERPAPI_API_KEY=your_serpapi_key
```

**DuckDuckGo (No API key required):**
```bash
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=duckduckgo
```

### 3. Basic Usage

```python
import asyncio
from src.agents.web_search_agent import WebSearchAgent
from src.adapters.web_search import create_web_search_client

async def main():
    # Create search client from settings
    search_client = create_web_search_client()

    # Create web search agent
    agent = WebSearchAgent(
        model_adapter=your_llm_adapter,  # Optional
        search_client=search_client,
        vector_store=your_vector_store,  # Optional
    )

    # Perform search
    result = await agent.process(
        "What are the latest developments in LangGraph?"
    )

    print(f"Answer: {result['response']}")
    print(f"Sources: {len(result['sources'])}")

    await agent.close()

asyncio.run(main())
```

## Search Providers

### Tavily AI Search

**Best for**: RAG applications, AI-powered search

- ✅ Optimized for AI/LLM use cases
- ✅ Built-in content extraction
- ✅ Relevance scoring
- ✅ AI-generated answers
- ❌ Requires API key (paid)

```python
from src.adapters.web_search import TavilySearchClient

client = TavilySearchClient(api_key="your_key")
response = await client.search(
    "quantum computing applications",
    max_results=5,
    include_raw_content=True,
)
```

### SerpAPI (Google Search)

**Best for**: Google Search results, rich metadata

- ✅ Access to Google Search
- ✅ Structured data extraction
- ✅ Answer boxes and knowledge graphs
- ❌ Requires API key (paid)

```python
from src.adapters.web_search import SerpAPISearchClient

client = SerpAPISearchClient(api_key="your_key")
response = await client.search(
    "climate change solutions",
    max_results=10,
    search_type="news",  # or "general"
)
```

### DuckDuckGo Search

**Best for**: Privacy, no authentication

- ✅ No API key required
- ✅ Privacy-focused
- ✅ Free to use
- ❌ Less reliable (HTML scraping)
- ❌ No raw content extraction

```python
from src.adapters.web_search import DuckDuckGoSearchClient

client = DuckDuckGoSearchClient()
response = await client.search(
    "machine learning tutorials",
    max_results=5,
)
```

## Integration with LangGraph Workflow

The web search agent integrates seamlessly into the LangGraph multi-agent workflow:

```python
from src.framework.graph import GraphBuilder, IntegratedFramework
from src.agents.web_search_agent import create_web_search_agent

# Create web search agent
web_search_agent = create_web_search_agent(
    model_adapter=model_adapter,
    logger=logger,
    vector_store=vector_store,
)

# Add to graph builder
graph_builder = GraphBuilder(
    hrm_agent=hrm_agent,
    trm_agent=trm_agent,
    web_search_agent=web_search_agent,  # Add here
    model_adapter=model_adapter,
    logger=logger,
    vector_store=vector_store,
)

# Use in workflow
result = await framework.process(
    query="What are the latest AI research trends?",
    use_web_search=True,  # Enable web search
    use_rag=True,
    use_mcts=False,
)
```

## Advanced Features

### Result Synthesis

The web search agent can synthesize results using an LLM:

```python
from src.agents.web_search_agent import WebSearchAgentConfig

config = WebSearchAgentConfig(
    max_results=5,
    enable_synthesis=True,  # Enable LLM synthesis
    include_raw_content=True,  # Include full content for RAG
)

agent = WebSearchAgent(
    model_adapter=llm_adapter,
    config=config,
)

result = await agent.process("Explain quantum entanglement")
# Returns synthesized answer from multiple sources
```

### Multi-Query Search

Search multiple related queries concurrently:

```python
queries = [
    "LangGraph architecture",
    "Multi-agent systems",
    "MCTS algorithms",
]

result = await agent.multi_query_search(queries, max_results=3)
print(f"Total unique sources: {len(result['sources'])}")
```

### Vector Storage

Automatically store search results in a vector database for RAG:

```python
config = WebSearchAgentConfig(
    enable_vector_storage=True,  # Auto-store in vector DB
)

agent = WebSearchAgent(
    vector_store=chroma_db,  # Or FAISS, Pinecone, etc.
    config=config,
)

# Results are automatically stored for future RAG queries
await agent.process("AI safety research")
```

### Caching

Built-in intelligent caching reduces API calls and costs:

```python
# First search (cache miss)
result1 = await agent.process("Python async programming")

# Second search (cache hit - instant)
result2 = await agent.process("Python async programming")

# Check cache statistics
stats = agent.stats
print(f"Cache hit rate: {stats['search_client']['cache']['hit_rate']:.1%}")
```

## Configuration Options

All configuration is done via environment variables:

```bash
# Core settings
WEB_SEARCH_ENABLED=True                      # Enable/disable web search
WEB_SEARCH_PROVIDER=tavily                   # Provider: tavily, serpapi, duckduckgo

# API Keys
TAVILY_API_KEY=your_key
SERPAPI_API_KEY=your_key

# Search parameters
WEB_SEARCH_MAX_RESULTS=5                     # Max results per search (1-20)
WEB_SEARCH_TIMEOUT_SECONDS=10                # Request timeout
WEB_SEARCH_RATE_LIMIT_PER_MINUTE=10          # Rate limit
WEB_SEARCH_CACHE_TTL_SECONDS=3600            # Cache TTL (1 hour)
WEB_SEARCH_INCLUDE_RAW_CONTENT=True          # Fetch full content
WEB_SEARCH_USER_AGENT="Your-Agent/1.0"       # User agent string

# Storage
SEARCH_RESULTS_STORAGE_ENABLED=True          # Store in vector DB
SEARCH_RESULTS_VECTOR_STORE=chroma           # Vector store type
```

## Monitoring and Debugging

### Agent Statistics

```python
stats = agent.stats
print(f"Searches: {stats['search_count']}")
print(f"Results retrieved: {stats['total_results_retrieved']}")
print(f"Avg execution time: {stats['avg_execution_time_ms']:.0f}ms")
print(f"Cache hit rate: {stats['search_client']['cache']['hit_rate']:.1%}")
```

### Logging

Enable debug logging for detailed insights:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.adapters.web_search")
logger.setLevel(logging.DEBUG)

# Now see detailed search logs
result = await agent.process("test query")
```

## Example: Complete Integration

```python
import asyncio
from src.framework.graph import IntegratedFramework
from src.agents.web_search_agent import create_web_search_agent
from src.config.settings import get_settings

async def main():
    settings = get_settings()

    # Enable web search
    settings.WEB_SEARCH_ENABLED = True

    # Create framework with web search
    framework = IntegratedFramework(
        model_adapter=model_adapter,
        logger=logger,
        vector_store=vector_store,
    )

    # Add web search agent
    framework.graph_builder.web_search_agent = create_web_search_agent(
        model_adapter=model_adapter,
        logger=logger,
        vector_store=vector_store,
    )

    # Process query with all agents
    result = await framework.process(
        query="What are the latest breakthroughs in quantum computing?",
        use_rag=True,
        use_web_search=True,  # Enable web search
        use_mcts=True,
    )

    print(f"Response: {result['response']}")
    print(f"Agents used: {result['metadata']['agents_used']}")

asyncio.run(main())
```

## Best Practices

1. **Choose the Right Provider**:
   - Use Tavily for RAG applications
   - Use SerpAPI for Google-quality results
   - Use DuckDuckGo for testing/development

2. **Enable Caching**: Reduce costs and latency with caching

3. **Set Rate Limits**: Respect API quotas with built-in rate limiting

4. **Use Vector Storage**: Store results for enhanced RAG capabilities

5. **Monitor Usage**: Track statistics to optimize performance

6. **Handle Errors**: Implement proper error handling for network issues

7. **Secure API Keys**: Use environment variables, never hardcode

## Troubleshooting

**Issue**: `WebSearchError: search_client_unavailable`
- **Solution**: Ensure WEB_SEARCH_ENABLED=True and provider is configured

**Issue**: `WebSearchAuthError: Invalid API key`
- **Solution**: Check your API key is correct and has credit

**Issue**: `WebSearchRateLimitError: Rate limit exceeded`
- **Solution**: Reduce WEB_SEARCH_RATE_LIMIT_PER_MINUTE or wait

**Issue**: DuckDuckGo returns few results
- **Solution**: This is expected with HTML scraping; use Tavily/SerpAPI for better results

## Demo and Examples

Run the included demo:

```bash
python examples/web_search_demo.py
```

See tests for more examples:
```bash
pytest tests/unit/test_web_search_agent.py -v
```

## Performance

Typical performance metrics:

- **Tavily**: 200-500ms per search
- **SerpAPI**: 300-700ms per search
- **DuckDuckGo**: 400-1000ms per search
- **Cache Hit**: <10ms

## Roadmap

Future enhancements:

- [ ] More search providers (Bing, Brave Search)
- [ ] Image search support
- [ ] Advanced filtering and ranking
- [ ] Automatic query expansion
- [ ] Result deduplication
- [ ] Multi-lingual search

## Support

For issues or questions:
- GitHub Issues: https://github.com/ianshank/langgraph_multi_agent_mcts/issues
- Documentation: See `/docs` directory

---

*Last updated: 2025-01-XX*
