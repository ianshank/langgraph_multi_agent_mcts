# Web Search Integration - Quick Start Guide

## 🎉 What's New

The LangGraph Multi-Agent MCTS framework now includes comprehensive web search capabilities! This allows agents to augment their reasoning with real-time information from the internet.

## 🚀 Key Features

- ✅ **Multiple Search Providers**: Tavily AI, SerpAPI, and DuckDuckGo
- ✅ **RAG Integration**: Automatic storage of search results in vector databases
- ✅ **LLM-Powered Synthesis**: Intelligent result synthesis
- ✅ **Intelligent Caching**: Built-in caching with configurable TTL
- ✅ **Rate Limiting**: Automatic rate limiting to respect API quotas
- ✅ **No Hardcoded Values**: All configuration via environment variables
- ✅ **Comprehensive Testing**: Full unit and integration test suite
- ✅ **Production Ready**: Comprehensive logging and debugging

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

New dependencies include:
- `tavily-python` - Tavily AI Search
- `google-search-results` - SerpAPI
- `duckduckgo-search` - DuckDuckGo Search
- `beautifulsoup4` - HTML parsing
- `lxml` - XML/HTML processing

## ⚙️ Configuration

### Option 1: DuckDuckGo (No API Key Required)

Perfect for testing and development:

```bash
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=duckduckgo
```

### Option 2: Tavily (Recommended for Production)

Best for RAG applications:

```bash
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=your_tavily_api_key

# Get your API key from: https://tavily.com/
```

### Option 3: SerpAPI (Google Search)

For Google Search quality:

```bash
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=serpapi
export SERPAPI_API_KEY=your_serpapi_key

# Get your API key from: https://serpapi.com/
```

### Full Configuration

See `.env.example` for all available options.

## 🎯 Usage Examples

### Basic Usage

```python
import asyncio
from src.agents.web_search_agent import WebSearchAgent
from src.adapters.web_search import create_web_search_client

async def main():
    # Create search client
    search_client = create_web_search_client()

    # Create agent
    agent = WebSearchAgent(
        search_client=search_client,
        model_adapter=your_llm_adapter,  # Optional for synthesis
    )

    # Perform search
    result = await agent.process(
        "What are the latest developments in quantum computing?"
    )

    print(f"Answer: {result['response']}")
    print(f"Sources: {len(result['sources'])}")

    await agent.close()

asyncio.run(main())
```

### Integration with LangGraph Workflow

```python
from src.framework.graph import GraphBuilder
from src.agents.web_search_agent import create_web_search_agent

# Create web search agent
web_search_agent = create_web_search_agent(
    model_adapter=model_adapter,
    logger=logger,
    vector_store=vector_store,
)

# Add to graph
graph_builder = GraphBuilder(
    hrm_agent=hrm_agent,
    trm_agent=trm_agent,
    web_search_agent=web_search_agent,
    model_adapter=model_adapter,
    logger=logger,
)

# Use in workflow
result = await framework.process(
    query="Latest AI research trends",
    use_web_search=True,
    use_rag=True,
)
```

## 🧪 Running Tests

```bash
# Run all web search tests
pytest tests/unit/test_web_search_adapters.py -v
pytest tests/unit/test_web_search_agent.py -v

# Run with coverage
pytest tests/unit/test_web_search*.py --cov=src/adapters/web_search --cov=src/agents/web_search_agent
```

## 📚 Documentation

Comprehensive documentation available in:
- `docs/web_search_integration.md` - Full integration guide
- `examples/web_search_demo.py` - Interactive demonstration

## 🎮 Demo

Run the interactive demo:

```bash
# With DuckDuckGo (no API key needed)
export WEB_SEARCH_PROVIDER=duckduckgo
python examples/web_search_demo.py

# With Tavily
export WEB_SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=your_key
python examples/web_search_demo.py
```

## 🏗️ Architecture

### New Components

```
src/adapters/web_search/
├── __init__.py              # Public API
├── base.py                  # Base classes and protocols
├── exceptions.py            # Exception hierarchy
├── tavily_client.py         # Tavily AI Search client
├── serpapi_client.py        # SerpAPI client
├── duckduckgo_client.py     # DuckDuckGo client
└── factory.py               # Client factory

src/agents/
└── web_search_agent.py      # Web Search Agent

src/config/
└── settings.py              # Updated with web search config

src/framework/
└── graph.py                 # Updated with web search node

tests/unit/
├── test_web_search_adapters.py
└── test_web_search_agent.py

examples/
└── web_search_demo.py

docs/
└── web_search_integration.md
```

### Workflow Integration

The web search agent integrates into the LangGraph workflow as a new node:

```
Entry → Retrieve Context → Route Decision → Web Search → Aggregate → Synthesize
                              ↓             ↓
                           HRM/TRM        MCTS
```

## 🔧 Troubleshooting

### Issue: "Search client not available"

**Solution**: Ensure `WEB_SEARCH_ENABLED=True` in your environment.

### Issue: "Invalid API key"

**Solution**: Check your API key is correct and has available credits.

### Issue: "Rate limit exceeded"

**Solution**: Reduce `WEB_SEARCH_RATE_LIMIT_PER_MINUTE` or wait.

### Issue: DuckDuckGo returns few results

**Solution**: This is expected with HTML scraping; use Tavily/SerpAPI for production.

## 🎯 Best Practices

1. **Use Tavily for Production**: Best results for RAG applications
2. **Enable Caching**: Reduce costs and latency
3. **Set Rate Limits**: Respect API quotas
4. **Monitor Usage**: Track statistics to optimize
5. **Secure API Keys**: Use environment variables, never hardcode

## 📊 Performance

Typical performance (cached results < 10ms):

- **Tavily**: 200-500ms per search
- **SerpAPI**: 300-700ms per search
- **DuckDuckGo**: 400-1000ms per search

## 🚀 Next Steps

1. Review full documentation: `docs/web_search_integration.md`
2. Run the demo: `python examples/web_search_demo.py`
3. Explore tests: `tests/unit/test_web_search*.py`
4. Configure for your use case: `.env.example`
5. Integrate into your workflow

## 🤝 Contributing

To extend web search functionality:

1. Add new provider in `src/adapters/web_search/`
2. Implement `WebSearchClient` protocol
3. Register in factory: `src/adapters/web_search/factory.py`
4. Add tests in `tests/unit/`
5. Update documentation

## 📝 Examples

### Multi-Query Search

```python
queries = [
    "LangGraph architecture",
    "Multi-agent systems",
    "MCTS algorithms",
]

result = await agent.multi_query_search(queries)
print(f"Total sources: {len(result['sources'])}")
```

### With Vector Storage

```python
from langchain.vectorstores import Chroma

config = WebSearchAgentConfig(
    enable_vector_storage=True,
    enable_synthesis=True,
)

agent = WebSearchAgent(
    vector_store=Chroma(...),
    config=config,
)

# Results automatically stored for future RAG queries
await agent.process("AI safety research")
```

## 🔗 Resources

- [Tavily Documentation](https://tavily.com/docs)
- [SerpAPI Documentation](https://serpapi.com/docs)
- [DuckDuckGo Search](https://duckduckgo.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## ⚡ Quick Commands

```bash
# Setup (DuckDuckGo - no API key)
export WEB_SEARCH_ENABLED=True
export WEB_SEARCH_PROVIDER=duckduckgo

# Run demo
python examples/web_search_demo.py

# Run tests
pytest tests/unit/test_web_search*.py -v

# Check coverage
pytest tests/unit/test_web_search*.py --cov=src --cov-report=html
```

## 📖 Learn More

- Full documentation: `docs/web_search_integration.md`
- Example code: `examples/web_search_demo.py`
- Test examples: `tests/unit/test_web_search*.py`
- Configuration: `.env.example`

---

**Happy Searching!** 🔍

For questions or issues, please open a GitHub issue.
