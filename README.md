# LangGraph Multi-Agent Framework with MCTS Integration

A production-ready, modular multi-agent framework combining Monte Carlo Tree Search (MCTS), hierarchical reasoning (HRM), and iterative refinement (TRM) agents with LangGraph state machine architecture.

## Features

- **Provider-Agnostic LLM Adapters**: Support for OpenAI, Anthropic, and LM Studio with unified interface
- **Deterministic MCTS**: Seeded RNG, progressive widening, simulation caching, and configurable policies
- **Async-First Architecture**: Full async/await support with parallel agent execution
- **Production Observability**: JSON logging, OpenTelemetry tracing, metrics collection, and S3 storage
- **Security & Validation**: Pydantic input validation, secrets management, and comprehensive security audit
- **CI/CD Ready**: GitHub Actions pipeline with linting, type-checking, security scanning, and coverage

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ianshank/langgraph_multi_agent_mcts.git
cd langgraph_multi_agent_mcts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Configuration

Copy the environment template and configure:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Key configuration options:
- `LLM_PROVIDER`: openai, anthropic, or lmstudio
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: API credentials
- `MCTS_ITERATIONS`: Number of MCTS simulations (default: 200)
- `MCTS_C`: Exploration weight for UCB1 (default: 1.414)
- `SEED`: Random seed for determinism (default: 42)
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR

### Basic Usage

```python
import asyncio
from src.adapters.llm import create_client
from src.config.settings import get_settings
from langgraph_multi_agent_mcts import LangGraphMultiAgentFramework
import logging

async def main():
    settings = get_settings()

    # Create provider-agnostic LLM client
    model_adapter = create_client(
        settings.LLM_PROVIDER,
        model="gpt-4-turbo-preview",  # or "claude-3-sonnet", etc.
        timeout=60.0,
        max_retries=3,
    )

    # Initialize framework
    framework = LangGraphMultiAgentFramework(
        model_adapter=model_adapter,
        logger=logging.getLogger("mcts.framework"),
        mcts_iterations=settings.MCTS_ITERATIONS,
        mcts_exploration_weight=settings.MCTS_C,
    )

    # Process query with MCTS
    result = await framework.process(
        query="Recommend defensive positions for night attack scenario",
        use_rag=True,
        use_mcts=True,
    )

    print(f"Response: {result['response']}")
    print(f"Confidence: {result['metadata']['consensus_score']:.2f}")
    print(f"MCTS Stats: {result['metadata'].get('mcts_stats', {})}")

asyncio.run(main())
```

### Using the Deterministic MCTS Engine

```python
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.config import BALANCED_CONFIG
from src.framework.mcts.policies import RandomRolloutPolicy

async def run_mcts():
    # Configuration with seed for determinism
    config = BALANCED_CONFIG.copy(seed=42)

    # Create engine
    engine = MCTSEngine(
        seed=config.seed,
        exploration_weight=config.exploration_weight,
        progressive_widening_k=config.progressive_widening_k,
    )

    # Define domain-specific functions
    def action_generator(state):
        return ["action_A", "action_B", "action_C"]

    def state_transition(state, action):
        return MCTSState(
            state_id=f"{state.state_id}_{action}",
            features=state.features.copy()
        )

    # Create root
    root = MCTSNode(
        state=MCTSState(state_id="root", features={}),
        rng=engine.rng,
    )

    # Create rollout policy
    rollout_policy = RandomRolloutPolicy()

    # Run search
    best_action, stats = await engine.search(
        root=root,
        num_iterations=100,
        action_generator=action_generator,
        state_transition=state_transition,
        rollout_policy=rollout_policy,
    )

    print(f"Best Action: {best_action}")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")

asyncio.run(run_mcts())
```

### MCP Server Integration

The framework includes a Model Context Protocol (MCP) server for tool integration:

```bash
# Start the MCP server
python3 tools/mcp/server.py
```

**Available MCP Tools:**
- `run_mcts` - Execute MCTS search with configurable parameters
- `query_agent` - Query HRM, TRM, or MCTS agents directly
- `get_artifact` - Retrieve stored search results
- `list_artifacts` - List available artifacts
- `get_config` - Get current framework configuration
- `health_check` - Check system health

**MCP Configuration (mcp_config.example.json):**
```json
{
  "mcpServers": {
    "mcts-framework": {
      "command": "python3",
      "args": ["tools/mcp/server.py"],
      "cwd": "/path/to/langgraph_multi_agent_mcts",
      "env": {
        "LLM_PROVIDER": "lmstudio",
        "LMSTUDIO_BASE_URL": "http://localhost:1234/v1"
      }
    }
  }
}
```

## Architecture Overview

The LangGraph Multi-Agent MCTS Framework is built on a state machine architecture that orchestrates multiple specialized agents to solve complex reasoning tasks. The system combines hierarchical reasoning, iterative refinement, and Monte Carlo Tree Search to provide robust, statistically-validated decision support.

### High-Level Architecture

The framework follows a **LangGraph state machine pattern** where queries flow through a series of nodes, each responsible for a specific aspect of reasoning:

1. **Entry & Context Retrieval**: Parse user query and retrieve relevant context from vector stores (RAG)
2. **Intelligent Routing**: Determine which agents to invoke based on query complexity and current state
3. **Parallel Agent Execution**: Run HRM (Hierarchical Reasoning) and TRM (Task Refinement) agents concurrently
4. **MCTS Simulation**: Perform Monte Carlo Tree Search for tactical planning and decision validation
5. **Result Aggregation**: Combine outputs from all agents with confidence scoring
6. **Consensus Evaluation**: Check if agents agree or if additional iterations are needed
7. **Synthesis**: Generate final response by synthesizing all agent outputs

### Core Components

#### 1. LangGraph State Machine (`src/framework/graph.py`)

The heart of the framework is a **StateGraph** that manages workflow execution:

- **Shared State (AgentState)**: TypedDict containing query, agent results, MCTS tree, confidence scores, and metadata
- **Nodes**: Each node is an async function that processes state and returns updates
- **Conditional Edges**: Routes queries to appropriate agents based on state analysis
- **Memory/Checkpointing**: Built-in support for conversation history and state persistence

**Workflow Flow:**
```
Entry → RAG Retrieval → Route Decision → [HRM | TRM | MCTS] → Aggregate → Evaluate → [Synthesize | Iterate]
```

#### 2. Multi-Agent System

**HRM Agent (Hierarchical Reasoning Model)**:
- Decomposes complex queries into sub-problems
- Processes sub-problems in parallel
- Provides structured, hierarchical analysis
- Best for: Multi-faceted problems requiring decomposition

**TRM Agent (Task Refinement Model)**:
- Iteratively refines solutions through multiple passes
- Applies quality scoring to each refinement
- Focuses on improving solution quality over iterations
- Best for: Tasks requiring iterative improvement

**MCTS Simulator**:
- Performs Monte Carlo Tree Search for tactical planning
- Uses UCB1 algorithm for exploration vs exploitation balance
- Simulates multiple scenarios to find optimal actions
- Best for: Decision-making under uncertainty, adversarial scenarios

#### 3. MCTS Engine (`src/framework/mcts/`)

A production-ready MCTS implementation with:

- **Deterministic Execution**: Seeded RNG ensures reproducible results
- **Progressive Widening**: Controls tree branching with `k * n^alpha` formula
- **Simulation Caching**: SHA-256 based caching for repeated states
- **Parallel Rollouts**: Bounded concurrency with asyncio.Semaphore
- **Multiple Selection Policies**: UCB1, max visits, max value, robust child

**MCTS Phases:**
1. **Selection**: Traverse tree using UCB1 to find leaf node
2. **Expansion**: Generate new child states/actions
3. **Simulation**: Evaluate state using rollout policy (can use HRM/TRM)
4. **Backpropagation**: Update ancestor node values and visit counts

#### 4. Neural Meta-Controllers (`src/agents/meta_controller/`)

Intelligent routing decisions using neural networks:

- **RNN Meta-Controller**: GRU-based model using 10D normalized feature vectors
- **BERT Meta-Controller**: Transformer-based model using text descriptions of state
- **Feature Extraction**: Converts agent state to normalized features or text
- **Vector Storage**: Stores decisions in Pinecone (10D vectors) for future training

#### 5. Provider-Agnostic LLM Adapters (`src/adapters/llm/`)

Unified interface for multiple LLM providers:

- **OpenAI Client**: GPT-4, GPT-3.5 with Chat Completions API
- **Anthropic Client**: Claude models with Messages API
- **LM Studio Client**: Local inference with OpenAI-compatible API
- **Common Features**: Retry logic, circuit breakers, rate limiting, timeout handling

#### 6. Observability Stack (`src/observability/`)

Production-ready monitoring and debugging:

- **JSON Structured Logging**: Correlation IDs, log levels, structured data
- **OpenTelemetry Tracing**: Distributed tracing with spans and context propagation
- **Prometheus Metrics**: Request counts, latencies, error rates
- **MCTS Tree Visualization**: Text and DOT format tree dumps
- **Performance Profiling**: Memory and CPU usage tracking

#### 7. Storage & Persistence (`src/storage/`)

- **S3 Client**: Async artifact storage with retry logic and exponential backoff
- **Pinecone Vector Store**: 10D feature vectors for meta-controller training data
- **Memory Checkpointing**: LangGraph's MemorySaver for conversation state

### External Systems Integration

The framework integrates with several external services:

- **LLM Providers**: OpenAI, Anthropic, LM Studio (local)
- **Vector Databases**: Pinecone (10D vectors), Chroma, FAISS
- **Experiment Tracking**: Weights & Biases, Braintrust
- **Observability**: OpenTelemetry Collector, Prometheus, Grafana
- **Storage**: AWS S3 for artifacts and model checkpoints

### User Journey

1. **User submits query** via CLI, REST API, or MCP server
2. **Framework initializes** state and loads conversation history (if available)
3. **RAG retrieval** fetches relevant context from vector store
4. **Routing decision** determines which agents to invoke (can use neural meta-controller)
5. **Agents execute** in parallel (HRM + TRM) or sequentially (MCTS)
6. **Results aggregated** with confidence scores and metadata
7. **Consensus evaluated** - if threshold not met, iterate with additional agent passes
8. **Final synthesis** combines all agent outputs into coherent response
9. **Response returned** with full metadata, confidence scores, and MCTS statistics

### Architecture Diagrams

Comprehensive C4 architecture diagrams are available in `docs/`:

- **C1 System Context**: High-level system and external dependencies
- **C2 Containers**: Major application components and their relationships
- **C3 Components**: Detailed component interactions within containers
- **C4 Sequence**: Key workflows and data flows
- **User Journey Flow**: End-to-end user interaction flow
- **Agent Workflow**: Detailed agent execution and coordination
- **Training Pipeline**: Neural meta-controller training process
- **External Systems**: Integration points with external services

Diagrams are available in multiple formats:
- **Mermaid** (`.mmd` files in `docs/mermaid/`)
- **Draw.io** (`.drawio` file in `docs/drawio/` - importable to Lucidchart)
- **PNG Images** (generated via `python scripts/export_architecture_diagrams.py`)

### Project Structure

```
langgraph_multi_agent_mcts/
├── src/
│   ├── adapters/llm/           # Provider-agnostic LLM clients
│   │   ├── base.py             # LLMClient Protocol & types
│   │   ├── openai_client.py    # OpenAI adapter with retries
│   │   ├── anthropic_client.py # Anthropic Messages API
│   │   └── lmstudio_client.py  # Local LM Studio support
│   ├── config/
│   │   └── settings.py         # Pydantic Settings v2
│   ├── framework/
│   │   ├── agents/base.py      # Async agent base class
│   │   ├── graph.py            # LangGraph wiring
│   │   └── mcts/               # MCTS core implementation
│   │       ├── core.py         # MCTSNode, MCTSEngine
│   │       ├── policies.py     # UCB1, rollout strategies
│   │       ├── config.py       # Configuration presets
│   │       └── experiments.py  # Experiment tracking
│   ├── agents/
│   │   └── meta_controller/    # Neural routing controllers
│   │       ├── rnn.py          # GRU-based meta-controller
│   │       ├── bert.py         # BERT-based meta-controller
│   │       └── utils.py        # Feature extraction utilities
│   ├── models/
│   │   └── validation.py       # Pydantic input validation
│   ├── observability/
│   │   ├── logging.py          # JSON structured logging
│   │   ├── tracing.py          # OpenTelemetry integration
│   │   ├── metrics.py          # Performance metrics
│   │   ├── debug.py            # MCTS tree visualization
│   │   └── profiling.py        # Performance profiling
│   └── storage/
│       ├── s3_client.py        # Async S3 with retries
│       └── pinecone_store.py   # Vector store integration
├── tools/
│   ├── mcp/
│   │   └── server.py           # MCP server with async tools
│   └── cli/                    # CLI entrypoints
├── training/                   # Neural meta-controller training
│   ├── agent_trainer.py        # Training orchestration
│   ├── data_pipeline.py        # Data preparation
│   └── evaluation.py           # Model evaluation
├── tests/                      # Test suite
│   ├── fixtures/               # Test fixtures
│   └── test_e2e_providers.py   # E2E provider tests
├── examples/                   # Usage examples
├── docs/                       # Documentation
│   ├── mermaid/                # Mermaid diagram sources
│   ├── drawio/                 # Draw.io diagram files
│   └── img/                    # Generated diagram images
├── scripts/
│   └── export_architecture_diagrams.py  # Diagram generation script
├── .github/workflows/ci.yml    # CI/CD pipeline
├── pyproject.toml              # Project configuration
├── mcp_config.example.json     # MCP server configuration template
└── .env.example                # Environment template
```

## Key Components

### LLM Provider Abstraction

Switch between providers without code changes:

```python
# OpenAI
client = create_client("openai", model="gpt-4")

# Anthropic
client = create_client("anthropic", model="claude-3-opus")

# Local LM Studio
client = create_client("lmstudio", base_url="http://localhost:1234/v1")
```

### MCTS Features

- **Deterministic Execution**: Same seed = identical results
- **Progressive Widening**: Control tree branching with `k * n^alpha` formula
- **Simulation Caching**: SHA-256 based caching for repeated states
- **Parallel Rollouts**: Bounded concurrency with asyncio.Semaphore
- **Multiple Policies**: UCB1, max visits, max value, robust child selection

### Observability

Enable comprehensive debugging:

```bash
export LOG_LEVEL=DEBUG
export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
```

Features:
- JSON-structured logs with correlation IDs
- OpenTelemetry traces for full request lifecycle
- MCTS tree visualization (text and DOT format)
- Performance profiling and memory tracking
- S3 artifact storage with retry logic

## Development

To ensure code quality and consistency, this project uses `pre-commit` hooks for automated linting and formatting.

### Setup Pre-commit Hooks

First, install the `pre-commit` package:

```bash
pip install pre-commit
```

Then, install the git hooks:

```bash
pre-commit install
```

Now, `ruff` will automatically lint and format your code before each commit.

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
black --check src/
```

## Security

- All API keys managed via environment variables (never hardcoded)
- Pydantic validation on all external inputs
- Query sanitization with injection detection
- Secrets redacted from logs automatically
- Security audit available in `docs/security_audit.md`

## CI/CD Pipeline

The GitHub Actions workflow includes:
1. **Ruff Linting** - Fast Python linting
2. **MyPy Type Checking** - Static type analysis
3. **Black Formatting** - Code style verification
4. **Bandit Security Scan** - SAST for vulnerabilities
5. **pip-audit** - Dependency CVE scanning
6. **Pytest with Coverage** - Unit and integration tests
7. **Codecov Integration** - Coverage reporting

## Configuration Presets

```python
from src.framework.mcts.config import MCTSConfig

# Fast exploration
config = MCTSConfig.fast(seed=42)  # 25 iterations

# Balanced (default)
config = MCTSConfig.balanced(seed=42)  # 100 iterations

# Thorough analysis
config = MCTSConfig.thorough(seed=42)  # 500 iterations
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Built on [LangGraph](https://github.com/langchain-ai/langgraph) for state machine architecture
- Inspired by 2025 research in multi-agent systems and MCTS algorithms
- Uses [OpenTelemetry](https://opentelemetry.io/) for observability
