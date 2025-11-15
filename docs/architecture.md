# Multi-Agent MCTS Framework Architecture

## C4 Model Diagrams

### Level 1: System Context

```mermaid
C4Context
    title System Context Diagram - Multi-Agent MCTS Framework

    Person(user, "User/Developer", "Interacts with framework via API or MCP")

    System(mcts_framework, "Multi-Agent MCTS Framework", "Provides intelligent decision support using MCTS, HRM, and TRM agents")

    System_Ext(openai, "OpenAI API", "Cloud LLM service")
    System_Ext(anthropic, "Anthropic API", "Cloud LLM service")
    System_Ext(lmstudio, "LM Studio", "Local LLM inference server")
    System_Ext(s3, "AWS S3", "Artifact storage")
    System_Ext(otel, "OpenTelemetry Collector", "Distributed tracing")

    Rel(user, mcts_framework, "Queries, configures, monitors")
    Rel(mcts_framework, openai, "Generates completions", "HTTPS/JSON")
    Rel(mcts_framework, anthropic, "Generates completions", "HTTPS/JSON")
    Rel(mcts_framework, lmstudio, "Generates completions", "HTTP/JSON")
    Rel(mcts_framework, s3, "Stores artifacts", "HTTPS")
    Rel(mcts_framework, otel, "Exports traces", "gRPC/OTLP")
```

### Level 2: Container Diagram

```mermaid
C4Container
    title Container Diagram - Multi-Agent MCTS Framework

    Person(user, "User/Developer")

    Container_Boundary(framework, "Multi-Agent MCTS Framework") {
        Container(mcp_server, "MCP Server", "Python/asyncio", "Exposes tools via Model Context Protocol")
        Container(llm_adapters, "LLM Adapters", "Python/httpx", "Provider-agnostic LLM clients")
        Container(mcts_engine, "MCTS Engine", "Python/numpy", "Monte Carlo Tree Search implementation")
        Container(agents, "Agent Framework", "Python/LangGraph", "HRM, TRM, and orchestration agents")
        Container(observability, "Observability Stack", "Python/OTel", "Logging, tracing, metrics")
        Container(config, "Configuration", "Pydantic Settings", "Environment and settings management")
        Container(storage, "Storage Client", "Python/aioboto3", "S3 artifact persistence")
    }

    System_Ext(openai, "OpenAI API")
    System_Ext(anthropic, "Anthropic API")
    System_Ext(lmstudio, "LM Studio")
    System_Ext(s3, "AWS S3")
    System_Ext(otel_collector, "OTel Collector")

    Rel(user, mcp_server, "JSON-RPC calls", "stdio/TCP")
    Rel(mcp_server, mcts_engine, "Executes searches")
    Rel(mcp_server, agents, "Queries agents")
    Rel(mcp_server, config, "Reads settings")

    Rel(agents, llm_adapters, "Generates text")
    Rel(mcts_engine, agents, "Simulates rollouts")

    Rel(llm_adapters, openai, "HTTPS")
    Rel(llm_adapters, anthropic, "HTTPS")
    Rel(llm_adapters, lmstudio, "HTTP")

    Rel(observability, otel_collector, "gRPC")
    Rel(storage, s3, "HTTPS")

    Rel(agents, observability, "Logs, traces")
    Rel(mcts_engine, observability, "Metrics")
    Rel(mcts_engine, storage, "Saves artifacts")
```

### Level 3: Component Diagram - MCTS Engine

```mermaid
C4Component
    title Component Diagram - MCTS Engine

    Container_Boundary(mcts, "MCTS Engine") {
        Component(engine, "MCTSEngine", "Class", "Orchestrates MCTS phases with seeded RNG")
        Component(node, "MCTSNode", "Dataclass", "Tree node with state, visits, value")
        Component(state, "MCTSState", "Dataclass", "Hashable state representation")
        Component(policies, "Policies", "Module", "UCB1, rollout, selection strategies")
        Component(config, "MCTSConfig", "Dataclass", "Configuration with presets")
        Component(cache, "SimulationCache", "Dict", "SHA-256 keyed result cache")
        Component(rng, "RandomGenerator", "numpy.random.Generator", "Seeded deterministic RNG")
    }

    Container(agents, "Agent Framework")
    Container(storage, "Storage Client")
    Container(observability, "Observability")

    Rel(engine, node, "Creates, traverses")
    Rel(engine, policies, "Applies UCB1, rollout")
    Rel(engine, config, "Reads parameters")
    Rel(engine, cache, "Caches simulations")
    Rel(engine, rng, "Generates random values")
    Rel(node, state, "Contains")
    Rel(engine, agents, "Evaluates states")
    Rel(engine, storage, "Persists results")
    Rel(engine, observability, "Records metrics")
```

### Level 3: Component Diagram - LLM Adapters

```mermaid
C4Component
    title Component Diagram - LLM Provider Abstraction

    Container_Boundary(adapters, "LLM Adapters") {
        Component(protocol, "LLMClient Protocol", "Protocol", "Structural typing interface")
        Component(base, "BaseLLMClient", "ABC", "Common retry, timeout logic")
        Component(openai_client, "OpenAIClient", "Class", "OpenAI Chat Completions API")
        Component(anthropic_client, "AnthropicClient", "Class", "Anthropic Messages API")
        Component(lmstudio_client, "LMStudioClient", "Class", "OpenAI-compatible local API")
        Component(factory, "ClientFactory", "Function", "Creates client from config")
        Component(circuit_breaker, "CircuitBreaker", "Class", "Failure protection pattern")
        Component(exceptions, "Exceptions", "Module", "Structured error types")
    }

    Container(config, "Configuration")
    System_Ext(openai, "OpenAI API")
    System_Ext(anthropic, "Anthropic API")
    System_Ext(lmstudio, "LM Studio")

    Rel(factory, config, "Reads provider settings")
    Rel(factory, openai_client, "Creates")
    Rel(factory, anthropic_client, "Creates")
    Rel(factory, lmstudio_client, "Creates")

    Rel(openai_client, base, "Extends")
    Rel(anthropic_client, base, "Extends")
    Rel(lmstudio_client, base, "Extends")

    Rel(base, protocol, "Implements")
    Rel(base, circuit_breaker, "Uses for resilience")
    Rel(base, exceptions, "Throws on errors")

    Rel(openai_client, openai, "HTTPS/JSON")
    Rel(anthropic_client, anthropic, "HTTPS/JSON")
    Rel(lmstudio_client, lmstudio, "HTTP/JSON")
```

### Level 3: Component Diagram - MCP Server

```mermaid
C4Component
    title Component Diagram - MCP Server

    Container_Boundary(mcp, "MCP Server") {
        Component(server, "MCPServer", "Class", "Main server handling JSON-RPC requests")
        Component(tools, "Tool Registry", "Dict", "Maps tool names to handlers")
        Component(validation, "Input Validators", "Pydantic", "Validates tool arguments")
        Component(artifacts, "Artifact Store", "Dict", "In-memory result storage")
        Component(run_mcts, "run_mcts", "Async Handler", "Executes MCTS search")
        Component(query_agent, "query_agent", "Async Handler", "Queries specific agents")
        Component(health_check, "health_check", "Async Handler", "System health monitoring")
        Component(stdio, "STDIO Transport", "asyncio", "JSON-RPC over stdin/stdout")
    }

    Container(mcts_engine, "MCTS Engine")
    Container(llm_adapters, "LLM Adapters")
    Container(config, "Configuration")
    Person(client, "MCP Client")

    Rel(client, stdio, "JSON-RPC 2.0")
    Rel(stdio, server, "Routes requests")
    Rel(server, tools, "Dispatches to handler")
    Rel(server, validation, "Validates inputs")

    Rel(run_mcts, mcts_engine, "Executes search")
    Rel(run_mcts, artifacts, "Stores results")
    Rel(query_agent, llm_adapters, "Generates response")
    Rel(health_check, config, "Checks settings")
    Rel(health_check, llm_adapters, "Verifies connectivity")
```

## Data Flow Diagrams

### MCTS Search Flow

```mermaid
sequenceDiagram
    participant Client
    participant MCPServer
    participant MCTSEngine
    participant Policy
    participant Cache
    participant RNG

    Client->>MCPServer: run_mcts(query, iterations, seed)
    MCPServer->>MCTSEngine: search(root, iterations)

    loop For each iteration
        MCTSEngine->>Policy: select(node) via UCB1
        Policy-->>MCTSEngine: best_child

        alt Should expand
            MCTSEngine->>RNG: generate action
            RNG-->>MCTSEngine: random action
            MCTSEngine->>MCTSEngine: expand(node, action)
        end

        MCTSEngine->>Cache: check(state_hash)
        alt Cache hit
            Cache-->>MCTSEngine: cached_value
        else Cache miss
            MCTSEngine->>Policy: simulate(state)
            Policy-->>MCTSEngine: value
            MCTSEngine->>Cache: store(state_hash, value)
        end

        MCTSEngine->>MCTSEngine: backpropagate(value)
    end

    MCTSEngine-->>MCPServer: best_action, stats
    MCPServer->>MCPServer: store_artifact(results)
    MCPServer-->>Client: {success, best_action, artifact_id}
```

### Provider Selection Flow

```mermaid
flowchart TD
    A[Load .env] --> B{LLM_PROVIDER}
    B -->|openai| C[Create OpenAIClient]
    B -->|anthropic| D[Create AnthropicClient]
    B -->|lmstudio| E[Create LMStudioClient]

    C --> F[Validate OPENAI_API_KEY]
    D --> G[Validate ANTHROPIC_API_KEY]
    E --> H[Validate LMSTUDIO_BASE_URL]

    F --> I[Configure httpx with retries]
    G --> I
    H --> I

    I --> J[Apply circuit breaker]
    J --> K[Return LLMClient instance]

    K --> L{generate() called}
    L --> M[Build request]
    M --> N[Send with timeout]
    N --> O{Success?}

    O -->|Yes| P[Parse response]
    O -->|No| Q{Retry?}
    Q -->|Yes| N
    Q -->|No| R[Raise LLMClientError]

    P --> S[Return LLMResponse]
```

## Deployment Architecture

```mermaid
C4Deployment
    title Deployment Diagram - Production Environment

    Deployment_Node(dev_machine, "Developer Machine", "macOS/Linux") {
        Container(venv, "Python Virtual Environment", "Python 3.11+")
        Container(mcp_client, "MCP Client", "Claude Desktop / IDE")
    }

    Deployment_Node(lmstudio_server, "LM Studio Server", "Local Network") {
        Container(lmstudio, "LM Studio", "liquid/lfm2-1.2b model")
    }

    Deployment_Node(cloud, "Cloud Services", "Internet") {
        Container(openai_api, "OpenAI API", "gpt-3.5-turbo, gpt-4")
        Container(anthropic_api, "Anthropic API", "claude-3-haiku, sonnet, opus")
        Container(aws_s3, "AWS S3", "Artifact storage bucket")
        Container(otel_backend, "Observability Backend", "Jaeger/Prometheus")
    }

    Rel(mcp_client, venv, "JSON-RPC over stdio")
    Rel(venv, lmstudio, "HTTP/REST", "localhost:1234")
    Rel(venv, openai_api, "HTTPS/REST")
    Rel(venv, anthropic_api, "HTTPS/REST")
    Rel(venv, aws_s3, "HTTPS/S3 Protocol")
    Rel(venv, otel_backend, "gRPC/OTLP")
```

## Security Architecture

```mermaid
flowchart LR
    subgraph External["External Input"]
        A[User Query]
        B[MCP Request]
        C[Config File]
    end

    subgraph Validation["Input Validation Layer"]
        D[Pydantic Models]
        E[Query Sanitization]
        F[Parameter Bounds]
        G[Injection Detection]
    end

    subgraph Secrets["Secrets Management"]
        H[SecretStr Protection]
        I[.env File]
        J[No Hardcoded Values]
    end

    subgraph Network["Network Security"]
        K[HTTPS Enforcement]
        L[Timeout Controls]
        M[Retry Limits]
        N[Circuit Breaker]
    end

    subgraph Logging["Safe Logging"]
        O[JSON Structured]
        P[Secret Redaction]
        Q[Correlation IDs]
    end

    A --> D
    B --> D
    C --> H

    D --> E
    D --> F
    D --> G

    H --> I
    I --> J

    E --> K
    F --> L
    G --> M

    K --> N
    L --> O
    M --> P
    N --> Q
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | Python 3.11+ | Core language with async support |
| **Web Framework** | None (pure asyncio) | Lightweight, no overhead |
| **HTTP Client** | httpx | Async HTTP with connection pooling |
| **Validation** | Pydantic v2 | Type-safe configuration and input validation |
| **Configuration** | pydantic-settings | Environment variable management |
| **State Machine** | LangGraph | Agent orchestration and graph workflows |
| **Tracing** | OpenTelemetry SDK | Distributed tracing and metrics |
| **Storage** | aioboto3 | Async AWS S3 operations |
| **Testing** | pytest + pytest-asyncio | Async test support |
| **Linting** | ruff | Fast Python linting |
| **Type Checking** | mypy | Static type analysis |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
