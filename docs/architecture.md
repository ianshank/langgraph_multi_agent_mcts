# Architecture Overview (C4 Views + Neural Network Diagrams)

This document provides a high-level C4 model (Context, Containers, Components, and a key Sequence) plus detailed neural network diagrams for the meta-controller implementations.

## System Context (C1)

Mermaid:

```mermaid
graph TB
  user[Developer / Operator]:::person
  system[LangGraph Multi‑Agent MCTS Framework]:::system

  subgraph External Systems
    openai[OpenAI API]
    anthropic[Anthropic API]
    lmstudio[LM Studio (Local LLM)]
    pinecone[(Pinecone Vector DB)]
    braintrust[Braintrust]
    wandb[Weights & Biases]
    s3[(S3 Object Storage)]
  end

  user -- runs demos/tests/training --> system
  system -- prompts/completions --> openai
  system -- prompts/completions --> anthropic
  system -- prompts/completions --> lmstudio
  system -- upsert/query 10D vectors --> pinecone
  system -- experiments/metrics --> braintrust
  system -- runs/metrics/artifacts --> wandb
  system -- model checkpoints/artifacts --> s3

  classDef person fill:#ffd,stroke:#333
  classDef system fill:#bdf,stroke:#333
```

Image export:
![C1 System Context](./img/c1_system_context.png)

## Containers (C2)

Mermaid:

```mermaid
graph TB
  cli[CLI/Demos/Tests\n- demos/neural_meta_controller_demo.py\n- tests/*]:::app
  training[Training Pipelines\n- src/training/train_rnn.py\n- src/training/train_bert_lora.py]:::app
  framework[Core Framework\n- src/framework/*]:::core
  agents[Meta‑Controllers\n- src/agents/meta_controller/*]:::core
  adapters[LLM Adapters\n- src/adapters/llm/*]:::core
  storage[Storage\n- src/storage/s3_client.py\n- src/storage/pinecone_store.py]:::infra
  observ[Observability\n- src/observability/*]:::infra
  config[Configuration\n- src/config/settings.py]:::infra
  tools[MCP Server\n- tools/mcp/server.py]:::app

  cli --> agents
  cli --> training
  training --> agents
  agents --> adapters
  agents --> storage
  agents --> observ
  framework --- agents
  framework --- adapters
  framework --- storage
  framework --- observ
  training --> observ
  config --> agents
  config --> adapters
  config --> storage
  config --> observ
  tools --> framework

  classDef app fill:#e7f7ff,stroke:#333
  classDef core fill:#d7ffd7,stroke:#333
  classDef infra fill:#fff1cc,stroke:#333
```

Image export:
![C2 Containers](./img/c2_containers.png)

## Components (C3)

Mermaid:

```mermaid
graph TB
  subgraph Meta‑Controller
    utils[utils.py\nnormalize_features(10D)\nfeatures_to_text()]:::code
    rnn[RNNMetaController\nGRU(hidden_dim, num_layers)\nDropout -> Linear(3)]:::code
    bert[BERTMetaController\nHF AutoModelForSeqCls\nOptional LoRA(r=4, alpha=16)]:::code
  end

  subgraph Integrations
    pstore[PineconeVectorStore\nVECTOR_DIMENSION=10\nupsert/query in namespace]:::code
    btracker[BraintrustTracker\nexperiment->log_*->summarize]:::code
    s3c[S3StorageClient (async, retries)]:::code
    llm[Adapters: OpenAI, Anthropic, LM Studio]:::code
  end

  utils --> rnn
  utils --> bert
  rnn --> pstore
  bert --> pstore
  rnn --> btracker
  bert --> btracker
  training[Training (RNN/BERT)] --> rnn
  training --> btracker
  rnn --> llm
  bert --> llm
  checkpoints((Checkpoints)) --> s3c

  classDef code fill:#f6f6ff,stroke:#333
```

Image export:
![C3 Components](./img/c3_components.png)

## Key Sequence (C4)

Mermaid:

```mermaid
sequenceDiagram
  participant App as Demo/Framework
  participant MC as Meta‑Controller (RNN/BERT)
  participant LLM as LLM Adapter
  participant PC as Pinecone
  participant BT as Braintrust

  App->>MC: features (state) or text (BERT)
  MC->>LLM: (optional) prompt for feature extraction/aux
  LLM-->>MC: completion
  MC-->>App: MetaControllerPrediction(agent, confidence, probs)

  App->>PC: upsert 10D vector + metadata (agent, probs, len, iter)
  App->>BT: log hyperparams/epoch or prediction
  Note over PC,BT: buffering when not configured
```

Image export:
![C4 Sequence](./img/c4_sequence.png)

## Neural Network Diagrams (Generated via Matplotlib)

- RNN Meta‑Controller (GRU + Dropout + Linear → 3 agents)

![RNN Meta‑Controller](./img/rnn_meta_controller.png)

- BERT Meta‑Controller (Tokenizer → BERT Sequence Classification → Softmax)
  - Optional LoRA adapters: r=4, α=16, dropout=0.1, target_modules=["query","value"]

![BERT Meta‑Controller](./img/bert_meta_controller.png)

## Notes

- Feature vector is fixed 10‑D and normalized to [0,1] (see `src/agents/meta_controller/utils.py`).
- Pinecone index must be configured with dimension 10 (cosine recommended) and can use namespace per environment.
- Training integrates optional Braintrust experiment tracking and W&B for visualization.

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

## Production Architecture

### REST API Endpoints

The production REST API provides secure, authenticated access to the Multi-Agent MCTS Framework. Built on FastAPI with OpenAPI 3.1 compliance.

#### Endpoint Summary

| Endpoint | Method | Authentication | Description |
|----------|--------|----------------|-------------|
| `/health` | GET | None | Liveness probe for load balancers |
| `/ready` | GET | None | Readiness probe for Kubernetes |
| `/metrics` | GET | None | Prometheus metrics scrape target |
| `/query` | POST | API Key Required | Main reasoning endpoint |
| `/stats` | GET | API Key Required | Client usage statistics |
| `/docs` | GET | None | Swagger UI documentation |
| `/redoc` | GET | None | ReDoc API documentation |

#### `/health` - Liveness Probe

**Request:**
```http
GET /health HTTP/1.1
Host: api.example.com
```

**Response Schema:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00.000000",
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Service unhealthy

#### `/ready` - Readiness Probe

**Request:**
```http
GET /ready HTTP/1.1
Host: api.example.com
```

**Response Schema:**
```json
{
  "ready": true,
  "checks": {
    "imports_available": true,
    "authenticator_configured": true,
    "llm_client_available": true,
    "prometheus_available": true
  }
}
```

**Status Codes:**
- `200 OK` - Service ready to accept traffic
- `503 Service Unavailable` - Service not ready (fails K8s readiness check)

#### `/query` - Main Reasoning Endpoint

**Request:**
```http
POST /query HTTP/1.1
Host: api.example.com
Content-Type: application/json
X-API-Key: your-api-key-here

{
  "query": "Recommend defensive positions for night attack scenario",
  "use_mcts": true,
  "use_rag": true,
  "mcts_iterations": 200,
  "thread_id": "session_123"
}
```

**Request Schema (QueryRequest):**
```typescript
{
  query: string;           // Required: 1-10000 characters
  use_mcts?: boolean;      // Default: true - Enable MCTS simulation
  use_rag?: boolean;       // Default: true - Enable RAG retrieval
  mcts_iterations?: number; // Optional: 1-10000, override default
  thread_id?: string;      // Optional: pattern ^[a-zA-Z0-9_-]+$, max 100 chars
}
```

**Response Schema (QueryResponse):**
```json
{
  "response": "Based on analysis by multiple agents...",
  "confidence": 0.85,
  "agents_used": ["hrm", "trm", "mcts"],
  "mcts_stats": {
    "iterations": 200,
    "best_action": "recommended_action",
    "root_visits": 200
  },
  "processing_time_ms": 1523.45,
  "metadata": {
    "client_id": "client_0",
    "thread_id": "session_123",
    "rag_enabled": true
  }
}
```

**Status Codes:**
- `200 OK` - Query processed successfully
- `400 Bad Request` - Validation error (invalid input)
- `401 Unauthorized` - Missing or invalid API key
- `429 Too Many Requests` - Rate limit exceeded (includes `Retry-After` header)
- `500 Internal Server Error` - Framework processing error

#### `/stats` - Client Usage Statistics

**Request:**
```http
GET /stats HTTP/1.1
Host: api.example.com
X-API-Key: your-api-key-here
```

**Response Schema:**
```json
{
  "client_id": "client_0",
  "roles": ["user"],
  "total_requests_today": 150,
  "requests_last_hour": 25,
  "requests_last_minute": 3,
  "rate_limits": {
    "per_minute": 60,
    "per_hour": 1000,
    "per_day": 10000
  }
}
```

#### `/metrics` - Prometheus Metrics

Returns Prometheus text format metrics for scraping:

```
# HELP mcts_requests_total Total number of requests
# TYPE mcts_requests_total counter
mcts_requests_total{method="POST",endpoint="/query",status="200"} 1523

# HELP mcts_request_duration_seconds Request latency in seconds
# TYPE mcts_request_duration_seconds histogram
mcts_request_duration_seconds_bucket{method="POST",endpoint="/query",le="1.0"} 1200

# HELP mcts_active_requests Number of active requests
# TYPE mcts_active_requests gauge
mcts_active_requests 5

# HELP mcts_errors_total Total number of errors
# TYPE mcts_errors_total counter
mcts_errors_total{error_type="validation"} 23
```

#### Authentication Flow

The API uses API key authentication with SHA-256 hashing for secure key storage:

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI Server
    participant Auth as APIKeyAuthenticator
    participant RateLimit as Rate Limiter

    Client->>API: POST /query with X-API-Key header
    API->>Auth: verify_api_key(api_key)
    Auth->>Auth: SHA-256 hash key
    Auth->>Auth: Lookup hash in _key_to_client

    alt Key not found
        Auth-->>API: AuthenticationError
        API-->>Client: 401 Unauthorized
    else Key valid
        Auth->>RateLimit: _check_rate_limit(client_id)

        alt Rate limit exceeded
            RateLimit-->>Auth: RateLimitError
            Auth-->>API: RateLimitError
            API-->>Client: 429 Too Many Requests + Retry-After header
        else Within limits
            RateLimit-->>Auth: OK
            Auth-->>API: ClientInfo(client_id, roles)
            API->>API: Process request
            API-->>Client: 200 OK with response
        end
    end
```

**Rate Limiting Tiers:**
- **Burst Limit**: 100 requests/second (prevents abuse)
- **Per Minute**: 60 requests (standard flow control)
- **Per Hour**: 1000 requests (sustained usage)
- **Per Day**: 10000 requests (daily quota)

**ClientInfo Structure:**
```python
@dataclass
class ClientInfo:
    client_id: str                           # Unique client identifier
    roles: Set[str] = {"user"}               # Default role set
    created_at: datetime = datetime.utcnow() # Registration timestamp
    last_access: datetime = datetime.utcnow()# Last activity
    request_count: int = 0                   # Total requests made
```

---

### Data Models

This section documents the core data structures used throughout the framework.

#### AgentState TypedDict

The central state object for LangGraph workflow management:

```python
class AgentState(TypedDict):
    """Shared state for LangGraph agent framework."""

    # Input fields
    query: str                                    # User query to process
    use_mcts: bool                                # Enable MCTS simulation
    use_rag: bool                                 # Enable RAG retrieval

    # RAG context (optional fields)
    rag_context: NotRequired[str]                 # Retrieved context string
    retrieved_docs: NotRequired[List[Dict]]       # Document metadata

    # Agent results
    hrm_results: NotRequired[Dict]                # HRM agent output
    trm_results: NotRequired[Dict]                # TRM agent output
    agent_outputs: Annotated[List[Dict], operator.add]  # Aggregated outputs

    # MCTS simulation
    mcts_root: NotRequired[MCTSNode]              # Root of MCTS tree
    mcts_iterations: NotRequired[int]             # Iterations performed
    mcts_best_action: NotRequired[str]            # Selected action
    mcts_stats: NotRequired[Dict]                 # Simulation statistics

    # Evaluation
    confidence_scores: NotRequired[Dict[str, float]]  # Per-agent scores
    consensus_reached: NotRequired[bool]          # Consensus flag
    consensus_score: NotRequired[float]           # Aggregated score

    # Control flow
    iteration: int                                # Current iteration number
    max_iterations: int                           # Maximum allowed iterations

    # Output
    final_response: NotRequired[str]              # Synthesized response
    metadata: NotRequired[Dict]                   # Response metadata
```

**Agent Output Structure:**
```python
{
    "agent": "hrm" | "trm" | "mcts",
    "response": str,          # Agent's textual response
    "confidence": float       # 0.0 to 1.0 confidence score
}
```

#### MCTSNode Data Structure

Two implementations exist: a simplified version in the main framework and a production version in `src/framework/mcts/core.py`.

**Simplified MCTSNode (langgraph_multi_agent_mcts.py):**
```python
class MCTSNode:
    state_id: str                           # Unique state identifier
    parent: Optional[MCTSNode]              # Parent node reference
    action: Optional[str]                   # Action taken to reach this node
    children: List[MCTSNode]                # Child nodes
    visits: int                             # Visit count
    value: float                            # Cumulative value
    terminal: bool                          # Terminal state flag

    def ucb1(exploration_weight: float = 1.414) -> float
    def best_child() -> Optional[MCTSNode]
    def add_child(action: str, state_id: str) -> MCTSNode
```

**Production MCTSNode (src/framework/mcts/core.py):**
```python
@dataclass
class MCTSState:
    """Hashable state representation for caching."""
    state_id: str
    features: Dict[str, Any]

    def to_hash_key() -> str  # SHA-256 hash for cache key

class MCTSNode:
    state: MCTSState                        # Hashable state object
    parent: Optional[MCTSNode]
    action: Optional[str]
    children: List[MCTSNode]
    visits: int
    value_sum: float                        # Sum of backpropagated values
    terminal: bool
    expanded_actions: set                   # Actions already expanded
    available_actions: List[str]            # All possible actions
    _rng: np.random.Generator               # Seeded RNG for determinism

    @property
    def value() -> float                    # Average value (value_sum/visits)
    @property
    def is_fully_expanded() -> bool
    def select_child(exploration_weight) -> MCTSNode
    def add_child(action, child_state) -> MCTSNode
    def get_unexpanded_action() -> Optional[str]
```

**MCTSEngine Statistics Output:**
```python
{
    "iterations": int,                      # Total iterations run
    "root_visits": int,                     # Root node visit count
    "root_value": float,                    # Root average value
    "num_children": int,                    # Direct children of root
    "best_action": str,                     # Selected best action
    "best_action_visits": int,              # Visits to best action
    "best_action_value": float,             # Value of best action
    "action_stats": Dict[str, {             # Per-action statistics
        "visits": int,
        "value": float,
        "value_sum": float,
        "num_children": int
    }],
    "total_simulations": int,               # Total simulations performed
    "cache_hits": int,                      # Cache hit count
    "cache_misses": int,                    # Cache miss count
    "cache_hit_rate": float,                # Cache efficiency ratio
    "seed": int                             # Random seed used
}
```

#### Vector Storage Schema (Pinecone 10D)

The neural meta-controller uses a fixed 10-dimensional feature vector for Pinecone vector storage:

**Feature Vector Structure (10 dimensions):**
```python
[
    hrm_confidence,        # [0] HRM agent confidence (0.0-1.0)
    trm_confidence,        # [1] TRM agent confidence (0.0-1.0)
    mcts_value,            # [2] MCTS simulation value (0.0-1.0)
    consensus_score,       # [3] Agent consensus score (0.0-1.0)
    last_agent_hrm,        # [4] One-hot: last agent was HRM (0 or 1)
    last_agent_trm,        # [5] One-hot: last agent was TRM (0 or 1)
    last_agent_mcts,       # [6] One-hot: last agent was MCTS (0 or 1)
    iteration_normalized,  # [7] Current iteration / 20 (0.0-1.0)
    query_length_normalized, # [8] Query length / 10000 (0.0-1.0)
    has_rag_context        # [9] Binary: RAG context available (0 or 1)
]
```

**MetaControllerFeatures Input Structure:**
```python
@dataclass
class MetaControllerFeatures:
    hrm_confidence: float       # 0.0-1.0
    trm_confidence: float       # 0.0-1.0
    mcts_value: float           # 0.0-1.0
    consensus_score: float      # 0.0-1.0
    last_agent: str             # 'hrm', 'trm', 'mcts', or 'none'
    iteration: int              # Current iteration number
    query_length: int           # Character count
    has_rag_context: bool       # Whether RAG context is available
```

**Pinecone Upsert Metadata:**
```python
{
    "agent": str,               # Selected agent name
    "probs": List[float],       # Probability distribution over agents
    "query_length": int,        # Original query length
    "iteration": int,           # Iteration when decision was made
    "timestamp": str            # ISO format timestamp
}
```

**Pinecone Configuration Requirements:**
- **Dimension**: 10 (fixed)
- **Metric**: Cosine similarity (recommended)
- **Namespace**: Per-environment separation (e.g., "production", "staging")
- **Index Type**: Serverless or Pod-based

#### API Request/Response Models

**QueryRequest (Pydantic Model):**
```python
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User query to process"
    )
    use_mcts: bool = Field(default=True)
    use_rag: bool = Field(default=True)
    mcts_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000
    )
    thread_id: Optional[str] = Field(
        default=None,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$'
    )
```

**QueryResponse (Pydantic Model):**
```python
class QueryResponse(BaseModel):
    response: str                           # Synthesized final response
    confidence: float                       # 0.0-1.0 overall confidence
    agents_used: List[str]                  # Contributing agents
    mcts_stats: Optional[Dict[str, Any]]    # MCTS simulation stats
    processing_time_ms: float               # Total processing time
    metadata: Dict[str, Any]                # Additional context
```

**ErrorResponse (Pydantic Model):**
```python
class ErrorResponse(BaseModel):
    error: bool = True
    error_code: str                         # Machine-readable code
    message: str                            # Human-readable message
    timestamp: str                          # ISO format timestamp
```

**Custom Exception Hierarchy:**
```
FrameworkError (base)
├── ValidationError      # Input validation failures
├── AuthenticationError  # API key validation failures
├── AuthorizationError   # Permission denied
├── RateLimitError       # Rate limit exceeded
├── LLMError             # LLM provider failures
├── MCTSError            # MCTS simulation failures
├── RAGError             # Vector retrieval failures
├── TimeoutError         # Operation timeouts
└── ConfigurationError   # Configuration errors
```

---

### Configuration Architecture

The framework uses a layered configuration system with Pydantic Settings v2 for type-safe environment variable management.

#### Environment Variable Hierarchy

Configuration is loaded in priority order (highest to lowest):

1. **Explicit Environment Variables** - Set in shell/container
2. **`.env` File** - Project-level configuration file
3. **Default Values** - Defined in Settings class
4. **Computed Defaults** - Validated at runtime

**Loading Mechanism:**
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",              # Load from .env file
        env_file_encoding="utf-8",    # UTF-8 encoding
        case_sensitive=True,          # Exact case matching
        extra="ignore",               # Ignore unknown variables
        validate_default=True,        # Validate defaults on load
    )
```

#### Core Configuration Categories

**LLM Provider Configuration:**
```bash
LLM_PROVIDER=openai|anthropic|lmstudio  # Default: openai
OPENAI_API_KEY=sk-...                    # Required if provider=openai
ANTHROPIC_API_KEY=sk-ant-...             # Required if provider=anthropic
LMSTUDIO_BASE_URL=http://localhost:1234/v1  # Required if provider=lmstudio
LMSTUDIO_MODEL=liquid/lfm2-1.2b         # Optional model identifier
```

**MCTS Configuration:**
```bash
MCTS_ITERATIONS=100          # Default: 100, Range: 1-10000
MCTS_C=1.414                 # Default: 1.414, Range: 0.0-10.0 (UCB1 constant)
SEED=42                      # Optional: Random seed for reproducibility
```

**Integration Services:**
```bash
BRAINTRUST_API_KEY=bt-...    # Optional: Experiment tracking
PINECONE_API_KEY=pc-...      # Optional: Vector storage
PINECONE_HOST=https://index-name.svc.environment.pinecone.io
```

**Storage Configuration:**
```bash
S3_BUCKET=my-bucket-name     # Optional: Artifact storage
S3_PREFIX=mcts-artifacts     # Default: mcts-artifacts
S3_REGION=us-east-1          # Default: us-east-1
```

**Network & Security:**
```bash
HTTP_TIMEOUT_SECONDS=30              # Range: 1-300
HTTP_MAX_RETRIES=3                   # Range: 0-10
MAX_QUERY_LENGTH=10000               # Range: 1-100000
RATE_LIMIT_REQUESTS_PER_MINUTE=60    # Range: 1-1000
```

**Observability:**
```bash
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL  # Default: INFO
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4317  # OpenTelemetry
```

#### Settings.py Integration

The `src/config/settings.py` module provides:

**1. Type-Safe Loading:**
```python
from src.config.settings import get_settings, Settings

settings = get_settings()  # Singleton pattern, cached after first load
```

**2. Secret Protection:**
```python
# API keys use SecretStr to prevent accidental exposure
settings.OPENAI_API_KEY        # Returns SecretStr object
settings.get_api_key()         # Returns actual string value (use carefully)
settings.safe_dict()           # Returns dict with secrets masked as ***MASKED***
```

**3. Provider Validation:**
```python
@model_validator(mode="after")
def validate_provider_credentials(self) -> "Settings":
    """Ensure required API keys are provided for selected provider."""
    if self.LLM_PROVIDER == LLMProvider.OPENAI:
        if self.OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY is required...")
    # Similar checks for other providers
```

**4. Format Validation:**
```python
@field_validator("OPENAI_API_KEY")
def validate_openai_key_format(cls, v):
    # Checks for placeholder values, correct prefix (sk-), minimum length
    pass

@field_validator("PINECONE_HOST")
def validate_pinecone_host(cls, v):
    # Ensures https:// prefix and pinecone.io domain
    pass

@field_validator("S3_BUCKET")
def validate_s3_bucket_name(cls, v):
    # Validates S3 bucket naming rules (3-63 chars, no special chars)
    pass
```

**5. Safe Logging:**
```python
def __repr__(self) -> str:
    """Safe string representation that doesn't expose secrets."""
    return f"Settings(LLM_PROVIDER={self.LLM_PROVIDER}, LOG_LEVEL={self.LOG_LEVEL})"
```

#### Optional Dependency Flags

The framework uses runtime import checking for optional features:

**Pattern:**
```python
try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Later in code:
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT.labels(...).inc()
```

**Optional Dependencies:**
- `IMPORTS_AVAILABLE` - Core framework imports (validation models, auth)
- `PROMETHEUS_AVAILABLE` - Prometheus metrics collection
- `VALIDATION_AVAILABLE` - Input validation models
- `EXCEPTIONS_AVAILABLE` - Custom exception hierarchy
- `Chroma` / `FAISS` - Vector store implementations
- `OpenAIEmbeddings` - Embedding model support

**Graceful Degradation:**
```python
if not IMPORTS_AVAILABLE:
    raise HTTPException(status_code=500, detail="Authentication module not available")

if not PROMETHEUS_AVAILABLE:
    raise HTTPException(status_code=501, detail="Prometheus metrics not available")
```

---

### Component Interactions

This section details how the REST API integrates with the core framework and how the neural meta-controller makes routing decisions.

#### REST API to Framework Connection

```mermaid
sequenceDiagram
    participant Client as API Client
    participant FastAPI as REST Server
    participant Auth as Authenticator
    participant Validator as Pydantic Validator
    participant Framework as MCTS Framework
    participant Agents as HRM/TRM Agents
    participant MCTS as MCTS Engine

    Client->>FastAPI: POST /query
    FastAPI->>Auth: verify_api_key()
    Auth-->>FastAPI: ClientInfo

    FastAPI->>Validator: QueryRequest validation
    Validator-->>FastAPI: Validated input

    FastAPI->>Framework: framework.process(query, use_mcts, use_rag)

    par Agent Execution
        Framework->>Agents: hrm_agent.process()
        Agents-->>Framework: HRM results
        Framework->>Agents: trm_agent.process()
        Agents-->>Framework: TRM results
    end

    alt MCTS Enabled
        Framework->>MCTS: mcts_simulator_node()
        MCTS-->>Framework: MCTS stats & best action
    end

    Framework->>Framework: aggregate_results_node()
    Framework->>Framework: evaluate_consensus_node()
    Framework->>Framework: synthesize_node()

    Framework-->>FastAPI: {response, metadata, state}
    FastAPI-->>Client: QueryResponse
```

**Integration Points:**

1. **Request Validation Layer:**
   ```python
   # REST Server validates request
   validated_input = QueryInput(
       query=request.query,
       use_rag=request.use_rag,
       use_mcts=request.use_mcts,
       thread_id=request.thread_id,
   )
   ```

2. **Framework Invocation:**
   ```python
   result = await framework.process(
       query=validated_query,
       use_rag=request.use_rag,
       use_mcts=request.use_mcts,
       mcts_iterations=request.mcts_iterations,
       thread_id=request.thread_id,
   )
   ```

3. **Response Construction:**
   ```python
   return QueryResponse(
       response=result["response"],
       confidence=result["metadata"]["consensus_score"],
       agents_used=result["metadata"]["agents_used"],
       mcts_stats=result["metadata"].get("mcts_stats"),
       processing_time_ms=elapsed_time,
       metadata={"client_id": client_info.client_id, ...}
   )
   ```

#### Neural Meta-Controller Routing Decision Flow

The neural meta-controller (RNN or BERT) dynamically selects which agent to execute based on current state features:

```mermaid
flowchart TD
    A[Current AgentState] --> B[Extract Features]
    B --> C{Meta-Controller Type}

    C -->|RNN| D[normalize_features]
    D --> E[10D Feature Vector]
    E --> F[RNN Forward Pass]
    F --> G[GRU + Dropout + Linear]

    C -->|BERT| H[features_to_text]
    H --> I[Text Description]
    I --> J[BERT Tokenizer]
    J --> K[BERT Sequence Classification]

    G --> L[Softmax Output]
    K --> L

    L --> M{Agent Selection}
    M -->|P(HRM) highest| N[Execute HRM Agent]
    M -->|P(TRM) highest| O[Execute TRM Agent]
    M -->|P(MCTS) highest| P[Execute MCTS Simulation]

    N --> Q[Update State]
    O --> Q
    P --> Q

    Q --> R{Consensus Reached?}
    R -->|No| A
    R -->|Yes| S[Synthesize Response]

    subgraph Feature Extraction
        B
        D
        E
        H
        I
    end

    subgraph Neural Network
        F
        G
        J
        K
        L
    end

    subgraph Agent Execution
        N
        O
        P
    end
```

**Feature Extraction Process:**

1. **RNN Meta-Controller:**
   ```python
   # Extract 10D normalized features
   features = normalize_features(MetaControllerFeatures(
       hrm_confidence=state["confidence_scores"].get("hrm", 0.0),
       trm_confidence=state["confidence_scores"].get("trm", 0.0),
       mcts_value=state["mcts_stats"]["best_action_value"] if state.get("mcts_stats") else 0.0,
       consensus_score=state.get("consensus_score", 0.0),
       last_agent=state["agent_outputs"][-1]["agent"] if state["agent_outputs"] else "none",
       iteration=state["iteration"],
       query_length=len(state["query"]),
       has_rag_context=bool(state.get("rag_context"))
   ))
   # features -> [hrm_conf, trm_conf, mcts_val, consensus, hrm_onehot, trm_onehot, mcts_onehot, iter_norm, len_norm, has_rag]
   ```

2. **BERT Meta-Controller:**
   ```python
   # Convert to structured text
   text = features_to_text(features)
   # Example output:
   # "Agent State Features:
   #  HRM confidence: 0.800
   #  TRM confidence: 0.600
   #  MCTS value: 0.750
   #  Consensus score: 0.700
   #  Last agent used: hrm
   #  Current iteration: 2
   #  Query length: 150 characters
   #  RAG context: available"
   ```

**Routing Decision Logic:**

```python
class RNNMetaController:
    def forward(self, x):
        # x: (batch_size, 10) normalized features
        gru_out, hidden = self.gru(x.unsqueeze(1))  # Add sequence dim
        dropped = self.dropout(gru_out[:, -1, :])    # Take last output
        logits = self.fc(dropped)                    # Linear to 3 classes
        return F.softmax(logits, dim=1)              # Probability over agents

    def predict(self, features):
        probs = self.forward(features)  # [P(HRM), P(TRM), P(MCTS)]
        agent_idx = torch.argmax(probs)
        return {
            "agent": ["hrm", "trm", "mcts"][agent_idx],
            "confidence": probs[agent_idx].item(),
            "probs": probs.tolist()
        }
```

**Integration with LangGraph:**

```python
# In route_decision_node or custom routing logic
def route_with_meta_controller(state: AgentState) -> str:
    # Build features from current state
    features = extract_meta_controller_features(state)

    # Get prediction from neural controller
    prediction = meta_controller.predict(features)

    # Log decision to Pinecone for future training
    if pinecone_store:
        pinecone_store.upsert(
            vectors=[(state_id, normalize_features(features))],
            metadata={"agent": prediction["agent"], "probs": prediction["probs"]}
        )

    # Log to Braintrust for experiment tracking
    if braintrust_tracker:
        braintrust_tracker.log_prediction(prediction)

    return prediction["agent"]
```

**Confidence-Based Fallback:**

```python
def route_with_confidence_threshold(state: AgentState, threshold: float = 0.6) -> str:
    prediction = meta_controller.predict(state_features)

    if prediction["confidence"] >= threshold:
        return prediction["agent"]
    else:
        # Fallback to rule-based routing when confidence is low
        if "hrm_results" not in state:
            return "hrm"
        elif "trm_results" not in state:
            return "trm"
        else:
            return "mcts" if state["use_mcts"] else "aggregate"
```

---

### Docker Deployment Stack

```mermaid
graph TB
    subgraph "Production Environment"
        subgraph "Application Layer"
            api[REST API Server<br/>FastAPI/Uvicorn<br/>Port 8000]
            auth[Authentication<br/>API Key + Rate Limiting]
            validation[Input Validation<br/>Pydantic Models]
        end

        subgraph "Monitoring Stack"
            prometheus[Prometheus<br/>Metrics Collection<br/>Port 9092]
            grafana[Grafana<br/>Dashboards<br/>Port 3000]
            jaeger[Jaeger<br/>Distributed Tracing<br/>Port 16686]
            alertmanager[AlertManager<br/>Alert Routing<br/>Port 9093]
            otel[OpenTelemetry<br/>Collector<br/>Port 4317]
        end

        subgraph "Infrastructure"
            redis[Redis<br/>Rate Limit Cache<br/>Port 6380]
            docker[Docker Network<br/>172.30.0.0/16]
        end
    end

    api --> auth
    auth --> validation
    validation --> otel
    api --> prometheus
    prometheus --> grafana
    prometheus --> alertmanager
    otel --> jaeger
    auth --> redis

    classDef app fill:#bdf,stroke:#333
    classDef monitor fill:#ffd,stroke:#333
    classDef infra fill:#dfd,stroke:#333

    class api,auth,validation app
    class prometheus,grafana,jaeger,alertmanager,otel monitor
    class redis,docker infra
```

### REST API Architecture

```mermaid
C4Component
    title Component Diagram - Production REST API

    Container_Boundary(api, "REST API Server") {
        Component(fastapi, "FastAPI App", "ASGI", "OpenAPI 3.1 compliant REST endpoints")
        Component(auth_middleware, "Auth Middleware", "Python", "API key validation and rate limiting")
        Component(error_handler, "Error Handler", "Python", "Exception sanitization for production")
        Component(health, "/health", "Endpoint", "Liveness probe")
        Component(ready, "/ready", "Endpoint", "Readiness probe")
        Component(metrics, "/metrics", "Endpoint", "Prometheus scrape target")
        Component(query, "/query", "Endpoint", "Main reasoning endpoint")
        Component(stats, "/stats", "Endpoint", "Client statistics")
    }

    Container(framework, "MCTS Framework")
    Container(monitoring, "Monitoring Stack")
    System_Ext(client, "API Client")

    Rel(client, fastapi, "HTTPS/REST", "JSON")
    Rel(fastapi, auth_middleware, "Validates requests")
    Rel(auth_middleware, error_handler, "Sanitizes errors")
    Rel(fastapi, health, "Routes /health")
    Rel(fastapi, ready, "Routes /ready")
    Rel(fastapi, metrics, "Routes /metrics")
    Rel(fastapi, query, "Routes /query")
    Rel(fastapi, stats, "Routes /stats")
    Rel(query, framework, "Processes queries")
    Rel(metrics, monitoring, "Exports telemetry")
```

### Kubernetes Deployment

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "Ingress Layer"
            ingress[Ingress Controller<br/>TLS Termination]
        end

        subgraph "Service Layer"
            svc[ClusterIP Service<br/>Port 8000, 9090]
        end

        subgraph "Workload Layer"
            deploy[Deployment<br/>3 Replicas]
            hpa[Horizontal Pod Autoscaler<br/>2-10 Replicas<br/>CPU 70% / Memory 80%]
            pdb[Pod Disruption Budget<br/>minAvailable: 2]
        end

        subgraph "Pod Configuration"
            pod[Pod<br/>CPU: 2-4 cores<br/>Memory: 4-8Gi]
            liveness[Liveness Probe<br/>/health every 30s]
            readiness[Readiness Probe<br/>/ready every 10s]
            security[Security Context<br/>Non-root, Read-only FS]
        end

        subgraph "Resource Management"
            rbac[RBAC<br/>ServiceAccount<br/>Minimal Permissions]
            configmap[ConfigMap<br/>prometheus.yml<br/>alerts.yml]
            secrets[Secrets<br/>API Keys<br/>TLS Certs]
        end
    end

    ingress --> svc
    svc --> deploy
    deploy --> pod
    hpa --> deploy
    pdb --> deploy
    pod --> liveness
    pod --> readiness
    pod --> security
    deploy --> rbac
    deploy --> configmap
    deploy --> secrets

    classDef ingress fill:#f9f,stroke:#333
    classDef service fill:#bdf,stroke:#333
    classDef workload fill:#ffd,stroke:#333
    classDef pod fill:#dfd,stroke:#333
    classDef resource fill:#fbb,stroke:#333

    class ingress ingress
    class svc service
    class deploy,hpa,pdb workload
    class pod,liveness,readiness,security pod
    class rbac,configmap,secrets resource
```

### Service Level Architecture

```mermaid
flowchart TB
    subgraph "SLA Tiers"
        standard[Standard Tier<br/>99.5% Uptime<br/>P95 < 30s]
        production[Production Tier<br/>99.9% Uptime<br/>P95 < 15s]
        enterprise[Enterprise Tier<br/>99.95% Uptime<br/>P95 < 10s]
    end

    subgraph "SLIs - Service Level Indicators"
        availability[Availability SLI<br/>Health check success rate]
        latency[Latency SLI<br/>P95 response time]
        error_rate[Error Rate SLI<br/>5xx / Total requests]
        throughput[Throughput SLI<br/>Requests per second]
    end

    subgraph "SLOs - Service Level Objectives"
        uptime[Uptime: 99.9%<br/>43.2 min/month budget]
        p95_target[P95 Latency: < 15s<br/>5% budget]
        error_budget[Error Rate: < 0.1%<br/>0.1% budget]
    end

    subgraph "Incident Response"
        p0[P0 Critical<br/>Response: 15 min]
        p1[P1 High<br/>Response: 1 hour]
        p2[P2 Medium<br/>Response: 4 hours]
        escalation[Escalation Matrix<br/>On-call -> Lead -> CTO]
    end

    standard --> availability
    production --> latency
    enterprise --> error_rate

    availability --> uptime
    latency --> p95_target
    error_rate --> error_budget
    throughput --> uptime

    uptime --> p0
    p95_target --> p1
    error_budget --> p2
    p0 --> escalation
    p1 --> escalation

    classDef tier fill:#bdf,stroke:#333
    classDef sli fill:#ffd,stroke:#333
    classDef slo fill:#dfd,stroke:#333
    classDef incident fill:#fbb,stroke:#333

    class standard,production,enterprise tier
    class availability,latency,error_rate,throughput sli
    class uptime,p95_target,error_budget slo
    class p0,p1,p2,escalation incident
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | Python 3.11+ | Core language with async support |
| **Web Framework** | FastAPI 0.110 | Production REST API with OpenAPI 3.1 |
| **ASGI Server** | Uvicorn 0.28 | High-performance async server |
| **HTTP Client** | httpx 0.25 | Async HTTP with connection pooling |
| **Validation** | Pydantic v2.6 | Type-safe configuration and input validation |
| **Configuration** | pydantic-settings | Environment variable management |
| **State Machine** | LangGraph 0.0.66 | Agent orchestration and graph workflows |
| **Authentication** | PyJWT 2.8 | API key and JWT token validation |
| **Rate Limiting** | Redis 7 | Distributed rate limit enforcement |
| **Containerization** | Docker + multi-stage | Security-hardened production images |
| **Orchestration** | Kubernetes | Auto-scaling, health checks, rolling updates |
| **Metrics** | Prometheus 2.48 | Time-series metrics and alerting |
| **Dashboards** | Grafana 10.2 | Visualization and monitoring |
| **Tracing** | Jaeger + OpenTelemetry | Distributed tracing and spans |
| **Storage** | aioboto3 12.3 | Async AWS S3 operations |
| **Testing** | pytest 8.1 + coverage | Comprehensive test suite with coverage |
| **Load Testing** | Custom + Hypothesis | Performance and property-based testing |
| **Chaos Testing** | Custom resilience tests | Fault injection and degradation testing |
| **Linting** | ruff 0.3.4 | Fast Python linting |
| **Type Checking** | mypy 1.9 | Static type analysis |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **Documentation** | OpenAPI + Swagger UI | Interactive API documentation |
