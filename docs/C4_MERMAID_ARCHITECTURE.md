# C4 Architecture Diagrams - Multi-Agent MCTS Platform

> Complete C4 model representation using Mermaid diagrams
> Version: 2.0 | Last Updated: January 28, 2026

---

## Table of Contents

1. [Level 1: System Context](#level-1-system-context)
2. [Level 2: Container Diagram](#level-2-container-diagram)
3. [Level 3: Component Diagrams](#level-3-component-diagrams)
4. [Level 4: Code Diagrams](#level-4-code-diagrams)
5. [Sequence Diagrams](#sequence-diagrams)
6. [State Machine Diagrams](#state-machine-diagrams)
7. [Data Flow Diagrams](#data-flow-diagrams)

---

## Level 1: System Context

### Primary System Context

```mermaid
flowchart TB
    subgraph Users["Users"]
        DEV[("Developer<br/>Uses API & CLI")]
        DS[("Data Scientist<br/>Training & Evaluation")]
        ANALYST[("Business Analyst<br/>Query Interface")]
        ADMIN[("Administrator<br/>Monitoring & Config")]
    end

    subgraph Core["Multi-Agent MCTS Platform"]
        PLATFORM[["Multi-Agent MCTS<br/>Platform<br/><br/>Orchestrates intelligent agents<br/>for complex reasoning tasks"]]
    end

    subgraph External["External Systems"]
        OPENAI[("OpenAI API<br/>GPT-4/GPT-4o")]
        ANTHROPIC[("Anthropic API<br/>Claude")]
        LMSTUDIO[("LM Studio<br/>Local LLM")]
        PINECONE[("Pinecone<br/>Vector Database")]
        WANDB[("Weights & Biases<br/>Experiment Tracking")]
        PROMETHEUS[("Prometheus<br/>Metrics Collection")]
        S3[("S3/MinIO<br/>Model Storage")]
    end

    DEV -->|REST API| PLATFORM
    DS -->|Training API| PLATFORM
    ANALYST -->|Web UI, CLI| PLATFORM
    ADMIN -->|Admin API| PLATFORM

    PLATFORM -->|LLM Inference| OPENAI
    PLATFORM -->|LLM Inference| ANTHROPIC
    PLATFORM -->|Local Inference| LMSTUDIO
    PLATFORM -->|Vector Search| PINECONE
    PLATFORM -->|Experiment Logs| WANDB
    PLATFORM -->|Metrics Export| PROMETHEUS
    PLATFORM -->|Model Artifacts| S3

    style PLATFORM fill:#1168bd,stroke:#0b4884,color:#fff
    style OPENAI fill:#74aa9c,stroke:#5a8a7c
    style ANTHROPIC fill:#d4a574,stroke:#b48554
    style PINECONE fill:#00a98f,stroke:#008570
```

### Extended Context with Data Flows

```mermaid
flowchart LR
    subgraph Users["User Touchpoints"]
        direction TB
        WEB[Web Interface]
        CLI[CLI Tool]
        API[REST API]
        SDK[Python SDK]
    end

    subgraph Platform["Multi-Agent Platform"]
        direction TB
        GW[API Gateway]
        ORCH[Orchestrator]
        AGENTS[Agent Pool]
        MCTS[MCTS Engine]
    end

    subgraph Intelligence["Intelligence Layer"]
        direction TB
        LLM[LLM Providers]
        RAG[RAG System]
        NEURAL[Neural Models]
    end

    subgraph Persistence["Data Layer"]
        direction TB
        CACHE[(Redis Cache)]
        VECTOR[(Vector DB)]
        MODELS[(Model Store)]
        LOGS[(Log Store)]
    end

    WEB --> GW
    CLI --> GW
    API --> GW
    SDK --> GW

    GW --> ORCH
    ORCH --> AGENTS
    ORCH --> MCTS
    AGENTS --> LLM
    AGENTS --> RAG
    MCTS --> NEURAL

    LLM --> CACHE
    RAG --> VECTOR
    NEURAL --> MODELS
    ORCH --> LOGS

    style Platform fill:#e3f2fd
    style Intelligence fill:#fff3e0
    style Persistence fill:#f3e5f5
```

---

## Level 2: Container Diagram

### Main Container Architecture

```mermaid
flowchart TB
    subgraph Client["Client Applications"]
        GRADIO[Gradio Web UI<br/>Interactive Demo]
        RESTCLIENT[REST Clients<br/>curl, Postman, SDKs]
        JUPYTER[Jupyter/Colab<br/>Notebooks]
    end

    subgraph API["API Layer"]
        FASTAPI[FastAPI Server<br/>REST Endpoints<br/>Port 8000]
    end

    subgraph Orchestration["Orchestration Layer"]
        LANGGRAPH[LangGraph<br/>State Machine<br/>Workflow Engine]
        SCHEDULER[Task Scheduler<br/>Async Queue<br/>Priority Management]
    end

    subgraph AgentLayer["Agent Layer"]
        HRM[HRM Agent<br/>Hierarchical<br/>Reasoning]
        TRM[TRM Agent<br/>Task<br/>Refinement]
        HYBRID[Hybrid Agent<br/>LLM + Neural]
        SYMBOLIC[Symbolic Agent<br/>Neuro-Symbolic<br/>Reasoning]
    end

    subgraph Intelligence["Intelligence Layer"]
        MCTS[MCTS Engine<br/>Monte Carlo<br/>Tree Search]
        META[Meta-Controller<br/>Neural Routing<br/>RNN + BERT]
        POLICY[Policy Networks<br/>Action Selection]
        VALUE[Value Networks<br/>State Evaluation]
    end

    subgraph Adapters["Adapter Layer"]
        OPENAI_ADAPTER[OpenAI Adapter<br/>GPT-4, GPT-4o]
        ANTHROPIC_ADAPTER[Anthropic Adapter<br/>Claude 3]
        LMSTUDIO_ADAPTER[LM Studio Adapter<br/>Local Models]
        RAG_ADAPTER[RAG Retriever<br/>Context Fetching]
    end

    subgraph Storage["Storage Layer"]
        REDIS[(Redis<br/>LRU Cache<br/>Session Store)]
        PINECONE_DB[(Pinecone<br/>Vector Store<br/>RAG Index)]
        S3_STORE[(S3/MinIO<br/>Model Artifacts<br/>Checkpoints)]
        POSTGRES[(PostgreSQL<br/>Audit Logs<br/>Metadata)]
    end

    subgraph Observability["Observability"]
        PROMETHEUS_SVC[Prometheus<br/>Metrics]
        GRAFANA_SVC[Grafana<br/>Dashboards]
        OTEL[OpenTelemetry<br/>Tracing]
        LOGGING[Structured<br/>Logging]
    end

    GRADIO --> FASTAPI
    RESTCLIENT --> FASTAPI
    JUPYTER --> FASTAPI

    FASTAPI --> LANGGRAPH
    LANGGRAPH --> SCHEDULER

    SCHEDULER --> HRM
    SCHEDULER --> TRM
    SCHEDULER --> HYBRID
    SCHEDULER --> SYMBOLIC

    LANGGRAPH --> MCTS
    LANGGRAPH --> META

    HRM --> OPENAI_ADAPTER
    HRM --> ANTHROPIC_ADAPTER
    TRM --> OPENAI_ADAPTER
    HYBRID --> LMSTUDIO_ADAPTER
    SYMBOLIC --> RAG_ADAPTER

    MCTS --> POLICY
    MCTS --> VALUE
    META --> POLICY

    OPENAI_ADAPTER --> REDIS
    RAG_ADAPTER --> PINECONE_DB
    POLICY --> S3_STORE
    VALUE --> S3_STORE
    LANGGRAPH --> POSTGRES

    FASTAPI --> PROMETHEUS_SVC
    FASTAPI --> OTEL
    LANGGRAPH --> LOGGING
    PROMETHEUS_SVC --> GRAFANA_SVC

    style LANGGRAPH fill:#1976d2,stroke:#1565c0,color:#fff
    style MCTS fill:#388e3c,stroke:#2e7d32,color:#fff
    style META fill:#f57c00,stroke:#ef6c00,color:#fff
```

### Container Communication Matrix

```mermaid
flowchart LR
    subgraph Sync["Synchronous Communication"]
        direction TB
        HTTP[HTTP/REST<br/>JSON Payloads]
    end

    subgraph Async["Asynchronous Communication"]
        direction TB
        WEBSOCKET[WebSocket<br/>Streaming]
        QUEUE[Message Queue<br/>Redis Pub/Sub]
    end

    subgraph Data["Data Protocols"]
        direction TB
        PROM[Prometheus<br/>Metrics Scraping]
        OTLP[OTLP<br/>Trace Export]
        S3_PROTO[S3 Protocol<br/>Object Storage]
    end

    API_SVC[API Services] --> HTTP
    API_SVC --> WEBSOCKET

    AGENTS_SVC[Agent Services] --> QUEUE
    TRAINING_SVC[Training Services] --> QUEUE

    MONITORING[Monitoring] --> PROM
    MONITORING --> OTLP
    MODELS_SVC[Model Services] --> S3_PROTO
```

---

## Level 3: Component Diagrams

### 3.1 MCTS Engine Components

```mermaid
flowchart TB
    subgraph MCTSEngine["MCTS Engine"]
        CORE[MCTSCore<br/>Main Algorithm<br/>Orchestration]

        subgraph TreeStructure["Tree Structure"]
            NODE[MCTSNode<br/>Tree Node<br/>UCB1 Selection]
            STATE[MCTSState<br/>Hashable State<br/>Game Representation]
        end

        subgraph Phases["Search Phases"]
            SELECT[Selection<br/>Tree Traversal]
            EXPAND[Expansion<br/>Node Creation]
            SIMULATE[Simulation<br/>Rollout]
            BACKPROP[Backpropagation<br/>Value Update]
        end

        subgraph Policies["Policies"]
            UCB1[UCB1 Policy<br/>Exploration/Exploitation]
            PUCT[PUCT Policy<br/>Prior-guided]
            ROLLOUT[Rollout Policy<br/>Simulation Strategy]
        end

        subgraph Optimization["Optimization"]
            PARALLEL[Parallel MCTS<br/>AsyncIO Workers<br/>Virtual Loss]
            CACHE[Simulation Cache<br/>LRU Strategy<br/>Result Memoization]
            PROGRESSIVE[Progressive Widening<br/>Action Space Control]
        end

        subgraph LLMGuided["LLM-Guided MCTS"]
            LLM_EXPAND[LLM Expansion<br/>Action Generation]
            LLM_EVAL[LLM Evaluation<br/>State Scoring]
            LLM_ROLLOUT[LLM Rollout<br/>Guided Simulation]
        end
    end

    CORE --> SELECT
    SELECT --> NODE
    NODE --> STATE
    SELECT --> UCB1
    SELECT --> PUCT

    CORE --> EXPAND
    EXPAND --> NODE
    EXPAND --> PROGRESSIVE
    EXPAND --> LLM_EXPAND

    CORE --> SIMULATE
    SIMULATE --> ROLLOUT
    SIMULATE --> LLM_ROLLOUT
    SIMULATE --> CACHE

    CORE --> BACKPROP
    BACKPROP --> NODE

    PARALLEL --> CORE

    style CORE fill:#2196f3,stroke:#1976d2,color:#fff
    style PARALLEL fill:#4caf50,stroke:#388e3c,color:#fff
```

### 3.2 Agent Layer Components

```mermaid
flowchart TB
    subgraph AgentFactory["Agent Factory"]
        FACTORY[AgentFactory<br/>Dependency Injection<br/>Configuration]
    end

    subgraph HRMAgentDetail["HRM Agent"]
        HRM_MAIN[HRM Main<br/>Coordinator]
        H_MODULE[H-Module<br/>High-Level Planning<br/>Problem Decomposition]
        L_MODULE[L-Module<br/>Low-Level Execution<br/>Sub-task Processing]
        ACT[ACT Controller<br/>Adaptive Computation<br/>Dynamic Depth]
        HALT[Halting Mechanism<br/>Confidence Threshold<br/>Early Stopping]
    end

    subgraph TRMAgentDetail["TRM Agent"]
        TRM_MAIN[TRM Main<br/>Refinement Coordinator]
        REFINE[Refinement Loop<br/>Iterative Improvement]
        CONVERGE[Convergence Detector<br/>Quality Assessment]
        MEMORY[Memory Manager<br/>Efficient Recursion]
    end

    subgraph HybridAgentDetail["Hybrid Agent"]
        HYBRID_MAIN[Hybrid Main<br/>Mode Selector]
        LLM_MODE[LLM Mode<br/>Full Capability]
        NEURAL_MODE[Neural Mode<br/>Cost-Effective]
        CONFIDENCE[Confidence Router<br/>Quality vs Cost]
    end

    subgraph SymbolicAgentDetail["Symbolic Agent"]
        SYMBOLIC_MAIN[Symbolic Main<br/>Reasoning Engine]
        CONSTRAINTS[Constraint System<br/>Logical Rules]
        PROOF[Proof Generator<br/>Tree Construction]
        INTEGRATION[MCTS Integration<br/>Constraint Pruning]
    end

    FACTORY --> HRM_MAIN
    FACTORY --> TRM_MAIN
    FACTORY --> HYBRID_MAIN
    FACTORY --> SYMBOLIC_MAIN

    HRM_MAIN --> H_MODULE
    HRM_MAIN --> L_MODULE
    HRM_MAIN --> ACT
    ACT --> HALT

    TRM_MAIN --> REFINE
    REFINE --> CONVERGE
    TRM_MAIN --> MEMORY

    HYBRID_MAIN --> LLM_MODE
    HYBRID_MAIN --> NEURAL_MODE
    HYBRID_MAIN --> CONFIDENCE

    SYMBOLIC_MAIN --> CONSTRAINTS
    SYMBOLIC_MAIN --> PROOF
    SYMBOLIC_MAIN --> INTEGRATION

    style FACTORY fill:#9c27b0,stroke:#7b1fa2,color:#fff
    style HRM_MAIN fill:#2196f3,stroke:#1976d2,color:#fff
    style TRM_MAIN fill:#4caf50,stroke:#388e3c,color:#fff
    style HYBRID_MAIN fill:#ff9800,stroke:#f57c00,color:#fff
    style SYMBOLIC_MAIN fill:#e91e63,stroke:#c2185b,color:#fff
```

### 3.3 Meta-Controller Components

```mermaid
flowchart TB
    subgraph MetaController["Neural Meta-Controller"]
        HYBRID_CTRL[Hybrid Controller<br/>Ensemble Decision]

        subgraph RNNBranch["RNN Branch"]
            GRU[GRU Layers<br/>Sequential Features]
            RNN_HEAD[RNN Classification Head<br/>Agent Probabilities]
        end

        subgraph BERTBranch["BERT Branch"]
            DEBERTA[DeBERTa Encoder<br/>Semantic Understanding]
            LORA[LoRA Adapters<br/>Efficient Fine-tuning]
            BERT_HEAD[BERT Classification Head<br/>Agent Probabilities]
        end

        subgraph FeatureExtraction["Feature Extraction"]
            QUERY_FEAT[Query Features<br/>Length, Complexity]
            CONTEXT_FEAT[Context Features<br/>RAG Results]
            HISTORY_FEAT[History Features<br/>Previous Decisions]
        end

        subgraph Ensemble["Ensemble Layer"]
            WEIGHTED[Weighted Average<br/>Learned Weights]
            CALIBRATION[Calibration Layer<br/>Confidence Adjustment]
            THRESHOLD[Threshold Router<br/>Multi-Agent Trigger]
        end
    end

    subgraph Output["Routing Output"]
        DECISION[Agent Selection<br/>HRM / TRM / MCTS / Multi]
        CONFIDENCE_OUT[Confidence Score<br/>0.0 - 1.0]
    end

    QUERY_FEAT --> GRU
    CONTEXT_FEAT --> GRU
    HISTORY_FEAT --> GRU
    GRU --> RNN_HEAD

    QUERY_FEAT --> DEBERTA
    DEBERTA --> LORA
    LORA --> BERT_HEAD

    RNN_HEAD --> WEIGHTED
    BERT_HEAD --> WEIGHTED
    WEIGHTED --> CALIBRATION
    CALIBRATION --> THRESHOLD

    HYBRID_CTRL --> DECISION
    HYBRID_CTRL --> CONFIDENCE_OUT

    THRESHOLD --> HYBRID_CTRL

    style HYBRID_CTRL fill:#ff5722,stroke:#e64a19,color:#fff
    style GRU fill:#3f51b5,stroke:#303f9f,color:#fff
    style DEBERTA fill:#009688,stroke:#00796b,color:#fff
```

### 3.4 LangGraph Orchestration Components

```mermaid
flowchart TB
    subgraph GraphBuilder["GraphBuilder"]
        BUILDER[Graph Builder<br/>State Machine Factory]

        subgraph Nodes["Graph Nodes"]
            ENTRY[Entry Node<br/>Input Processing]
            RETRIEVE[Retrieve Node<br/>RAG Context]
            ROUTE[Route Node<br/>Meta-Controller]
            HRM_NODE[HRM Node<br/>Hierarchical Agent]
            TRM_NODE[TRM Node<br/>Refinement Agent]
            MCTS_NODE[MCTS Node<br/>Search Agent]
            AGGREGATE[Aggregate Node<br/>Result Collection]
            CONSENSUS[Consensus Node<br/>Agreement Check]
            SYNTHESIZE[Synthesize Node<br/>Response Generation]
        end

        subgraph Edges["Conditional Edges"]
            ROUTE_EDGE{Route Decision}
            CONSENSUS_EDGE{Consensus Check}
            ITERATE_EDGE{Iteration Check}
        end

        subgraph State["AgentState TypedDict"]
            QUERY_STATE[query: str]
            RESULTS_STATE[*_results: dict]
            META_STATE[metadata: dict]
            FINAL_STATE[final_response: str]
        end
    end

    BUILDER --> ENTRY
    ENTRY --> RETRIEVE
    RETRIEVE --> ROUTE

    ROUTE --> ROUTE_EDGE
    ROUTE_EDGE -->|"HRM Selected"| HRM_NODE
    ROUTE_EDGE -->|"TRM Selected"| TRM_NODE
    ROUTE_EDGE -->|"MCTS Selected"| MCTS_NODE
    ROUTE_EDGE -->|"Multi-Agent"| HRM_NODE
    ROUTE_EDGE -->|"Multi-Agent"| TRM_NODE
    ROUTE_EDGE -->|"Multi-Agent"| MCTS_NODE

    HRM_NODE --> AGGREGATE
    TRM_NODE --> AGGREGATE
    MCTS_NODE --> AGGREGATE

    AGGREGATE --> CONSENSUS
    CONSENSUS --> CONSENSUS_EDGE
    CONSENSUS_EDGE -->|"Agreed"| SYNTHESIZE
    CONSENSUS_EDGE -->|"Disagree"| ITERATE_EDGE

    ITERATE_EDGE -->|"iteration < max"| ROUTE
    ITERATE_EDGE -->|"iteration >= max"| SYNTHESIZE

    SYNTHESIZE --> FINAL_STATE

    style BUILDER fill:#673ab7,stroke:#512da8,color:#fff
    style ROUTE_EDGE fill:#ffc107,stroke:#ffa000
    style CONSENSUS_EDGE fill:#ffc107,stroke:#ffa000
```

---

## Level 4: Code Diagrams

### 4.1 MCTS Core Classes

```mermaid
classDiagram
    class MCTSEngine {
        -int _iterations
        -float _c
        -int _seed
        -Random _rng
        -LRUCache _cache
        -bool _progressive_widening
        +search(state: MCTSState) SearchResult
        +select(node: MCTSNode) MCTSNode
        +expand(node: MCTSNode) MCTSNode
        +simulate(node: MCTSNode) float
        +backpropagate(node: MCTSNode, value: float)
        -_should_expand(node: MCTSNode) bool
    }

    class MCTSNode {
        +MCTSState state
        +MCTSNode parent
        +List~MCTSNode~ children
        +Action action
        +int visits
        +float value
        +float prior
        +bool is_terminal
        +select_child(c: float) MCTSNode
        +expand(action: Action, state: MCTSState) MCTSNode
        +backpropagate(value: float)
        +ucb1(c: float) float
        +puct(c: float) float
        +is_fully_expanded() bool
        +best_child() MCTSNode
    }

    class MCTSState {
        <<interface>>
        +Any data
        +__hash__() int
        +__eq__(other: MCTSState) bool
        +is_terminal() bool
        +get_legal_actions() List~Action~
        +apply_action(action: Action) MCTSState
        +evaluate() float
    }

    class Action {
        +str name
        +dict parameters
        +float prior_probability
    }

    class SearchResult {
        +Action best_action
        +int total_visits
        +float best_value
        +MCTSNode root
        +List~ActionStats~ action_stats
        +float search_time_ms
    }

    class SelectionPolicy {
        <<interface>>
        +select(node: MCTSNode, c: float) MCTSNode
    }

    class UCB1Policy {
        +select(node: MCTSNode, c: float) MCTSNode
    }

    class PUCTPolicy {
        -float prior_weight
        +select(node: MCTSNode, c: float) MCTSNode
    }

    class RolloutPolicy {
        <<interface>>
        +rollout(state: MCTSState, max_depth: int) float
    }

    class RandomRollout {
        +rollout(state: MCTSState, max_depth: int) float
    }

    class HeuristicRollout {
        -Callable heuristic
        +rollout(state: MCTSState, max_depth: int) float
    }

    MCTSEngine --> MCTSNode : creates/manages
    MCTSEngine --> MCTSState : operates on
    MCTSEngine --> SelectionPolicy : uses
    MCTSEngine --> RolloutPolicy : uses
    MCTSNode --> MCTSState : contains
    MCTSNode --> Action : stores
    MCTSNode --> MCTSNode : parent/children
    SearchResult --> Action : best_action
    SearchResult --> MCTSNode : root
    UCB1Policy ..|> SelectionPolicy
    PUCTPolicy ..|> SelectionPolicy
    RandomRollout ..|> RolloutPolicy
    HeuristicRollout ..|> RolloutPolicy
```

### 4.2 Agent Interfaces

```mermaid
classDiagram
    class AgentProtocol {
        <<protocol>>
        +process(query: str, context: dict) AgentResult
        +get_confidence() float
        +get_name() str
    }

    class HRMAgent {
        -LLMClient _llm
        -int _max_depth
        -float _halt_threshold
        -bool _use_act
        -Logger _logger
        +process(query: str, context: dict) HRMResult
        +decompose(problem: str) List~SubProblem~
        +execute_subproblem(sub: SubProblem) SubResult
        +aggregate_results(results: List) HRMResult
        -_h_module(problem: str) HierarchicalPlan
        -_l_module(plan: HierarchicalPlan) ExecutionResult
        -_should_halt(confidence: float) bool
    }

    class TRMAgent {
        -LLMClient _llm
        -int _max_iterations
        -float _convergence_threshold
        -Logger _logger
        +process(query: str, context: dict) TRMResult
        +refine(solution: str, feedback: str) str
        +check_convergence(prev: str, curr: str) bool
        -_compute_improvement(prev: str, curr: str) float
    }

    class HybridAgent {
        -LLMClient _llm
        -NeuralModel _neural
        -float _neural_threshold
        +process(query: str, context: dict) HybridResult
        +select_mode(query: str) Mode
        -_llm_inference(query: str) str
        -_neural_inference(query: str) str
    }

    class SymbolicAgent {
        -ConstraintSystem _constraints
        -ProofEngine _prover
        -MCTSEngine _mcts
        +process(query: str, context: dict) SymbolicResult
        +reason(premises: List) Conclusion
        +generate_proof(conclusion: Conclusion) ProofTree
    }

    class AgentResult {
        +str response
        +float confidence
        +dict metadata
        +List~str~ reasoning_trace
        +float latency_ms
    }

    class HRMResult {
        +List~SubProblem~ decomposition
        +List~SubResult~ sub_results
        +str synthesized_response
        +int depth_used
    }

    class TRMResult {
        +str final_solution
        +int iterations_used
        +List~str~ refinement_history
        +float improvement_score
    }

    HRMAgent ..|> AgentProtocol
    TRMAgent ..|> AgentProtocol
    HybridAgent ..|> AgentProtocol
    SymbolicAgent ..|> AgentProtocol
    HRMAgent --> HRMResult : returns
    TRMAgent --> TRMResult : returns
    HRMResult --|> AgentResult
    TRMResult --|> AgentResult
```

### 4.3 Configuration Classes

```mermaid
classDiagram
    class Settings {
        +SecretStr OPENAI_API_KEY
        +SecretStr ANTHROPIC_API_KEY
        +str LLM_PROVIDER
        +str LLM_MODEL
        +bool MCTS_ENABLED
        +int MCTS_ITERATIONS
        +float MCTS_C
        +int SEED
        +str LOG_LEVEL
        +str OTEL_ENDPOINT
        +get_api_key() str
        +get_llm_client() LLMClient
        +model_config: SettingsConfigDict
    }

    class MCTSConfig {
        +int iterations
        +float c
        +int seed
        +bool progressive_widening
        +float widening_alpha
        +int max_depth
        +int cache_size
        +validate_iterations(v: int) int
        +validate_c(v: float) float
    }

    class AgentConfig {
        +str agent_type
        +int max_depth
        +float halt_threshold
        +int max_iterations
        +float convergence_threshold
        +bool use_act
    }

    class MetaControllerConfig {
        +str rnn_model_path
        +str bert_model_path
        +float routing_threshold
        +List~str~ agent_labels
        +int hidden_dim
        +float dropout
    }

    class ObservabilityConfig {
        +str log_level
        +bool enable_tracing
        +str otel_endpoint
        +bool enable_metrics
        +int metrics_port
        +List~str~ sensitive_keys
    }

    Settings --> MCTSConfig : contains
    Settings --> AgentConfig : contains
    Settings --> MetaControllerConfig : contains
    Settings --> ObservabilityConfig : contains
```

---

## Sequence Diagrams

### Query Processing Sequence

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant API as FastAPI
    participant Graph as LangGraph
    participant Meta as MetaController
    participant HRM as HRM Agent
    participant TRM as TRM Agent
    participant MCTS as MCTS Engine
    participant LLM as LLM Provider
    participant Cache as Redis Cache

    User->>API: POST /query
    API->>API: Validate request
    API->>Graph: process(query, config)

    Graph->>Graph: Entry node (extract features)
    Graph->>Cache: Check RAG cache
    Cache-->>Graph: Cache miss
    Graph->>LLM: Embed query
    LLM-->>Graph: Embedding vector
    Graph->>Graph: Retrieve context (RAG)

    Graph->>Meta: predict(query, features)
    Meta->>Meta: Extract features
    Meta->>Meta: RNN forward pass
    Meta->>Meta: BERT forward pass
    Meta->>Meta: Ensemble prediction
    Meta-->>Graph: {agent: "hrm", confidence: 0.89}

    alt HRM Selected
        Graph->>HRM: process(query, context)
        HRM->>HRM: H-Module decomposition
        loop For each subproblem
            HRM->>LLM: Generate solution
            LLM-->>HRM: Sub-solution
        end
        HRM->>HRM: L-Module execution
        HRM->>HRM: Aggregate results
        HRM-->>Graph: HRMResult
    else Multi-Agent Mode
        par Parallel Execution
            Graph->>HRM: process(query)
            Graph->>TRM: process(query)
            Graph->>MCTS: search(state)
        end
        HRM-->>Graph: HRMResult
        TRM-->>Graph: TRMResult
        MCTS-->>Graph: MCTSResult
    end

    Graph->>Graph: Aggregate results
    Graph->>Graph: Evaluate consensus

    alt Consensus Reached
        Graph->>LLM: Synthesize response
        LLM-->>Graph: Final response
    else No Consensus
        Graph->>Graph: Iterate (if < max)
    end

    Graph-->>API: GraphResult
    API-->>User: JSON Response
```

### MCTS Search Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Engine as MCTSEngine
    participant Root as Root Node
    participant Child as Child Nodes
    participant Policy as Selection Policy
    participant Cache as Simulation Cache
    participant Rollout as Rollout Policy

    Engine->>Root: Initialize search

    loop For each iteration
        Note over Engine: Selection Phase
        Engine->>Root: Start traversal
        Root->>Policy: select_child(c=1.414)
        Policy->>Policy: Calculate UCB1 scores
        Policy-->>Root: Best child
        Root->>Child: Continue selection

        alt Node fully expanded
            Child->>Policy: select_child(c)
            Policy-->>Child: Best grandchild
        else Node not expanded
            Note over Engine: Expansion Phase
            Engine->>Child: expand()
            Child->>Child: Generate legal actions
            Child->>Child: Create new node
        end

        Note over Engine: Simulation Phase
        Engine->>Cache: Check cache(state_hash)
        alt Cache hit
            Cache-->>Engine: Cached value
        else Cache miss
            Engine->>Rollout: rollout(state)
            Rollout->>Rollout: Simulate to terminal
            Rollout-->>Engine: Terminal value
            Engine->>Cache: Store(state_hash, value)
        end

        Note over Engine: Backpropagation Phase
        loop Until root
            Engine->>Child: backpropagate(value)
            Child->>Child: Update visits += 1
            Child->>Child: Update value += reward
            Child-->>Engine: Propagate to parent
        end
    end

    Engine->>Root: Get best child
    Root-->>Engine: SearchResult
```

### Training Pipeline Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Orch as Orchestrator
    participant DataGen as Data Generator
    participant Buffer as Replay Buffer
    participant RNN as RNN Trainer
    participant BERT as BERT Trainer
    participant Eval as Evaluator
    participant Store as Model Store

    Orch->>DataGen: generate_dataset(config)
    DataGen->>DataGen: Create synthetic queries
    DataGen->>DataGen: Label with agent types
    DataGen-->>Orch: Training dataset

    Orch->>Buffer: load_dataset(data)
    Buffer->>Buffer: Shuffle and batch

    loop For each epoch
        Orch->>Buffer: get_batch()
        Buffer-->>Orch: Batch data

        par Train in parallel
            Orch->>RNN: train_step(batch)
            RNN->>RNN: Forward pass
            RNN->>RNN: Compute loss
            RNN->>RNN: Backward pass
            RNN-->>Orch: RNN metrics

            Orch->>BERT: train_step(batch)
            BERT->>BERT: Forward pass (LoRA)
            BERT->>BERT: Compute loss
            BERT->>BERT: Backward pass
            BERT-->>Orch: BERT metrics
        end

        alt Validation interval
            Orch->>Eval: evaluate(models, val_set)
            Eval->>Eval: Compute accuracy
            Eval->>Eval: Compute F1 score
            Eval-->>Orch: Validation metrics

            alt Improved
                Orch->>Store: save_checkpoint(models)
                Store-->>Orch: Checkpoint saved
            end
        end
    end

    Orch->>Store: save_final_models()
    Store-->>Orch: Models saved
```

---

## State Machine Diagrams

### LangGraph State Machine

```mermaid
stateDiagram-v2
    [*] --> Entry: Query Received

    Entry --> ValidateInput: Parse Request
    ValidateInput --> Error: Invalid Input
    ValidateInput --> ExtractFeatures: Valid Input

    Error --> [*]: Return Error Response

    ExtractFeatures --> RetrieveContext: Features Extracted

    state RetrieveContext {
        [*] --> CheckCache
        CheckCache --> CacheHit: Found
        CheckCache --> EmbedQuery: Not Found
        EmbedQuery --> VectorSearch
        VectorSearch --> CacheResult
        CacheHit --> [*]
        CacheResult --> [*]
    }

    RetrieveContext --> RouteDecision: Context Retrieved

    state RouteDecision {
        [*] --> MetaController
        MetaController --> HighConfidence: conf > 0.7
        MetaController --> LowConfidence: conf <= 0.7
        HighConfidence --> SingleAgent
        LowConfidence --> MultiAgent
        SingleAgent --> [*]
        MultiAgent --> [*]
    }

    RouteDecision --> AgentExecution: Routing Complete

    state AgentExecution {
        [*] --> SelectAgents

        state SingleAgentMode {
            [*] --> RunSelected
            RunSelected --> [*]
        }

        state MultiAgentMode {
            [*] --> Fork
            Fork --> RunHRM
            Fork --> RunTRM
            Fork --> RunMCTS
            RunHRM --> Join
            RunTRM --> Join
            RunMCTS --> Join
            Join --> [*]
        }

        SelectAgents --> SingleAgentMode: Single
        SelectAgents --> MultiAgentMode: Multi
        SingleAgentMode --> CollectResults
        MultiAgentMode --> CollectResults
        CollectResults --> [*]
    }

    AgentExecution --> EvaluateConsensus: Results Collected

    state EvaluateConsensus {
        [*] --> ComputeAgreement
        ComputeAgreement --> Agreed: score > threshold
        ComputeAgreement --> Disagreed: score <= threshold
        Agreed --> [*]
        Disagreed --> CheckIteration
        CheckIteration --> CanIterate: iter < max
        CheckIteration --> ForceConsensus: iter >= max
        CanIterate --> [*]
        ForceConsensus --> [*]
    }

    EvaluateConsensus --> Synthesize: Consensus/Forced
    EvaluateConsensus --> RouteDecision: Iterate

    state Synthesize {
        [*] --> CombineResults
        CombineResults --> GenerateResponse
        GenerateResponse --> AddMetadata
        AddMetadata --> [*]
    }

    Synthesize --> [*]: Return Response
```

### Meta-Controller State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> FeatureExtraction: Query Received

    state FeatureExtraction {
        [*] --> ExtractQueryFeatures
        ExtractQueryFeatures --> ExtractContextFeatures
        ExtractContextFeatures --> ExtractHistoryFeatures
        ExtractHistoryFeatures --> CombineFeatures
        CombineFeatures --> [*]
    }

    FeatureExtraction --> RNNInference: Features Ready

    state RNNInference {
        [*] --> EmbedSequence
        EmbedSequence --> GRUForward
        GRUForward --> ClassificationHead
        ClassificationHead --> [*]
    }

    state BERTInference {
        [*] --> Tokenize
        Tokenize --> DeBERTaForward
        DeBERTaForward --> LoRAAdaptation
        LoRAAdaptation --> ClassificationHead2
        ClassificationHead2 --> [*]
    }

    RNNInference --> BERTInference: RNN Complete
    BERTInference --> Ensemble: BERT Complete

    state Ensemble {
        [*] --> WeightedAverage
        WeightedAverage --> Calibration
        Calibration --> ThresholdCheck
        ThresholdCheck --> [*]
    }

    Ensemble --> OutputDecision: Ensemble Complete

    state OutputDecision {
        [*] --> HighConfidence: conf > 0.7
        [*] --> LowConfidence: conf <= 0.7
        HighConfidence --> SingleAgentOutput
        LowConfidence --> MultiAgentOutput
        SingleAgentOutput --> [*]
        MultiAgentOutput --> [*]
    }

    OutputDecision --> Idle: Decision Made
```

---

## Data Flow Diagrams

### Request Data Flow

```mermaid
flowchart LR
    subgraph Input["Input Processing"]
        REQ[HTTP Request] --> VAL[Validation]
        VAL --> PARSE[JSON Parse]
        PARSE --> EXTRACT[Feature Extraction]
    end

    subgraph Processing["Core Processing"]
        EXTRACT --> RAG[RAG Retrieval]
        RAG --> ROUTE[Routing Decision]
        ROUTE --> AGENTS[Agent Execution]
        AGENTS --> AGG[Aggregation]
        AGG --> CONSENSUS[Consensus Check]
    end

    subgraph Output["Output Processing"]
        CONSENSUS --> SYNTH[Synthesis]
        SYNTH --> FORMAT[Format Response]
        FORMAT --> RESP[HTTP Response]
    end

    subgraph SideEffects["Side Effects"]
        EXTRACT -.-> LOG1[Log Entry]
        ROUTE -.-> LOG2[Log Routing]
        AGENTS -.-> METRICS[Update Metrics]
        CONSENSUS -.-> TRACE[Trace Span]
    end

    style Processing fill:#e3f2fd
    style SideEffects fill:#fff3e0
```

### Training Data Flow

```mermaid
flowchart TB
    subgraph DataSources["Data Sources"]
        LOGS[Query Logs]
        SYNTH[Synthetic Generator]
        MANUAL[Manual Labels]
    end

    subgraph Preprocessing["Preprocessing"]
        CLEAN[Data Cleaning]
        FEAT[Feature Engineering]
        SPLIT[Train/Val/Test Split]
    end

    subgraph Training["Training Loop"]
        BATCH[Batch Loading]
        FORWARD[Forward Pass]
        LOSS[Loss Computation]
        BACKWARD[Backward Pass]
        UPDATE[Weight Update]
    end

    subgraph Evaluation["Evaluation"]
        VAL_METRICS[Validation Metrics]
        COMPARE[Comparison]
        CHECKPOINT[Checkpoint Decision]
    end

    subgraph Deployment["Model Deployment"]
        SAVE[Save Model]
        VERSION[Version Control]
        DEPLOY[Deploy to Production]
    end

    LOGS --> CLEAN
    SYNTH --> CLEAN
    MANUAL --> CLEAN
    CLEAN --> FEAT
    FEAT --> SPLIT

    SPLIT --> BATCH
    BATCH --> FORWARD
    FORWARD --> LOSS
    LOSS --> BACKWARD
    BACKWARD --> UPDATE
    UPDATE --> BATCH

    UPDATE -.-> VAL_METRICS
    VAL_METRICS --> COMPARE
    COMPARE --> CHECKPOINT

    CHECKPOINT -->|Improved| SAVE
    SAVE --> VERSION
    VERSION --> DEPLOY

    style Training fill:#e8f5e9
    style Evaluation fill:#fff8e1
```

### Observability Data Flow

```mermaid
flowchart TB
    subgraph Application["Application Layer"]
        API[API Server]
        AGENTS[Agent Pool]
        MCTS[MCTS Engine]
        TRAINING[Training Pipeline]
    end

    subgraph Collection["Data Collection"]
        STRUCT_LOG[Structured Logs]
        PROM_METRICS[Prometheus Metrics]
        OTEL_TRACES[OTLP Traces]
    end

    subgraph Storage["Storage Layer"]
        LOKI[Loki / ELK]
        PROM_DB[Prometheus TSDB]
        JAEGER[Jaeger / Tempo]
    end

    subgraph Visualization["Visualization"]
        GRAFANA[Grafana Dashboards]
        ALERTS[Alert Manager]
        TRACE_UI[Trace Explorer]
    end

    API --> STRUCT_LOG
    AGENTS --> STRUCT_LOG
    MCTS --> STRUCT_LOG
    TRAINING --> STRUCT_LOG

    API --> PROM_METRICS
    AGENTS --> PROM_METRICS
    MCTS --> PROM_METRICS

    API --> OTEL_TRACES
    AGENTS --> OTEL_TRACES

    STRUCT_LOG --> LOKI
    PROM_METRICS --> PROM_DB
    OTEL_TRACES --> JAEGER

    LOKI --> GRAFANA
    PROM_DB --> GRAFANA
    PROM_DB --> ALERTS
    JAEGER --> TRACE_UI

    style Collection fill:#e3f2fd
    style Storage fill:#f3e5f5
    style Visualization fill:#e8f5e9
```

---

## Deployment Architecture

### Kubernetes Deployment

```mermaid
flowchart TB
    subgraph Internet["Internet"]
        USERS[Users]
    end

    subgraph K8s["Kubernetes Cluster"]
        subgraph Ingress["Ingress Layer"]
            NGINX[NGINX Ingress]
            CERT[Cert Manager]
        end

        subgraph Services["Service Layer"]
            API_SVC[API Service<br/>ClusterIP]
            TRAINING_SVC[Training Service<br/>ClusterIP]
            METRICS_SVC[Metrics Service<br/>ClusterIP]
        end

        subgraph Workloads["Workloads"]
            API_DEPLOY[API Deployment<br/>HPA: 2-10 replicas]
            TRAINING_DEPLOY[Training StatefulSet<br/>GPU Nodes]
            WORKER_DEPLOY[Worker Deployment<br/>HPA: 1-5 replicas]
        end

        subgraph Storage["Persistent Storage"]
            PVC_MODELS[Models PVC<br/>ReadWriteMany]
            PVC_CACHE[Cache PVC<br/>ReadWriteOnce]
        end

        subgraph ConfigMaps["Configuration"]
            CM_APP[App ConfigMap]
            SECRET_KEYS[API Keys Secret]
        end
    end

    subgraph External["External Services"]
        S3[S3 Model Storage]
        PINECONE[Pinecone Vector DB]
        LLM_PROVIDERS[LLM Providers]
    end

    USERS --> NGINX
    NGINX --> API_SVC
    API_SVC --> API_DEPLOY
    API_DEPLOY --> WORKER_DEPLOY

    TRAINING_SVC --> TRAINING_DEPLOY
    METRICS_SVC --> API_DEPLOY

    API_DEPLOY --> PVC_CACHE
    TRAINING_DEPLOY --> PVC_MODELS

    API_DEPLOY --> CM_APP
    API_DEPLOY --> SECRET_KEYS

    API_DEPLOY --> S3
    API_DEPLOY --> PINECONE
    API_DEPLOY --> LLM_PROVIDERS

    style K8s fill:#e3f2fd
    style External fill:#fff3e0
```

---

*Generated: January 2026*
*Framework: Multi-Agent MCTS v0.1.0*
