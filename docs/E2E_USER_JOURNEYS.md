# End-to-End Workflows & User Journeys

> Comprehensive documentation of user journeys, E2E workflows, and practical scenarios
> Version: 2.0 | Last Updated: January 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Core User Journeys](#core-user-journeys)
3. [E2E Workflow Specifications](#e2e-workflow-specifications)
4. [Industry-Specific Scenarios](#industry-specific-scenarios)
5. [Integration Patterns](#integration-patterns)
6. [Testing & Validation](#testing--validation)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Overview

### User Personas

| Persona | Description | Primary Use Cases |
|---------|-------------|-------------------|
| **Developer** | Software engineer building applications | API integration, code review, architecture design |
| **Data Scientist** | ML/AI practitioner | Model training, experiment tracking, evaluation |
| **Business Analyst** | Strategic decision maker | Report generation, competitive analysis, due diligence |
| **Platform Admin** | DevOps/SRE engineer | Deployment, monitoring, scaling |
| **Researcher** | Academic or R&D professional | Literature review, hypothesis generation |

### Journey Categories

```mermaid
mindmap
  root((User Journeys))
    Technical
      Code Analysis
      Architecture Design
      Bug Investigation
      Performance Optimization
    Business
      Due Diligence
      Market Research
      Risk Assessment
      Strategy Planning
    Research
      Literature Review
      Experiment Design
      Data Analysis
      Hypothesis Generation
    Operations
      Incident Response
      Capacity Planning
      Cost Optimization
      Compliance Audit
```

---

## Core User Journeys

### Journey 1: Developer - Architecture Design

```mermaid
journey
    title Developer Journey: Design Scalable Architecture
    section Discovery
      Developer identifies need: 5: Developer
      Researches existing solutions: 4: Developer
      Formulates requirements: 5: Developer
    section Query Submission
      Opens CLI/Web interface: 5: Developer
      Submits architecture question: 5: Developer
      System validates input: 5: System
    section Analysis
      Meta-controller routes to HRM: 5: System
      HRM decomposes requirements: 5: HRM Agent
      MCTS explores design options: 4: MCTS
      TRM refines selected design: 5: TRM Agent
    section Review
      Developer reviews response: 5: Developer
      Examines reasoning trace: 4: Developer
      Asks follow-up questions: 5: Developer
    section Implementation
      Exports design document: 5: Developer
      Implements architecture: 5: Developer
      Validates against analysis: 4: Developer
```

#### Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEVELOPER ARCHITECTURE DESIGN FLOW                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │  Query  │────▶│ Feature │────▶│  Route  │────▶│ Execute │          │
│  │  Input  │     │ Extract │     │ Decision│     │ Agents  │          │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘          │
│       │               │               │               │                │
│       ▼               ▼               ▼               ▼                │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │Validate │     │  Query  │     │  Meta   │     │  HRM    │          │
│  │ & Parse │     │Embedding│     │Controller│     │  TRM    │          │
│  │ Input   │     │   +RAG  │     │ Neural  │     │  MCTS   │          │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘          │
│                                                       │                │
│                                                       ▼                │
│                                              ┌─────────────────┐       │
│                                              │   Aggregate &   │       │
│                                              │    Consensus    │       │
│                                              └────────┬────────┘       │
│                                                       │                │
│                                                       ▼                │
│                                              ┌─────────────────┐       │
│                                              │   Synthesize    │       │
│                                              │    Response     │       │
│                                              └────────┬────────┘       │
│                                                       │                │
│                                                       ▼                │
│                                              ┌─────────────────┐       │
│                                              │  Return with    │       │
│                                              │   Metadata      │       │
│                                              └─────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Example Interaction

**User Query:**
```
Design a microservices architecture for a high-traffic e-commerce platform
that needs to handle 10,000 concurrent users with real-time inventory updates
and support for flash sales.
```

**System Processing:**

1. **Feature Extraction:**
   - Query length: 150 chars (medium complexity)
   - Keywords detected: `microservices`, `architecture`, `high-traffic`, `real-time`
   - Task type: Architecture design
   - Domain: E-commerce

2. **Routing Decision:**
   - Meta-controller confidence: 0.91
   - Selected agent: HRM (hierarchical decomposition ideal for architecture)
   - Backup: Enable MCTS for design exploration

3. **HRM Decomposition:**
   ```
   Problem: E-commerce Microservices Architecture
   ├── SP1: Core Service Boundaries
   │   ├── User Service
   │   ├── Product Catalog Service
   │   ├── Inventory Service
   │   ├── Order Service
   │   └── Payment Service
   ├── SP2: Data Consistency Patterns
   │   ├── Saga pattern for orders
   │   ├── Event sourcing for inventory
   │   └── CQRS for product catalog
   ├── SP3: Real-time Communication
   │   ├── WebSocket for inventory updates
   │   ├── Message queue (Kafka) for events
   │   └── Cache invalidation strategy
   └── SP4: Scalability Design
       ├── Auto-scaling rules
       ├── Database sharding strategy
       └── CDN for static content
   ```

4. **MCTS Exploration (100 iterations):**
   - Explored 47 unique architecture variants
   - Best variant: Event-driven with CQRS
   - Win rate: 0.78 (handles flash sales well)

5. **Final Response:**
   ```markdown
   ## Recommended Architecture

   ### Service Boundaries
   [Detailed service descriptions...]

   ### Data Patterns
   [CQRS, Event Sourcing details...]

   ### Trade-offs
   - Complexity: Medium-High
   - Consistency: Eventual (acceptable for e-commerce)
   - Scalability: Excellent

   ### Confidence: 91%
   ```

---

### Journey 2: Data Scientist - Model Training

```mermaid
journey
    title Data Scientist Journey: Train Meta-Controller
    section Preparation
      Prepare training data: 5: DS
      Configure hyperparameters: 4: DS
      Set up experiment tracking: 5: DS
    section Training
      Launch training pipeline: 5: DS
      Monitor training metrics: 4: System
      Evaluate checkpoints: 5: System
    section Evaluation
      Run validation suite: 5: System
      Compare with baseline: 4: DS
      Analyze error cases: 5: DS
    section Deployment
      Select best model: 5: DS
      Deploy to staging: 5: System
      A/B test in production: 4: System
    section Monitoring
      Track production metrics: 5: System
      Detect model drift: 4: System
      Trigger retraining: 3: System
```

#### Training Pipeline Flow

```mermaid
flowchart TB
    subgraph DataPrep["Data Preparation"]
        A[Query Logs] --> B[Label Extraction]
        C[Synthetic Generation] --> B
        B --> D[Feature Engineering]
        D --> E[Train/Val/Test Split]
    end

    subgraph Training["Model Training"]
        E --> F[RNN Meta-Controller]
        E --> G[BERT+LoRA Controller]

        F --> H[Validation Loop]
        G --> H

        H --> I{Improved?}
        I -->|Yes| J[Save Checkpoint]
        I -->|No| K[Early Stop Check]
        K -->|Continue| F
        K -->|Stop| L[Best Model]
    end

    subgraph Evaluation["Evaluation Suite"]
        L --> M[Accuracy Metrics]
        L --> N[Latency Benchmarks]
        L --> O[A/B Test Setup]

        M --> P[Report Generation]
        N --> P
        O --> P
    end

    subgraph Deployment["Model Deployment"]
        P --> Q{Pass Threshold?}
        Q -->|Yes| R[Deploy to Staging]
        Q -->|No| S[Rollback]
        R --> T[Production A/B]
        T --> U[Full Rollout]
    end

    style Training fill:#e3f2fd
    style Evaluation fill:#fff3e0
    style Deployment fill:#e8f5e9
```

#### Training Commands

```bash
# Start training with default configuration
python -m training.unified_orchestrator \
    --config training/configs/default.yaml \
    --experiment-name "meta-controller-v2" \
    --wandb-project "multiagent-mcts"

# Training with custom hyperparameters
python -m training.unified_orchestrator \
    --rnn-hidden-dim 256 \
    --bert-lora-rank 8 \
    --learning-rate 1e-4 \
    --batch-size 32 \
    --epochs 50 \
    --early-stopping-patience 5

# Evaluation only
python -m training.evaluate \
    --model-path models/meta_controller_v2.pt \
    --test-set data/test.json \
    --output-report reports/evaluation.md
```

---

### Journey 3: Business Analyst - Due Diligence

```mermaid
journey
    title Business Analyst Journey: M&A Due Diligence
    section Document Collection
      Receive target company docs: 5: Analyst
      Upload to platform: 5: Analyst
      System indexes documents: 5: System
    section Analysis Request
      Define analysis scope: 5: Analyst
      Submit due diligence queries: 5: Analyst
      System processes in parallel: 4: System
    section Deep Analysis
      HRM structures framework: 5: HRM
      Symbolic checks compliance: 4: Symbolic
      TRM refines findings: 5: TRM
    section Report Generation
      Aggregate findings: 5: System
      Generate structured report: 5: System
      Highlight risk factors: 5: System
    section Review & Decision
      Review with full audit trail: 5: Analyst
      Make investment decision: 5: Analyst
      Archive for compliance: 5: Analyst
```

#### Due Diligence Workflow

```
                    ┌──────────────────────────────────────┐
                    │        DOCUMENT INGESTION            │
                    └──────────────────┬───────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   Financial   │           │    Legal      │           │  Operational  │
│   Documents   │           │   Contracts   │           │    Data       │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │       RAG INDEXING & EMBEDDING       │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │        MULTI-AGENT ANALYSIS          │
                    │                                      │
                    │  ┌────────┐ ┌────────┐ ┌────────┐   │
                    │  │  HRM   │ │Symbolic│ │  TRM   │   │
                    │  │Analysis│ │Compliance│ │Refine │   │
                    │  └────┬───┘ └────┬───┘ └────┬───┘   │
                    │       │          │          │       │
                    │       └──────────┼──────────┘       │
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         RISK ASSESSMENT              │
                    │                                      │
                    │  • Financial Risks    [Score: 7/10]  │
                    │  • Legal Risks        [Score: 4/10]  │
                    │  • Operational Risks  [Score: 6/10]  │
                    │  • Regulatory Risks   [Score: 3/10]  │
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │        EXECUTIVE REPORT              │
                    │                                      │
                    │  • Executive Summary                 │
                    │  • Key Findings                      │
                    │  • Risk Matrix                       │
                    │  • Recommendations                   │
                    │  • Audit Trail                       │
                    │                                      │
                    └──────────────────────────────────────┘
```

---

### Journey 4: Platform Admin - Incident Response

```mermaid
journey
    title Platform Admin Journey: Incident Response
    section Detection
      Alert triggered: 5: System
      Admin receives notification: 5: Admin
      Initial assessment: 4: Admin
    section Investigation
      Query system for root cause: 5: Admin
      MCTS explores failure modes: 4: MCTS
      HRM analyzes dependencies: 5: HRM
    section Resolution
      Identify root cause: 5: System
      Generate remediation steps: 5: System
      Apply fix: 5: Admin
    section Post-Mortem
      Generate incident report: 5: System
      Review with team: 5: Admin
      Update runbooks: 4: Admin
```

#### Incident Response Commands

```bash
# Quick diagnosis query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API response times increased 3x in the last hour. What could be causing this?",
    "use_mcts": true,
    "context": {
      "service": "api-gateway",
      "metrics": {"p99_latency": "3000ms", "error_rate": "2%"},
      "recent_changes": ["deployment at 14:30", "config update at 14:45"]
    }
  }'

# Get remediation suggestions
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Suggest remediation steps for database connection pool exhaustion",
    "use_mcts": false,
    "agent_preference": "hrm"
  }'
```

---

### Journey 5: Researcher - Literature Review

```mermaid
journey
    title Researcher Journey: Systematic Literature Review
    section Planning
      Define research question: 5: Researcher
      Set inclusion criteria: 5: Researcher
      Configure search parameters: 4: Researcher
    section Search
      Submit search query: 5: Researcher
      System retrieves papers: 5: System
      RAG indexes abstracts: 5: System
    section Analysis
      HRM categorizes papers: 5: HRM
      TRM extracts key findings: 5: TRM
      MCTS identifies gaps: 4: MCTS
    section Synthesis
      Generate literature matrix: 5: System
      Identify themes: 5: System
      Highlight contradictions: 4: System
    section Output
      Export to reference manager: 5: Researcher
      Generate review draft: 5: System
      Refine and publish: 5: Researcher
```

---

## E2E Workflow Specifications

### Workflow 1: Simple Query Processing

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant A as API Gateway
    participant V as Validator
    participant G as GraphBuilder
    participant M as MetaController
    participant Ag as Agent
    participant L as LLM

    U->>A: POST /query
    A->>V: Validate request
    V-->>A: Valid
    A->>G: process(query)
    G->>G: Extract features
    G->>M: predict(features)
    M-->>G: {agent: "hrm", conf: 0.85}
    G->>Ag: execute(query)
    Ag->>L: generate(prompt)
    L-->>Ag: response
    Ag-->>G: AgentResult
    G->>G: Synthesize
    G-->>A: GraphResult
    A-->>U: JSON Response
```

### Workflow 2: Multi-Agent Consensus

```mermaid
sequenceDiagram
    autonumber
    participant G as GraphBuilder
    participant M as MetaController
    participant H as HRM
    participant T as TRM
    participant MC as MCTS
    participant C as Consensus

    G->>M: predict(query)
    M-->>G: {conf: 0.55, multi: true}

    par Execute in Parallel
        G->>H: process(query)
        G->>T: process(query)
        G->>MC: search(state)
    end

    H-->>G: HRMResult
    T-->>G: TRMResult
    MC-->>G: MCTSResult

    G->>C: evaluate([results])
    C->>C: Compute agreement

    alt Consensus Reached
        C-->>G: {agreed: true, score: 0.92}
        G->>G: Synthesize best response
    else No Consensus
        C-->>G: {agreed: false, score: 0.45}
        G->>G: Iterate or force synthesis
    end
```

### Workflow 3: Training Pipeline E2E

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator
    participant D as DataGenerator
    participant R as ReplayBuffer
    participant RNN as RNNTrainer
    participant B as BERTTrainer
    participant E as Evaluator
    participant S as ModelStore

    O->>D: generate(config)
    D-->>O: Dataset
    O->>R: load(dataset)

    loop Each Epoch
        O->>R: get_batch()
        R-->>O: Batch

        par Train Models
            O->>RNN: train_step(batch)
            O->>B: train_step(batch)
        end

        RNN-->>O: loss, metrics
        B-->>O: loss, metrics

        alt Validation Interval
            O->>E: evaluate(models)
            E-->>O: val_metrics

            alt Improved
                O->>S: save_checkpoint()
            end
        end
    end

    O->>S: save_final()
    S-->>O: model_path
```

---

## Industry-Specific Scenarios

### Financial Services

```yaml
Scenario: Credit Risk Assessment
Actors: Risk Analyst, Compliance Officer
Flow:
  1. Upload loan application documents
  2. System extracts financial indicators
  3. HRM structures risk assessment framework
  4. MCTS explores scenario analysis
  5. Symbolic agent checks regulatory compliance
  6. Generate risk score with explanation

Key Metrics:
  - Accuracy: 94%
  - Compliance rate: 100%
  - Processing time: 45 seconds

Regulatory Compliance:
  - GDPR: Personal data handling
  - Basel III: Capital adequacy
  - SOX: Audit trail requirements
```

### Healthcare

```yaml
Scenario: Clinical Decision Support
Actors: Physician, Medical Coder
Flow:
  1. Input patient symptoms and history
  2. RAG retrieves relevant medical literature
  3. HRM structures differential diagnosis
  4. TRM refines based on test results
  5. Generate treatment recommendations

Key Metrics:
  - Diagnostic accuracy: 89%
  - Literature coverage: 95%
  - Response time: 30 seconds

Compliance:
  - HIPAA: Patient data privacy
  - FDA: Clinical decision support guidelines
```

### Legal

```yaml
Scenario: Contract Analysis
Actors: Legal Counsel, Paralegal
Flow:
  1. Upload contract documents
  2. System identifies clause types
  3. HRM decomposes contract structure
  4. Symbolic agent checks against templates
  5. Highlight risky clauses
  6. Generate summary with recommendations

Key Metrics:
  - Clause identification: 97%
  - Risk detection: 91%
  - Time savings: 75%
```

### Manufacturing

```yaml
Scenario: Root Cause Analysis
Actors: Quality Engineer, Production Manager
Flow:
  1. Input defect description and data
  2. MCTS explores failure mode tree
  3. HRM structures analysis framework
  4. TRM refines hypothesis
  5. Generate corrective action plan

Key Metrics:
  - Root cause accuracy: 87%
  - Time to resolution: -60%
  - Defect recurrence: -45%
```

---

## Integration Patterns

### Pattern 1: REST API Integration

```python
import httpx
import asyncio

async def query_multiagent(query: str, use_mcts: bool = True) -> dict:
    """Integrate with Multi-Agent MCTS via REST API."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/query",
            json={
                "query": query,
                "use_mcts": use_mcts,
                "use_rag": True,
                "max_iterations": 3
            },
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()

# Usage
result = asyncio.run(query_multiagent(
    "Design a caching strategy for user sessions"
))
print(f"Response: {result['response']}")
print(f"Confidence: {result['metadata']['confidence']}")
```

### Pattern 2: Python SDK Integration

```python
from multiagent_mcts import Client, QueryConfig

# Initialize client
client = Client(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Configure query
config = QueryConfig(
    use_mcts=True,
    use_rag=True,
    agent_preference="auto",
    max_iterations=3,
    timeout_seconds=60
)

# Execute query
result = client.query(
    "Optimize database indexing for read-heavy workloads",
    config=config
)

# Access results
print(result.response)
print(result.confidence)
print(result.agents_used)
print(result.reasoning_trace)
```

### Pattern 3: Webhook Integration

```python
from fastapi import FastAPI, BackgroundTasks
import httpx

app = FastAPI()

@app.post("/process")
async def process_request(
    query: str,
    callback_url: str,
    background_tasks: BackgroundTasks
):
    """Process query asynchronously with webhook callback."""

    async def process_and_callback():
        # Call multi-agent system
        async with httpx.AsyncClient() as client:
            result = await client.post(
                "http://multiagent:8000/api/v1/query",
                json={"query": query}
            )

            # Send result to callback URL
            await client.post(
                callback_url,
                json=result.json()
            )

    background_tasks.add_task(process_and_callback)
    return {"status": "processing", "query": query}
```

### Pattern 4: Streaming Integration

```python
import asyncio
import websockets
import json

async def stream_query(query: str):
    """Stream query results via WebSocket."""
    uri = "ws://localhost:8000/ws/query"

    async with websockets.connect(uri) as websocket:
        # Send query
        await websocket.send(json.dumps({
            "query": query,
            "stream": True
        }))

        # Receive streaming updates
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "progress":
                print(f"Progress: {data['agent']} - {data['status']}")
            elif data["type"] == "partial":
                print(f"Partial: {data['content']}", end="")
            elif data["type"] == "complete":
                print(f"\n\nFinal: {data['response']}")
                break

# Usage
asyncio.run(stream_query("Explain microservices architecture"))
```

---

## Testing & Validation

### E2E Test Suite

```python
# tests/e2e/test_user_journeys.py

import pytest
from httpx import AsyncClient

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_developer_architecture_journey():
    """Test complete developer architecture design journey."""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # Submit architecture query
        response = await client.post(
            "/api/v1/query",
            json={
                "query": "Design a microservices architecture for e-commerce",
                "use_mcts": True,
                "use_rag": False
            }
        )

        assert response.status_code == 200
        result = response.json()

        # Verify response structure
        assert "response" in result
        assert "metadata" in result
        assert result["metadata"]["confidence"] > 0.7
        assert "hrm" in [a.lower() for a in result["metadata"]["agents_used"]]

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agent_consensus_journey():
    """Test multi-agent consensus workflow."""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/api/v1/query",
            json={
                "query": "Compare REST vs GraphQL for mobile backend",
                "use_mcts": True,
                "force_multi_agent": True
            }
        )

        assert response.status_code == 200
        result = response.json()

        # Verify multi-agent execution
        assert len(result["metadata"]["agents_used"]) >= 2
        assert "consensus_score" in result["metadata"]

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_handling_journey():
    """Test graceful error handling."""
    async with AsyncClient(base_url="http://localhost:8000") as client:
        # Test invalid input
        response = await client.post(
            "/api/v1/query",
            json={"query": ""}  # Empty query
        )

        assert response.status_code == 422  # Validation error

        # Test timeout handling
        response = await client.post(
            "/api/v1/query",
            json={
                "query": "Complex query requiring long processing",
                "timeout_seconds": 1  # Very short timeout
            }
        )

        # Should handle gracefully
        assert response.status_code in [200, 408, 504]
```

### Load Testing

```python
# tests/performance/test_load.py

import asyncio
from locust import HttpUser, task, between

class MultiAgentUser(HttpUser):
    """Load test user for multi-agent system."""

    wait_time = between(1, 3)

    @task(3)
    def simple_query(self):
        """Simple query - most common."""
        self.client.post(
            "/api/v1/query",
            json={
                "query": "What is dependency injection?",
                "use_mcts": False
            }
        )

    @task(2)
    def mcts_query(self):
        """MCTS-enabled query - medium frequency."""
        self.client.post(
            "/api/v1/query",
            json={
                "query": "Design a caching strategy",
                "use_mcts": True
            }
        )

    @task(1)
    def complex_query(self):
        """Complex multi-agent query - less frequent."""
        self.client.post(
            "/api/v1/query",
            json={
                "query": "Compare microservices vs monolithic architectures with trade-offs",
                "use_mcts": True,
                "force_multi_agent": True
            }
        )

# Run with: locust -f test_load.py --host=http://localhost:8000
```

---

## Troubleshooting Guide

### Common Issues

| Issue | Symptoms | Resolution |
|-------|----------|------------|
| Slow response times | >10s for simple queries | Check LLM provider latency, enable caching |
| Low confidence scores | Consistently <0.7 | Retrain meta-controller, check RAG index |
| Consensus failures | Multiple iterations without agreement | Adjust consensus threshold, check agent configs |
| Memory issues | OOM errors | Reduce batch size, enable streaming |
| API timeouts | 504 errors | Increase timeout, check rate limits |

### Diagnostic Commands

```bash
# Check system health
curl http://localhost:8000/health

# View detailed metrics
curl http://localhost:8000/metrics | grep multiagent

# Test individual agents
curl -X POST http://localhost:8000/api/v1/debug/agent \
  -H "Content-Type: application/json" \
  -d '{"agent": "hrm", "query": "test"}'

# Check meta-controller routing
curl -X POST http://localhost:8000/api/v1/debug/route \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'

# View recent traces
curl http://localhost:8000/api/v1/debug/traces?limit=10
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("multiagent_mcts").setLevel(logging.DEBUG)

# Or via environment
# LOG_LEVEL=DEBUG python -m src.api.rest_server
```

---

*Document generated: January 2026*
*Framework: Multi-Agent MCTS v0.1.0*
