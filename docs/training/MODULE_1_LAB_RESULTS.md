# Module 1: Architecture Deep Dive - Lab Results

**Student:** AI Implementation
**Date:** 2025-11-19
**Module:** 1 - System & Architecture Deep Dive

---

## Lab 1: Navigate the Codebase

**Objective:** Use architecture docs to find key code locations.

### Task 1: Find the FastAPI route that handles query/analysis requests

**Location:** [src/api/rest_server.py:336-404](../../src/api/rest_server.py#L336-L404)

**Details:**
- **Endpoint:** `POST /query`
- **Function:** `process_query()`
- **Request Model:** `QueryRequest` - accepts query text, MCTS/RAG flags, iterations, thread_id
- **Response Model:** `QueryResponse` - returns response, confidence, agents_used, mcts_stats, processing_time
- **Features:** API key authentication, rate limiting, Prometheus metrics integration

**Secondary Neural Inference Server:**
- **Location:** [src/api/inference_server.py:214-285](../../src/api/inference_server.py#L214-L285)
- **Endpoint:** `POST /inference`
- **Purpose:** Neural MCTS inference with HRM/TRM integration

---

### Task 2: Locate the HRM agent's task decomposition logic

**Location:** [src/agents/hrm_agent.py:324-346](../../src/agents/hrm_agent.py#L324-L346)

**Details:**
- **Function:** `HRMAgent.decompose_problem()` (async method)
- **Architecture Components:**
  - H-Module (lines 87-152): High-level planning and decomposition
  - L-Module (lines 154-216): Low-level execution
  - ACT Mechanism (lines 46-84): Adaptive Computation Time with dynamic halting
- **Key Data Structures:**
  - `SubProblem` dataclass (lines 25-33): Represents decomposed tasks
  - `HRMOutput` dataclass (lines 36-44): Contains final state, subproblems, halt metrics
- **Features:** Multi-head self-attention, progressive hierarchy decomposition, confidence-based halting

---

### Task 3: Find the MCTS UCB1 selection implementation

**Location:** [src/framework/mcts/policies.py:25-51](../../src/framework/mcts/policies.py#L25-L51)

**Details:**
- **Function:** `ucb1(value_sum, visits, parent_visits, c=1.414)`
- **Formula:** `Q(s,a) + c * sqrt(N(s)) / sqrt(N(s,a))`
- **Components:**
  - Exploitation term: `value_sum / visits`
  - Exploration term: `c * sqrt(parent_visits) / sqrt(visits)`
  - Returns `float("inf")` for unvisited nodes (prioritizes exploration)
- **Usage:** Called in [src/framework/mcts/core.py:90-117](../../src/framework/mcts/core.py#L90-L117) by `MCTSNode.select_child()`

**Related Implementations:**
- UCB1-Tuned (variance-aware): lines 54-91
- Selection Policies Enum: lines 94-108
- Rollout Policies: lines 110-282
- Progressive Widening: lines 284-344

---

### Task 4: Identify where LangSmith tracing is initialized

**Primary Location:** [tests/utils/langsmith_tracing.py](../../tests/utils/langsmith_tracing.py)

**Key Components:**
1. **Metadata Helpers:**
   - `get_test_metadata()` (lines 30-45): Extracts CI/CD environment metadata
   - `_build_trace_metadata()` (lines 48-106): Builds metadata and tags

2. **Decorators:**
   - `@trace_e2e_test()` (lines 109-202): Main E2E test tracing decorator
   - `@trace_api_endpoint()` (lines 243-290): API endpoint test tracing
   - `@trace_mcts_simulation()` (lines 293-343): MCTS simulation tracing

3. **Context Managers:**
   - `trace_e2e_workflow()` (lines 205-240): For workflow section tracing

4. **Runtime Helpers:**
   - `update_run_metadata()` (lines 346-367): Dynamic metadata injection
   - `add_run_tag()` (lines 370-390): Dynamic tag addition
   - `get_langsmith_client()` (lines 393-407): Client initialization

**Usage Examples:**
- [tests/e2e/test_complete_query_flow_traced.py](../../tests/e2e/test_complete_query_flow_traced.py)
- [tests/components/test_hrm_agent_traced.py](../../tests/components/test_hrm_agent_traced.py)
- [scripts/smoke_test_traced.py](../../scripts/smoke_test_traced.py)

**Configuration:**
- Environment variables set in `.env` file
- Tracing enabled via `LANGSMITH_TRACING=true`
- Project configured via `LANGSMITH_PROJECT` variable

---

## Summary Table

| Component | File Path | Line Range | Key Function/Class |
|-----------|-----------|------------|-------------------|
| FastAPI Query Route | src/api/rest_server.py | 336-404 | `process_query()` |
| HRM Decomposition | src/agents/hrm_agent.py | 324-346 | `HRMAgent.decompose_problem()` |
| MCTS UCB1 | src/framework/mcts/policies.py | 25-51 | `ucb1()` |
| LangSmith Tracing | tests/utils/langsmith_tracing.py | 1-460 | `@trace_e2e_test()` |

---

## Lab 2: Trace a Sample Query

**Objective:** Follow a complete query through the system using LangSmith tracing.

**Status:** Ready to execute
