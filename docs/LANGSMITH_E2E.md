# LangSmith End-to-End Workflow Tracing Guide

## Overview

This document describes how to use LangSmith for tracing end-to-end (E2E) test workflows in the LangGraph Multi-Agent MCTS framework. LangSmith provides comprehensive observability for LangChain/LangGraph applications, enabling you to:

- Trace complete E2E user journeys from test execution
- Debug failures with full execution context
- Monitor performance metrics (latency, token usage)
- Track MCTS simulation quality and decisions
- Correlate test runs with CI/CD pipeline metadata

## Table of Contents

1. [Configuration](#configuration)
2. [Tracing Architecture](#tracing-architecture)
3. [Using the Tracing Utilities](#using-the-tracing-utilities)
4. [E2E Test Instrumentation](#e2e-test-instrumentation)
5. [CI/CD Integration](#cicd-integration)
6. [Viewing Traces in LangSmith UI](#viewing-traces-in-langsmith-ui)
7. [Filtering and Dashboards](#filtering-and-dashboards)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Configuration

### Environment Variables

LangSmith tracing is configured via environment variables in `.env`:

```bash
# LangSmith Configuration
LANGSMITH_API_KEY=lsv2_pt_...           # Your LangSmith API key
LANGSMITH_ORG_ID=196445bb-...           # Organization ID
LANGSMITH_TRACING=true                   # Enable tracing
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=langgraph-multi-agent-mcts  # Project name
```

### Dependencies

The `langsmith` package is already included in project dependencies:

```bash
pip install langsmith  # Installed with langchain/langgraph
```

Verify installation:

```bash
pip show langsmith
```

---

## Tracing Architecture

### Trace Hierarchy

LangSmith traces are organized hierarchically:

```
Root Run (E2E Test)
├─ Input Validation
├─ Agent Processing
│  ├─ HRM Agent Call
│  │  └─ LLM Call (OpenAI/Anthropic)
│  └─ TRM Agent Call
│     └─ LLM Call
├─ MCTS Simulation (if enabled)
│  ├─ Selection Phase
│  ├─ Expansion Phase
│  ├─ Simulation/Rollout
│  └─ Backpropagation
├─ Consensus Calculation
└─ Response Generation
```

**Root Run**: Each E2E test creates a top-level trace (e.g., `e2e_tactical_analysis_flow`)
**Child Runs**: Nested LangChain/LangGraph operations are auto-traced
**Metadata**: Test context, CI info, performance metrics attached at each level

### Metadata Model

Common metadata fields attached to traces:

| Field | Description | Example |
|-------|-------------|---------|
| `test_suite` | Test category | `"e2e"`, `"api"`, `"mcts"` |
| `test_name` | Specific test name | `"e2e_tactical_analysis_flow"` |
| `phase` | Test phase | `"validation"`, `"processing"`, `"integration"` |
| `scenario_type` | Scenario being tested | `"tactical"`, `"cybersecurity"` |
| `provider` | LLM provider | `"openai"`, `"anthropic"` |
| `use_mcts` | Whether MCTS is enabled | `true` / `false` |
| `mcts_iterations` | Number of MCTS iterations | `100`, `200` |
| `branch` | Git branch | `"main"`, `"develop"` |
| `commit` | Git commit SHA | `"a1b2c3d..."` |
| `ci_run_id` | CI run identifier | GitHub Actions run ID |
| `consensus_score` | Agent agreement score | `0.835` |
| `processing_time_ms` | Execution time | `1500` |

### Tags

Tags enable filtering in the LangSmith UI:

- **Test suite**: `e2e`, `api`, `mcts`, `smoke`
- **Phase**: `phase:validation`, `phase:processing`
- **Scenario**: `scenario:tactical`, `scenario:cybersecurity`
- **Provider**: `provider:openai`, `provider:anthropic`
- **Features**: `mcts`, `rag`, `hrm`, `trm`, `consensus`
- **Performance**: `performance`, `latency`, `stress`

---

## Using the Tracing Utilities

### Import the Helpers

```python
from tests.utils.langsmith_tracing import (
    trace_e2e_test,
    trace_api_endpoint,
    trace_mcts_simulation,
    update_run_metadata,
    add_run_tag,
)
```

### 1. `@trace_e2e_test` Decorator

Trace a complete E2E test function:

```python
@pytest.mark.e2e
@pytest.mark.asyncio
@trace_e2e_test(
    "e2e_tactical_analysis_flow",
    phase="complete_flow",
    scenario_type="tactical",
    use_mcts=False,
    tags=["hrm", "trm", "consensus"],
)
async def test_tactical_analysis_flow(mock_llm_client, tactical_query):
    """Complete tactical analysis should produce valid results."""
    # Test implementation
    query_input = QueryInput(**tactical_query)

    # Process through agents
    hrm_response = await mock_llm_client.generate(...)
    trm_response = await mock_llm_client.generate(...)

    # Calculate consensus
    consensus = (hrm_conf + trm_conf) / 2

    # Update trace with runtime metrics
    update_run_metadata({
        "consensus_score": consensus,
        "processing_time_ms": 1500,
        "agents_consulted": ["HRM", "TRM"],
    })

    # Assertions
    assert final_response["confidence"] >= 0.75
```

**Parameters**:
- `test_name`: Unique identifier for the test
- `phase`: Test phase (e.g., `"complete_flow"`, `"validation"`)
- `scenario_type`: Type of scenario (e.g., `"tactical"`, `"cybersecurity"`)
- `use_mcts`: Whether MCTS is enabled
- `mcts_iterations`: Number of MCTS iterations (if applicable)
- `tags`: List of tags for filtering
- `metadata`: Additional custom metadata

### 2. `@trace_api_endpoint` Decorator

Trace API endpoint tests:

```python
@pytest.mark.api
@trace_api_endpoint(
    "/query",
    method="POST",
    use_mcts=True,
    use_rag=True,
    tags=["performance"],
)
async def test_query_with_mcts():
    """Test query endpoint with MCTS enabled."""
    response = await client.post(
        "/query",
        json={"query": "...", "use_mcts": True},
    )
    assert response.status_code == 200
```

### 3. `@trace_mcts_simulation` Decorator

Trace MCTS simulation tests:

```python
@pytest.mark.e2e
@pytest.mark.performance
@trace_mcts_simulation(
    iterations=200,
    scenario_type="tactical",
    seed=42,
    max_depth=10,
    tags=["performance", "stress"],
)
def test_200_iterations_latency(tactical_scenario):
    """200 MCTS iterations should complete in <30 seconds."""
    start_time = time.time()

    # Run MCTS simulation
    mcts_tree = MCTSTree(...)
    mcts_tree.simulate(iterations=200)

    elapsed = time.time() - start_time

    # Update trace with metrics
    update_run_metadata({
        "elapsed_time_seconds": elapsed,
        "iterations_per_second": 200 / elapsed,
    })

    assert elapsed < 30.0
```

### 4. `update_run_metadata(metadata: dict)`

Dynamically add metadata during test execution:

```python
update_run_metadata({
    "actual_latency_ms": 1234,
    "consensus_score": 0.85,
    "threat_identified": True,
    "severity": "HIGH",
})
```

This updates the current trace's metadata, visible in the LangSmith UI.

### 5. `add_run_tag(tag: str)`

Add tags dynamically based on runtime conditions:

```python
if error_occurred:
    add_run_tag("error")

if latency > threshold:
    add_run_tag("slow")
```

---

## E2E Test Instrumentation

### Core E2E Tests

**File**: `tests/e2e/test_complete_query_flow.py`

See `tests/e2e/test_complete_query_flow_traced.py` for a fully instrumented example.

**Key Tests to Instrument**:
1. `test_tactical_analysis_flow` - Complete tactical user journey
2. `test_cybersecurity_analysis_flow` - Cybersecurity threat analysis

**Pattern**:
```python
# Add import at top
from tests.utils.langsmith_tracing import trace_e2e_test, update_run_metadata

# Add decorator to test method
@pytest.mark.e2e
@pytest.mark.asyncio
@trace_e2e_test(
    "e2e_tactical_analysis_flow",
    phase="complete_flow",
    scenario_type="tactical",
    use_mcts=False,
    tags=["hrm", "trm", "consensus"],
)
async def test_tactical_analysis_flow(self, mock_llm_client, tactical_query):
    # ... test implementation ...

    # Update trace with results
    update_run_metadata({
        "consensus_score": consensus,
        "processing_time_ms": final_response["processing_time_ms"],
    })
```

### MCTS Simulation Tests

**File**: `tests/e2e/test_mcts_simulation_flow.py`

See `tests/e2e/test_mcts_simulation_flow_traced.py` for examples.

**Key Tests**:
1. `test_100_iterations_latency` - Performance test (100 iterations)
2. `test_200_iterations_latency` - Stress test (200 iterations)
3. `test_win_probability_estimation` - Decision quality

**Pattern**:
```python
from tests.utils.langsmith_tracing import trace_mcts_simulation, update_run_metadata

@pytest.mark.e2e
@pytest.mark.performance
@trace_mcts_simulation(
    iterations=200,
    scenario_type="tactical",
    seed=42,
    tags=["performance"],
)
def test_200_iterations_latency(tactical_scenario):
    # ... MCTS simulation ...

    update_run_metadata({
        "elapsed_time_seconds": elapsed,
        "iterations_per_second": 200 / elapsed,
    })
```

### API Tests

**File**: `tests/api/test_rest_endpoints.py`

Use `@trace_api_endpoint` for key API tests:

```python
from tests.utils.langsmith_tracing import trace_api_endpoint

@pytest.mark.api
@trace_api_endpoint("/query", use_mcts=True, use_rag=True)
async def test_valid_query_request(valid_query_request):
    # ... API test ...
```

### Smoke Tests

**File**: `scripts/smoke_test_traced.py`

A Python wrapper for the bash smoke tests that provides full tracing:

```bash
# Run traced smoke tests locally
python scripts/smoke_test_traced.py --port 8000

# Run in CI
LANGSMITH_TRACING=true python scripts/smoke_test_traced.py
```

The script traces all smoke test endpoints and reports aggregated results to LangSmith.

---

## CI/CD Integration

### GitHub Actions Workflow

A dedicated E2E test workflow with LangSmith tracing is provided:

**File**: `.github/workflows/e2e_with_langsmith.yml`

**Jobs**:
1. **e2e-tests-traced**: Runs all E2E tests with tracing
2. **api-smoke-tests-traced**: Runs API smoke tests
3. **mcts-performance-tests**: Runs MCTS performance tests

### Required Secrets

Configure these GitHub secrets:

| Secret | Description |
|--------|-------------|
| `LANGSMITH_API_KEY` | LangSmith API key |
| `LANGSMITH_ORG_ID` | LangSmith organization ID |
| `OPENAI_API_KEY` | OpenAI API key (for tests) |
| `ANTHROPIC_API_KEY` | Anthropic API key (for tests) |

### CI Metadata

The workflow automatically injects CI metadata into traces:

```yaml
env:
  LANGSMITH_TRACING: true
  LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
  LANGSMITH_PROJECT: langgraph-multi-agent-mcts
  CI: true
  GITHUB_REF_NAME: ${{ github.ref_name }}
  GITHUB_SHA: ${{ github.sha }}
  GITHUB_RUN_ID: ${{ github.run_id }}
```

This enables correlating traces with specific CI runs, branches, and commits.

### Running E2E Tests Locally with Tracing

```bash
# Ensure .env has LANGSMITH_TRACING=true
export LANGSMITH_TRACING=true

# Run E2E tests
pytest tests/e2e/ -m e2e -v

# Run specific test
pytest tests/e2e/test_complete_query_flow.py::TestCompleteFlow::test_tactical_analysis_flow -v

# Run MCTS performance tests
pytest tests/e2e/test_mcts_simulation_flow.py -m performance -v
```

---

## Viewing Traces in LangSmith UI

### Access Your Project

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your organization
3. Select project: **langgraph-multi-agent-mcts**

### Understanding the Trace View

**Trace List** (`Projects` → `langgraph-multi-agent-mcts`):
- Each row is a root-level trace (E2E test run)
- Columns: Name, Status, Latency, Tokens, Timestamp

**Trace Detail** (click on a trace):
- **Tree View**: Hierarchical execution flow
  - Expand nodes to see nested LLM calls
  - View inputs/outputs at each step
- **Metadata**: Test context, CI info, custom metrics
- **Tags**: Filterable labels (e.g., `e2e`, `mcts`, `scenario:tactical`)
- **Timeline**: Execution timeline with durations
- **Tokens**: Token usage for each LLM call

**Example Trace Hierarchy**:
```
e2e_tactical_analysis_flow (Root)
├─ Input Validation
│  └─ QueryInput model validation
├─ HRM Agent Processing
│  └─ ChatOpenAI.generate
│     └─ OpenAI API Call
├─ TRM Agent Processing
│  └─ ChatOpenAI.generate
│     └─ OpenAI API Call
├─ Consensus Calculation
└─ Response Generation
```

### Key Metrics to Monitor

1. **Latency**:
   - E2E test execution time
   - Individual agent processing time
   - LLM call latency

2. **Token Usage**:
   - Total tokens per test
   - Tokens per agent
   - Cost estimation

3. **Quality Metrics** (in metadata):
   - `consensus_score`: Agent agreement
   - `confidence`: Final recommendation confidence
   - `win_probability`: MCTS decision quality

4. **MCTS Metrics**:
   - `iterations_completed`
   - `iterations_per_second`
   - `tree_depth`

---

## Filtering and Dashboards

### Filtering Traces

Use the LangSmith UI filter bar to find specific traces:

**By Test Suite**:
```
tags: e2e
tags: api
tags: mcts
```

**By Scenario Type**:
```
tags: scenario:tactical
tags: scenario:cybersecurity
```

**By Phase**:
```
tags: phase:complete_flow
tags: phase:validation
```

**By Provider**:
```
tags: provider:openai
tags: provider:anthropic
```

**By CI Branch**:
```
metadata.ci_branch: main
metadata.ci_branch: develop
```

**By Test Name**:
```
name: e2e_tactical_analysis_flow
name: e2e_cybersecurity_analysis_flow
```

**By Performance**:
```
tags: performance
tags: latency
latency > 5000  # Traces slower than 5 seconds
```

### Creating Dashboard Views

**Recommended Saved Filters**:

1. **Complete Query Flow Tests**:
   - Filter: `tags: e2e AND name: *_flow`
   - Use: View all complete E2E user journeys

2. **MCTS Simulations**:
   - Filter: `tags: mcts`
   - Use: Monitor MCTS performance and quality

3. **API Smoke Tests**:
   - Filter: `tags: smoke`
   - Use: Track API endpoint health

4. **Failed Tests**:
   - Filter: `status: error OR status: failure`
   - Use: Debug test failures

5. **Slow Tests** (latency > 10s):
   - Filter: `latency > 10000`
   - Use: Identify performance bottlenecks

6. **CI Main Branch**:
   - Filter: `metadata.ci_branch: main`
   - Use: Production-quality test results

### Charts and Analytics

LangSmith provides built-in analytics:

- **Latency Over Time**: Track performance trends
- **Token Usage**: Monitor LLM costs
- **Error Rate**: Identify flaky tests
- **Test Success Rate**: Overall quality metrics

---

## Best Practices

### 1. Trace Naming Conventions

- Use descriptive, hierarchical names: `e2e_tactical_analysis_flow`
- Prefix by test suite: `e2e_*`, `api_*`, `mcts_*`
- Be consistent across tests

### 2. Tagging Strategy

- **Always tag** test suite (`e2e`, `api`, `mcts`)
- Tag scenario types for domain-specific filtering
- Tag providers when testing multi-provider scenarios
- Tag performance-related tests (`performance`, `stress`)

### 3. Metadata Guidelines

- Attach **metrics** that matter for analysis:
  - Latency, consensus scores, win probabilities
- Include **CI context** for correlation:
  - Branch, commit SHA, run ID
- Add **domain-specific** context:
  - Threat actor (cybersecurity), resource usage (tactical)

### 4. Update Metadata Dynamically

Use `update_run_metadata()` to add runtime metrics:

```python
# Good: Add actual results
update_run_metadata({
    "actual_consensus": 0.87,
    "actual_latency_ms": 1245,
})

# Avoid: Static metadata (use decorator params instead)
```

### 5. Avoid Over-Tracing

- Don't trace every helper function manually
- Let LangChain/LangGraph auto-trace LLM calls
- Focus on **E2E entry points** and **key decision points**

### 6. Test Locally Before CI

```bash
# Verify tracing works locally
export LANGSMITH_TRACING=true
pytest tests/e2e/test_complete_query_flow.py::TestCompleteFlow::test_tactical_analysis_flow -v

# Check trace in LangSmith UI
```

### 7. Use Tracing for Debugging

When a test fails in CI:
1. Find the trace in LangSmith using CI run ID
2. Expand the tree to see exact failure point
3. Inspect inputs/outputs at each step
4. Check metadata for anomalies (e.g., high latency)

---

## Troubleshooting

### Traces Not Appearing in LangSmith

**Check**:
1. `LANGSMITH_TRACING=true` in environment
2. `LANGSMITH_API_KEY` is valid
3. `LANGSMITH_PROJECT` matches your project name
4. No network/firewall issues blocking `api.smith.langchain.com`

**Debug**:
```python
from tests.utils.langsmith_tracing import is_tracing_enabled

print(f"Tracing enabled: {is_tracing_enabled()}")
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'tests.utils.langsmith_tracing'`

**Fix**:
- Ensure `tests/utils/__init__.py` exists
- Run tests from project root
- Install package in editable mode: `pip install -e .`

### Metadata Not Updating

**Error**: `update_run_metadata()` doesn't reflect in UI

**Cause**: Not inside a traced context

**Fix**:
- Only call `update_run_metadata()` inside a `@trace_*` decorated function
- Verify trace appears in UI first

### CI Secrets Not Configured

**Error**: `LANGSMITH_API_KEY` not found in CI

**Fix**:
- Configure GitHub secrets: Settings → Secrets and variables → Actions
- Add `LANGSMITH_API_KEY` and `LANGSMITH_ORG_ID`

### Performance Impact

**Concern**: Does tracing slow down tests?

**Answer**:
- Minimal overhead (<5% latency increase)
- LangSmith batches and buffers trace data
- Tracing is asynchronous and non-blocking

**Best Practice**:
- Enable tracing in CI and staging
- Optionally disable in high-frequency local test runs

---

## Advanced Topics

### Creating Test Datasets

For evaluation and regression testing:

```python
from tests.utils.langsmith_tracing import create_test_dataset

examples = [
    {
        "inputs": {"query": "Analyze tactical situation..."},
        "outputs": {"recommendation": "Secure Alpha position", "confidence": 0.85}
    },
    # ... more examples
]

dataset_id = create_test_dataset("tactical_scenarios", examples)
```

Use datasets for:
- Running evaluations over E2E scenarios
- Regression testing against prior outputs
- Scoring responses with custom evaluators

### Integrating with OpenTelemetry

LangSmith supports OpenTelemetry for unified observability:

- Export traces to OTLP-compatible backends (Datadog, Honeycomb, etc.)
- Correlate LangSmith traces with application metrics
- See [LangChain blog on OTEL](https://blog.langchain.com/end-to-end-opentelemetry-langsmith)

### Programmatic Trace Analysis

Use the LangSmith client for automation:

```python
from tests.utils.langsmith_tracing import get_langsmith_client

client = get_langsmith_client()

# List recent runs
runs = client.list_runs(
    project_name="langgraph-multi-agent-mcts",
    filter="tags: e2e AND metadata.ci_branch: main",
    limit=100,
)

for run in runs:
    print(f"{run.name}: {run.status}, latency={run.latency_ms}ms")
```

---

## Resources

- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangChain Tracing Guide](https://python.langchain.com/docs/langsmith/walkthrough)
- [OpenTelemetry + LangSmith](https://blog.langchain.com/end-to-end-opentelemetry-langsmith)
- [LangSmith Python Client](https://github.com/langchain-ai/langsmith-sdk)

---

## Quick Reference

### Environment Variables

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=langgraph-multi-agent-mcts
LANGSMITH_ORG_ID=196445bb-...
```

### Common Decorators

```python
# E2E test
@trace_e2e_test(
    "e2e_test_name",
    phase="complete_flow",
    scenario_type="tactical",
    use_mcts=False,
    tags=["hrm", "trm"],
)

# API test
@trace_api_endpoint(
    "/query",
    method="POST",
    use_mcts=True,
    tags=["performance"],
)

# MCTS test
@trace_mcts_simulation(
    iterations=200,
    scenario_type="tactical",
    seed=42,
    tags=["performance"],
)
```

### Updating Metadata

```python
update_run_metadata({
    "consensus_score": 0.87,
    "processing_time_ms": 1500,
})
```

### Running Tests

```bash
# Local E2E tests with tracing
export LANGSMITH_TRACING=true
pytest tests/e2e/ -m e2e -v

# Traced smoke tests
python scripts/smoke_test_traced.py --port 8000
```

### LangSmith UI

- **Project**: [smith.langchain.com/projects/langgraph-multi-agent-mcts](https://smith.langchain.com/)
- **Filter Examples**:
  - `tags: e2e`
  - `tags: scenario:tactical`
  - `metadata.ci_branch: main`
  - `latency > 5000`

---

**Last Updated**: 2025-01-17

For questions or issues, refer to the [LangSmith documentation](https://docs.smith.langchain.com/) or open an issue in the project repository.
