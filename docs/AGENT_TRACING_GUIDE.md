# Agent-Specific Tracing Guide (HRM, TRM, MCTS)

## Overview

This guide shows how to apply LangSmith tracing to all agents and subagents in the LangGraph Multi-Agent MCTS framework. We provide three layers of tracing coverage:

1. **Agent-Specific E2E Tests** - Isolate each agent (HRM, TRM, MCTS)
2. **Component-Level Tests** - Test agent internals and behavior
3. **LangGraph Node-Level Tracing** - Trace graph execution

## Table of Contents

- [Quick Start](#quick-start)
- [Agent-Specific E2E Tests](#agent-specific-e2e-tests)
- [Component-Level Tests](#component-level-tests)
- [LangGraph Integration](#langgraph-integration)
- [Filtering by Agent in LangSmith](#filtering-by-agent-in-langsmith)
- [Metadata Conventions](#metadata-conventions)

---

## Quick Start

### Run Agent-Specific Tests

```bash
# All agent-specific E2E tests
pytest tests/e2e/test_agent_specific_flows.py -v

# HRM-only tests
pytest tests/e2e/test_agent_specific_flows.py::TestHRMOnlyFlows -v

# TRM-only tests
pytest tests/e2e/test_agent_specific_flows.py::TestTRMOnlyFlows -v

# MCTS-only tests
pytest tests/e2e/test_agent_specific_flows.py::TestMCTSOnlyFlows -v

# Full-stack (HRM + TRM + MCTS)
pytest tests/e2e/test_agent_specific_flows.py::TestFullStackFlows -v
```

### Run Component Tests

```bash
# HRM component tests
pytest tests/components/test_hrm_agent_traced.py -v

# TRM component tests
pytest tests/components/test_trm_agent_traced.py -v

# MCTS component tests
pytest tests/components/test_mcts_agent_traced.py -v
```

---

## Agent-Specific E2E Tests

### File: `tests/e2e/test_agent_specific_flows.py`

This file contains isolated tests for each agent plus combined flows.

### 1. HRM-Only Tests

**Purpose**: Test Hierarchical Reasoning Model in isolation

**Pattern**:
```python
from tests.utils.langsmith_tracing import trace_e2e_test

@pytest.mark.e2e
@pytest.mark.asyncio
@trace_e2e_test(
    "e2e_hrm_tactical_flow",
    phase="hrm_only",
    scenario_type="tactical",
    provider="openai",
    use_mcts=False,
    tags=["hrm", "hierarchical_reasoning", "tactical"],
)
async def test_hrm_tactical_analysis(mock_llm_client, tactical_query):
    """HRM-only tactical analysis."""

    # Process through HRM only
    hrm_response = await mock_llm_client.generate(f"HRM: {query}")

    # Extract HRM-specific metrics
    objectives_count = ...
    subtasks_count = ...

    # Update trace with HRM metadata
    update_run_metadata({
        "agent": "hrm",
        "hierarchical_objectives": objectives_count,
        "subtasks_identified": subtasks_count,
        "hrm_confidence": 0.87,
        "decomposition_depth": 3,
    })
```

**Key Tests**:
- `test_hrm_tactical_analysis` - Tactical scenario with HRM
- `test_hrm_cybersecurity_analysis` - Cybersecurity scenario with HRM

**Metadata to Track**:
- `hierarchical_objectives`: Number of objectives identified
- `subtasks_identified`: Number of subtasks generated
- `decomposition_depth`: Depth of hierarchical decomposition
- `hrm_confidence`: HRM confidence score

### 2. TRM-Only Tests

**Purpose**: Test Task Refinement Model in isolation

**Pattern**:
```python
@pytest.mark.e2e
@pytest.mark.asyncio
@trace_e2e_test(
    "e2e_trm_tactical_flow",
    phase="trm_only",
    scenario_type="tactical",
    provider="openai",
    use_mcts=False,
    tags=["trm", "task_refinement", "tactical"],
)
async def test_trm_tactical_refinement(mock_llm_client, tactical_query):
    """TRM-only tactical refinement."""

    # Process through TRM only
    trm_response = await mock_llm_client.generate(f"TRM: {query}")

    # Extract TRM metrics
    refinement_cycles = ...
    alternatives_evaluated = ...

    # Update trace
    update_run_metadata({
        "agent": "trm",
        "refinement_cycles": refinement_cycles,
        "alternatives_evaluated": alternatives_evaluated,
        "trm_confidence": 0.83,
        "convergence_achieved": True,
    })
```

**Key Tests**:
- `test_trm_tactical_refinement` - Single refinement cycle
- `test_trm_multi_iteration_refinement` - Multiple iterations

**Metadata to Track**:
- `refinement_cycles`: Number of refinement iterations
- `alternatives_evaluated`: Number of alternatives considered
- `trm_confidence`: TRM confidence score
- `convergence_achieved`: Whether refinement converged

### 3. MCTS-Only Tests

**Purpose**: Test Monte Carlo Tree Search in isolation

**Pattern**:
```python
from tests.utils.langsmith_tracing import trace_mcts_simulation

@pytest.mark.e2e
@trace_mcts_simulation(
    iterations=100,
    scenario_type="tactical",
    seed=42,
    max_depth=10,
    tags=["mcts", "simulation", "tactical"],
)
def test_mcts_tactical_simulation(mcts_tactical_scenario):
    """MCTS-only tactical simulation."""

    # Run MCTS simulation
    # ... simulation code ...

    # Update trace with MCTS metrics
    update_run_metadata({
        "agent": "mcts",
        "total_simulations": 100,
        "best_action": best_action,
        "best_win_probability": 0.75,
        "tree_depth": 10,
        "exploration_rate": 0.85,
    })
```

**Key Tests**:
- `test_mcts_tactical_simulation` - 100 iteration tactical simulation
- `test_mcts_incident_response_simulation` - Cybersecurity MCTS

**Metadata to Track**:
- `total_simulations`: Number of MCTS simulations
- `best_action`: Recommended action
- `best_win_probability`: Win probability for best action
- `tree_depth`: Maximum tree depth reached
- `exploration_rate`: Percentage of action space explored

### 4. Full-Stack Combined Tests

**Purpose**: Test HRM + TRM + MCTS together

**Pattern**:
```python
@pytest.mark.e2e
@pytest.mark.asyncio
@trace_e2e_test(
    "e2e_full_stack_tactical_flow",
    phase="complete_flow",
    scenario_type="tactical",
    provider="openai",
    use_mcts=True,
    mcts_iterations=200,
    tags=["hrm", "trm", "mcts", "full_stack", "tactical"],
)
async def test_full_stack_tactical_analysis(...):
    """Full stack: HRM + TRM + MCTS."""

    # Step 1: HRM
    hrm_response = await mock_llm_client.generate(f"HRM: {query}")
    hrm_confidence = 0.87

    # Step 2: TRM
    trm_response = await mock_llm_client.generate(f"TRM: {query}")
    trm_confidence = 0.83

    # Step 3: MCTS
    mcts_best_action = ...
    mcts_win_prob = 0.75

    # Step 4: Consensus
    consensus = (hrm_confidence + trm_confidence + mcts_win_prob) / 3

    # Update comprehensive trace
    update_run_metadata({
        "agents_used": ["hrm", "trm", "mcts"],
        "hrm_confidence": hrm_confidence,
        "trm_confidence": trm_confidence,
        "mcts_win_probability": mcts_win_prob,
        "consensus_score": consensus,
    })
```

**Key Tests**:
- `test_full_stack_tactical_analysis` - Combined tactical flow
- `test_full_stack_cybersecurity_response` - Combined cybersecurity flow

---

## Component-Level Tests

Component tests focus on agent internals and behavior.

### HRM Component Tests

**File**: `tests/components/test_hrm_agent_traced.py`

**Key Tests**:
1. **Task Decomposition**:
   ```python
   @trace_e2e_test(
       "component_hrm_task_decomposition",
       phase="component",
       tags=["hrm", "component", "decomposition"],
   )
   async def test_hierarchical_decomposition_depth(...):
       # Test HRM decomposition depth
   ```

2. **Objective Identification**:
   ```python
   @trace_e2e_test(
       "component_hrm_objective_identification",
       phase="component",
       tags=["hrm", "component", "objectives"],
   )
   async def test_objective_identification_quality(...):
       # Test HRM objective quality
   ```

3. **Confidence Calibration**:
   ```python
   @trace_e2e_test(
       "component_hrm_confidence_calibration",
       phase="component",
       tags=["hrm", "component", "confidence"],
   )
   def test_confidence_score_calibration(...):
       # Test HRM confidence scores
   ```

### TRM Component Tests

**File**: `tests/components/test_trm_agent_traced.py`

**Key Tests**:
1. **Iterative Refinement**:
   ```python
   @trace_e2e_test(
       "component_trm_iterative_refinement",
       phase="component",
       tags=["trm", "component", "refinement"],
   )
   async def test_multi_cycle_refinement(...):
       # Test TRM improvement over cycles
   ```

2. **Convergence Detection**:
   ```python
   @trace_e2e_test(
       "component_trm_convergence_detection",
       phase="component",
       tags=["trm", "component", "convergence"],
   )
   async def test_convergence_detection(...):
       # Test TRM convergence
   ```

3. **Alternative Ranking**:
   ```python
   @trace_e2e_test(
       "component_trm_alternative_ranking",
       phase="component",
       tags=["trm", "component", "alternatives"],
   )
   def test_alternative_ranking_quality(...):
       # Test TRM ranking consistency
   ```

### MCTS Component Tests

**File**: `tests/components/test_mcts_agent_traced.py`

**Key Tests**:
1. **UCB1 Selection**:
   ```python
   @trace_mcts_simulation(
       iterations=100,
       scenario_type="tactical",
       tags=["mcts", "component", "ucb1"],
   )
   def test_ucb1_selection_correctness(...):
       # Test UCB1 algorithm
   ```

2. **Backpropagation**:
   ```python
   @trace_mcts_simulation(
       iterations=200,
       scenario_type="tactical",
       tags=["mcts", "component", "backpropagation"],
   )
   def test_backpropagation_accuracy(...):
       # Test MCTS backpropagation
   ```

3. **Win Probability Accuracy**:
   ```python
   @trace_mcts_simulation(
       iterations=500,
       scenario_type="tactical",
       tags=["mcts", "component", "win_probability"],
   )
   def test_win_probability_accuracy(...):
       # Test win probability estimates
   ```

---

## LangGraph Integration

### Tracing LangGraph Nodes

When you have a LangGraph app that uses HRM, TRM, and MCTS nodes:

```python
from langchain_core.runnables import RunnableConfig

# Create config with agent-specific tags
hrm_config = RunnableConfig(
    run_name="graph_hrm_node",
    tags=["graph", "hrm", "node"],
    metadata={
        "agent": "hrm",
        "scenario": "tactical",
    }
)

# Invoke with config
result = hrm_node.invoke(input_data, config=hrm_config)
```

### Full Graph Execution

```python
from tests.utils.langsmith_tracing import trace_e2e_workflow

# Wrap entire graph execution
with trace_e2e_workflow(
    "langgraph_full_execution",
    tags=["graph", "hrm", "trm", "mcts"],
    metadata={"scenario": "tactical"},
):
    result = app.invoke(query)
```

---

## Filtering by Agent in LangSmith

### Filter by Agent Tag

**HRM traces**:
```
tags: hrm
```

**TRM traces**:
```
tags: trm
```

**MCTS traces**:
```
tags: mcts
```

**Full-stack traces** (all agents):
```
tags: hrm AND tags: trm AND tags: mcts
```

### Filter by Phase

**Agent-only E2E**:
```
tags: hrm AND phase: hrm_only
tags: trm AND phase: trm_only
```

**Component tests**:
```
phase: component AND tags: hrm
phase: component AND tags: trm
phase: component AND tags: mcts
```

**Full flows**:
```
phase: complete_flow
```

### Filter by Test Type

**Decomposition tests**:
```
tags: hrm AND tags: decomposition
```

**Refinement tests**:
```
tags: trm AND tags: refinement
```

**Simulation tests**:
```
tags: mcts AND tags: simulation
```

---

## Metadata Conventions

### HRM-Specific Metadata

| Field | Type | Description |
|-------|------|-------------|
| `agent` | string | Always `"hrm"` |
| `hierarchical_objectives` | integer | Number of objectives identified |
| `subtasks_identified` | integer | Number of subtasks generated |
| `decomposition_depth` | integer | Depth of hierarchy |
| `hrm_confidence` | float | HRM confidence score |

### TRM-Specific Metadata

| Field | Type | Description |
|-------|------|-------------|
| `agent` | string | Always `"trm"` |
| `refinement_cycles` | integer | Number of iterations |
| `alternatives_evaluated` | integer | Number of alternatives |
| `trm_confidence` | float | TRM confidence score |
| `convergence_achieved` | boolean | Whether converged |
| `improvement` | float | Score improvement |

### MCTS-Specific Metadata

| Field | Type | Description |
|-------|------|-------------|
| `agent` | string | Always `"mcts"` |
| `total_simulations` | integer | Number of simulations |
| `best_action` | string | Recommended action |
| `best_win_probability` | float | Win probability |
| `tree_depth` | integer | Tree depth |
| `exploration_rate` | float | Action space explored |

### Multi-Agent Metadata

For full-stack tests with all agents:

| Field | Type | Description |
|-------|------|-------------|
| `agents_used` | list | `["hrm", "trm", "mcts"]` |
| `hrm_confidence` | float | HRM score |
| `trm_confidence` | float | TRM score |
| `mcts_win_probability` | float | MCTS win prob |
| `consensus_score` | float | Average of all scores |

---

## Dashboard Examples

### HRM Dashboard

**Filter**: `tags: hrm`

**Metrics to Track**:
- Decomposition depth over time
- Objective identification rate
- HRM confidence trends
- Latency per decomposition

### TRM Dashboard

**Filter**: `tags: trm`

**Metrics to Track**:
- Refinement cycles per test
- Convergence rate
- TRM confidence improvement
- Alternatives evaluated

### MCTS Dashboard

**Filter**: `tags: mcts`

**Metrics to Track**:
- Simulations per second
- Win probability accuracy
- Tree depth distribution
- Exploration vs exploitation balance

### Multi-Agent Dashboard

**Filter**: `tags: full_stack`

**Metrics to Track**:
- Consensus scores
- Agent agreement rates
- Combined latency
- Full-stack success rate

---

## Integration Checklist

- [ ] Run agent-specific E2E tests locally
- [ ] Verify traces in LangSmith for each agent
- [ ] Run component tests for each agent
- [ ] Check metadata is populated correctly
- [ ] Create saved filters for each agent
- [ ] Set up dashboards for HRM, TRM, MCTS
- [ ] Test full-stack combined flows
- [ ] Verify tags and filtering work as expected

---

## Resources

- [Main LangSmith E2E Guide](LANGSMITH_E2E.md)
- [Agent-Specific Tests](../tests/e2e/test_agent_specific_flows.py)
- [HRM Component Tests](../tests/components/test_hrm_agent_traced.py)
- [TRM Component Tests](../tests/components/test_trm_agent_traced.py)
- [MCTS Component Tests](../tests/components/test_mcts_agent_traced.py)
- [Tracing Utilities](../tests/utils/langsmith_tracing.py)

---

**Last Updated**: 2025-01-17
