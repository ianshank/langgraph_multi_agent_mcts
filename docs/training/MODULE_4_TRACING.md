# Module 4: LangSmith Tracing Utilities & Patterns

**Duration:** 10 hours (2 days)
**Format:** Workshop + Observability Lab
**Difficulty:** Intermediate
**Prerequisites:** Completed Module 3, basic understanding of distributed tracing

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Instrument code** with LangSmith tracing decorators and context managers
2. **Design trace hierarchies** for complex multi-agent workflows
3. **Add metadata and tags** for effective filtering and analysis
4. **Debug failures** using comprehensive trace data
5. **Build dashboards** for monitoring system behavior and performance

---

## Session 1: Tracing Fundamentals (2.5 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [LANGSMITH_E2E.md](../LANGSMITH_E2E.md) - E2E workflow tracing guide
- [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md) - Agent-specific tracing patterns
- [tests/utils/langsmith_tracing.py](../../tests/utils/langsmith_tracing.py) - Tracing utilities

### Lecture: LangSmith Architecture (60 minutes)

#### What is LangSmith?

**LangSmith** is an observability platform for LangChain/LangGraph applications that provides:
- **Distributed tracing:** Track requests across multiple agents and LLM calls
- **Performance monitoring:** Measure latency, token usage, and cost
- **Debugging:** Inspect inputs/outputs at every step
- **Experimentation:** Compare different models and configurations
- **Dataset management:** Create test sets for evaluation

#### Trace Hierarchy Model

**Hierarchical Structure:**
```
Root Run (E2E Test or API Request)
├─ Input Validation Run
├─ RAG Retrieval Run
│  └─ Vector Search Run
├─ Agent Processing Run
│  ├─ HRM Agent Run
│  │  ├─ Task Decomposition Run
│  │  │  └─ LLM Call (gpt-4o)
│  │  └─ Confidence Calculation Run
│  ├─ TRM Agent Run
│  │  ├─ Initial Generation Run
│  │  │  └─ LLM Call (gpt-4o)
│  │  ├─ Critique Run
│  │  │  └─ LLM Call (gpt-4o)
│  │  ├─ Refinement Run
│  │  │  └─ LLM Call (gpt-4o)
│  │  └─ Convergence Check Run
│  └─ MCTS Simulation Run (optional)
│     ├─ Selection Phase Run
│     ├─ Expansion Phase Run
│     ├─ Simulation Rollout Run
│     └─ Backpropagation Run
├─ Consensus Calculation Run
└─ Response Generation Run
```

**Key Concepts:**
- **Root Run:** Top-level operation (e.g., API request, E2E test)
- **Child Runs:** Nested operations that inherit trace context
- **Span:** Individual operation with start/end times
- **Metadata:** Key-value pairs attached to runs
- **Tags:** Labels for filtering and categorization

#### Automatic vs. Manual Tracing

**Automatic Tracing:**
- LangChain/LangGraph operations auto-traced when `LANGSMITH_TRACING=true`
- LLM calls, chains, and agents traced by default
- No code changes required

**Manual Tracing:**
- Custom functions and logic need explicit instrumentation
- Use `@traceable` decorator or context managers
- Required for non-LangChain code

**Example:**
```python
from langsmith import traceable

# Automatic: LangChain operations
chain = LLMChain(llm=ChatOpenAI(), prompt=prompt)
result = chain.invoke({"query": "test"})  # Auto-traced

# Manual: Custom function
@traceable(name="custom_logic", run_type="tool")
def custom_processing(data):
    # Your logic here
    return processed_data

result = custom_processing(data)  # Now traced!
```

### Lecture: Configuration & Setup (45 minutes)

#### Environment Variables

**Required Configuration:**
```bash
# .env file
LANGSMITH_API_KEY=lsv2_pt_...              # Your API key
LANGSMITH_ORG_ID=196445bb-...              # Organization ID
LANGSMITH_TRACING=true                      # Enable tracing
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=langgraph-multi-agent-mcts  # Project name
```

**Programmatic Configuration:**
```python
import os

# Set before importing LangChain
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"

# Or use LangChain settings
from langchain_core.tracers.langchain import LangChainTracer

tracer = LangChainTracer(project_name="my-project")
```

#### Project Organization

**Best Practices:**
- **One project per application:** e.g., "langgraph-multi-agent-mcts"
- **Use tags for categorization:** e.g., `["e2e", "production", "experiment"]`
- **Metadata for filtering:** e.g., `{"environment": "staging", "version": "v1.2.0"}`

#### Client Initialization

**Using LangSmith Client:**
```python
from langsmith import Client

# Initialize client
client = Client()

# List recent runs
runs = list(client.list_runs(
    project_name="langgraph-multi-agent-mcts",
    limit=10
))

# Get specific run
run = client.read_run(run_id="...")

# Create dataset
dataset = client.create_dataset(
    dataset_name="test_scenarios",
    description="Test scenarios for evaluation"
)
```

### Live Demo: First Trace (15 minutes)

**Instructor Demo:**

```python
import os
from langsmith import traceable
from langchain_openai import ChatOpenAI

# Ensure tracing is enabled
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "demo-project"

@traceable(name="demo_function", run_type="tool")
def process_query(query: str) -> dict:
    """Process a query with tracing."""
    # This will be traced
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(query)

    return {
        "query": query,
        "response": response.content,
        "tokens": response.response_metadata.get("token_usage", {})
    }

# Execute
result = process_query("What is the capital of France?")
print(result)

# Check trace: https://smith.langchain.com/
```

### Discussion & Q&A (10 minutes)

**Key Questions:**
1. What's the performance overhead of tracing?
2. How long are traces retained?
3. Can you disable tracing for specific functions?
4. How do you handle sensitive data in traces?

---

## Session 2: Tracing Decorators (2.5 hours)

### Pre-Reading (30 minutes)

- [tests/utils/langsmith_tracing.py](../../tests/utils/langsmith_tracing.py) - Custom decorators
- LangSmith Python SDK docs: https://docs.smith.langchain.com/

### Lecture: Decorator Patterns (60 minutes)

#### Basic @traceable Decorator

**Signature:**
```python
@traceable(
    name: str = None,           # Run name (defaults to function name)
    run_type: str = "chain",    # Run type: "llm", "chain", "tool", "retriever"
    tags: list = None,          # Tags for filtering
    metadata: dict = None,      # Additional metadata
)
```

**Example:**
```python
from langsmith import traceable

@traceable(name="decompose_task", run_type="tool", tags=["hrm", "decomposition"])
def decompose_task(query: str) -> list[str]:
    """Decompose query into subtasks."""
    # Function logic
    tasks = [...]
    return tasks
```

#### Custom E2E Test Decorator

**Implementation from tests/utils/langsmith_tracing.py:**
```python
def trace_e2e_test(
    scenario: str,
    phase: str = None,
    tags: list[str] = None,
    metadata: dict[str, Any] = None,
    **kwargs
):
    """
    Decorator for E2E tests with comprehensive tracing.

    Args:
        scenario: Test scenario name
        phase: Test phase (validation, processing, integration)
        tags: Additional tags for filtering
        metadata: Custom metadata
        **kwargs: Additional traceable parameters

    Usage:
        @trace_e2e_test("tactical_analysis", phase="validation", tags=["tactical"])
        def test_tactical_flow():
            ...
    """
    def decorator(func):
        # Build metadata
        run_metadata = get_test_metadata()
        run_metadata.update({
            "test_suite": "e2e",
            "scenario": scenario,
            "phase": phase or "unknown",
        })
        if metadata:
            run_metadata.update(metadata)

        # Build tags
        run_tags = ["e2e", "test"]
        if tags:
            run_tags.extend(tags)
        if phase:
            run_tags.append(f"phase:{phase}")

        # Apply traceable decorator
        return traceable(
            name=f"e2e_{scenario}",
            run_type="chain",
            tags=run_tags,
            metadata=run_metadata,
            **kwargs
        )(func)

    return decorator
```

**Usage:**
```python
@trace_e2e_test(
    scenario="hrm_tactical_decomposition",
    phase="agent_processing",
    tags=["hrm", "tactical"],
    metadata={"test_type": "unit"}
)
def test_hrm_tactical():
    """Test HRM with tactical query."""
    agent = HRMAgent(llm=ChatOpenAI())
    result = agent.decompose("Urban warfare tactics")
    assert result["confidence"] >= 0.7
```

#### Agent-Specific Decorators

**HRM Agent Tracing:**
```python
def trace_hrm_agent(
    query_type: str = "general",
    expected_confidence: float = 0.7
):
    """Trace HRM agent execution."""
    def decorator(func):
        return trace_e2e_test(
            scenario=f"hrm_{query_type}",
            phase="decomposition",
            tags=["hrm", query_type],
            metadata={
                "agent": "hrm",
                "expected_confidence": expected_confidence,
            }
        )(func)
    return decorator

# Usage
@trace_hrm_agent(query_type="tactical", expected_confidence=0.8)
def test_hrm_tactical_query():
    ...
```

**TRM Agent Tracing:**
```python
def trace_trm_agent(
    max_iterations: int = 5,
    convergence_threshold: float = 0.05
):
    """Trace TRM agent execution."""
    def decorator(func):
        return trace_e2e_test(
            scenario="trm_refinement",
            phase="refinement",
            tags=["trm", "refinement"],
            metadata={
                "agent": "trm",
                "max_iterations": max_iterations,
                "convergence_threshold": convergence_threshold,
            }
        )(func)
    return decorator
```

**MCTS Tracing:**
```python
def trace_mcts_search(
    iterations: int = 100,
    exploration_constant: float = 1.41
):
    """Trace MCTS search execution."""
    def decorator(func):
        return trace_e2e_test(
            scenario="mcts_search",
            phase="simulation",
            tags=["mcts", "search"],
            metadata={
                "agent": "mcts",
                "iterations": iterations,
                "exploration_constant": exploration_constant,
            }
        )(func)
    return decorator
```

### Hands-On Exercise: Create Custom Decorators (60 minutes)

**Exercise 1: Performance Monitoring Decorator**

**Objective:** Create a decorator that tracks execution time and adds performance metadata.

**Requirements:**
1. Measure function execution time
2. Add timing metadata to trace
3. Log warning if execution exceeds threshold
4. Support both sync and async functions

**Starter Code:**
```python
import time
import functools
from langsmith import traceable

def trace_performance(
    max_latency_ms: float = 1000,
    warn_on_slow: bool = True
):
    """
    Trace function with performance monitoring.

    Args:
        max_latency_ms: Maximum expected latency
        warn_on_slow: Whether to log warnings for slow execution
    """
    def decorator(func):
        @functools.wraps(func)
        @traceable(
            name=func.__name__,
            run_type="tool",
            tags=["performance"],
        )
        def wrapper(*args, **kwargs):
            # TODO: Measure execution time
            start = time.time()

            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # TODO: Calculate elapsed time
                elapsed_ms = (time.time() - start) * 1000

                # TODO: Add metadata to current trace
                # Hint: Use langsmith.run_helpers.get_current_run_tree()

                # TODO: Log warning if slow
                if warn_on_slow and elapsed_ms > max_latency_ms:
                    print(f"⚠️  Slow execution: {func.__name__} took {elapsed_ms:.2f}ms")

            return result

        return wrapper
    return decorator

# TODO: Test your decorator
@trace_performance(max_latency_ms=500, warn_on_slow=True)
def slow_function():
    """Simulate slow function."""
    time.sleep(0.6)  # 600ms
    return "done"

slow_function()  # Should log warning
```

**Deliverable:** Complete decorator with tests

---

## Session 3: Advanced Patterns (3 hours)

### Pre-Reading (30 minutes)

- [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md) - Advanced tracing patterns
- [examples/langgraph_multi_agent_mcts.py](../../examples/langgraph_multi_agent_mcts.py) - Production tracing

### Lecture: Context Managers (45 minutes)

#### Tracing Context Manager

**Basic Usage:**
```python
from langsmith.run_helpers import tracing_context

# Create traced context
with tracing_context(
    project_name="my-project",
    tags=["context", "test"],
    metadata={"version": "1.0"}
):
    # All LangChain operations traced with context
    llm = ChatOpenAI()
    result = llm.invoke("test query")
```

**Custom Context Manager:**
```python
from contextlib import contextmanager
from langsmith.run_helpers import tracing_context
import time

@contextmanager
def trace_e2e_workflow(
    scenario: str,
    tags: list[str] = None,
    metadata: dict = None
):
    """
    Context manager for tracing E2E workflows.

    Usage:
        with trace_e2e_workflow("tactical_analysis", tags=["tactical"]):
            result = run_workflow()
    """
    # Setup
    run_metadata = get_test_metadata()
    if metadata:
        run_metadata.update(metadata)

    run_tags = ["e2e", "workflow"]
    if tags:
        run_tags.extend(tags)

    start_time = time.time()

    # Enter traced context
    with tracing_context(
        project_name=os.getenv("LANGSMITH_PROJECT"),
        tags=run_tags,
        metadata=run_metadata
    ):
        try:
            yield
        finally:
            # Cleanup / final metrics
            elapsed = time.time() - start_time
            print(f"Workflow '{scenario}' completed in {elapsed:.2f}s")
```

#### Nested Tracing

**Example: Nested Operations**
```python
from langsmith import traceable

@traceable(name="outer_operation", run_type="chain")
def outer_operation(query: str):
    """Outer operation with nested tracing."""
    # Preprocessing
    processed = preprocess(query)

    # Call inner operation (will be nested)
    result = inner_operation(processed)

    # Postprocessing
    return postprocess(result)

@traceable(name="inner_operation", run_type="tool")
def inner_operation(data: str):
    """Inner operation (nested trace)."""
    # This will appear as child in trace hierarchy
    return data.upper()

@traceable(name="preprocess", run_type="tool")
def preprocess(text: str):
    """Preprocessing step."""
    return text.strip().lower()

# Execute - creates nested trace hierarchy
result = outer_operation("  Test Query  ")
```

### Lecture: Metadata & Tags Strategy (60 minutes)

#### Metadata Design

**Structured Metadata:**
```python
metadata = {
    # Test Context
    "test_suite": "e2e",
    "test_name": "tactical_analysis_flow",
    "test_id": "test_001",

    # Environment
    "environment": "ci",  # ci, local, staging, production
    "ci_branch": "main",
    "ci_commit": "a1b2c3d",
    "ci_run_id": "12345",

    # Configuration
    "model": "gpt-4o",
    "temperature": 0.7,
    "use_mcts": True,
    "mcts_iterations": 200,

    # Agent-Specific
    "agent": "hrm",
    "confidence_threshold": 0.7,
    "max_refinement_iterations": 5,

    # Performance
    "elapsed_ms": 1234.56,
    "token_count": 1500,
    "cost_usd": 0.05,

    # Outcome
    "success": True,
    "error_type": None,
    "confidence_score": 0.85,
}
```

**Adding Metadata to Traces:**
```python
from langsmith.run_helpers import get_current_run_tree

@traceable(name="my_function")
def my_function(query: str):
    """Function with dynamic metadata."""
    # Get current trace
    run_tree = get_current_run_tree()

    # Execute logic
    result = process(query)

    # Add metadata dynamically
    if run_tree:
        run_tree.add_metadata({
            "result_length": len(result),
            "processing_stage": "complete",
        })

    return result
```

#### Tag Strategy

**Tag Categories:**

1. **Test Type:** `"e2e"`, `"unit"`, `"integration"`, `"performance"`
2. **Agent:** `"hrm"`, `"trm"`, `"mcts"`
3. **Phase:** `"phase:validation"`, `"phase:processing"`, `"phase:synthesis"`
4. **Domain:** `"scenario:tactical"`, `"scenario:cybersecurity"`
5. **Environment:** `"env:ci"`, `"env:production"`
6. **Feature:** `"feature:rag"`, `"feature:consensus"`
7. **Experiment:** `"experiment"`, `"exp:mcts_200"`

**Example Tag Usage:**
```python
@trace_e2e_test(
    scenario="tactical_hrm_trm",
    tags=[
        "e2e",
        "hrm",
        "trm",
        "scenario:tactical",
        "phase:integration",
        "env:ci"
    ]
)
def test_tactical_hrm_trm():
    ...
```

### Hands-On Exercise: Implement Comprehensive Tracing (75 minutes)

**Exercise 2: Trace a Multi-Agent Workflow**

**Objective:** Add comprehensive tracing to a complete workflow with metadata and tags.

**Tasks:**

1. **Instrument workflow nodes:**
```python
@traceable(name="initialize_workflow", run_type="chain", tags=["setup"])
def initialize_state(state: AgentState) -> AgentState:
    """Initialize workflow with tracing."""
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_metadata({
            "query_length": len(state["query"]),
            "use_rag": state.get("use_rag", False),
            "use_mcts": state.get("use_mcts", False),
        })

    # Initialization logic
    return {**state, "current_phase": "initialized"}
```

2. **Add agent-specific tracing:**
```python
@traceable(name="hrm_agent", run_type="chain", tags=["hrm", "decomposition"])
def run_hrm_agent(state: AgentState) -> AgentState:
    """Run HRM with detailed tracing."""
    start = time.time()

    # Execute HRM
    result = hrm.decompose(state["query"])

    # Add performance metadata
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_metadata({
            "hrm_confidence": result["confidence"],
            "task_count": len(result["tasks"]),
            "elapsed_ms": (time.time() - start) * 1000,
        })

    return {
        **state,
        "hrm_results": result,
        "confidence_scores": {
            **state.get("confidence_scores", {}),
            "hrm": result["confidence"]
        }
    }
```

3. **Add conditional routing tracing:**
```python
@traceable(name="route_decision", run_type="tool", tags=["routing"])
def should_run_mcts(state: AgentState) -> str:
    """Routing decision with tracing."""
    decision = "run_mcts" if state.get("use_mcts") else "skip_mcts"

    # Add routing metadata
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.add_metadata({
            "routing_decision": decision,
            "use_mcts": state.get("use_mcts", False),
            "confidence": state["confidence_scores"].get("trm", 0),
        })

    return decision
```

4. **Write test with full tracing:**
```python
@trace_e2e_test(
    scenario="multi_agent_traced_workflow",
    phase="integration",
    tags=["hrm", "trm", "mcts", "full_stack"],
    metadata={"test_category": "comprehensive"}
)
def test_traced_workflow():
    """Test fully traced workflow."""
    workflow = create_traced_workflow()

    result = workflow.invoke({
        "query": "Develop urban defense strategy",
        "use_rag": True,
        "use_mcts": True,
    })

    # Verify tracing
    assert result["current_phase"] == "completed"
    # Check trace in LangSmith UI
```

**Deliverable:** Traced workflow with comprehensive metadata and tests

---

## Session 4: Dashboards and Filtering (2 hours)

### Lecture: LangSmith UI Navigation (30 minutes)

#### Projects & Runs View

**Project Dashboard:**
- Overview of all traces in project
- Performance metrics (latency, cost, token usage)
- Success/failure rates
- Recent runs

**Filtering Syntax:**

1. **By Tags:**
```
tags: experiment
tags: e2e AND tags: tactical
tags: hrm OR tags: trm
```

2. **By Metadata:**
```
metadata.environment: "ci"
metadata.use_mcts: true
metadata.elapsed_ms > 1000
metadata.confidence > 0.8
```

3. **By Date:**
```
created_at > "2025-01-15"
created_at < "2025-01-20"
```

4. **By Success:**
```
success: true
success: false
error_type: "ValidationError"
```

5. **Combined Filters:**
```
tags: experiment AND metadata.model: "gpt-4o" AND success: true
```

### Hands-On Exercise: Create Custom Dashboards (60 minutes)

**Exercise 3: Build Performance Dashboard**

**Objective:** Create dashboards for monitoring agent performance.

**Tasks:**

1. **Agent Performance Dashboard**
   - Filter: `tags: e2e AND (tags: hrm OR tags: trm OR tags: mcts)`
   - Metrics:
     - Average latency by agent
     - Confidence score distribution
     - Success rate by agent

2. **Experiment Comparison Dashboard**
   - Filter: `tags: experiment`
   - Group by: `metadata.experiment_name`
   - Metrics:
     - Latency comparison across experiments
     - Confidence trends
     - Cost analysis

3. **Error Analysis Dashboard**
   - Filter: `success: false`
   - Group by: `error_type`
   - Metrics:
     - Error frequency
     - Error distribution by agent
     - Time to failure

4. **CI/CD Monitoring Dashboard**
   - Filter: `metadata.environment: "ci"`
   - Group by: `metadata.ci_branch`
   - Metrics:
     - Test pass rate
     - Average test duration
     - Flaky test identification

**Deliverable:** Screenshots and saved dashboard configurations

### Live Demo: Debugging with Traces (30 minutes)

**Instructor Demo:**

**Scenario:** TRM agent not converging

1. **Find failing test in LangSmith:**
```
tags: e2e AND tags: trm AND success: false
```

2. **Inspect trace hierarchy:**
   - Navigate to TRM agent span
   - Check refinement iterations
   - Examine confidence scores per iteration

3. **Analyze inputs/outputs:**
   - Review initial solution quality
   - Check critique outputs
   - Identify where improvement plateaued

4. **Identify root cause:**
   - Convergence threshold too strict
   - Max iterations too low
   - Critique prompts not effective

5. **Fix and verify:**
   - Adjust configuration
   - Re-run test with tracing
   - Confirm fix in new trace

---

## Module 4 Assessment

### Practical Assessment

**Task:** Implement comprehensive tracing for a multi-agent workflow

**Requirements:**
1. Create custom decorators for each agent type (HRM, TRM, MCTS)
2. Instrument complete workflow with nested tracing
3. Add performance monitoring with metadata
4. Implement error tracking and debugging aids
5. Create custom dashboard with 3+ charts
6. Write test suite with 80%+ tracing coverage
7. Document tracing strategy

**Deliverable:**
- Traced workflow implementation (30 points)
- Custom decorators and utilities (25 points)
- Dashboard with analysis (20 points)
- Test suite (15 points)
- Documentation (10 points)

**Total:** 100 points (passing: 70+)

**Submission:** Git branch + LangSmith dashboard URLs

---

## Assessment Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Tracing Implementation** | 30% | Complete workflow instrumentation |
| **Custom Decorators** | 25% | Reusable, well-designed decorators |
| **Dashboard** | 20% | Useful charts and filtering |
| **Tests** | 15% | Comprehensive test coverage |
| **Documentation** | 10% | Clear tracing strategy documentation |

**Minimum Passing:** 70% overall

---

## Additional Resources

### Reading
- [LANGSMITH_E2E.md](../LANGSMITH_E2E.md) - E2E tracing guide
- [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md) - Agent tracing patterns
- LangSmith Docs: https://docs.smith.langchain.com/

### Code Examples
- [tests/utils/langsmith_tracing.py](../../tests/utils/langsmith_tracing.py) - Tracing utilities
- [tests/e2e/test_agent_specific_flows.py](../../tests/e2e/test_agent_specific_flows.py) - E2E tests

### Video Tutorials
- LangSmith UI walkthrough (to be recorded)
- Debugging with traces tutorial (to be recorded)

### Office Hours
- When: [Schedule TBD]
- Topics: Tracing patterns, dashboard design, debugging strategies

---

## Next Module

Continue to [MODULE_5_EXPERIMENTS.md](MODULE_5_EXPERIMENTS.md) - Experiments & Datasets in LangSmith

**Prerequisites for Module 5:**
- Completed Module 4 practical assessment
- Familiarity with LangSmith UI and filtering
- Understanding of experiment design principles
