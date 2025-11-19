# Module 3: E2E Flows & LangGraph Orchestration

**Duration:** 10 hours (2 days)
**Format:** Workshop + Integration Lab
**Difficulty:** Intermediate
**Prerequisites:** Completed Modules 1 & 2, basic async/await knowledge

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand LangGraph state machine** design and execution model
2. **Implement E2E workflows** with multiple agents and conditional routing
3. **Test E2E flows** using comprehensive test patterns
4. **Design custom state transitions** and implement complex routing logic
5. **Debug E2E failures** using trace data and state inspection

---

## Session 1: LangGraph State Machine (2.5 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [langgraph_mcts_architecture.md](../langgraph_mcts_architecture.md) - Complete architectural design
- [examples/langgraph_multi_agent_mcts.py](../../examples/langgraph_multi_agent_mcts.py) - Main implementation
- LangGraph documentation: https://langchain-ai.github.io/langgraph/

### Lecture: State Management (60 minutes)

#### State Schema Design

**TypedDict for Type Safety:**
```python
from typing import TypedDict, List, Optional, Dict, Any

class AgentState(TypedDict):
    """Shared state across all graph nodes."""
    # Input
    query: str                           # Original user query
    use_rag: bool                        # Enable RAG retrieval
    use_mcts: bool                       # Enable MCTS search

    # RAG Context
    rag_context: Optional[str]           # Retrieved context from vector DB

    # Agent Results
    hrm_results: Optional[Dict[str, Any]]  # HRM decomposition
    trm_results: Optional[Dict[str, Any]]  # TRM refinement
    mcts_tree: Optional[Any]              # MCTS search tree

    # Metadata
    confidence_scores: Dict[str, float]   # Agent confidence scores
    agent_outputs: List[Dict[str, Any]]   # All agent responses
    iteration: int                        # Current iteration
    current_phase: str                    # Current execution phase

    # Performance
    start_time: float                     # Execution start timestamp
    latency_ms: Optional[float]          # Total latency
```

**Benefits of TypedDict:**
- Type checking at development time
- IDE autocompletion
- Self-documenting state structure
- Runtime validation (with Pydantic)

#### Node Functions

**Node Function Signature:**
```python
def node_function(state: AgentState) -> AgentState:
    """
    Process state and return updated state.

    Args:
        state: Current graph state

    Returns:
        Updated state dictionary (can be partial update)
    """
    # Process state
    result = do_work(state["query"])

    # Return state update
    return {
        "current_phase": "next_phase",
        "agent_outputs": state["agent_outputs"] + [result],
    }
```

**Important Rules:**
1. **Must return dict**: Return value updates the state
2. **Partial updates**: Only modified keys need to be returned
3. **Immutability**: Don't modify state in-place, return new values
4. **Type consistency**: Return values must match TypedDict schema

#### Graph Construction

**Example Graph Definition:**
```python
from langgraph.graph import StateGraph, END

# Create graph with state schema
graph = StateGraph(AgentState)

# Add nodes (each is a function)
graph.add_node("initialize", initialize_state)
graph.add_node("retrieve_context", retrieve_rag_context)
graph.add_node("hrm", run_hrm_agent)
graph.add_node("trm", run_trm_agent)
graph.add_node("mcts", run_mcts_agent)
graph.add_node("synthesize", synthesize_results)

# Set entry point
graph.set_entry_point("initialize")

# Add linear edges
graph.add_edge("initialize", "retrieve_context")
graph.add_edge("retrieve_context", "hrm")
graph.add_edge("hrm", "trm")

# Add conditional edge
graph.add_conditional_edges(
    "trm",                    # Source node
    should_run_mcts,          # Decision function
    {
        "run_mcts": "mcts",   # If True, go to MCTS
        "skip_mcts": "synthesize"  # If False, go to synthesis
    }
)

graph.add_edge("mcts", "synthesize")
graph.add_edge("synthesize", END)

# Compile graph
app = graph.compile()
```

### Lecture: Conditional Routing (45 minutes)

#### Router Function Design

**Router Function Signature:**
```python
def should_run_mcts(state: AgentState) -> str:
    """
    Decide whether to run MCTS based on state.

    Args:
        state: Current graph state

    Returns:
        Next node name or routing key
    """
    # Check if MCTS was requested
    if not state.get("use_mcts", False):
        return "skip_mcts"

    # Check if query is suitable for MCTS
    query = state["query"].lower()
    decision_keywords = ["decide", "choose", "optimal", "best", "compare"]

    if any(kw in query for kw in decision_keywords):
        return "run_mcts"

    return "skip_mcts"
```

**Routing Patterns:**

1. **Binary Decision:**
```python
graph.add_conditional_edges(
    "node_a",
    lambda s: "yes" if condition(s) else "no",
    {"yes": "node_b", "no": "node_c"}
)
```

2. **Multi-way Routing:**
```python
def route_by_domain(state: AgentState) -> str:
    """Route to domain-specific handler."""
    domain = detect_domain(state["query"])
    return f"process_{domain}"

graph.add_conditional_edges(
    "classify",
    route_by_domain,
    {
        "process_tactical": "tactical_handler",
        "process_cybersecurity": "cyber_handler",
        "process_general": "general_handler",
    }
)
```

3. **Loop Detection:**
```python
def check_convergence(state: AgentState) -> str:
    """Loop back if not converged."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"

    if state["confidence_scores"]["overall"] >= THRESHOLD:
        return "end"

    return "continue"

graph.add_conditional_edges(
    "evaluate",
    check_convergence,
    {
        "continue": "refine",  # Loop back
        "end": END
    }
)
```

### Live Demo: Graph Execution (15 minutes)

**Instructor Demo:**
```python
from examples.langgraph_multi_agent_mcts import create_workflow

# Create workflow
workflow = create_workflow()

# Execute with state
initial_state = {
    "query": "What's the best defensive strategy for urban warfare?",
    "use_rag": True,
    "use_mcts": True,
    "iteration": 0,
    "agent_outputs": [],
    "confidence_scores": {},
}

# Run workflow
result = workflow.invoke(initial_state)

# Inspect final state
print(f"Final phase: {result['current_phase']}")
print(f"Iterations: {result['iteration']}")
print(f"Confidence: {result['confidence_scores']}")
print(f"Output: {result['agent_outputs'][-1]}")
```

### Discussion & Q&A (10 minutes)

**Key Questions:**
1. What happens if a node function raises an exception?
2. How do you handle state that grows indefinitely?
3. Can nodes execute in parallel?
4. How do checkpoints work for long-running workflows?

---

## Session 2: E2E Flow Patterns (3 hours)

### Pre-Reading (30 minutes)

- [tests/e2e/test_agent_specific_flows.py](../../tests/e2e/test_agent_specific_flows.py) - E2E test patterns
- [AGENT_TRACING_GUIDE.md](../AGENT_TRACING_GUIDE.md) - Agent interaction patterns

### Lecture: Common E2E Patterns (60 minutes)

#### Pattern 1: Linear Pipeline

**Use Case:** Simple sequential processing
```python
# Query → RAG → HRM → TRM → Response

graph.add_edge("rag", "hrm")
graph.add_edge("hrm", "trm")
graph.add_edge("trm", "response")
```

**Example:**
```python
def create_linear_pipeline():
    """Simple pipeline for basic queries."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("decompose", run_hrm)
    graph.add_node("refine", run_trm)
    graph.add_node("respond", generate_response)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decompose")
    graph.add_edge("decompose", "refine")
    graph.add_edge("refine", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
```

#### Pattern 2: Conditional Branching

**Use Case:** Different paths based on query type
```python
# Query → Classify → [Tactical | Cyber | General] → Response

graph.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "tactical": "tactical_agent",
        "cyber": "cyber_agent",
        "general": "general_agent",
    }
)
```

**Example:**
```python
def route_by_query_type(state: AgentState) -> str:
    """Route to domain-specific agent."""
    query = state["query"].lower()

    if any(kw in query for kw in ["warfare", "military", "tactical"]):
        return "tactical"
    elif any(kw in query for kw in ["cyber", "security", "threat"]):
        return "cyber"
    else:
        return "general"

graph.add_node("tactical_agent", handle_tactical)
graph.add_node("cyber_agent", handle_cyber)
graph.add_node("general_agent", handle_general)

graph.add_conditional_edges(
    "classify",
    route_by_query_type,
    {
        "tactical": "tactical_agent",
        "cyber": "cyber_agent",
        "general": "general_agent",
    }
)

# Converge back
graph.add_edge("tactical_agent", "finalize")
graph.add_edge("cyber_agent", "finalize")
graph.add_edge("general_agent", "finalize")
```

#### Pattern 3: Iterative Refinement Loop

**Use Case:** Multi-round refinement until convergence
```python
# Query → Initial → Refine → Evaluate → [Continue → Refine | Done → Response]

graph.add_conditional_edges(
    "evaluate",
    check_convergence,
    {
        "continue": "refine",  # Loop back
        "done": "respond"
    }
)
```

**Example:**
```python
def create_refinement_loop():
    """Iterative refinement with convergence check."""
    graph = StateGraph(AgentState)

    graph.add_node("generate_initial", generate_initial_solution)
    graph.add_node("refine", refine_solution)
    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("finalize", finalize_solution)

    graph.set_entry_point("generate_initial")
    graph.add_edge("generate_initial", "refine")
    graph.add_edge("refine", "evaluate")

    def check_done(state: AgentState) -> str:
        """Check if refinement should continue."""
        # Max iterations check
        if state["iteration"] >= 5:
            return "done"

        # Quality threshold check
        if state["confidence_scores"].get("quality", 0) >= 0.9:
            return "done"

        # Improvement plateaued
        if len(state["agent_outputs"]) >= 2:
            prev_score = state["agent_outputs"][-2].get("quality", 0)
            curr_score = state["agent_outputs"][-1].get("quality", 0)
            if curr_score - prev_score < 0.05:
                return "done"

        return "continue"

    graph.add_conditional_edges(
        "evaluate",
        check_done,
        {
            "continue": "refine",
            "done": "finalize"
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()
```

#### Pattern 4: Parallel Fan-Out / Fan-In

**Use Case:** Execute multiple agents in parallel, aggregate results
```python
# Query → [HRM | TRM | MCTS] → Aggregate → Response

# Note: LangGraph doesn't support true parallelism in basic mode
# Use Send API for parallel execution (advanced)
```

**Example (Sequential Fan-Out):**
```python
def create_multi_agent_consensus():
    """Run multiple agents and aggregate."""
    graph = StateGraph(AgentState)

    graph.add_node("hrm", run_hrm)
    graph.add_node("trm", run_trm)
    graph.add_node("aggregate", aggregate_agent_results)

    graph.set_entry_point("hrm")
    graph.add_edge("hrm", "trm")
    graph.add_edge("trm", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()

def aggregate_agent_results(state: AgentState) -> AgentState:
    """Aggregate results from multiple agents."""
    outputs = state["agent_outputs"]

    # Weighted voting based on confidence
    total_confidence = sum(
        o.get("confidence", 0) for o in outputs
    )

    weighted_scores = {}
    for output in outputs:
        weight = output.get("confidence", 0) / total_confidence
        # Aggregate scores...

    return {
        "confidence_scores": weighted_scores,
        "current_phase": "completed"
    }
```

### Hands-On Exercise: Implement Custom Flow (90 minutes)

**Exercise 1: Domain-Aware Routing**

**Objective:** Implement a graph that routes queries to domain-specific agents.

**Requirements:**
1. Create a classifier node that detects query domain
2. Route to one of three domain handlers (tactical, cyber, general)
3. Each handler has different processing logic
4. All paths converge to a synthesis node
5. Add proper error handling

**Starter Code:**
```python
from langgraph.graph import StateGraph, END

def create_domain_router():
    """Create domain-aware routing graph."""
    graph = StateGraph(AgentState)

    # TODO: Add nodes
    graph.add_node("classify", classify_domain)
    graph.add_node("tactical", handle_tactical_query)
    graph.add_node("cyber", handle_cyber_query)
    graph.add_node("general", handle_general_query)
    graph.add_node("synthesize", synthesize_final_response)

    # TODO: Set entry point
    graph.set_entry_point("classify")

    # TODO: Add conditional routing
    def route_to_handler(state: AgentState) -> str:
        """Route based on detected domain."""
        # Your implementation here
        pass

    graph.add_conditional_edges(
        "classify",
        route_to_handler,
        {
            "tactical": "tactical",
            "cyber": "cyber",
            "general": "general",
        }
    )

    # TODO: Add convergence edges
    graph.add_edge("tactical", "synthesize")
    graph.add_edge("cyber", "synthesize")
    graph.add_edge("general", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()

# TODO: Implement node functions
def classify_domain(state: AgentState) -> AgentState:
    """Classify query domain."""
    pass

def handle_tactical_query(state: AgentState) -> AgentState:
    """Handle tactical domain queries."""
    pass

# TODO: Write tests
def test_domain_routing_tactical():
    """Test routing to tactical handler."""
    workflow = create_domain_router()

    result = workflow.invoke({
        "query": "What's the best urban warfare tactic?",
        "iteration": 0,
        "agent_outputs": [],
        "confidence_scores": {},
    })

    # Verify tactical handler was used
    assert result["metadata"]["domain"] == "tactical"
    # Add more assertions...
```

**Deliverable:** Complete implementation with passing tests

---

## Session 3: Testing E2E Workflows (2.5 hours)

### Pre-Reading (30 minutes)

- [tests/e2e/test_agent_specific_flows.py](../../tests/e2e/test_agent_specific_flows.py)
- [tests/utils/langsmith_tracing.py](../../tests/utils/langsmith_tracing.py)

### Lecture: E2E Test Patterns (60 minutes)

#### Test Structure

**Canonical E2E Test:**
```python
import pytest
from tests.utils.langsmith_tracing import trace_e2e_test

@trace_e2e_test(
    scenario="tactical_analysis_flow",
    phase="validation",
    tags=["e2e", "tactical"]
)
async def test_tactical_analysis_e2e():
    """Test complete tactical analysis workflow."""
    # 1. Setup
    workflow = create_workflow()

    # 2. Prepare input
    input_state = {
        "query": "Develop defensive strategy for urban environment",
        "use_rag": True,
        "use_mcts": False,
        "iteration": 0,
        "agent_outputs": [],
        "confidence_scores": {},
    }

    # 3. Execute workflow
    result = workflow.invoke(input_state)

    # 4. Assertions
    assert result["current_phase"] == "completed"
    assert result["hrm_results"] is not None
    assert result["trm_results"] is not None
    assert result["confidence_scores"]["overall"] >= 0.7

    # 5. Verify output structure
    final_output = result["agent_outputs"][-1]
    assert "defensive_strategy" in final_output
    assert "tactical_recommendations" in final_output

    # 6. Check performance
    assert result["latency_ms"] < 5000  # 5 second max
```

#### Test Categories

**1. Happy Path Tests:**
```python
@trace_e2e_test(scenario="happy_path_hrm_trm")
def test_hrm_trm_success():
    """Test successful HRM → TRM flow."""
    result = workflow.invoke({
        "query": "Simple tactical query",
        "use_mcts": False,
    })

    assert result["current_phase"] == "completed"
    assert result["hrm_results"]["confidence"] >= 0.7
    assert result["trm_results"]["converged"] is True
```

**2. Error Handling Tests:**
```python
@trace_e2e_test(scenario="error_handling_invalid_input")
def test_invalid_input_handling():
    """Test handling of invalid input."""
    with pytest.raises(ValidationError):
        workflow.invoke({
            "query": "",  # Empty query
        })
```

**3. Edge Case Tests:**
```python
@trace_e2e_test(scenario="edge_case_max_iterations")
def test_max_iterations_reached():
    """Test behavior when max iterations reached."""
    result = workflow.invoke({
        "query": "Complex query requiring many iterations",
        "use_mcts": True,
    })

    # Should still complete gracefully
    assert result["current_phase"] == "completed"
    assert result["iteration"] <= MAX_ITERATIONS
```

**4. Performance Tests:**
```python
@pytest.mark.slow
@trace_e2e_test(scenario="performance_mcts_200")
def test_mcts_performance():
    """Test MCTS with 200 iterations completes in time."""
    import time

    start = time.time()
    result = workflow.invoke({
        "query": "Optimal decision with 5 alternatives",
        "use_mcts": True,
    })
    elapsed = time.time() - start

    assert elapsed < 10.0  # 10 second max
    assert result["mcts_tree"] is not None
```

#### Test Fixtures

**Reusable Fixtures:**
```python
@pytest.fixture
def workflow():
    """Create workflow instance."""
    return create_workflow()

@pytest.fixture
def base_state():
    """Base state for tests."""
    return {
        "query": "",
        "use_rag": False,
        "use_mcts": False,
        "iteration": 0,
        "agent_outputs": [],
        "confidence_scores": {},
    }

@pytest.fixture
def tactical_query():
    """Sample tactical query."""
    return "What's the best defensive posture for urban warfare?"

# Usage
def test_with_fixtures(workflow, base_state, tactical_query):
    """Test using fixtures."""
    state = {**base_state, "query": tactical_query}
    result = workflow.invoke(state)
    assert result["current_phase"] == "completed"
```

### Hands-On Exercise: Write E2E Tests (60 minutes)

**Exercise 2: Comprehensive Test Suite**

**Objective:** Write tests for all paths in your domain router from Exercise 1.

**Tasks:**

1. **Happy path for each domain:**
```python
def test_tactical_domain_flow():
    """Test tactical domain routing and processing."""
    pass

def test_cyber_domain_flow():
    """Test cyber domain routing and processing."""
    pass

def test_general_domain_flow():
    """Test general domain routing and processing."""
    pass
```

2. **Error cases:**
```python
def test_empty_query():
    """Test handling of empty query."""
    pass

def test_ambiguous_domain():
    """Test handling of ambiguous domain classification."""
    pass
```

3. **Performance:**
```python
@pytest.mark.slow
def test_end_to_end_latency():
    """Test overall latency is acceptable."""
    pass
```

4. **State validation:**
```python
def test_state_consistency():
    """Test state remains consistent throughout flow."""
    pass
```

**Deliverable:** Test file with 80%+ coverage

---

## Session 4: Custom State Transitions (2 hours)

### Lecture: Advanced Routing (45 minutes)

#### Dynamic Node Selection

**Example: Adaptive Agent Selection**
```python
def select_agents(state: AgentState) -> str:
    """Dynamically select which agents to run."""
    query_complexity = estimate_complexity(state["query"])

    if query_complexity < 0.3:
        return "simple_path"  # HRM only
    elif query_complexity < 0.7:
        return "medium_path"  # HRM + TRM
    else:
        return "complex_path"  # HRM + TRM + MCTS

graph.add_conditional_edges(
    "analyze_query",
    select_agents,
    {
        "simple_path": "hrm_only",
        "medium_path": "hrm_trm_pipeline",
        "complex_path": "full_stack_with_mcts",
    }
)
```

#### State-Based Routing

**Example: Confidence-Based Iteration**
```python
def check_confidence(state: AgentState) -> str:
    """Route based on confidence scores."""
    confidence = state["confidence_scores"].get("overall", 0)

    if confidence >= 0.9:
        return "high_confidence"
    elif confidence >= 0.7:
        return "medium_confidence"
    else:
        return "low_confidence"

graph.add_conditional_edges(
    "evaluate_confidence",
    check_confidence,
    {
        "high_confidence": "finalize",
        "medium_confidence": "additional_refinement",
        "low_confidence": "request_human_review",
    }
)
```

#### Multi-Criteria Routing

**Example: Complex Decision Logic**
```python
def smart_router(state: AgentState) -> str:
    """Route based on multiple criteria."""
    confidence = state["confidence_scores"].get("overall", 0)
    iterations = state["iteration"]
    has_mcts = state.get("use_mcts", False)

    # Max iterations reached
    if iterations >= MAX_ITERATIONS:
        return "force_complete"

    # High confidence, finish
    if confidence >= 0.9:
        return "complete"

    # Low confidence but MCTS not tried yet
    if confidence < 0.7 and has_mcts and not state.get("mcts_tree"):
        return "try_mcts"

    # Default: refine
    return "refine"

graph.add_conditional_edges(
    "decision_point",
    smart_router,
    {
        "force_complete": "finalize",
        "complete": "finalize",
        "try_mcts": "mcts",
        "refine": "trm",
    }
)
```

### Hands-On Exercise: Implement Adaptive Workflow (60 minutes)

**Exercise 3: Confidence-Based Adaptive Flow**

**Objective:** Create a workflow that adapts based on confidence scores.

**Requirements:**
1. Run HRM and evaluate confidence
2. If confidence > 0.85, skip TRM and go directly to response
3. If confidence 0.6-0.85, run TRM
4. If confidence < 0.6, run TRM + MCTS
5. After each step, re-evaluate and adapt

**Starter Code:**
```python
def create_adaptive_workflow():
    """Create confidence-based adaptive workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("hrm", run_hrm)
    graph.add_node("evaluate_hrm", evaluate_hrm_confidence)
    graph.add_node("trm", run_trm)
    graph.add_node("evaluate_trm", evaluate_trm_confidence)
    graph.add_node("mcts", run_mcts)
    graph.add_node("finalize", finalize_response)

    # Entry point
    graph.set_entry_point("hrm")

    # HRM → Evaluate
    graph.add_edge("hrm", "evaluate_hrm")

    # TODO: Add routing logic
    def route_after_hrm(state: AgentState) -> str:
        """Route based on HRM confidence."""
        # Your implementation here
        pass

    graph.add_conditional_edges(
        "evaluate_hrm",
        route_after_hrm,
        {
            "skip_to_response": "finalize",
            "run_trm": "trm",
        }
    )

    # TODO: Add TRM evaluation routing
    # TODO: Add MCTS routing

    return graph.compile()
```

**Deliverable:** Adaptive workflow with tests demonstrating all paths

### Discussion: Production Considerations (15 minutes)

**Key Topics:**
1. **State Size Management:** How to prevent state from growing too large
2. **Checkpointing:** Persisting state for long-running workflows
3. **Error Recovery:** Implementing retry logic and fallbacks
4. **Monitoring:** Adding observability to custom workflows
5. **Versioning:** Managing workflow schema changes

---

## Module 3 Assessment

### Practical Assessment

**Task:** Implement a complete E2E workflow with custom routing

**Scenario:** Build a "Smart Query Router" that:
1. Classifies query complexity (simple, moderate, complex)
2. Routes to appropriate agent pipeline:
   - Simple: HRM only
   - Moderate: HRM → TRM
   - Complex: HRM → TRM → MCTS
3. Implements confidence-based fallback:
   - If confidence < 0.6 after initial path, escalate to more complex path
4. Handles errors gracefully with fallback responses
5. Tracks performance metrics at each step

**Requirements:**
- Complete graph implementation (30 points)
- Comprehensive test suite (25 points)
- LangSmith tracing integration (15 points)
- Error handling and fallbacks (15 points)
- Documentation and code quality (15 points)

**Total:** 100 points (passing: 70+)

**Submission:** Git branch with implementation + test results

---

## Assessment Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Graph Implementation** | 30% | Correct state schema, nodes, and routing logic |
| **Test Coverage** | 25% | All paths tested, edge cases covered |
| **Tracing** | 15% | Proper LangSmith instrumentation |
| **Error Handling** | 15% | Graceful failures, fallbacks implemented |
| **Code Quality** | 15% | Type hints, documentation, formatting |

**Minimum Passing:** 70% overall

---

## Additional Resources

### Reading
- [langgraph_mcts_architecture.md](../langgraph_mcts_architecture.md) - Complete architecture
- [LANGSMITH_E2E.md](../LANGSMITH_E2E.md) - E2E tracing patterns
- LangGraph documentation: https://langchain-ai.github.io/langgraph/

### Code Examples
- [examples/langgraph_multi_agent_mcts.py](../../examples/langgraph_multi_agent_mcts.py) - Main workflow
- [tests/e2e/test_agent_specific_flows.py](../../tests/e2e/test_agent_specific_flows.py) - E2E tests

### Office Hours
- When: [Schedule TBD]
- Topics: Graph design, testing patterns, debugging state issues

---

## Next Module

Continue to [MODULE_4_TRACING.md](MODULE_4_TRACING.md) - LangSmith Tracing Utilities & Patterns

**Prerequisites for Module 4:**
- Completed Module 3 practical assessment
- Familiarity with LangSmith UI
- Understanding of distributed tracing concepts
