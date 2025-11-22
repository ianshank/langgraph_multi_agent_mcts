# RAM Agent Architecture Modification Plan

## Overview
Add a Risk Assessment Module (RAM) agent before TRM to evaluate risks.

## Proposed Changes

### 1. AgentState Schema Modifications
```python
class AgentState(TypedDict):
    query: str
    decomposed_tasks: List[str]
    ram_assessment: Optional[Dict[str, Any]]  # NEW FIELD
    refined_solution: str
    mcts_result: Optional[dict]
    current_phase: str
    metadata: dict
```

### 2. RAM Node Implementation
Location: `examples/langgraph_multi_agent_mcts.py`

```python
@traceable(name="ram_agent", run_type="chain", tags=["ram", "risk"])
def run_ram_agent(state: AgentState) -> AgentState:
    """Run Risk Assessment Module."""
    risks = assess_risks(state["decomposed_tasks"])
    return {
        **state,
        "ram_assessment": risks,
        "current_phase": "trm",
        "metadata": {**state["metadata"], "ram_confidence": risks["confidence"]}
    }
```

### 3. Conditional Routing Logic
```python
def should_run_ram(state: AgentState) -> str:
    """Decide if RAM should run based on query type."""
    if requires_risk_assessment(state["query"]):
        return "run_ram"
    return "skip_ram"
```

### 4. Tracing Integration
- Use `@traceable` decorator for RAM node
- Add metadata: risk_level, confidence, assessment_type
- Tag with: ["ram", "risk", "assessment"]
- Context manager for RAM subprocess tracing

## Implementation Steps
1. Define RAM agent class in `src/agents/ram_agent.py`
2. Add RAM node to LangGraph workflow
3. Implement conditional routing
4. Add comprehensive test coverage
5. Integrate LangSmith tracing
