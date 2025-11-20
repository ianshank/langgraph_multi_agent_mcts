"""
Test data builders for creating test fixtures.

This module provides builder classes for creating test data in a
fluent, readable way. Builders enable:
- Readable test setup
- Reusable test data patterns
- Easy test maintenance
- Reduce test boilerplate

Best Practices 2025:
- Use builder pattern for complex objects
- Provide sensible defaults
- Support method chaining
- Enable partial object creation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class AgentContextBuilder:
    """
    Builder for creating AgentContext test fixtures.

    Example:
        >>> context = (
        ...     AgentContextBuilder()
        ...     .with_query("Test query")
        ...     .with_session_id("test-123")
        ...     .with_rag_context("Some context")
        ...     .build()
        ... )
    """

    _query: str = "Default test query"
    _session_id: str = "test-session-123"
    _rag_context: str | None = None
    _metadata: dict = field(default_factory=dict)
    _conversation_history: list[dict] = field(default_factory=list)
    _max_iterations: int = 5
    _temperature: float = 0.7
    _additional_context: dict = field(default_factory=dict)

    def with_query(self, query: str) -> AgentContextBuilder:
        """Set the query."""
        self._query = query
        return self

    def with_session_id(self, session_id: str) -> AgentContextBuilder:
        """Set the session ID."""
        self._session_id = session_id
        return self

    def with_rag_context(self, context: str) -> AgentContextBuilder:
        """Set RAG context."""
        self._rag_context = context
        return self

    def with_metadata(self, metadata: dict) -> AgentContextBuilder:
        """Set metadata."""
        self._metadata = metadata
        return self

    def with_conversation_history(self, history: list[dict]) -> AgentContextBuilder:
        """Set conversation history."""
        self._conversation_history = history
        return self

    def with_max_iterations(self, max_iterations: int) -> AgentContextBuilder:
        """Set max iterations."""
        self._max_iterations = max_iterations
        return self

    def with_temperature(self, temperature: float) -> AgentContextBuilder:
        """Set temperature."""
        self._temperature = temperature
        return self

    def build(self) -> dict:
        """Build the AgentContext."""
        from src.framework.agents.base import AgentContext

        return AgentContext(
            query=self._query,
            session_id=self._session_id,
            rag_context=self._rag_context,
            metadata=self._metadata,
            conversation_history=self._conversation_history,
            max_iterations=self._max_iterations,
            temperature=self._temperature,
            additional_context=self._additional_context,
        )


@dataclass
class AgentResultBuilder:
    """
    Builder for creating AgentResult test fixtures.

    Example:
        >>> result = (
        ...     AgentResultBuilder()
        ...     .with_response("Test response")
        ...     .with_confidence(0.95)
        ...     .with_agent_name("HRM")
        ...     .with_success(True)
        ...     .build()
        ... )
    """

    _response: str = "Default test response"
    _confidence: float = 0.8
    _metadata: dict = field(default_factory=dict)
    _agent_name: str = "TestAgent"
    _processing_time_ms: float = 100.0
    _token_usage: dict = field(default_factory=lambda: {"prompt_tokens": 10, "completion_tokens": 20})
    _intermediate_steps: list[dict] = field(default_factory=list)
    _error: str | None = None
    _success: bool = True

    def with_response(self, response: str) -> AgentResultBuilder:
        """Set the response."""
        self._response = response
        return self

    def with_confidence(self, confidence: float) -> AgentResultBuilder:
        """Set confidence score."""
        self._confidence = confidence
        return self

    def with_metadata(self, metadata: dict) -> AgentResultBuilder:
        """Set metadata."""
        self._metadata = metadata
        return self

    def with_agent_name(self, name: str) -> AgentResultBuilder:
        """Set agent name."""
        self._agent_name = name
        return self

    def with_processing_time(self, time_ms: float) -> AgentResultBuilder:
        """Set processing time."""
        self._processing_time_ms = time_ms
        return self

    def with_token_usage(self, usage: dict) -> AgentResultBuilder:
        """Set token usage."""
        self._token_usage = usage
        return self

    def with_intermediate_steps(self, steps: list[dict]) -> AgentResultBuilder:
        """Set intermediate steps."""
        self._intermediate_steps = steps
        return self

    def with_error(self, error: str) -> AgentResultBuilder:
        """Set error message."""
        self._error = error
        self._success = False
        return self

    def with_success(self, success: bool) -> AgentResultBuilder:
        """Set success status."""
        self._success = success
        return self

    def build(self) -> dict:
        """Build the AgentResult."""
        from src.framework.agents.base import AgentResult

        return AgentResult(
            response=self._response,
            confidence=self._confidence,
            metadata=self._metadata,
            agent_name=self._agent_name,
            processing_time_ms=self._processing_time_ms,
            token_usage=self._token_usage,
            intermediate_steps=self._intermediate_steps,
            error=self._error,
            success=self._success,
        )


@dataclass
class MCTSStateBuilder:
    """
    Builder for creating MCTSState test fixtures.

    Example:
        >>> state = (
        ...     MCTSStateBuilder()
        ...     .with_state_id("root")
        ...     .with_feature("position", [0, 0])
        ...     .with_feature("score", 100)
        ...     .build()
        ... )
    """

    _state_id: str = "test-state-1"
    _features: dict[str, Any] = field(default_factory=dict)

    def with_state_id(self, state_id: str) -> MCTSStateBuilder:
        """Set state ID."""
        self._state_id = state_id
        return self

    def with_feature(self, key: str, value: Any) -> MCTSStateBuilder:
        """Add a feature."""
        self._features[key] = value
        return self

    def with_features(self, features: dict[str, Any]) -> MCTSStateBuilder:
        """Set all features."""
        self._features = features
        return self

    def build(self) -> dict:
        """Build the MCTSState."""
        from src.framework.mcts.core import MCTSState

        return MCTSState(state_id=self._state_id, features=self._features)


@dataclass
class LLMResponseBuilder:
    """
    Builder for creating LLMResponse test fixtures.

    Example:
        >>> response = (
        ...     LLMResponseBuilder()
        ...     .with_content("Test response content")
        ...     .with_model("gpt-4")
        ...     .with_tokens(100, 50)
        ...     .build()
        ... )
    """

    _content: str = "Test LLM response"
    _model: str = "gpt-4-turbo-preview"
    _prompt_tokens: int = 50
    _completion_tokens: int = 100
    _total_tokens: int | None = None
    _finish_reason: str = "stop"
    _metadata: dict = field(default_factory=dict)

    def with_content(self, content: str) -> LLMResponseBuilder:
        """Set response content."""
        self._content = content
        return self

    def with_model(self, model: str) -> LLMResponseBuilder:
        """Set model name."""
        self._model = model
        return self

    def with_tokens(self, prompt: int, completion: int) -> LLMResponseBuilder:
        """Set token counts."""
        self._prompt_tokens = prompt
        self._completion_tokens = completion
        self._total_tokens = prompt + completion
        return self

    def with_finish_reason(self, reason: str) -> LLMResponseBuilder:
        """Set finish reason."""
        self._finish_reason = reason
        return self

    def with_metadata(self, metadata: dict) -> LLMResponseBuilder:
        """Set metadata."""
        self._metadata = metadata
        return self

    def build(self) -> dict:
        """Build the LLMResponse."""
        from src.adapters.llm.base import LLMResponse

        total = self._total_tokens or (self._prompt_tokens + self._completion_tokens)

        return LLMResponse(
            content=self._content,
            model=self._model,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            total_tokens=total,
            finish_reason=self._finish_reason,
            metadata=self._metadata,
        )


class TacticalScenarioBuilder:
    """
    Builder for creating tactical scenario test data.

    Example:
        >>> scenario = (
        ...     TacticalScenarioBuilder()
        ...     .with_type("defensive")
        ...     .with_complexity("high")
        ...     .with_units(100)
        ...     .with_time_constraint(timedelta(hours=2))
        ...     .build()
        ... )
    """

    def __init__(self):
        self._type = "defensive"
        self._complexity = "medium"
        self._units = 50
        self._terrain = "urban"
        self._weather = "clear"
        self._time_constraint = timedelta(hours=4)
        self._objectives = []
        self._constraints = []

    def with_type(self, scenario_type: str) -> TacticalScenarioBuilder:
        """Set scenario type (defensive, offensive, reconnaissance)."""
        self._type = scenario_type
        return self

    def with_complexity(self, complexity: str) -> TacticalScenarioBuilder:
        """Set complexity level (low, medium, high)."""
        self._complexity = complexity
        return self

    def with_units(self, count: int) -> TacticalScenarioBuilder:
        """Set number of units."""
        self._units = count
        return self

    def with_terrain(self, terrain: str) -> TacticalScenarioBuilder:
        """Set terrain type."""
        self._terrain = terrain
        return self

    def with_weather(self, weather: str) -> TacticalScenarioBuilder:
        """Set weather conditions."""
        self._weather = weather
        return self

    def with_time_constraint(self, duration: timedelta) -> TacticalScenarioBuilder:
        """Set time constraint."""
        self._time_constraint = duration
        return self

    def add_objective(self, objective: str) -> TacticalScenarioBuilder:
        """Add an objective."""
        self._objectives.append(objective)
        return self

    def add_constraint(self, constraint: str) -> TacticalScenarioBuilder:
        """Add a constraint."""
        self._constraints.append(constraint)
        return self

    def build(self) -> dict:
        """Build the scenario dictionary."""
        return {
            "type": self._type,
            "complexity": self._complexity,
            "units": self._units,
            "terrain": self._terrain,
            "weather": self._weather,
            "time_constraint_hours": self._time_constraint.total_seconds() / 3600,
            "objectives": self._objectives or [f"Default {self._type} objective"],
            "constraints": self._constraints,
            "timestamp": datetime.utcnow().isoformat(),
        }


# Convenience functions for common test data patterns
def minimal_agent_context() -> dict:
    """Create minimal agent context for simple tests."""
    return AgentContextBuilder().build()


def successful_agent_result(agent_name: str = "TestAgent") -> dict:
    """Create a successful agent result."""
    return AgentResultBuilder().with_agent_name(agent_name).with_success(True).build()


def failed_agent_result(error: str, agent_name: str = "TestAgent") -> dict:
    """Create a failed agent result."""
    return AgentResultBuilder().with_agent_name(agent_name).with_error(error).build()


def simple_mcts_state(state_id: str = "root") -> dict:
    """Create a simple MCTS state."""
    return MCTSStateBuilder().with_state_id(state_id).build()


def basic_llm_response(content: str = "Test response") -> dict:
    """Create a basic LLM response."""
    return LLMResponseBuilder().with_content(content).build()


def defensive_scenario() -> dict:
    """Create a standard defensive scenario."""
    return (
        TacticalScenarioBuilder()
        .with_type("defensive")
        .with_complexity("medium")
        .add_objective("Secure perimeter")
        .add_objective("Establish observation posts")
        .build()
    )


def offensive_scenario() -> dict:
    """Create a standard offensive scenario."""
    return (
        TacticalScenarioBuilder()
        .with_type("offensive")
        .with_complexity("high")
        .add_objective("Breach enemy lines")
        .add_constraint("Minimize casualties")
        .build()
    )
