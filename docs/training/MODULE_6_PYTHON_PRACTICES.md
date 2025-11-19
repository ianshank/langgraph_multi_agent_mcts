# Module 6: 2025 Python Coding & Testing Practices

**Duration:** 8 hours (1.5 days)
**Format:** Workshop + Code Quality Lab
**Difficulty:** Intermediate
**Prerequisites:** Solid Python foundation, completed Modules 1-5

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Apply modern Python standards** including type hints, dataclasses, and pattern matching
2. **Write async/await code** for concurrent operations
3. **Implement comprehensive tests** with pytest best practices
4. **Use code quality tools** (Ruff, MyPy, Black) effectively
5. **Refactor legacy code** to meet 2025 standards

---

## Session 1: Modern Python Standards (2 hours)

### Pre-Reading (30 minutes)

Before the session, review:
- [pyproject.toml](../../pyproject.toml) - Project configuration and tool settings
- Python 3.11+ release notes: https://docs.python.org/3/whatsnew/3.11.html
- Python 3.12+ release notes: https://docs.python.org/3/whatsnew/3.12.html

### Lecture: Type Hints (45 minutes)

#### Why Type Hints?

**Benefits:**
- **Early error detection:** Catch type errors before runtime
- **Better IDE support:** Improved autocomplete and refactoring
- **Self-documentation:** Code is more readable
- **Safer refactoring:** Type checker validates changes

#### Basic Type Hints

**Function signatures:**
```python
def process_query(query: str, max_tokens: int = 1000) -> dict[str, Any]:
    """Process query with type hints."""
    return {
        "response": "...",
        "tokens": max_tokens,
    }
```

**Variables:**
```python
# Simple types
name: str = "John"
age: int = 30
is_active: bool = True

# Collections
tasks: list[str] = ["task1", "task2"]
scores: dict[str, float] = {"hrm": 0.85, "trm": 0.92}
coordinates: tuple[float, float] = (1.0, 2.0)

# Optional types
result: str | None = None  # Python 3.10+
# or
from typing import Optional
result: Optional[str] = None
```

#### Advanced Type Hints

**Generic types:**
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Cache(Generic[T]):
    """Generic cache for any type."""
    def __init__(self) -> None:
        self._data: dict[str, T] = {}

    def get(self, key: str) -> T | None:
        return self._data.get(key)

    def set(self, key: str, value: T) -> None:
        self._data[key] = value

# Usage
string_cache: Cache[str] = Cache()
int_cache: Cache[int] = Cache()
```

**Protocol types (structural subtyping):**
```python
from typing import Protocol

class Evaluator(Protocol):
    """Protocol for evaluation functions."""
    def evaluate(self, response: str) -> float:
        """Evaluate response and return score."""
        ...

# Any class with evaluate() method satisfies this protocol
class HRMEvaluator:
    def evaluate(self, response: str) -> float:
        return 0.85

class TRMEvaluator:
    def evaluate(self, response: str) -> float:
        return 0.92

def run_evaluation(evaluator: Evaluator, response: str) -> float:
    """Works with any evaluator."""
    return evaluator.evaluate(response)
```

**TypedDict for structured dictionaries:**
```python
from typing import TypedDict

class AgentState(TypedDict):
    """Typed dictionary for agent state."""
    query: str
    confidence: float
    results: list[str]
    metadata: dict[str, Any]

# Usage with type checking
state: AgentState = {
    "query": "test",
    "confidence": 0.85,
    "results": ["result1"],
    "metadata": {},
}

# Type checker will error on missing or wrong-typed keys
# state: AgentState = {"query": 123}  # Error: wrong type
```

#### Type Hints for LangGraph

**State schema:**
```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class WorkflowState(TypedDict):
    """LangGraph workflow state."""
    # Simple fields
    query: str
    iteration: int

    # Messages with reducer (append)
    messages: Annotated[list, add_messages]

    # Optional fields
    hrm_results: dict[str, Any] | None
    trm_results: dict[str, Any] | None

# Node function signature
def hrm_node(state: WorkflowState) -> WorkflowState:
    """HRM node with type hints."""
    return {
        "hrm_results": {"confidence": 0.85},
        "iteration": state["iteration"] + 1,
    }
```

### Lecture: Dataclasses and Pydantic (30 minutes)

#### Dataclasses (Python 3.10+)

**Basic dataclass:**
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    model: str
    temperature: float = 0.7
    use_mcts: bool = False
    mcts_iterations: Optional[int] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "model": self.model,
            "temperature": self.temperature,
            "use_mcts": self.use_mcts,
            "mcts_iterations": self.mcts_iterations,
        }

# Usage
config = ExperimentConfig(
    name="exp_mcts_200",
    model="gpt-4o",
    use_mcts=True,
    mcts_iterations=200,
)
print(config.name)  # Type-safe access
```

**Dataclass features:**
```python
@dataclass(frozen=True)  # Immutable
class ImmutableConfig:
    name: str
    value: int

@dataclass(order=True)  # Comparable
class SortableItem:
    priority: int
    name: str

# Auto-generated methods: __init__, __repr__, __eq__, __hash__
```

#### Pydantic Models

**Pydantic for validation:**
```python
from pydantic import BaseModel, Field, validator

class AgentConfig(BaseModel):
    """Agent configuration with validation."""
    name: str = Field(..., min_length=1, max_length=100)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_iterations: int = Field(5, gt=0, le=100)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

    @validator('name')
    def validate_name(cls, v):
        """Custom validation for name."""
        if not v.isalnum():
            raise ValueError('Name must be alphanumeric')
        return v

# Usage - automatic validation
try:
    config = AgentConfig(
        name="hrm_agent",
        confidence_threshold=0.85,
        max_iterations=10,
    )
except ValidationError as e:
    print(e.json())
```

### Hands-On Exercise: Add Type Hints (15 minutes)

**Exercise 1: Type-Hint Legacy Code**

**Objective:** Add comprehensive type hints to existing code.

**Task:**
```python
# Before (no type hints)
def process_agent_results(results, config):
    output = []
    for result in results:
        if result["confidence"] >= config["threshold"]:
            output.append(result)
    return output

# TODO: Add type hints
# After (with type hints)
def process_agent_results(
    results: list[dict[str, Any]],
    config: dict[str, float]
) -> list[dict[str, Any]]:
    """Filter results by confidence threshold."""
    output: list[dict[str, Any]] = []
    for result in results:
        if result["confidence"] >= config["threshold"]:
            output.append(result)
    return output

# Better: Use Pydantic or dataclass for structured data
@dataclass
class AgentResult:
    confidence: float
    output: str
    metadata: dict[str, Any]

@dataclass
class FilterConfig:
    threshold: float

def process_agent_results_typed(
    results: list[AgentResult],
    config: FilterConfig
) -> list[AgentResult]:
    """Filter results by confidence threshold (type-safe)."""
    return [r for r in results if r.confidence >= config.threshold]
```

**Deliverable:** Type-hinted version of provided code

---

## Session 2: Testing Standards (2.5 hours)

### Pre-Reading (30 minutes)

- pytest documentation: https://docs.pytest.org/
- [tests/unit/](../../tests/unit/) - Unit test examples
- [pyproject.toml](../../pyproject.toml) - pytest configuration

### Lecture: Pytest Best Practices (60 minutes)

#### Test Structure

**Arrange-Act-Assert pattern:**
```python
def test_hrm_decomposition():
    """Test HRM task decomposition."""
    # Arrange - Set up test data
    query = "Develop urban defense strategy"
    agent = HRMAgent(llm=MockLLM())

    # Act - Execute the operation
    result = agent.decompose(query)

    # Assert - Verify expectations
    assert result["confidence"] >= 0.7
    assert len(result["tasks"]) >= 3
    assert "defense" in str(result["tasks"]).lower()
```

#### Fixtures

**Basic fixtures:**
```python
import pytest

@pytest.fixture
def hrm_agent():
    """Create HRM agent for testing."""
    return HRMAgent(llm=MockLLM())

@pytest.fixture
def tactical_query():
    """Sample tactical query."""
    return "What's the best defensive strategy?"

# Usage
def test_with_fixtures(hrm_agent, tactical_query):
    """Test using fixtures."""
    result = hrm_agent.decompose(tactical_query)
    assert result["confidence"] > 0.0
```

**Fixture scopes:**
```python
@pytest.fixture(scope="function")  # Default: new for each test
def fresh_agent():
    return HRMAgent()

@pytest.fixture(scope="module")  # Shared across module
def expensive_resource():
    resource = setup_expensive_resource()
    yield resource
    resource.cleanup()

@pytest.fixture(scope="session")  # Shared across entire test session
def database_connection():
    conn = create_db_connection()
    yield conn
    conn.close()
```

#### Parametrization

**Test multiple inputs:**
```python
@pytest.mark.parametrize(
    "query,expected_domain",
    [
        ("Urban warfare tactics", "tactical"),
        ("SQL injection prevention", "cybersecurity"),
        ("General question", "general"),
    ]
)
def test_domain_detection(query, expected_domain):
    """Test domain detection with multiple inputs."""
    domain = detect_domain(query)
    assert domain == expected_domain
```

**Multiple parameters:**
```python
@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini"])
@pytest.mark.parametrize("temperature", [0.5, 0.7, 0.9])
def test_model_temperature_combinations(model, temperature):
    """Test all model-temperature combinations."""
    agent = create_agent(model=model, temperature=temperature)
    result = agent.process("test query")
    assert result is not None
```

#### Mocking

**Using pytest-mock:**
```python
def test_with_mock(mocker):
    """Test with mocked LLM."""
    # Mock LLM call
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = "Mocked response"

    agent = HRMAgent(llm=mock_llm)
    result = agent.decompose("test query")

    # Verify mock was called
    mock_llm.invoke.assert_called_once()
    assert result is not None
```

**Patching:**
```python
def test_with_patch(mocker):
    """Test with patched function."""
    # Patch expensive function
    mock_expensive = mocker.patch(
        'src.module.expensive_function',
        return_value="mocked result"
    )

    result = call_function_that_uses_expensive()

    mock_expensive.assert_called_once()
    assert result == "mocked result"
```

#### Async Testing

**Testing async functions:**
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_process_query("test")
    assert result is not None

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent async operations."""
    tasks = [
        async_operation_1(),
        async_operation_2(),
        async_operation_3(),
    ]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
```

### Lecture: Test Organization (30 minutes)

#### Test File Structure

**Recommended structure:**
```
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_hrm_agent.py
│   ├── test_trm_agent.py
│   └── test_mcts_engine.py
├── integration/             # Integration tests (moderate speed)
│   ├── test_workflow_integration.py
│   └── test_database_integration.py
├── e2e/                     # End-to-end tests (slow)
│   ├── test_agent_specific_flows.py
│   └── test_full_workflow.py
├── fixtures/                # Shared fixtures
│   ├── __init__.py
│   └── common_fixtures.py
└── conftest.py             # Global pytest configuration
```

#### Test Markers

**Define custom markers:**
```python
# pyproject.toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
    "e2e: end-to-end tests",
]

# Usage in tests
@pytest.mark.slow
def test_expensive_operation():
    """Slow test."""
    pass

@pytest.mark.integration
def test_database_integration():
    """Integration test."""
    pass

# Run specific markers
# pytest -m "not slow"  # Skip slow tests
# pytest -m integration  # Run only integration tests
```

### Hands-On Exercise: Write Comprehensive Tests (30 minutes)

**Exercise 2: Test Suite for Agent**

**Objective:** Write comprehensive test suite for an agent.

**Requirements:**
1. Unit tests for core methods
2. Parametrized tests for multiple scenarios
3. Mocked external dependencies
4. Async test for concurrent operations
5. Integration test for full workflow

**Template:**
```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def agent():
    """Create agent for testing."""
    return MyAgent(llm=Mock())

@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "Simple query",
        "Complex tactical query with multiple parts",
        "Cybersecurity incident response query",
    ]

# Unit test
def test_core_method(agent):
    """Test core method."""
    result = agent.core_method("input")
    assert result is not None

# Parametrized test
@pytest.mark.parametrize(
    "query,expected_length",
    [
        ("Short", 1),
        ("Medium length query", 3),
        ("Very detailed and complex query with many parts", 5),
    ]
)
def test_query_processing(agent, query, expected_length):
    """Test processing different query lengths."""
    # TODO: Implement test
    pass

# Mocked test
def test_with_mocked_llm(mocker):
    """Test with mocked LLM."""
    # TODO: Implement test
    pass

# Async test
@pytest.mark.asyncio
async def test_async_operation(agent):
    """Test async operation."""
    # TODO: Implement test
    pass

# Integration test
@pytest.mark.integration
def test_integration(agent, sample_queries):
    """Test integration with real components."""
    # TODO: Implement test
    pass
```

**Deliverable:** Complete test suite with 80%+ coverage

---

## Session 3: Code Quality Tools (2 hours)

### Pre-Reading (30 minutes)

- Ruff documentation: https://docs.astral.sh/ruff/
- MyPy documentation: https://mypy.readthedocs.io/
- [pyproject.toml](../../pyproject.toml) - Tool configurations

### Lecture: Linting with Ruff (45 minutes)

#### What is Ruff?

**Ruff** is an extremely fast Python linter and formatter that combines functionality of:
- **Flake8:** Style and error checking
- **isort:** Import sorting
- **Black:** Code formatting
- **pyupgrade:** Syntax modernization

#### Ruff Configuration

**pyproject.toml configuration:**
```toml
[tool.ruff]
target-version = "py311"
line-length = 120
indent-width = 4

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
]
ignore = [
    "E501",   # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

#### Running Ruff

**Command line usage:**
```bash
# Check code
ruff check .

# Check with fix
ruff check --fix .

# Format code
ruff format .

# Check formatting without applying
ruff format --check .
```

**Common issues and fixes:**

```python
# Before: Unused import
import os
import sys

def main():
    print("Hello")

# After: Ruff removes unused imports
def main():
    print("Hello")

# Before: Unsorted imports
from z_module import z
from a_module import a
import sys
import os

# After: Ruff sorts imports
import os
import sys

from a_module import a
from z_module import z

# Before: Complex comprehension
result = []
for item in items:
    if item.value > 0:
        result.append(item.name)

# After: Ruff suggests list comprehension
result = [item.name for item in items if item.value > 0]
```

### Lecture: Type Checking with MyPy (45 minutes)

#### What is MyPy?

**MyPy** is a static type checker that validates type hints in Python code.

#### MyPy Configuration

**pyproject.toml configuration:**
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
check_untyped_defs = true
disallow_untyped_defs = false
no_implicit_optional = true
strict_equality = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
```

#### Running MyPy

**Command line usage:**
```bash
# Check entire codebase
mypy src/

# Check specific file
mypy src/agents/hrm_agent.py

# Check with strict mode
mypy --strict src/
```

#### Common Type Errors

**Error: Incompatible types:**
```python
# Error
def process(value: int) -> str:
    return value  # Error: Incompatible return type "int" (expected "str")

# Fix
def process(value: int) -> str:
    return str(value)
```

**Error: Missing type annotation:**
```python
# Error
def calculate_score(results):  # Error: Missing type annotation
    return sum(results) / len(results)

# Fix
def calculate_score(results: list[float]) -> float:
    return sum(results) / len(results)
```

**Error: Optional handling:**
```python
# Error
def get_value(data: dict[str, int]) -> int:
    value = data.get("key")  # Type is int | None
    return value + 1  # Error: Unsupported operand types for + ("None" and "int")

# Fix
def get_value(data: dict[str, int]) -> int:
    value = data.get("key")
    if value is None:
        return 0
    return value + 1

# Or with default
def get_value(data: dict[str, int]) -> int:
    value = data.get("key", 0)  # Type is int
    return value + 1
```

### Hands-On Exercise: Apply Quality Tools (30 minutes)

**Exercise 3: Fix Code Quality Issues**

**Objective:** Use Ruff and MyPy to improve code quality.

**Task:**
```python
# messy_code.py - Fix all issues

import sys
import time
import os
from typing import Any
from datetime import datetime

def process_data(data):
    result = []
    for item in data:
        if item["value"] > 0:
            result.append(item)
    return result

def calculate(x, y, operation):
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    else:
        return None

class Agent:
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def process(self, query):
        # Process query
        start = time.time()
        result = self._do_processing(query)
        elapsed = time.time() - start
        return result

    def _do_processing(self, query):
        # Implementation
        return {"result": "processed"}

# TODO: Run ruff check messy_code.py
# TODO: Run ruff format messy_code.py
# TODO: Add type hints
# TODO: Run mypy messy_code.py
# TODO: Fix all issues
```

**Deliverable:** Clean code that passes Ruff and MyPy checks

---

## Session 4: Refactor Lab (1.5 hours)

### Lecture: Refactoring Strategies (30 minutes)

#### Refactoring Checklist

**1. Add type hints:**
- Function signatures
- Class attributes
- Return types

**2. Extract functions:**
- Break long functions into smaller ones
- Single responsibility principle

**3. Use dataclasses:**
- Replace dictionary-based data with dataclasses
- Better type safety and IDE support

**4. Simplify conditionals:**
- Replace nested ifs with guard clauses
- Use pattern matching (Python 3.10+)

**5. Modern Python features:**
- Use walrus operator (`:=`) where appropriate
- Pattern matching for complex conditions
- Structural pattern matching for data validation

#### Example Refactoring

**Before:**
```python
def process_agent_output(output, config):
    if output is not None:
        if "confidence" in output:
            if output["confidence"] >= config["threshold"]:
                result = {"status": "success", "data": output}
                return result
            else:
                result = {"status": "low_confidence", "data": output}
                return result
        else:
            result = {"status": "error", "message": "Missing confidence"}
            return result
    else:
        result = {"status": "error", "message": "No output"}
        return result
```

**After:**
```python
from dataclasses import dataclass

@dataclass
class ProcessResult:
    status: str
    data: dict[str, Any] | None = None
    message: str | None = None

def process_agent_output(
    output: dict[str, Any] | None,
    config: dict[str, float]
) -> ProcessResult:
    """Process agent output with validation."""
    # Guard clauses
    if output is None:
        return ProcessResult(status="error", message="No output")

    if "confidence" not in output:
        return ProcessResult(status="error", message="Missing confidence")

    # Main logic
    threshold = config["threshold"]
    if output["confidence"] >= threshold:
        return ProcessResult(status="success", data=output)

    return ProcessResult(status="low_confidence", data=output)
```

### Hands-On Exercise: Refactor Module (60 minutes)

**Exercise 4: Complete Refactoring**

**Objective:** Refactor a legacy module to meet 2025 standards.

**Requirements:**
1. Add comprehensive type hints
2. Replace dictionaries with dataclasses
3. Simplify complex functions
4. Add docstrings
5. Pass Ruff and MyPy checks
6. Maintain or improve test coverage

**Deliverable:** Refactored module with before/after comparison

---

## Module 6 Assessment

### Practical Assessment

**Task:** Refactor a legacy component to 2025 standards

**Requirements:**
1. Add type hints to all functions and methods (25 points)
2. Use dataclasses or Pydantic models (20 points)
3. Write comprehensive test suite (25 points)
4. Pass Ruff linting (10 points)
5. Pass MyPy type checking (10 points)
6. Documentation and docstrings (10 points)

**Total:** 100 points (passing: 70+)

**Submission:** Git branch with refactored code + test results

---

## Assessment Rubric

| Criteria | Weight | Description |
|----------|--------|-------------|
| **Type Hints** | 25% | Comprehensive and correct type annotations |
| **Structured Data** | 20% | Proper use of dataclasses/Pydantic |
| **Tests** | 25% | Comprehensive test coverage (80%+) |
| **Linting** | 10% | Passes Ruff checks |
| **Type Checking** | 10% | Passes MyPy checks |
| **Documentation** | 10% | Clear docstrings and comments |

**Minimum Passing:** 70% overall

---

## Additional Resources

### Reading
- [pyproject.toml](../../pyproject.toml) - Tool configurations
- Python type hints: https://docs.python.org/3/library/typing.html
- pytest documentation: https://docs.pytest.org/

### Tools
- Ruff: https://docs.astral.sh/ruff/
- MyPy: https://mypy.readthedocs.io/
- pytest: https://docs.pytest.org/

### Office Hours
- When: [Schedule TBD]
- Topics: Type hints, testing strategies, refactoring patterns

---

## Next Module

Continue to [MODULE_7_CICD.md](MODULE_7_CICD.md) - CI/CD & Observability Integration

**Prerequisites for Module 7:**
- Completed Module 6 practical assessment
- Understanding of CI/CD concepts
- Basic Docker knowledge
