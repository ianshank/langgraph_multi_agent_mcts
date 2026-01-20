# LangGraph Multi-Agent MCTS - Claude Code Implementation Template

> **Design Principle:** This template treats prompting as constraint programming, not instruction writing. Define the feasible region, objective function, and search parameters—then let the agent solve.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: An enterprise-grade LangGraph + Multi-Agent + Monte Carlo Tree Search framework
that combines AlphaGo-style strategic exploration with modern LLM orchestration for complex
decision-making in domains like M&A due diligence, clinical trial optimization, and regulatory compliance.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All unit tests pass: `pytest tests/unit -v --tb=short`
- [ ] All integration tests pass: `pytest tests/integration -v`
- [ ] Type checking passes: `mypy src/ --strict`
- [ ] Linting passes: `ruff check src/ tests/`
- [ ] Code coverage ≥80% on core modules: `pytest --cov=src --cov-fail-under=80`
- [ ] No hardcoded values in configuration (all via Pydantic Settings)
- [ ] All components use dependency injection (no global state)
- [ ] Logging captures correlation IDs across async boundaries
- [ ] MCTS engine processes ≥100 iterations/second on reference hardware
- [ ] API latency <500ms for standard queries (p95)
- [ ] All sensitive data sanitized in logs (API keys, secrets masked)
```

### 1.3 Problem Description (The "Three Paragraphs")

```
COORDINATION LOGIC:
The system orchestrates multiple specialized AI agents (HRM for hierarchical reasoning,
TRM for task refinement, Neural Meta-Controller for routing) that collaborate to solve
complex multi-step problems. A Monte Carlo Tree Search engine explores decision spaces
strategically, using UCB1 selection, neural policy/value networks for guidance, and
progressive widening for action space management. The Meta-Controller routes incoming
queries to the most appropriate agent based on extracted features (query complexity,
technical nature, confidence scores).

DATA FLOWS & STATE:
Queries flow: REST API → Framework Service → Meta-Controller → Agent(s) → LLM/Neural → Response.
State maintained: AgentState (TypedDict) with query, agent_outputs (accumulator pattern),
MCTS tree (root node, best action), confidence scores, consensus status, iteration counters.
Synchronization via LangGraph StateGraph with checkpointing for conversation memory.
MCTS state includes: node statistics (visits, value), simulation cache (LRU OrderedDict),
tree structure (parent/child relationships with depth tracking).

FAILURE MODES & INVARIANTS:
- LLM provider failure: Retry with exponential backoff (tenacity), fallback to alternate provider
- MCTS timeout: Return best action found so far, log incomplete search
- Agent disagreement: Consensus threshold check, escalate to meta-controller arbitration
- Memory pressure: LRU cache eviction, progressive widening limits
- Network partition: Circuit breaker pattern, graceful degradation
Invariants: sum(node.visits for children) ≤ parent.visits; UCB1 exploration always ≥0;
correlation_id present in all log entries within request scope.
```

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
LANGUAGE/RUNTIME:
- Python 3.10+ (type hints, asyncio, dataclasses)
- Async-first design (asyncio, aiohttp, aioboto3)

REQUIRED DEPENDENCIES:
- langgraph>=0.0.1 (graph orchestration)
- langchain>=0.1.0, langchain-core>=0.1.0 (LLM abstraction)
- pydantic>=2.0.0, pydantic-settings (configuration, validation)
- openai>=1.0.0 (LLM provider)
- httpx>=0.25.0 (async HTTP)
- tenacity>=8.2.0 (retry logic)
- opentelemetry-api, opentelemetry-sdk (distributed tracing)

SECURITY:
- No hardcoded secrets (use SecretStr, environment variables)
- All inputs validated via Pydantic models
- Sensitive data sanitized in logs (regex patterns for API keys, passwords)
- Rate limiting on API endpoints (configurable requests/minute)
- Authentication required for all API calls

COMPATIBILITY:
- Must run on Linux (Ubuntu 22.04+), macOS, Windows WSL2
- PostgreSQL 15+ for persistence (optional)
- Redis 7+ for distributed locks (optional)
- S3-compatible storage for artifacts
- Pinecone for vector storage (optional)
```

### 2.2 Soft Constraints (Preferences)

```
STYLE:
- Google Python Style Guide
- Line length: 120 characters (Black formatter)
- Docstrings: Google format with type annotations
- Import ordering: isort with black profile

ARCHITECTURE:
- Prefer composition over inheritance
- Use Protocol classes for interfaces (not ABC where possible)
- Dependency injection via constructor parameters
- Factory pattern for component creation
- Dataclasses for configuration objects
- TypedDict for state containers

PERFORMANCE:
- Async I/O for all network operations
- Connection pooling for HTTP clients
- LRU caching for expensive computations
- Semaphore for concurrent operation limits
- Progressive widening for MCTS action spaces

TESTING:
- pytest with asyncio_mode="auto"
- Fixtures in tests/fixtures/ directory
- Markers: @pytest.mark.{unit,integration,e2e,slow,benchmark}
- Property-based testing with hypothesis for core algorithms
- Coverage threshold: 80% for src/, 50% overall minimum
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure file organization within src/ for clarity
- Add PyPI dependencies not explicitly listed (document in pyproject.toml)
- Refactor adjacent code for consistency with new patterns
- Choose neural network architectures (RNN, Transformer, hybrid)
- Implement additional MCTS variants (PUCT, AlphaZero-style)
- Add new agent types following established patterns
- Create new test fixtures and scenarios
- Extend configuration with new Pydantic Settings fields
- Add observability (metrics, traces, logs) beyond minimum requirements
- Optimize algorithms without changing external interfaces
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- src/**/*.py (all source code)
- tests/**/*.py (all test code)
- tests/fixtures/**/*.py (test fixtures)
- config/**/*.yaml, config/**/*.json (configuration files)
- pyproject.toml (dependencies, tool config)
- docs/**/*.md (documentation updates)
- examples/**/*.py (usage examples)

OUT OF SCOPE:
- .github/**/* (CI/CD workflows - requires separate review)
- kubernetes/**/* (deployment manifests - ops team)
- monitoring/grafana/**/* (dashboard configs)
- training/checkpoints/**/* (model weights - version controlled separately)
- Files with header: # DO NOT MODIFY - GENERATED

RESTRICTED (modify with caution):
- src/config/settings.py (core settings - ensure backward compatibility)
- src/api/server.py (API surface - maintain stability)
- src/framework/graph.py (orchestration core - test thoroughly)
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within src/, tests/, docs/
- Dependency installation (add to pyproject.toml)
- Running tests, linters, type checkers
- Refactoring for consistency (rename, reorganize)
- Adding logging, metrics, tracing
- Creating test fixtures and scenarios
- Implementing new components following established patterns
- Bug fixes with clear root cause

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >3 modules
- Breaking changes to public API (src/api/*)
- Deletions of >100 lines without replacement
- New external service integrations
- Changes to authentication/authorization logic
- Database schema modifications
- Configuration schema changes that break backward compatibility

PROHIBITED (do not attempt):
- Commits directly to main branch (use feature branches)
- External API calls with side effects in tests (use mocks)
- Modifications to files marked OUT OF SCOPE
- Removing existing tests without replacement
- Disabling security features (auth, rate limiting, input validation)
- Adding hardcoded credentials or API keys
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5 (for complex debugging)
- Max files to modify in single pass: 25
- Time-boxed exploration: ≤15 min on research before asking
- Max test runtime for single test file: 60 seconds
- Max lines per function: 50 (prefer decomposition)
- Max cyclomatic complexity: 10 per function
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Format code (auto-fix)
black src/ tests/ --line-length 120
isort src/ tests/ --profile black

# 2. Lint (must pass)
ruff check src/ tests/ --fix
ruff check src/ tests/  # Verify no remaining issues

# 3. Type check (must pass)
mypy src/ --strict --ignore-missing-imports

# 4. Unit tests (must pass)
pytest tests/unit -v --tb=short -x  # Stop on first failure

# 5. Integration tests (must pass)
pytest tests/integration -v --tb=short

# 6. Coverage check (must meet threshold)
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=50

# 7. Security scan (should pass)
bandit -r src/ -ll

# 8. Smoke test
python -c "from src.config.settings import get_settings; print(get_settings().safe_dict())"
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
→ Run `ruff check --fix` to auto-fix
→ Review remaining issues manually
→ Fix and re-run

ON TYPE ERROR:
→ Read error message carefully
→ Check if type stub is missing (install types-* package)
→ Fix type annotations
→ If complex, add # type: ignore with explanation comment
→ Re-run mypy

ON TEST FAILURE:
→ Read failure output completely
→ Identify root cause (implementation bug vs test bug)
→ If implementation bug: fix implementation, re-run
→ If test bug: fix test with justification comment
→ If flaky: add @pytest.mark.flaky or investigate async timing

ON IMPORT ERROR:
→ Check if dependency is in pyproject.toml
→ Run `pip install -e ".[dev]"` to reinstall
→ Verify virtual environment is activated

ON REPEATED FAILURE (same error 3x):
→ Stop and document findings
→ List attempted solutions
→ Request human guidance with specific question
```

### 4.3 Success Verification

```
Before reporting completion:

1. All verification commands pass (Section 4.1)

2. Manual smoke tests:
   - Settings load: python -c "from src.config.settings import get_settings; s = get_settings(); print(s.LLM_PROVIDER)"
   - MCTS runs: python -c "from src.framework.mcts.core import MCTSEngine, MCTSConfig; e = MCTSEngine(MCTSConfig()); print('MCTS OK')"
   - Logging works: python -c "from src.observability.logging import setup_logging; setup_logging(); print('Logging OK')"

3. Generate change summary:
   - Files added/modified/deleted
   - New dependencies added
   - Configuration changes
   - Breaking changes (if any)
   - Test coverage delta
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory (CLAUDE.md)

```markdown
# CLAUDE.md - Project Context for AI Assistants

## Quick Start
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,neural]"
cp .env.example .env  # Configure API keys

# Verify
pytest tests/unit -v --tb=short
```

## Build Commands
- `pip install -e ".[dev]"`: Install with dev dependencies
- `pip install -e ".[dev,neural]"`: Include PyTorch for neural MCTS
- `black src/ tests/`: Format code
- `ruff check src/ tests/ --fix`: Lint with auto-fix
- `mypy src/`: Type check

## Test Commands
- `pytest tests/unit -v`: Run unit tests
- `pytest tests/integration -v`: Run integration tests
- `pytest tests/ -k "mcts"`: Run MCTS-related tests
- `pytest tests/ --cov=src`: Run with coverage
- `pytest tests/ -m "not slow"`: Skip slow tests

## Architecture Decisions
- [2024-01] Use Pydantic Settings v2 for configuration: Type safety, validation, env loading
- [2024-01] LangGraph for orchestration: Native async, checkpointing, visualization
- [2024-02] Protocol-based LLM adapters: Provider agnosticism without ABC overhead
- [2024-03] TypedDict for AgentState: Better IDE support than dataclass for state machines
- [2024-04] Correlation IDs via ContextVar: Async-safe request tracking

## Known Issues
- LMStudio adapter requires local server running (skip tests with LMSTUDIO_SKIP=1)
- Pinecone tests require valid API key (use mock in CI)
- Neural MCTS requires CUDA for reasonable performance (CPU fallback works but slow)

## Environment Variables (Required)
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`: LLM provider credentials
- `LLM_PROVIDER`: "openai" | "anthropic" | "lmstudio"
- `LOG_LEVEL`: DEBUG | INFO | WARNING | ERROR

## Environment Variables (Optional)
- `MCTS_ENABLED`: Enable MCTS exploration (default: true)
- `MCTS_ITERATIONS`: Search iterations (default: 100)
- `LANGSMITH_API_KEY`: For tracing
- `PINECONE_API_KEY`: For vector storage
```

### 5.2 Information to Preserve Across Sessions

```
ALWAYS PRESERVE:
- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
- Test patterns that work for async code
- Configuration patterns for different environments

UPDATE WHEN CHANGED:
- New dependencies added
- New test markers introduced
- Configuration schema changes
- API endpoint changes
```

### 5.3 Information That Can Be Re-derived

```
CAN BE RE-DERIVED (don't persist):
- File structure (scan with find/glob)
- Dependency versions (in pyproject.toml, pip freeze)
- Current test status (re-run pytest)
- Coverage numbers (re-run coverage)
- Lint status (re-run ruff)
```

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```bash
# 1. Read project context
cat CLAUDE.md 2>/dev/null || echo "No CLAUDE.md found"

# 2. Scan project structure
find src -name "*.py" -type f | head -30
find tests -name "*.py" -type f | head -20

# 3. Check configuration
cat pyproject.toml | head -50
ls -la src/config/

# 4. Identify entry points
grep -r "def main\|if __name__" src/ --include="*.py" | head -10

# 5. Run existing tests to establish baseline
pytest tests/unit -v --tb=short -q 2>&1 | tail -20
```

### 6.2 Implementation Order

```
1. UNDERSTAND (read before write)
   - Read existing code in the area you'll modify
   - Identify patterns already in use
   - Check for related tests

2. DESIGN (plan before code)
   - Use TodoWrite to break down into tasks
   - Identify dependencies between components
   - Consider edge cases and failure modes

3. IMPLEMENT CORE (smallest working version)
   - Start with data structures (dataclasses, TypedDict)
   - Implement core logic without optimizations
   - Use dependency injection from the start

4. ADD ERROR HANDLING
   - Define custom exceptions if needed
   - Add try/except with specific exception types
   - Ensure cleanup in finally blocks

5. ADD LOGGING & OBSERVABILITY
   - Use structured logging (src/observability/logging.py)
   - Add correlation ID tracking
   - Include timing for performance-critical paths

6. WRITE TESTS
   - Unit tests for individual functions
   - Integration tests for component interactions
   - Use existing fixtures from tests/fixtures/

7. RUN VERIFICATION LOOP
   - Execute all commands from Section 4.1
   - Fix issues iteratively
   - Re-run until all pass

8. REFACTOR IF NEEDED
   - Extract common patterns
   - Ensure consistency with existing code
   - Remove duplication

9. UPDATE DOCUMENTATION
   - Update CLAUDE.md with new commands/decisions
   - Add docstrings to new public functions
   - Update README if user-facing changes
```

### 6.3 Completion Checklist

```
□ All success criteria from Section 1.2 met
□ All verification commands from Section 4.1 pass
□ No hardcoded values (check with: grep -r "api_key.*=" src/)
□ Dependency injection used (no global state)
□ Logging includes correlation IDs
□ Tests cover happy path and error cases
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided to user
□ Known limitations documented
```

---

## SECTION 7: COMPONENT PATTERNS (Reference)

### 7.1 Configuration Pattern

```python
# CORRECT: Use Pydantic Settings with validation
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class MyComponentSettings(BaseSettings):
    """Settings for MyComponent - loaded from environment."""

    model_config = {"env_prefix": "MY_COMPONENT_"}

    enabled: bool = Field(default=True, description="Enable component")
    max_iterations: int = Field(default=100, ge=1, le=10000)
    timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("max_iterations")
    @classmethod
    def validate_iterations(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_iterations must be positive")
        return v

# WRONG: Hardcoded values
class BadComponent:
    def __init__(self):
        self.max_iterations = 100  # WRONG: hardcoded
        self.api_key = "sk-xxx"    # WRONG: hardcoded secret
```

### 7.2 Agent Pattern

```python
# CORRECT: Protocol-based with dependency injection
from typing import Protocol
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Agent configuration - no hardcoded values."""
    confidence_threshold: float = 0.7
    max_retries: int = 3

class AgentProtocol(Protocol):
    """Interface for all agents."""

    async def process(self, query: str, context: dict) -> AgentResult:
        """Process a query and return result."""
        ...

    def get_confidence(self) -> float:
        """Return confidence score for last result."""
        ...

class MyAgent:
    """Concrete agent implementation."""

    def __init__(
        self,
        config: AgentConfig,
        llm_client: LLMClient,
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._logger = logger

    async def process(self, query: str, context: dict) -> AgentResult:
        self._logger.info("Processing query", extra={"query_length": len(query)})
        # Implementation...
```

### 7.3 Factory Pattern

```python
# CORRECT: Factory with settings injection
from typing import Protocol

class ComponentFactory(Protocol):
    """Factory protocol for creating components."""

    def create(self, **kwargs) -> Component:
        ...

class LLMClientFactory:
    """Factory for creating LLM clients."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create(self, provider: str | None = None) -> LLMClient:
        provider = provider or self._settings.LLM_PROVIDER

        if provider == "openai":
            return OpenAIClient(
                api_key=self._settings.get_api_key(),
                timeout=self._settings.HTTP_TIMEOUT,
            )
        elif provider == "anthropic":
            return AnthropicClient(...)
        else:
            raise ValueError(f"Unknown provider: {provider}")
```

### 7.4 Logging Pattern

```python
# CORRECT: Structured logging with correlation
import logging
from contextvars import ContextVar
from src.observability.logging import get_correlation_id, sanitize_dict

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

class MyComponent:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    async def process(self, data: dict) -> Result:
        correlation_id = get_correlation_id()

        # Sanitize sensitive data before logging
        safe_data = sanitize_dict(data)

        self._logger.info(
            "Processing request",
            extra={
                "correlation_id": correlation_id,
                "data": safe_data,
                "component": self.__class__.__name__,
            }
        )

        try:
            result = await self._do_work(data)
            self._logger.info("Request completed", extra={"correlation_id": correlation_id})
            return result
        except Exception as e:
            self._logger.error(
                "Request failed",
                extra={"correlation_id": correlation_id, "error": str(e)},
                exc_info=True,
            )
            raise
```

### 7.5 Test Pattern

```python
# CORRECT: Async test with fixtures and markers
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_llm_client() -> AsyncMock:
    """Create mock LLM client for testing."""
    client = AsyncMock(spec=LLMClient)
    client.generate.return_value = LLMResponse(
        content="Test response",
        model="gpt-4",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )
    return client

@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test configuration - no hardcoded values in tests either."""
    return AgentConfig(
        confidence_threshold=0.5,  # Lower for testing
        max_retries=1,
    )

@pytest.mark.unit
@pytest.mark.asyncio
async def test_agent_processes_query(
    mock_llm_client: AsyncMock,
    agent_config: AgentConfig,
) -> None:
    """Test that agent correctly processes a query."""
    # Arrange
    agent = MyAgent(
        config=agent_config,
        llm_client=mock_llm_client,
        logger=MagicMock(),
    )
    query = "What is the capital of France?"

    # Act
    result = await agent.process(query, context={})

    # Assert
    assert result.content is not None
    assert result.confidence >= 0.0
    mock_llm_client.generate.assert_called_once()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_handles_llm_failure(
    agent_config: AgentConfig,
) -> None:
    """Test that agent gracefully handles LLM failures."""
    # Arrange
    failing_client = AsyncMock(spec=LLMClient)
    failing_client.generate.side_effect = LLMError("API unavailable")

    agent = MyAgent(
        config=agent_config,
        llm_client=failing_client,
        logger=MagicMock(),
    )

    # Act & Assert
    with pytest.raises(AgentError) as exc_info:
        await agent.process("test query", context={})

    assert "API unavailable" in str(exc_info.value)
```

### 7.6 MCTS Integration Pattern

```python
# CORRECT: MCTS with configurable parameters
from dataclasses import dataclass
from src.framework.mcts.core import MCTSEngine, MCTSConfig, MCTSState

@dataclass
class MCTSIntegrationConfig:
    """Configuration for MCTS integration - all values configurable."""
    enabled: bool = True
    iterations: int = 100
    exploration_weight: float = 1.414
    max_depth: int = 10
    simulation_timeout: float = 5.0

class MCTSOrchestrator:
    """Orchestrates MCTS with multi-agent system."""

    def __init__(
        self,
        config: MCTSIntegrationConfig,
        mcts_engine: MCTSEngine,
        agents: dict[str, AgentProtocol],
        logger: logging.Logger,
    ) -> None:
        self._config = config
        self._engine = mcts_engine
        self._agents = agents
        self._logger = logger

    async def search(self, initial_state: MCTSState) -> SearchResult:
        """Execute MCTS search with agent evaluation."""
        if not self._config.enabled:
            self._logger.info("MCTS disabled, returning default action")
            return SearchResult(action=None, confidence=0.0)

        self._logger.info(
            "Starting MCTS search",
            extra={
                "iterations": self._config.iterations,
                "exploration_weight": self._config.exploration_weight,
            }
        )

        result = await self._engine.search(
            initial_state,
            num_iterations=self._config.iterations,
            exploration_weight=self._config.exploration_weight,
        )

        return result
```

---

## SECTION 8: SUB-AGENT SPECIFICATIONS

### 8.1 Available Sub-Agents

```yaml
Explore:
  purpose: "Codebase exploration and understanding"
  use_when: "Need to understand structure, find patterns, answer questions about code"
  tools: [Glob, Grep, Read, WebFetch, WebSearch]

Plan:
  purpose: "Architecture and implementation planning"
  use_when: "Need to design approach, identify critical files, consider trade-offs"
  tools: [All tools]

Bash:
  purpose: "Command execution and git operations"
  use_when: "Need to run commands, git operations, terminal tasks"
  tools: [Bash]

general-purpose:
  purpose: "Complex multi-step tasks"
  use_when: "Task requires multiple rounds of search, read, and analysis"
  tools: [All tools]
```

### 8.2 Sub-Agent Usage Patterns

```markdown
## When to use Explore agent:
- Understanding existing code patterns
- Finding where functionality is implemented
- Answering "how does X work?" questions
- Discovering test patterns and fixtures

## When to use Plan agent:
- Designing new features or components
- Determining implementation strategy
- Identifying files that need modification
- Considering architectural trade-offs

## When to use Bash agent:
- Running test suites
- Git operations (commit, push, branch)
- Installing dependencies
- Running linters and formatters

## When to use general-purpose agent:
- Complex debugging requiring multiple searches
- Refactoring across multiple files
- Research tasks requiring web search
```

---

## SECTION 9: ENTERPRISE USE CASE TEMPLATES

### 9.1 M&A Due Diligence Platform

```yaml
use_case: "Autonomous M&A Due Diligence Platform"
description: "MCTS-guided multi-agent system exploring due diligence pathways"
target_buyers: ["Investment banks", "PE firms", "Corporate M&A teams"]

agents_required:
  - name: "DocumentAnalysisAgent"
    purpose: "Extract key terms from contracts, financials, legal docs"
    pattern: "HRM (hierarchical decomposition of document types)"

  - name: "RiskIdentificationAgent"
    purpose: "Identify hidden risks and red flags"
    pattern: "TRM (iterative refinement of risk categories)"

  - name: "SynergyExplorationAgent"
    purpose: "Explore potential synergies and value creation"
    pattern: "MCTS (explore combination scenarios)"

  - name: "ComplianceCheckAgent"
    purpose: "Verify regulatory compliance across jurisdictions"
    pattern: "Rule-based with LLM fallback"

mcts_configuration:
  state_space: "Due diligence pathways (financial, legal, operational, tech)"
  actions: "Deep-dive into specific area, cross-reference findings, escalate risk"
  reward: "Information gain + risk discovery + timeline efficiency"

configuration_required:
  - DOCUMENT_STORAGE_BUCKET: "S3 bucket for deal documents"
  - RISK_THRESHOLD: "Minimum risk score to flag (0.0-1.0)"
  - MAX_ANALYSIS_DEPTH: "Maximum levels of document drill-down"
  - PARALLEL_ANALYSES: "Number of concurrent analysis threads"
```

### 9.2 Clinical Trial Design Optimizer

```yaml
use_case: "Pharmaceutical Clinical Trial Design Optimizer"
description: "MCTS simulation of trial designs to maximize approval probability"
target_buyers: ["Pharma companies", "CROs", "Biotech"]

agents_required:
  - name: "CohortStrategyAgent"
    purpose: "Define patient inclusion/exclusion criteria"
    pattern: "HRM (hierarchical criteria decomposition)"

  - name: "EndpointOptimizationAgent"
    purpose: "Select primary/secondary endpoints"
    pattern: "MCTS (explore endpoint combinations)"

  - name: "RegulatoryAlignmentAgent"
    purpose: "Ensure FDA/EMA guideline compliance"
    pattern: "Rule-based + LLM interpretation"

  - name: "StatisticalPowerAgent"
    purpose: "Calculate sample sizes and power"
    pattern: "Deterministic + sensitivity analysis"

mcts_configuration:
  state_space: "Trial design parameters (cohorts, endpoints, sites, duration)"
  actions: "Modify criteria, add/remove endpoint, adjust sample size"
  reward: "Approval probability * (-cost) * (-timeline)"

configuration_required:
  - REGULATORY_GUIDELINE_VERSION: "FDA guidance version to use"
  - MIN_STATISTICAL_POWER: "Minimum acceptable power (0.8 typical)"
  - MAX_TRIAL_DURATION_MONTHS: "Maximum acceptable trial length"
  - BUDGET_CONSTRAINT_USD: "Total budget ceiling"
```

### 9.3 Regulatory Compliance Automation

```yaml
use_case: "Enterprise Regulatory Compliance Automation"
description: "Navigate multi-jurisdictional regulations, predict enforcement"
target_buyers: ["Fortune 500", "Big 4 consulting", "Legal tech"]

agents_required:
  - name: "RegulatoryMappingAgent"
    purpose: "Map applicable regulations to business operations"
    pattern: "HRM (jurisdiction → regulation → requirement)"

  - name: "GapAnalysisAgent"
    purpose: "Identify compliance gaps and remediation needs"
    pattern: "TRM (iterative gap refinement)"

  - name: "EnforcementPredictionAgent"
    purpose: "Predict regulatory enforcement priorities"
    pattern: "MCTS (explore enforcement scenarios)"

  - name: "RemediationPlannerAgent"
    purpose: "Generate prioritized remediation plans"
    pattern: "Optimization with constraints"

mcts_configuration:
  state_space: "Compliance posture across jurisdictions"
  actions: "Implement control, document evidence, request exemption"
  reward: "Risk reduction * (-implementation cost) * urgency"

configuration_required:
  - JURISDICTIONS: "List of applicable jurisdictions"
  - REGULATION_UPDATE_FREQUENCY: "How often to check for updates"
  - RISK_TOLERANCE: "Organization risk appetite (conservative/moderate/aggressive)"
  - REMEDIATION_BUDGET: "Annual compliance budget"
```

---

## SECTION 10: TESTING STRATEGY

### 10.1 Test Categories

```yaml
unit_tests:
  location: "tests/unit/"
  purpose: "Test individual functions and classes in isolation"
  patterns:
    - Mock all external dependencies
    - Test single behavior per test
    - Use descriptive test names: test_<function>_<scenario>_<expected>
  coverage_target: "90%"

integration_tests:
  location: "tests/integration/"
  purpose: "Test component interactions"
  patterns:
    - Use real components where feasible
    - Mock only external services (LLM APIs, databases)
    - Test data flow between components
  coverage_target: "70%"

e2e_tests:
  location: "tests/e2e/"
  purpose: "Test complete user scenarios"
  patterns:
    - Simulate real API calls
    - Use test fixtures for consistent data
    - Verify end-to-end behavior
  coverage_target: "50%"

property_tests:
  location: "tests/property/"
  purpose: "Test invariants with random inputs"
  patterns:
    - Use hypothesis for property-based testing
    - Focus on MCTS invariants (visit counts, UCB bounds)
    - Test serialization round-trips

benchmark_tests:
  location: "tests/benchmark/"
  purpose: "Measure and track performance"
  patterns:
    - Use pytest-benchmark
    - Track MCTS iterations/second
    - Monitor memory usage
```

### 10.2 Required Test Fixtures

```python
# tests/fixtures/llm_fixtures.py
@pytest.fixture
def mock_openai_response() -> dict:
    """Standard OpenAI API response structure."""
    return {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "model": "gpt-4",
    }

# tests/fixtures/mcts_fixtures.py
@pytest.fixture
def simple_mcts_state() -> MCTSState:
    """Simple state for MCTS testing."""
    return MCTSState(
        features={"query": "test", "depth": 0},
        available_actions=["action_a", "action_b", "action_c"],
    )

@pytest.fixture
def mcts_config() -> MCTSConfig:
    """Standard MCTS configuration for tests."""
    return MCTSConfig(
        num_iterations=10,  # Low for fast tests
        exploration_weight=1.414,
        seed=42,  # Reproducible
    )

# tests/fixtures/agent_fixtures.py
@pytest.fixture
def hrm_agent(mock_llm_client, test_settings) -> HRMAgent:
    """Configured HRM agent for testing."""
    return HRMAgent(
        config=HRMConfig.from_settings(test_settings),
        llm_client=mock_llm_client,
        logger=logging.getLogger("test"),
    )
```

### 10.3 Test Configuration

```toml
# pyproject.toml additions for testing
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
timeout = 300
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow tests (>10s)",
    "benchmark: Performance benchmarks",
    "property: Property-based tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 50
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

---

## SECTION 11: DEBUGGING & OBSERVABILITY

### 11.1 Logging Levels

```python
# Use appropriate log levels:
logger.debug("Detailed diagnostic info - variable values, loop iterations")
logger.info("Normal operation - request received, task completed")
logger.warning("Unexpected but recoverable - retry needed, fallback used")
logger.error("Operation failed - exception caught, task aborted")
logger.critical("System failure - cannot continue, immediate attention needed")
```

### 11.2 Debugging Checklist

```markdown
When debugging issues:

1. [ ] Check logs with correlation ID: grep <correlation_id> logs/
2. [ ] Verify configuration loaded correctly: settings.safe_dict()
3. [ ] Check MCTS tree state: engine.get_tree_stats()
4. [ ] Verify agent confidence scores: agent.get_confidence()
5. [ ] Check async task status: asyncio.all_tasks()
6. [ ] Review metrics: /metrics endpoint or prometheus
7. [ ] Check distributed trace: LangSmith dashboard
8. [ ] Verify rate limits: check 429 responses in logs
```

### 11.3 Performance Profiling

```python
# Use built-in profiling utilities
from src.observability.profiling import profile_async, MemoryTracker

@profile_async
async def my_expensive_operation():
    ...

# Memory tracking
tracker = MemoryTracker()
tracker.start()
# ... do work ...
report = tracker.stop()
print(f"Peak memory: {report.peak_mb}MB")
```

---

## APPENDIX A: QUICK REFERENCE

### Command Cheatsheet

```bash
# Setup
pip install -e ".[dev,neural]"          # Full install
cp .env.example .env                     # Configure

# Development
black src/ tests/                        # Format
ruff check src/ tests/ --fix            # Lint
mypy src/                                # Type check

# Testing
pytest tests/unit -v                     # Unit tests
pytest tests/ -k "mcts"                  # MCTS tests only
pytest tests/ --cov=src                  # With coverage
pytest tests/ -m "not slow"              # Skip slow

# Running
python -m src.api.server                 # Start API
python -m src.cli query "test"           # CLI query
```

### File Locations

```
Configuration:    src/config/settings.py
Logging:          src/observability/logging.py
MCTS Core:        src/framework/mcts/core.py
Agents:           src/agents/{hrm,trm,hybrid}_agent.py
Meta-Controller:  src/agents/meta_controller/
LLM Adapters:     src/adapters/llm/
Test Fixtures:    tests/fixtures/
```

### Environment Variables

```bash
# Required
LLM_PROVIDER=openai|anthropic|lmstudio
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# Optional
LOG_LEVEL=INFO
MCTS_ENABLED=true
MCTS_ITERATIONS=100
LANGSMITH_API_KEY=ls-...
```

---

## APPENDIX B: ANTI-PATTERNS TO AVOID

```python
# ❌ WRONG: Hardcoded values
class BadAgent:
    def __init__(self):
        self.api_key = "sk-123"  # Never hardcode secrets
        self.max_retries = 3     # Should be configurable
        self.model = "gpt-4"     # Should come from settings

# ✅ CORRECT: Dependency injection with configuration
class GoodAgent:
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        self._config = config
        self._llm = llm_client

# ❌ WRONG: Global state
_global_client = None
def get_client():
    global _global_client
    if _global_client is None:
        _global_client = Client()
    return _global_client

# ✅ CORRECT: Factory with explicit lifecycle
class ClientFactory:
    def __init__(self, settings: Settings):
        self._settings = settings

    def create(self) -> Client:
        return Client(config=self._settings)

# ❌ WRONG: Logging secrets
logger.info(f"Using API key: {api_key}")

# ✅ CORRECT: Sanitize before logging
logger.info(f"Using API key: {sanitize_dict({'key': api_key})}")

# ❌ WRONG: Sync I/O in async context
async def bad_fetch():
    response = requests.get(url)  # Blocks event loop

# ✅ CORRECT: Async I/O
async def good_fetch():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
```

---

*Template Version: 1.0*
*Last Updated: 2025-01*
*Aligned with: langgraph_multi_agent_mcts architecture patterns*
