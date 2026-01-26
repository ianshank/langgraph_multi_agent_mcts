# Phase 2: LLM-Guided MCTS for Code Generation - Implementation Template

> **Version:** 2.0 | **Status:** Implementation Ready | **Last Updated:** 2025-01

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
Building a production-ready LLM-guided Monte Carlo Tree Search (MCTS) system
for code generation with integrated training data collection for neural network distillation.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
Phase 2 succeeds when:
- [ ] All unit tests pass (>95% pass rate)
- [ ] Training data collector produces valid JSONL files for PyTorch DataLoader
- [ ] Neural policy/value networks train on collected data with decreasing loss
- [ ] HumanEval benchmark achieves >50% pass@1 with balanced preset
- [ ] Code generation latency <10s per problem with fast preset
- [ ] Memory usage <2GB during search
- [ ] Integration tests pass with HRM, TRM, Meta-Controller pipeline
- [ ] RAG context injection improves solution quality (A/B measurable)
```

### 1.3 Problem Description

**Paragraph 1: Core Coordination Logic**
The system orchestrates LLM-guided search for code generation problems. Given a problem
description and test cases, MCTS explores the solution space by generating code variants
(expansion), evaluating them against tests (simulation), and learning from results
(backpropagation). The LLM acts as both policy (suggesting promising variants) and value
(estimating success probability) networks, with training data collected to distill this
knowledge into faster neural networks.

**Paragraph 2: Data Flows and State**
Data flows: Problem → Root Node → [Select → Expand → Evaluate → Backpropagate]* → Best Solution.
State maintained: MCTS tree (nodes with visit counts, Q-values), episode training data
(state-action-value tuples), execution results cache. Synchronization: Training data
finalized only after episode completion; MCTS tree is single-threaded within an episode
but multiple episodes can run in parallel.

**Paragraph 3: Failure Modes and Invariants**
Failure modes: LLM API failures (retry with backoff), code execution timeout (mark as
failed, assign low value), memory exhaustion (limit tree size), invalid code syntax
(catch at execution). Invariants: Visit counts monotonically increase; Q-values bounded
[-1, 1]; training examples always have episode outcome assigned; no hardcoded API keys.

---

## SECTION 2: C4 ARCHITECTURE

### 2.1 Level 1: System Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM-Guided MCTS System                            │
│                         (Code Generation Platform)                          │
└───────────────────┬─────────────────────────────┬───────────────────────────┘
                    │                             │
        ┌───────────▼───────────┐     ┌───────────▼───────────┐
        │   LLM Providers       │     │   Storage Systems     │
        │ ┌─────────────────┐   │     │ ┌─────────────────┐   │
        │ │  OpenAI API     │   │     │ │   Pinecone      │   │
        │ │  Anthropic API  │   │     │ │   (RAG Store)   │   │
        │ │  Local LMStudio │   │     │ └─────────────────┘   │
        │ └─────────────────┘   │     │ ┌─────────────────┐   │
        └───────────────────────┘     │ │   S3/Local FS   │   │
                                      │ │ (Training Data) │   │
        ┌───────────────────────┐     │ └─────────────────┘   │
        │   Observability       │     └───────────────────────┘
        │ ┌─────────────────┐   │
        │ │   LangSmith     │   │     ┌───────────────────────┐
        │ │   W&B           │   │     │   External APIs       │
        │ │   Prometheus    │   │     │ ┌─────────────────┐   │
        │ └─────────────────┘   │     │ │   HumanEval     │   │
        └───────────────────────┘     │ │   (Benchmark)   │   │
                                      │ └─────────────────┘   │
                                      └───────────────────────┘
```

### 2.2 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LLM-Guided MCTS Container                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │  REST API       │    │  Framework      │    │  Storage        │             │
│  │  (FastAPI)      │───▶│  Service        │───▶│  Adapters       │             │
│  │                 │    │                 │    │                 │             │
│  └─────────────────┘    └────────┬────────┘    └─────────────────┘             │
│                                  │                                              │
│         ┌────────────────────────┼────────────────────────┐                    │
│         │                        │                        │                    │
│         ▼                        ▼                        ▼                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │  Meta-Controller│    │  LLM-Guided     │    │  Training       │             │
│  │  (Routing)      │    │  MCTS Engine    │    │  Pipeline       │             │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘             │
│           │                      │                      │                      │
│    ┌──────┴──────┐        ┌──────┴──────┐        ┌──────┴──────┐               │
│    │             │        │             │        │             │               │
│    ▼             ▼        ▼             ▼        ▼             ▼               │
│ ┌─────┐      ┌─────┐  ┌─────────┐  ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│ │ HRM │      │ TRM │  │Generator│  │Reflector│ │Data     │ │Neural   │         │
│ │Agent│      │Agent│  │ Agent   │  │ Agent   │ │Collector│ │Networks │         │
│ └─────┘      └─────┘  └─────────┘  └─────────┘ └─────────┘ └─────────┘         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Level 3: Component Diagram (LLM-Guided MCTS)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LLM-Guided MCTS Engine                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                          Search Controller                                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐          │  │
│  │  │  Select    │  │  Expand    │  │  Evaluate  │  │ Backprop   │          │  │
│  │  │  (UCB1)    │──▶│  (LLM)    │──▶│  (Test)   │──▶│  (Update)  │          │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                    │
│  │    Generator Agent      │    │    Reflector Agent      │                    │
│  │  ┌───────────────────┐  │    │  ┌───────────────────┐  │                    │
│  │  │ Prompt Builder    │  │    │  │ Prompt Builder    │  │                    │
│  │  │ Response Parser   │  │    │  │ Response Parser   │  │                    │
│  │  │ Variant Scorer    │  │    │  │ Value Estimator   │  │                    │
│  │  └───────────────────┘  │    │  └───────────────────┘  │                    │
│  └─────────────────────────┘    └─────────────────────────┘                    │
│                                                                                 │
│  ┌─────────────────────────┐    ┌─────────────────────────┐                    │
│  │    Code Executor        │    │    Data Collector       │                    │
│  │  ┌───────────────────┐  │    │  ┌───────────────────┐  │                    │
│  │  │ Sandbox Runner    │  │    │  │ Episode Manager   │  │                    │
│  │  │ Test Validator    │  │    │  │ Example Buffer    │  │                    │
│  │  │ Error Capture     │  │    │  │ JSONL Writer      │  │                    │
│  │  └───────────────────┘  │    │  └───────────────────┘  │                    │
│  └─────────────────────────┘    └─────────────────────────┘                    │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                          MCTS Tree Structure                              │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │  │
│  │  │ MCTSNode │───▶│ MCTSNode │───▶│ MCTSNode │───▶│ MCTSNode │            │  │
│  │  │ (Root)   │    │ (Var 1)  │    │ (Var 1a) │    │ (Leaf)   │            │  │
│  │  │ visits=N │    │ visits=M │    │ visits=K │    │ terminal │            │  │
│  │  │ Q=0.5    │    │ Q=0.7    │    │ Q=0.9    │    │ Q=1.0    │            │  │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Level 4: Code Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Key Classes & Interfaces                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  src/framework/mcts/llm_guided/                                                 │
│  ├── __init__.py              # Module exports                                  │
│  ├── config.py                # LLMGuidedMCTSConfig, presets                    │
│  ├── node.py                  # LLMGuidedMCTSNode, NodeState                    │
│  ├── agents.py                # GeneratorAgent, ReflectorAgent                  │
│  ├── executor.py              # CodeExecutor, CodeExecutionResult               │
│  ├── data_collector.py        # TrainingDataCollector, TrainingExample          │
│  ├── engine.py                # LLMGuidedMCTSEngine, MCTSSearchResult           │
│  ├── integration.py           # UnifiedSearchOrchestrator                       │
│  │                                                                              │
│  │  # Phase 2 Additions                                                         │
│  ├── training/                                                                  │
│  │   ├── __init__.py                                                            │
│  │   ├── dataset.py           # MCTSDataset (PyTorch)                           │
│  │   ├── networks.py          # PolicyNetwork, ValueNetwork                     │
│  │   ├── trainer.py           # DistillationTrainer                             │
│  │   └── metrics.py           # TrainingMetrics, EvaluationMetrics              │
│  │                                                                              │
│  ├── benchmark/                                                                 │
│  │   ├── __init__.py                                                            │
│  │   ├── humaneval.py         # HumanEvalBenchmark                              │
│  │   ├── runner.py            # BenchmarkRunner                                 │
│  │   └── metrics.py           # pass@k, execution accuracy                      │
│  │                                                                              │
│  └── rag/                                                                       │
│      ├── __init__.py                                                            │
│      ├── context.py           # RAGContextProvider                              │
│      └── prompts.py           # RAG-enhanced prompt templates                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## SECTION 3: FEASIBLE REGION (Constraints)

### 3.1 Hard Constraints (Violations = Failure)

```python
# Language/Runtime
- Python 3.11+
- asyncio for all I/O operations

# Required Dependencies
- Pydantic v2 for configuration and validation
- LangGraph for orchestration (optional but preferred)
- NumPy for numerical operations
- PyTorch for neural network training (optional for inference)

# Security
- NO hardcoded API keys (use settings from environment)
- Code execution in sandboxed environment
- Input validation on all external data

# Architecture
- All configuration via src/config/settings.py
- Dependency injection via factories
- Structured logging with correlation IDs
```

### 3.2 Soft Constraints (Preferences)

```python
# Style
- Black formatting with line length 120
- Type hints everywhere (mypy --strict compatible)
- Google-style docstrings

# Architecture
- Protocol classes for interfaces
- Dataclasses for immutable data structures
- TypedDict for state management
- Factory pattern for component creation

# Performance
- Async all I/O paths
- Cache LLM responses where appropriate
- Minimize tree traversals

# Testing
- pytest with >80% coverage on core logic
- Property-based tests for MCTS invariants
- Integration tests with mocked LLM clients
```

### 3.3 Anti-Constraints (Explicit Freedoms)

```python
# You ARE permitted to:
- Add new configuration presets
- Create additional agent types
- Modify prompt templates for better quality
- Add caching layers for performance
- Restructure training data format if needed
- Choose neural network architectures
- Add new evaluation metrics
```

---

## SECTION 4: PERMISSION ARCHITECTURE

### 4.1 Scope (What You Can Touch)

```
IN SCOPE:
- src/framework/mcts/llm_guided/       # Core MCTS components
- src/framework/mcts/llm_guided/training/  # NEW: Training pipeline
- src/framework/mcts/llm_guided/benchmark/ # NEW: Benchmark suite
- src/framework/mcts/llm_guided/rag/       # NEW: RAG integration
- tests/unit/test_llm_guided_*.py
- tests/integration/test_mcts_*.py
- docs/PHASE2_*.md

OUT OF SCOPE:
- src/config/settings.py (read only, add new settings via PR)
- src/agents/hrm_agent.py (use via adapters)
- src/agents/trm_agent.py (use via adapters)
- Production deployment configs
```

### 4.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation within src/framework/mcts/llm_guided/
- Test file creation
- Documentation updates
- Bug fixes in existing code
- Performance optimizations

CONFIRM FIRST (ask before proceeding):
- Changes to public API signatures
- New dependencies in pyproject.toml
- Breaking changes to training data format
- Changes affecting HRM/TRM integration

PROHIBITED (do not attempt):
- Commits to main branch directly
- Modifications to settings.py without PR
- Removal of existing tests
- Hardcoding API keys or secrets
```

---

## SECTION 5: FEEDBACK LOOP SPECIFICATION

### 5.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Format
black src/framework/mcts/llm_guided/ tests/ --line-length 120
isort src/framework/mcts/llm_guided/ tests/ --profile black

# 2. Lint
ruff check src/framework/mcts/llm_guided/ tests/ --fix

# 3. Type check
mypy src/framework/mcts/llm_guided/ --strict

# 4. Unit tests
pytest tests/unit/test_llm_guided_*.py -v --tb=short

# 5. Integration tests
pytest tests/integration/test_mcts_*.py -v --tb=short

# 6. Coverage
pytest tests/ -k "llm_guided" --cov=src/framework/mcts/llm_guided --cov-report=term-missing
```

### 5.2 Error Handling Protocol

```python
ON LINT FAILURE:
  → Fix automatically with --fix, re-run

ON TYPE ERROR:
  → Analyze error carefully
  → Add type hints or fix signatures
  → Re-run mypy

ON TEST FAILURE:
  → Read failure output completely
  → Identify root cause (not just symptoms)
  → Fix implementation (tests are usually correct)
  → Re-run specific test first, then full suite

ON REPEATED FAILURE (same error 3x):
  → Stop and analyze the pattern
  → Check if test assumptions are wrong
  → Consider if architecture needs adjustment
  → Report analysis for human review
```

### 5.3 Success Verification

```bash
# Before reporting completion:

# 1. All verification commands pass
# 2. Manual smoke test
python -c "
from src.framework.mcts.llm_guided import (
    LLMGuidedMCTSEngine,
    LLMGuidedMCTSConfig,
    create_llm_mcts_preset,
    LLMGuidedMCTSPreset,
)
print('Imports successful')
config = create_llm_mcts_preset(LLMGuidedMCTSPreset.FAST)
print(f'Config created: {config.name}')
"

# 3. Training data format validation
python -c "
from src.framework.mcts.llm_guided.training import MCTSDataset
print('Training module imports successful')
"
```

---

## SECTION 6: IMPLEMENTATION PATTERNS

### 6.1 Configuration Pattern

```python
# CORRECT - All config from settings
from src.config.settings import get_settings

class MyComponent:
    def __init__(self, config: MyConfig | None = None):
        settings = get_settings()
        self._config = config or self._default_config(settings)

    def _default_config(self, settings) -> MyConfig:
        return MyConfig(
            api_key=settings.get_api_key(),
            timeout=settings.HTTP_TIMEOUT,
        )

# WRONG - Hardcoded values
class BadComponent:
    def __init__(self):
        self.api_key = "sk-..."  # NEVER
        self.timeout = 30  # Should be from settings
```

### 6.2 Async Pattern

```python
# CORRECT - Async for all I/O
async def generate_code(self, problem: str) -> str:
    response = await self._llm_client.complete(
        prompt=self._build_prompt(problem),
        temperature=self._config.temperature,
    )
    return self._parse_response(response)

# WRONG - Blocking I/O
def generate_code_sync(self, problem: str) -> str:
    response = self._llm_client.complete_sync(...)  # NEVER in async codebase
    return response
```

### 6.3 Factory Pattern

```python
# CORRECT - Factory function for component creation
def create_mcts_engine(
    llm_client: LLMClientProtocol,
    preset: LLMGuidedMCTSPreset = LLMGuidedMCTSPreset.BALANCED,
    **overrides,
) -> LLMGuidedMCTSEngine:
    """Create engine with preset configuration."""
    config = create_llm_mcts_preset(preset)
    if overrides:
        config = config.copy(**overrides)
    return LLMGuidedMCTSEngine(llm_client, config)

# Usage
engine = create_mcts_engine(client, preset=LLMGuidedMCTSPreset.FAST)
```

### 6.4 Logging Pattern

```python
from src.observability.logging import get_structured_logger, get_correlation_id

logger = get_structured_logger(__name__)

class MyComponent:
    async def process(self, query: str) -> Result:
        logger.info(
            "Processing query",
            query_length=len(query),
            correlation_id=get_correlation_id(),
        )
        try:
            result = await self._do_work(query)
            logger.info("Query processed", result_size=len(result))
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            raise
```

### 6.5 Testing Pattern

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""
    client = MagicMock()
    client.complete = AsyncMock(return_value='{"variants": [{"code": "def f(): pass", "confidence": 0.8}]}')
    return client

@pytest.fixture
def mcts_config():
    """Create test configuration."""
    return LLMGuidedMCTSConfig(
        num_iterations=5,  # Small for fast tests
        collect_training_data=True,
        training_data_dir="./test_data",
    )

@pytest.mark.asyncio
async def test_search_finds_solution(mock_llm_client, mcts_config):
    """Test that MCTS search can find a solution."""
    engine = LLMGuidedMCTSEngine(mock_llm_client, mcts_config)

    result = await engine.search(
        problem="Write a function that returns 42",
        test_cases=["assert solution() == 42"],
    )

    assert result.num_iterations > 0
    assert result.tree_size >= 1
```

---

## SECTION 7: PHASE 2 IMPLEMENTATION CHECKLIST

### 7.1 Core Components

```
□ Enhanced TrainingDataCollector
  □ PyTorch Dataset integration
  □ Feature extraction for policy/value networks
  □ Efficient batching and shuffling
  □ Train/val/test split utilities

□ Neural Network Components
  □ PolicyNetwork (code variant selection)
  □ ValueNetwork (success probability estimation)
  □ Code encoder (transformer-based)
  □ DistillationTrainer

□ Benchmark Suite
  □ HumanEval problem loader
  □ Benchmark runner with metrics
  □ pass@k calculation
  □ Results reporting

□ RAG Integration
  □ RAGContextProvider
  □ Enhanced prompts with context
  □ Pinecone integration
```

### 7.2 Testing Requirements

```
□ Unit Tests
  □ test_mcts_dataset.py
  □ test_policy_network.py
  □ test_value_network.py
  □ test_distillation_trainer.py
  □ test_humaneval_benchmark.py
  □ test_rag_context.py

□ Integration Tests
  □ test_mcts_training_pipeline.py
  □ test_mcts_benchmark_flow.py
  □ test_mcts_rag_integration.py

□ Property Tests
  □ Training data format invariants
  □ Network output bounds
  □ MCTS tree structure invariants
```

### 7.3 Documentation

```
□ API documentation for new modules
□ Training guide with examples
□ Benchmark results template
□ RAG configuration guide
```

---

## SECTION 8: EXECUTION PROTOCOL

### 8.1 Initial Actions

```bash
# 1. Verify environment
python -c "from src.framework.mcts.llm_guided import *; print('Phase 1 OK')"

# 2. Run existing tests
pytest tests/unit/test_llm_guided*.py -v --tb=short

# 3. Check current structure
find src/framework/mcts/llm_guided -type f -name "*.py"
```

### 8.2 Implementation Order

```
1. Create training/ directory structure
2. Implement MCTSDataset (PyTorch integration)
3. Implement neural networks (PolicyNetwork, ValueNetwork)
4. Implement DistillationTrainer
5. Add training tests

6. Create benchmark/ directory structure
7. Implement HumanEval loader
8. Implement BenchmarkRunner
9. Add benchmark tests

10. Create rag/ directory structure
11. Implement RAGContextProvider
12. Update prompts with RAG
13. Add RAG tests

14. Update __init__.py exports
15. Run full test suite
16. Update documentation
```

### 8.3 Completion Checklist

```
□ All success criteria met (Section 1.2)
□ All verification commands pass (Section 5.1)
□ CLAUDE.md updated with new commands
□ Tests added for all new functionality
□ Documentation complete
□ Code follows all patterns (Section 6)
□ No hardcoded values
□ Summary of changes provided
```

---

## APPENDIX A: DATA STRUCTURES

### A.1 Training Example Schema

```python
@dataclass
class TrainingExample:
    # State representation
    state_code: str           # Current code
    state_problem: str        # Problem description
    state_hash: str           # Unique state identifier
    depth: int                # Tree depth

    # Policy targets
    llm_action_probs: dict[str, float]   # LLM teacher labels
    mcts_action_probs: dict[str, float]  # MCTS improved policy

    # Value targets
    llm_value_estimate: float  # LLM value prediction
    outcome: float             # Final episode outcome [-1, 1]

    # Metadata
    episode_id: str
    timestamp: float
    visits: int
    q_value: float
```

### A.2 Benchmark Result Schema

```python
@dataclass
class BenchmarkResult:
    benchmark_name: str        # e.g., "humaneval"
    config_name: str           # MCTS configuration used

    # Metrics
    pass_at_1: float           # Single-attempt success rate
    pass_at_k: dict[int, float]  # Multi-attempt success rates
    execution_accuracy: float  # Code that runs without error

    # Statistics
    total_problems: int
    problems_solved: int
    avg_iterations: float
    avg_time_ms: float
    total_tokens: int

    # Per-problem results
    problem_results: list[ProblemResult]
```

---

## APPENDIX B: CONFIGURATION PRESETS

```python
# Fast: Quick iteration, minimal exploration
FAST = {
    "num_iterations": 10,
    "max_depth": 5,
    "num_variants": 2,
    "collect_training_data": False,
}

# Balanced: Default for most use cases
BALANCED = {
    "num_iterations": 30,
    "max_depth": 10,
    "num_variants": 3,
    "collect_training_data": True,
}

# Thorough: Maximum quality, longer runtime
THOROUGH = {
    "num_iterations": 100,
    "max_depth": 15,
    "num_variants": 4,
    "exploration_weight": 2.0,
}

# Data Collection: Optimized for training
DATA_COLLECTION = {
    "num_iterations": 50,
    "early_termination_on_solution": False,
    "save_mcts_policy": True,
}

# Benchmark: For HumanEval evaluation
BENCHMARK = {
    "num_iterations": 30,
    "execution_timeout_seconds": 10.0,
    "collect_training_data": True,
}
```

---

*Template Version: 2.0 | Generated for Phase 2 LLM-Guided MCTS Implementation*
