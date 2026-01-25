# Continuous Play & Learning Implementation Plan

## Executive Summary

This plan implements a production-ready continuous play and learning system for the LangGraph Multi-Agent MCTS framework, following 2025 best practices with dynamic/modular/reusable components, comprehensive testing, and no hardcoded values.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Design](#2-component-design)
3. [Configuration System](#3-configuration-system)
4. [Testing Strategy](#4-testing-strategy)
5. [Implementation Phases](#5-implementation-phases)
6. [Metrics & Monitoring](#6-metrics--monitoring)
7. [Validation & Execution](#7-validation--execution)

---

## 1. Architecture Overview

### Existing Components Identified

| Component | Location | Purpose |
|-----------|----------|---------|
| `ContinuousLearningSession` | `src/games/chess/continuous_learning.py` | Self-play loop orchestration |
| `OnlineLearner` | `src/games/chess/continuous_learning.py` | Experience buffer & online training |
| `ScoreCard` | `src/games/chess/continuous_learning.py` | Win/loss/Elo tracking |
| `LearningDashboard` | `examples/chess_demo/learning_dashboard.py` | Real-time analytics |
| `PerformanceMonitor` | `src/training/performance_monitor.py` | Latency/memory tracking |
| `ChessEnsembleAgent` | `src/games/chess/ensemble_agent.py` | Multi-agent ensemble |
| `Chess UI` | `src/games/chess/ui.py` | Gradio interface |

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ContinuousPlayOrchestrator                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ SessionConfig│  │ MetricsConfig│  │ LearningConfig        │  │
│  │ (from env)   │  │ (from env)   │  │ (from env)            │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              ContinuousLearningSession                    │   │
│  │  - Self-play game loop                                    │   │
│  │  - Temperature scheduling                                 │   │
│  │  - Experience collection                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │ MetricsCollector│  │ OnlineLearner │  │ ScoreCard      │    │
│  │ - Prometheus   │  │ - Replay buf  │  │ - Elo tracking │    │
│  │ - W&B/Braintrust│  │ - SGD updates │  │ - Streaks      │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   ReportGenerator                         │   │
│  │  - HTML dashboard export                                  │   │
│  │  - JSON metrics export                                    │   │
│  │  - Improvement analysis                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 ContinuousPlayOrchestrator

**Location**: `src/training/continuous_play_orchestrator.py`

```python
@dataclass
class OrchestratorConfig:
    """All configuration loaded from environment or config file."""
    session_duration_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_DURATION_MIN", "20")))
    max_games: int = field(default_factory=lambda: int(os.getenv("MAX_GAMES", "100")))
    checkpoint_interval_games: int = field(default_factory=lambda: int(os.getenv("CHECKPOINT_INTERVAL", "10")))
    metrics_export_interval_sec: int = field(default_factory=lambda: int(os.getenv("METRICS_INTERVAL", "30")))
    enable_wandb: bool = field(default_factory=lambda: os.getenv("ENABLE_WANDB", "false").lower() == "true")
    enable_prometheus: bool = field(default_factory=lambda: os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true")
    report_output_dir: str = field(default_factory=lambda: os.getenv("REPORT_DIR", "./reports"))
```

**Key Methods**:
- `run_session()` - Main entry point
- `_game_completed_callback()` - Per-game metrics
- `_learning_update_callback()` - Training progress
- `generate_improvement_report()` - Final analysis

### 2.2 MetricsAggregator

**Location**: `src/training/metrics_aggregator.py`

Unified metrics collection combining:
- Prometheus counters/gauges/histograms
- W&B experiment tracking
- Braintrust logging
- In-memory statistics for real-time dashboard

**Metrics Tracked**:

| Category | Metric | Type |
|----------|--------|------|
| Games | total_games | Counter |
| Games | games_per_minute | Gauge |
| Games | white_win_rate | Gauge |
| Games | draw_rate | Gauge |
| Learning | training_loss | Histogram |
| Learning | positions_learned | Counter |
| Learning | buffer_size | Gauge |
| Performance | avg_move_time_ms | Histogram |
| Performance | avg_game_duration_ms | Histogram |
| Improvement | elo_estimate | Gauge |
| Improvement | elo_delta_per_10_games | Gauge |

### 2.3 ImprovementAnalyzer

**Location**: `src/training/improvement_analyzer.py`

Analyzes learning progress:
- Elo rating progression
- Win rate trends (rolling window)
- Loss convergence
- Move quality metrics
- Statistical significance testing

---

## 3. Configuration System

### 3.1 Environment Variables (No Hardcoded Values)

```bash
# Session Configuration
SESSION_DURATION_MIN=20
MAX_GAMES=100
MAX_MOVES_PER_GAME=150

# Learning Configuration
LEARN_EVERY_N_GAMES=5
MIN_GAMES_BEFORE_LEARNING=10
LEARNING_BATCH_SIZE=256
LEARNING_RATE=0.001

# Temperature Scheduling
TEMPERATURE_SCHEDULE=linear_decay
INITIAL_TEMPERATURE=1.0
FINAL_TEMPERATURE=0.1
TEMPERATURE_DECAY_GAMES=50

# Metrics & Monitoring
ENABLE_WANDB=false
ENABLE_PROMETHEUS=true
ENABLE_BRAINTRUST=false
METRICS_EXPORT_INTERVAL=30
PROMETHEUS_PORT=8000

# Reporting
REPORT_DIR=./reports
CHECKPOINT_DIR=./checkpoints

# Chess Configuration
CHESS_PRESET=small  # small, medium, large
DEVICE=cpu  # cpu, cuda
```

### 3.2 Configuration Loader

```python
# src/training/config_loader.py
@dataclass
class ContinuousPlayConfig:
    """Unified configuration loaded entirely from environment."""

    @classmethod
    def from_env(cls) -> "ContinuousPlayConfig":
        """Load all configuration from environment variables."""
        return cls(
            session=SessionConfig.from_env(),
            learning=LearningConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            chess=ChessConfig.from_preset(os.getenv("CHESS_PRESET", "small")),
        )
```

---

## 4. Testing Strategy

### 4.1 Test Pyramid

```
                    ┌─────────────────┐
                    │   E2E Tests     │  (5%)
                    │  User Journeys  │
                   ─┼─────────────────┼─
                  / │  Performance    │ \
                 /  │    Tests        │  \  (10%)
                /   │  Sanity/Smoke   │   \
               ─────┼─────────────────┼─────
              /     │  Integration    │     \
             /      │    Tests        │      \  (25%)
            /       │ Component Tests │       \
           ─────────┼─────────────────┼─────────
          /         │   Unit Tests    │         \
         /          │                 │          \  (60%)
        /           │  Fixtures &     │           \
       /            │   Mocks         │            \
      ──────────────┴─────────────────┴──────────────
```

### 4.2 Test Categories

#### Unit Tests (`tests/unit/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_orchestrator_config.py` | Config loading from env |
| `test_metrics_aggregator.py` | Metrics collection |
| `test_improvement_analyzer.py` | Elo/win rate analysis |
| `test_scorecard.py` | Win/loss tracking |
| `test_online_learner.py` | Experience buffer |
| `test_temperature_scheduling.py` | Temp decay algorithms |

**Example**:
```python
@pytest.mark.unit
class TestOrchestratorConfig:
    def test_loads_from_environment(self, monkeypatch):
        monkeypatch.setenv("SESSION_DURATION_MIN", "30")
        monkeypatch.setenv("MAX_GAMES", "50")
        config = OrchestratorConfig.from_env()
        assert config.session_duration_minutes == 30
        assert config.max_games == 50

    def test_uses_defaults_when_env_missing(self):
        config = OrchestratorConfig()
        assert config.session_duration_minutes == 20  # default
```

#### Integration Tests (`tests/integration/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_learning_pipeline.py` | Game → Experience → Training flow |
| `test_metrics_export.py` | Prometheus/W&B integration |
| `test_checkpoint_restore.py` | Save/load session state |
| `test_agent_ensemble.py` | Multi-agent coordination |

**Example**:
```python
@pytest.mark.integration
async def test_game_generates_training_data():
    """Verify game play produces valid training examples."""
    session = create_learning_session(preset="small", max_games=1)

    record = await session.play_single_game("test_001", temperature=1.0)

    assert record.result != GameResult.IN_PROGRESS
    assert len(record.moves) > 0
    assert session.learner.get_buffer_size() > 0
```

#### Functional Tests (`tests/functional/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_session_lifecycle.py` | Start → Pause → Resume → Stop |
| `test_learning_callbacks.py` | Callback invocation |
| `test_scorecard_updates.py` | Statistics accuracy |
| `test_report_generation.py` | HTML/JSON output |

#### E2E Tests (`tests/e2e/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_full_learning_session.py` | Complete 5-game mini session |
| `test_user_journey_ui.py` | UI interaction flow |
| `test_metrics_dashboard.py` | Dashboard data flow |

**Example**:
```python
@pytest.mark.e2e
@pytest.mark.timeout(300)  # 5 minute timeout
async def test_complete_mini_session():
    """Run a complete mini learning session."""
    config = ContinuousPlayConfig.from_env()
    config.session.max_games = 3
    config.session.session_duration_minutes = 2

    orchestrator = ContinuousPlayOrchestrator(config)
    result = await orchestrator.run_session()

    assert result.scorecard.total_games == 3
    assert result.report_path.exists()
    assert result.metrics.elo_estimate >= 1400
```

#### Performance Tests (`tests/performance/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_game_throughput.py` | Games per minute |
| `test_learning_latency.py` | Training step timing |
| `test_memory_stability.py` | Memory growth over time |
| `test_concurrent_sessions.py` | Multiple parallel sessions |

**Example**:
```python
@pytest.mark.performance
@pytest.mark.timeout(120)
async def test_game_throughput():
    """Verify minimum game throughput."""
    session = create_learning_session(preset="small", max_games=10)

    start = time.time()
    await session.run_session(max_games=10)
    elapsed = time.time() - start

    games_per_minute = 10 / (elapsed / 60)
    assert games_per_minute >= 2.0, f"Too slow: {games_per_minute:.2f} games/min"
```

#### Regression Tests (`tests/regression/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_deterministic_games.py` | Seeded reproducibility |
| `test_known_good_outcomes.py` | Golden file validation |
| `test_elo_calculation.py` | Elo formula accuracy |

#### Sanity/Smoke Tests (`tests/smoke/continuous_play/`)

| Test File | Coverage |
|-----------|----------|
| `test_imports.py` | All modules importable |
| `test_config_valid.py` | Config loads without error |
| `test_session_starts.py` | Session initializes |

### 4.3 Test Fixtures

```python
# tests/fixtures/continuous_play_fixtures.py

@pytest.fixture
def minimal_config():
    """Minimal config for fast tests."""
    return ContinuousPlayConfig(
        session=SessionConfig(max_games=1, session_duration_minutes=1),
        learning=LearningConfig(learn_every_n_games=1),
        chess=ChessConfig.from_preset("small"),
    )

@pytest.fixture
def mock_ensemble_agent():
    """Mock agent that returns deterministic moves."""
    agent = Mock(spec=ChessEnsembleAgent)
    agent.get_best_move = AsyncMock(return_value=MockMoveResponse("e2e4"))
    return agent

@pytest.fixture
def seeded_session(minimal_config):
    """Session with fixed seed for reproducibility."""
    return ContinuousLearningSession(
        chess_config=minimal_config.chess,
        seed=42,
    )
```

### 4.4 Test Markers

```python
# pyproject.toml additions
markers = [
    "continuous_play: Continuous play system tests",
    "learning: Online learning tests",
    "metrics: Metrics collection tests",
    "session: Session lifecycle tests",
]
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Tests First)

**Files to Create**:
1. `src/training/continuous_play_orchestrator.py`
2. `src/training/continuous_play_config.py`
3. `tests/unit/continuous_play/test_orchestrator_config.py`
4. `tests/unit/continuous_play/test_metrics_aggregator.py`

**Deliverables**:
- [ ] Configuration loader with env var support
- [ ] Unit tests for config (100% coverage)
- [ ] Smoke test for imports

### Phase 2: Metrics & Monitoring

**Files to Create**:
1. `src/training/metrics_aggregator.py`
2. `src/training/improvement_analyzer.py`
3. `tests/unit/continuous_play/test_improvement_analyzer.py`
4. `tests/integration/continuous_play/test_metrics_export.py`

**Deliverables**:
- [ ] Unified metrics collector
- [ ] Prometheus integration tests
- [ ] Improvement analysis algorithms

### Phase 3: Session Integration

**Files to Modify**:
1. `src/games/chess/continuous_learning.py` (add callbacks)
2. `src/games/chess/ui.py` (integrate orchestrator)

**Files to Create**:
1. `tests/integration/continuous_play/test_learning_pipeline.py`
2. `tests/functional/continuous_play/test_session_lifecycle.py`

**Deliverables**:
- [ ] Full session lifecycle
- [ ] Integration tests passing
- [ ] Functional tests passing

### Phase 4: Reporting & Analysis

**Files to Create**:
1. `src/training/report_generator.py`
2. `tests/functional/continuous_play/test_report_generation.py`

**Deliverables**:
- [ ] HTML report generation
- [ ] JSON export
- [ ] Improvement walkthrough

### Phase 5: E2E & Performance Validation

**Files to Create**:
1. `tests/e2e/continuous_play/test_full_learning_session.py`
2. `tests/performance/continuous_play/test_game_throughput.py`
3. `tests/regression/continuous_play/test_deterministic_games.py`

**Deliverables**:
- [ ] Complete E2E test suite
- [ ] Performance benchmarks
- [ ] Regression baseline

---

## 6. Metrics & Monitoring

### 6.1 Real-Time Dashboard Metrics

```python
# Exposed via /metrics endpoint
mcts_continuous_play_games_total{result="white_win|black_win|draw"}
mcts_continuous_play_session_duration_seconds
mcts_continuous_play_elo_estimate
mcts_continuous_play_training_loss
mcts_continuous_play_positions_learned_total
mcts_continuous_play_buffer_size
mcts_continuous_play_games_per_minute
```

### 6.2 Improvement Indicators

| Indicator | Measurement | Target |
|-----------|-------------|--------|
| Elo Growth | Elo delta over session | > +50 per 20 min |
| Win Rate Trend | Rolling 10-game avg | Increasing |
| Loss Convergence | Training loss stddev | Decreasing |
| Game Length | Avg moves per game | Stable or increasing |
| Decision Quality | High-confidence moves % | > 60% |

### 6.3 Alert Thresholds

```python
ALERTS = {
    "elo_regression": lambda elo_delta: elo_delta < -100,
    "loss_spike": lambda loss, avg: loss > avg * 3.0,
    "memory_growth": lambda mb_delta: mb_delta > 500,
    "game_timeout_rate": lambda rate: rate > 0.1,
}
```

---

## 7. Validation & Execution

### 7.1 Pre-Run Checklist

```bash
# Dependency verification
pip install -r requirements.txt
python -c "from src.games.chess.continuous_learning import ContinuousLearningSession"

# Config validation
python -c "from src.training.continuous_play_config import ContinuousPlayConfig; ContinuousPlayConfig.from_env()"

# Smoke tests
pytest tests/smoke/continuous_play/ -v

# Unit tests
pytest tests/unit/continuous_play/ -v --cov=src/training

# Integration tests
pytest tests/integration/continuous_play/ -v
```

### 7.2 20-Minute Session Execution

```bash
# Set environment
export SESSION_DURATION_MIN=20
export MAX_GAMES=100
export CHESS_PRESET=small
export ENABLE_PROMETHEUS=true
export REPORT_DIR=./reports/$(date +%Y%m%d_%H%M%S)

# Launch UI with continuous learning
python -c "
from src.games.chess.ui import create_chess_ui
demo = create_chess_ui()
demo.launch(server_name='0.0.0.0', server_port=7860)
"
```

### 7.3 Post-Session Analysis

```bash
# Generate report
python -m src.training.report_generator --input ./reports/latest --output ./analysis

# View metrics
cat ./reports/latest/metrics.json | jq '.improvement'

# Open dashboard
open ./reports/latest/dashboard.html
```

### 7.4 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Session Completes | 100% | No crashes |
| Games Played | >= 10 | Counter metric |
| Elo Change | Any direction | Tracked |
| Reports Generated | HTML + JSON | File exists |
| Tests Pass | 100% | CI green |
| Memory Stable | < 500MB growth | Profiler |

---

## Appendix A: File Structure

```
src/training/
├── __init__.py
├── continuous_play_orchestrator.py  # NEW
├── continuous_play_config.py        # NEW
├── metrics_aggregator.py            # NEW
├── improvement_analyzer.py          # NEW
├── report_generator.py              # NEW
├── performance_monitor.py           # EXISTING
├── experiment_tracker.py            # EXISTING
└── replay_buffer.py                 # EXISTING

tests/
├── unit/continuous_play/
│   ├── __init__.py
│   ├── test_orchestrator_config.py
│   ├── test_metrics_aggregator.py
│   ├── test_improvement_analyzer.py
│   └── test_scorecard.py
├── integration/continuous_play/
│   ├── __init__.py
│   ├── test_learning_pipeline.py
│   └── test_metrics_export.py
├── functional/continuous_play/
│   ├── __init__.py
│   ├── test_session_lifecycle.py
│   └── test_report_generation.py
├── e2e/continuous_play/
│   ├── __init__.py
│   └── test_full_learning_session.py
├── performance/continuous_play/
│   ├── __init__.py
│   └── test_game_throughput.py
├── regression/continuous_play/
│   ├── __init__.py
│   └── test_deterministic_games.py
└── smoke/continuous_play/
    ├── __init__.py
    └── test_imports.py

docs/
└── implementation_plan.md           # THIS FILE
```

---

## Appendix B: Dependencies

```
# requirements-continuous-play.txt
torch>=2.1.0
numpy>=1.24.0
gradio>=4.0.0
python-chess>=1.10.0
prometheus-client>=0.18.0
wandb>=0.16.0  # optional
psutil>=5.9.0
pyyaml>=6.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-timeout>=2.1.0
hypothesis>=6.88.0
```

---

**Document Version**: 1.0.0
**Created**: 2025-12-12
**Author**: Claude Code
**Status**: Ready for Implementation
