"""
Unit tests for LLM-Guided MCTS Benchmark Module (Phase 2).

Tests:
- HumanEval problem loading and parsing
- Benchmark metrics computation (pass@k)
- Benchmark runner configuration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Check for numpy availability
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Skip all tests if numpy not available (required for metrics)
pytestmark = pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="numpy not available")


class TestHumanEvalProblem:
    """Tests for HumanEvalProblem."""

    def test_problem_creation(self):
        """Test creating a HumanEval problem."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalProblem

        problem = HumanEvalProblem(
            task_id="HumanEval/0",
            prompt='def foo(x):\n    """Return x + 1."""\n',
            canonical_solution="    return x + 1\n",
            entry_point="foo",
            test="def check(candidate):\n    assert candidate(1) == 2",
        )

        assert problem.task_id == "HumanEval/0"
        assert problem.function_name == "foo"
        assert "Return x + 1" in problem.docstring

    def test_problem_extract_test_cases(self):
        """Test extracting test cases from problem."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalProblem

        problem = HumanEvalProblem(
            task_id="HumanEval/0",
            prompt='def foo(x):\n    """Return x + 1."""\n',
            canonical_solution="    return x + 1\n",
            entry_point="foo",
            test="""
def check(candidate):
    assert candidate(1) == 2
    assert candidate(0) == 1
    assert candidate(-1) == 0
""",
        )

        assert len(problem.test_cases) == 3
        assert "assert candidate(1) == 2" in problem.test_cases

    def test_problem_difficulty_estimation(self):
        """Test difficulty estimation."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalProblem

        # Simple problem
        simple = HumanEvalProblem(
            task_id="HumanEval/0",
            prompt='def foo(x):\n    """Return x."""\n',
            canonical_solution="    return x\n",
            entry_point="foo",
            test="def check(c):\n    assert c(1) == 1",
        )
        assert simple.difficulty == "easy"

    def test_problem_to_dict(self):
        """Test converting problem to dictionary."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalProblem

        problem = HumanEvalProblem(
            task_id="HumanEval/0",
            prompt='def foo(x):\n    """Doc."""\n',
            canonical_solution="    return x\n",
            entry_point="foo",
            test="def check(c): pass",
        )

        d = problem.to_dict()

        assert d["task_id"] == "HumanEval/0"
        assert d["entry_point"] == "foo"
        assert "prompt" in d
        assert "canonical_solution" in d

    def test_problem_from_dict(self):
        """Test creating problem from dictionary."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalProblem

        data = {
            "task_id": "HumanEval/1",
            "prompt": "def bar(): pass",
            "canonical_solution": "pass",
            "entry_point": "bar",
            "test": "check(bar)",
        }

        problem = HumanEvalProblem.from_dict(data)

        assert problem.task_id == "HumanEval/1"
        assert problem.entry_point == "bar"

    def test_problem_description(self):
        """Test getting formatted problem description."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalProblem

        problem = HumanEvalProblem(
            task_id="HumanEval/0",
            prompt='def foo(x):\n    """Doc."""\n',
            canonical_solution="    return x\n",
            entry_point="foo",
            test="def check(c): assert c(1) == 1",
        )

        desc = problem.get_problem_description()

        assert "HumanEval/0" in desc
        assert "foo" in desc


class TestHumanEvalBenchmark:
    """Tests for HumanEvalBenchmark."""

    def test_benchmark_creation_default(self):
        """Test creating benchmark with default (sample) problems."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalBenchmark

        benchmark = HumanEvalBenchmark()

        # Should have sample problems
        assert len(benchmark) > 0

    def test_benchmark_iteration(self):
        """Test iterating over benchmark problems."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalBenchmark

        benchmark = HumanEvalBenchmark()

        problems = list(benchmark)

        assert len(problems) == len(benchmark)
        for p in problems:
            assert p.task_id is not None

    def test_benchmark_getitem(self):
        """Test getting problem by ID."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalBenchmark

        benchmark = HumanEvalBenchmark()
        problem_ids = benchmark.get_problem_ids()

        if problem_ids:
            problem = benchmark[problem_ids[0]]
            assert problem.task_id == problem_ids[0]

    def test_benchmark_sample(self):
        """Test sampling problems."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalBenchmark

        benchmark = HumanEvalBenchmark()

        sample = benchmark.sample(2, seed=42)

        assert len(sample) == min(2, len(benchmark))

    def test_benchmark_load_from_file(self, tmp_path):
        """Test loading benchmark from file."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalBenchmark

        # Create a test file
        data_file = tmp_path / "humaneval.jsonl"
        problems = [
            {
                "task_id": "HumanEval/100",
                "prompt": "def test(): pass",
                "canonical_solution": "pass",
                "entry_point": "test",
                "test": "check(test)",
            },
        ]

        with open(data_file, "w") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")

        benchmark = HumanEvalBenchmark(data_path=data_file)

        assert len(benchmark) == 1
        assert "HumanEval/100" in benchmark.get_problem_ids()


class TestBenchmarkMetrics:
    """Tests for benchmark metrics."""

    def test_pass_at_k_computation(self):
        """Test pass@k computation."""
        from src.framework.mcts.llm_guided.benchmark.metrics import compute_pass_at_k

        # If we have 10 samples and 3 correct, pass@1 should be ~0.3
        # Using the unbiased estimator
        result = compute_pass_at_k(n=10, c=3, k=1)

        assert 0 < result < 1

        # If all correct, pass@k = 1
        assert compute_pass_at_k(n=10, c=10, k=1) == 1.0

        # If none correct, pass@k = 0
        assert compute_pass_at_k(n=10, c=0, k=1) == 0.0

    def test_pass_at_k_for_problems(self):
        """Test pass@k computation for multiple problems."""
        from src.framework.mcts.llm_guided.benchmark.metrics import (
            ProblemResult,
            compute_pass_at_k_for_problems,
        )

        results = [
            ProblemResult(task_id="p1", solved=True, num_attempts=5, num_passed=2),
            ProblemResult(task_id="p2", solved=False, num_attempts=5, num_passed=0),
            ProblemResult(task_id="p3", solved=True, num_attempts=5, num_passed=5),
        ]

        pass_at_k = compute_pass_at_k_for_problems(results, k_values=[1, 5])

        assert 1 in pass_at_k
        assert 5 in pass_at_k
        assert 0 <= pass_at_k[1] <= 1
        assert 0 <= pass_at_k[5] <= 1

    def test_execution_accuracy(self):
        """Test execution accuracy computation."""
        from src.framework.mcts.llm_guided.benchmark.metrics import (
            ProblemResult,
            compute_execution_accuracy,
        )

        results = [
            ProblemResult(
                task_id="p1",
                solved=True,
                num_attempts=10,
                num_passed=8,
                syntax_errors=1,
                runtime_errors=1,
                timeout_errors=0,
            ),
            ProblemResult(
                task_id="p2",
                solved=False,
                num_attempts=10,
                num_passed=0,
                syntax_errors=5,
                runtime_errors=3,
                timeout_errors=2,
            ),
        ]

        exec_acc, syntax_rate, runtime_rate, timeout_rate = compute_execution_accuracy(results)

        # Total attempts = 20, errors = 12
        assert exec_acc == pytest.approx(0.4, abs=0.01)
        assert syntax_rate == pytest.approx(0.3, abs=0.01)
        assert runtime_rate == pytest.approx(0.2, abs=0.01)
        assert timeout_rate == pytest.approx(0.1, abs=0.01)

    def test_aggregate_metrics(self):
        """Test aggregating metrics from results."""
        from src.framework.mcts.llm_guided.benchmark.metrics import (
            ProblemResult,
            aggregate_metrics,
        )

        results = [
            ProblemResult(
                task_id="p1",
                solved=True,
                num_attempts=5,
                num_passed=3,
                total_time_ms=1000,
                first_solution_time_ms=500,
                num_iterations=20,
                tree_size=50,
            ),
            ProblemResult(
                task_id="p2",
                solved=False,
                num_attempts=5,
                num_passed=0,
                total_time_ms=2000,
                num_iterations=30,
                tree_size=80,
            ),
        ]

        metrics = aggregate_metrics(results)

        assert metrics.total_problems == 2
        assert metrics.problems_solved == 1
        assert metrics.total_time_ms == 3000
        assert metrics.avg_iterations == pytest.approx(25.0)

    def test_benchmark_metrics_summary(self):
        """Test metrics summary generation."""
        from src.framework.mcts.llm_guided.benchmark.metrics import BenchmarkMetrics

        metrics = BenchmarkMetrics(
            total_problems=10,
            problems_solved=7,
            pass_at_1=0.7,
            pass_at_k={1: 0.7, 5: 0.9},
            execution_accuracy=0.85,
            total_time_ms=60000,
        )

        summary = metrics.summary()

        assert "70.0%" in summary
        assert "pass@1" in summary


class TestProblemResult:
    """Tests for ProblemResult."""

    def test_problem_result_to_dict(self):
        """Test converting result to dictionary."""
        from src.framework.mcts.llm_guided.benchmark.metrics import ProblemResult

        result = ProblemResult(
            task_id="HumanEval/0",
            solved=True,
            num_attempts=5,
            num_passed=3,
            best_solution="def foo(): return 42",
            total_time_ms=1500.5,
        )

        d = result.to_dict()

        assert d["task_id"] == "HumanEval/0"
        assert d["solved"] is True
        assert d["num_passed"] == 3
        assert d["best_solution"] == "def foo(): return 42"


class TestBenchmarkRunnerConfig:
    """Tests for BenchmarkRunnerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from src.framework.mcts.llm_guided.benchmark.runner import BenchmarkRunnerConfig

        config = BenchmarkRunnerConfig()

        assert config.num_samples_per_problem == 1
        assert config.timeout_per_problem_seconds == 60.0
        assert config.max_concurrent == 1

    def test_config_validation(self):
        """Test configuration validation."""
        from src.framework.mcts.llm_guided.benchmark.runner import BenchmarkRunnerConfig

        config = BenchmarkRunnerConfig(num_samples_per_problem=0)

        with pytest.raises(ValueError, match="num_samples_per_problem must be >= 1"):
            config.validate()

    def test_custom_config(self):
        """Test custom configuration."""
        from src.framework.mcts.llm_guided.benchmark.runner import BenchmarkRunnerConfig

        config = BenchmarkRunnerConfig(
            num_problems=50,
            num_samples_per_problem=5,
            timeout_per_problem_seconds=30.0,
            seed=123,
        )

        config.validate()
        assert config.num_problems == 50
        assert config.seed == 123


class TestBenchmarkReport:
    """Tests for BenchmarkReport."""

    def test_report_save_load(self, tmp_path):
        """Test saving and loading benchmark report."""
        from src.framework.mcts.llm_guided.benchmark.metrics import BenchmarkMetrics
        from src.framework.mcts.llm_guided.benchmark.runner import BenchmarkReport

        metrics = BenchmarkMetrics(
            total_problems=5,
            problems_solved=3,
            pass_at_1=0.6,
        )

        report = BenchmarkReport(
            run_id="test_001",
            timestamp="2025-01-01T00:00:00",
            mcts_config_name="balanced",
            runner_config={"num_samples": 5},
            metrics=metrics,
            python_version="3.11.0",
        )

        filepath = tmp_path / "report.json"
        report.save(filepath)

        loaded = BenchmarkReport.load(filepath)

        assert loaded.run_id == "test_001"
        assert loaded.metrics.total_problems == 5
        assert loaded.metrics.pass_at_1 == 0.6


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    @pytest.fixture
    def mock_mcts_engine(self):
        """Create mock MCTS engine."""
        engine = MagicMock()

        # Create a mock search result
        search_result = MagicMock()
        search_result.solution_found = True
        search_result.best_code = "def solution(): return 42"
        search_result.num_iterations = 10
        search_result.tree_size = 25
        search_result.max_depth = 5

        engine.search = AsyncMock(return_value=search_result)
        engine._config = MagicMock()
        engine._config.name = "test_config"

        return engine

    @pytest.mark.asyncio
    async def test_runner_solves_problems(self, mock_mcts_engine, tmp_path):
        """Test runner solves problems."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import HumanEvalBenchmark
        from src.framework.mcts.llm_guided.benchmark.runner import (
            BenchmarkRunner,
            BenchmarkRunnerConfig,
        )

        config = BenchmarkRunnerConfig(
            num_problems=2,
            num_samples_per_problem=1,
            output_dir=tmp_path,
            verbose=False,
        )

        runner = BenchmarkRunner(mock_mcts_engine, config)
        report = await runner.run_humaneval()

        assert report.metrics.total_problems == 2
        assert report.metrics.problems_solved == 2
        assert mock_mcts_engine.search.call_count == 2


class TestLoadHumanEvalProblems:
    """Tests for load_humaneval_problems function."""

    def test_load_problems_default(self):
        """Test loading problems with defaults."""
        from src.framework.mcts.llm_guided.benchmark.humaneval import load_humaneval_problems

        problems = load_humaneval_problems()

        assert len(problems) > 0
        assert all(p.task_id is not None for p in problems)
