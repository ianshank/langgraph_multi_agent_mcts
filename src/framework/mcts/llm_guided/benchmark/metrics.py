"""
Benchmark Metrics for Code Generation.

Provides:
- pass@k computation
- Execution accuracy
- Per-problem result tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


@dataclass
class ProblemResult:
    """Result for a single problem."""

    task_id: str
    """Problem identifier."""

    solved: bool
    """Whether the problem was solved."""

    num_attempts: int
    """Number of solution attempts."""

    num_passed: int
    """Number of attempts that passed all tests."""

    best_solution: str | None = None
    """Best solution code (if solved)."""

    # Timing
    total_time_ms: float = 0.0
    """Total time spent on this problem."""

    first_solution_time_ms: float | None = None
    """Time to first successful solution."""

    # MCTS statistics
    num_iterations: int = 0
    """Total MCTS iterations."""

    tree_size: int = 0
    """Final tree size."""

    max_depth_reached: int = 0
    """Maximum tree depth reached."""

    # Code quality
    solution_length: int = 0
    """Length of solution in characters."""

    num_test_cases: int = 0
    """Number of test cases."""

    # Errors
    syntax_errors: int = 0
    """Number of syntax errors encountered."""

    runtime_errors: int = 0
    """Number of runtime errors encountered."""

    timeout_errors: int = 0
    """Number of timeouts."""

    error_messages: list[str] = field(default_factory=list)
    """Collected error messages."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "solved": self.solved,
            "num_attempts": self.num_attempts,
            "num_passed": self.num_passed,
            "best_solution": self.best_solution,
            "total_time_ms": self.total_time_ms,
            "first_solution_time_ms": self.first_solution_time_ms,
            "num_iterations": self.num_iterations,
            "tree_size": self.tree_size,
            "max_depth_reached": self.max_depth_reached,
            "solution_length": self.solution_length,
            "num_test_cases": self.num_test_cases,
            "syntax_errors": self.syntax_errors,
            "runtime_errors": self.runtime_errors,
            "timeout_errors": self.timeout_errors,
            "error_messages": self.error_messages,
        }


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a benchmark run."""

    # Overall metrics
    total_problems: int = 0
    """Total number of problems."""

    problems_solved: int = 0
    """Number of problems solved."""

    problems_attempted: int = 0
    """Number of problems attempted."""

    # pass@k metrics
    pass_at_1: float = 0.0
    """Pass rate with 1 attempt per problem."""

    pass_at_k: dict[int, float] = field(default_factory=dict)
    """Pass rate with k attempts per problem."""

    # Execution metrics
    execution_accuracy: float = 0.0
    """Fraction of generated code that executes without errors."""

    syntax_error_rate: float = 0.0
    """Fraction of attempts with syntax errors."""

    runtime_error_rate: float = 0.0
    """Fraction of attempts with runtime errors."""

    timeout_rate: float = 0.0
    """Fraction of attempts that timed out."""

    # Timing
    avg_time_per_problem_ms: float = 0.0
    """Average time per problem."""

    total_time_ms: float = 0.0
    """Total benchmark time."""

    avg_time_to_solution_ms: float = 0.0
    """Average time to first solution (for solved problems)."""

    # MCTS statistics
    avg_iterations: float = 0.0
    """Average MCTS iterations per problem."""

    avg_tree_size: float = 0.0
    """Average tree size."""

    avg_max_depth: float = 0.0
    """Average maximum depth reached."""

    # Token/cost tracking
    total_tokens: int = 0
    """Total LLM tokens used."""

    total_cost_usd: float = 0.0
    """Estimated total cost in USD."""

    # By difficulty breakdown
    metrics_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    """Metrics broken down by difficulty level."""

    # Per-problem results
    problem_results: list[ProblemResult] = field(default_factory=list)
    """Individual problem results."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_problems": self.total_problems,
            "problems_solved": self.problems_solved,
            "problems_attempted": self.problems_attempted,
            "pass_at_1": self.pass_at_1,
            "pass_at_k": self.pass_at_k,
            "execution_accuracy": self.execution_accuracy,
            "syntax_error_rate": self.syntax_error_rate,
            "runtime_error_rate": self.runtime_error_rate,
            "timeout_rate": self.timeout_rate,
            "avg_time_per_problem_ms": self.avg_time_per_problem_ms,
            "total_time_ms": self.total_time_ms,
            "avg_time_to_solution_ms": self.avg_time_to_solution_ms,
            "avg_iterations": self.avg_iterations,
            "avg_tree_size": self.avg_tree_size,
            "avg_max_depth": self.avg_max_depth,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "metrics_by_difficulty": self.metrics_by_difficulty,
            "problem_results": [r.to_dict() for r in self.problem_results],
        }

    def summary(self) -> str:
        """Get a text summary of the metrics."""
        lines = [
            "=" * 60,
            "BENCHMARK RESULTS",
            "=" * 60,
            f"Problems: {self.problems_solved}/{self.total_problems} solved "
            f"({100 * self.problems_solved / max(1, self.total_problems):.1f}%)",
            "",
            "Pass@k Metrics:",
            f"  pass@1: {self.pass_at_1:.2%}",
        ]

        for k, rate in sorted(self.pass_at_k.items()):
            if k != 1:
                lines.append(f"  pass@{k}: {rate:.2%}")

        lines.extend(
            [
                "",
                "Execution Metrics:",
                f"  Execution accuracy: {self.execution_accuracy:.2%}",
                f"  Syntax errors: {self.syntax_error_rate:.2%}",
                f"  Runtime errors: {self.runtime_error_rate:.2%}",
                f"  Timeouts: {self.timeout_rate:.2%}",
                "",
                "Performance:",
                f"  Total time: {self.total_time_ms / 1000:.1f}s",
                f"  Avg time/problem: {self.avg_time_per_problem_ms / 1000:.2f}s",
                f"  Avg iterations: {self.avg_iterations:.1f}",
                f"  Avg tree size: {self.avg_tree_size:.1f}",
                "",
            ]
        )

        if self.metrics_by_difficulty:
            lines.append("By Difficulty:")
            for diff, metrics in self.metrics_by_difficulty.items():
                rate = metrics.get("pass_rate", 0)
                count = int(metrics.get("count", 0))
                lines.append(f"  {diff}: {rate:.1%} ({count} problems)")

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_pass_at_k(
    n: int,
    c: int,
    k: int,
) -> float:
    """
    Compute pass@k metric.

    This uses the unbiased estimator from the Codex paper:
    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of attempts allowed

    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0

    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_pass_at_k_for_problems(
    results: list[ProblemResult],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """
    Compute pass@k for a set of problem results.

    Args:
        results: List of problem results
        k_values: Values of k to compute (default: [1, 5, 10])

    Returns:
        Dictionary mapping k to pass@k value
    """
    if k_values is None:
        k_values = [1, 5, 10]
    pass_at_k = {}

    for k in k_values:
        # For each problem, compute pass@k and average
        problem_pass_rates = []

        for result in results:
            n = result.num_attempts
            c = result.num_passed

            if n == 0:
                # No attempts, treat as 0
                problem_pass_rates.append(0.0)
            elif n < k:
                # Not enough samples, use actual pass rate
                problem_pass_rates.append(1.0 if c > 0 else 0.0)
            else:
                problem_pass_rates.append(compute_pass_at_k(n, c, k))

        pass_at_k[k] = np.mean(problem_pass_rates) if problem_pass_rates else 0.0

    return pass_at_k


def compute_execution_accuracy(results: list[ProblemResult]) -> tuple[float, float, float, float]:
    """
    Compute execution accuracy and error rates.

    Args:
        results: List of problem results

    Returns:
        Tuple of (execution_accuracy, syntax_error_rate, runtime_error_rate, timeout_rate)
    """
    total_attempts = sum(r.num_attempts for r in results)

    if total_attempts == 0:
        return 0.0, 0.0, 0.0, 0.0

    total_syntax_errors = sum(r.syntax_errors for r in results)
    total_runtime_errors = sum(r.runtime_errors for r in results)
    total_timeouts = sum(r.timeout_errors for r in results)

    total_errors = total_syntax_errors + total_runtime_errors + total_timeouts
    execution_accuracy = 1.0 - (total_errors / total_attempts)

    syntax_error_rate = total_syntax_errors / total_attempts
    runtime_error_rate = total_runtime_errors / total_attempts
    timeout_rate = total_timeouts / total_attempts

    return execution_accuracy, syntax_error_rate, runtime_error_rate, timeout_rate


def aggregate_metrics(
    results: list[ProblemResult],
    k_values: list[int] | None = None,
) -> BenchmarkMetrics:
    """
    Aggregate metrics from problem results.

    Args:
        results: List of problem results
        k_values: Values of k for pass@k computation (default: [1, 5, 10])

    Returns:
        Aggregated BenchmarkMetrics
    """
    if k_values is None:
        k_values = [1, 5, 10]
    if not results:
        return BenchmarkMetrics()

    total_problems = len(results)
    problems_solved = sum(1 for r in results if r.solved)
    problems_attempted = sum(1 for r in results if r.num_attempts > 0)

    # pass@k
    pass_at_k = compute_pass_at_k_for_problems(results, k_values)
    pass_at_1 = pass_at_k.get(1, 0.0)

    # Execution metrics
    exec_acc, syntax_err, runtime_err, timeout_r = compute_execution_accuracy(results)

    # Timing
    total_time = sum(r.total_time_ms for r in results)
    avg_time = total_time / total_problems if total_problems > 0 else 0.0

    solved_times = [r.first_solution_time_ms for r in results if r.first_solution_time_ms is not None]
    avg_time_to_solution = np.mean(solved_times) if solved_times else 0.0

    # MCTS stats
    avg_iterations = np.mean([r.num_iterations for r in results]) if results else 0.0
    avg_tree_size = np.mean([r.tree_size for r in results]) if results else 0.0
    avg_max_depth = np.mean([r.max_depth_reached for r in results]) if results else 0.0

    # Note: By difficulty metrics would require problem difficulty info
    # which is not available from ProblemResult alone

    return BenchmarkMetrics(
        total_problems=total_problems,
        problems_solved=problems_solved,
        problems_attempted=problems_attempted,
        pass_at_1=pass_at_1,
        pass_at_k=pass_at_k,
        execution_accuracy=exec_acc,
        syntax_error_rate=syntax_err,
        runtime_error_rate=runtime_err,
        timeout_rate=timeout_r,
        avg_time_per_problem_ms=avg_time,
        total_time_ms=total_time,
        avg_time_to_solution_ms=avg_time_to_solution,
        avg_iterations=avg_iterations,
        avg_tree_size=avg_tree_size,
        avg_max_depth=avg_max_depth,
        problem_results=results,
    )
