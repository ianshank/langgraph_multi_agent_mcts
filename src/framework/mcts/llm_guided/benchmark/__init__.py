"""
Benchmark Module for LLM-Guided MCTS Code Generation.

Provides:
- HumanEvalBenchmark: HumanEval problem set loader
- BenchmarkRunner: Run MCTS on benchmark problems
- BenchmarkMetrics: pass@k and other metrics
"""

from .humaneval import (
    HumanEvalProblem,
    HumanEvalBenchmark,
    load_humaneval_problems,
)
from .metrics import (
    BenchmarkMetrics,
    ProblemResult,
    compute_pass_at_k,
    compute_execution_accuracy,
)
from .runner import (
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    BenchmarkReport,
    run_benchmark,
)

__all__ = [
    # HumanEval
    "HumanEvalProblem",
    "HumanEvalBenchmark",
    "load_humaneval_problems",
    # Metrics
    "BenchmarkMetrics",
    "ProblemResult",
    "compute_pass_at_k",
    "compute_execution_accuracy",
    # Runner
    "BenchmarkRunner",
    "BenchmarkRunnerConfig",
    "BenchmarkReport",
    "run_benchmark",
]
