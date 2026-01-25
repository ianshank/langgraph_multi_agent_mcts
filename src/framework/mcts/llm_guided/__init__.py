"""
LLM-Guided MCTS Module - Phase 2 Implementation.

This module implements the LLM-guided Monte Carlo Tree Search system
for code generation with integrated training data collection for
neural network distillation.

Phase 1 Components:
- LLMGuidedMCTSConfig: Configuration for the LLM-guided MCTS
- LLMGuidedMCTSNode: Enhanced MCTS node with data collection fields
- TrainingDataCollector: Collects (state, action, value) tuples
- GeneratorAgent: LLM agent for code generation
- ReflectorAgent: LLM agent for code evaluation
- CodeExecutor: Safe code execution sandbox
- LLMGuidedMCTSEngine: Main orchestrator
- UnifiedSearchOrchestrator: Integration with HRM, TRM, Meta-Controller

Phase 2 Components:
- training/: PyTorch Dataset, PolicyNetwork, ValueNetwork, DistillationTrainer
- benchmark/: HumanEval benchmark loader and runner with pass@k metrics
- rag/: RAG context provider and enhanced prompts
"""

from .agents import (
    GeneratorAgent,
    GeneratorOutput,
    ReflectorAgent,
    ReflectorOutput,
)
from .benchmark import (  # noqa: F401 - re-exports
    BenchmarkMetrics,
    BenchmarkReport,
    BenchmarkRunner,
    BenchmarkRunnerConfig,
    HumanEvalBenchmark,
    HumanEvalProblem,
    ProblemResult,
    compute_execution_accuracy,
    compute_pass_at_k,
    load_humaneval_problems,
    run_benchmark,
)
from .config import (
    GeneratorConfig,
    LLMGuidedMCTSConfig,
    LLMGuidedMCTSPreset,
    ReflectorConfig,
    create_llm_mcts_preset,
    get_preset_config,
)
from .data_collector import (
    EpisodeMetadata,
    TrainingDataCollector,
    TrainingExample,
)
from .engine import LLMGuidedMCTSEngine, MCTSSearchResult
from .executor import CodeExecutionResult, CodeExecutor
from .integration import (
    AgentType,
    HRMAdapter,
    IntegrationConfig,
    MetaControllerAdapter,
    RefinementResult,
    RoutingDecision,
    SubProblemDecomposition,
    TRMAdapter,
    UnifiedSearchOrchestrator,
    UnifiedSearchResult,
    create_unified_orchestrator,
)
from .node import LLMGuidedMCTSNode, NodeState
from .rag import (  # noqa: F401 - re-exports
    RAGContext,
    RAGContextProvider,
    RAGContextProviderConfig,
    RAGPromptBuilder,
    build_generator_prompt_with_rag,
    build_reflector_prompt_with_rag,
    create_rag_provider,
)

# Phase 2: Training module (optional, requires PyTorch)
_TRAINING_AVAILABLE = False
try:
    from .training import (  # noqa: F401 - re-exports
        DistillationTrainer,
        DistillationTrainerConfig,
        EvaluationMetrics,
        MCTSDataset,
        MCTSDatasetConfig,
        PolicyNetwork,
        PolicyNetworkConfig,
        TrainingBatch,
        TrainingCheckpoint,
        TrainingMetrics,
        ValueNetwork,
        ValueNetworkConfig,
        create_dataloaders,
        create_policy_network,
        create_trainer,
        create_value_network,
    )

    _TRAINING_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # Config
    "LLMGuidedMCTSConfig",
    "LLMGuidedMCTSPreset",
    "GeneratorConfig",
    "ReflectorConfig",
    "create_llm_mcts_preset",
    "get_preset_config",
    # Node
    "LLMGuidedMCTSNode",
    "NodeState",
    # Data Collection
    "TrainingDataCollector",
    "TrainingExample",
    "EpisodeMetadata",
    # Executor
    "CodeExecutor",
    "CodeExecutionResult",
    # Agents
    "GeneratorAgent",
    "GeneratorOutput",
    "ReflectorAgent",
    "ReflectorOutput",
    # Engine
    "LLMGuidedMCTSEngine",
    "MCTSSearchResult",
    # Integration
    "AgentType",
    "IntegrationConfig",
    "HRMAdapter",
    "TRMAdapter",
    "MetaControllerAdapter",
    "SubProblemDecomposition",
    "RefinementResult",
    "RoutingDecision",
    "UnifiedSearchOrchestrator",
    "UnifiedSearchResult",
    "create_unified_orchestrator",
    # Phase 2: Benchmark
    "HumanEvalProblem",
    "HumanEvalBenchmark",
    "load_humaneval_problems",
    "BenchmarkMetrics",
    "ProblemResult",
    "compute_pass_at_k",
    "compute_execution_accuracy",
    "BenchmarkRunner",
    "BenchmarkRunnerConfig",
    "BenchmarkReport",
    "run_benchmark",
    # Phase 2: RAG
    "RAGContextProvider",
    "RAGContextProviderConfig",
    "RAGContext",
    "create_rag_provider",
    "RAGPromptBuilder",
    "build_generator_prompt_with_rag",
    "build_reflector_prompt_with_rag",
]

# Add training exports if available
if _TRAINING_AVAILABLE:
    __all__.extend(
        [
            "MCTSDataset",
            "MCTSDatasetConfig",
            "TrainingBatch",
            "create_dataloaders",
            "PolicyNetwork",
            "PolicyNetworkConfig",
            "ValueNetwork",
            "ValueNetworkConfig",
            "create_policy_network",
            "create_value_network",
            "DistillationTrainer",
            "DistillationTrainerConfig",
            "TrainingCheckpoint",
            "create_trainer",
            "TrainingMetrics",
            "EvaluationMetrics",
        ]
    )
