"""
Multi-Agent MCTS Training Pipeline

A comprehensive training pipeline for training multi-agent systems including:
- Hierarchical Reasoning Model (HRM)
- Task Refinement Model (TRM)
- Monte Carlo Tree Search (MCTS)
- RAG retrieval systems
- Neural meta-controllers
"""

__version__ = "1.0.0"

from training.data_pipeline import (
    DABStepLoader,
    PRIMUSProcessor,
    DataOrchestrator,
    TrainingDataset,
)
from training.agent_trainer import (
    HRMTrainer,
    TRMTrainer,
    MCTSTrainer,
    AgentTrainingOrchestrator,
)
from training.rag_builder import (
    VectorIndexBuilder,
    ChunkingStrategy,
    RetrievalOptimizer,
    RAGIndexManager,
)
from training.meta_controller import (
    ExecutionTraceCollector,
    NeuralRouter,
    EnsembleAggregator,
    MetaControllerTrainer,
)
from training.evaluation import (
    DABStepBenchmark,
    MultiAgentEvaluator,
    PerformanceProfiler,
    ProductionValidator,
)
from training.orchestrator import (
    TrainingPipeline,
    PhaseManager,
    ExperimentTracker,
)
from training.continual_learning import (
    FeedbackCollector,
    IncrementalTrainer,
    DriftDetector,
    ABTestFramework,
)
from training.monitoring import (
    TrainingMonitor,
    MetricsDashboard,
    AlertManager,
)
from training.integrate import (
    ModelIntegrator,
    ConfigurationManager,
    HotSwapper,
)

__all__ = [
    # Data Pipeline
    "DABStepLoader",
    "PRIMUSProcessor",
    "DataOrchestrator",
    "TrainingDataset",
    # Agent Training
    "HRMTrainer",
    "TRMTrainer",
    "MCTSTrainer",
    "AgentTrainingOrchestrator",
    # RAG Builder
    "VectorIndexBuilder",
    "ChunkingStrategy",
    "RetrievalOptimizer",
    "RAGIndexManager",
    # Meta-Controller
    "ExecutionTraceCollector",
    "NeuralRouter",
    "EnsembleAggregator",
    "MetaControllerTrainer",
    # Evaluation
    "DABStepBenchmark",
    "MultiAgentEvaluator",
    "PerformanceProfiler",
    "ProductionValidator",
    # Orchestration
    "TrainingPipeline",
    "PhaseManager",
    "ExperimentTracker",
    # Continual Learning
    "FeedbackCollector",
    "IncrementalTrainer",
    "DriftDetector",
    "ABTestFramework",
    # Monitoring
    "TrainingMonitor",
    "MetricsDashboard",
    "AlertManager",
    # Integration
    "ModelIntegrator",
    "ConfigurationManager",
    "HotSwapper",
]
