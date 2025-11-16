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

# Optional imports - allow package to be importable even without all dependencies
__all__ = []

try:
    from training.data_pipeline import (
        DABStepLoader,
        DataOrchestrator,
        PRIMUSProcessor,
        TrainingDataset,
    )

    __all__.extend(
        [
            "DABStepLoader",
            "PRIMUSProcessor",
            "DataOrchestrator",
            "TrainingDataset",
        ]
    )
except ImportError:
    pass

try:
    from training.agent_trainer import (
        AgentTrainingOrchestrator,
        HRMTrainer,
        MCTSTrainer,
        TRMTrainer,
    )

    __all__.extend(
        [
            "HRMTrainer",
            "TRMTrainer",
            "MCTSTrainer",
            "AgentTrainingOrchestrator",
        ]
    )
except ImportError:
    pass

try:
    from training.rag_builder import (
        ChunkingStrategy,
        RAGIndexManager,
        RetrievalOptimizer,
        VectorIndexBuilder,
    )

    __all__.extend(
        [
            "VectorIndexBuilder",
            "ChunkingStrategy",
            "RetrievalOptimizer",
            "RAGIndexManager",
        ]
    )
except ImportError:
    pass

try:
    from training.meta_controller import (
        EnsembleAggregator,
        ExecutionTraceCollector,
        MetaControllerTrainer,
        NeuralRouter,
    )

    __all__.extend(
        [
            "ExecutionTraceCollector",
            "NeuralRouter",
            "EnsembleAggregator",
            "MetaControllerTrainer",
        ]
    )
except ImportError:
    pass

try:
    from training.evaluation import (
        DABStepBenchmark,
        MultiAgentEvaluator,
        PerformanceProfiler,
        ProductionValidator,
    )

    __all__.extend(
        [
            "DABStepBenchmark",
            "MultiAgentEvaluator",
            "PerformanceProfiler",
            "ProductionValidator",
        ]
    )
except ImportError:
    pass

try:
    from training.orchestrator import (
        ExperimentTracker,
        PhaseManager,
        TrainingPipeline,
    )

    __all__.extend(
        [
            "TrainingPipeline",
            "PhaseManager",
            "ExperimentTracker",
        ]
    )
except ImportError:
    pass

try:
    from training.continual_learning import (
        ABTestFramework,
        DriftDetector,
        FeedbackCollector,
        IncrementalTrainer,
    )

    __all__.extend(
        [
            "FeedbackCollector",
            "IncrementalTrainer",
            "DriftDetector",
            "ABTestFramework",
        ]
    )
except ImportError:
    pass

try:
    from training.monitoring import (
        AlertManager,
        MetricsDashboard,
        TrainingMonitor,
    )

    __all__.extend(
        [
            "TrainingMonitor",
            "MetricsDashboard",
            "AlertManager",
        ]
    )
except ImportError:
    pass

try:
    from training.integrate import (
        ConfigurationManager,
        HotSwapper,
        ModelIntegrator,
    )

    __all__.extend(
        [
            "ModelIntegrator",
            "ConfigurationManager",
            "HotSwapper",
        ]
    )
except ImportError:
    pass
