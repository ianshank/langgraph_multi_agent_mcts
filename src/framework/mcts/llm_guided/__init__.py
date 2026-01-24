"""
LLM-Guided MCTS Module - Phase 1 Prototype.

This module implements the LLM-guided Monte Carlo Tree Search system
for code generation with training data collection for future neural
network distillation.

Components:
- LLMGuidedMCTSConfig: Configuration for the LLM-guided MCTS
- LLMGuidedMCTSNode: Enhanced MCTS node with data collection fields
- TrainingDataCollector: Collects (state, action, value) tuples
- GeneratorAgent: LLM agent for code generation
- ReflectorAgent: LLM agent for code evaluation
- CodeExecutor: Safe code execution sandbox
- LLMGuidedMCTSEngine: Main orchestrator
- UnifiedSearchOrchestrator: Integration with HRM, TRM, Meta-Controller
"""

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
from .executor import CodeExecutionResult, CodeExecutor
from .node import LLMGuidedMCTSNode, NodeState
from .agents import GeneratorAgent, GeneratorOutput, ReflectorAgent, ReflectorOutput
from .engine import LLMGuidedMCTSEngine, MCTSSearchResult
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
]
