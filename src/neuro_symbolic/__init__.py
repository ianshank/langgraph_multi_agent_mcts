"""
Neuro-Symbolic AI Module for LangGraph Multi-Agent MCTS.

This module provides neuro-symbolic AI capabilities that bridge symbolic reasoning
with neural network-based approaches for enhanced:
- Explainability through proof trees and logical derivations
- Reliability through formal constraint satisfaction
- Sample efficiency through symbolic pruning
- Compositional generalization through logical composition

Components:
- Configuration: Centralized neuro-symbolic configuration management
- State: Unified neuro-symbolic state representation
- Constraints: Symbolic constraint system for MCTS
- Reasoning: Symbolic reasoning agent with logic engine
- Integration: Framework integration utilities

Best Practices 2025:
- Protocol-based interfaces for extensibility
- Dependency injection for testability
- Async-first design for scalability
- Type-safe configurations with dataclasses
"""

from .config import (
    ConstraintConfig,
    LogicEngineConfig,
    NeuroSymbolicConfig,
    ProofConfig,
    SymbolicAgentConfig,
    get_default_config,
    get_high_precision_config,
    get_low_latency_config,
)
from .constraints import (
    Constraint,
    ConstraintResult,
    ConstraintSatisfactionLevel,
    ConstraintSystem,
    ConstraintValidator,
)
from .integration import (
    HybridConfidenceAggregator,
    NeuroSymbolicMCTSConfig,
    NeuroSymbolicMCTSIntegration,
    SymbolicAgentGraphExtension,
    SymbolicAgentNodeConfig,
    create_neuro_symbolic_extension,
    extend_graph_builder,
)
from .reasoning import (
    LogicEngine,
    Predicate,
    Proof,
    ProofStep,
    ProofTree,
    SymbolicReasoner,
    SymbolicReasoningAgent,
)
from .state import (
    Fact,
    NeuroSymbolicState,
    SimpleStateEncoder,
    StateTransition,
    SymbolicFact,
    SymbolicFactType,
)

__all__ = [
    # Configuration
    "NeuroSymbolicConfig",
    "SymbolicAgentConfig",
    "ConstraintConfig",
    "ProofConfig",
    "LogicEngineConfig",
    "get_default_config",
    "get_high_precision_config",
    "get_low_latency_config",
    # State
    "NeuroSymbolicState",
    "SymbolicFact",
    "SymbolicFactType",
    "Fact",
    "StateTransition",
    "SimpleStateEncoder",
    # Constraints
    "Constraint",
    "ConstraintResult",
    "ConstraintSatisfactionLevel",
    "ConstraintSystem",
    "ConstraintValidator",
    # Reasoning
    "LogicEngine",
    "SymbolicReasoner",
    "SymbolicReasoningAgent",
    "Predicate",
    "Proof",
    "ProofStep",
    "ProofTree",
    # Integration
    "NeuroSymbolicMCTSConfig",
    "NeuroSymbolicMCTSIntegration",
    "SymbolicAgentGraphExtension",
    "SymbolicAgentNodeConfig",
    "HybridConfidenceAggregator",
    "create_neuro_symbolic_extension",
    "extend_graph_builder",
]
