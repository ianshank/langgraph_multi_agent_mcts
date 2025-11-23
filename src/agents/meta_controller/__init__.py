"""
Neural Meta-Controller package for Multi-Agent MCTS Framework.

This package provides the base infrastructure for neural network-based
meta-controllers that dynamically select which agent to route queries to.

Includes Assembly Theory integration for hybrid neural+rule-based routing.
"""

from src.agents.meta_controller.base import (
    AbstractMetaController,
    MetaControllerFeatures,
    MetaControllerPrediction,
)
from src.agents.meta_controller.rnn_controller import (
    RNNMetaController,
    RNNMetaControllerModel,
)
from src.agents.meta_controller.utils import (
    features_to_tensor,
    features_to_text,
    normalize_features,
    one_hot_encode_agent,
)

# Import Assembly Theory components
from src.agents.meta_controller.assembly_router import (
    AssemblyRouter,
    RoutingDecision,
)
from src.agents.meta_controller.hybrid_controller import (
    HybridMetaController,
    HybridPrediction,
)

# Import BERT controller (may not be available if transformers/peft not installed)
try:
    from src.agents.meta_controller.bert_controller import BERTMetaController  # noqa: F401

    _bert_available = True
except ImportError:
    _bert_available = False

__all__ = [
    "AbstractMetaController",
    "MetaControllerFeatures",
    "MetaControllerPrediction",
    "normalize_features",
    "one_hot_encode_agent",
    "features_to_tensor",
    "features_to_text",
    "RNNMetaController",
    "RNNMetaControllerModel",
    "AssemblyRouter",
    "RoutingDecision",
    "HybridMetaController",
    "HybridPrediction",
]

if _bert_available:
    __all__.append("BERTMetaController")
