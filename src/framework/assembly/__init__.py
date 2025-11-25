"""
Assembly Theory integration for LangGraph Multi-Agent MCTS Framework.

This module implements Assembly Theory concepts for quantifying complexity
and guiding agent decisions based on construction pathways and reusability.

Key Components:
- AssemblyIndexCalculator: Core complexity measurement
- ConceptExtractor: Extract concepts and build dependency graphs
- AssemblyGraph: Specialized graph structure for assembly pathways
- SubstructureLibrary: Pattern reuse tracking
- AssemblyFeatureExtractor: Feature extraction for ML models

Based on:
Assembly Theory by Cronin, Walker, et al.
"""

from .calculator import AssemblyIndexCalculator
from .concept_extractor import Concept, ConceptExtractor
from .config import AssemblyConfig
from .features import AssemblyFeatureExtractor, AssemblyFeatures
from .graph import AssemblyGraph, AssemblyNode
from .substructure_library import Match, SubstructureLibrary

__all__ = [
    "AssemblyIndexCalculator",
    "ConceptExtractor",
    "Concept",
    "AssemblyGraph",
    "AssemblyNode",
    "SubstructureLibrary",
    "Match",
    "AssemblyFeatureExtractor",
    "AssemblyFeatures",
    "AssemblyConfig",
]

__version__ = "0.1.0"
