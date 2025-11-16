"""
Dataset Integration Module for Multi-Agent MCTS Training.

This module provides utilities for loading, preprocessing, and managing
open-source datasets for training HRM/TRM agents and neural meta-controllers.

Supported Datasets:
- DABStep: Multi-step reasoning tasks (CC-BY-4.0)
- PRIMUS-Seed: Cybersecurity domain knowledge (ODC-BY)
- PRIMUS-Instruct: Instruction fine-tuning data (ODC-BY)
"""

from .dataset_loader import DatasetLoader, DABStepLoader, PRIMUSLoader
from .preprocessing import TextPreprocessor, TokenizerWrapper
from .tactical_augmentation import TacticalAugmenter
from .train_test_split import DataSplitter, StratifiedSplitter

__all__ = [
    "DatasetLoader",
    "DABStepLoader",
    "PRIMUSLoader",
    "TextPreprocessor",
    "TokenizerWrapper",
    "TacticalAugmenter",
    "DataSplitter",
    "StratifiedSplitter",
]

__version__ = "1.0.0"
