"""Pytest configuration for training tests."""

import sys
from unittest.mock import MagicMock, Mock

# Create Pinecone mock with proper structure
pinecone_mock = MagicMock()
pinecone_mock.Pinecone = MagicMock
pinecone_mock.ServerlessSpec = MagicMock
sys.modules['pinecone'] = pinecone_mock

# Mock heavy dependencies that may not be installed
for module in [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.optim.lr_scheduler',
    'torch.utils',
    'torch.utils.data',
    'transformers',
    'peft',
    'datasets',
    'sentence_transformers',
    'faiss',
    'wandb',
    'mlflow',
    'tensorboard',
    'scipy',
    'scipy.stats',
    'sklearn',
    'sklearn.metrics',
    'seaborn',
    'matplotlib',
    'matplotlib.pyplot',
]:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()
