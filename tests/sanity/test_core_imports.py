"""
CI/CD Sanity Tests - Core Imports
==================================

Fast tests to verify core project modules can be imported without errors.
These tests should always pass and run in <10 seconds.
"""

import pytest


@pytest.mark.sanity
class TestCoreAgentImports:
    """Verify agent modules import correctly."""

    def test_import_meta_controller_base(self):
        """Test meta controller base imports."""
        from src.agents.meta_controller.base import (
            MetaControllerFeatures,
            MetaControllerPrediction,
        )
        assert MetaControllerFeatures is not None
        assert MetaControllerPrediction is not None

    def test_import_meta_controller_utils(self):
        """Test meta controller utils imports."""
        from src.agents.meta_controller.utils import normalize_features
        assert normalize_features is not None


@pytest.mark.sanity
class TestStorageImports:
    """Verify storage modules import correctly."""

    def test_import_pinecone_store(self):
        """Test Pinecone vector store imports."""
        from src.storage.pinecone_store import (
            PINECONE_AVAILABLE,
            PineconeVectorStore,
        )
        assert PineconeVectorStore is not None
        # PINECONE_AVAILABLE should be a boolean
        assert isinstance(PINECONE_AVAILABLE, bool)

    def test_import_s3_client(self):
        """Test S3 client imports."""
        from src.storage.s3_client import S3Config, S3StorageClient
        assert S3Config is not None
        assert S3StorageClient is not None


@pytest.mark.sanity
class TestObservabilityImports:
    """Verify observability modules import correctly."""

    def test_import_logging(self):
        """Test logging module imports."""
        from src.observability.logging import get_logger
        assert get_logger is not None

    def test_import_tracing(self):
        """Test tracing module imports."""
        from src.observability.tracing import TracingManager
        assert TracingManager is not None


@pytest.mark.sanity
class TestTrainingImports:
    """Verify training modules import correctly."""

    def test_import_continuous_play_config(self):
        """Test continuous play config imports."""
        from src.training.continuous_play_config import (
            ContinuousPlayConfig,
            SessionConfig,
        )
        assert ContinuousPlayConfig is not None
        assert SessionConfig is not None

    def test_import_metrics_aggregator(self):
        """Test metrics aggregator imports."""
        from src.training.metrics_aggregator import MetricsAggregator
        assert MetricsAggregator is not None
