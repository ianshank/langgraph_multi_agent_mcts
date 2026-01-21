"""
CI/CD Sanity Tests - External Dependencies
===========================================

Fast tests to verify critical external dependencies are installed correctly.
These tests should always pass and run in <5 seconds.
"""

import importlib.metadata
import sys

import pytest


@pytest.mark.sanity
class TestPineconeInstallation:
    """Verify Pinecone is installed correctly."""

    def test_pinecone_package_installed(self):
        """Test that pinecone package is installed (not pinecone-client)."""
        try:
            version = importlib.metadata.version("pinecone")
            assert version is not None
        except importlib.metadata.PackageNotFoundError:
            pytest.fail("pinecone package not installed")

    def test_pinecone_imports_correctly(self):
        """Test that Pinecone imports with the v3+ API style."""
        from pinecone import Pinecone
        assert Pinecone is not None

    def test_pinecone_client_not_conflicting(self):
        """Test that pinecone-client stub is not causing conflicts."""
        # Should not raise "The official Pinecone python client..." error
        import pinecone
        # If we get here, no conflict error was raised
        assert pinecone is not None


@pytest.mark.sanity
class TestOpenTelemetryInstallation:
    """Verify OpenTelemetry packages are installed correctly."""

    def test_opentelemetry_api_installed(self):
        """Test OpenTelemetry API is installed."""
        from opentelemetry import trace
        assert trace is not None

    def test_opentelemetry_sdk_installed(self):
        """Test OpenTelemetry SDK is installed."""
        from opentelemetry.sdk.trace import TracerProvider
        assert TracerProvider is not None

    def test_opentelemetry_exporter_installed(self):
        """Test OpenTelemetry OTLP exporter is installed."""
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        assert OTLPSpanExporter is not None

    def test_opentelemetry_httpx_instrumentation_installed(self):
        """Test OpenTelemetry HTTPX instrumentation is installed."""
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        assert HTTPXClientInstrumentor is not None


@pytest.mark.sanity
class TestPyTorchInstallation:
    """Verify PyTorch is installed correctly."""

    def test_torch_installed(self):
        """Test that torch is installed."""
        import torch
        assert torch is not None

    def test_torch_has_safe_globals(self):
        """Test that torch has add_safe_globals (PyTorch 2.6+)."""
        import torch
        assert hasattr(torch.serialization, "add_safe_globals"), \
            "torch.serialization.add_safe_globals not found - PyTorch 2.6+ required"


@pytest.mark.sanity
class TestLangChainInstallation:
    """Verify LangChain/LangGraph packages are installed correctly (optional)."""

    def test_langchain_installed(self):
        """Test that langchain is installed (optional)."""
        try:
            import langchain
            assert langchain is not None
        except ImportError:
            pytest.skip("langchain not installed (optional dependency)")

    def test_langgraph_installed(self):
        """Test that langgraph is installed (optional)."""
        try:
            import langgraph
            assert langgraph is not None
        except ImportError:
            pytest.skip("langgraph not installed (optional dependency)")


@pytest.mark.sanity
class TestCriticalDependencies:
    """Verify other critical dependencies are installed."""

    def test_pydantic_installed(self):
        """Test that pydantic is installed."""
        import pydantic
        assert pydantic is not None

    def test_pydantic_settings_installed(self):
        """Test that pydantic-settings is installed."""
        from pydantic_settings import BaseSettings
        assert BaseSettings is not None

    def test_httpx_installed(self):
        """Test that httpx is installed."""
        import httpx
        assert httpx is not None

    def test_aioboto3_installed(self):
        """Test that aioboto3 is installed."""
        import aioboto3
        assert aioboto3 is not None

    def test_tenacity_installed(self):
        """Test that tenacity is installed."""
        import tenacity
        assert tenacity is not None
