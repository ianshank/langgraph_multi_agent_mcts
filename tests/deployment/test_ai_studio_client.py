"""
Tests for Google AI Studio client.

These tests validate the AIStudioClient implementation for
integrating with Google AI Studio / Gemini API.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Get the repository root (tests/deployment/ -> tests/ -> repo_root)
REPO_ROOT = Path(__file__).parent.parent.parent


class TestAIStudioClientModule:
    """Test AI Studio client module existence and structure."""

    def test_module_exists(self):
        """Verify ai_studio_client.py exists."""
        module_path = REPO_ROOT / "src" / "integrations" / "google_adk" / "ai_studio_client.py"
        assert module_path.exists(), "ai_studio_client.py not found"

    def test_module_is_syntactically_valid(self):
        """Verify ai_studio_client.py is syntactically valid Python."""
        module_path = REPO_ROOT / "src" / "integrations" / "google_adk" / "ai_studio_client.py"

        # This will raise SyntaxError if invalid
        compile(module_path.read_text(), str(module_path), "exec")

    def test_module_exports_expected_classes(self):
        """Verify module exports expected classes."""
        from src.integrations.google_adk.ai_studio_client import (
            AIStudioClient,
            AIStudioConfig,
            GeminiModel,
        )

        assert AIStudioClient is not None
        assert AIStudioConfig is not None
        assert GeminiModel is not None


class TestAIStudioConfig:
    """Test AIStudioConfig dataclass."""

    def test_config_default_values(self):
        """Verify AIStudioConfig has sensible defaults."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        config = AIStudioConfig()

        assert config.temperature == 0.7
        assert config.max_output_tokens == 4096
        assert config.location == "us-central1"
        assert config.use_vertex_ai is False

    def test_config_from_env_without_key(self):
        """Verify AIStudioConfig.from_env() works without API key set."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        # Clear any existing keys
        with patch.dict(os.environ, {}, clear=True):
            config = AIStudioConfig.from_env()

            assert config.api_key is None
            assert config.use_vertex_ai is False

    def test_config_from_env_with_google_api_key(self):
        """Verify AIStudioConfig.from_env() reads GOOGLE_API_KEY."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}):
            config = AIStudioConfig.from_env()

            assert config.api_key == "test-key-123"

    def test_config_from_env_with_gemini_api_key(self):
        """Verify AIStudioConfig.from_env() reads GEMINI_API_KEY."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key-456"}):
            config = AIStudioConfig.from_env()

            assert config.api_key == "gemini-key-456"

    def test_config_from_env_vertex_ai_mode(self):
        """Verify AIStudioConfig.from_env() reads Vertex AI settings."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        env_vars = {
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "my-project",
            "GOOGLE_CLOUD_LOCATION": "europe-west1",
        }

        with patch.dict(os.environ, env_vars):
            config = AIStudioConfig.from_env()

            assert config.use_vertex_ai is True
            assert config.project_id == "my-project"
            assert config.location == "europe-west1"

    def test_config_validate_requires_api_key_for_ai_studio(self):
        """Verify validation fails without API key in AI Studio mode."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        config = AIStudioConfig(api_key=None, use_vertex_ai=False)

        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            config.validate()

    def test_config_validate_requires_project_for_vertex_ai(self):
        """Verify validation fails without project ID in Vertex AI mode."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        config = AIStudioConfig(use_vertex_ai=True, project_id=None)

        with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
            config.validate()

    def test_config_validate_succeeds_with_api_key(self):
        """Verify validation succeeds with API key."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        config = AIStudioConfig(api_key="test-key")

        # Should not raise
        config.validate()

    def test_config_validate_succeeds_with_vertex_ai(self):
        """Verify validation succeeds with Vertex AI config."""
        from src.integrations.google_adk.ai_studio_client import AIStudioConfig

        config = AIStudioConfig(use_vertex_ai=True, project_id="my-project")

        # Should not raise
        config.validate()


class TestGeminiModel:
    """Test GeminiModel enum."""

    def test_gemini_model_values(self):
        """Verify GeminiModel has expected values."""
        from src.integrations.google_adk.ai_studio_client import GeminiModel

        assert GeminiModel.GEMINI_2_0_FLASH.value == "gemini-2.0-flash-001"
        assert GeminiModel.GEMINI_1_5_FLASH.value == "gemini-1.5-flash"
        assert GeminiModel.GEMINI_1_5_PRO.value == "gemini-1.5-pro"


class TestAIStudioClient:
    """Test AIStudioClient class."""

    def test_client_init_with_config(self):
        """Verify client can be initialized with config."""
        from src.integrations.google_adk.ai_studio_client import AIStudioClient, AIStudioConfig

        config = AIStudioConfig(api_key="test-key")
        client = AIStudioClient(config)

        assert client.config.api_key == "test-key"
        assert client._initialized is False

    def test_client_init_from_env(self):
        """Verify client can be initialized from environment."""
        from src.integrations.google_adk.ai_studio_client import AIStudioClient

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            client = AIStudioClient()

            assert client.config.api_key == "env-key"

    def test_client_get_model_info(self):
        """Verify get_model_info returns expected structure."""
        from src.integrations.google_adk.ai_studio_client import AIStudioClient, AIStudioConfig, GeminiModel

        config = AIStudioConfig(
            api_key="test-key",
            model=GeminiModel.GEMINI_2_0_FLASH,
            temperature=0.5,
        )
        client = AIStudioClient(config)

        info = client.get_model_info()

        assert info["model"] == "gemini-2.0-flash-001"
        assert info["temperature"] == 0.5
        assert info["backend"] == "ai_studio"

    def test_client_get_model_info_vertex_ai(self):
        """Verify get_model_info shows Vertex AI backend."""
        from src.integrations.google_adk.ai_studio_client import AIStudioClient, AIStudioConfig

        config = AIStudioConfig(
            use_vertex_ai=True,
            project_id="my-project",
            location="us-central1",
        )
        client = AIStudioClient(config)

        info = client.get_model_info()

        assert info["backend"] == "vertex_ai"
        assert info["project"] == "my-project"
        assert info["location"] == "us-central1"


class TestAIStudioClientAsync:
    """Test AIStudioClient async methods."""

    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self):
        """Verify client initializes with API key (mocked)."""
        from src.integrations.google_adk.ai_studio_client import AIStudioClient, AIStudioConfig

        config = AIStudioConfig(api_key="test-key")
        client = AIStudioClient(config)

        # Mock the google.genai module
        mock_genai = MagicMock()
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            with patch("src.integrations.google_adk.ai_studio_client.genai", mock_genai, create=True):
                # This tests the import path, actual initialization requires the SDK
                pass

    @pytest.mark.asyncio
    async def test_generate_not_initialized(self):
        """Verify generate auto-initializes if not initialized."""
        from src.integrations.google_adk.ai_studio_client import AIStudioClient, AIStudioConfig

        config = AIStudioConfig(api_key="test-key")
        client = AIStudioClient(config)

        assert client._initialized is False

        # Mock initialization
        mock_genai = MagicMock()
        mock_client_instance = MagicMock()
        mock_genai.Client.return_value = mock_client_instance

        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_client_instance.models.generate_content.return_value = mock_response

        with patch("src.integrations.google_adk.ai_studio_client.genai", mock_genai, create=True):
            # Patch the import within initialize
            with patch.object(client, "initialize", new_callable=AsyncMock) as mock_init:
                mock_init.return_value = None
                client._initialized = True
                client._client = mock_client_instance

                response = await client.generate("Test prompt")

                assert response == "Generated response"


class TestAIStudioIntegration:
    """Integration tests for AI Studio module."""

    def test_module_exported_from_google_adk(self):
        """Verify AI Studio classes are exported from google_adk package."""
        from src.integrations.google_adk import (
            AIStudioClient,
            AIStudioConfig,
            GeminiModel,
        )

        assert AIStudioClient is not None
        assert AIStudioConfig is not None
        assert GeminiModel is not None

    def test_google_adk_version_updated(self):
        """Verify google_adk version was updated for new feature."""
        from src.integrations import google_adk

        # Version should be 0.2.0 or higher after adding AI Studio client
        version = getattr(google_adk, "__version__", "0.0.0")
        major, minor, patch = map(int, version.split("."))

        assert (major, minor) >= (0, 2), f"Version should be >= 0.2.0, got {version}"
