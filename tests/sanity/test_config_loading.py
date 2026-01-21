"""
CI/CD Sanity Tests - Configuration Loading
===========================================

Fast tests to verify project configurations can be loaded and validated.
These tests should always pass and run in <5 seconds.
"""

from pathlib import Path
import tomllib

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.sanity
class TestProjectConfig:
    """Verify project configuration files are valid."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml not found"

    def test_pyproject_toml_parseable(self):
        """Test that pyproject.toml can be parsed."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        assert "project" in config
        assert "name" in config["project"]
        assert config["project"]["name"] == "langgraph-multi-agent-mcts"

    def test_pytest_markers_defined(self):
        """Test that required pytest markers are defined."""
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        markers = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
        marker_names = [m.split(":")[0] for m in markers]

        # Check for essential markers
        assert "smoke" in marker_names, "smoke marker not defined"
        assert "integration" in marker_names, "integration marker not defined"
        assert "unit" in marker_names, "unit marker not defined"
        assert "sanity" in marker_names, "sanity marker not defined"


@pytest.mark.sanity
class TestTrainingConfig:
    """Verify training configuration files are valid."""

    def test_training_config_directory_exists(self):
        """Test that training configs directory exists."""
        config_dir = PROJECT_ROOT / "training" / "configs"
        assert config_dir.exists(), "training/configs directory not found"

    def test_default_config_exists(self):
        """Test that default training config exists."""
        config_path = PROJECT_ROOT / "training" / "config.yaml"
        # Config might be in configs subdirectory
        alt_config_path = PROJECT_ROOT / "training" / "configs" / "config.yaml"

        assert config_path.exists() or alt_config_path.exists(), "No default training config found"


@pytest.mark.sanity
class TestEnvironmentConfig:
    """Verify environment configuration loads correctly."""

    def test_continuous_play_config_from_env(self):
        """Test ContinuousPlayConfig loads from environment."""
        from src.training.continuous_play_config import ContinuousPlayConfig

        config = ContinuousPlayConfig.from_env()
        assert config is not None
        assert config.session is not None
        assert config.learning is not None

    def test_config_validation_runs(self):
        """Test that config validation can be executed."""
        from src.training.continuous_play_config import ContinuousPlayConfig

        config = ContinuousPlayConfig.from_env()
        errors = config.validate()

        # Should return a list (possibly empty)
        assert isinstance(errors, list)
