"""
Tests for Google Cloud deployment configuration.

These tests validate the deployment configuration files for:
- Cloud Run
- Vertex AI Agent Engine
- Google AI Studio integration
"""

import os
import re
from pathlib import Path

import pytest
import yaml

# Get the repository root (tests/deployment/ -> tests/ -> repo_root)
REPO_ROOT = Path(__file__).parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy" / "google_cloud"


class TestGoogleCloudDeploymentConfig:
    """Test Google Cloud deployment configuration files."""

    def test_deploy_directory_exists(self):
        """Verify the deployment directory exists."""
        assert DEPLOY_DIR.exists(), f"Deploy directory not found: {DEPLOY_DIR}"

    def test_required_files_exist(self):
        """Verify all required deployment files exist."""
        required_files = [
            "README.md",
            "Dockerfile",
            "requirements.txt",
            "config.yaml",
            "vertex_ai_app.py",
            "deploy_cloud_run.sh",
            "deploy_vertex_ai.sh",
        ]

        for filename in required_files:
            filepath = DEPLOY_DIR / filename
            assert filepath.exists(), f"Required file not found: {filename}"


class TestGoogleCloudDockerfile:
    """Test Google Cloud Dockerfile configuration."""

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found"

    def test_dockerfile_has_python_base(self):
        """Verify Dockerfile uses Python base image."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "FROM python:" in content, "Dockerfile should use Python base image"

    def test_dockerfile_exposes_cloud_run_port(self):
        """Verify Dockerfile exposes Cloud Run port 8080."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "EXPOSE 8080" in content, "Dockerfile should expose port 8080 for Cloud Run"

    def test_dockerfile_has_healthcheck(self):
        """Verify Dockerfile has a health check."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "HEALTHCHECK" in content, "Dockerfile should have a HEALTHCHECK"

    def test_dockerfile_sets_port_env(self):
        """Verify Dockerfile sets PORT environment variable."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "PORT=8080" in content or "PORT" in content, "Dockerfile should set PORT for Cloud Run"


class TestGoogleCloudConfig:
    """Test Google Cloud config.yaml configuration."""

    def test_config_exists(self):
        """Verify config.yaml exists."""
        config_path = DEPLOY_DIR / "config.yaml"
        assert config_path.exists(), "config.yaml not found"

    def test_config_is_valid_yaml(self):
        """Verify config.yaml is valid YAML."""
        config_path = DEPLOY_DIR / "config.yaml"
        content = config_path.read_text()

        # Should parse without error
        config = yaml.safe_load(content)
        assert config is not None, "config.yaml should not be empty"

    def test_config_has_project_section(self):
        """Verify config.yaml has project configuration."""
        config_path = DEPLOY_DIR / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        assert "project" in config, "config.yaml should have project section"

    def test_config_has_vertex_ai_section(self):
        """Verify config.yaml has Vertex AI configuration."""
        config_path = DEPLOY_DIR / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        assert "vertex_ai" in config, "config.yaml should have vertex_ai section"

    def test_config_has_cloud_run_section(self):
        """Verify config.yaml has Cloud Run configuration."""
        config_path = DEPLOY_DIR / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        assert "cloud_run" in config, "config.yaml should have cloud_run section"

    def test_config_has_iam_section(self):
        """Verify config.yaml has IAM configuration."""
        config_path = DEPLOY_DIR / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        assert "iam" in config, "config.yaml should have iam section"


class TestGoogleCloudRequirements:
    """Test Google Cloud requirements.txt configuration."""

    def test_requirements_exists(self):
        """Verify requirements.txt exists."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        assert requirements_path.exists(), "requirements.txt not found"

    def test_requirements_has_google_cloud_aiplatform(self):
        """Verify requirements.txt includes google-cloud-aiplatform."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "google-cloud-aiplatform" in content, "requirements.txt must include google-cloud-aiplatform"

    def test_requirements_has_google_genai(self):
        """Verify requirements.txt includes google-genai."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "google-genai" in content, "requirements.txt must include google-genai"

    def test_requirements_has_fastapi(self):
        """Verify requirements.txt includes FastAPI."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "fastapi" in content, "requirements.txt must include fastapi"

    def test_requirements_has_uvicorn(self):
        """Verify requirements.txt includes uvicorn."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "uvicorn" in content, "requirements.txt must include uvicorn"

    def test_requirements_has_langgraph(self):
        """Verify requirements.txt includes langgraph."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "langgraph" in content, "requirements.txt must include langgraph"


class TestVertexAIApp:
    """Test Vertex AI FastAPI application."""

    def test_vertex_ai_app_exists(self):
        """Verify vertex_ai_app.py exists."""
        app_path = DEPLOY_DIR / "vertex_ai_app.py"
        assert app_path.exists(), "vertex_ai_app.py not found"

    def test_vertex_ai_app_imports_fastapi(self):
        """Verify vertex_ai_app.py imports FastAPI."""
        app_path = DEPLOY_DIR / "vertex_ai_app.py"
        content = app_path.read_text()

        assert "from fastapi import" in content or "import fastapi" in content, (
            "vertex_ai_app.py should import FastAPI"
        )

    def test_vertex_ai_app_has_health_endpoint(self):
        """Verify vertex_ai_app.py has health endpoint."""
        app_path = DEPLOY_DIR / "vertex_ai_app.py"
        content = app_path.read_text()

        assert '"/health"' in content or "'/health'" in content, (
            "vertex_ai_app.py should have /health endpoint"
        )

    def test_vertex_ai_app_has_query_endpoint(self):
        """Verify vertex_ai_app.py has query endpoint."""
        app_path = DEPLOY_DIR / "vertex_ai_app.py"
        content = app_path.read_text()

        assert '"/query"' in content or "'/query'" in content, (
            "vertex_ai_app.py should have /query endpoint"
        )

    def test_vertex_ai_app_is_syntactically_valid(self):
        """Verify vertex_ai_app.py is syntactically valid Python."""
        app_path = DEPLOY_DIR / "vertex_ai_app.py"

        # This will raise SyntaxError if invalid
        compile(app_path.read_text(), str(app_path), "exec")


class TestDeployScripts:
    """Test deployment scripts."""

    def test_cloud_run_script_exists(self):
        """Verify deploy_cloud_run.sh exists."""
        script_path = DEPLOY_DIR / "deploy_cloud_run.sh"
        assert script_path.exists(), "deploy_cloud_run.sh not found"

    def test_cloud_run_script_is_executable(self):
        """Verify deploy_cloud_run.sh is executable."""
        script_path = DEPLOY_DIR / "deploy_cloud_run.sh"
        assert os.access(script_path, os.X_OK), "deploy_cloud_run.sh should be executable"

    def test_cloud_run_script_has_shebang(self):
        """Verify deploy_cloud_run.sh has proper shebang."""
        script_path = DEPLOY_DIR / "deploy_cloud_run.sh"
        content = script_path.read_text()

        assert content.startswith("#!/bin/bash"), "deploy_cloud_run.sh should start with #!/bin/bash"

    def test_cloud_run_script_has_error_handling(self):
        """Verify deploy_cloud_run.sh has error handling."""
        script_path = DEPLOY_DIR / "deploy_cloud_run.sh"
        content = script_path.read_text()

        assert "set -e" in content, "deploy_cloud_run.sh should use 'set -e' for error handling"

    def test_cloud_run_script_uses_gcloud(self):
        """Verify deploy_cloud_run.sh uses gcloud commands."""
        script_path = DEPLOY_DIR / "deploy_cloud_run.sh"
        content = script_path.read_text()

        assert "gcloud" in content, "deploy_cloud_run.sh should use gcloud commands"

    def test_vertex_ai_script_exists(self):
        """Verify deploy_vertex_ai.sh exists."""
        script_path = DEPLOY_DIR / "deploy_vertex_ai.sh"
        assert script_path.exists(), "deploy_vertex_ai.sh not found"

    def test_vertex_ai_script_is_executable(self):
        """Verify deploy_vertex_ai.sh is executable."""
        script_path = DEPLOY_DIR / "deploy_vertex_ai.sh"
        assert os.access(script_path, os.X_OK), "deploy_vertex_ai.sh should be executable"

    def test_vertex_ai_script_has_shebang(self):
        """Verify deploy_vertex_ai.sh has proper shebang."""
        script_path = DEPLOY_DIR / "deploy_vertex_ai.sh"
        content = script_path.read_text()

        assert content.startswith("#!/bin/bash"), "deploy_vertex_ai.sh should start with #!/bin/bash"


class TestGoogleCloudSanity:
    """Sanity tests for Google Cloud deployment."""

    def test_config_default_region(self):
        """Verify config has sensible default region."""
        config_path = DEPLOY_DIR / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        # Check project location
        project = config.get("project", {})
        location = project.get("location", "")

        assert "us-central1" in location or "${GOOGLE_CLOUD_LOCATION" in location, (
            "Default region should be us-central1 or use environment variable"
        )

    def test_cloud_run_memory_reasonable(self):
        """Verify Cloud Run memory configuration is reasonable."""
        config_path = DEPLOY_DIR / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        cloud_run = config.get("cloud_run", {}).get("container", {})
        memory = cloud_run.get("memory", "")

        # Memory should be at least 2Gi for ML workloads
        assert "Gi" in memory or "Mi" in memory, "Memory should be specified in Gi or Mi"

    def test_scripts_reference_project_env_var(self):
        """Verify scripts use GOOGLE_CLOUD_PROJECT environment variable."""
        for script_name in ["deploy_cloud_run.sh", "deploy_vertex_ai.sh"]:
            script_path = DEPLOY_DIR / script_name
            content = script_path.read_text()

            assert "GOOGLE_CLOUD_PROJECT" in content, (
                f"{script_name} should reference GOOGLE_CLOUD_PROJECT"
            )
