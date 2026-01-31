"""
Tests for Hugging Face Spaces deployment configuration.

These tests validate the deployment configuration files and ensure
they meet Hugging Face Spaces requirements.
"""

import os
import re
from pathlib import Path

import pytest
import yaml

# Get the repository root (tests/deployment/ -> tests/ -> repo_root)
REPO_ROOT = Path(__file__).parent.parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy" / "huggingface"


class TestHuggingFaceDeploymentConfig:
    """Test Hugging Face deployment configuration files."""

    def test_deploy_directory_exists(self):
        """Verify the deployment directory exists."""
        assert DEPLOY_DIR.exists(), f"Deploy directory not found: {DEPLOY_DIR}"

    def test_required_files_exist(self):
        """Verify all required deployment files exist."""
        required_files = [
            "README.md",
            "Dockerfile",
            "requirements.txt",
            "deploy.sh",
        ]

        for filename in required_files:
            filepath = DEPLOY_DIR / filename
            assert filepath.exists(), f"Required file not found: {filename}"

    def test_readme_has_yaml_frontmatter(self):
        """Verify README.md has valid YAML frontmatter for Hugging Face."""
        readme_path = DEPLOY_DIR / "README.md"
        content = readme_path.read_text()

        # Check for YAML frontmatter delimiters
        assert content.startswith("---"), "README.md must start with YAML frontmatter"
        assert content.count("---") >= 2, "README.md must have closing YAML delimiter"

        # Extract YAML frontmatter
        yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        assert yaml_match, "Could not extract YAML frontmatter"

        # Parse YAML
        yaml_content = yaml.safe_load(yaml_match.group(1))
        assert yaml_content is not None, "YAML frontmatter is empty"

        # Check required fields
        required_fields = ["title", "emoji", "sdk"]
        for field in required_fields:
            assert field in yaml_content, f"Missing required YAML field: {field}"

    def test_readme_yaml_sdk_is_docker(self):
        """Verify SDK is set to docker for containerized deployment."""
        readme_path = DEPLOY_DIR / "README.md"
        content = readme_path.read_text()

        yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        yaml_content = yaml.safe_load(yaml_match.group(1))

        assert yaml_content.get("sdk") == "docker", "SDK should be 'docker' for containerized deployment"

    def test_readme_yaml_has_app_port(self):
        """Verify app_port is configured for Docker deployment."""
        readme_path = DEPLOY_DIR / "README.md"
        content = readme_path.read_text()

        yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        yaml_content = yaml.safe_load(yaml_match.group(1))

        assert "app_port" in yaml_content, "Docker SDK requires app_port configuration"
        assert yaml_content["app_port"] == 7860, "Default Gradio port should be 7860"


class TestHuggingFaceDockerfile:
    """Test Hugging Face Dockerfile configuration."""

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found"

    def test_dockerfile_has_python_base(self):
        """Verify Dockerfile uses Python base image."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "FROM python:" in content, "Dockerfile should use Python base image"

    def test_dockerfile_exposes_gradio_port(self):
        """Verify Dockerfile exposes the Gradio port."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "EXPOSE 7860" in content, "Dockerfile should expose port 7860 for Gradio"

    def test_dockerfile_has_healthcheck(self):
        """Verify Dockerfile has a health check."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "HEALTHCHECK" in content, "Dockerfile should have a HEALTHCHECK"

    def test_dockerfile_uses_non_root_user(self):
        """Verify Dockerfile creates and uses non-root user (Hugging Face requirement)."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        # Check for user creation
        assert "useradd" in content or "USER" in content, "Dockerfile should create/use non-root user"


class TestHuggingFaceRequirements:
    """Test Hugging Face requirements.txt configuration."""

    def test_requirements_exists(self):
        """Verify requirements.txt exists."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        assert requirements_path.exists(), "requirements.txt not found"

    def test_requirements_has_gradio(self):
        """Verify requirements.txt includes gradio."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "gradio" in content, "requirements.txt must include gradio"

    def test_requirements_has_torch(self):
        """Verify requirements.txt includes torch for neural models."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "torch" in content, "requirements.txt must include torch"

    def test_requirements_has_transformers(self):
        """Verify requirements.txt includes transformers."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text().lower()

        assert "transformers" in content, "requirements.txt must include transformers"

    def test_requirements_parseable(self):
        """Verify requirements.txt is properly formatted."""
        requirements_path = DEPLOY_DIR / "requirements.txt"
        content = requirements_path.read_text()

        # Each non-comment, non-empty line should be a valid requirement
        for line_num, line in enumerate(content.split("\n"), 1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                # Basic format check: should contain package name
                assert re.match(r"^[\w\-\[\]]+", stripped), f"Invalid requirement on line {line_num}: {line}"


class TestHuggingFaceDeployScript:
    """Test Hugging Face deployment script."""

    def test_deploy_script_exists(self):
        """Verify deploy.sh exists."""
        script_path = DEPLOY_DIR / "deploy.sh"
        assert script_path.exists(), "deploy.sh not found"

    def test_deploy_script_is_executable(self):
        """Verify deploy.sh is executable."""
        script_path = DEPLOY_DIR / "deploy.sh"
        assert os.access(script_path, os.X_OK), "deploy.sh should be executable"

    def test_deploy_script_has_shebang(self):
        """Verify deploy.sh has proper shebang."""
        script_path = DEPLOY_DIR / "deploy.sh"
        content = script_path.read_text()

        assert content.startswith("#!/bin/bash"), "deploy.sh should start with #!/bin/bash"

    def test_deploy_script_has_error_handling(self):
        """Verify deploy.sh has error handling."""
        script_path = DEPLOY_DIR / "deploy.sh"
        content = script_path.read_text()

        assert "set -e" in content, "deploy.sh should use 'set -e' for error handling"

    def test_deploy_script_checks_prerequisites(self):
        """Verify deploy.sh checks for huggingface-cli."""
        script_path = DEPLOY_DIR / "deploy.sh"
        content = script_path.read_text()

        assert "huggingface-cli" in content, "deploy.sh should check for huggingface-cli"


class TestHuggingFaceSanity:
    """Sanity tests for Hugging Face deployment."""

    def test_app_py_exists_in_repo(self):
        """Verify main app.py exists in repository root."""
        app_path = REPO_ROOT / "app.py"
        assert app_path.exists(), "app.py must exist in repository root for Hugging Face deployment"

    def test_app_py_uses_gradio(self):
        """Verify app.py uses Gradio."""
        app_path = REPO_ROOT / "app.py"
        content = app_path.read_text()

        assert "import gradio" in content or "from gradio" in content, "app.py should use Gradio"

    def test_models_directory_referenced(self):
        """Verify Dockerfile references models directory."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "models" in content, "Dockerfile should reference models directory"

    def test_src_directory_referenced(self):
        """Verify Dockerfile references src directory."""
        dockerfile_path = DEPLOY_DIR / "Dockerfile"
        content = dockerfile_path.read_text()

        assert "src" in content, "Dockerfile should reference src directory"
