"""
Sanity tests for deployment configurations.

These tests perform quick validation of all deployment configurations
to ensure they are properly set up and ready for deployment.
"""

import os
import re
from pathlib import Path

import pytest
import yaml

# Get the repository root (tests/deployment/ -> tests/ -> repo_root)
REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.mark.smoke
class TestDeploymentSanity:
    """Sanity tests that run quickly to validate deployment setup."""

    def test_deploy_directory_structure(self):
        """Verify the deploy directory has expected structure."""
        deploy_dir = REPO_ROOT / "deploy"
        assert deploy_dir.exists(), "deploy/ directory must exist"

        # Check subdirectories
        assert (deploy_dir / "huggingface").exists(), "deploy/huggingface/ must exist"
        assert (deploy_dir / "google_cloud").exists(), "deploy/google_cloud/ must exist"

    def test_all_dockerfiles_valid_syntax(self):
        """Verify all Dockerfiles have valid basic syntax."""
        deploy_dir = REPO_ROOT / "deploy"

        dockerfiles = list(deploy_dir.glob("**/Dockerfile"))
        assert len(dockerfiles) >= 2, "Expected at least 2 Dockerfiles"

        for dockerfile in dockerfiles:
            content = dockerfile.read_text()

            # Must have FROM instruction
            assert "FROM" in content, f"{dockerfile} must have FROM instruction"

            # Must have at least one RUN, COPY, or CMD
            has_instruction = any(
                instr in content for instr in ["RUN", "COPY", "CMD", "ENTRYPOINT"]
            )
            assert has_instruction, f"{dockerfile} must have build/run instructions"

    def test_all_shell_scripts_executable(self):
        """Verify all deployment shell scripts are executable."""
        deploy_dir = REPO_ROOT / "deploy"

        scripts = list(deploy_dir.glob("**/*.sh"))
        assert len(scripts) >= 2, "Expected at least 2 shell scripts"

        for script in scripts:
            assert os.access(script, os.X_OK), f"{script} must be executable"

    def test_all_shell_scripts_have_shebang(self):
        """Verify all shell scripts have proper shebang."""
        deploy_dir = REPO_ROOT / "deploy"

        scripts = list(deploy_dir.glob("**/*.sh"))

        for script in scripts:
            content = script.read_text()
            assert content.startswith("#!/bin/bash"), f"{script} must start with #!/bin/bash"

    def test_all_requirements_files_exist(self):
        """Verify requirements.txt files exist in deploy directories."""
        deploy_dir = REPO_ROOT / "deploy"

        requirements = list(deploy_dir.glob("**/requirements.txt"))
        assert len(requirements) >= 2, "Expected at least 2 requirements.txt files"

    def test_all_yaml_files_valid(self):
        """Verify all YAML files in deploy directories are valid."""
        deploy_dir = REPO_ROOT / "deploy"

        yaml_files = list(deploy_dir.glob("**/*.yaml")) + list(deploy_dir.glob("**/*.yml"))

        for yaml_file in yaml_files:
            content = yaml_file.read_text()
            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {yaml_file}: {e}")

    def test_env_example_has_google_cloud_vars(self):
        """Verify .env.example has Google Cloud configuration."""
        env_example = REPO_ROOT / ".env.example"
        assert env_example.exists(), ".env.example must exist"

        content = env_example.read_text()

        assert "GOOGLE_CLOUD_PROJECT" in content, ".env.example should have GOOGLE_CLOUD_PROJECT"
        assert "GOOGLE_CLOUD_LOCATION" in content, ".env.example should have GOOGLE_CLOUD_LOCATION"
        assert "GOOGLE_API_KEY" in content, ".env.example should have GOOGLE_API_KEY"

    def test_env_example_has_huggingface_vars(self):
        """Verify .env.example has Hugging Face configuration."""
        env_example = REPO_ROOT / ".env.example"
        content = env_example.read_text()

        assert "HF_TOKEN" in content, ".env.example should have HF_TOKEN"

    def test_deployment_guide_exists(self):
        """Verify deployment guide documentation exists."""
        guide = REPO_ROOT / "docs" / "DEPLOYMENT_GUIDE.md"
        assert guide.exists(), "docs/DEPLOYMENT_GUIDE.md must exist"

        content = guide.read_text()
        assert len(content) > 1000, "Deployment guide should be comprehensive"

        # Check it covers both platforms
        assert "Hugging Face" in content, "Guide should cover Hugging Face"
        assert "Google Cloud" in content, "Guide should cover Google Cloud"

    def test_ai_studio_client_importable(self):
        """Verify AI Studio client can be imported."""
        try:
            from src.integrations.google_adk import AIStudioClient, AIStudioConfig
        except ImportError as e:
            pytest.fail(f"Failed to import AI Studio client: {e}")

    def test_google_adk_exports_updated(self):
        """Verify google_adk __init__.py exports new classes."""
        init_path = REPO_ROOT / "src" / "integrations" / "google_adk" / "__init__.py"
        content = init_path.read_text()

        assert "AIStudioClient" in content, "__init__.py should export AIStudioClient"
        assert "AIStudioConfig" in content, "__init__.py should export AIStudioConfig"
        assert "GeminiModel" in content, "__init__.py should export GeminiModel"


@pytest.mark.smoke
class TestHuggingFaceSanity:
    """Quick sanity tests for Hugging Face deployment."""

    def test_readme_yaml_valid(self):
        """Verify Hugging Face README has valid YAML."""
        readme_path = REPO_ROOT / "deploy" / "huggingface" / "README.md"
        content = readme_path.read_text()

        # Extract YAML
        yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        assert yaml_match, "README should have YAML frontmatter"

        yaml_content = yaml.safe_load(yaml_match.group(1))
        assert yaml_content.get("sdk") == "docker", "SDK should be docker"

    def test_dockerfile_app_port_matches_readme(self):
        """Verify Dockerfile EXPOSE matches README app_port."""
        readme_path = REPO_ROOT / "deploy" / "huggingface" / "README.md"
        dockerfile_path = REPO_ROOT / "deploy" / "huggingface" / "Dockerfile"

        # Get port from README YAML
        readme_content = readme_path.read_text()
        yaml_match = re.match(r"^---\n(.*?)\n---", readme_content, re.DOTALL)
        yaml_content = yaml.safe_load(yaml_match.group(1))
        app_port = yaml_content.get("app_port", 7860)

        # Check Dockerfile exposes this port
        dockerfile_content = dockerfile_path.read_text()
        assert f"EXPOSE {app_port}" in dockerfile_content, (
            f"Dockerfile should EXPOSE {app_port}"
        )


@pytest.mark.smoke
class TestGoogleCloudSanity:
    """Quick sanity tests for Google Cloud deployment."""

    def test_cloud_run_port_consistent(self):
        """Verify Cloud Run port is consistent across files."""
        dockerfile_path = REPO_ROOT / "deploy" / "google_cloud" / "Dockerfile"
        config_path = REPO_ROOT / "deploy" / "google_cloud" / "config.yaml"

        # Check Dockerfile
        dockerfile_content = dockerfile_path.read_text()
        assert "8080" in dockerfile_content, "Dockerfile should reference port 8080"

        # Check config
        config = yaml.safe_load(config_path.read_text())
        container_port = config.get("cloud_run", {}).get("container", {}).get("port", 8080)
        assert container_port == 8080, "Cloud Run port should be 8080"

    def test_vertex_ai_app_has_cors(self):
        """Verify Vertex AI app has CORS middleware."""
        app_path = REPO_ROOT / "deploy" / "google_cloud" / "vertex_ai_app.py"
        content = app_path.read_text()

        assert "CORSMiddleware" in content, "App should have CORS middleware"

    def test_deploy_scripts_check_project_id(self):
        """Verify deploy scripts check for project ID."""
        for script_name in ["deploy_cloud_run.sh", "deploy_vertex_ai.sh"]:
            script_path = REPO_ROOT / "deploy" / "google_cloud" / script_name
            content = script_path.read_text()

            # Should check if PROJECT_ID is set
            assert "PROJECT_ID" in content, f"{script_name} should reference PROJECT_ID"
            # Should have some validation
            assert "if" in content, f"{script_name} should have conditional checks"


@pytest.mark.smoke
class TestCrossDeploymentConsistency:
    """Tests for consistency across deployment configurations."""

    def test_python_version_consistent(self):
        """Verify Python version is consistent across Dockerfiles."""
        hf_dockerfile = REPO_ROOT / "deploy" / "huggingface" / "Dockerfile"
        gc_dockerfile = REPO_ROOT / "deploy" / "google_cloud" / "Dockerfile"

        hf_content = hf_dockerfile.read_text()
        gc_content = gc_dockerfile.read_text()

        # Both should use Python 3.11
        assert "python:3.11" in hf_content, "HF Dockerfile should use Python 3.11"
        assert "python:3.11" in gc_content, "GC Dockerfile should use Python 3.11"

    def test_core_dependencies_present_in_both(self):
        """Verify core dependencies are in both requirements files."""
        hf_reqs = (REPO_ROOT / "deploy" / "huggingface" / "requirements.txt").read_text().lower()
        gc_reqs = (REPO_ROOT / "deploy" / "google_cloud" / "requirements.txt").read_text().lower()

        core_deps = ["torch", "transformers", "peft"]

        for dep in core_deps:
            assert dep in hf_reqs, f"HF requirements should have {dep}"
            assert dep in gc_reqs, f"GC requirements should have {dep}"

    def test_both_dockerfiles_use_non_root_user(self):
        """Verify both Dockerfiles use non-root users."""
        hf_dockerfile = REPO_ROOT / "deploy" / "huggingface" / "Dockerfile"
        gc_dockerfile = REPO_ROOT / "deploy" / "google_cloud" / "Dockerfile"

        hf_content = hf_dockerfile.read_text()
        gc_content = gc_dockerfile.read_text()

        assert "USER" in hf_content, "HF Dockerfile should set USER"
        assert "USER" in gc_content or "useradd" in gc_content, "GC Dockerfile should create/use non-root user"
