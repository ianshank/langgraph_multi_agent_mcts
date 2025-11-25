#!/usr/bin/env python3
"""
Deployment Verification Script for Hugging Face Spaces
=======================================================

This script performs comprehensive dependency and compatibility checks
before deployment to identify and resolve conflicts early.

Usage:
    python verify_deployment.py [--fix] [--verbose]

The script implements backpropagation principles:
1. Identify errors at the surface level
2. Trace back through the dependency chain
3. Fix conflicts at each level
4. Validate the complete stack
"""

import argparse
import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


@dataclass
class DependencyCheck:
    """Result of a dependency check."""

    name: str
    required_version: str
    installed_version: str | None
    compatible: bool
    error_message: str | None = None


@dataclass
class CompatibilityIssue:
    """Compatibility issue between packages."""

    package1: str
    package2: str
    issue: str
    resolution: str


class DeploymentVerifier:
    """Comprehensive deployment verification."""

    # Known compatibility issues and their resolutions
    KNOWN_ISSUES = [
        {
            "packages": ["peft", "transformers"],
            "condition": lambda v: v["peft"].startswith("0.10") and v["transformers"].startswith("4.46"),
            "issue": "PEFT 0.10.0 requires transformers<4.46 (missing modeling_layers)",
            "resolution": "Upgrade to peft>=0.12.0 for transformers>=4.46.0 compatibility",
        },
        {
            "packages": ["sentence-transformers", "transformers"],
            "condition": lambda v: "sentence-transformers" in v and v["transformers"].startswith("4.46"),
            "issue": "sentence-transformers may be incompatible with transformers 4.46.0",
            "resolution": "Either downgrade transformers to 4.40-4.45 or disable sentence-transformers",
        },
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues = []
        self.checks = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if level == "ERROR":
            prefix = "❌"
        elif level == "WARNING":
            prefix = "⚠️"
        elif level == "SUCCESS":
            prefix = "✅"
        else:
            prefix = "ℹ️"

        print(f"{prefix} {message}")

    def verbose_log(self, message: str):
        """Log a verbose message."""
        if self.verbose:
            print(f"  → {message}")

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        self.log("Checking Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            self.log(f"Python {version.major}.{version.minor} is too old. Requires Python 3.10+", "ERROR")
            return False

        self.log(f"Python {version.major}.{version.minor}.{version.micro}", "SUCCESS")
        return True

    def get_installed_version(self, package: str) -> str | None:
        """Get installed version of a package."""
        try:
            version = importlib.metadata.version(package)
            return version
        except importlib.metadata.PackageNotFoundError:
            return None

    def parse_requirement(self, req: str) -> tuple[str, str]:
        """Parse a requirement string into (package, version_spec)."""
        req = req.strip()

        # Remove comments
        if "#" in req:
            req = req.split("#")[0].strip()

        if not req:
            return "", ""

        # Parse package name and version
        for op in [">=", "<=", "==", "!=", ">", "<", "~="]:
            if op in req:
                parts = req.split(op)
                return parts[0].strip(), req

        # No version specified
        return req, req

    def check_requirements_file(self, requirements_path: Path) -> list[DependencyCheck]:
        """Check all requirements in a file."""
        self.log(f"Checking requirements file: {requirements_path}")

        if not requirements_path.exists():
            self.log(f"Requirements file not found: {requirements_path}", "ERROR")
            return []

        checks = []
        with open(requirements_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                package_name, version_spec = self.parse_requirement(line)
                if not package_name:
                    continue

                installed = self.get_installed_version(package_name)

                check = DependencyCheck(
                    name=package_name,
                    required_version=version_spec,
                    installed_version=installed,
                    compatible=installed is not None,
                )

                if installed:
                    self.verbose_log(f"{package_name}: {installed}")
                else:
                    self.log(f"{package_name}: NOT INSTALLED", "WARNING")

                checks.append(check)

        self.checks.extend(checks)
        return checks

    def check_import_compatibility(self) -> list[str]:
        """Check if critical imports work."""
        self.log("Checking critical imports...")

        critical_imports = [
            ("gradio", "Gradio UI framework"),
            ("torch", "PyTorch"),
            ("transformers", "Transformers library"),
            ("peft", "PEFT library (optional for LoRA)"),
        ]

        errors = []
        for module_name, description in critical_imports:
            try:
                __import__(module_name)
                self.verbose_log(f"{module_name}: OK")
            except ImportError as e:
                error_msg = f"{description} import failed: {e}"
                self.log(error_msg, "ERROR")
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"{description} unexpected error: {type(e).__name__}: {e}"
                self.log(error_msg, "ERROR")
                errors.append(error_msg)

        if not errors:
            self.log("All critical imports successful", "SUCCESS")

        return errors

    def detect_compatibility_issues(self) -> list[CompatibilityIssue]:
        """Detect known compatibility issues."""
        self.log("Detecting compatibility issues...")

        issues = []

        # Get installed versions
        versions = {}
        for check in self.checks:
            if check.installed_version:
                versions[check.name] = check.installed_version

        # Check known issues
        for known_issue in self.KNOWN_ISSUES:
            packages = known_issue["packages"]

            # Check if all packages are installed
            if all(pkg in versions for pkg in packages):
                # Check condition
                if known_issue["condition"](versions):
                    issue = CompatibilityIssue(
                        package1=packages[0],
                        package2=packages[1] if len(packages) > 1 else "N/A",
                        issue=known_issue["issue"],
                        resolution=known_issue["resolution"],
                    )
                    issues.append(issue)
                    self.log(f"{issue.issue}", "WARNING")
                    self.log(f"  Resolution: {issue.resolution}", "INFO")

        if not issues:
            self.log("No known compatibility issues detected", "SUCCESS")

        self.issues.extend(issues)
        return issues

    def test_transformers_modeling_layers(self) -> bool:
        """Test if transformers.modeling_layers is available."""
        self.log("Testing transformers.modeling_layers module...")

        try:
            from transformers import modeling_layers
            self.log("transformers.modeling_layers available", "SUCCESS")
            return True
        except ImportError:
            self.log("transformers.modeling_layers NOT available (may cause PEFT issues)", "WARNING")
            return False

    def generate_fix_recommendations(self) -> list[str]:
        """Generate specific fix recommendations."""
        recommendations = []

        # Check for PEFT + transformers issue
        peft_version = self.get_installed_version("peft")
        transformers_version = self.get_installed_version("transformers")

        if peft_version and transformers_version:
            if peft_version.startswith("0.10") and transformers_version.startswith("4.46"):
                recommendations.append(
                    "CRITICAL: Upgrade PEFT to >=0.12.0 or downgrade transformers to <4.46.0"
                )

        # Check for sentence-transformers
        sentence_transformers_version = self.get_installed_version("sentence-transformers")
        if sentence_transformers_version and transformers_version:
            if transformers_version.startswith("4.46"):
                recommendations.append(
                    "WARNING: sentence-transformers may be incompatible with transformers 4.46.0. "
                    "Consider using heuristic-based feature extraction instead."
                )

        return recommendations

    def run_full_verification(self, requirements_path: Path) -> bool:
        """Run full verification suite."""
        self.log("=" * 80)
        self.log("Starting Deployment Verification")
        self.log("=" * 80)

        all_passed = True

        # 1. Python version
        if not self.check_python_version():
            all_passed = False

        print()

        # 2. Requirements
        self.check_requirements_file(requirements_path)

        print()

        # 3. Import compatibility
        import_errors = self.check_import_compatibility()
        if import_errors:
            all_passed = False

        print()

        # 4. Compatibility issues
        self.detect_compatibility_issues()

        print()

        # 5. Specific tests
        self.test_transformers_modeling_layers()

        print()

        # 6. Recommendations
        recommendations = self.generate_fix_recommendations()
        if recommendations:
            self.log("Fix Recommendations:", "INFO")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
            print()

        # Summary
        self.log("=" * 80)
        if all_passed and not self.issues:
            self.log("✅ ALL CHECKS PASSED - Deployment ready!", "SUCCESS")
        else:
            self.log("❌ ISSUES DETECTED - Fix before deployment", "ERROR")
            all_passed = False
        self.log("=" * 80)

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify Hugging Face Space deployment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--requirements",
        "-r",
        type=Path,
        default=Path("requirements.txt"),
        help="Path to requirements.txt file",
    )

    args = parser.parse_args()

    verifier = DeploymentVerifier(verbose=args.verbose)
    success = verifier.run_full_verification(args.requirements)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
