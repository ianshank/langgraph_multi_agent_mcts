#!/usr/bin/env python3
"""
Cache-Busting Script for Hugging Face Spaces
=============================================

This script implements aggressive cache-busting techniques to force
Hugging Face Spaces to rebuild with fresh dependencies and code.

Techniques:
1. Add unique timestamps/hashes to requirements.txt
2. Update version markers in code files
3. Create cache-busting commit messages
4. Generate deployment markers

Usage:
    python force_cache_bust.py [--commit] [--push]
"""

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output
if sys.stdout.encoding != "utf-8":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class CacheBuster:
    """Aggressive cache-busting for HF Spaces."""

    def __init__(self, space_dir: Path):
        self.space_dir = Path(space_dir)
        self.timestamp = datetime.now()
        self.hash = hashlib.sha256(str(self.timestamp.timestamp()).encode()).hexdigest()[:8]

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        prefix_map = {
            "ERROR": "❌",
            "WARNING": "⚠️",
            "SUCCESS": "✅",
            "INFO": "ℹ️",
        }
        prefix = prefix_map.get(level, "ℹ️")
        print(f"{prefix} {message}")

    def update_requirements_txt(self) -> bool:
        """
        Update requirements.txt with cache-busting markers.

        Strategy:
        1. Add timestamp comment
        2. Update version marker
        3. Add unique rebuild hash
        """
        requirements_path = self.space_dir / "requirements.txt"

        if not requirements_path.exists():
            self.log(f"requirements.txt not found at {requirements_path}", "ERROR")
            return False

        self.log(f"Updating {requirements_path}...")

        # Read current content
        with open(requirements_path) as f:
            lines = f.readlines()

        # Remove old cache-bust markers
        filtered_lines = []
        for line in lines:
            # Skip old cache-bust comments
            if "Force rebuild timestamp:" in line or "Cache-bust hash:" in line:
                continue
            # Skip old version markers
            if line.strip().startswith("# Force Rebuild"):
                continue
            filtered_lines.append(line)

        # Add new cache-bust markers at the end
        timestamp_str = self.timestamp.strftime("%a, %b %d, %Y %I:%M:%S %p")
        cache_bust_markers = [
            f"\n# Force Rebuild {self.timestamp.strftime('%Y-%m-%d-%H-%M-%S')}\n",
            f"# Force rebuild timestamp: {timestamp_str}\n",
            f"# Cache-bust hash: {self.hash}\n",
        ]

        # Write updated content
        with open(requirements_path, "w") as f:
            f.writelines(filtered_lines)
            f.writelines(cache_bust_markers)

        self.log(f"Updated requirements.txt with timestamp: {timestamp_str}", "SUCCESS")
        return True

    def update_app_version(self) -> bool:
        """
        Update version markers in app.py to force code reload.
        """
        app_path = self.space_dir / "app.py"

        if not app_path.exists():
            self.log(f"app.py not found at {app_path}", "WARNING")
            return False

        self.log(f"Updating version in {app_path}...")

        # Read content
        with open(app_path) as f:
            content = f.read()

        # Update APP_VERSION if it exists
        version_pattern = r'APP_VERSION\s*=\s*"[^"]*"'
        new_version = f'APP_VERSION = "{self.timestamp.strftime("%Y-%m-%d-%H%M%S")}-{self.hash}"'

        if re.search(version_pattern, content):
            content = re.sub(version_pattern, new_version, content)
            self.log("Updated APP_VERSION", "SUCCESS")
        else:
            self.log("APP_VERSION not found in app.py", "WARNING")

        # Update VERSION comment if it exists
        version_comment_pattern = r"VERSION:\s*[\w\-]+"
        new_version_comment = f"VERSION: {self.timestamp.strftime('%Y-%m-%d-%H%M%S')}"

        if re.search(version_comment_pattern, content):
            content = re.sub(version_comment_pattern, new_version_comment, content)
            self.log("Updated VERSION comment", "SUCCESS")

        # Write updated content
        with open(app_path, "w") as f:
            f.write(content)

        return True

    def create_deployment_marker(self) -> bool:
        """
        Create a deployment marker file.
        """
        marker_path = self.space_dir / ".deployment_marker"

        marker_content = f"""# Deployment Marker
# This file forces Hugging Face Spaces to recognize changes

Timestamp: {self.timestamp.isoformat()}
Hash: {self.hash}
Deployment ID: {self.timestamp.strftime("%Y%m%d%H%M%S")}-{self.hash}
"""

        with open(marker_path, "w") as f:
            f.write(marker_content)

        self.log(f"Created deployment marker: {marker_path}", "SUCCESS")
        return True

    def generate_commit_message(self) -> str:
        """Generate a unique commit message."""
        return f"""chore: force rebuild - cache bust {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

Deployment ID: {self.hash}
Timestamp: {self.timestamp.isoformat()}

This commit forces Hugging Face Spaces to rebuild with fresh dependencies.

Changes:
- Updated requirements.txt cache-bust markers
- Updated app.py version markers
- Created deployment marker

[skip ci] [cache-bust] [rebuild-{self.hash}]
"""

    def run_git_operations(self, push: bool = False) -> bool:
        """
        Run git operations to commit and optionally push changes.
        """
        try:
            # Check if in git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.space_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.log("Not in a git repository", "ERROR")
                return False

            # Add changes
            self.log("Staging changes...")
            subprocess.run(
                ["git", "add", "requirements.txt", "app.py", ".deployment_marker"],
                cwd=self.space_dir,
                check=True,
            )

            # Commit
            commit_message = self.generate_commit_message()
            self.log("Creating commit...")
            subprocess.run(
                ["git", "commit", "-m", commit_message],
                cwd=self.space_dir,
                check=True,
            )
            self.log("Committed changes", "SUCCESS")

            # Push if requested
            if push:
                self.log("Pushing to remote...")
                subprocess.run(
                    ["git", "push"],
                    cwd=self.space_dir,
                    check=True,
                )
                self.log("Pushed to remote", "SUCCESS")

            return True

        except subprocess.CalledProcessError as e:
            self.log(f"Git operation failed: {e}", "ERROR")
            return False

    def run_full_cache_bust(self, commit: bool = False, push: bool = False) -> bool:
        """
        Run full cache-busting procedure.
        """
        self.log("=" * 80)
        self.log("Starting Cache-Busting Operations")
        self.log("=" * 80)
        self.log(f"Timestamp: {self.timestamp.isoformat()}")
        self.log(f"Hash: {self.hash}")
        print()

        success = True

        # 1. Update requirements.txt
        if not self.update_requirements_txt():
            success = False

        # 2. Update app.py version
        self.update_app_version()

        # 3. Create deployment marker
        self.create_deployment_marker()

        print()

        # 4. Git operations
        if commit:
            self.log("Committing changes...")
            if not self.run_git_operations(push=push):
                success = False

        # Summary
        self.log("=" * 80)
        if success:
            self.log("✅ Cache-busting complete!", "SUCCESS")

            if commit:
                self.log("Changes have been committed", "INFO")
                if push:
                    self.log("Changes have been pushed to remote", "INFO")
                    self.log(
                        "Hugging Face Space should rebuild automatically",
                        "INFO",
                    )
                else:
                    self.log("Run 'git push' to deploy to Hugging Face Space", "INFO")
            else:
                self.log("Run with --commit to create a git commit", "INFO")
        else:
            self.log("❌ Some operations failed", "ERROR")

        self.log("=" * 80)

        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Force cache-busting for HF Spaces")
    parser.add_argument(
        "--commit",
        "-c",
        action="store_true",
        help="Commit changes to git",
    )
    parser.add_argument(
        "--push",
        "-p",
        action="store_true",
        help="Push changes to remote (implies --commit)",
    )
    parser.add_argument(
        "--space-dir",
        "-d",
        type=Path,
        default=Path.cwd(),
        help="Path to Hugging Face Space directory",
    )

    args = parser.parse_args()

    # If push is specified, commit is implied
    if args.push:
        args.commit = True

    buster = CacheBuster(args.space_dir)
    success = buster.run_full_cache_bust(commit=args.commit, push=args.push)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
