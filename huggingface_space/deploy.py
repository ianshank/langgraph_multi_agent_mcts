#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Deployment Script for Hugging Face Spaces
================================================

This script orchestrates the complete deployment workflow:
1. Verify dependencies
2. Resolve conflicts
3. Cache-bust
4. Deploy

It implements intelligent backpropagation-based error handling.

Usage:
    python deploy.py [--strategy STRATEGY] [--verify-only] [--auto]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class DeploymentOrchestrator:
    """Orchestrates the complete deployment workflow."""

    def __init__(self, space_dir: Path, verbose: bool = False):
        self.space_dir = Path(space_dir)
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        prefix_map = {
            "ERROR": "❌",
            "WARNING": "⚠️",
            "SUCCESS": "✅",
            "INFO": "ℹ️",
            "STEP": "🔧",
        }
        prefix = prefix_map.get(level, "ℹ️")
        print(f"{prefix} {message}")

    def run_script(self, script: str, args: list[str]) -> tuple[bool, str]:
        """
        Run a Python script and return (success, output).
        """
        script_path = self.space_dir / script

        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False, ""

        cmd = [sys.executable, str(script_path)] + args

        try:
            result = subprocess.run(
                cmd,
                cwd=self.space_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if self.verbose:
                print(result.stdout)

            if result.returncode != 0:
                print(result.stderr)
                return False, result.stderr

            return True, result.stdout

        except subprocess.TimeoutExpired:
            self.log(f"Script timeout: {script}", "ERROR")
            return False, "Timeout"
        except Exception as e:
            self.log(f"Script error: {e}", "ERROR")
            return False, str(e)

    def step_verify(self) -> bool:
        """Step 1: Verify current state."""
        self.log("=" * 80)
        self.log("STEP 1: Verifying Deployment")
        self.log("=" * 80)

        success, output = self.run_script("verify_deployment.py", ["--verbose"])

        if success:
            self.log("Verification passed", "SUCCESS")
            return True
        else:
            self.log("Verification found issues", "WARNING")
            self.warnings.append("Verification issues detected")
            return False

    def step_fix_dependencies(self, strategy: str) -> bool:
        """Step 2: Fix dependency conflicts."""
        self.log("=" * 80)
        self.log("STEP 2: Resolving Dependencies")
        self.log("=" * 80)

        if not strategy:
            # Get recommendation
            self.log("Getting strategy recommendation...")
            success, output = self.run_script("fix_dependencies.py", ["--recommend"])

            if "Recommended strategy:" in output:
                # Parse recommendation
                for line in output.split("\n"):
                    if "Recommended strategy:" in line:
                        strategy = line.split(":")[-1].strip()
                        break

            if not strategy:
                strategy = "conservative"

        self.log(f"Using strategy: {strategy}", "INFO")

        # Apply strategy
        success, output = self.run_script(
            "fix_dependencies.py", ["--strategy", strategy, "--apply"]
        )

        if success:
            self.log(f"Applied {strategy} strategy", "SUCCESS")
            return True
        else:
            self.log("Failed to apply strategy", "ERROR")
            self.errors.append("Strategy application failed")
            return False

    def step_cache_bust(self, push: bool = False) -> bool:
        """Step 3: Force cache-busting."""
        self.log("=" * 80)
        self.log("STEP 3: Cache-Busting")
        self.log("=" * 80)

        args = ["--commit"]
        if push:
            args.append("--push")

        success, output = self.run_script("force_cache_bust.py", args)

        if success:
            self.log("Cache-bust complete", "SUCCESS")
            return True
        else:
            self.log("Cache-bust failed", "ERROR")
            self.errors.append("Cache-bust failed")
            return False

    def step_final_verification(self) -> bool:
        """Step 4: Final verification before deployment."""
        self.log("=" * 80)
        self.log("STEP 4: Final Verification")
        self.log("=" * 80)

        success, output = self.run_script("verify_deployment.py", [])

        if success:
            self.log("Final verification passed", "SUCCESS")
            return True
        else:
            self.log("Final verification failed", "WARNING")
            self.warnings.append("Final verification issues")
            return False

    def interactive_strategy_selection(self) -> str:
        """Interactively select a deployment strategy."""
        self.log("=" * 80)
        self.log("Strategy Selection")
        self.log("=" * 80)

        strategies = {
            "1": ("minimal", "Zero ML dependencies (fastest, most reliable)"),
            "2": ("conservative", "Stable versions (recommended for full features)"),
            "3": ("modern", "Latest versions (may have compatibility issues)"),
            "4": ("no-lora", "Disable LoRA adapters"),
            "5": ("cpu-only", "CPU-only PyTorch"),
        }

        print("\nAvailable strategies:")
        for key, (name, desc) in strategies.items():
            print(f"  {key}. {name:15s} - {desc}")

        print()
        choice = input("Select strategy (1-5) or press Enter for recommendation: ").strip()

        if choice in strategies:
            strategy = strategies[choice][0]
            self.log(f"Selected: {strategy}", "INFO")
            return strategy
        else:
            self.log("Using automatic recommendation", "INFO")
            return ""

    def run_deployment(
        self,
        strategy: Optional[str] = None,
        verify_only: bool = False,
        push: bool = False,
        auto: bool = False,
    ) -> bool:
        """
        Run the complete deployment workflow.

        Args:
            strategy: Dependency resolution strategy to use
            verify_only: Only run verification, don't deploy
            push: Push changes to remote
            auto: Non-interactive mode

        Returns:
            True if successful
        """
        self.log("=" * 80)
        self.log("Hugging Face Space Deployment Orchestrator")
        self.log("=" * 80)
        self.log(f"Space Directory: {self.space_dir}")
        print()

        # Step 1: Initial verification
        verification_passed = self.step_verify()

        if verify_only:
            return verification_passed

        print()

        # Interactive strategy selection if not auto mode
        if not auto and not strategy:
            strategy = self.interactive_strategy_selection()
            print()

        # Step 2: Fix dependencies
        if not self.step_fix_dependencies(strategy or ""):
            self.log("Dependency resolution failed", "ERROR")
            return False

        print()

        # Step 3: Cache-bust
        if not self.step_cache_bust(push=push):
            self.log("Cache-busting failed", "ERROR")
            return False

        print()

        # Step 4: Final verification
        final_verification_passed = self.step_final_verification()

        print()

        # Summary
        self.log("=" * 80)
        self.log("Deployment Summary")
        self.log("=" * 80)

        if self.errors:
            self.log(f"Errors: {len(self.errors)}", "ERROR")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            self.log(f"Warnings: {len(self.warnings)}", "WARNING")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and final_verification_passed:
            self.log("Deployment ready!", "SUCCESS")
            print()

            if push:
                self.log("Changes pushed to Hugging Face Space", "SUCCESS")
                self.log(
                    "Monitor build at: https://huggingface.co/spaces/ianshank/langgraph-mcts-demo",
                    "INFO",
                )
            else:
                self.log("Run with --push to deploy to Hugging Face Space", "INFO")

            return True
        else:
            self.log("Deployment has issues", "WARNING")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy to Hugging Face Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify only
  python deploy.py --verify-only

  # Interactive deployment
  python deploy.py

  # Auto deployment with specific strategy
  python deploy.py --strategy minimal --auto --push

  # Deploy with recommendation
  python deploy.py --auto --push
        """,
    )

    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        help="Dependency resolution strategy (minimal, conservative, modern, no-lora, cpu-only)",
    )
    parser.add_argument(
        "--verify-only",
        "-v",
        action="store_true",
        help="Only run verification, don't deploy",
    )
    parser.add_argument(
        "--push",
        "-p",
        action="store_true",
        help="Push changes to remote (deploy to HF Space)",
    )
    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help="Non-interactive mode (use defaults/recommendations)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--space-dir",
        "-d",
        type=Path,
        default=Path.cwd(),
        help="Path to Hugging Face Space directory",
    )

    args = parser.parse_args()

    orchestrator = DeploymentOrchestrator(args.space_dir, verbose=args.verbose)

    success = orchestrator.run_deployment(
        strategy=args.strategy,
        verify_only=args.verify_only,
        push=args.push,
        auto=args.auto,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
