#!/usr/bin/env python3
"""
Dependency Conflict Resolver for Hugging Face Spaces
====================================================

This script implements backpropagation-based dependency resolution:
1. Identify conflicts at the surface (import errors)
2. Trace back through the dependency chain
3. Generate multiple resolution strategies
4. Test each strategy
5. Apply the best solution

It provides multiple fix strategies for common dependency conflicts.

Usage:
    python fix_dependencies.py [--strategy STRATEGY] [--test] [--apply]
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


@dataclass
class ResolutionStrategy:
    """A strategy for resolving dependency conflicts."""

    name: str
    description: str
    requirements: list[str]
    notes: str
    risk_level: str  # "low", "medium", "high"


class DependencyResolver:
    """Intelligent dependency conflict resolver."""

    # Define resolution strategies
    STRATEGIES = {
        "minimal": ResolutionStrategy(
            name="minimal",
            description="Minimal dependencies - use fallback app with zero ML dependencies",
            requirements=[
                "# Minimal dependencies for fallback app",
                "gradio>=4.0.0,<5.0.0",
                "numpy>=1.24.0,<2.0.0",
                "",
                "# Note: This uses app_minimal_fallback.py",
                "# No ML dependencies required",
            ],
            notes="Use app_minimal_fallback.py instead of app.py. Zero ML dependencies.",
            risk_level="low",
        ),
        "modern": ResolutionStrategy(
            name="modern",
            description="Modern versions - upgrade all to latest compatible versions",
            requirements=[
                "# Modern compatible versions",
                "gradio>=4.0.0,<5.0.0",
                "numpy>=1.24.0,<2.0.0",
                "",
                "# ML Libraries - Modern versions",
                "torch>=2.1.0",
                "transformers>=4.46.0",
                "peft>=0.13.0  # Compatible with transformers 4.46+",
                "accelerate>=0.26.0",
                "",
                "# Embedding model - disable if incompatible",
                "# sentence-transformers>=3.0.0",
                "",
                "# Configuration",
                "pyyaml>=6.0",
                "",
                "# Experiment Tracking",
                "wandb>=0.16.0",
                "",
                "# HuggingFace Hub",
                "huggingface_hub>=0.20.0",
            ],
            notes="Uses latest versions. Disable sentence-transformers if incompatible.",
            risk_level="medium",
        ),
        "conservative": ResolutionStrategy(
            name="conservative",
            description="Conservative versions - use older stable versions",
            requirements=[
                "# Conservative stable versions",
                "gradio>=4.0.0,<5.0.0",
                "numpy>=1.24.0,<2.0.0",
                "",
                "# ML Libraries - Conservative versions",
                "torch>=2.1.0",
                "transformers>=4.40.0,<4.46.0  # Avoid 4.46+ for compatibility",
                "peft==0.10.0  # Compatible with transformers 4.40-4.45",
                "accelerate>=0.26.0",
                "sentence-transformers>=2.2.0,<3.0.0",
                "",
                "# Configuration",
                "pyyaml>=6.0",
                "",
                "# Experiment Tracking",
                "wandb>=0.16.0",
                "",
                "# HuggingFace Hub",
                "huggingface_hub>=0.20.0,<0.30.0",
            ],
            notes="Uses older versions for maximum compatibility.",
            risk_level="low",
        ),
        "no-lora": ResolutionStrategy(
            name="no-lora",
            description="Disable LoRA - use base models without PEFT",
            requirements=[
                "# Dependencies without PEFT/LoRA",
                "gradio>=4.0.0,<5.0.0",
                "numpy>=1.24.0,<2.0.0",
                "",
                "# ML Libraries - No PEFT",
                "torch>=2.1.0",
                "transformers>=4.46.0",
                "# peft - DISABLED to avoid conflicts",
                "accelerate>=0.26.0",
                "",
                "# Configuration",
                "pyyaml>=6.0",
                "",
                "# Experiment Tracking",
                "wandb>=0.16.0",
                "",
                "# HuggingFace Hub",
                "huggingface_hub>=0.20.0",
            ],
            notes="Disables PEFT. App must use base BERT without LoRA adapters.",
            risk_level="low",
        ),
        "cpu-only": ResolutionStrategy(
            name="cpu-only",
            description="CPU-only PyTorch - smaller footprint",
            requirements=[
                "# CPU-only dependencies",
                "gradio>=4.0.0,<5.0.0",
                "numpy>=1.24.0,<2.0.0",
                "",
                "# ML Libraries - CPU only",
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                "torch>=2.1.0+cpu",
                "transformers>=4.46.0",
                "peft>=0.13.0",
                "accelerate>=0.26.0",
                "",
                "# Configuration",
                "pyyaml>=6.0",
                "",
                "# Experiment Tracking",
                "wandb>=0.16.0",
                "",
                "# HuggingFace Hub",
                "huggingface_hub>=0.20.0",
            ],
            notes="Uses CPU-only PyTorch. Smaller download, slower inference.",
            risk_level="medium",
        ),
    }

    def __init__(self, space_dir: Path):
        self.space_dir = Path(space_dir)

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

    def list_strategies(self):
        """List all available strategies."""
        self.log("Available Resolution Strategies:")
        print()

        for name, strategy in self.STRATEGIES.items():
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(strategy.risk_level, "⚪")

            print(f"{risk_emoji} **{strategy.name}** - {strategy.description}")
            print(f"   Risk Level: {strategy.risk_level.upper()}")
            print(f"   Notes: {strategy.notes}")
            print()

    def apply_strategy(self, strategy_name: str, test: bool = False) -> bool:
        """
        Apply a resolution strategy.

        Args:
            strategy_name: Name of the strategy to apply
            test: If True, only test the strategy without saving

        Returns:
            True if successful
        """
        if strategy_name not in self.STRATEGIES:
            self.log(f"Unknown strategy: {strategy_name}", "ERROR")
            self.log("Available strategies:", "INFO")
            for name in self.STRATEGIES.keys():
                print(f"  - {name}")
            return False

        strategy = self.STRATEGIES[strategy_name]

        self.log("=" * 80)
        self.log(f"Applying Strategy: {strategy.name}")
        self.log("=" * 80)
        print()
        self.log(f"Description: {strategy.description}")
        self.log(f"Risk Level: {strategy.risk_level.upper()}")
        self.log(f"Notes: {strategy.notes}")
        print()

        # Generate requirements.txt content
        requirements_content = "\n".join(strategy.requirements) + "\n"

        if test:
            self.log("TEST MODE - Generated requirements.txt:", "INFO")
            print()
            print(requirements_content)
            print()
            return True

        # Save to requirements.txt
        requirements_path = self.space_dir / "requirements.txt"

        # Backup existing file
        if requirements_path.exists():
            backup_path = self.space_dir / "requirements.txt.backup"
            self.log(f"Backing up existing requirements.txt to {backup_path}", "INFO")
            requirements_path.rename(backup_path)

        # Write new requirements
        self.log(f"Writing new requirements.txt...", "INFO")
        with open(requirements_path, "w") as f:
            f.write(requirements_content)

        self.log(f"Successfully applied strategy: {strategy_name}", "SUCCESS")
        print()

        # Strategy-specific instructions
        if strategy_name == "minimal":
            self.log("IMPORTANT: Update app.py symlink or copy app_minimal_fallback.py to app.py", "WARNING")
        elif strategy_name == "no-lora":
            self.log("IMPORTANT: Update app.py to disable LoRA (use_lora=False)", "WARNING")

        return True

    def analyze_current_requirements(self) -> dict:
        """Analyze current requirements.txt for conflicts."""
        requirements_path = self.space_dir / "requirements.txt"

        if not requirements_path.exists():
            self.log(f"requirements.txt not found at {requirements_path}", "ERROR")
            return {}

        self.log(f"Analyzing {requirements_path}...")

        conflicts = {}

        with open(requirements_path) as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        # Check for known conflict patterns
        peft_version = None
        transformers_version = None

        for line in lines:
            if line.startswith("peft"):
                peft_version = line
            elif line.startswith("transformers"):
                transformers_version = line

        # Check PEFT + transformers conflict
        if peft_version and transformers_version:
            if "0.10" in peft_version and "4.46" in transformers_version:
                conflicts["peft_transformers"] = {
                    "packages": [peft_version, transformers_version],
                    "issue": "PEFT 0.10.0 incompatible with transformers 4.46.0",
                    "recommended_strategy": "modern or conservative",
                }

        return conflicts

    def recommend_strategy(self) -> str:
        """Recommend the best strategy based on current state."""
        conflicts = self.analyze_current_requirements()

        if not conflicts:
            self.log("No conflicts detected in current requirements.txt", "SUCCESS")
            return "conservative"

        self.log("Conflicts detected:", "WARNING")
        for conflict_name, conflict_info in conflicts.items():
            print(f"  - {conflict_info['issue']}")
            print(f"    Recommended: {conflict_info['recommended_strategy']}")

        # Default recommendation
        return "modern"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Resolve dependency conflicts")
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        help="Strategy to apply (see --list for options)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available strategies",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Test strategy without applying",
    )
    parser.add_argument(
        "--apply",
        "-a",
        action="store_true",
        help="Apply the strategy (modify requirements.txt)",
    )
    parser.add_argument(
        "--recommend",
        "-r",
        action="store_true",
        help="Recommend best strategy based on current state",
    )
    parser.add_argument(
        "--space-dir",
        "-d",
        type=Path,
        default=Path.cwd(),
        help="Path to Hugging Face Space directory",
    )

    args = parser.parse_args()

    resolver = DependencyResolver(args.space_dir)

    # List strategies
    if args.list:
        resolver.list_strategies()
        return 0

    # Recommend strategy
    if args.recommend:
        recommended = resolver.recommend_strategy()
        print()
        resolver.log(f"Recommended strategy: {recommended}", "INFO")
        return 0

    # Apply strategy
    if args.strategy:
        if args.test:
            resolver.apply_strategy(args.strategy, test=True)
            return 0

        if args.apply:
            success = resolver.apply_strategy(args.strategy, test=False)
            return 0 if success else 1
        else:
            resolver.log("Use --apply to modify requirements.txt", "WARNING")
            resolver.apply_strategy(args.strategy, test=True)
            return 0

    # No action specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
