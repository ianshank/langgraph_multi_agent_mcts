"""
Continuous Learning Manager.

Orchestrates the lifecycle of:
1. Data Collection Monitoring
2. Model Training (Distillation)
3. Model Evaluation
4. Model Promotion (Registry Update)
"""

from __future__ import annotations

import time
from pathlib import Path

from src.framework.mcts.llm_guided.data_collector import TrainingDataCollector
from src.observability.logging import get_structured_logger
from src.training.distillation_orchestrator import DistillationOrchestrator
from src.training.model_registry import ModelRegistry
from src.training.system_config import SystemConfig

logger = get_structured_logger(__name__)


class ContinuousLearningManager:
    """
    Automates the self-improvement loop.
    """

    def __init__(
        self,
        config: SystemConfig,
        data_dir: str | Path,
        registry_dir: str | Path,
        min_samples_for_training: int = 100,
        training_interval_seconds: int = 3600,
    ):
        self.config = config
        self.data_dir = Path(data_dir)

        self.registry = ModelRegistry(registry_dir)
        self.orchestrator = DistillationOrchestrator(config, data_dir, registry_dir / "temp_builds")
        self.data_collector = TrainingDataCollector(output_dir=data_dir)

        self.min_samples = min_samples_for_training
        self.interval = training_interval_seconds

        self.last_trained_count = 0
        self.is_running = False

    def start_loop(self):
        """Start the continuous learning loop (blocking)."""
        logger.info("Starting Continuous Learning Loop")
        self.is_running = True

        while self.is_running:
            try:
                self.check_and_train()
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")

            time.sleep(self.interval)

    def check_and_train(self):
        """Check availability of data and trigger training."""
        stats = self.data_collector.get_statistics()
        total_examples = stats["total_examples"]

        new_samples = total_examples - self.last_trained_count

        logger.info(f"Data Monitor: {new_samples} new samples (Total: {total_examples})")

        if new_samples >= self.min_samples:
            logger.info("Threshold reached. Triggering distillation...")
            self.run_distillation_cycle()
            self.last_trained_count = total_examples
        else:
            logger.info(f"Waiting for {self.min_samples - new_samples} more samples.")

    def run_distillation_cycle(self):
        """Run full distillation and promotion cycle."""

        # 1. Distill HRM
        try:
            metrics_hrm = self.orchestrator.distill_hrm(num_epochs=5)
            if metrics_hrm:
                self._register_and_promote(
                    model_name="hrm_agent",
                    metrics=metrics_hrm,
                    model_type="hrm",
                    sub_components={"decoder": "system_decoder.pt"},
                )
        except Exception as e:
            logger.error(f"HRM Distillation failed: {e}")

        # 2. Distill TRM
        try:
            metrics_trm = self.orchestrator.distill_trm(num_epochs=5)
            if metrics_trm:
                self._register_and_promote(
                    model_name="trm_agent",
                    metrics=metrics_trm,
                    model_type="trm",
                    sub_components={"decoder": "system_decoder.pt"},
                )
        except Exception as e:
            logger.error(f"TRM Distillation failed: {e}")

        # 3. Distill Meta-Controller
        try:
            metrics_meta = self.orchestrator.distill_meta_controller(num_epochs=5)
            if metrics_meta:
                self._register_and_promote(
                    model_name="meta_controller", metrics=metrics_meta, model_type="meta_controller"
                )
        except Exception as e:
            logger.error(f"Meta-Controller Distillation failed: {e}")

    def _register_and_promote(
        self, model_name: str, metrics: dict[str, float], model_type: str, sub_components: dict[str, str] | None = None
    ):
        """Register model and promote if validation loss improves."""
        # Simple promotion logic: Always promote if it trains successfully (Validation Loss < X?)
        # Better: Compare with previous best.

        built_path = self.orchestrator.output_dir / f"{model_name}.pt"
        # Meta-Controller uses save_pretrained (directory)
        if not built_path.exists():
            # Check if directory
            built_path = self.orchestrator.output_dir / model_name
            if not built_path.exists():
                logger.warning(f"Artifact {built_path} not found.")
                return

        # Prepare sub components paths
        subs = {}
        if sub_components:
            for key, filename in sub_components.items():
                path = self.orchestrator.output_dir / filename
                if path.exists():
                    subs[key] = path

        # Register
        current_loss = metrics.get("avg_loss", float("inf"))

        version_id = self.registry.register_model(
            source_path=built_path, model_type=model_type, metrics=metrics, sub_components=subs, tags=["candidate"]
        )

        # Compare with previous best
        best_version = self.registry.get_best_model_version(model_type)
        if best_version:
            best_loss = best_version.metrics.get("avg_loss", float("inf"))
            if current_loss < best_loss:
                logger.info(f"New model {version_id} is better ({current_loss:.4f} < {best_loss:.4f}). Promoting.")
                self.registry.promote_to_best(version_id)
            else:
                logger.info(
                    f"New model {version_id} ({current_loss:.4f}) not better than {best_version.version_id} ({best_loss:.4f})."
                )
        else:
            # First model
            logger.info(f"First model {version_id}. Promoting to best.")
            self.registry.promote_to_best(version_id)
