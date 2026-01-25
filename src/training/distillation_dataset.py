"""
Distillation Dataset for Neural Agent Bootstrapping.

Loads and processes LLM-generated training examples for:
- HRM (Hierarchical Reasoning Module)
- TRM (Tiny Recursive Model)
- Meta-Controller (Routing)
- Policy-Value Networks

Best Practices:
- 2025 Standards: Type hints, strict validation, modular design
- Dynamic Loading: Supports JSONL streaming for large datasets
- Flexible Mapping: Decouples source data format from model inputs
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.framework.mcts.llm_guided.data_collector import EpisodeMetadata, TrainingExample
from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)


class DistillationTask(str, Enum):
    """Supported distillation task types."""

    POLICY_VALUE = "policy_value"
    HRM_DECOMPOSITION = "hrm_decomposition"
    TRM_REFINEMENT = "trm_refinement"
    META_CONTROLLER = "meta_controller"


class DistillationDataset(Dataset):
    """
    Dataset for distilling LLM knowledge into neural models.

    Attributes:
        data_paths (list[Path]): List of paths to JSONL data files.
        task_type (DistillationTask): The specific distillation task.
        transform (Callable): Optional transform to apply to samples.
        max_samples (int | None): Limit on number of samples to load.
    """

    def __init__(
        self,
        data_dir: str | Path,
        task_type: DistillationTask,
        transform: Callable | None = None,
        file_pattern: str = "episode_*.jsonl",
        max_samples: int | None = None,
    ):
        """
        Initialize the distillation dataset.

        Args:
            data_dir: Directory containing training data files.
            task_type: The type of distillation task.
            transform: Optional transform function.
            file_pattern: Glob pattern to match data files.
            max_samples: Optional limit on total samples.
        """
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.transform = transform
        self.max_samples = max_samples

        self.examples: list[TrainingExample] = []
        self.metadata: list[EpisodeMetadata] = []
        self.episode_agent_map: dict[str, str] = {}

        self._load_data(file_pattern)

    def _load_data(self, file_pattern: str) -> None:
        """
        Load data from JSONL files matching the pattern.

        Args:
            file_pattern: Glob pattern to match files.
        """
        files = list(self.data_dir.glob(file_pattern))
        if not files:
            logger.warning("No training data found", data_dir=str(self.data_dir), pattern=file_pattern)
            return

        logger.info(
            "Loading distillation data", task_type=self.task_type, num_files=len(files), max_samples=self.max_samples
        )

        loaded_count = 0

        for file_path in files:
            if self.max_samples and loaded_count >= self.max_samples:
                break

            try:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        if self.max_samples and loaded_count >= self.max_samples:
                            break

                        data = json.loads(line)

                        # Handle metadata vs example
                        if "_metadata" in data:
                            try:
                                meta = EpisodeMetadata(**data["_metadata"])
                                self.metadata.append(meta)
                                if hasattr(meta, "agent_strategy"):
                                    self.episode_agent_map[meta.episode_id] = meta.agent_strategy
                            except Exception as e:
                                logger.debug(f"Skipping malformed metadata in {file_path}: {e}")
                        else:
                            try:
                                example = TrainingExample.from_dict(data)

                                # Filter relevant examples based on task
                                if self._is_relevant_for_task(example):
                                    self.examples.append(example)
                                    loaded_count += 1

                            except Exception as e:
                                logger.debug(f"Skipping malformed example in {file_path}: {e}")

            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")

        logger.info("Distillation data loaded", total_samples=len(self.examples), task_type=self.task_type)

    def _is_relevant_for_task(self, example: TrainingExample) -> bool:
        """
        Filter examples relevant to the specific distillation task.

        Args:
            example: Candidate training example.

        Returns:
            True if valid for the current task.
        """
        if self.task_type == DistillationTask.POLICY_VALUE:
            # Needs valid MCTS action probabilities (teacher)
            return bool(example.mcts_action_probs)

        elif self.task_type == DistillationTask.HRM_DECOMPOSITION:
            # Placeholder: Check if example has decomposition data
            # Typically stored in `test_results` or specific metadata
            # For now, we assume all code generation tasks are relevant
            return True

        elif self.task_type == DistillationTask.TRM_REFINEMENT:
            # Needs to be part of a refinement chain
            # Check if it has parent info or refinement iteration metadata
            return example.depth > 0 or example.action == "refine"

        elif self.task_type == DistillationTask.META_CONTROLLER:
            # Needs to have routing info or outcome
            return True

        return False

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Any]:
        """
        Get a training sample tailored to the specific task.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing model inputs and targets.
        """
        example = self.examples[idx]

        if self.task_type == DistillationTask.POLICY_VALUE:
            return self._prepare_policy_value_sample(example)

        elif self.task_type == DistillationTask.HRM_DECOMPOSITION:
            return self._prepare_hrm_sample(example)

        elif self.task_type == DistillationTask.TRM_REFINEMENT:
            return self._prepare_trm_sample(example)

        elif self.task_type == DistillationTask.META_CONTROLLER:
            return self._prepare_meta_controller_sample(example)

        raise ValueError(f"Unknown task type: {self.task_type}")

    # =========================================================================
    # Task-Specific Preparation Methods
    # =========================================================================

    def _prepare_policy_value_sample(self, example: TrainingExample) -> dict[str, Any]:
        """
        Prepare sample for Policy-Value network training.

        Mapping:
        - Input: State code/problem representation
        - Target Policy: MCTS visit distribution (teacher)
        - Target Value: Final episode outcome
        """
        # Note: Actual encoding to tensors assumes existence of a tokenizer/encoder
        # This function returns raw data or pre-processed features for collating

        return {
            "state_code": example.state_code,
            "state_problem": example.state_problem,
            "target_policy": example.mcts_action_probs,
            "target_value": example.outcome,
        }

    def _prepare_hrm_sample(self, example: TrainingExample) -> dict[str, Any]:
        """
        Prepare sample for HRM (Decomposition) training.

        Mapping:
        - Input: Problem statement
        - Target: Sub-problem decomposition (from LLM trace)
        """
        # Hypothetical: extracting decomposition from test_results or specialized field
        # In a real implementation, we'd parse the LLM reasoning chain
        decomposition = example.test_results.get("decomposition", []) if example.test_results else []

        return {"problem": example.state_problem, "target_decomposition": decomposition}

    def _prepare_trm_sample(self, example: TrainingExample) -> dict[str, Any]:
        """
        Prepare sample for TRM (Refinement) training.

        Mapping:
        - Input: Critical state / buggy code
        - Target: Refined code / fix
        """
        return {
            "initial_code": example.state_code,
            "target_code": example.test_results.get("refined_code", example.state_code)
            if example.test_results
            else example.state_code,
            "success": example.outcome > 0,
        }

    def _prepare_meta_controller_sample(self, example: TrainingExample) -> dict[str, Any]:
        """
        Prepare sample for Meta-Controller training.

        Mapping:
        - Input: Meta-features (confidence, complexity, history)
        - Target: Success/Failure signal for the chosen agent strategy
        """
        return {
            "features": {
                "visits": example.visits,
                "depth": example.depth,
                "llm_confidence": example.llm_value_estimate,
                "q_value": example.q_value,
            },
            "outcome": example.outcome,
            "agent_strategy": self.episode_agent_map.get(example.episode_id, "LLM_MCTS"),
        }
