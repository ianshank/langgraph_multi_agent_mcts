"""
Model Registry for Continuous Learning.

Manages model versions, metadata, and "best" model pointers.
Now backed by SQL Repository.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.observability.logging import get_structured_logger
from src.training.registry.models import ModelVersionDB
from src.training.registry.repository import ModelRegistryRepository

logger = get_structured_logger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a specific model version."""

    version_id: str
    model_type: str
    filepath: str
    sub_components: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_db(cls, db_model: ModelVersionDB) -> ModelVersion:
        """Convert DB model to dataclass."""
        return cls(
            version_id=db_model.version_id,
            model_type=db_model.model_type,
            filepath=db_model.filepath,
            sub_components=db_model.sub_components,
            created_at=db_model.created_at.isoformat(),
            metrics=db_model.metrics,
            tags=[t.tag for t in db_model.tags],
        )


class ModelRegistry:
    """
    Manages model artifacts and versions.
    Wrapper around SQL Repository.
    """

    def __init__(self, registry_dir: str | Path, db_url: str = "sqlite:///./registry.db"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.repo = ModelRegistryRepository(db_url)

    def register_model(
        self,
        source_path: str | Path,
        model_type: str,
        metrics: dict[str, float],
        sub_components: dict[str, str | Path] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """
        Register a new model version.
        Moves/Copies the model file into the registry.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source model not found: {source_path}")

        # Generate Version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_type}_{timestamp}"

        # Target Directory
        target_dir = self.registry_dir / model_type / version_id
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy Main Artifact
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)

        # Copy Sub-components
        stored_subs = {}
        if sub_components:
            for key, path in sub_components.items():
                p = Path(path)
                if p.exists():
                    sub_target = target_dir / p.name
                    shutil.copy2(p, sub_target)
                    stored_subs[key] = str(sub_target.relative_to(self.registry_dir))

        # Register in DB
        rel_path = str(target_path.relative_to(self.registry_dir))
        self.repo.register_model_version(
            version_id=version_id,
            model_type=model_type,
            filepath=rel_path,
            metrics=metrics,
            sub_components=stored_subs,
            tags=tags or ["latest"],
        )

        logger.info(f"Registered model {version_id}", metrics=metrics)
        return version_id

    def get_best_model_version(self, model_type: str) -> ModelVersion | None:
        """Get the version tagged 'best' for a type."""
        db_model = self.repo.get_best_version(model_type)
        if db_model:
            return ModelVersion.from_db(db_model)
        return None

    def get_latest_model_version(self, model_type: str) -> ModelVersion | None:
        """Get the most recently created version for a type."""
        # Not efficiently implemented in repo yet, assuming usage pattern relies on explicit get
        # For now return None as Phase 4 focuses on explicitly managed deployments
        return None

    def promote_to_best(self, version_id: str) -> None:
        """Mark a version as 'best', demoting previous best."""
        version = self.repo.get_version(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        self.repo.demote_current_best(version.model_type)
        self.repo.add_tag(version_id, "best")

        logger.info(f"Promoted {version_id} to BEST")

    def get_model_path(self, version_id: str) -> Path:
        """Get absolute path to model file."""
        version = self.repo.get_version(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")
        return self.registry_dir / version.filepath
