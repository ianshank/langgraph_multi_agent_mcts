"""
Repository for SQL Model Registry operations.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from datetime import timezone

from sqlalchemy import select, delete, create_engine
from sqlalchemy.orm import Session, sessionmaker, joinedload

from src.training.registry.models import Base, ModelVersionDB, ModelTagDB
from src.observability.logging import get_structured_logger

logger = get_structured_logger(__name__)

class ModelRegistryRepository:
    """Handles database operations for the Model Registry."""

    def __init__(self, db_url: str = "sqlite:///./registry.db"):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        Base.metadata.create_all(self.engine)

    def register_model_version(
        self,
        version_id: str,
        model_type: str,
        filepath: str,
        metrics: dict,
        sub_components: dict,
        tags: list[str]
    ) -> ModelVersionDB:
        """Create a new model version record."""
        with self.SessionLocal() as session:
            version = ModelVersionDB(
                version_id=version_id,
                model_type=model_type,
                filepath=filepath,
                metrics=metrics,
                sub_components=sub_components,
                created_at=datetime.now(timezone.utc)
            )
            
            for tag_name in tags:
                version.tags.append(ModelTagDB(tag=tag_name))
            
            session.add(version)
            session.commit()
            session.refresh(version)
            return version

    def get_version(self, version_id: str) -> Optional[ModelVersionDB]:
        """Get a specific model version."""
        with self.SessionLocal() as session:
            stmt = select(ModelVersionDB).options(joinedload(ModelVersionDB.tags)).where(ModelVersionDB.version_id == version_id)
            return session.scalar(stmt)

    def get_best_version(self, model_type: str) -> Optional[ModelVersionDB]:
        """Get the version tagged 'best' for a specific model type."""
        with self.SessionLocal() as session:
            stmt = (
                select(ModelVersionDB)
                .options(joinedload(ModelVersionDB.tags))
                .join(ModelVersionDB.tags)
                .where(ModelVersionDB.model_type == model_type)
                .where(ModelTagDB.tag == "best")
            )
            return session.scalar(stmt)

    def demote_current_best(self, model_type: str):
        """Remove 'best' tag from current best model of type."""
        with self.SessionLocal() as session:
            stmt = (
                delete(ModelTagDB)
                .where(ModelTagDB.tag == "best")
                .where(ModelTagDB.version_id.in_(
                    select(ModelVersionDB.version_id)
                    .where(ModelVersionDB.model_type == model_type)
                ))
            )
            session.execute(stmt)
            session.commit()

    def add_tag(self, version_id: str, tag: str):
        """Add a tag to a model version."""
        with self.SessionLocal() as session:
            version = session.get(ModelVersionDB, version_id)
            if version:
                # Check exist
                exists = any(t.tag == tag for t in version.tags)
                if not exists:
                    version.tags.append(ModelTagDB(tag=tag))
                    session.commit()
