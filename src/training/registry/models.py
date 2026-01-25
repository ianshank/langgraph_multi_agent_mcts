"""
SQLAlchemy models for the Model Registry.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ModelVersionDB(Base):
    """Database model for a model version."""

    __tablename__ = "model_versions"

    version_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_type: Mapped[str] = mapped_column(String, index=True)  # hrm, trm, etc.
    filepath: Mapped[str] = mapped_column(String)  # Relative path to artifact
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Validation metrics (loss, win_rate, etc.)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default={})

    # Sub-components map (e.g. {"decoder": "decoder_v1.pt"})
    sub_components: Mapped[dict[str, str]] = mapped_column(JSON, default={})

    # Relationships
    tags: Mapped[list["ModelTagDB"]] = relationship(
        "ModelTagDB", back_populates="model_version", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_type": self.model_type,
            "filepath": self.filepath,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "sub_components": self.sub_components,
            "tags": [t.tag for t in self.tags],
        }


class ModelTagDB(Base):
    """Tags for model versions (e.g., 'best', 'latest', 'candidate')."""

    __tablename__ = "model_tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.version_id"))
    tag: Mapped[str] = mapped_column(String, index=True)

    model_version: Mapped["ModelVersionDB"] = relationship(back_populates="tags")
