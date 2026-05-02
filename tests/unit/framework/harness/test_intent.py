"""Unit tests for intent normalisation and SPEC parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.framework.harness import HarnessSettings
from src.framework.harness.intent import DefaultIntentNormalizer, SpecLoader, SpecParseError

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_normalize_string_intent() -> None:
    """A string becomes a Task with ``goal`` populated."""
    n = DefaultIntentNormalizer()
    task = await n.normalize("ship the feature", HarnessSettings())
    assert task.goal == "ship the feature"
    assert task.id


@pytest.mark.asyncio
async def test_normalize_dict_intent_with_criteria() -> None:
    """Dict input copies criteria, constraints, and metadata."""
    n = DefaultIntentNormalizer()
    task = await n.normalize(
        {
            "id": "T1",
            "goal": "fix bug",
            "acceptance_criteria": [
                "lints clean",
                {"id": "c2", "description": "tests pass", "check": "passed"},
            ],
            "constraints": ["no new deps"],
            "metadata": {"owner": "alice"},
        },
        HarnessSettings(),
    )
    assert task.id == "T1"
    assert len(task.acceptance_criteria) == 2
    assert task.acceptance_criteria[0].id == "c0"
    assert task.acceptance_criteria[1].check == "passed"
    assert task.constraints == ("no new deps",)
    assert task.metadata == {"owner": "alice"}


@pytest.mark.asyncio
async def test_normalize_rejects_empty() -> None:
    """An empty intent is a programming error."""
    n = DefaultIntentNormalizer()
    with pytest.raises(ValueError):
        await n.normalize("   ", HarnessSettings())
    with pytest.raises(ValueError):
        await n.normalize({"goal": ""}, HarnessSettings())


@pytest.mark.asyncio
async def test_normalize_unsupported_type() -> None:
    """Non-str, non-dict payloads are rejected."""
    n = DefaultIntentNormalizer()
    with pytest.raises(TypeError):
        await n.normalize(42, HarnessSettings())  # type: ignore[arg-type]


def test_spec_loader_parses_frontmatter_and_sections(tmp_path: Path) -> None:
    """Frontmatter, goal, criteria, and constraints are extracted."""
    text = (
        "---\n"
        "owner: alice\n"
        "version: 1\n"
        "---\n"
        "# Goal\n"
        "Add the feature.\n\n"
        "# Acceptance Criteria\n"
        "- ruff clean\n"
        "- tests pass\n\n"
        "# Constraints\n"
        "- no new deps\n"
    )
    spec_file = tmp_path / "spec.md"
    spec_file.write_text(text)
    spec = SpecLoader().load(spec_file)
    assert spec.goal == "Add the feature."
    assert spec.acceptance_criteria == ["ruff clean", "tests pass"]
    assert spec.constraints == ["no new deps"]
    assert spec.frontmatter == {"owner": "alice", "version": "1"}


def test_spec_loader_handles_missing_frontmatter() -> None:
    """No frontmatter is fine — body is parsed as-is."""
    spec = SpecLoader().parse("# Goal\nDo stuff.\n# Acceptance\n- a\n")
    assert spec.goal == "Do stuff."
    assert spec.acceptance_criteria == ["a"]


def test_spec_loader_missing_file_raises(tmp_path: Path) -> None:
    """Missing files raise a clear, typed error."""
    with pytest.raises(SpecParseError):
        SpecLoader().load(tmp_path / "absent.md")
