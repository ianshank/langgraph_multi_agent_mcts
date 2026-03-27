"""
Tests for assembly theory configuration module.

Tests AssemblyConfig defaults, YAML serialization, validation, and to_dict.
"""

import pytest

from src.framework.assembly.config import AssemblyConfig


@pytest.mark.unit
class TestAssemblyConfig:
    """Tests for AssemblyConfig dataclass."""

    def test_defaults(self):
        cfg = AssemblyConfig()
        assert cfg.mcts_ucb_weight == 0.2
        assert cfg.max_complexity_threshold == 15
        assert cfg.routing_simple_threshold == 3
        assert cfg.routing_medium_threshold == 7
        assert cfg.trm_complexity_penalty == 0.1
        assert cfg.substructure_reuse_threshold == 0.7
        assert cfg.enable_assembly_routing is True
        assert cfg.cache_assembly_indices is True
        assert cfg.max_cache_size == 10000

    def test_custom(self):
        cfg = AssemblyConfig(mcts_ucb_weight=0.5, max_cache_size=500)
        assert cfg.mcts_ucb_weight == 0.5
        assert cfg.max_cache_size == 500

    def test_to_dict(self):
        cfg = AssemblyConfig()
        d = cfg.to_dict()
        assert "mcts" in d
        assert d["mcts"]["ucb_weight"] == 0.2
        assert "routing" in d
        assert "trm" in d
        assert "substructure" in d
        assert "feature_flags" in d
        assert d["feature_flags"]["routing"] is True
        assert "performance" in d
        assert "consensus" in d
        assert "concept_extraction" in d

    def test_validate_valid(self):
        cfg = AssemblyConfig()
        cfg.validate()  # Should not raise

    def test_validate_invalid_ucb_weight(self):
        cfg = AssemblyConfig(mcts_ucb_weight=1.5)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_validate_invalid_thresholds(self):
        cfg = AssemblyConfig(routing_simple_threshold=10, routing_medium_threshold=5)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_validate_negative_cache(self):
        cfg = AssemblyConfig(max_cache_size=-1)
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_from_yaml_nonexistent(self):
        cfg = AssemblyConfig.from_yaml("/nonexistent/path.yaml")
        assert cfg.mcts_ucb_weight == 0.2  # Returns defaults

    def test_from_yaml_valid(self, tmp_path):
        yaml_content = """
assembly:
  mcts:
    ucb_weight: 0.4
    max_complexity_threshold: 20
  routing:
    simple_threshold: 5
    medium_threshold: 10
  feature_flags:
    routing: false
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)
        cfg = AssemblyConfig.from_yaml(str(path))
        assert cfg.mcts_ucb_weight == 0.4
        assert cfg.max_complexity_threshold == 20
        assert cfg.routing_simple_threshold == 5
        assert cfg.enable_assembly_routing is False

    def test_save_and_load_yaml(self, tmp_path):
        cfg = AssemblyConfig(mcts_ucb_weight=0.35)
        path = str(tmp_path / "output.yaml")
        cfg.save_yaml(path)
        loaded = AssemblyConfig.from_yaml(path)
        assert loaded.mcts_ucb_weight == pytest.approx(0.35)

    def test_consensus_defaults(self):
        cfg = AssemblyConfig()
        assert cfg.consensus_pathway_weight == 0.7
        assert cfg.consensus_complexity_weight == 0.3
        assert cfg.complexity_selection_weight == 0.1

    def test_concept_extraction_defaults(self):
        cfg = AssemblyConfig()
        assert cfg.min_concept_frequency == 1
        assert cfg.max_concepts == 100
        assert cfg.use_technical_terms is True
