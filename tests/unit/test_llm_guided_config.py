"""
Unit tests for LLM-Guided MCTS Configuration.

Tests configuration validation, serialization, and preset management.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.config.settings import LLMProvider
from src.framework.mcts.llm_guided.config import (
    GeneratorConfig,
    LLMGuidedMCTSConfig,
    LLMGuidedMCTSPreset,
    ReflectorConfig,
    create_llm_mcts_preset,
    get_preset_config,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig validation."""

    def test_default_config(self) -> None:
        """Default config should be valid."""
        config = GeneratorConfig()
        config.validate()  # Should not raise
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.num_variants == 3

    def test_temperature_bounds_low(self) -> None:
        """Temperature must be >= 0."""
        config = GeneratorConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_temperature_bounds_high(self) -> None:
        """Temperature must be <= 2."""
        config = GeneratorConfig(temperature=2.5)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_max_tokens_bounds_low(self) -> None:
        """max_tokens must be >= 1."""
        config = GeneratorConfig(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()

    def test_max_tokens_bounds_high(self) -> None:
        """max_tokens must be <= 8000."""
        config = GeneratorConfig(max_tokens=10000)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()

    def test_num_variants_bounds_low(self) -> None:
        """num_variants must be >= 1."""
        config = GeneratorConfig(num_variants=0)
        with pytest.raises(ValueError, match="num_variants"):
            config.validate()

    def test_num_variants_bounds_high(self) -> None:
        """num_variants must be <= 10."""
        config = GeneratorConfig(num_variants=15)
        with pytest.raises(ValueError, match="num_variants"):
            config.validate()

    def test_top_p_bounds(self) -> None:
        """top_p must be in [0, 1]."""
        config = GeneratorConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()

    @pytest.mark.parametrize(
        "temperature,max_tokens,num_variants",
        [
            (0.0, 1, 1),  # minimum valid
            (2.0, 8000, 10),  # maximum valid
            (0.5, 1000, 3),  # typical
        ],
    )
    def test_valid_configurations(
        self,
        temperature: float,
        max_tokens: int,
        num_variants: int,
    ) -> None:
        """Valid configurations should pass validation."""
        config = GeneratorConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            num_variants=num_variants,
        )
        config.validate()  # Should not raise


class TestReflectorConfig:
    """Tests for ReflectorConfig validation."""

    def test_default_config(self) -> None:
        """Default config should be valid."""
        config = ReflectorConfig()
        config.validate()  # Should not raise
        assert config.model == "gpt-4o"
        assert config.temperature == 0.3

    def test_temperature_validation(self) -> None:
        """Temperature must be in [0, 2]."""
        config = ReflectorConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_max_tokens_validation(self) -> None:
        """max_tokens must be in valid range."""
        config = ReflectorConfig(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()


class TestLLMGuidedMCTSConfig:
    """Tests for LLMGuidedMCTSConfig validation and serialization."""

    def test_default_config(self) -> None:
        """Default config should be valid."""
        config = LLMGuidedMCTSConfig()
        config.validate()  # Should not raise
        assert config.num_iterations == 30
        assert config.exploration_weight == pytest.approx(1.414)

    def test_num_iterations_bounds(self) -> None:
        """num_iterations must be >= 1."""
        config = LLMGuidedMCTSConfig(num_iterations=0)
        with pytest.raises(ValueError, match="num_iterations"):
            config.validate()

    def test_exploration_weight_bounds(self) -> None:
        """exploration_weight must be > 0."""
        config = LLMGuidedMCTSConfig(exploration_weight=-0.1)
        with pytest.raises(ValueError, match="exploration_weight"):
            config.validate()

    def test_max_depth_bounds(self) -> None:
        """max_depth must be >= 1."""
        config = LLMGuidedMCTSConfig(max_depth=0)
        with pytest.raises(ValueError, match="max_depth"):
            config.validate()

    def test_max_children_bounds(self) -> None:
        """max_children must be >= 1."""
        config = LLMGuidedMCTSConfig(max_children=0)
        with pytest.raises(ValueError, match="max_children"):
            config.validate()

    def test_solution_threshold_bounds(self) -> None:
        """solution_confidence_threshold must be in (0, 1]."""
        config = LLMGuidedMCTSConfig(solution_confidence_threshold=0)
        with pytest.raises(ValueError, match="solution_confidence_threshold"):
            config.validate()

        config = LLMGuidedMCTSConfig(solution_confidence_threshold=1.5)
        with pytest.raises(ValueError, match="solution_confidence_threshold"):
            config.validate()

    def test_execution_timeout_bounds(self) -> None:
        """execution_timeout_seconds must be > 0."""
        config = LLMGuidedMCTSConfig(execution_timeout_seconds=0)
        with pytest.raises(ValueError, match="execution_timeout"):
            config.validate()

    def test_nested_config_validation(self) -> None:
        """Nested configs should also be validated."""
        config = LLMGuidedMCTSConfig(
            generator_config=GeneratorConfig(temperature=5.0)  # Invalid
        )
        with pytest.raises(ValueError, match="temperature"):
            config.validate()


class TestLLMGuidedMCTSConfigSerialization:
    """Tests for configuration serialization."""

    def test_to_dict_and_from_dict_round_trip(self) -> None:
        """Config -> dict -> Config should preserve values."""
        original = LLMGuidedMCTSConfig(
            name="test_config",
            num_iterations=50,
            exploration_weight=2.0,
            max_depth=15,
            generator_config=GeneratorConfig(temperature=0.9, num_variants=5),
            reflector_config=ReflectorConfig(temperature=0.2),
            llm_provider=LLMProvider.ANTHROPIC,
        )

        dict_repr = original.to_dict()
        restored = LLMGuidedMCTSConfig.from_dict(dict_repr)

        assert restored.name == original.name
        assert restored.num_iterations == original.num_iterations
        assert restored.exploration_weight == pytest.approx(original.exploration_weight)
        assert restored.max_depth == original.max_depth
        assert restored.generator_config.temperature == pytest.approx(0.9)
        assert restored.generator_config.num_variants == 5
        assert restored.reflector_config.temperature == pytest.approx(0.2)
        assert restored.llm_provider == LLMProvider.ANTHROPIC

    def test_to_json_and_from_json_round_trip(self) -> None:
        """Config -> JSON -> Config should preserve values."""
        original = LLMGuidedMCTSConfig(
            name="json_test",
            num_iterations=100,
            seed=123,
        )

        json_str = original.to_json()
        restored = LLMGuidedMCTSConfig.from_json(json_str)

        assert restored.name == "json_test"
        assert restored.num_iterations == 100
        assert restored.seed == 123

    def test_save_and_load_from_file(self) -> None:
        """Config can be saved to and loaded from file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = LLMGuidedMCTSConfig(
                name="file_test",
                num_iterations=75,
            )
            filepath = Path(tmp_dir) / "config.json"

            config.save(filepath)
            assert filepath.exists()

            loaded = LLMGuidedMCTSConfig.load(filepath)
            assert loaded.name == "file_test"
            assert loaded.num_iterations == 75

    def test_to_dict_contains_expected_keys(self) -> None:
        """to_dict should contain all expected keys."""
        config = LLMGuidedMCTSConfig()
        d = config.to_dict()

        expected_keys = {
            "name",
            "num_iterations",
            "exploration_weight",
            "max_depth",
            "max_children",
            "solution_confidence_threshold",
            "generator_config",
            "reflector_config",
            "llm_provider",
            "early_termination_on_solution",
            "collect_training_data",
            "training_data_dir",
            "execution_timeout_seconds",
            "max_memory_mb",
            "seed",
            "verbose",
        }

        for key in expected_keys:
            assert key in d, f"Missing expected key: {key}"


class TestLLMGuidedMCTSConfigCopy:
    """Tests for configuration copying."""

    def test_copy_creates_independent_instance(self) -> None:
        """copy() should create an independent instance."""
        original = LLMGuidedMCTSConfig(
            name="original",
            num_iterations=50,
        )

        copied = original.copy()
        copied.name = "copied"
        copied.num_iterations = 100

        assert original.name == "original"
        assert original.num_iterations == 50
        assert copied.name == "copied"
        assert copied.num_iterations == 100

    def test_copy_with_overrides(self) -> None:
        """copy() should allow overriding values."""
        original = LLMGuidedMCTSConfig(
            name="original",
            num_iterations=50,
            max_depth=10,
        )

        copied = original.copy(num_iterations=200, max_depth=20)

        assert original.num_iterations == 50
        assert original.max_depth == 10
        assert copied.num_iterations == 200
        assert copied.max_depth == 20
        assert copied.name == "original"  # Preserved from original


class TestLLMGuidedMCTSPresets:
    """Tests for preset configurations."""

    def test_fast_preset(self) -> None:
        """Fast preset should have low iterations."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.FAST)
        config.validate()

        assert config.name == "fast"
        assert config.num_iterations == 10
        assert config.max_depth <= 10
        assert config.generator_config.num_variants <= 3

    def test_balanced_preset(self) -> None:
        """Balanced preset should have moderate settings."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.BALANCED)
        config.validate()

        assert config.name == "balanced"
        assert config.num_iterations == 30
        assert config.collect_training_data is True

    def test_thorough_preset(self) -> None:
        """Thorough preset should have high iterations."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.THOROUGH)
        config.validate()

        assert config.name == "thorough"
        assert config.num_iterations == 100
        assert config.verbose is True

    def test_benchmark_preset(self) -> None:
        """Benchmark preset should be optimized for evaluation."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.BENCHMARK)
        config.validate()

        assert config.name == "benchmark"
        assert config.num_iterations >= 50

    def test_creative_preset(self) -> None:
        """Creative preset should have higher temperatures."""
        config = create_llm_mcts_preset(LLMGuidedMCTSPreset.CREATIVE)
        config.validate()

        assert config.name == "creative"
        assert config.generator_config.temperature >= 0.8
        assert config.exploration_weight > 1.414  # Higher exploration

    def test_all_presets_valid(self) -> None:
        """All presets should produce valid configurations."""
        for preset in LLMGuidedMCTSPreset:
            config = create_llm_mcts_preset(preset)
            config.validate()  # Should not raise

    def test_get_preset_config_by_name(self) -> None:
        """get_preset_config should work with string names."""
        for name in ["fast", "balanced", "thorough", "benchmark", "creative"]:
            config = get_preset_config(name)
            assert config.name == name

    def test_get_preset_config_invalid_name(self) -> None:
        """get_preset_config should raise for invalid names."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("nonexistent")

    def test_get_preset_config_case_insensitive(self) -> None:
        """get_preset_config should be case-insensitive."""
        config1 = get_preset_config("FAST")
        config2 = get_preset_config("fast")
        config3 = get_preset_config("Fast")

        assert config1.name == config2.name == config3.name == "fast"


class TestMultipleValidationErrors:
    """Tests for validation error aggregation."""

    def test_multiple_errors_reported(self) -> None:
        """Multiple validation errors should all be reported."""
        config = LLMGuidedMCTSConfig(
            num_iterations=0,  # Invalid
            exploration_weight=-1,  # Invalid
            max_depth=0,  # Invalid
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_msg = str(exc_info.value)
        assert "num_iterations" in error_msg
        assert "exploration_weight" in error_msg
        assert "max_depth" in error_msg


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_iterations(self) -> None:
        """Very large iteration count should be valid but warn-worthy."""
        config = LLMGuidedMCTSConfig(num_iterations=999)
        config.validate()  # Should not raise

    def test_zero_seed(self) -> None:
        """Zero seed should be valid."""
        config = LLMGuidedMCTSConfig(seed=0)
        config.validate()

    def test_negative_seed(self) -> None:
        """Negative seed should be valid (Python allows it)."""
        config = LLMGuidedMCTSConfig(seed=-1)
        config.validate()

    def test_empty_training_data_dir(self) -> None:
        """Empty training_data_dir should still be valid."""
        config = LLMGuidedMCTSConfig(
            training_data_dir="",
            collect_training_data=False,
        )
        config.validate()
