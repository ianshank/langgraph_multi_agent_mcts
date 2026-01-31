"""
Comprehensive unit tests for personality trait modules.

Tests cover:
- Input validation and sanitization
- Range enforcement [0.0-1.0]
- Type safety
- Memory safety of bounded collections
- Module functionality
- Integration patterns
"""

from __future__ import annotations

import math
import threading
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from pydantic import ValidationError

# Import personality module components
from src.framework.personality.profiles import (
    DEFAULT_TRAIT_VALUE,
    MAX_TRAIT_VALUE,
    MIN_TRAIT_VALUE,
    PersonalityProfile,
    PersonalityTraits,
    validate_trait_value,
)
from src.framework.personality.collections import (
    BoundedCounter,
    BoundedHistory,
    TimeAwareBoundedHistory,
)
from src.framework.personality.config import (
    AspirationConfig,
    CuriosityConfig,
    EthicalConfig,
    LoyaltyConfig,
    PersonalityConfig,
    TransparencyConfig,
)
from src.framework.personality.exceptions import (
    ErrorSeverity,
    EthicalViolationError,
    MemoryLimitError,
    PersonalityError,
    TraitValidationError,
)
from src.framework.personality.modules import (
    AspirationModule,
    CuriosityModule,
    EthicalReasoningModule,
    LoyaltyModule,
    TransparencyModule,
)


class TestPersonalityProfile:
    """Test suite for PersonalityProfile Pydantic model."""

    def test_default_profile(self):
        """Test default personality profile creation."""
        profile = PersonalityProfile()

        # Check all traits are at default value
        assert profile.loyalty == DEFAULT_TRAIT_VALUE
        assert profile.curiosity == DEFAULT_TRAIT_VALUE
        assert profile.aspiration == DEFAULT_TRAIT_VALUE
        assert profile.ethical_weight == DEFAULT_TRAIT_VALUE
        assert profile.transparency == DEFAULT_TRAIT_VALUE

        # Check metadata
        assert profile.profile_version == "1.0.0"
        assert isinstance(profile.created_at, datetime)

    def test_custom_profile(self):
        """Test custom personality trait values."""
        profile = PersonalityProfile(
            loyalty=0.95,
            curiosity=0.85,
            aspiration=0.9,
            ethical_weight=0.92,
            transparency=0.88,
        )

        assert profile.loyalty == 0.95
        assert profile.curiosity == 0.85
        assert profile.aspiration == 0.9
        assert profile.ethical_weight == 0.92
        assert profile.transparency == 0.88

    @pytest.mark.parametrize(
        "trait_name,value",
        [
            ("loyalty", 0.0),
            ("loyalty", 1.0),
            ("curiosity", 0.5),
            ("aspiration", 0.25),
            ("ethical_weight", 0.75),
            ("transparency", 0.333),
        ],
    )
    def test_valid_trait_values(self, trait_name: str, value: float):
        """Test valid trait values are accepted."""
        profile = PersonalityProfile(**{trait_name: value})
        assert getattr(profile, trait_name) == value

    @pytest.mark.parametrize(
        "trait_name,invalid_value",
        [
            ("loyalty", -0.1),
            ("loyalty", 1.1),
            ("curiosity", -1.0),
            ("aspiration", 2.0),
            ("ethical_weight", -0.001),
            ("transparency", 999.0),
        ],
    )
    def test_invalid_trait_values_rejected(
        self, trait_name: str, invalid_value: float
    ):
        """Test out-of-range trait values are rejected."""
        with pytest.raises(ValidationError):
            PersonalityProfile(**{trait_name: invalid_value})

    @pytest.mark.parametrize(
        "nan_inf_value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_nan_inf_values_rejected(self, nan_inf_value: float):
        """Test NaN and Inf values are rejected for security."""
        with pytest.raises(ValidationError):
            PersonalityProfile(loyalty=nan_inf_value)

    def test_profile_immutability(self):
        """Test that profile is immutable (frozen)."""
        profile = PersonalityProfile(loyalty=0.7)

        with pytest.raises(ValidationError):
            profile.loyalty = 0.8  # type: ignore

    def test_with_trait_creates_new_profile(self):
        """Test with_trait creates new profile without mutating original."""
        original = PersonalityProfile(loyalty=0.7)
        modified = original.with_trait("loyalty", 0.9)

        assert original.loyalty == 0.7
        assert modified.loyalty == 0.9
        assert original is not modified

    def test_trait_vector_property(self):
        """Test trait_vector returns numpy array."""
        profile = PersonalityProfile(
            loyalty=0.1,
            curiosity=0.2,
            aspiration=0.3,
            ethical_weight=0.4,
            transparency=0.5,
        )

        vector = profile.trait_vector
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) == 5
        np.testing.assert_array_almost_equal(
            vector, [0.1, 0.2, 0.3, 0.4, 0.5]
        )

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = PersonalityProfile(
            loyalty=0.8,
            curiosity=0.7,
        )

        serialized = original.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["loyalty"] == 0.8

        restored = PersonalityProfile.from_dict(serialized)
        assert restored.loyalty == original.loyalty
        assert restored.curiosity == original.curiosity

    def test_preset_profiles(self):
        """Test preset profile factory methods."""
        default = PersonalityProfile.default()
        assert all(
            getattr(default, t) == DEFAULT_TRAIT_VALUE
            for t in ["loyalty", "curiosity", "aspiration"]
        )

        high_perf = PersonalityProfile.high_performer()
        assert high_perf.aspiration > 0.8

        explorer = PersonalityProfile.explorer()
        assert explorer.curiosity > 0.9

        principled = PersonalityProfile.principled()
        assert principled.ethical_weight > 0.9


class TestBoundedHistory:
    """Test suite for memory-safe bounded history."""

    def test_initialization(self):
        """Test bounded history initializes correctly."""
        history = BoundedHistory[str](max_size=100)

        assert len(history) == 0
        assert history.max_size == 100
        assert history.eviction_count == 0
        assert history.utilization == 0.0

    def test_append_and_retrieve(self):
        """Test appending and retrieving items."""
        history = BoundedHistory[int](max_size=5)

        for i in range(5):
            history.append(i)

        assert len(history) == 5
        assert history.get_all() == [0, 1, 2, 3, 4]
        assert history.is_full

    def test_automatic_eviction(self):
        """Test automatic FIFO eviction when full."""
        history = BoundedHistory[int](max_size=3)

        history.append(1)
        history.append(2)
        history.append(3)

        # Fourth append evicts first
        evicted = history.append(4)

        assert evicted == 1
        assert len(history) == 3
        assert history.get_all() == [2, 3, 4]
        assert history.eviction_count == 1

    def test_get_recent(self):
        """Test getting recent items."""
        history = BoundedHistory[int](max_size=10)

        for i in range(10):
            history.append(i)

        recent_3 = history.get_recent(3)
        assert recent_3 == [7, 8, 9]

    def test_thread_safety(self):
        """Test bounded history is thread-safe."""
        history = BoundedHistory[int](max_size=1000)

        def append_items(start: int, count: int):
            for i in range(start, start + count):
                history.append(i)

        threads = [
            threading.Thread(target=append_items, args=(i * 100, 100))
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 1000 items (at capacity)
        assert len(history) == 1000

    def test_invalid_max_size_rejected(self):
        """Test invalid max_size raises error."""
        with pytest.raises(ValueError):
            BoundedHistory[int](max_size=0)

        with pytest.raises(ValueError):
            BoundedHistory[int](max_size=-10)


class TestBoundedCounter:
    """Test suite for bounded counter."""

    def test_increment(self):
        """Test counter increment."""
        counter = BoundedCounter(max_count=100)

        assert counter.increment("key1") == 1
        assert counter.increment("key1") == 2
        assert counter.increment("key1", amount=5) == 7

    def test_overflow_protection(self):
        """Test overflow protection."""
        counter = BoundedCounter(max_count=10)

        counter.increment("key", amount=10)

        with pytest.raises(ValueError, match="overflow"):
            counter.increment("key", amount=1)

    def test_underflow_protection(self):
        """Test underflow protection."""
        counter = BoundedCounter()

        counter.increment("key", amount=5)

        with pytest.raises(ValueError, match="underflow"):
            counter.increment("key", amount=-10)

    def test_max_keys_limit(self):
        """Test maximum keys limit."""
        counter = BoundedCounter(max_keys=3)

        counter.increment("key1")
        counter.increment("key2")
        counter.increment("key3")

        with pytest.raises(MemoryLimitError):
            counter.increment("key4")

    def test_top_k(self):
        """Test top_k retrieval."""
        counter = BoundedCounter()

        counter.increment("a", amount=10)
        counter.increment("b", amount=5)
        counter.increment("c", amount=15)

        top_2 = counter.top_k(2)
        assert top_2[0] == ("c", 15)
        assert top_2[1] == ("a", 10)


class TestLoyaltyModule:
    """Test suite for LoyaltyModule."""

    @pytest.fixture
    def high_loyalty_module(self):
        """Create high-loyalty module fixture."""
        profile = PersonalityProfile(loyalty=0.95)
        return LoyaltyModule(profile)

    @pytest.fixture
    def low_loyalty_module(self):
        """Create low-loyalty module fixture."""
        profile = PersonalityProfile(loyalty=0.3)
        return LoyaltyModule(profile)

    def test_module_initialization(self, high_loyalty_module):
        """Test module initializes correctly."""
        assert high_loyalty_module.trait_value == 0.95
        assert high_loyalty_module.module_name == "loyalty"

    def test_commit_to_goal(self, high_loyalty_module):
        """Test goal commitment."""
        high_loyalty_module.commit_to_goal("test_goal", priority=0.8)

        assert "test_goal" in high_loyalty_module.get_active_goals()

    def test_goal_alignment_evaluation(self, high_loyalty_module):
        """Test goal alignment scoring."""
        high_loyalty_module.commit_to_goal("goal1", priority=0.9)

        # Same goal - should align
        alignment = high_loyalty_module.evaluate_goal_alignment(
            "action1", ["goal1"]
        )
        assert alignment > 0.8

        # Different goal - should penalize
        alignment = high_loyalty_module.evaluate_goal_alignment(
            "action2", ["goal2"]
        )
        assert alignment < 1.0

    def test_persistence_threshold_scaled_by_loyalty(
        self, high_loyalty_module, low_loyalty_module
    ):
        """Test high loyalty = higher persistence threshold."""
        high_loyalty_module.commit_to_goal("goal1", priority=0.9)
        low_loyalty_module.commit_to_goal("goal1", priority=0.9)

        # High loyalty should persist longer
        should_persist_high, _ = high_loyalty_module.should_persist_on_goal(
            "goal1", difficulty=0.5, attempts=10
        )

        should_persist_low, _ = low_loyalty_module.should_persist_on_goal(
            "goal1", difficulty=0.5, attempts=10
        )

        # High loyalty more likely to persist
        assert should_persist_high or not should_persist_low


class TestCuriosityModule:
    """Test suite for CuriosityModule."""

    @pytest.fixture
    def high_curiosity_module(self):
        """Create high-curiosity module fixture."""
        profile = PersonalityProfile(curiosity=0.9)
        return CuriosityModule(profile)

    def test_intrinsic_reward_calculation(self, high_curiosity_module):
        """Test intrinsic reward computation."""
        # First visit - high novelty
        reward1 = high_curiosity_module.compute_intrinsic_reward(
            "state1", "action1", uncertainty=0.5
        )
        assert reward1 > 0

        # Second visit - lower novelty
        reward2 = high_curiosity_module.compute_intrinsic_reward(
            "state1", "action1", uncertainty=0.5
        )
        assert reward2 < reward1  # Novelty decreases

    def test_exploration_bonus(self, high_curiosity_module):
        """Test exploration bonus calculation."""
        base_weight = 1.414

        bonus = high_curiosity_module.get_exploration_bonus(
            base_weight=base_weight,
            iteration=0,
            state_novelty=0.8,
        )

        assert bonus > base_weight  # Curiosity adds bonus

    def test_rollout_policy_modification(self, high_curiosity_module):
        """Test rollout policy is modified by curiosity."""
        # High curiosity should favor exploration
        policy = high_curiosity_module.modify_rollout_policy(
            "greedy", {}
        )
        assert policy in ["random", "hybrid"]


class TestEthicalReasoningModule:
    """Test suite for EthicalReasoningModule."""

    @pytest.fixture
    def ethics_module(self):
        """Create ethical reasoning module fixture."""
        profile = PersonalityProfile(ethical_weight=0.9)
        return EthicalReasoningModule(profile)

    def test_ethical_evaluation(self, ethics_module):
        """Test ethical action evaluation."""
        score, assessment = ethics_module.evaluate_action_ethics(
            action="help_user",
            context={},
            consequences={
                "benefits": [{"magnitude": 0.8}],
                "harms": [{"magnitude": 0.1}],
            },
        )

        assert 0.0 <= score <= 1.0
        assert "final_score" in assessment

    def test_prohibition_detection(self, ethics_module):
        """Test absolute prohibition detection."""
        score, assessment = ethics_module.evaluate_action_ethics(
            action="deceive_user",
            context={"action_info": {"deception": True}},
            consequences={},
        )

        assert score == 0.0
        assert assessment.get("prohibited", False)

    def test_ethical_constraint_check(self, ethics_module):
        """Test ethical constraint checking."""
        # Allowed action
        is_allowed, reason = ethics_module.check_ethical_constraints("help")
        assert is_allowed
        assert reason is None

        # Prohibited action
        is_allowed, reason = ethics_module.check_ethical_constraints("deceive user")
        assert not is_allowed
        assert reason is not None


class TestTransparencyModule:
    """Test suite for TransparencyModule."""

    @pytest.fixture
    def transparency_module(self):
        """Create transparency module fixture."""
        profile = PersonalityProfile(transparency=0.9)
        return TransparencyModule(profile)

    @pytest.mark.asyncio
    async def test_decision_logging(self, transparency_module):
        """Test decision logging functionality."""
        log = await transparency_module.log_decision(
            state_id="s1",
            action="explore",
            rationale={"score": 0.8},
            confidence=0.9,
        )

        assert log is not None
        assert log.action == "explore"
        assert log.confidence == 0.9

    @pytest.mark.asyncio
    async def test_pii_masking(self, transparency_module):
        """Test PII is masked in logs."""
        log = await transparency_module.log_decision(
            state_id="s1",
            action="contact user at test@example.com",
            rationale={"phone": "555-123-4567"},
            confidence=0.8,
        )

        assert "test@example.com" not in log.action
        assert "[REDACTED_EMAIL]" in log.action

    @pytest.mark.asyncio
    async def test_low_transparency_no_logging(self):
        """Test low transparency doesn't log."""
        profile = PersonalityProfile(transparency=0.3)
        module = TransparencyModule(profile)

        log = await module.log_decision(
            state_id="s1",
            action="action",
            rationale={},
            confidence=0.5,
        )

        assert log is None  # Not logged due to low transparency


class TestPersonalityConfig:
    """Test suite for PersonalityConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PersonalityConfig()

        assert not config.enabled
        assert isinstance(config.default_profile, PersonalityProfile)

    def test_enabled_config(self):
        """Test enabled configuration."""
        config = PersonalityConfig.with_defaults()

        assert config.enabled
        assert config.default_profile is not None

    def test_preset_configs(self):
        """Test preset configurations."""
        high_perf = PersonalityConfig.high_performance()
        assert high_perf.enabled
        assert high_perf.hrm_profile is not None

        explorer = PersonalityConfig.exploration_focused()
        assert explorer.mcts_curiosity_weight > 0.3

        ethics = PersonalityConfig.ethics_focused()
        assert ethics.ethical_veto_threshold > 0.3

    def test_agent_profile_selection(self):
        """Test profile selection by agent name."""
        config = PersonalityConfig(
            enabled=True,
            hrm_profile=PersonalityProfile(loyalty=0.9),
            trm_profile=PersonalityProfile(loyalty=0.7),
        )

        hrm_profile = config.get_profile_for_agent("hrm_agent")
        assert hrm_profile.loyalty == 0.9

        trm_profile = config.get_profile_for_agent("trm_model")
        assert trm_profile.loyalty == 0.7

        unknown_profile = config.get_profile_for_agent("unknown")
        assert unknown_profile == config.default_profile


class TestExceptions:
    """Test suite for custom exceptions."""

    def test_personality_error(self):
        """Test base PersonalityError."""
        error = PersonalityError(
            message="Test error",
            severity=ErrorSeverity.HIGH,
            context={"key": "value"},
        )

        assert "Test error" in str(error)
        assert "[HIGH]" in str(error)

    def test_trait_validation_error(self):
        """Test TraitValidationError."""
        error = TraitValidationError(
            message="",
            trait_name="loyalty",
            trait_value=1.5,
        )

        assert "loyalty" in str(error)
        assert "1.5" in str(error)

    def test_ethical_violation_error(self):
        """Test EthicalViolationError."""
        error = EthicalViolationError(
            message="",
            action="deceive",
            framework="deontological",
            violation_type="prohibition",
        )

        assert error.severity == ErrorSeverity.HIGH
        assert "deceive" in str(error)


class TestPersonalityTraitsLegacy:
    """Test legacy dataclass for backward compatibility."""

    def test_legacy_traits_creation(self):
        """Test legacy PersonalityTraits creation."""
        traits = PersonalityTraits(
            loyalty=0.8,
            curiosity=0.7,
        )

        assert traits.loyalty == 0.8
        assert traits.curiosity == 0.7

    def test_conversion_to_pydantic(self):
        """Test conversion to Pydantic model."""
        traits = PersonalityTraits(loyalty=0.9)
        profile = traits.to_pydantic()

        assert isinstance(profile, PersonalityProfile)
        assert profile.loyalty == 0.9


class TestValidateTraitValue:
    """Test validate_trait_value helper."""

    def test_valid_values(self):
        """Test valid values pass validation."""
        assert validate_trait_value(0.5) == 0.5
        assert validate_trait_value(0.0) == 0.0
        assert validate_trait_value(1.0) == 1.0

    def test_invalid_values(self):
        """Test invalid values raise errors."""
        with pytest.raises(ValueError):
            validate_trait_value(-0.1)

        with pytest.raises(ValueError):
            validate_trait_value(1.1)

        with pytest.raises(ValueError):
            validate_trait_value(float("nan"))


# Run with: pytest tests/unit/test_personality.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
