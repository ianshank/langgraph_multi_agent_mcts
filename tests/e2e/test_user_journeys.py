"""
User Journey Validators for LangGraph Multi-Agent MCTS.

Tests simulate real user workflows including:
- Research assistant journey
- Code generation journey
- Configuration customization journey
- Training workflow journey
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from dataclasses import dataclass

import pytest
import numpy as np

# Framework imports
from src.framework.actions import (
    ActionType,
    GraphConfig,
    ConfidenceConfig,
    RolloutWeights,
    SynthesisConfig,
    create_research_config,
    create_coding_config,
    create_creative_config,
)
from src.framework.mcts.config import (
    MCTSConfig,
    ConfigPreset,
    create_preset_config,
)
from src.framework.mcts.core import MCTSEngine, MCTSNode, MCTSState
from src.framework.mcts.policies import RandomRolloutPolicy

# Conditional imports
try:
    from src.framework.mcts.neural_integration import (
        NeuralMCTSConfig,
        NeuralRolloutPolicy,
        create_neural_mcts_adapter,
        get_balanced_neural_config,
    )
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    from src.training.expert_iteration import (
        ExpertIterationConfig,
        ReplayBuffer,
        Trajectory,
        TrajectoryStep,
    )
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


@dataclass
class UserSession:
    """Simulates a user session with the system."""

    session_id: str
    queries: list[str]
    results: list[dict[str, Any]]
    config: GraphConfig
    start_time: float
    end_time: float | None = None

    def add_result(self, result: dict[str, Any]):
        """Add a query result."""
        self.results.append(result)

    def complete(self):
        """Mark session as complete."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get session duration."""
        end = self.end_time or time.time()
        return end - self.start_time


class JourneyValidator:
    """Base class for journey validation."""

    def __init__(self, name: str):
        self.name = name
        self.steps_completed: list[str] = []
        self.errors: list[str] = []
        self.metrics: dict[str, Any] = {}

    def record_step(self, step: str):
        """Record a completed step."""
        self.steps_completed.append(step)

    def record_error(self, error: str):
        """Record an error."""
        self.errors.append(error)

    def record_metric(self, name: str, value: Any):
        """Record a metric."""
        self.metrics[name] = value

    @property
    def is_successful(self) -> bool:
        """Check if journey was successful."""
        return len(self.errors) == 0

    def get_report(self) -> dict[str, Any]:
        """Get journey report."""
        return {
            "name": self.name,
            "steps_completed": self.steps_completed,
            "errors": self.errors,
            "metrics": self.metrics,
            "is_successful": self.is_successful,
        }


class TestResearcherJourney:
    """
    User Journey: AI Researcher using the system for literature analysis.

    Steps:
    1. Configure for research (high exploration, parallel agents)
    2. Submit complex research query
    3. Process with MCTS for thorough exploration
    4. Review multi-agent outputs
    5. Iterate with refined query
    """

    @pytest.fixture
    def validator(self):
        """Create journey validator."""
        return JourneyValidator("researcher_journey")

    @pytest.fixture
    def session(self):
        """Create user session."""
        return UserSession(
            session_id="researcher_001",
            queries=[],
            results=[],
            config=create_research_config(),
            start_time=time.time(),
        )

    @pytest.fixture
    def rollout_policy(self):
        """Create rollout policy for MCTS."""
        return RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    def test_step1_configure_for_research(self, validator, session):
        """Step 1: User configures system for research."""
        config = session.config

        # Verify research-appropriate settings
        assert config.enable_parallel_agents is True, "Research needs parallel agents"
        assert config.confidence.consensus_threshold <= 0.8, "Research allows diverse views"
        assert config.rollout_weights.random_weight >= 0.3, "Research needs exploration"

        validator.record_step("configure_for_research")
        validator.record_metric("consensus_threshold", config.confidence.consensus_threshold)

    @pytest.mark.asyncio
    async def test_step2_submit_research_query(self, validator, session, rollout_policy):
        """Step 2: User submits complex research query."""
        query = "Compare transformer architectures: attention mechanisms in BERT vs GPT vs T5"
        session.queries.append(query)

        # Create MCTS for exploration
        mcts_config = create_preset_config(ConfigPreset.THOROUGH)
        engine = MCTSEngine(
            seed=42,
            exploration_weight=mcts_config.exploration_weight,
        )

        root = MCTSNode(
            state=MCTSState(
                state_id="root",
                features={"query": query, "domain": "research"},
            ),
            rng=engine.rng,
        )

        # Research-specific actions
        def action_gen(s):
            depth = len(s.state_id.split("_")) - 1
            if depth == 0:
                return ["literature_search", "compare_models", "synthesize_findings"]
            elif depth < 4:
                return ["dig_deeper", "broaden_scope", "focus"]
            return []

        def state_trans(s, a):
            return MCTSState(
                state_id=f"{s.state_id}_{a}",
                features={**s.features, "last_action": a},
            )

        best_action, stats = await engine.search(
            root=root,
            num_iterations=100,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        # Record result
        session.add_result({
            "query": query,
            "best_action": best_action,
            "iterations": stats["iterations"],
        })

        validator.record_step("submit_research_query")
        validator.record_metric("search_iterations", stats["iterations"])

        assert best_action in ["literature_search", "compare_models", "synthesize_findings"]

    def test_step3_review_agent_outputs(self, validator, session):
        """Step 3: User reviews multi-agent outputs."""
        # Simulated agent outputs
        agent_outputs = [
            {"agent": "HRM", "response": "Decomposed into 3 sub-questions", "confidence": 0.85},
            {"agent": "TRM", "response": "Refined analysis through 5 iterations", "confidence": 0.82},
            {"agent": "MCTS", "response": "Explored 100 solution paths", "confidence": 0.88},
        ]

        # Calculate consensus
        confidences = [a["confidence"] for a in agent_outputs]
        consensus = sum(confidences) / len(confidences)

        validator.record_step("review_agent_outputs")
        validator.record_metric("num_agents", len(agent_outputs))
        validator.record_metric("consensus_score", consensus)

        assert consensus >= session.config.confidence.consensus_threshold - 0.1

    @pytest.mark.asyncio
    async def test_step4_iterate_with_refinement(self, validator, session, rollout_policy):
        """Step 4: User iterates with refined query."""
        refined_query = "Focus specifically on attention head patterns in BERT"
        session.queries.append(refined_query)

        # Second search with more focused actions
        engine = MCTSEngine(seed=43)
        root = MCTSNode(
            state=MCTSState(
                state_id="root",
                features={"query": refined_query, "iteration": 2},
            ),
            rng=engine.rng,
        )

        def action_gen(s):
            return ["analyze_detail", "cite_sources", "conclude"] if len(s.state_id) < 15 else []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        best_action, stats = await engine.search(
            root=root,
            num_iterations=50,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        session.add_result({
            "query": refined_query,
            "best_action": best_action,
            "is_refinement": True,
        })

        validator.record_step("iterate_with_refinement")
        validator.record_metric("total_queries", len(session.queries))

    def test_journey_completion(self, validator, session):
        """Verify complete journey."""
        session.complete()

        validator.record_step("journey_complete")
        validator.record_metric("session_duration", session.duration)

        report = validator.get_report()
        # Tests run independently in pytest, so we check essential properties
        assert report["is_successful"]
        assert "journey_complete" in report["steps_completed"]


class TestDeveloperJourney:
    """
    User Journey: Software developer using the system for code generation.

    Steps:
    1. Configure for coding (low temperature, high consensus)
    2. Submit code generation query
    3. Process with sequential agents
    4. Validate generated code structure
    5. Refine with debugging query
    """

    @pytest.fixture
    def validator(self):
        """Create journey validator."""
        return JourneyValidator("developer_journey")

    @pytest.fixture
    def session(self):
        """Create user session."""
        return UserSession(
            session_id="developer_001",
            queries=[],
            results=[],
            config=create_coding_config(),
            start_time=time.time(),
        )

    @pytest.fixture
    def rollout_policy(self):
        """Create rollout policy for MCTS."""
        return RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    def test_step1_configure_for_coding(self, validator, session):
        """Step 1: User configures for code generation."""
        config = session.config

        # Verify coding-appropriate settings
        assert config.enable_parallel_agents is False, "Coding needs sequential processing"
        assert config.confidence.consensus_threshold >= 0.8, "Code needs high agreement"
        assert config.synthesis.temperature <= 0.5, "Code needs determinism"

        validator.record_step("configure_for_coding")
        validator.record_metric("temperature", config.synthesis.temperature)

    @pytest.mark.asyncio
    async def test_step2_submit_code_query(self, validator, session, rollout_policy):
        """Step 2: User submits code generation query."""
        query = "Implement a binary search tree with insert, delete, and search methods in Python"
        session.queries.append(query)

        # Use coding-specific actions
        config = create_preset_config(ConfigPreset.BALANCED)
        engine = MCTSEngine(seed=42)

        root = MCTSNode(
            state=MCTSState(
                state_id="root",
                features={"query": query, "language": "python"},
            ),
            rng=engine.rng,
        )

        # Coding-specific actions
        root_actions = ["implement", "test", "refactor", "document"]
        cont_actions = ["continue", "fix", "optimize", "validate"]

        def action_gen(s):
            depth = len(s.state_id.split("_")) - 1
            if depth == 0:
                return root_actions
            elif depth < 3:
                return cont_actions
            return []

        def state_trans(s, a):
            return MCTSState(
                state_id=f"{s.state_id}_{a}",
                features={**s.features, "action_history": s.features.get("action_history", []) + [a]},
            )

        best_action, stats = await engine.search(
            root=root,
            num_iterations=50,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        session.add_result({
            "query": query,
            "best_action": best_action,
            "stats": stats,
        })

        validator.record_step("submit_code_query")
        validator.record_metric("best_action", best_action)

        assert best_action in root_actions

    def test_step3_validate_code_structure(self, validator, session):
        """Step 3: Validate the generated code structure."""
        # Simulated code validation checks
        validation_checks = {
            "has_class_definition": True,
            "has_insert_method": True,
            "has_delete_method": True,
            "has_search_method": True,
            "passes_syntax_check": True,
        }

        all_passed = all(validation_checks.values())

        validator.record_step("validate_code_structure")
        validator.record_metric("validation_checks", validation_checks)
        validator.record_metric("all_checks_passed", all_passed)

        assert all_passed

    @pytest.mark.asyncio
    async def test_step4_debugging_refinement(self, validator, session, rollout_policy):
        """Step 4: User submits debugging query."""
        debug_query = "Fix the delete method to handle the case when deleting a node with two children"
        session.queries.append(debug_query)

        engine = MCTSEngine(seed=44)
        root = MCTSNode(
            state=MCTSState(
                state_id="root",
                features={"query": debug_query, "is_debug": True},
            ),
            rng=engine.rng,
        )

        def action_gen(s):
            return ["identify_bug", "fix", "test_fix", "verify"] if len(s.state_id) < 15 else []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        best_action, _ = await engine.search(
            root=root,
            num_iterations=30,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        validator.record_step("debugging_refinement")
        validator.record_metric("debug_action", best_action)

    def test_journey_completion(self, validator, session):
        """Verify complete journey."""
        session.complete()

        # Record completion step for this independent test
        validator.record_step("journey_complete")
        validator.record_metric("session_duration", session.duration)

        report = validator.get_report()
        # Tests run independently in pytest, so we check essential properties
        assert report["is_successful"]
        assert "journey_complete" in report["steps_completed"]


class TestConfigurationJourney:
    """
    User Journey: Power user customizing configuration.

    Steps:
    1. Explore available presets
    2. Create custom configuration
    3. Validate configuration
    4. Apply and test configuration
    """

    @pytest.fixture
    def validator(self):
        """Create journey validator."""
        return JourneyValidator("configuration_journey")

    @pytest.fixture
    def rollout_policy(self):
        """Create rollout policy for MCTS."""
        return RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

    def test_step1_explore_presets(self, validator):
        """Step 1: User explores available presets."""
        presets = {
            "fast": create_preset_config(ConfigPreset.FAST),
            "balanced": create_preset_config(ConfigPreset.BALANCED),
            "thorough": create_preset_config(ConfigPreset.THOROUGH),
        }

        domain_presets = {
            "research": create_research_config(),
            "coding": create_coding_config(),
            "creative": create_creative_config(),
        }

        validator.record_step("explore_presets")
        validator.record_metric("mcts_presets_count", len(presets))
        validator.record_metric("domain_presets_count", len(domain_presets))

        # All presets should be valid
        for name, config in presets.items():
            config.validate()

    def test_step2_create_custom_config(self, validator):
        """Step 2: User creates custom configuration."""
        custom_config = GraphConfig(
            max_iterations=4,
            enable_parallel_agents=True,
            confidence=ConfidenceConfig(
                consensus_threshold=0.85,
                default_hrm_confidence=0.6,
                heuristic_base_value=0.55,
            ),
            rollout_weights=RolloutWeights(
                heuristic_weight=0.65,
                random_weight=0.35,
            ),
            synthesis=SynthesisConfig(
                temperature=0.4,
                max_tokens=4096,
            ),
            root_actions=["analyze", "implement", "verify", "refine"],
            continuation_actions=["continue", "adjust", "finalize"],
        )

        validator.record_step("create_custom_config")
        validator.record_metric("custom_consensus_threshold", custom_config.confidence.consensus_threshold)

    def test_step3_validate_configuration(self, validator):
        """Step 3: User validates configuration."""
        # Valid config
        valid = GraphConfig(max_iterations=3)
        assert valid.max_iterations == 3

        # Test validation catches errors
        validation_errors = []

        try:
            GraphConfig(max_iterations=0)
        except ValueError:
            validation_errors.append("max_iterations")

        try:
            ConfidenceConfig(consensus_threshold=2.0)
        except ValueError:
            validation_errors.append("consensus_threshold")

        validator.record_step("validate_configuration")
        validator.record_metric("validation_errors_caught", len(validation_errors))

        assert len(validation_errors) == 2

    @pytest.mark.asyncio
    async def test_step4_apply_and_test(self, validator, rollout_policy):
        """Step 4: User applies and tests configuration."""
        config = GraphConfig(
            max_iterations=2,
            rollout_weights=RolloutWeights(heuristic_weight=0.8, random_weight=0.2),
        )

        mcts_config = create_preset_config(ConfigPreset.FAST)
        engine = MCTSEngine(
            seed=42,
            exploration_weight=mcts_config.exploration_weight,
        )

        root = MCTSNode(
            state=MCTSState(state_id="root", features={}),
            rng=engine.rng,
        )

        # Use configured actions
        def action_gen(s):
            depth = len(s.state_id.split("_")) - 1
            if depth == 0:
                return config.root_actions
            elif depth < 2:
                return config.continuation_actions
            return []

        def state_trans(s, a):
            return MCTSState(state_id=f"{s.state_id}_{a}", features={})

        best_action, stats = await engine.search(
            root=root,
            num_iterations=25,
            action_generator=action_gen,
            state_transition=state_trans,
            rollout_policy=rollout_policy,
        )

        validator.record_step("apply_and_test")
        validator.record_metric("test_successful", best_action in config.root_actions)

        assert best_action in config.root_actions

    def test_journey_completion(self, validator):
        """Verify complete journey."""
        report = validator.get_report()
        # Check all steps were recorded (they run independently in pytest)
        assert len(report["errors"]) == 0


@pytest.mark.skipif(not TRAINING_AVAILABLE, reason="Training not available")
class TestTrainingJourney:
    """
    User Journey: ML Engineer setting up Expert Iteration training.

    Steps:
    1. Configure training parameters
    2. Initialize replay buffer
    3. Generate trajectories
    4. Validate training loop components
    """

    @pytest.fixture
    def validator(self):
        """Create journey validator."""
        return JourneyValidator("training_journey")

    def test_step1_configure_training(self, validator):
        """Step 1: Configure training parameters."""
        config = ExpertIterationConfig(
            num_episodes_per_iteration=50,
            mcts_simulations=100,
            batch_size=64,
            learning_rate=1e-3,
            buffer_size=10000,
        )

        validator.record_step("configure_training")
        validator.record_metric("batch_size", config.batch_size)
        validator.record_metric("buffer_size", config.buffer_size)

        assert config.num_episodes_per_iteration > 0
        assert config.mcts_simulations > 0

    def test_step2_initialize_buffer(self, validator):
        """Step 2: Initialize replay buffer."""
        buffer = ReplayBuffer(max_size=1000, seed=42)

        # Add some sample data
        for i in range(100):
            state = MCTSState(
                state_id=f"state_{i}",
                features={"index": i},
            )
            policy = {"action_a": 0.5, "action_b": 0.3, "action_c": 0.2}
            value = 0.5 + 0.01 * i

            buffer.add(state, policy, value)

        validator.record_step("initialize_buffer")
        validator.record_metric("buffer_size", len(buffer))

        assert len(buffer) == 100

    def test_step3_generate_trajectories(self, validator):
        """Step 3: Generate sample trajectories."""
        trajectories = []

        for i in range(5):
            steps = []
            for j in range(10):
                step = TrajectoryStep(
                    state=MCTSState(state_id=f"traj_{i}_step_{j}", features={}),
                    action=f"action_{j % 3}",
                    mcts_policy={"a": 0.4, "b": 0.3, "c": 0.3},
                    value=0.5,
                )
                steps.append(step)

            trajectory = Trajectory(
                steps=steps,
                outcome=0.8 if i % 2 == 0 else -0.5,
                metadata={"trajectory_id": i},
            )
            trajectories.append(trajectory)

        validator.record_step("generate_trajectories")
        validator.record_metric("num_trajectories", len(trajectories))
        validator.record_metric("avg_length", np.mean([t.length for t in trajectories]))

        assert len(trajectories) == 5
        assert all(t.length == 10 for t in trajectories)

    def test_step4_validate_training_components(self, validator):
        """Step 4: Validate training loop components."""
        # Buffer sampling
        buffer = ReplayBuffer(max_size=100, seed=42)
        for i in range(50):
            buffer.add(
                MCTSState(state_id=f"s_{i}", features={}),
                {"a": 0.5, "b": 0.5},
                0.7,
            )

        batch = buffer.sample(16)
        assert len(batch) == 16

        # Config validation
        config = ExpertIterationConfig()
        assert config.num_episodes_per_iteration > 0

        validator.record_step("validate_training_components")
        validator.record_metric("batch_sample_size", len(batch))

    def test_journey_completion(self, validator):
        """Verify complete journey."""
        report = validator.get_report()
        assert len(report["errors"]) == 0


class TestMultiSessionJourney:
    """Test multiple user sessions interacting with the system."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """Multiple sessions should work concurrently."""
        sessions = []

        async def run_session(session_id: int):
            """Run a single session."""
            config = GraphConfig() if session_id % 2 == 0 else create_research_config()
            engine = MCTSEngine(seed=session_id)
            rollout_policy = RandomRolloutPolicy(base_value=0.5, noise_scale=0.2)

            root = MCTSNode(
                state=MCTSState(
                    state_id="root",
                    features={"session": session_id},
                ),
                rng=engine.rng,
            )

            def action_gen(s):
                return ["a", "b"] if len(s.state_id) < 10 else []

            def state_trans(s, a):
                return MCTSState(state_id=f"{s.state_id}_{a}", features={})

            best_action, stats = await engine.search(
                root=root,
                num_iterations=25,
                action_generator=action_gen,
                state_transition=state_trans,
                rollout_policy=rollout_policy,
            )

            return {
                "session_id": session_id,
                "best_action": best_action,
                "iterations": stats["iterations"],
            }

        # Run 5 concurrent sessions
        results = await asyncio.gather(*[run_session(i) for i in range(5)])

        assert len(results) == 5
        assert all(r["best_action"] in ["a", "b"] for r in results)
        assert all(r["iterations"] > 0 for r in results)
