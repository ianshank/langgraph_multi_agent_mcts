"""
Experiment Tracking Integration Tests.

Tests for:
- Braintrust experiment tracking
- Weights & Biases (W&B) integration
- Metric logging and hyperparameter tracking
- Model artifact versioning

Expected outcomes:
- Full experiment auditability
- Training metrics logged
- Model versions tracked
"""


import pytest

from tests.mocks.mock_external_services import (
    create_mock_braintrust,
    create_mock_wandb,
)


@pytest.fixture
def braintrust_tracker():
    """Create mock Braintrust tracker."""
    return create_mock_braintrust(project="neural-meta-controller")


@pytest.fixture
def wandb_run():
    """Create mock W&B run."""
    return create_mock_wandb(project="mcts-training", name="rnn_training_v1")


@pytest.fixture
def training_config():
    """Sample training configuration."""
    return {
        "model_type": "rnn",
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "seed": 42,
        "dropout": 0.1,
    }


@pytest.fixture
def training_metrics():
    """Sample training metrics."""
    return [
        {"epoch": 1, "train_loss": 0.85, "val_loss": 0.90, "accuracy": 0.65},
        {"epoch": 2, "train_loss": 0.72, "val_loss": 0.78, "accuracy": 0.72},
        {"epoch": 3, "train_loss": 0.61, "val_loss": 0.68, "accuracy": 0.78},
        {"epoch": 4, "train_loss": 0.52, "val_loss": 0.62, "accuracy": 0.82},
        {"epoch": 5, "train_loss": 0.45, "val_loss": 0.58, "accuracy": 0.85},
    ]


class TestBraintrustExperimentInit:
    """Test Braintrust experiment initialization."""

    @pytest.mark.training
    def test_experiment_creation(self, braintrust_tracker):
        """Should create experiment successfully."""
        exp_id = braintrust_tracker.init_experiment(name="test_experiment")

        assert exp_id is not None
        assert "exp_" in exp_id
        assert braintrust_tracker.current_experiment is not None

    @pytest.mark.training
    def test_experiment_naming(self, braintrust_tracker):
        """Experiments should have descriptive names."""
        exp_id = braintrust_tracker.init_experiment(name="rnn_meta_controller_v2_lr0001")

        summary = braintrust_tracker.get_experiment_summary(exp_id)

        assert summary["name"] == "rnn_meta_controller_v2_lr0001"
        assert summary["project"] == "neural-meta-controller"

    @pytest.mark.training
    def test_multiple_experiments(self, braintrust_tracker):
        """Should track multiple experiments."""
        exp1 = braintrust_tracker.init_experiment(name="exp_1")
        braintrust_tracker.end_experiment()

        exp2 = braintrust_tracker.init_experiment(name="exp_2")
        braintrust_tracker.end_experiment()

        assert exp1 != exp2
        assert len(braintrust_tracker.experiments) == 2


class TestBraintrustMetricLogging:
    """Test metric logging to Braintrust."""

    @pytest.mark.training
    def test_log_training_loss(self, braintrust_tracker, training_metrics):
        """Should log training loss metrics."""
        braintrust_tracker.init_experiment(name="loss_tracking")

        for metric in training_metrics:
            braintrust_tracker.log_metric("train_loss", metric["train_loss"])
            braintrust_tracker.log_metric("val_loss", metric["val_loss"])

        exp = braintrust_tracker.current_experiment

        assert "train_loss" in exp.metrics
        assert "val_loss" in exp.metrics
        assert len(exp.metrics["train_loss"]) == 5
        assert exp.metrics["train_loss"][-1] == 0.45  # Last value

    @pytest.mark.training
    def test_log_accuracy_metrics(self, braintrust_tracker, training_metrics):
        """Should log accuracy metrics over time."""
        braintrust_tracker.init_experiment(name="accuracy_tracking")

        for metric in training_metrics:
            braintrust_tracker.log_metric("accuracy", metric["accuracy"])

        exp = braintrust_tracker.current_experiment

        # Verify accuracy improves over epochs
        accuracies = exp.metrics["accuracy"]
        assert accuracies[-1] > accuracies[0]  # Should improve
        assert accuracies[-1] == 0.85  # Final accuracy

    @pytest.mark.training
    def test_log_custom_metrics(self, braintrust_tracker):
        """Should log custom application-specific metrics."""
        braintrust_tracker.init_experiment(name="custom_metrics")

        braintrust_tracker.log_metric("consensus_score", 0.83)
        braintrust_tracker.log_metric("agent_routing_accuracy", 0.79)
        braintrust_tracker.log_metric("mcts_simulation_time_ms", 1250)

        exp = braintrust_tracker.current_experiment

        assert "consensus_score" in exp.metrics
        assert "agent_routing_accuracy" in exp.metrics
        assert "mcts_simulation_time_ms" in exp.metrics


class TestBraintrustHyperparameters:
    """Test hyperparameter tracking."""

    @pytest.mark.training
    def test_log_hyperparameters(self, braintrust_tracker, training_config):
        """Should log all hyperparameters."""
        braintrust_tracker.init_experiment(name="hyperparams_test")

        braintrust_tracker.log_hyperparameters(training_config)

        exp = braintrust_tracker.current_experiment

        assert exp.hyperparameters["model_type"] == "rnn"
        assert exp.hyperparameters["hidden_size"] == 64
        assert exp.hyperparameters["learning_rate"] == 0.001
        assert exp.hyperparameters["epochs"] == 10

    @pytest.mark.training
    def test_update_hyperparameters(self, braintrust_tracker):
        """Should allow updating hyperparameters."""
        braintrust_tracker.init_experiment(name="hyperparam_update")

        initial_params = {"learning_rate": 0.001, "batch_size": 32}
        braintrust_tracker.log_hyperparameters(initial_params)

        # Update after tuning
        updated_params = {"learning_rate": 0.0005}  # Reduced LR
        braintrust_tracker.log_hyperparameters(updated_params)

        exp = braintrust_tracker.current_experiment

        assert exp.hyperparameters["learning_rate"] == 0.0005
        assert exp.hyperparameters["batch_size"] == 32  # Preserved


class TestBraintrustArtifacts:
    """Test artifact logging."""

    @pytest.mark.training
    def test_log_model_checkpoint(self, braintrust_tracker):
        """Should log model checkpoint artifacts."""
        braintrust_tracker.init_experiment(name="checkpoint_test")

        braintrust_tracker.log_artifact(
            "/models/rnn_meta_controller_epoch_10.pt",
            name="best_model",
        )

        exp = braintrust_tracker.current_experiment

        assert "/models/rnn_meta_controller_epoch_10.pt" in exp.artifacts

    @pytest.mark.training
    def test_log_multiple_artifacts(self, braintrust_tracker):
        """Should log multiple artifacts."""
        braintrust_tracker.init_experiment(name="multi_artifact")

        braintrust_tracker.log_artifact("/models/model.pt")
        braintrust_tracker.log_artifact("/configs/training_config.yaml")
        braintrust_tracker.log_artifact("/data/train_split.json")

        exp = braintrust_tracker.current_experiment

        assert len(exp.artifacts) == 3


class TestBraintrustExperimentSummary:
    """Test experiment summary generation."""

    @pytest.mark.training
    def test_experiment_summary(self, braintrust_tracker, training_config, training_metrics):
        """Should generate comprehensive experiment summary."""
        braintrust_tracker.init_experiment(name="full_experiment")

        # Log everything
        braintrust_tracker.log_hyperparameters(training_config)

        for metric in training_metrics:
            braintrust_tracker.log_metric("train_loss", metric["train_loss"])
            braintrust_tracker.log_metric("accuracy", metric["accuracy"])

        braintrust_tracker.log_artifact("/models/final_model.pt")

        summary = braintrust_tracker.end_experiment()

        # Verify summary completeness
        assert "id" in summary
        assert "name" in summary
        assert "metrics" in summary
        assert "hyperparameters" in summary
        assert "artifacts" in summary

        # Verify data
        assert summary["hyperparameters"]["epochs"] == 10
        assert len(summary["metrics"]["train_loss"]) == 5
        assert len(summary["artifacts"]) == 1


class TestWandBIntegration:
    """Test Weights & Biases integration."""

    @pytest.mark.training
    def test_wandb_run_initialization(self, wandb_run):
        """Should initialize W&B run."""
        assert wandb_run.project == "mcts-training"
        assert wandb_run.name == "rnn_training_v1"
        assert wandb_run._finished is False

    @pytest.mark.training
    def test_wandb_log_metrics(self, wandb_run, training_metrics):
        """Should log metrics to W&B."""
        for i, metric in enumerate(training_metrics):
            wandb_run.log(
                {
                    "train_loss": metric["train_loss"],
                    "val_loss": metric["val_loss"],
                    "accuracy": metric["accuracy"],
                },
                step=i,
            )

        summary = wandb_run.get_summary()

        assert "train_loss" in summary["metrics"]
        assert "val_loss" in summary["metrics"]
        assert "accuracy" in summary["metrics"]
        assert summary["steps"] == 5

    @pytest.mark.training
    def test_wandb_config_update(self, wandb_run, training_config):
        """Should update W&B config."""
        wandb_run.update_config(training_config)

        summary = wandb_run.get_summary()

        assert summary["config"]["model_type"] == "rnn"
        assert summary["config"]["hidden_size"] == 64
        assert summary["config"]["learning_rate"] == 0.001

    @pytest.mark.training
    def test_wandb_run_finish(self, wandb_run):
        """Should finish W&B run properly."""
        wandb_run.log({"test_metric": 1.0})
        wandb_run.finish()

        assert wandb_run._finished is True

        # Should not allow logging after finish
        with pytest.raises(RuntimeError):
            wandb_run.log({"another_metric": 2.0})


class TestExperimentComparison:
    """Test experiment comparison and analysis."""

    @pytest.mark.training
    def test_compare_experiments(self, braintrust_tracker):
        """Should enable experiment comparison."""
        # Experiment 1: Lower LR
        exp1_id = braintrust_tracker.init_experiment(name="lr_0001")
        braintrust_tracker.log_hyperparameters({"learning_rate": 0.001})
        braintrust_tracker.log_metric("final_accuracy", 0.82)
        braintrust_tracker.end_experiment()

        # Experiment 2: Higher LR
        exp2_id = braintrust_tracker.init_experiment(name="lr_0005")
        braintrust_tracker.log_hyperparameters({"learning_rate": 0.005})
        braintrust_tracker.log_metric("final_accuracy", 0.78)
        braintrust_tracker.end_experiment()

        # Compare
        exp1_summary = braintrust_tracker.get_experiment_summary(exp1_id)
        exp2_summary = braintrust_tracker.get_experiment_summary(exp2_id)

        # Lower LR had better accuracy
        assert exp1_summary["metrics"]["final_accuracy"][0] > exp2_summary["metrics"]["final_accuracy"][0]

    @pytest.mark.training
    def test_best_experiment_selection(self, braintrust_tracker):
        """Should identify best performing experiment."""
        experiments = []

        for i in range(3):
            exp_id = braintrust_tracker.init_experiment(name=f"exp_{i}")
            braintrust_tracker.log_metric("accuracy", 0.75 + i * 0.05)
            braintrust_tracker.end_experiment()
            experiments.append(exp_id)

        # Find best
        best_acc = 0.0
        best_exp = None

        for exp_id in experiments:
            summary = braintrust_tracker.get_experiment_summary(exp_id)
            acc = summary["metrics"]["accuracy"][0]
            if acc > best_acc:
                best_acc = acc
                best_exp = exp_id

        assert best_acc == 0.85
        assert best_exp == experiments[2]


class TestOfflineMode:
    """Test offline experiment tracking."""

    @pytest.mark.training
    def test_wandb_offline_mode(self):
        """W&B should support offline mode for testing."""
        # Simulate offline mode

        # Mock run should work without network
        offline_run = create_mock_wandb(project="offline-test")
        offline_run.log({"metric": 1.0})

        assert offline_run._finished is False

    @pytest.mark.training
    def test_metrics_buffering(self, braintrust_tracker):
        """Metrics should be buffered when service unavailable."""
        braintrust_tracker.init_experiment(name="buffered_metrics")

        # Log multiple metrics (would be buffered if offline)
        for i in range(100):
            braintrust_tracker.log_metric("iteration_loss", 1.0 / (i + 1))

        exp = braintrust_tracker.current_experiment

        assert len(exp.metrics["iteration_loss"]) == 100


class TestTrainingPipelineTracking:
    """Test full training pipeline with experiment tracking."""

    @pytest.mark.training
    @pytest.mark.integration
    def test_complete_training_run_tracking(self, braintrust_tracker, wandb_run, training_config, training_metrics):
        """Should track complete training run."""
        # Initialize tracking
        braintrust_tracker.init_experiment(name="complete_run")
        braintrust_tracker.log_hyperparameters(training_config)
        wandb_run.update_config(training_config)

        # Simulate training
        for epoch, metric in enumerate(training_metrics):
            # Log to Braintrust
            braintrust_tracker.log_metric("train_loss", metric["train_loss"])
            braintrust_tracker.log_metric("val_loss", metric["val_loss"])
            braintrust_tracker.log_metric("accuracy", metric["accuracy"])

            # Log to W&B
            wandb_run.log(
                {
                    "train_loss": metric["train_loss"],
                    "val_loss": metric["val_loss"],
                    "accuracy": metric["accuracy"],
                },
                step=epoch,
            )

        # Save best model
        braintrust_tracker.log_artifact("/models/best_model.pt")

        # Finalize
        bt_summary = braintrust_tracker.end_experiment()
        wandb_run.finish()
        wb_summary = wandb_run.get_summary()

        # Verify both trackers have consistent data
        assert len(bt_summary["metrics"]["train_loss"]) == len(wb_summary["metrics"]["train_loss"])
        assert bt_summary["hyperparameters"]["epochs"] == wb_summary["config"]["epochs"]

    @pytest.mark.training
    def test_training_cost_tracking(self, braintrust_tracker):
        """Should track training cost metrics."""
        braintrust_tracker.init_experiment(name="cost_tracking")

        # Log cost-related metrics
        braintrust_tracker.log_metric("gpu_hours", 4.5)
        braintrust_tracker.log_metric("estimated_cost_usd", 9.0)  # $2/GPU-hour
        braintrust_tracker.log_metric("samples_processed", 10000)

        exp = braintrust_tracker.current_experiment

        # Verify cost is under budget
        cost = exp.metrics["estimated_cost_usd"][0]
        assert cost < 100, f"Training cost ${cost} exceeds $100 budget"

    @pytest.mark.training
    def test_reproducibility_tracking(self, braintrust_tracker, training_config):
        """Should track info needed for reproducibility."""
        braintrust_tracker.init_experiment(name="reproducibility")

        reproducibility_info = {
            **training_config,
            "random_seed": 42,
            "pytorch_version": "2.0.0",
            "transformers_version": "4.30.0",
            "dataset_version": "DABStep_v1.0",
            "git_commit": "abc123def456",
        }

        braintrust_tracker.log_hyperparameters(reproducibility_info)

        exp = braintrust_tracker.current_experiment

        # All reproducibility info should be logged
        assert exp.hyperparameters["random_seed"] == 42
        assert exp.hyperparameters["pytorch_version"] == "2.0.0"
        assert exp.hyperparameters["dataset_version"] == "DABStep_v1.0"
        assert exp.hyperparameters["git_commit"] == "abc123def456"
