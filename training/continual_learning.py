"""
Continual Learning Module

Implements online learning capabilities including:
- Feedback collection from production
- Incremental training without catastrophic forgetting
- Data drift detection
- A/B testing framework
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class FeedbackSample:
    """Feedback sample from production deployment."""
    sample_id: str
    input_data: Dict[str, Any]
    model_output: Any
    user_feedback: str  # "positive", "negative", "neutral"
    corrected_output: Optional[Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Report of detected data drift."""
    timestamp: str
    drift_type: str  # "feature", "label", "concept"
    severity: float  # 0.0 to 1.0
    affected_features: List[str]
    p_value: float
    recommendation: str


class FeedbackCollector:
    """Collect and manage feedback from production deployments."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feedback collector.

        Args:
            config: Continual learning configuration
        """
        self.config = config
        self.buffer_size = config.get("buffer_size", 100000)
        self.sample_rate = config.get("sample_rate", 0.1)

        self.feedback_buffer = deque(maxlen=self.buffer_size)
        self.storage_path = Path("training/data/feedback")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.statistics = {
            "total_collected": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }

        logger.info(f"FeedbackCollector initialized with buffer size {self.buffer_size}")

    def add_feedback(self, feedback: FeedbackSample) -> None:
        """
        Add feedback sample to buffer.

        Args:
            feedback: Feedback sample
        """
        # Apply sampling rate
        if np.random.random() > self.sample_rate:
            return

        self.feedback_buffer.append(feedback)
        self.statistics["total_collected"] += 1
        self.statistics[feedback.user_feedback] = self.statistics.get(feedback.user_feedback, 0) + 1

        # Persist periodically
        if len(self.feedback_buffer) % 1000 == 0:
            self._persist_feedback()

        logger.debug(f"Added feedback sample {feedback.sample_id}")

    def _persist_feedback(self) -> None:
        """Persist feedback to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.storage_path / f"feedback_{timestamp}.jsonl"

        with open(file_path, 'w') as f:
            for sample in list(self.feedback_buffer)[-1000:]:
                record = {
                    "sample_id": sample.sample_id,
                    "input_data": sample.input_data,
                    "model_output": str(sample.model_output),
                    "user_feedback": sample.user_feedback,
                    "corrected_output": str(sample.corrected_output) if sample.corrected_output else None,
                    "timestamp": sample.timestamp,
                    "metadata": sample.metadata
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Persisted feedback to {file_path}")

    def get_training_samples(self, min_quality: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get high-quality samples for retraining.

        Args:
            min_quality: Minimum quality threshold

        Returns:
            List of training samples
        """
        training_samples = []

        for feedback in self.feedback_buffer:
            # Prioritize samples with corrections
            if feedback.corrected_output is not None:
                quality = 1.0
            elif feedback.user_feedback == "positive":
                quality = 0.8
            elif feedback.user_feedback == "neutral":
                quality = 0.5
            else:
                quality = 0.3

            if quality >= min_quality:
                sample = {
                    "input": feedback.input_data,
                    "target": feedback.corrected_output or feedback.model_output,
                    "weight": quality
                }
                training_samples.append(sample)

        logger.info(f"Retrieved {len(training_samples)} training samples from feedback")
        return training_samples

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback collection statistics."""
        stats = self.statistics.copy()
        stats["buffer_size"] = len(self.feedback_buffer)
        stats["negative_rate"] = stats["negative"] / max(1, stats["total_collected"])
        return stats

    def clear_buffer(self) -> None:
        """Clear feedback buffer after processing."""
        self._persist_feedback()
        self.feedback_buffer.clear()
        logger.info("Feedback buffer cleared")


class IncrementalTrainer:
    """Train models incrementally without catastrophic forgetting."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize incremental trainer.

        Args:
            config: Incremental training configuration
        """
        self.config = config
        self.retrain_threshold = config.get("retrain_threshold", 1000)
        self.forgetting_prevention = config.get("forgetting_prevention", "elastic_weight_consolidation")
        self.ewc_lambda = config.get("ewc_lambda", 1000.0)

        self.fisher_information = {}
        self.optimal_params = {}
        self.accumulated_samples = 0

        logger.info(f"IncrementalTrainer using {self.forgetting_prevention} method")

    def should_retrain(self, num_new_samples: int) -> bool:
        """
        Check if retraining should be triggered.

        Args:
            num_new_samples: Number of new samples available

        Returns:
            True if retraining should occur
        """
        self.accumulated_samples += num_new_samples
        return self.accumulated_samples >= self.retrain_threshold

    def compute_fisher_information(self, model: Any, dataloader: Any) -> Dict[str, Any]:
        """
        Compute Fisher Information Matrix for EWC.

        Args:
            model: Neural network model
            dataloader: Data loader with old task data

        Returns:
            Fisher information for each parameter
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping Fisher computation")
            return {}

        fisher_info = {}

        # Set model to evaluation mode
        model.eval()

        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)

        # Compute Fisher Information
        num_samples = 0
        for batch in dataloader:
            model.zero_grad()

            # Get model output (simplified)
            if hasattr(model, "forward"):
                inputs = batch.get("input_ids", batch.get("features"))
                output = model(inputs)
            else:
                continue

            # Compute log probability for Fisher Information
            # Fisher Information = E[grad(log p(y|x))^2]
            if isinstance(output, dict):
                logits = output.get("logits", None)
                if logits is not None:
                    # Compute log softmax for classification
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Use the predicted class probability
                    pred_classes = logits.argmax(dim=-1)
                    # Get log probability of predicted class
                    batch_size = logits.size(0)
                    log_likelihood = log_probs[range(batch_size), pred_classes].sum()
                else:
                    log_likelihood = output.get("output", torch.zeros(1)).sum()
            else:
                # For regression or other outputs, use negative squared error
                log_likelihood = -0.5 * (output ** 2).sum()

            # Compute gradients
            log_likelihood.backward()

            # Accumulate Fisher information (squared gradients)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2

            num_samples += 1

        # Average over number of batches
        if num_samples > 0:
            for name in fisher_info:
                fisher_info[name] /= num_samples

        self.fisher_information = fisher_info

        # Store optimal parameters
        self.optimal_params = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

        logger.info(f"Computed Fisher Information for {len(fisher_info)} parameters")
        return fisher_info

    def ewc_loss(self, model: Any) -> float:
        """
        Compute EWC regularization loss.

        Args:
            model: Current model

        Returns:
            EWC loss value
        """
        if not HAS_TORCH or not self.fisher_information:
            return 0.0

        ewc_loss = 0.0

        for name, param in model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]

                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return self.ewc_lambda * ewc_loss

    def incremental_update(
        self,
        model: Any,
        new_dataloader: Any,
        old_dataloader: Optional[Any] = None,
        num_epochs: int = 3
    ) -> Dict[str, float]:
        """
        Perform incremental model update.

        Args:
            model: Model to update
            new_dataloader: New task data
            old_dataloader: Old task data (for EWC)
            num_epochs: Number of training epochs

        Returns:
            Training metrics
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping incremental update")
            return {}

        logger.info("Performing incremental model update")

        # Compute Fisher Information from old task
        if old_dataloader and self.forgetting_prevention == "elastic_weight_consolidation":
            self.compute_fisher_information(model, old_dataloader)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        metrics = {"losses": [], "ewc_losses": []}

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_ewc_loss = 0.0

            for batch in new_dataloader:
                optimizer.zero_grad()

                # Forward pass (simplified)
                if hasattr(model, "forward"):
                    output = model(batch)
                else:
                    continue

                # Task loss
                if isinstance(output, dict):
                    task_loss = output.get("loss", torch.zeros(1))
                else:
                    task_loss = output.mean()

                # Add EWC regularization
                ewc_reg = self.ewc_loss(model)

                loss = task_loss + ewc_reg
                loss.backward()
                optimizer.step()

                total_loss += task_loss.item()
                total_ewc_loss += ewc_reg if isinstance(ewc_reg, float) else ewc_reg.item()

            avg_loss = total_loss / len(new_dataloader)
            avg_ewc = total_ewc_loss / len(new_dataloader)

            metrics["losses"].append(avg_loss)
            metrics["ewc_losses"].append(avg_ewc)

            logger.info(f"Incremental epoch {epoch + 1}: Loss={avg_loss:.4f}, EWC={avg_ewc:.4f}")

        # Reset sample counter
        self.accumulated_samples = 0

        return metrics


class DriftDetector:
    """Detect data drift in production."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize drift detector.

        Args:
            config: Drift detection configuration
        """
        self.config = config
        self.window_size = config.get("window_size", 1000)
        self.threshold = config.get("threshold", 0.1)
        self.detection_method = config.get("detection_method", "kolmogorov_smirnov")

        self.reference_distribution = None
        self.current_window = deque(maxlen=self.window_size)
        self.drift_history = []

        logger.info(f"DriftDetector using {self.detection_method} method")

    def set_reference_distribution(self, data: np.ndarray) -> None:
        """
        Set reference distribution from training data.

        Args:
            data: Reference data array
        """
        self.reference_distribution = data
        logger.info(f"Set reference distribution with {len(data)} samples")

    def add_sample(self, sample: np.ndarray) -> Optional[DriftReport]:
        """
        Add new sample and check for drift.

        Args:
            sample: New data sample

        Returns:
            DriftReport if drift detected, None otherwise
        """
        self.current_window.append(sample)

        if len(self.current_window) < self.window_size:
            return None

        # Check for drift
        drift_report = self._detect_drift()

        if drift_report:
            self.drift_history.append(drift_report)
            logger.warning(f"Data drift detected: {drift_report.drift_type}, severity={drift_report.severity:.3f}")

        return drift_report

    def _detect_drift(self) -> Optional[DriftReport]:
        """Detect drift using configured method."""
        if self.reference_distribution is None:
            return None

        current_data = np.array(list(self.current_window))

        if self.detection_method == "kolmogorov_smirnov":
            return self._ks_test(current_data)
        elif self.detection_method == "population_stability_index":
            return self._psi_test(current_data)
        else:
            return self._ks_test(current_data)

    def _ks_test(self, current_data: np.ndarray) -> Optional[DriftReport]:
        """Kolmogorov-Smirnov test for drift detection."""
        from scipy import stats

        # Test each feature
        affected_features = []
        p_values = []

        num_features = min(current_data.shape[1] if current_data.ndim > 1 else 1,
                          self.reference_distribution.shape[1] if self.reference_distribution.ndim > 1 else 1)

        for i in range(num_features):
            if current_data.ndim > 1:
                current_feature = current_data[:, i]
                reference_feature = self.reference_distribution[:, i]
            else:
                current_feature = current_data
                reference_feature = self.reference_distribution

            statistic, p_value = stats.ks_2samp(current_feature, reference_feature)

            if p_value < self.threshold:
                affected_features.append(f"feature_{i}")
                p_values.append(p_value)

        if affected_features:
            severity = 1.0 - np.mean(p_values)
            return DriftReport(
                timestamp=datetime.now().isoformat(),
                drift_type="feature",
                severity=severity,
                affected_features=affected_features,
                p_value=np.mean(p_values),
                recommendation="Consider retraining or feature recalibration"
            )

        return None

    def _psi_test(self, current_data: np.ndarray) -> Optional[DriftReport]:
        """Population Stability Index test."""
        # Simplified PSI calculation
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        if self.reference_distribution.ndim == 1:
            ref_data = self.reference_distribution.reshape(-1, 1)
        else:
            ref_data = self.reference_distribution

        psi_values = []
        affected_features = []

        for i in range(current_data.shape[1]):
            psi = self._calculate_psi(ref_data[:, i], current_data[:, i])
            if psi > 0.25:  # High drift threshold
                affected_features.append(f"feature_{i}")
            psi_values.append(psi)

        if affected_features:
            return DriftReport(
                timestamp=datetime.now().isoformat(),
                drift_type="feature",
                severity=np.mean(psi_values),
                affected_features=affected_features,
                p_value=0.0,  # Not applicable for PSI
                recommendation="Significant distribution shift detected"
            )

        return None

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate PSI between two distributions."""
        # Create buckets based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Add small constant to avoid division by zero
        expected_percents = expected_counts / len(expected) + 1e-6
        actual_percents = actual_counts / len(actual) + 1e-6

        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

        return psi

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {"total_drifts": 0}

        summary = {
            "total_drifts": len(self.drift_history),
            "avg_severity": np.mean([d.severity for d in self.drift_history]),
            "recent_drifts": [
                {
                    "timestamp": d.timestamp,
                    "type": d.drift_type,
                    "severity": d.severity
                }
                for d in self.drift_history[-5:]
            ]
        }

        return summary


class ABTestFramework:
    """A/B testing framework for model updates."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize A/B testing framework.

        Args:
            config: A/B testing configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.traffic_split = config.get("traffic_split", 0.1)
        self.min_samples = config.get("min_samples", 1000)
        self.confidence_level = config.get("confidence_level", 0.95)

        self.tests = {}
        self.results = {}

        logger.info(f"ABTestFramework initialized, traffic split: {self.traffic_split}")

    def create_test(
        self,
        test_name: str,
        model_a: Any,
        model_b: Any,
        metric_fn: Callable[[Any, Any], float]
    ) -> str:
        """
        Create a new A/B test.

        Args:
            test_name: Name of the test
            model_a: Control model (current production)
            model_b: Treatment model (new candidate)
            metric_fn: Function to compute success metric

        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.tests[test_id] = {
            "name": test_name,
            "model_a": model_a,
            "model_b": model_b,
            "metric_fn": metric_fn,
            "samples_a": [],
            "samples_b": [],
            "created_at": datetime.now().isoformat(),
            "status": "running"
        }

        logger.info(f"Created A/B test: {test_id}")
        return test_id

    def assign_group(self, test_id: str, request_id: str) -> str:
        """
        Assign request to test group.

        Args:
            test_id: Test ID
            request_id: Request identifier

        Returns:
            "A" or "B"
        """
        if test_id not in self.tests:
            return "A"  # Default to control

        # Hash-based assignment for consistency
        hash_val = hash(f"{test_id}_{request_id}") % 100
        group = "B" if hash_val < self.traffic_split * 100 else "A"

        return group

    def record_result(
        self,
        test_id: str,
        group: str,
        input_data: Any,
        output: Any,
        success_metric: float
    ) -> None:
        """
        Record test result.

        Args:
            test_id: Test ID
            group: Test group ("A" or "B")
            input_data: Request input
            output: Model output
            success_metric: Success metric value
        """
        if test_id not in self.tests:
            return

        sample = {
            "input": input_data,
            "output": output,
            "metric": success_metric,
            "timestamp": datetime.now().isoformat()
        }

        if group == "A":
            self.tests[test_id]["samples_a"].append(sample)
        else:
            self.tests[test_id]["samples_b"].append(sample)

        # Check if we have enough samples
        if self._has_enough_samples(test_id):
            self._analyze_test(test_id)

    def _has_enough_samples(self, test_id: str) -> bool:
        """Check if test has enough samples for analysis."""
        test = self.tests[test_id]
        return (
            len(test["samples_a"]) >= self.min_samples and
            len(test["samples_b"]) >= self.min_samples * self.traffic_split
        )

    def _analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        from scipy import stats

        test = self.tests[test_id]

        metrics_a = [s["metric"] for s in test["samples_a"]]
        metrics_b = [s["metric"] for s in test["samples_b"]]

        mean_a = np.mean(metrics_a)
        mean_b = np.mean(metrics_b)

        # Statistical significance test
        t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)

        is_significant = p_value < (1 - self.confidence_level)
        improvement = (mean_b - mean_a) / mean_a if mean_a > 0 else 0

        result = {
            "test_id": test_id,
            "mean_control": mean_a,
            "mean_treatment": mean_b,
            "improvement": improvement,
            "p_value": p_value,
            "is_significant": is_significant,
            "recommendation": "Deploy B" if is_significant and improvement > 0 else "Keep A",
            "analyzed_at": datetime.now().isoformat()
        }

        self.results[test_id] = result
        test["status"] = "analyzed"

        logger.info(f"A/B test {test_id} analyzed: {result['recommendation']}")
        return result

    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of a test."""
        if test_id not in self.tests:
            return {"error": "Test not found"}

        test = self.tests[test_id]

        status = {
            "test_id": test_id,
            "name": test["name"],
            "status": test["status"],
            "samples_control": len(test["samples_a"]),
            "samples_treatment": len(test["samples_b"]),
            "progress": min(len(test["samples_a"]) / self.min_samples, 1.0)
        }

        if test_id in self.results:
            status["result"] = self.results[test_id]

        return status

    def end_test(self, test_id: str) -> Dict[str, Any]:
        """End an A/B test and return final results."""
        if test_id not in self.tests:
            return {"error": "Test not found"}

        if self.tests[test_id]["status"] != "analyzed":
            result = self._analyze_test(test_id)
        else:
            result = self.results[test_id]

        self.tests[test_id]["status"] = "completed"

        logger.info(f"A/B test {test_id} completed")
        return result


if __name__ == "__main__":
    # Test continual learning module
    logging.basicConfig(level=logging.INFO)

    logger.info("Testing Continual Learning Module")

    # Load config
    config_path = "training/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cl_config = config.get("continual_learning", {})

    # Test FeedbackCollector
    feedback_config = cl_config.get("feedback", {})
    collector = FeedbackCollector(feedback_config)

    # Add some feedback samples
    for i in range(100):
        feedback = FeedbackSample(
            sample_id=f"sample_{i}",
            input_data={"query": f"Test query {i}"},
            model_output=f"Result {i}",
            user_feedback=["positive", "negative", "neutral"][i % 3],
            corrected_output=f"Corrected {i}" if i % 5 == 0 else None,
            timestamp=float(i)
        )
        collector.add_feedback(feedback)

    stats = collector.get_statistics()
    logger.info(f"Feedback statistics: {stats}")

    # Test IncrementalTrainer
    incremental_config = cl_config.get("incremental", {})
    trainer = IncrementalTrainer(incremental_config)

    should_retrain = trainer.should_retrain(500)
    logger.info(f"Should retrain (500 samples): {should_retrain}")

    should_retrain = trainer.should_retrain(600)
    logger.info(f"Should retrain (1100 samples): {should_retrain}")

    # Test DriftDetector
    drift_config = cl_config.get("drift_detection", {})
    detector = DriftDetector(drift_config)

    # Set reference distribution
    reference_data = np.random.randn(1000, 5)
    detector.set_reference_distribution(reference_data)

    # Add samples (with drift)
    for i in range(1000):
        # Simulate drift by shifting distribution
        sample = np.random.randn(5) + (i / 1000) * 0.5  # Gradual drift
        drift_report = detector.add_sample(sample)

        if drift_report:
            logger.info(f"Drift detected at sample {i}")

    drift_summary = detector.get_drift_summary()
    logger.info(f"Drift summary: {drift_summary}")

    # Test ABTestFramework
    ab_config = cl_config.get("ab_testing", {})
    ab_framework = ABTestFramework(ab_config)

    # Create test
    test_id = ab_framework.create_test(
        "model_v2_test",
        model_a="model_v1",
        model_b="model_v2",
        metric_fn=lambda inp, out: np.random.random()
    )

    # Simulate test traffic
    for i in range(1500):
        group = ab_framework.assign_group(test_id, f"request_{i}")
        metric = 0.7 + (0.1 if group == "B" else 0)  # B is slightly better
        metric += np.random.randn() * 0.1

        ab_framework.record_result(test_id, group, {"req": i}, "output", metric)

    status = ab_framework.get_test_status(test_id)
    logger.info(f"A/B test status: {status}")

    logger.info("Continual Learning Module test complete")
