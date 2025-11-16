"""
Mock implementations of external services for testing.

These mocks provide deterministic, offline-capable versions of:
- Pinecone vector database
- Braintrust experiment tracking
- Weights & Biases
- LLM providers
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

logger = logging.getLogger(__name__)


@dataclass
class MockVectorRecord:
    """Represents a vector stored in mock Pinecone."""

    id: str
    values: List[float]
    metadata: Dict[str, Any]
    namespace: str = "default"


class MockPineconeClient:
    """
    In-memory mock of Pinecone vector database.

    Simulates:
    - Vector upsert operations
    - Similarity search queries
    - Namespace isolation
    - Connection lifecycle
    """

    def __init__(self, dimension: int = 10):
        """
        Initialize mock Pinecone client.

        Args:
            dimension: Vector dimension (default 10 for meta-controller features)
        """
        self.dimension = dimension
        self.vectors: Dict[str, Dict[str, MockVectorRecord]] = {}  # namespace -> id -> record
        self._connected = False
        self._operations_log: List[Dict[str, Any]] = []

    def connect(self):
        """Simulate connection to Pinecone."""
        self._connected = True
        self._log_operation("connect", {})
        logger.info("MockPineconeClient connected")

    def disconnect(self):
        """Simulate disconnection."""
        self._connected = False
        self._log_operation("disconnect", {})

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """
        Upsert vectors into mock database.

        Args:
            vectors: List of vector dictionaries with 'id', 'values', 'metadata'
            namespace: Namespace to store vectors in

        Returns:
            Upsert result summary
        """
        if not self._connected:
            raise ConnectionError("Not connected to Pinecone")

        if namespace not in self.vectors:
            self.vectors[namespace] = {}

        upserted_count = 0
        for vec in vectors:
            if len(vec["values"]) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {len(vec['values'])}"
                )

            record = MockVectorRecord(
                id=vec["id"],
                values=vec["values"],
                metadata=vec.get("metadata", {}),
                namespace=namespace,
            )
            self.vectors[namespace][vec["id"]] = record
            upserted_count += 1

        self._log_operation("upsert", {"namespace": namespace, "count": upserted_count})

        return {"upserted_count": upserted_count}

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "default",
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Query for similar vectors.

        Uses simple cosine similarity for testing purposes.

        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Namespace to search
            include_metadata: Whether to include metadata in results

        Returns:
            Query results with matches
        """
        if not self._connected:
            raise ConnectionError("Not connected to Pinecone")

        if len(vector) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}")

        if namespace not in self.vectors:
            return {"matches": []}

        # Calculate similarities
        matches = []
        for record in self.vectors[namespace].values():
            similarity = self._cosine_similarity(vector, record.values)
            match_data = {
                "id": record.id,
                "score": similarity,
            }
            if include_metadata:
                match_data["metadata"] = record.metadata
            matches.append(match_data)

        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x["score"], reverse=True)
        matches = matches[:top_k]

        self._log_operation("query", {"namespace": namespace, "top_k": top_k, "results": len(matches)})

        return {"matches": matches}

    def delete(self, ids: List[str], namespace: str = "default") -> Dict[str, Any]:
        """Delete vectors by ID."""
        if not self._connected:
            raise ConnectionError("Not connected to Pinecone")

        deleted = 0
        if namespace in self.vectors:
            for vec_id in ids:
                if vec_id in self.vectors[namespace]:
                    del self.vectors[namespace][vec_id]
                    deleted += 1

        self._log_operation("delete", {"namespace": namespace, "count": deleted})
        return {"deleted_count": deleted}

    def describe_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        total_vectors = sum(len(ns_vectors) for ns_vectors in self.vectors.values())
        return {
            "dimension": self.dimension,
            "index_fullness": 0.0,
            "total_vector_count": total_vectors,
            "namespaces": {ns: {"vector_count": len(vecs)} for ns, vecs in self.vectors.items()},
        }

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _log_operation(self, operation: str, details: Dict[str, Any]):
        """Log operation for testing verification."""
        self._operations_log.append(
            {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "details": details,
            }
        )

    def get_operations_log(self) -> List[Dict[str, Any]]:
        """Get all operations for test verification."""
        return self._operations_log

    def reset(self):
        """Reset mock to initial state."""
        self.vectors = {}
        self._operations_log = []
        self._connected = False


@dataclass
class MockExperiment:
    """Represents a tracked experiment."""

    id: str
    name: str
    project: str
    start_time: datetime
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


class MockBraintrustTracker:
    """
    Mock Braintrust experiment tracker.

    Simulates:
    - Experiment creation
    - Metric logging
    - Hyperparameter tracking
    - Artifact management
    """

    def __init__(self, project_name: str = "test-project"):
        """
        Initialize mock Braintrust tracker.

        Args:
            project_name: Project name for experiments
        """
        self.project_name = project_name
        self.experiments: Dict[str, MockExperiment] = {}
        self.current_experiment: Optional[MockExperiment] = None
        self._experiment_counter = 0

    def init_experiment(self, name: Optional[str] = None) -> str:
        """
        Initialize a new experiment.

        Args:
            name: Experiment name (auto-generated if None)

        Returns:
            Experiment ID
        """
        self._experiment_counter += 1
        exp_id = f"exp_{self._experiment_counter}"
        exp_name = name or f"experiment_{self._experiment_counter}"

        experiment = MockExperiment(
            id=exp_id,
            name=exp_name,
            project=self.project_name,
            start_time=datetime.now(),
        )

        self.experiments[exp_id] = experiment
        self.current_experiment = experiment

        logger.info(f"MockBraintrust: Initialized experiment {exp_name}")
        return exp_id

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.

        Args:
            name: Metric name
            value: Metric value
            step: Optional training step
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment. Call init_experiment first.")

        if name not in self.current_experiment.metrics:
            self.current_experiment.metrics[name] = []

        self.current_experiment.metrics[name].append(value)
        logger.debug(f"MockBraintrust: Logged {name}={value}")

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment.")

        self.current_experiment.hyperparameters.update(params)
        logger.debug(f"MockBraintrust: Logged hyperparameters {params}")

    def log_artifact(self, artifact_path: str, name: Optional[str] = None):
        """
        Log an artifact (model, data file, etc.).

        Args:
            artifact_path: Path to artifact
            name: Optional artifact name
        """
        if not self.current_experiment:
            raise RuntimeError("No active experiment.")

        self.current_experiment.artifacts.append(artifact_path)
        logger.debug(f"MockBraintrust: Logged artifact {artifact_path}")

    def end_experiment(self) -> Dict[str, Any]:
        """
        End current experiment.

        Returns:
            Experiment summary
        """
        if not self.current_experiment:
            return {}

        summary = {
            "id": self.current_experiment.id,
            "name": self.current_experiment.name,
            "metrics": self.current_experiment.metrics,
            "hyperparameters": self.current_experiment.hyperparameters,
            "artifacts": self.current_experiment.artifacts,
        }

        self.current_experiment = None
        return summary

    def get_experiment_summary(self, exp_id: str) -> Dict[str, Any]:
        """Get summary of a specific experiment."""
        if exp_id not in self.experiments:
            raise KeyError(f"Experiment {exp_id} not found")

        exp = self.experiments[exp_id]
        return {
            "id": exp.id,
            "name": exp.name,
            "project": exp.project,
            "metrics": exp.metrics,
            "hyperparameters": exp.hyperparameters,
            "artifacts": exp.artifacts,
        }


class MockWandBRun:
    """
    Mock Weights & Biases run tracker.

    Simulates:
    - Run initialization
    - Metric logging
    - Configuration tracking
    - Artifact logging
    """

    def __init__(self, project: str = "test-project", name: Optional[str] = None):
        """
        Initialize mock W&B run.

        Args:
            project: Project name
            name: Run name
        """
        self.project = project
        self.name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, List[float]] = {}
        self.summary: Dict[str, Any] = {}
        self._step = 0
        self._finished = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            data: Dictionary of metrics to log
            step: Optional step number
        """
        if self._finished:
            raise RuntimeError("Run already finished")

        if step is not None:
            self._step = step

        for key, value in data.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

        self._step += 1
        logger.debug(f"MockWandB: Logged {data}")

    def update_config(self, config: Dict[str, Any]):
        """Update run configuration."""
        self.config.update(config)

    def finish(self):
        """Finish the run."""
        self._finished = True
        logger.info(f"MockWandB: Run {self.name} finished")

    def get_summary(self) -> Dict[str, Any]:
        """Get run summary."""
        return {
            "project": self.project,
            "name": self.name,
            "config": self.config,
            "metrics": self.metrics,
            "steps": self._step,
        }


@dataclass
class MockLLMResponse:
    """Mock LLM response."""

    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str = "stop"


class MockLLMClient:
    """
    Mock LLM client for deterministic testing.

    Provides:
    - Predefined responses
    - Response patterns based on input
    - Configurable behavior (delays, errors, etc.)
    """

    def __init__(self, provider: str = "mock"):
        """
        Initialize mock LLM client.

        Args:
            provider: Provider name for identification
        """
        self.provider = provider
        self.responses: List[MockLLMResponse] = []
        self.call_history: List[Dict[str, Any]] = []
        self._response_index = 0
        self._should_fail = False
        self._failure_message = ""

    def set_responses(self, responses: List[str]):
        """
        Set predefined responses.

        Args:
            responses: List of response strings to return in order
        """
        self.responses = [
            MockLLMResponse(
                content=resp,
                model=f"{self.provider}-mock",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            )
            for resp in responses
        ]
        self._response_index = 0

    def set_failure_mode(self, should_fail: bool, message: str = "Mock failure"):
        """
        Configure client to fail on next call.

        Args:
            should_fail: Whether to fail
            message: Failure message
        """
        self._should_fail = should_fail
        self._failure_message = message

    async def generate(self, prompt: str, **kwargs) -> MockLLMResponse:
        """
        Generate mock response.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            MockLLMResponse
        """
        self.call_history.append(
            {
                "prompt": prompt,
                "kwargs": kwargs,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if self._should_fail:
            self._should_fail = False
            raise Exception(self._failure_message)

        if self.responses and self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
            return response

        # Default response based on prompt
        return MockLLMResponse(
            content=f"Mock response for: {prompt[:100]}...",
            model=f"{self.provider}-mock",
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 50, "total_tokens": len(prompt) // 4 + 50},
        )

    def get_call_count(self) -> int:
        """Get number of calls made."""
        return len(self.call_history)

    def get_last_call(self) -> Optional[Dict[str, Any]]:
        """Get the last call made."""
        return self.call_history[-1] if self.call_history else None

    def reset(self):
        """Reset mock to initial state."""
        self.responses = []
        self.call_history = []
        self._response_index = 0
        self._should_fail = False


def create_mock_pinecone() -> MockPineconeClient:
    """Factory function for creating mock Pinecone client."""
    client = MockPineconeClient(dimension=10)
    client.connect()
    return client


def create_mock_braintrust(project: str = "test-project") -> MockBraintrustTracker:
    """Factory function for creating mock Braintrust tracker."""
    return MockBraintrustTracker(project_name=project)


def create_mock_wandb(project: str = "test-project", name: Optional[str] = None) -> MockWandBRun:
    """Factory function for creating mock W&B run."""
    return MockWandBRun(project=project, name=name)


def create_mock_llm(provider: str = "mock", responses: Optional[List[str]] = None) -> MockLLMClient:
    """Factory function for creating mock LLM client."""
    client = MockLLMClient(provider=provider)
    if responses:
        client.set_responses(responses)
    return client
