"""
Framework Service for REST API Integration.

Provides a clean interface between the REST API and the underlying
LangGraph Multi-Agent MCTS framework.

Best Practices 2025:
- Dependency injection via factory pattern
- Configuration-driven (no hardcoded values)
- Async-first design
- Comprehensive error handling
- Type-safe interfaces
- Lazy initialization for resource efficiency
- Structured logging with correlation IDs
- RAG integration for context-aware responses
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.config.settings import Settings, get_settings

# Try to import structured logging
try:
    from src.observability.logging import get_correlation_id, get_structured_logger, set_correlation_id

    _structured_logger = get_structured_logger(__name__)
    _HAS_STRUCTURED_LOGGING = True
except ImportError:
    _structured_logger = logging.getLogger(__name__)
    _HAS_STRUCTURED_LOGGING = False

    def get_correlation_id() -> str:
        return "unknown"

    def set_correlation_id(cid: str) -> None:
        pass


class FrameworkState(Enum):
    """Framework lifecycle states."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass(frozen=True)
class FrameworkConfig:
    """
    Configuration for the framework service.

    All values are loaded from Settings - no hardcoded defaults.
    """

    mcts_enabled: bool
    mcts_iterations: int
    mcts_exploration_weight: float
    seed: int | None
    max_iterations: int
    consensus_threshold: float
    top_k_retrieval: int
    enable_parallel_agents: bool
    timeout_seconds: float

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> FrameworkConfig:
        """Create config from application settings."""
        settings = settings or get_settings()
        return cls(
            mcts_enabled=settings.MCTS_ENABLED,
            mcts_iterations=settings.MCTS_ITERATIONS,
            mcts_exploration_weight=settings.MCTS_C,
            seed=settings.SEED,
            max_iterations=settings.FRAMEWORK_MAX_ITERATIONS,
            consensus_threshold=settings.FRAMEWORK_CONSENSUS_THRESHOLD,
            top_k_retrieval=settings.FRAMEWORK_TOP_K_RETRIEVAL,
            enable_parallel_agents=settings.FRAMEWORK_ENABLE_PARALLEL_AGENTS,
            timeout_seconds=float(settings.HTTP_TIMEOUT_SECONDS),
        )


@dataclass
class QueryResult:
    """Result of a framework query."""

    response: str
    confidence: float
    agents_used: list[str]
    mcts_stats: dict[str, Any] | None
    processing_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "response": self.response,
            "confidence": self.confidence,
            "agents_used": self.agents_used,
            "mcts_stats": self.mcts_stats,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


@runtime_checkable
class FrameworkProtocol(Protocol):
    """Protocol defining the framework interface."""

    async def process(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ) -> dict[str, Any]: ...


class FrameworkService:
    """
    Service layer for the LangGraph Multi-Agent MCTS Framework.

    Provides:
    - Lazy initialization
    - Thread-safe singleton access
    - Graceful degradation
    - Health monitoring
    - Metrics collection
    - RAG integration for context-aware responses
    - Structured logging with correlation IDs
    """

    _instance: FrameworkService | None = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        config: FrameworkConfig | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize framework service.

        Args:
            config: Framework configuration (uses settings if not provided)
            settings: Application settings (uses global if not provided)
        """
        self._settings = settings or get_settings()
        self._config = config or FrameworkConfig.from_settings(self._settings)
        self._logger = _structured_logger
        self._state = FrameworkState.UNINITIALIZED
        self._framework: FrameworkProtocol | None = None
        self._rag_retriever: Any | None = None
        self._init_error: Exception | None = None
        self._start_time: float | None = None
        self._request_count: int = 0
        self._error_count: int = 0
        self._total_processing_time_ms: float = 0.0

    @classmethod
    async def get_instance(
        cls,
        config: FrameworkConfig | None = None,
        settings: Settings | None = None,
    ) -> FrameworkService:
        """
        Get or create the singleton framework service instance.

        Thread-safe singleton pattern with async support.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config=config, settings=settings)
            return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.shutdown()
            cls._instance = None

    async def initialize(self) -> bool:
        """
        Initialize the framework components.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._state == FrameworkState.READY:
            return True

        if self._state == FrameworkState.INITIALIZING:
            # Wait for ongoing initialization
            while self._state == FrameworkState.INITIALIZING:
                await asyncio.sleep(0.1)
            return self._state == FrameworkState.READY

        self._state = FrameworkState.INITIALIZING
        self._start_time = time.time()
        init_start = time.perf_counter()
        correlation_id = get_correlation_id()

        if _HAS_STRUCTURED_LOGGING:
            self._logger.info(
                "Initializing framework service",
                correlation_id=correlation_id,
                mcts_enabled=self._config.mcts_enabled,
                mcts_iterations=self._config.mcts_iterations,
            )
        else:
            self._logger.info("Initializing framework service...")

        try:
            # Import framework components
            from src.framework.factories import LLMClientFactory
            from src.framework.graph import IntegratedFramework

            # Create LLM client
            llm_factory = LLMClientFactory(settings=self._settings)
            llm_client = None

            try:
                llm_client = llm_factory.create_from_settings()
                if _HAS_STRUCTURED_LOGGING:
                    self._logger.info(
                        "LLM client created successfully",
                        correlation_id=correlation_id,
                        provider=self._settings.LLM_PROVIDER.value,
                    )
            except Exception as e:
                if _HAS_STRUCTURED_LOGGING:
                    self._logger.warning(
                        "LLM client creation failed, using mock",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                else:
                    self._logger.warning(f"LLM client creation failed: {e}, using mock")
                llm_client = MockLLMClient()

            # Initialize RAG retriever
            try:
                from src.api.rag_retriever import create_rag_retriever

                self._rag_retriever = create_rag_retriever(settings=self._settings)
                await self._rag_retriever.initialize()
                if _HAS_STRUCTURED_LOGGING:
                    self._logger.info(
                        "RAG retriever initialized",
                        correlation_id=correlation_id,
                        backends=self._rag_retriever.available_backends,
                    )
            except Exception as e:
                if _HAS_STRUCTURED_LOGGING:
                    self._logger.warning(
                        "RAG retriever initialization failed",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                else:
                    self._logger.warning(f"RAG retriever initialization failed: {e}")
                self._rag_retriever = None

            # Create MCTS config
            from src.framework.mcts.config import MCTSConfig

            mcts_config = MCTSConfig(
                num_iterations=self._config.mcts_iterations,
                exploration_weight=self._config.mcts_exploration_weight,
                seed=self._config.seed or 42,
            )

            # Try to create full framework, fall back to lightweight mode
            try:
                self._framework = IntegratedFramework(
                    model_adapter=llm_client,
                    logger=self._logger,
                    mcts_config=mcts_config,
                    top_k_retrieval=self._config.top_k_retrieval,
                    max_iterations=self._config.max_iterations,
                    consensus_threshold=self._config.consensus_threshold,
                    enable_parallel_agents=self._config.enable_parallel_agents,
                )
                if _HAS_STRUCTURED_LOGGING:
                    self._logger.info(
                        "Full framework initialized",
                        correlation_id=correlation_id,
                        framework_type="integrated",
                    )
            except (ImportError, NotImplementedError) as e:
                if _HAS_STRUCTURED_LOGGING:
                    self._logger.warning(
                        "Full framework unavailable, using lightweight mode",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                else:
                    self._logger.warning(f"Full framework unavailable: {e}, using lightweight mode")
                self._framework = LightweightFramework(
                    llm_client=llm_client,
                    config=self._config,
                    logger=self._logger,
                    rag_retriever=self._rag_retriever,
                )

            self._state = FrameworkState.READY
            init_time = (time.perf_counter() - init_start) * 1000

            if _HAS_STRUCTURED_LOGGING:
                self._logger.info(
                    "Framework service initialized successfully",
                    correlation_id=correlation_id,
                    init_time_ms=round(init_time, 2),
                    framework_ready=True,
                    rag_available=self._rag_retriever is not None,
                )
            else:
                self._logger.info("Framework service initialized successfully")
            return True

        except Exception as e:
            self._state = FrameworkState.ERROR
            self._init_error = e
            if _HAS_STRUCTURED_LOGGING:
                self._logger.error(
                    "Framework initialization failed",
                    correlation_id=correlation_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
            else:
                self._logger.error(f"Framework initialization failed: {e}")
            return False

    async def process_query(
        self,
        query: str,
        use_mcts: bool | None = None,
        use_rag: bool = True,
        thread_id: str | None = None,
        mcts_iterations: int | None = None,
    ) -> QueryResult:
        """
        Process a query through the framework.

        Args:
            query: User query to process
            use_mcts: Enable MCTS (uses config default if None)
            use_rag: Enable RAG context retrieval
            thread_id: Optional conversation thread ID
            mcts_iterations: Override MCTS iterations

        Returns:
            QueryResult with response and metadata

        Raises:
            RuntimeError: If framework not initialized
            TimeoutError: If processing exceeds timeout
            ValueError: If query is invalid
        """
        # Generate request ID for tracking
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        set_correlation_id(request_id)
        correlation_id = get_correlation_id()

        # Ensure initialized
        if self._state != FrameworkState.READY and not await self.initialize():
            raise RuntimeError(
                f"Framework not available: {self._init_error or 'initialization failed'}"
            )

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        start_time = time.perf_counter()
        self._request_count += 1

        # Use config defaults if not specified
        use_mcts = use_mcts if use_mcts is not None else self._config.mcts_enabled

        if _HAS_STRUCTURED_LOGGING:
            self._logger.info(
                "Processing query",
                correlation_id=correlation_id,
                request_id=request_id,
                query_length=len(query),
                use_mcts=use_mcts,
                use_rag=use_rag,
                thread_id=thread_id,
            )
        else:
            self._logger.info(f"Processing query: {query[:50]}...")

        try:
            # Retrieve RAG context if enabled
            rag_context = ""
            rag_metadata: dict[str, Any] = {}

            if use_rag and self._rag_retriever is not None:
                try:
                    retrieval_result = await self._rag_retriever.retrieve(
                        query=query,
                        top_k=self._config.top_k_retrieval,
                    )
                    rag_context = retrieval_result.context
                    rag_metadata = {
                        "documents_retrieved": len(retrieval_result.documents),
                        "retrieval_time_ms": retrieval_result.retrieval_time_ms,
                        "retrieval_backend": retrieval_result.backend,
                    }

                    if _HAS_STRUCTURED_LOGGING:
                        self._logger.debug(
                            "RAG context retrieved",
                            correlation_id=correlation_id,
                            documents=len(retrieval_result.documents),
                            retrieval_time_ms=round(retrieval_result.retrieval_time_ms, 2),
                            backend=retrieval_result.backend,
                            context_length=len(rag_context),
                        )
                except Exception as e:
                    if _HAS_STRUCTURED_LOGGING:
                        self._logger.warning(
                            "RAG retrieval failed, proceeding without context",
                            correlation_id=correlation_id,
                            error=str(e),
                        )
                    else:
                        self._logger.warning(f"RAG retrieval failed: {e}")

            # Build config for this request
            config: dict[str, Any] = {
                "configurable": {"thread_id": thread_id or "default"},
                "rag_context": rag_context,  # Pass RAG context to framework
            }

            if mcts_iterations is not None:
                config["mcts_iterations"] = mcts_iterations

            # Process with timeout
            assert self._framework is not None

            framework_start = time.perf_counter()
            result = await asyncio.wait_for(
                self._framework.process(
                    query=query,
                    use_rag=use_rag,
                    use_mcts=use_mcts,
                    config=config,
                ),
                timeout=self._config.timeout_seconds,
            )
            framework_time = (time.perf_counter() - framework_start) * 1000

            processing_time = (time.perf_counter() - start_time) * 1000
            self._total_processing_time_ms += processing_time

            # Extract metadata
            metadata = result.get("metadata", {})
            state = result.get("state", {})

            # Build response metadata
            response_metadata = {
                "thread_id": thread_id,
                "rag_enabled": use_rag,
                "mcts_enabled": use_mcts,
                "iterations": metadata.get("iterations", 0),
                "request_id": request_id,
                **rag_metadata,
            }

            query_result = QueryResult(
                response=result.get("response", ""),
                confidence=metadata.get("consensus_score", 0.0),
                agents_used=metadata.get("agents_used", []),
                mcts_stats=state.get("mcts_stats") if use_mcts else None,
                processing_time_ms=processing_time,
                metadata=response_metadata,
            )

            if _HAS_STRUCTURED_LOGGING:
                self._logger.info(
                    "Query processed successfully",
                    correlation_id=correlation_id,
                    request_id=request_id,
                    processing_time_ms=round(processing_time, 2),
                    framework_time_ms=round(framework_time, 2),
                    confidence=round(query_result.confidence, 3),
                    agents_used=query_result.agents_used,
                    response_length=len(query_result.response),
                )

            return query_result

        except TimeoutError:
            self._error_count += 1
            processing_time = (time.perf_counter() - start_time) * 1000
            if _HAS_STRUCTURED_LOGGING:
                self._logger.error(
                    "Query processing timed out",
                    correlation_id=correlation_id,
                    request_id=request_id,
                    timeout_seconds=self._config.timeout_seconds,
                    processing_time_ms=round(processing_time, 2),
                )
            raise TimeoutError(
                f"Query processing timed out after {self._config.timeout_seconds}s"
            )
        except Exception as e:
            self._error_count += 1
            if _HAS_STRUCTURED_LOGGING:
                self._logger.error(
                    "Query processing failed",
                    correlation_id=correlation_id,
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
            else:
                self._logger.error(f"Query processing failed: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """
        Check framework health status.

        Returns:
            Dictionary with health information
        """
        uptime = time.time() - self._start_time if self._start_time else 0
        avg_processing_time = (
            self._total_processing_time_ms / self._request_count
            if self._request_count > 0
            else 0.0
        )

        return {
            "status": self._state.value,
            "uptime_seconds": round(uptime, 2),
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": round(self._error_count / max(self._request_count, 1), 4),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "framework_ready": self._framework is not None,
            "rag_available": self._rag_retriever is not None,
            "config": {
                "mcts_enabled": self._config.mcts_enabled,
                "mcts_iterations": self._config.mcts_iterations,
                "top_k_retrieval": self._config.top_k_retrieval,
                "consensus_threshold": self._config.consensus_threshold,
                "max_iterations": self._config.max_iterations,
                "timeout_seconds": self._config.timeout_seconds,
            },
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the framework service."""
        if self._state == FrameworkState.SHUTTING_DOWN:
            return

        self._state = FrameworkState.SHUTTING_DOWN
        self._logger.info("Shutting down framework service...")

        # Cleanup resources
        self._framework = None
        self._state = FrameworkState.UNINITIALIZED

    @property
    def is_ready(self) -> bool:
        """Check if framework is ready to process queries."""
        return self._state == FrameworkState.READY

    @property
    def state(self) -> FrameworkState:
        """Get current framework state."""
        return self._state


class MockLLMClient:
    """Mock LLM client for testing and fallback."""

    async def generate(self, prompt: str, temperature: float = 0.5) -> Any:
        """Generate mock response."""

        @dataclass
        class MockResponse:
            text: str = "This is a mock response for testing purposes."

        return MockResponse()


class LightweightFramework:
    """
    Lightweight framework for when full framework is unavailable.

    Provides basic query processing without full agent orchestration.
    Supports RAG context injection for improved responses.
    """

    def __init__(
        self,
        llm_client: Any,
        config: FrameworkConfig,
        logger: logging.Logger,
        rag_retriever: Any | None = None,
    ):
        self._llm_client = llm_client
        self._config = config
        self._logger = logger
        self._rag_retriever = rag_retriever

    async def process(
        self,
        query: str,
        use_rag: bool = True,
        use_mcts: bool = False,
        config: dict | None = None,
    ) -> dict[str, Any]:
        """Process query with lightweight implementation."""
        correlation_id = get_correlation_id()

        # Use try/except to handle both structured and standard loggers
        try:
            self._logger.info(
                "Processing query (lightweight mode)",
                correlation_id=correlation_id,
                query_length=len(query),
                use_rag=use_rag,
                use_mcts=use_mcts,
            )
        except TypeError:
            self._logger.info(f"Processing query (lightweight mode): {query[:50]}...")

        # Retrieve RAG context if available
        rag_context = ""
        if use_rag and self._rag_retriever is not None:
            try:
                retrieval_result = await self._rag_retriever.retrieve(
                    query=query,
                    top_k=self._config.top_k_retrieval,
                )
                rag_context = retrieval_result.context
                try:
                    self._logger.debug(
                        "RAG context retrieved for lightweight processing",
                        correlation_id=correlation_id,
                        documents=len(retrieval_result.documents),
                    )
                except TypeError:
                    self._logger.debug(f"RAG context retrieved: {len(retrieval_result.documents)} documents")
            except Exception as e:
                # Use try/except to handle both structured and standard loggers
                try:
                    self._logger.warning(
                        "RAG retrieval failed in lightweight mode",
                        correlation_id=correlation_id,
                        error=str(e),
                    )
                except TypeError:
                    # Fall back to standard logging if structured logging not supported
                    self._logger.warning(f"RAG retrieval failed in lightweight mode: {e}")

        # Build prompt with optional context
        if rag_context:
            prompt = f"""Based on the following context, answer the query.

Context:
{rag_context}

Query: {query}

Provide a comprehensive and accurate answer:"""
        else:
            prompt = f"Answer this query concisely: {query}"

        # Simple LLM call
        try:
            response = await self._llm_client.generate(
                prompt=prompt,
                temperature=0.7,
            )
            response_text = response.text
            # Higher confidence when RAG context is used
            confidence = 0.8 if rag_context else 0.7
        except Exception as e:
            try:
                self._logger.warning(
                    "LLM call failed in lightweight mode",
                    correlation_id=correlation_id,
                    error=str(e),
                )
            except TypeError:
                self._logger.warning(f"LLM call failed: {e}")
            response_text = f"Unable to process query: {query[:100]}"
            confidence = 0.3

        # Build mock MCTS stats if requested
        mcts_stats = None
        if use_mcts:
            mcts_stats = {
                "iterations": self._config.mcts_iterations,
                "best_action": "direct_response",
                "best_action_visits": self._config.mcts_iterations,
                "best_action_value": confidence,
            }

        return {
            "response": response_text,
            "metadata": {
                "agents_used": ["lightweight"],
                "consensus_score": confidence,
                "iterations": 1,
                "rag_context_used": bool(rag_context),
            },
            "state": {
                "mcts_stats": mcts_stats,
            },
        }


# Convenience function for getting the service
async def get_framework_service() -> FrameworkService:
    """Get the framework service singleton."""
    return await FrameworkService.get_instance()
