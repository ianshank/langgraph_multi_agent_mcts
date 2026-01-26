"""
Unit tests for LLM-Guided MCTS RAG Module (Phase 2).

Tests:
- RAGContextProvider configuration and retrieval
- RAGContext data structure
- RAG-enhanced prompt building
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# Check for numpy availability (required by parent modules)
try:
    import numpy as np  # noqa: F401 - used for availability check

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

# Skip all tests if numpy not available (required by parent MCTS modules)
pytestmark = pytest.mark.skipif(not _NUMPY_AVAILABLE, reason="numpy not available")


class TestRAGContext:
    """Tests for RAGContext."""

    def test_context_creation(self):
        """Test creating RAG context."""
        from src.framework.mcts.llm_guided.rag.context import RAGContext

        context = RAGContext(
            similar_solutions=[{"code": "def foo(): pass", "description": "Example"}],
            code_patterns=[{"name": "Pattern1", "code": "x = [i for i in lst]"}],
            query="test query",
        )

        assert len(context.similar_solutions) == 1
        assert len(context.code_patterns) == 1
        assert context.query == "test query"

    def test_context_is_empty(self):
        """Test empty context detection."""
        from src.framework.mcts.llm_guided.rag.context import RAGContext

        empty = RAGContext()
        non_empty = RAGContext(similar_solutions=[{"code": "def x(): pass"}])

        assert empty.is_empty()
        assert not non_empty.is_empty()

    def test_context_to_text(self):
        """Test converting context to text."""
        from src.framework.mcts.llm_guided.rag.context import RAGContext

        context = RAGContext(
            similar_solutions=[{"code": "def foo(): return 1", "description": "Returns 1"}],
            code_patterns=[{"name": "List Comp", "code": "[x for x in lst]"}],
        )

        text = context.to_text()

        assert "Similar Solutions" in text
        assert "def foo()" in text
        assert "List Comp" in text

    def test_context_to_text_truncation(self):
        """Test text truncation for long context."""
        from src.framework.mcts.llm_guided.rag.context import RAGContext

        # Create context with long code
        long_code = "x = 1\n" * 500
        context = RAGContext(similar_solutions=[{"code": long_code, "description": "Long example"}])

        text = context.to_text(max_length=500)

        assert len(text) <= 600  # Some buffer for truncation message
        assert "[Context truncated...]" in text

    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        from src.framework.mcts.llm_guided.rag.context import RAGContext

        context = RAGContext(
            similar_solutions=[{"code": "x"}],
            query="test",
            num_results=1,
            retrieval_time_ms=10.5,
        )

        d = context.to_dict()

        assert d["query"] == "test"
        assert d["num_results"] == 1
        assert d["retrieval_time_ms"] == 10.5


class TestRAGContextProviderConfig:
    """Tests for RAGContextProviderConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from src.framework.mcts.llm_guided.rag.context import RAGContextProviderConfig

        config = RAGContextProviderConfig()

        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.include_solutions is True

    def test_config_validation(self):
        """Test configuration validation."""
        from src.framework.mcts.llm_guided.rag.context import RAGContextProviderConfig

        config = RAGContextProviderConfig(top_k=0)

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            config.validate()

    def test_config_validation_threshold(self):
        """Test threshold validation."""
        from src.framework.mcts.llm_guided.rag.context import RAGContextProviderConfig

        config = RAGContextProviderConfig(similarity_threshold=1.5)

        with pytest.raises(ValueError, match="similarity_threshold"):
            config.validate()


class TestRAGContextProvider:
    """Tests for RAGContextProvider."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = MagicMock()
        store.query = AsyncMock(
            return_value=[
                {
                    "text": "def example(): return 42",
                    "type": "solution",
                    "score": 0.9,
                    "metadata": {"description": "Example solution"},
                },
                {
                    "text": "[x for x in items]",
                    "type": "pattern",
                    "score": 0.85,
                    "metadata": {"name": "List Comprehension"},
                },
            ]
        )
        return store

    @pytest.mark.asyncio
    async def test_get_context_with_vector_store(self, mock_vector_store):
        """Test getting context from vector store."""
        from src.framework.mcts.llm_guided.rag.context import RAGContextProvider

        provider = RAGContextProvider(vector_store=mock_vector_store)

        context = await provider.get_context(
            problem="Write a function",
            current_code="def foo(): pass",
        )

        assert not context.is_empty()
        assert len(context.similar_solutions) == 1
        assert len(context.code_patterns) == 1
        mock_vector_store.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_fallback(self):
        """Test fallback context when no vector store."""
        from src.framework.mcts.llm_guided.rag.context import RAGContextProvider

        provider = RAGContextProvider(vector_store=None)

        context = await provider.get_context(problem="Write a function")

        # Should have fallback patterns
        assert not context.is_empty()
        assert len(context.code_patterns) > 0

    @pytest.mark.asyncio
    async def test_context_caching(self, mock_vector_store):
        """Test context result caching."""
        from src.framework.mcts.llm_guided.rag.context import (
            RAGContextProvider,
            RAGContextProviderConfig,
        )

        config = RAGContextProviderConfig(cache_results=True)
        provider = RAGContextProvider(config=config, vector_store=mock_vector_store)

        # First call
        _context1 = await provider.get_context(problem="Test problem")
        # Second call with same query
        _context2 = await provider.get_context(problem="Test problem")

        # Should only call vector store once (cache hit on second call)
        assert mock_vector_store.query.call_count == 1
        # Verify contexts were returned (not None)
        assert _context1 is not None
        assert _context2 is not None

    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, mock_vector_store):
        """Test filtering by similarity threshold."""
        from src.framework.mcts.llm_guided.rag.context import (
            RAGContextProvider,
            RAGContextProviderConfig,
        )

        # Return results with varying scores
        mock_vector_store.query = AsyncMock(
            return_value=[
                {"text": "good", "type": "solution", "score": 0.9, "metadata": {}},
                {"text": "bad", "type": "solution", "score": 0.5, "metadata": {}},
            ]
        )

        config = RAGContextProviderConfig(similarity_threshold=0.7, cache_results=False)
        provider = RAGContextProvider(config=config, vector_store=mock_vector_store)

        context = await provider.get_context(problem="Test")

        # Should only include the high-score result
        assert len(context.similar_solutions) == 1

    async def test_clear_cache(self, mock_vector_store):
        """Test clearing context cache."""
        from src.framework.mcts.llm_guided.rag.context import RAGContextProvider

        provider = RAGContextProvider(vector_store=mock_vector_store)
        provider._cache["key"] = ("value", 0)

        await provider.clear_cache()

        assert len(provider._cache) == 0


class TestCreateRAGProvider:
    """Tests for create_rag_provider factory."""

    def test_create_provider(self):
        """Test creating provider with factory."""
        from src.framework.mcts.llm_guided.rag.context import create_rag_provider

        provider = create_rag_provider(
            top_k=10,
            similarity_threshold=0.8,
            cache_results=False,
        )

        assert provider._config.top_k == 10
        assert provider._config.similarity_threshold == 0.8
        assert provider._config.cache_results is False


class TestRAGPromptBuilder:
    """Tests for RAGPromptBuilder."""

    def test_builder_default_config(self):
        """Test builder default configuration."""
        from src.framework.mcts.llm_guided.rag.prompts import RAGPromptBuilder

        builder = RAGPromptBuilder()

        assert builder.max_context_length == 4000
        assert builder.max_examples == 3

    def test_build_generator_prompt_basic(self):
        """Test building basic generator prompt."""
        from src.framework.mcts.llm_guided.rag.prompts import RAGPromptBuilder

        builder = RAGPromptBuilder()

        prompt = builder.build_generator_prompt(
            problem="Write a function that adds two numbers",
            current_code=None,
            rag_context=None,
            num_variants=3,
        )

        assert "Write a function" in prompt
        assert "3" in prompt  # num_variants
        assert "JSON" in prompt

    def test_build_generator_prompt_with_context(self):
        """Test building generator prompt with RAG context."""
        from src.framework.mcts.llm_guided.rag.context import RAGContext
        from src.framework.mcts.llm_guided.rag.prompts import RAGPromptBuilder

        builder = RAGPromptBuilder()

        context = RAGContext(
            similar_solutions=[{"code": "def add(a, b): return a + b", "description": "Simple addition"}]
        )

        prompt = builder.build_generator_prompt(
            problem="Add two numbers",
            current_code="def solution():",
            rag_context=context,
            num_variants=2,
        )

        assert "Similar Solutions" in prompt
        assert "def add(a, b)" in prompt
        assert "def solution():" in prompt

    def test_build_generator_prompt_with_feedback(self):
        """Test building prompt with feedback."""
        from src.framework.mcts.llm_guided.rag.prompts import RAGPromptBuilder

        builder = RAGPromptBuilder()

        prompt = builder.build_generator_prompt(
            problem="Test problem",
            current_code=None,
            rag_context=None,
            num_variants=2,
            iteration=3,
            feedback="Previous attempt failed on edge case",
        )

        assert "iteration 3" in prompt
        assert "Previous attempt failed" in prompt

    def test_build_reflector_prompt_basic(self):
        """Test building basic reflector prompt."""
        from src.framework.mcts.llm_guided.rag.prompts import RAGPromptBuilder

        builder = RAGPromptBuilder()

        prompt = builder.build_reflector_prompt(
            problem="Write add function",
            code="def add(a, b): return a + b",
            test_results=None,
            rag_context=None,
        )

        assert "Write add function" in prompt
        assert "def add(a, b)" in prompt
        assert "Value Estimate" in prompt

    def test_build_reflector_prompt_with_test_results(self):
        """Test building reflector prompt with test results."""
        from src.framework.mcts.llm_guided.rag.prompts import RAGPromptBuilder

        builder = RAGPromptBuilder()

        test_results = {
            "passed": False,
            "num_passed": 2,
            "num_total": 3,
            "errors": ["AssertionError: add(-1, 1) != 0"],
        }

        prompt = builder.build_reflector_prompt(
            problem="Test",
            code="def x(): pass",
            test_results=test_results,
            rag_context=None,
        )

        assert "2/3" in prompt
        assert "AssertionError" in prompt


class TestPromptConvenienceFunctions:
    """Tests for prompt convenience functions."""

    def test_build_generator_prompt_with_rag(self):
        """Test convenience function for generator prompt."""
        from src.framework.mcts.llm_guided.rag.prompts import build_generator_prompt_with_rag

        prompt = build_generator_prompt_with_rag(
            problem="Add numbers",
            num_variants=2,
        )

        assert "Add numbers" in prompt
        assert "2" in prompt

    def test_build_reflector_prompt_with_rag(self):
        """Test convenience function for reflector prompt."""
        from src.framework.mcts.llm_guided.rag.prompts import build_reflector_prompt_with_rag

        prompt = build_reflector_prompt_with_rag(
            problem="Test",
            code="def foo(): pass",
        )

        assert "Test" in prompt
        assert "def foo()" in prompt
