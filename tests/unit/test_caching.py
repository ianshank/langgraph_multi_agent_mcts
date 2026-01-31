"""
Unit tests for query-level caching.

Tests cache functionality, TTL expiration, and LRU eviction.

Based on: MULTI_AGENT_MCTS_TEMPLATE.md Section 10
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

# Set environment variables before importing modules that depend on settings
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-testing-only")
os.environ.setdefault("LLM_PROVIDER", "openai")

# Import caching modules with graceful fallback
try:
    from src.framework.caching import (
        CacheEntry,
        CacheStats,
        LLMResponseCache,
        MCTSSimulationCache,
        QueryCache,
        get_llm_cache,
        get_mcts_cache,
        get_query_cache,
        reset_caches,
    )

    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not CACHING_AVAILABLE, reason="Caching module not available"),
]


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_entry_not_expired(self):
        """Test entry is not expired when within TTL."""
        entry = CacheEntry(value="test", ttl_seconds=60.0)

        assert entry.is_expired is False

    def test_entry_expired(self):
        """Test entry is expired after TTL."""
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 100,  # Created 100s ago
            ttl_seconds=60.0,
        )

        assert entry.is_expired is True

    def test_entry_age(self):
        """Test entry age calculation."""
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 10,
        )

        assert entry.age_seconds >= 10


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)

        assert stats.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        """Test hit rate with zero total."""
        stats = CacheStats(hits=0, misses=0)

        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test stats serialization."""
        stats = CacheStats(hits=10, misses=5, evictions=2, size=8)

        d = stats.to_dict()

        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["size"] == 8
        assert "hit_rate" in d


class TestQueryCache:
    """Test QueryCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        return QueryCache(max_size=10, default_ttl=60.0, name="test")

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("key1", "value1")

        found, value = await cache.get("key1")

        assert found is True
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        """Test cache miss."""
        found, value = await cache.get("nonexistent")

        assert found is False
        assert value is None

    @pytest.mark.asyncio
    async def test_get_expired(self, cache):
        """Test expired entry returns miss."""
        # Set with very short TTL
        await cache.set("key1", "value1", ttl=0.01)

        # Wait for expiration
        await asyncio.sleep(0.02)

        found, value = await cache.get("key1")

        assert found is False
        assert cache._stats.expirations == 1

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self, cache):
        """Test get_or_compute with cache hit."""
        await cache.set("key1", "cached_value")

        compute_called = False

        def compute_fn():
            nonlocal compute_called
            compute_called = True
            return "computed_value"

        result = await cache.get_or_compute("key1", compute_fn)

        assert result == "cached_value"
        assert compute_called is False

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self, cache):
        """Test get_or_compute with cache miss."""
        compute_called = False

        def compute_fn():
            nonlocal compute_called
            compute_called = True
            return "computed_value"

        result = await cache.get_or_compute("new_key", compute_fn)

        assert result == "computed_value"
        assert compute_called is True

    @pytest.mark.asyncio
    async def test_get_or_compute_async_fn(self, cache):
        """Test get_or_compute with async compute function."""

        async def async_compute():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await cache.get_or_compute("async_key", async_compute)

        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")

        # Add one more, should evict oldest
        await cache.set("key_new", "value_new")

        # First key should be evicted
        found, _ = await cache.get("key0")
        assert found is False

        # New key should exist
        found, value = await cache.get("key_new")
        assert found is True
        assert value == "value_new"

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        """Test cache invalidation."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        removed = await cache.invalidate("key1")

        assert removed is True

        found, _ = await cache.get("key1")
        assert found is False

        found, _ = await cache.get("key2")
        assert found is True

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self, cache):
        """Test invalidating nonexistent key."""
        removed = await cache.invalidate("nonexistent")

        assert removed is False

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self, cache):
        """Test pattern-based invalidation."""
        await cache.set("user:1:data", "data1")
        await cache.set("user:2:data", "data2")
        await cache.set("other:key", "other")

        count = await cache.invalidate_pattern("user:")

        assert count == 2

        # User keys should be gone
        found, _ = await cache.get("user:1:data")
        assert found is False

        # Other key should remain
        found, _ = await cache.get("other:key")
        assert found is True

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing cache."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        count = await cache.clear()

        assert count == 2
        assert cache._stats.size == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self, cache):
        """Test statistics tracking."""
        # Generate some activity
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key1")  # Hit
        await cache.get("nonexistent")  # Miss

        stats = cache.stats

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_compute_key(self, cache):
        """Test cache key computation."""
        key1 = cache._compute_key("arg1", "arg2", foo="bar")
        key2 = cache._compute_key("arg1", "arg2", foo="bar")
        key3 = cache._compute_key("arg1", "arg2", foo="baz")

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache):
        """Test concurrent access to cache."""

        async def writer(key: str):
            for i in range(10):
                await cache.set(f"{key}_{i}", f"value_{i}")
                await asyncio.sleep(0.001)

        async def reader(key: str):
            for i in range(10):
                await cache.get(f"{key}_{i}")
                await asyncio.sleep(0.001)

        # Run concurrent readers and writers
        await asyncio.gather(
            writer("a"),
            writer("b"),
            reader("a"),
            reader("b"),
        )

        # Should not raise any errors
        assert cache._stats.size <= cache.max_size


class TestLLMResponseCache:
    """Test LLM response cache."""

    @pytest.fixture
    def llm_cache(self):
        """Create LLM cache for testing."""
        return LLMResponseCache(max_size=100, default_ttl=300)

    def test_compute_key_deterministic(self, llm_cache):
        """Test LLM cache key is deterministic for same input."""
        messages = [{"role": "user", "content": "Hello"}]

        key1 = llm_cache.compute_key_from_messages(messages, "gpt-4", 0.0)
        key2 = llm_cache.compute_key_from_messages(messages, "gpt-4", 0.0)

        assert key1 == key2

    def test_compute_key_different_for_different_messages(self, llm_cache):
        """Test different messages produce different keys."""
        key1 = llm_cache.compute_key_from_messages([{"role": "user", "content": "Hello"}], "gpt-4", 0.0)
        key2 = llm_cache.compute_key_from_messages([{"role": "user", "content": "Hi"}], "gpt-4", 0.0)

        assert key1 != key2

    def test_no_cache_for_non_deterministic(self, llm_cache):
        """Test no caching for non-deterministic requests (temperature > 0)."""
        key = llm_cache.compute_key_from_messages(
            [{"role": "user", "content": "Hello"}],
            "gpt-4",
            0.7,  # Non-zero temperature
        )

        assert key == ""

    def test_token_savings_tracking(self, llm_cache):
        """Test token savings tracking."""
        llm_cache.record_token_savings(100)
        llm_cache.record_token_savings(50)

        assert llm_cache.token_savings == 150


class TestMCTSSimulationCache:
    """Test MCTS simulation cache."""

    @pytest.fixture
    def mcts_cache(self):
        """Create MCTS cache for testing."""
        return MCTSSimulationCache(max_size=1000, default_ttl=600)

    def test_compute_key_from_state(self, mcts_cache):
        """Test state-based key computation."""
        key = mcts_cache.compute_key_from_state("state_hash_123")

        assert len(key) == 16  # SHA256 truncated

    def test_compute_key_with_action(self, mcts_cache):
        """Test state-action key computation."""
        key1 = mcts_cache.compute_key_from_state("state_hash", "action_a")
        key2 = mcts_cache.compute_key_from_state("state_hash", "action_b")
        key3 = mcts_cache.compute_key_from_state("state_hash")

        assert key1 != key2
        assert key1 != key3


class TestGlobalCaches:
    """Test global cache instances."""

    @pytest.fixture(autouse=True)
    def reset_global_caches(self):
        """Reset global caches before each test."""
        reset_caches()
        yield
        reset_caches()

    def test_get_query_cache_singleton(self):
        """Test query cache is singleton."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()

        assert cache1 is cache2

    def test_get_llm_cache_singleton(self):
        """Test LLM cache is singleton."""
        cache1 = get_llm_cache()
        cache2 = get_llm_cache()

        assert cache1 is cache2

    def test_get_mcts_cache_singleton(self):
        """Test MCTS cache is singleton."""
        cache1 = get_mcts_cache()
        cache2 = get_mcts_cache()

        assert cache1 is cache2

    def test_reset_caches(self):
        """Test cache reset functionality."""
        cache1 = get_query_cache()
        reset_caches()
        cache2 = get_query_cache()

        assert cache1 is not cache2
