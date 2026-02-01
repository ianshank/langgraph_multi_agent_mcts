"""
Unit tests for embedding cache.

Tests LRU caching, batch processing, and memory management.

Based on: NEXT_STEPS_PLAN.md Phase 5.1
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Test Markers
# =============================================================================

pytestmark = [
    pytest.mark.unit,
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    from src.performance.embedding_cache import EmbeddingCacheConfig

    return EmbeddingCacheConfig(
        max_entries=100,
        max_memory_mb=50.0,
        embedding_dim=128,
        ttl_seconds=60.0,
    )


@pytest.fixture
def embedding_cache(cache_config):
    """Create test embedding cache."""
    from src.performance.embedding_cache import EmbeddingCache

    return EmbeddingCache(cache_config)


@pytest.fixture
def mock_compute_fn():
    """Create mock embedding compute function."""

    def compute(text: str) -> list[float]:
        # Return deterministic fake embedding based on text hash
        return [float(hash(text) % 1000) / 1000] * 128

    return compute


# =============================================================================
# EmbeddingCacheConfig Tests
# =============================================================================


class TestEmbeddingCacheConfig:
    """Tests for EmbeddingCacheConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        from src.performance.embedding_cache import EmbeddingCacheConfig

        config = EmbeddingCacheConfig()

        assert config.max_entries == 10000
        assert config.max_memory_mb == 500.0
        assert config.embedding_dim == 768
        assert config.ttl_seconds == 3600.0

    def test_custom_config_values(self):
        """Test custom configuration values."""
        from src.performance.embedding_cache import EmbeddingCacheConfig

        config = EmbeddingCacheConfig(
            max_entries=1000,
            embedding_dim=256,
        )

        assert config.max_entries == 1000
        assert config.embedding_dim == 256

    def test_estimated_memory_per_entry(self):
        """Test memory estimation per entry."""
        from src.performance.embedding_cache import EmbeddingCacheConfig

        config = EmbeddingCacheConfig(
            embedding_dim=768,
            bytes_per_float=4,
        )
        entry_bytes = config.estimated_memory_per_entry_bytes()

        # 768 floats * 4 bytes + 100 overhead
        assert entry_bytes == 768 * 4 + 100

    def test_max_entries_for_memory(self):
        """Test max entries calculation based on memory."""
        from src.performance.embedding_cache import EmbeddingCacheConfig

        # Small memory limit
        config = EmbeddingCacheConfig(
            max_entries=10000,
            max_memory_mb=1.0,  # 1 MB
            embedding_dim=768,
        )
        max_entries = config.max_entries_for_memory()

        # Should be less than max_entries due to memory constraint
        assert max_entries < 10000
        assert max_entries > 0


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test cache entry creation."""
        from src.performance.embedding_cache import CacheEntry

        embedding = [0.1, 0.2, 0.3]
        entry = CacheEntry(embedding=embedding)

        assert entry.embedding == embedding
        assert entry.access_count == 0
        assert entry.created_at > 0

    def test_entry_not_expired(self):
        """Test entry is not expired when fresh."""
        from src.performance.embedding_cache import CacheEntry

        entry = CacheEntry(embedding=[0.1, 0.2])

        assert entry.is_expired(ttl_seconds=60.0) is False

    def test_entry_expired(self):
        """Test entry is expired after TTL."""
        from src.performance.embedding_cache import CacheEntry

        entry = CacheEntry(embedding=[0.1, 0.2])
        # Simulate old entry
        entry.created_at = time.time() - 120  # 2 minutes ago

        assert entry.is_expired(ttl_seconds=60.0) is True


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_stats_initialization(self):
        """Test stats start at zero."""
        from src.performance.embedding_cache import CacheStats

        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_hit_rate_calculation(self):
        """Test hit rate is calculated correctly."""
        from src.performance.embedding_cache import CacheStats

        stats = CacheStats(hits=80, misses=20)

        assert stats.hit_rate == 0.8

    def test_hit_rate_zero_when_empty(self):
        """Test hit rate is zero when no requests."""
        from src.performance.embedding_cache import CacheStats

        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.performance.embedding_cache import CacheStats

        stats = CacheStats(hits=10, misses=5)
        data = stats.to_dict()

        assert data["hits"] == 10
        assert data["misses"] == 5
        assert "hit_rate" in data


# =============================================================================
# EmbeddingCache Basic Tests
# =============================================================================


class TestEmbeddingCacheBasic:
    """Basic tests for EmbeddingCache class."""

    def test_cache_creation(self, embedding_cache):
        """Test cache can be created."""
        assert embedding_cache is not None
        assert embedding_cache.size == 0

    def test_cache_put_and_get(self, embedding_cache):
        """Test putting and getting embeddings."""
        text = "Hello, world!"
        embedding = [0.1, 0.2, 0.3]

        embedding_cache.put(text, embedding)
        result = embedding_cache.get(text)

        assert result == embedding
        assert embedding_cache.size == 1

    def test_cache_miss_returns_none(self, embedding_cache):
        """Test cache miss returns None."""
        result = embedding_cache.get("nonexistent text")

        assert result is None

    def test_cache_tracks_hits_and_misses(self, embedding_cache):
        """Test cache tracks hit/miss statistics."""
        embedding_cache.put("text1", [0.1])
        embedding_cache.get("text1")  # Hit
        embedding_cache.get("text2")  # Miss

        stats = embedding_cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1


# =============================================================================
# EmbeddingCache LRU Tests
# =============================================================================


class TestEmbeddingCacheLRU:
    """Tests for LRU eviction behavior."""

    def test_lru_eviction(self):
        """Test oldest entries are evicted."""
        from src.performance.embedding_cache import EmbeddingCache, EmbeddingCacheConfig

        config = EmbeddingCacheConfig(
            max_entries=3,
            max_memory_mb=1000.0,  # Don't limit by memory
        )
        cache = EmbeddingCache(config)

        # Fill cache
        cache.put("text1", [1.0])
        cache.put("text2", [2.0])
        cache.put("text3", [3.0])

        # Add new entry, should evict text1
        cache.put("text4", [4.0])

        assert cache.get("text1") is None  # Evicted
        assert cache.get("text4") is not None  # Present
        assert cache.size == 3

    def test_access_updates_lru_order(self):
        """Test accessing entry moves it to end."""
        from src.performance.embedding_cache import EmbeddingCache, EmbeddingCacheConfig

        config = EmbeddingCacheConfig(max_entries=3, max_memory_mb=1000.0)
        cache = EmbeddingCache(config)

        cache.put("text1", [1.0])
        cache.put("text2", [2.0])
        cache.put("text3", [3.0])

        # Access text1, moving it to end
        cache.get("text1")

        # Add new entry, should evict text2 (now oldest)
        cache.put("text4", [4.0])

        assert cache.get("text1") is not None  # Still present
        assert cache.get("text2") is None  # Evicted

    def test_eviction_count_tracked(self):
        """Test eviction count is tracked."""
        from src.performance.embedding_cache import EmbeddingCache, EmbeddingCacheConfig

        config = EmbeddingCacheConfig(max_entries=2, max_memory_mb=1000.0)
        cache = EmbeddingCache(config)

        cache.put("text1", [1.0])
        cache.put("text2", [2.0])
        cache.put("text3", [3.0])  # Evicts text1

        stats = cache.get_stats()

        assert stats["evictions"] == 1


# =============================================================================
# EmbeddingCache TTL Tests
# =============================================================================


class TestEmbeddingCacheTTL:
    """Tests for TTL-based expiration."""

    def test_expired_entry_not_returned(self):
        """Test expired entries are not returned."""
        from src.performance.embedding_cache import EmbeddingCache, EmbeddingCacheConfig

        config = EmbeddingCacheConfig(
            max_entries=100,
            ttl_seconds=0.1,  # Very short TTL
        )
        cache = EmbeddingCache(config)

        cache.put("text1", [1.0])
        time.sleep(0.15)  # Wait for expiration

        result = cache.get("text1")

        assert result is None

    def test_expiration_count_tracked(self):
        """Test expiration count is tracked."""
        from src.performance.embedding_cache import EmbeddingCache, EmbeddingCacheConfig

        config = EmbeddingCacheConfig(ttl_seconds=0.1)
        cache = EmbeddingCache(config)

        cache.put("text1", [1.0])
        time.sleep(0.15)
        cache.get("text1")  # Triggers expiration check

        stats = cache.get_stats()

        assert stats["expirations"] == 1


# =============================================================================
# EmbeddingCache Compute Function Tests
# =============================================================================


class TestEmbeddingCacheCompute:
    """Tests for compute function integration."""

    def test_get_or_compute_uses_cache(self, embedding_cache, mock_compute_fn):
        """Test get_or_compute returns cached value."""
        embedding_cache.set_compute_function(mock_compute_fn)

        # First call computes
        result1 = embedding_cache.get_or_compute("test text")

        # Second call uses cache
        result2 = embedding_cache.get_or_compute("test text")

        assert result1 == result2
        assert embedding_cache._stats.hits == 1
        assert embedding_cache._stats.misses == 1

    def test_get_or_compute_raises_without_function(self, embedding_cache):
        """Test get_or_compute raises if no compute function."""
        with pytest.raises(ValueError, match="No compute function set"):
            embedding_cache.get_or_compute("test text")

    def test_compute_time_tracked(self, embedding_cache, mock_compute_fn):
        """Test compute time is tracked."""
        embedding_cache.set_compute_function(mock_compute_fn)
        embedding_cache.get_or_compute("test text")

        stats = embedding_cache.get_stats()

        assert stats["total_compute_time_ms"] > 0


# =============================================================================
# EmbeddingCache Batch Tests
# =============================================================================


class TestEmbeddingCacheBatch:
    """Tests for batch processing."""

    def test_get_batch_returns_all_results(self, embedding_cache, mock_compute_fn):
        """Test get_batch returns results for all texts."""
        embedding_cache.set_compute_function(mock_compute_fn)

        texts = ["text1", "text2", "text3"]
        results = embedding_cache.get_batch(texts)

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_get_batch_uses_cache(self, embedding_cache, mock_compute_fn):
        """Test get_batch uses cached values."""
        embedding_cache.set_compute_function(mock_compute_fn)

        # Pre-cache one text
        embedding_cache.put("text1", [1.0, 2.0])

        texts = ["text1", "text2"]
        results = embedding_cache.get_batch(texts)

        # text1 from cache, text2 computed
        assert results[0] == [1.0, 2.0]
        assert embedding_cache._stats.hits == 1


# =============================================================================
# EmbeddingCache Clear Tests
# =============================================================================


class TestEmbeddingCacheClear:
    """Tests for cache clearing."""

    def test_clear_removes_all_entries(self, embedding_cache):
        """Test clear removes all entries."""
        embedding_cache.put("text1", [1.0])
        embedding_cache.put("text2", [2.0])

        embedding_cache.clear()

        assert embedding_cache.size == 0
        assert embedding_cache.get("text1") is None


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateEmbeddingCache:
    """Tests for create_embedding_cache factory function."""

    def test_factory_creates_cache(self):
        """Test factory creates cache with custom settings."""
        from src.performance.embedding_cache import create_embedding_cache

        cache = create_embedding_cache(
            max_entries=500,
            embedding_dim=256,
        )

        assert cache is not None
        assert cache.config.max_entries == 500
        assert cache.config.embedding_dim == 256

    def test_factory_sets_compute_function(self, mock_compute_fn):
        """Test factory sets compute function if provided."""
        from src.performance.embedding_cache import create_embedding_cache

        cache = create_embedding_cache(compute_fn=mock_compute_fn)

        # Should be able to use get_or_compute
        result = cache.get_or_compute("test")
        assert result is not None
