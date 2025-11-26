"""
Memory-safe bounded collections for personality modules.

Provides:
- Bounded deques with automatic eviction
- Thread-safe operations
- Memory usage tracking
- Automatic cleanup mechanisms

Security features:
- Fixed maximum size (prevents unbounded growth)
- Automatic FIFO eviction
- Thread-safe operations
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Generic, TypeVar

from .exceptions import MemoryLimitError

T = TypeVar("T")


@dataclass
class BoundedHistory(Generic[T]):
    """Thread-safe bounded history with automatic eviction.

    Security features:
    - Fixed maximum size (prevents unbounded growth)
    - Automatic FIFO eviction
    - Thread-safe operations
    - Memory usage tracking

    Attributes:
        max_size: Maximum number of items to store

    Example:
        >>> history = BoundedHistory[str](max_size=100)
        >>> history.append("item1")
        >>> len(history)
        1
    """

    max_size: int
    _data: deque[T] = field(init=False, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )
    _eviction_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate max_size and initialize deque."""
        if self.max_size <= 0:
            raise ValueError(
                f"max_size must be positive, got: {self.max_size}"
            )

        if self.max_size > 1_000_000:
            warnings.warn(
                f"Large max_size ({self.max_size}) may consume "
                "significant memory",
                UserWarning,
                stacklevel=2,
            )

        # Initialize with maxlen for automatic eviction
        self._data = deque(maxlen=self.max_size)

    def append(self, item: T) -> T | None:
        """Append item to history with automatic eviction.

        Args:
            item: Item to append

        Returns:
            Evicted item if history was full, None otherwise

        Thread-safe operation.
        """
        with self._lock:
            evicted: T | None = None
            if len(self._data) == self.max_size:
                evicted = self._data[0]  # Will be evicted
                self._eviction_count += 1

            self._data.append(item)
            return evicted

    def extend(self, items: list[T]) -> int:
        """Extend history with multiple items.

        Args:
            items: Items to append

        Returns:
            Number of items evicted

        Thread-safe operation.
        """
        with self._lock:
            initial_evictions = self._eviction_count
            for item in items:
                self.append(item)
            return self._eviction_count - initial_evictions

    def get_recent(self, n: int) -> list[T]:
        """Get n most recent items.

        Args:
            n: Number of items to retrieve

        Returns:
            List of recent items (may be fewer than n)

        Thread-safe operation.
        """
        with self._lock:
            if n <= 0:
                return []
            return list(self._data)[-n:]

    def get_all(self) -> list[T]:
        """Get all items in history.

        Returns:
            Copy of all items

        Thread-safe operation.
        """
        with self._lock:
            return list(self._data)

    def clear(self) -> int:
        """Clear all items from history.

        Returns:
            Number of items cleared

        Thread-safe operation.
        """
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def __len__(self) -> int:
        """Get current size (thread-safe)."""
        with self._lock:
            return len(self._data)

    def __contains__(self, item: T) -> bool:
        """Check if item in history (thread-safe)."""
        with self._lock:
            return item in self._data

    def __iter__(self):
        """Iterate over items (returns copy for thread-safety)."""
        with self._lock:
            return iter(list(self._data))

    @property
    def is_full(self) -> bool:
        """Check if history is at capacity."""
        with self._lock:
            return len(self._data) == self.max_size

    @property
    def eviction_count(self) -> int:
        """Get total number of evictions."""
        with self._lock:
            return self._eviction_count

    @property
    def utilization(self) -> float:
        """Get utilization percentage [0.0-1.0]."""
        with self._lock:
            return (
                len(self._data) / self.max_size
                if self.max_size > 0
                else 0.0
            )


@dataclass
class TimeAwareBoundedHistory(BoundedHistory[T]):
    """Bounded history with time-based expiration.

    Security features:
    - Size-based eviction (like BoundedHistory)
    - Time-based expiration
    - Automatic cleanup of expired entries

    Attributes:
        max_size: Maximum number of items
        retention_period: How long to retain items
    """

    retention_period: timedelta = field(
        default_factory=lambda: timedelta(days=30)
    )
    _timestamps: deque[datetime] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize with timestamp tracking."""
        super().__post_init__()
        self._timestamps = deque(maxlen=self.max_size)

    def append(self, item: T) -> T | None:
        """Append item with timestamp.

        Args:
            item: Item to append

        Returns:
            Evicted item if any
        """
        with self._lock:
            # Record timestamp
            self._timestamps.append(datetime.now(timezone.utc))

            # Cleanup expired entries first
            self._cleanup_expired()

            # Then append
            return super().append(item)

    def _cleanup_expired(self) -> int:
        """Remove expired entries based on retention_period.

        Returns:
            Number of entries removed

        Note: Must be called with lock held.
        """
        if not self._data:
            return 0

        cutoff_time = datetime.now(timezone.utc) - self.retention_period
        removed_count = 0

        # Remove from left while entries are expired
        while (
            self._data
            and self._timestamps
            and self._timestamps[0] < cutoff_time
        ):
            self._data.popleft()
            self._timestamps.popleft()
            removed_count += 1

        return removed_count

    def cleanup(self) -> int:
        """Manually trigger cleanup of expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            return self._cleanup_expired()


@dataclass
class BoundedCounter:
    """Thread-safe bounded counter with overflow protection.

    Security features:
    - Maximum count limit
    - Overflow protection
    - Thread-safe operations

    Attributes:
        max_count: Maximum count per key
        max_keys: Maximum number of keys to track

    Example:
        >>> counter = BoundedCounter(max_count=1000)
        >>> counter.increment("key1")
        1
    """

    max_count: int = 1_000_000
    max_keys: int = 100_000
    _counts: dict[Any, int] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )

    def increment(self, key: Any, amount: int = 1) -> int:
        """Increment counter for key.

        Args:
            key: Counter key
            amount: Amount to increment

        Returns:
            New count value

        Raises:
            ValueError: If increment would exceed max_count
            MemoryLimitError: If max_keys exceeded
        """
        with self._lock:
            # Check if adding new key
            if key not in self._counts and len(self._counts) >= self.max_keys:
                raise MemoryLimitError(
                    message="Maximum key count exceeded",
                    collection_name="BoundedCounter",
                    current_size=len(self._counts),
                    max_size=self.max_keys,
                )

            current = self._counts.get(key, 0)
            new_value = current + amount

            if new_value > self.max_count:
                raise ValueError(
                    f"Counter overflow: {new_value} exceeds "
                    f"max_count {self.max_count}"
                )

            if new_value < 0:
                raise ValueError(
                    f"Counter underflow: {new_value} is negative"
                )

            self._counts[key] = new_value
            return new_value

    def decrement(self, key: Any, amount: int = 1) -> int:
        """Decrement counter for key.

        Args:
            key: Counter key
            amount: Amount to decrement

        Returns:
            New count value

        Raises:
            ValueError: If decrement would go negative
        """
        return self.increment(key, -amount)

    def get(self, key: Any) -> int:
        """Get count for key.

        Args:
            key: Counter key

        Returns:
            Current count (0 if key doesn't exist)
        """
        with self._lock:
            return self._counts.get(key, 0)

    def reset(self, key: Any) -> int:
        """Reset counter for key.

        Args:
            key: Counter key

        Returns:
            Previous count value
        """
        with self._lock:
            return self._counts.pop(key, 0)

    def clear_all(self) -> int:
        """Clear all counters.

        Returns:
            Number of counters cleared
        """
        with self._lock:
            count = len(self._counts)
            self._counts.clear()
            return count

    def total_keys(self) -> int:
        """Get total number of tracked keys."""
        with self._lock:
            return len(self._counts)

    def total_count(self) -> int:
        """Get sum of all counts."""
        with self._lock:
            return sum(self._counts.values())

    def top_k(self, k: int = 10) -> list[tuple[Any, int]]:
        """Get top k keys by count.

        Args:
            k: Number of top keys to return

        Returns:
            List of (key, count) tuples sorted by count descending
        """
        with self._lock:
            sorted_items = sorted(
                self._counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_items[:k]


@dataclass
class AsyncBoundedCache(Generic[T]):
    """Async-safe bounded cache with TTL support.

    Provides:
    - Async-safe operations
    - TTL-based expiration
    - LRU eviction
    - Deduplication of concurrent computations

    Attributes:
        max_size: Maximum cache entries
        ttl_seconds: Time-to-live for entries (0 = no expiration)
    """

    max_size: int = 10_000
    ttl_seconds: float = 0
    _cache: dict[str, tuple[T, datetime]] = field(
        default_factory=dict, repr=False
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _computing: dict[str, asyncio.Event] = field(
        default_factory=dict, repr=False
    )
    _hits: int = field(default=0, repr=False)
    _misses: int = field(default=0, repr=False)

    async def get(self, key: str) -> T | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if self._is_valid(timestamp):
                    self._hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None

    async def set(self, key: str, value: T) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()

            self._cache[key] = (value, datetime.now(timezone.utc))

    async def get_or_compute(
        self,
        key: str,
        compute: Callable[[], T] | Callable[[], Any],
    ) -> T:
        """Get from cache or compute value.

        If multiple coroutines request the same key simultaneously,
        only one will compute, others will wait for the result.

        Args:
            key: Cache key
            compute: Function to compute value (sync or async)

        Returns:
            Cached or computed value
        """
        # Fast path: check cache
        event = None
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if self._is_valid(timestamp):
                    self._hits += 1
                    return value

            # Check if someone else is computing
            if key in self._computing:
                event = self._computing[key]
                # Release lock while waiting - event will be awaited below

        # Wait for other computation
        if event is not None:
            await event.wait()
            async with self._lock:
                if key in self._cache:
                    return self._cache[key][0]

        # We need to compute
        async with self._lock:
            # Double-check
            if key in self._cache:
                value, timestamp = self._cache[key]
                if self._is_valid(timestamp):
                    return value

            # Mark as computing
            event = asyncio.Event()
            self._computing[key] = event

        try:
            # Compute outside lock
            if asyncio.iscoroutinefunction(compute):
                value = await compute()  # type: ignore
            else:
                value = compute()

            # Store result
            await self.set(key, value)
            self._misses += 1
            return value

        finally:
            # Signal waiters
            async with self._lock:
                if key in self._computing:
                    del self._computing[key]
            event.set()

    def _is_valid(self, timestamp: datetime) -> bool:
        """Check if timestamp is still valid."""
        if self.ttl_seconds <= 0:
            return True
        elapsed = (datetime.now(timezone.utc) - timestamp).total_seconds()
        return elapsed < self.ttl_seconds

    def _evict_oldest(self) -> None:
        """Evict oldest entry (must hold lock)."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k][1],
        )
        del self._cache[oldest_key]

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    async def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
