"""
Async utility helpers for Night Shift Training Engine.

Provides:
- AsyncSemaphore: Bounded concurrency control
- TokenBucketRateLimiter: Rate limiting for API calls
- ParallelBatchProcessor: Parallel async batch processing
- AsyncRetry: Retry decorator with exponential backoff
- AsyncQueue: Bounded async producer/consumer queue

ADVANCED PATTERNS (v76.0):
- CircuitBreaker: Fault tolerance with automatic recovery
- Backpressure: Adaptive load management
- Bulkhead: Failure isolation between components
- DeadLetterQueue: Failed operation tracking and retry
- HealthMonitor: Component health tracking
- MetricsCollector: Observability and performance tracking
- AdaptiveRateLimiter: Dynamic rate limiting based on success rates
- TimeoutPolicy: Configurable timeout with fallback strategies
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import random
import statistics
import time
import traceback
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class AsyncSemaphore:
    """
    Enhanced async semaphore with metrics and timeout support.

    Example:
        semaphore = AsyncSemaphore(max_concurrent=10)

        async with semaphore:
            await do_work()

        # Or with timeout
        async with semaphore.acquire_with_timeout(5.0):
            await do_work()
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._current_count = 0
        self._total_acquisitions = 0
        self._total_wait_time = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncSemaphore":
        start = time.monotonic()
        await self._semaphore.acquire()
        wait_time = time.monotonic() - start

        async with self._lock:
            self._current_count += 1
            self._total_acquisitions += 1
            self._total_wait_time += wait_time

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        async with self._lock:
            self._current_count -= 1
        self._semaphore.release()

    async def acquire_with_timeout(
        self, timeout: float
    ) -> "AsyncSemaphoreContext":
        """Acquire with timeout, raising TimeoutError if exceeded."""
        return AsyncSemaphoreContext(self, timeout)

    @property
    def current_usage(self) -> int:
        """Current number of acquired permits."""
        return self._current_count

    @property
    def available(self) -> int:
        """Number of available permits."""
        return self._max_concurrent - self._current_count

    def get_metrics(self) -> Dict[str, Any]:
        """Get semaphore metrics."""
        return {
            "max_concurrent": self._max_concurrent,
            "current_usage": self._current_count,
            "available": self.available,
            "total_acquisitions": self._total_acquisitions,
            "avg_wait_time": (
                self._total_wait_time / self._total_acquisitions
                if self._total_acquisitions > 0
                else 0.0
            ),
        }


class AsyncSemaphoreContext:
    """Context manager for semaphore with timeout."""

    def __init__(self, semaphore: AsyncSemaphore, timeout: float):
        self._semaphore = semaphore
        self._timeout = timeout

    async def __aenter__(self) -> AsyncSemaphore:
        try:
            await asyncio.wait_for(
                self._semaphore.__aenter__(),
                timeout=self._timeout,
            )
            return self._semaphore
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Failed to acquire semaphore within {self._timeout}s"
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._semaphore.__aexit__(exc_type, exc_val, exc_tb)


@dataclass
class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.

    Supports:
    - Configurable requests per minute/second
    - Burst handling
    - Async-safe
    - Metrics tracking

    Example:
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        await limiter.acquire()  # Blocks if rate exceeded
        response = await api_call()
    """

    requests_per_minute: int = 60
    burst_size: Optional[int] = None  # Defaults to requests_per_minute

    # Internal state
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _total_requests: int = field(init=False, default=0)
    _total_wait_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()

    @property
    def rate(self) -> float:
        """Tokens per second."""
        return self.requests_per_minute / 60.0

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1).
        """
        async with self._lock:
            await self._wait_for_tokens(tokens)
            self._tokens -= tokens
            self._total_requests += tokens

    async def _wait_for_tokens(self, needed: int) -> None:
        """Wait until enough tokens are available."""
        while True:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rate,
            )

            if self._tokens >= needed:
                return

            # Calculate wait time
            deficit = needed - self._tokens
            wait_time = deficit / self.rate

            self._total_wait_time += wait_time
            await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Returns:
            True if tokens acquired, False otherwise.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rate,
            )

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_requests += tokens
                return True

            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
            "current_tokens": self._tokens,
            "total_requests": self._total_requests,
            "total_wait_time": self._total_wait_time,
            "avg_wait_time": (
                self._total_wait_time / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }


@dataclass
class BatchResult(Generic[T, R]):
    """Result from batch processing."""

    item: T
    result: Optional[R] = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.error is None


class ParallelBatchProcessor(Generic[T, R]):
    """
    Process items in parallel batches with rate limiting and error handling.

    Example:
        processor = ParallelBatchProcessor(
            process_fn=api_call,
            max_concurrent=10,
            rate_limiter=TokenBucketRateLimiter(60),
        )

        async for result in processor.process(items):
            if result.success:
                print(result.result)
            else:
                print(f"Error: {result.error}")
    """

    def __init__(
        self,
        process_fn: Callable[[T], Awaitable[R]],
        max_concurrent: int = 10,
        batch_size: int = 100,
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        retry_attempts: int = 0,
        retry_delay: float = 1.0,
        on_error: Optional[Callable[[T, Exception], Awaitable[None]]] = None,
    ):
        """
        Initialize batch processor.

        Args:
            process_fn: Async function to process each item.
            max_concurrent: Max concurrent processing tasks.
            batch_size: Items per batch for yielding results.
            rate_limiter: Optional rate limiter for API calls.
            retry_attempts: Number of retry attempts on failure.
            retry_delay: Delay between retries (exponential backoff).
            on_error: Optional error handler callback.
        """
        self.process_fn = process_fn
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.rate_limiter = rate_limiter
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.on_error = on_error

        self._semaphore = AsyncSemaphore(max_concurrent)
        self._processed_count = 0
        self._success_count = 0
        self._error_count = 0

    async def _process_single(self, item: T) -> BatchResult[T, R]:
        """Process a single item with rate limiting and retries."""
        start = time.monotonic()

        for attempt in range(self.retry_attempts + 1):
            try:
                # Rate limit
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                # Process with semaphore
                async with self._semaphore:
                    result = await self.process_fn(item)

                duration = (time.monotonic() - start) * 1000
                self._processed_count += 1
                self._success_count += 1

                return BatchResult(
                    item=item,
                    result=result,
                    duration_ms=duration,
                )

            except Exception as e:
                if attempt < self.retry_attempts:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.retry_attempts} after {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    duration = (time.monotonic() - start) * 1000
                    self._processed_count += 1
                    self._error_count += 1

                    if self.on_error:
                        await self.on_error(item, e)

                    return BatchResult(
                        item=item,
                        error=e,
                        duration_ms=duration,
                    )

        # Should never reach here, but satisfy type checker
        return BatchResult(item=item, error=Exception("Unexpected error"))

    async def process(
        self,
        items: Union[List[T], AsyncIterator[T]],
    ) -> AsyncIterator[BatchResult[T, R]]:
        """
        Process items in parallel batches.

        Args:
            items: List or async iterator of items to process.

        Yields:
            BatchResult for each processed item.
        """
        # Convert to list if needed for batching
        if isinstance(items, list):
            item_list = items
        else:
            item_list = [item async for item in items]

        # Process in batches
        for batch_start in range(0, len(item_list), self.batch_size):
            batch = item_list[batch_start : batch_start + self.batch_size]

            # Process batch in parallel
            tasks = [self._process_single(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    # This shouldn't happen since we catch exceptions in _process_single
                    yield BatchResult(
                        item=None,
                        error=result,
                    )
                else:
                    yield result

    async def process_all(
        self,
        items: Union[List[T], AsyncIterator[T]],
    ) -> List[BatchResult[T, R]]:
        """
        Process all items and return complete results.

        Args:
            items: Items to process.

        Returns:
            List of all BatchResults.
        """
        results = []
        async for result in self.process(items):
            results.append(result)
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        return {
            "processed_count": self._processed_count,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": (
                self._success_count / self._processed_count
                if self._processed_count > 0
                else 0.0
            ),
            "semaphore": self._semaphore.get_metrics(),
            "rate_limiter": (
                self.rate_limiter.get_metrics()
                if self.rate_limiter
                else None
            ),
        }


def async_retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for async retry with exponential backoff.

    Example:
        @async_retry(attempts=3, delay=1.0)
        async def api_call():
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{attempts} for {func.__name__} "
                            f"after {wait_time:.1f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

            raise last_exception

        return wrapper

    return decorator


class AsyncQueue(Generic[T]):
    """
    Bounded async producer/consumer queue with backpressure.

    Example:
        queue = AsyncQueue(maxsize=100)

        # Producer
        await queue.put(item)

        # Consumer
        async for item in queue:
            process(item)

        # Or get single item
        item = await queue.get()
    """

    def __init__(self, maxsize: int = 0):
        self._queue: asyncio.Queue[Optional[T]] = asyncio.Queue(maxsize)
        self._closed = False
        self._put_count = 0
        self._get_count = 0

    async def put(self, item: T) -> None:
        """Put item in queue, blocking if full."""
        if self._closed:
            raise RuntimeError("Queue is closed")
        await self._queue.put(item)
        self._put_count += 1

    def put_nowait(self, item: T) -> None:
        """Put item without blocking, raises QueueFull if full."""
        if self._closed:
            raise RuntimeError("Queue is closed")
        self._queue.put_nowait(item)
        self._put_count += 1

    async def get(self) -> T:
        """Get item from queue, blocking if empty."""
        item = await self._queue.get()
        if item is None and self._closed:
            raise StopAsyncIteration()
        self._get_count += 1
        return item

    def get_nowait(self) -> T:
        """Get item without blocking, raises QueueEmpty if empty."""
        item = self._queue.get_nowait()
        if item is None and self._closed:
            raise StopAsyncIteration()
        self._get_count += 1
        return item

    def close(self) -> None:
        """Close queue. No more items can be added."""
        self._closed = True
        # Put sentinel to unblock consumers
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def __aiter__(self) -> "AsyncQueue[T]":
        return self

    async def __anext__(self) -> T:
        try:
            item = await self.get()
            return item
        except StopAsyncIteration:
            raise

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "current_size": self.qsize,
            "put_count": self._put_count,
            "get_count": self._get_count,
            "closed": self._closed,
        }


async def gather_with_concurrency(
    coros: List[Awaitable[T]],
    max_concurrent: int = 10,
) -> List[T]:
    """
    Like asyncio.gather but with concurrency limit.

    Example:
        results = await gather_with_concurrency(
            [api_call(x) for x in items],
            max_concurrent=5,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[bounded_coro(c) for c in coros])


async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Run coroutine with timeout, returning default on timeout.

    Example:
        result = await run_with_timeout(api_call(), timeout=5.0, default=None)
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        return default


class ProgressTracker:
    """
    Track progress of async operations.

    Example:
        tracker = ProgressTracker(total=100)

        async for item in items:
            process(item)
            tracker.update()
            print(f"Progress: {tracker.percent:.1f}%")
    """

    def __init__(
        self,
        total: int,
        callback: Optional[Callable[[int, int, float], Awaitable[None]]] = None,
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items.
            callback: Optional async callback(current, total, percent).
        """
        self.total = total
        self.callback = callback
        self._current = 0
        self._start_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        async with self._lock:
            self._current += n

            if self.callback:
                await self.callback(self._current, self.total, self.percent)

    @property
    def current(self) -> int:
        return self._current

    @property
    def percent(self) -> float:
        return (self._current / self.total * 100) if self.total > 0 else 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self._current == 0:
            return None

        rate = self._current / self.elapsed
        remaining = self.total - self._current

        return remaining / rate if rate > 0 else None

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "current": self._current,
            "total": self.total,
            "percent": self.percent,
            "elapsed": self.elapsed,
            "eta_seconds": self.eta_seconds,
            "rate": self._current / self.elapsed if self.elapsed > 0 else 0,
        }


# =============================================================================
# ADVANCED PATTERNS (v76.0)
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time before testing recovery
    half_open_max_calls: int = 3  # Max calls in half-open state
    excluded_exceptions: Tuple[type, ...] = ()  # Don't count these as failures
    fallback: Optional[Callable[..., Awaitable[Any]]] = None


class CircuitBreaker:
    """
    Circuit Breaker pattern for fault tolerance.

    Prevents cascade failures by stopping calls to failing services
    and automatically recovering when the service is back.

    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Service failing, calls rejected immediately
    - HALF_OPEN: Testing recovery with limited calls

    Example:
        breaker = CircuitBreaker("external_api", CircuitBreakerConfig())

        @breaker
        async def call_external_api():
            ...

        # Or use directly
        result = await breaker.call(external_api_call)
    """

    # Global registry of circuit breakers for monitoring
    _registry: Dict[str, "CircuitBreaker"] = {}

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0
        self._state_changes: List[Tuple[datetime, CircuitState]] = []

        # Register
        CircuitBreaker._registry[name] = self
        logger.info(f"Circuit breaker '{name}' initialized")

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def _should_allow_call(self) -> bool:
        """Check if call should be allowed based on state."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        # Transition to half-open
                        self._set_state(CircuitState.HALF_OPEN)
                        self._half_open_calls = 0
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def _set_state(self, new_state: CircuitState) -> None:
        """Set state with logging and metrics."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._state_changes.append((datetime.now(), new_state))
            logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")

    async def _record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._set_state(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    async def _record_failure(self, exception: Exception) -> None:
        """Record failed call."""
        # Check if exception is excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return

        async with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._set_state(CircuitState.OPEN)
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._set_state(CircuitState.OPEN)

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with circuit breaker protection."""
        self._total_calls += 1

        # Check if call should be allowed
        if not await self._should_allow_call():
            self._total_rejections += 1
            if self.config.fallback:
                logger.warning(f"Circuit '{self.name}' is open, using fallback")
                return await self.config.fallback(*args, **kwargs)
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is {self._state.value}"
            )

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Use as decorator."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)
        return wrapper

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self._set_state(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
            "total_rejections": self._total_rejections,
            "failure_rate": (
                self._total_failures / self._total_calls
                if self._total_calls > 0 else 0
            ),
            "recent_state_changes": [
                {"time": t.isoformat(), "state": s.value}
                for t, s in self._state_changes[-10:]
            ],
        }

    @classmethod
    def get_all_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: cb.get_metrics() for name, cb in cls._registry.items()}


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and call is rejected."""
    pass


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""
    id: str
    operation: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    exception: str
    exception_type: str
    traceback: str
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "operation": self.operation,
            "exception": self.exception,
            "exception_type": self.exception_type,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "last_retry": self.last_retry.isoformat() if self.last_retry else None,
            "metadata": self.metadata,
        }


class DeadLetterQueue:
    """
    Dead Letter Queue for failed operations.

    Captures failed operations for later retry or manual intervention.
    Integrates with the training pipeline to prevent data loss.

    Features:
    - Persistent storage option
    - Automatic retry with backoff
    - Metrics and alerting
    - Manual retry/discard interface
    """

    def __init__(
        self,
        name: str = "default",
        max_size: int = 10000,
        persist_path: Optional[Path] = None,
        auto_retry_interval: float = 300.0,  # 5 minutes
    ):
        self.name = name
        self.max_size = max_size
        self.persist_path = persist_path
        self.auto_retry_interval = auto_retry_interval

        self._queue: Deque[DeadLetterEntry] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._retry_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._total_added = 0
        self._total_retried = 0
        self._total_succeeded = 0
        self._total_discarded = 0

        # Operation registry for retries
        self._operations: Dict[str, Callable[..., Awaitable[Any]]] = {}

        if persist_path:
            self._load_from_disk()

    def register_operation(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register an operation for automatic retry."""
        self._operations[name] = func

    async def add(
        self,
        operation: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        exception: Exception,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add failed operation to the queue."""
        entry = DeadLetterEntry(
            id=str(uuid.uuid4())[:12],
            operation=operation,
            args=args,
            kwargs=kwargs,
            exception=str(exception),
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        async with self._lock:
            self._queue.append(entry)
            self._total_added += 1

        logger.warning(
            f"DLQ '{self.name}': Added failed operation '{operation}' - {exception}"
        )

        if self.persist_path:
            await self._persist_to_disk()

        return entry.id

    async def retry(self, entry_id: str) -> bool:
        """Retry a specific entry."""
        async with self._lock:
            entry = self._find_entry(entry_id)
            if not entry:
                return False

            if entry.operation not in self._operations:
                logger.error(f"DLQ: Operation '{entry.operation}' not registered")
                return False

            func = self._operations[entry.operation]

        try:
            await func(*entry.args, **entry.kwargs)

            # Success - remove from queue
            async with self._lock:
                self._queue.remove(entry)
                self._total_succeeded += 1

            logger.info(f"DLQ: Successfully retried entry {entry_id}")
            return True

        except Exception as e:
            async with self._lock:
                entry.retry_count += 1
                entry.last_retry = datetime.now()
                self._total_retried += 1

            logger.warning(f"DLQ: Retry failed for {entry_id}: {e}")
            return False

    def _find_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """Find entry by ID."""
        for entry in self._queue:
            if entry.id == entry_id:
                return entry
        return None

    async def retry_all(self) -> Dict[str, bool]:
        """Retry all entries that haven't exceeded max retries."""
        results = {}

        entries_to_retry = []
        async with self._lock:
            for entry in list(self._queue):
                if entry.retry_count < entry.max_retries:
                    entries_to_retry.append(entry)

        for entry in entries_to_retry:
            results[entry.id] = await self.retry(entry.id)

        return results

    async def discard(self, entry_id: str) -> bool:
        """Discard an entry from the queue."""
        async with self._lock:
            entry = self._find_entry(entry_id)
            if entry:
                self._queue.remove(entry)
                self._total_discarded += 1
                logger.info(f"DLQ: Discarded entry {entry_id}")
                return True
        return False

    async def start_auto_retry(self) -> None:
        """Start automatic retry background task."""
        if self._running:
            return

        self._running = True
        self._retry_task = asyncio.create_task(self._auto_retry_loop())
        logger.info(f"DLQ '{self.name}': Auto-retry started")

    async def stop_auto_retry(self) -> None:
        """Stop automatic retry."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

    async def _auto_retry_loop(self) -> None:
        """Background loop for automatic retries."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_retry_interval)
                await self.retry_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DLQ auto-retry error: {e}")

    async def _persist_to_disk(self) -> None:
        """Persist queue to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = [entry.to_dict() for entry in self._queue]

        # Atomic write
        temp_path = self.persist_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(self.persist_path)

    def _load_from_disk(self) -> None:
        """Load queue from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path) as f:
                data = json.load(f)

            for item in data:
                # Reconstruct entry (without args/kwargs for security)
                entry = DeadLetterEntry(
                    id=item["id"],
                    operation=item["operation"],
                    args=(),
                    kwargs={},
                    exception=item["exception"],
                    exception_type=item["exception_type"],
                    traceback="",
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    retry_count=item["retry_count"],
                    max_retries=item["max_retries"],
                    metadata=item.get("metadata", {}),
                )
                self._queue.append(entry)

            logger.info(f"DLQ: Loaded {len(self._queue)} entries from disk")

        except Exception as e:
            logger.error(f"DLQ: Failed to load from disk: {e}")

    def __len__(self) -> int:
        return len(self._queue)

    def get_metrics(self) -> Dict[str, Any]:
        """Get DLQ metrics."""
        by_operation = defaultdict(int)
        by_exception = defaultdict(int)

        for entry in self._queue:
            by_operation[entry.operation] += 1
            by_exception[entry.exception_type] += 1

        return {
            "name": self.name,
            "size": len(self._queue),
            "max_size": self.max_size,
            "total_added": self._total_added,
            "total_retried": self._total_retried,
            "total_succeeded": self._total_succeeded,
            "total_discarded": self._total_discarded,
            "by_operation": dict(by_operation),
            "by_exception": dict(by_exception),
            "oldest_entry": (
                self._queue[0].timestamp.isoformat()
                if self._queue else None
            ),
        }


# =============================================================================
# BACKPRESSURE CONTROLLER
# =============================================================================

class BackpressureStrategy(Enum):
    """Backpressure handling strategies."""
    DROP_OLDEST = "drop_oldest"  # Drop oldest items when full
    DROP_NEWEST = "drop_newest"  # Reject new items when full
    BLOCK = "block"  # Block until space available
    ADAPTIVE = "adaptive"  # Dynamically adjust rate


@dataclass
class BackpressureConfig:
    """Configuration for backpressure controller."""
    strategy: BackpressureStrategy = BackpressureStrategy.ADAPTIVE
    high_water_mark: float = 0.8  # Start applying pressure
    low_water_mark: float = 0.5  # Release pressure
    max_queue_size: int = 1000
    adaptive_min_rate: float = 0.1  # Min rate multiplier
    adaptive_recovery_rate: float = 0.05  # Rate recovery per check


class BackpressureController:
    """
    Backpressure controller for flow control.

    Manages load between producers and consumers, preventing
    memory exhaustion and providing smooth degradation.

    Example:
        controller = BackpressureController(BackpressureConfig())

        async def producer():
            while True:
                await controller.acquire()
                data = await produce_data()
                await controller.enqueue(data)

        async def consumer():
            async for item in controller:
                await process(item)
    """

    def __init__(self, config: Optional[BackpressureConfig] = None):
        self.config = config or BackpressureConfig()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._pressure_level = 0.0
        self._rate_multiplier = 1.0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_dropped = 0
        self._total_blocked_time = 0.0
        self._pressure_history: Deque[Tuple[datetime, float]] = deque(maxlen=100)

        logger.info(f"Backpressure controller initialized with {self.config.strategy.value} strategy")

    @property
    def pressure_level(self) -> float:
        """Current pressure level (0.0 - 1.0)."""
        if self.config.max_queue_size == 0:
            return 0.0
        return self._queue.qsize() / self.config.max_queue_size

    @property
    def is_under_pressure(self) -> bool:
        """Whether system is under backpressure."""
        return self.pressure_level >= self.config.high_water_mark

    async def acquire(self) -> bool:
        """
        Acquire permission to produce.

        Returns True if allowed, False if rejected (DROP_NEWEST).
        May block for BLOCK strategy.
        """
        pressure = self.pressure_level

        if self.config.strategy == BackpressureStrategy.DROP_NEWEST:
            if pressure >= self.config.high_water_mark:
                self._total_dropped += 1
                return False
            return True

        elif self.config.strategy == BackpressureStrategy.BLOCK:
            if pressure >= self.config.high_water_mark:
                start = time.monotonic()
                while self.pressure_level >= self.config.low_water_mark:
                    await asyncio.sleep(0.01)
                self._total_blocked_time += time.monotonic() - start
            return True

        elif self.config.strategy == BackpressureStrategy.ADAPTIVE:
            # Slow down based on pressure
            async with self._lock:
                if pressure > self.config.high_water_mark:
                    self._rate_multiplier = max(
                        self.config.adaptive_min_rate,
                        self._rate_multiplier * 0.9
                    )
                elif pressure < self.config.low_water_mark:
                    self._rate_multiplier = min(
                        1.0,
                        self._rate_multiplier + self.config.adaptive_recovery_rate
                    )

            # Probabilistic admission based on rate
            if random.random() > self._rate_multiplier:
                self._total_dropped += 1
                return False
            return True

        return True

    async def enqueue(self, item: T) -> bool:
        """Add item to queue."""
        try:
            if self.config.strategy == BackpressureStrategy.DROP_OLDEST:
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._total_dropped += 1
                    except asyncio.QueueEmpty:
                        pass

            await self._queue.put(item)
            self._total_enqueued += 1

            # Record pressure
            self._pressure_history.append((datetime.now(), self.pressure_level))

            return True

        except asyncio.QueueFull:
            self._total_dropped += 1
            return False

    async def dequeue(self) -> T:
        """Get item from queue."""
        item = await self._queue.get()
        self._total_dequeued += 1
        return item

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        return await self.dequeue()

    def get_metrics(self) -> Dict[str, Any]:
        """Get backpressure metrics."""
        return {
            "strategy": self.config.strategy.value,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.config.max_queue_size,
            "pressure_level": self.pressure_level,
            "rate_multiplier": self._rate_multiplier,
            "is_under_pressure": self.is_under_pressure,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_dropped": self._total_dropped,
            "total_blocked_time": self._total_blocked_time,
            "drop_rate": (
                self._total_dropped / (self._total_enqueued + self._total_dropped)
                if (self._total_enqueued + self._total_dropped) > 0 else 0
            ),
        }


# =============================================================================
# BULKHEAD PATTERN
# =============================================================================

class Bulkhead:
    """
    Bulkhead pattern for failure isolation.

    Isolates different parts of the system so failures in one
    component don't cascade to others. Each bulkhead has its
    own resource pool (semaphore).

    Example:
        # Create bulkheads for different operations
        db_bulkhead = Bulkhead("database", max_concurrent=10)
        api_bulkhead = Bulkhead("external_api", max_concurrent=5)

        async with db_bulkhead:
            await db_query()

        async with api_bulkhead:
            await api_call()
    """

    # Global registry
    _registry: Dict[str, "Bulkhead"] = {}

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait_time: float = 30.0,
        on_rejection: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait_time = max_wait_time
        self.on_rejection = on_rejection

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timeouts = 0
        self._total_execution_time = 0.0

        Bulkhead._registry[name] = self
        logger.info(f"Bulkhead '{name}' initialized with {max_concurrent} slots")

    async def __aenter__(self) -> "Bulkhead":
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.max_wait_time,
            )
            if acquired:
                async with self._lock:
                    self._active_count += 1
                    self._total_acquired += 1
                return self

        except asyncio.TimeoutError:
            self._total_timeouts += 1
            if self.on_rejection:
                await self.on_rejection()
            raise BulkheadFullError(f"Bulkhead '{self.name}' is full")

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        async with self._lock:
            self._active_count -= 1
        self._semaphore.release()

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active_count

    @property
    def utilization(self) -> float:
        """Current utilization (0.0 - 1.0)."""
        return self._active_count / self.max_concurrent if self.max_concurrent > 0 else 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "available_slots": self.available_slots,
            "utilization": self.utilization,
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
            "total_timeouts": self._total_timeouts,
        }

    @classmethod
    def get_all_metrics(cls) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all bulkheads."""
        return {name: bh.get_metrics() for name, bh in cls._registry.items()}


class BulkheadFullError(Exception):
    """Raised when bulkhead is full and request is rejected."""
    pass


# =============================================================================
# ADAPTIVE RATE LIMITER
# =============================================================================

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on success/failure rates.

    Unlike fixed rate limiting, this automatically increases limits
    when things are working well and decreases when errors occur.

    Example:
        limiter = AdaptiveRateLimiter(
            initial_rate=100,
            min_rate=10,
            max_rate=1000,
        )

        async with limiter.acquire():
            result = await api_call()
            limiter.record_success()
        # or
        limiter.record_failure()
    """

    def __init__(
        self,
        initial_rate: float = 100,  # Requests per minute
        min_rate: float = 10,
        max_rate: float = 1000,
        increase_factor: float = 1.1,  # Increase on success
        decrease_factor: float = 0.5,  # Decrease on failure
        window_size: int = 100,  # Success/failure window
        adjustment_interval: float = 10.0,  # Seconds between adjustments
    ):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.window_size = window_size
        self.adjustment_interval = adjustment_interval

        self._current_rate = initial_rate
        self._limiter = TokenBucketRateLimiter(
            requests_per_minute=int(initial_rate)
        )

        self._success_window: Deque[bool] = deque(maxlen=window_size)
        self._lock = asyncio.Lock()
        self._last_adjustment = time.monotonic()

        # Metrics
        self._total_successes = 0
        self._total_failures = 0
        self._rate_adjustments: List[Tuple[datetime, float]] = []

    @property
    def current_rate(self) -> float:
        """Current rate limit (requests per minute)."""
        return self._current_rate

    @property
    def success_rate(self) -> float:
        """Recent success rate (0.0 - 1.0)."""
        if not self._success_window:
            return 1.0
        return sum(self._success_window) / len(self._success_window)

    async def acquire(self) -> "AdaptiveRateLimiterContext":
        """Acquire rate limit."""
        await self._limiter.acquire()
        return AdaptiveRateLimiterContext(self)

    def record_success(self) -> None:
        """Record successful operation."""
        self._success_window.append(True)
        self._total_successes += 1
        self._maybe_adjust_rate()

    def record_failure(self) -> None:
        """Record failed operation."""
        self._success_window.append(False)
        self._total_failures += 1
        self._maybe_adjust_rate()

    def _maybe_adjust_rate(self) -> None:
        """Adjust rate if needed."""
        now = time.monotonic()
        if now - self._last_adjustment < self.adjustment_interval:
            return

        if len(self._success_window) < self.window_size // 2:
            return

        self._last_adjustment = now
        success_rate = self.success_rate

        old_rate = self._current_rate

        if success_rate > 0.95:
            # Increase rate
            self._current_rate = min(
                self.max_rate,
                self._current_rate * self.increase_factor
            )
        elif success_rate < 0.8:
            # Decrease rate
            self._current_rate = max(
                self.min_rate,
                self._current_rate * self.decrease_factor
            )

        if old_rate != self._current_rate:
            # Update underlying limiter
            self._limiter = TokenBucketRateLimiter(
                requests_per_minute=int(self._current_rate)
            )
            self._rate_adjustments.append((datetime.now(), self._current_rate))
            logger.info(f"Rate adjusted: {old_rate:.0f} -> {self._current_rate:.0f} RPM")

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            "current_rate": self._current_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "success_rate": self.success_rate,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "recent_adjustments": [
                {"time": t.isoformat(), "rate": r}
                for t, r in self._rate_adjustments[-10:]
            ],
        }


class AdaptiveRateLimiterContext:
    """Context manager for adaptive rate limiter."""

    def __init__(self, limiter: AdaptiveRateLimiter):
        self._limiter = limiter

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._limiter.record_success()
        else:
            self._limiter.record_failure()


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat(),
            "metadata": self.metadata,
        }


class HealthMonitor:
    """
    Health monitoring for system components.

    Tracks health of various components and provides aggregated status.
    Supports automatic health checks and alerting.

    Example:
        monitor = HealthMonitor()

        # Register health checks
        monitor.register("database", check_db_health)
        monitor.register("cache", check_cache_health)

        # Start monitoring
        await monitor.start()

        # Get status
        status = await monitor.get_status()
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3,  # Consecutive failures before unhealthy
    ):
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold

        self._checks: Dict[str, Callable[[], Awaitable[HealthCheck]]] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_status_change: List[Callable[[str, HealthStatus, HealthStatus], Awaitable[None]]] = []

    def register(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheck]],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_func
        self._results[name] = HealthCheck(name=name, status=HealthStatus.UNKNOWN)

    def on_status_change(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], Awaitable[None]],
    ) -> None:
        """Register callback for status changes."""
        self._on_status_change.append(callback)

    async def check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not registered",
            )

        start = time.monotonic()
        old_status = self._results.get(name, HealthCheck(name=name, status=HealthStatus.UNKNOWN)).status

        try:
            result = await self._checks[name]()
            result.latency_ms = (time.monotonic() - start) * 1000
            result.last_check = datetime.now()

            async with self._lock:
                if result.status == HealthStatus.HEALTHY:
                    self._failure_counts[name] = 0
                else:
                    self._failure_counts[name] += 1

                    if self._failure_counts[name] >= self.unhealthy_threshold:
                        result.status = HealthStatus.UNHEALTHY

                self._results[name] = result

            # Notify on status change
            if result.status != old_status:
                for callback in self._on_status_change:
                    try:
                        await callback(name, old_status, result.status)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")

            return result

        except Exception as e:
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )

            async with self._lock:
                self._failure_counts[name] += 1
                self._results[name] = result

            return result

    async def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = await asyncio.gather(
            *[self.check(name) for name in self._checks],
            return_exceptions=True,
        )

        return {
            name: result if isinstance(result, HealthCheck)
            else HealthCheck(name=name, status=HealthStatus.UNHEALTHY, message=str(result))
            for name, result in zip(self._checks.keys(), results)
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get aggregated health status."""
        results = self._results.copy()

        # Determine overall status
        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return {
            "status": overall.value,
            "checks": {name: check.to_dict() for name, check in results.items()},
            "timestamp": datetime.now().isoformat(),
        }

    async def start(self) -> None:
        """Start automatic health checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop automatic health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.check_interval)


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Centralized metrics collection for observability.

    Collects and aggregates metrics from various components.
    Supports counters, gauges, histograms, and timers.

    Example:
        metrics = MetricsCollector()

        # Counter
        metrics.increment("requests_total")

        # Gauge
        metrics.set_gauge("active_connections", 42)

        # Timer
        with metrics.timer("request_duration"):
            await process_request()

        # Histogram
        metrics.record("response_size", 1024)
    """

    _instance: Optional["MetricsCollector"] = None

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        self._labels: Dict[str, Dict[str, str]] = {}
        self._lock = asyncio.Lock()

        # Configuration
        self._histogram_max_size = 1000

        self._initialized = True

    def increment(
        self,
        name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def decrement(
        self,
        name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Decrement a counter."""
        key = self._make_key(name, labels)
        self._counters[key] -= value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

        # Limit size
        if len(self._histograms[key]) > self._histogram_max_size:
            self._histograms[key] = self._histograms[key][-self._histogram_max_size:]

    @contextlib.contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time operations."""
        start = time.monotonic()
        try:
            yield
        finally:
            duration = (time.monotonic() - start) * 1000  # ms
            key = self._make_key(name, labels)
            self._timers[key].append(duration)

            if len(self._timers[key]) > self._histogram_max_size:
                self._timers[key] = self._timers[key][-self._histogram_max_size:]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "p50": self._percentile(values, 50),
            "p90": self._percentile(values, 90),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def get_timer_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get timer statistics."""
        key = self._make_key(name, labels)
        values = self._timers.get(key, [])

        if not values:
            return {}

        return {
            "count": len(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p50_ms": self._percentile(values, 50),
            "p90_ms": self._percentile(values, 90),
            "p95_ms": self._percentile(values, 95),
            "p99_ms": self._percentile(values, 99),
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in set(k.split("{")[0] for k in self._histograms)
            },
            "timers": {
                name: self.get_timer_stats(name)
                for name in set(k.split("{")[0] for k in self._timers)
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timers.clear()


# =============================================================================
# TIMEOUT POLICY
# =============================================================================

class TimeoutStrategy(Enum):
    """Timeout handling strategies."""
    RAISE = "raise"  # Raise TimeoutError
    RETURN_DEFAULT = "return_default"  # Return a default value
    RETRY = "retry"  # Retry the operation


@dataclass
class TimeoutConfig:
    """Configuration for timeout policy."""
    timeout_seconds: float = 30.0
    strategy: TimeoutStrategy = TimeoutStrategy.RAISE
    default_value: Any = None
    max_retries: int = 3
    retry_delay: float = 1.0


class TimeoutPolicy:
    """
    Configurable timeout policy with multiple strategies.

    Example:
        policy = TimeoutPolicy(TimeoutConfig(
            timeout_seconds=5.0,
            strategy=TimeoutStrategy.RETURN_DEFAULT,
            default_value={"error": "timeout"},
        ))

        result = await policy.execute(slow_operation)
    """

    def __init__(self, config: Optional[TimeoutConfig] = None):
        self.config = config or TimeoutConfig()

        # Metrics
        self._total_calls = 0
        self._total_timeouts = 0
        self._total_retries = 0

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with timeout policy."""
        self._total_calls += 1
        retry_count = 0

        while True:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._total_timeouts += 1

                if self.config.strategy == TimeoutStrategy.RAISE:
                    raise

                elif self.config.strategy == TimeoutStrategy.RETURN_DEFAULT:
                    logger.warning(f"Timeout, returning default value")
                    return self.config.default_value

                elif self.config.strategy == TimeoutStrategy.RETRY:
                    retry_count += 1
                    self._total_retries += 1

                    if retry_count >= self.config.max_retries:
                        logger.error(f"Max retries ({self.config.max_retries}) exceeded")
                        raise

                    logger.warning(f"Timeout, retry {retry_count}/{self.config.max_retries}")
                    await asyncio.sleep(self.config.retry_delay * retry_count)

    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Use as decorator."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.execute(func, *args, **kwargs)
        return wrapper

    def get_metrics(self) -> Dict[str, Any]:
        """Get timeout policy metrics."""
        return {
            "timeout_seconds": self.config.timeout_seconds,
            "strategy": self.config.strategy.value,
            "total_calls": self._total_calls,
            "total_timeouts": self._total_timeouts,
            "total_retries": self._total_retries,
            "timeout_rate": (
                self._total_timeouts / self._total_calls
                if self._total_calls > 0 else 0
            ),
        }


# =============================================================================
# GRACEFUL SHUTDOWN COORDINATOR
# =============================================================================

class GracefulShutdown:
    """
    Coordinate graceful shutdown of async components.

    Ensures all components are properly shutdown in order,
    with timeout handling for stuck components.

    Example:
        shutdown = GracefulShutdown()

        # Register components in shutdown order
        shutdown.register("server", server.shutdown)
        shutdown.register("database", db.close)
        shutdown.register("cache", cache.flush)

        # When shutting down
        await shutdown.execute()
    """

    def __init__(self, timeout_per_component: float = 30.0):
        self.timeout_per_component = timeout_per_component
        self._components: List[Tuple[str, Callable[[], Awaitable[None]]]] = []
        self._is_shutting_down = False
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        shutdown_func: Callable[[], Awaitable[None]],
        priority: int = 0,  # Lower = earlier shutdown
    ) -> None:
        """Register a component for shutdown."""
        self._components.append((name, shutdown_func))
        # Sort by priority (lower first)
        self._components.sort(key=lambda x: priority)

    async def execute(self) -> Dict[str, str]:
        """Execute graceful shutdown."""
        async with self._lock:
            if self._is_shutting_down:
                logger.warning("Shutdown already in progress")
                return {}

            self._is_shutting_down = True

        results = {}
        logger.info(f"Starting graceful shutdown of {len(self._components)} components")

        for name, shutdown_func in self._components:
            try:
                logger.info(f"Shutting down: {name}")
                await asyncio.wait_for(
                    shutdown_func(),
                    timeout=self.timeout_per_component,
                )
                results[name] = "success"
                logger.info(f"Shutdown complete: {name}")

            except asyncio.TimeoutError:
                results[name] = "timeout"
                logger.error(f"Shutdown timeout: {name}")

            except Exception as e:
                results[name] = f"error: {e}"
                logger.error(f"Shutdown error for {name}: {e}")

        logger.info("Graceful shutdown complete")
        return results

    @property
    def is_shutting_down(self) -> bool:
        return self._is_shutting_down


# =============================================================================
# RESOURCE POOL
# =============================================================================

class ResourcePool(Generic[T]):
    """
    Generic async resource pool with health checking.

    Manages a pool of reusable resources (connections, workers, etc.)
    with automatic health checking and replacement.

    Example:
        pool = ResourcePool(
            factory=create_connection,
            max_size=10,
            health_check=check_connection,
        )

        async with pool.acquire() as conn:
            await conn.execute(query)
    """

    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        max_size: int = 10,
        min_size: int = 1,
        health_check: Optional[Callable[[T], Awaitable[bool]]] = None,
        max_idle_time: float = 300.0,  # 5 minutes
        health_check_interval: float = 60.0,
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.health_check = health_check
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval

        self._pool: asyncio.Queue[Tuple[T, float]] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()
        self._closed = False

        # Metrics
        self._total_acquired = 0
        self._total_released = 0
        self._total_created = 0
        self._total_destroyed = 0
        self._health_check_failures = 0

    async def _create_resource(self) -> T:
        """Create a new resource."""
        resource = await self.factory()
        self._total_created += 1
        return resource

    async def _destroy_resource(self, resource: T) -> None:
        """Destroy a resource."""
        self._total_destroyed += 1
        if hasattr(resource, "close"):
            try:
                await resource.close()
            except Exception:
                pass

    async def acquire(self) -> "ResourcePoolContext[T]":
        """Acquire a resource from the pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")

        # Try to get from pool
        while True:
            try:
                resource, created_at = self._pool.get_nowait()

                # Check if too old
                if time.monotonic() - created_at > self.max_idle_time:
                    await self._destroy_resource(resource)
                    async with self._lock:
                        self._size -= 1
                    continue

                # Health check
                if self.health_check:
                    try:
                        if not await self.health_check(resource):
                            self._health_check_failures += 1
                            await self._destroy_resource(resource)
                            async with self._lock:
                                self._size -= 1
                            continue
                    except Exception:
                        self._health_check_failures += 1
                        await self._destroy_resource(resource)
                        async with self._lock:
                            self._size -= 1
                        continue

                self._total_acquired += 1
                return ResourcePoolContext(self, resource)

            except asyncio.QueueEmpty:
                pass

            # Create new if under limit
            async with self._lock:
                if self._size < self.max_size:
                    self._size += 1
                    break

            # Wait for available resource
            resource, _ = await self._pool.get()
            self._total_acquired += 1
            return ResourcePoolContext(self, resource)

        # Create new resource
        resource = await self._create_resource()
        self._total_acquired += 1
        return ResourcePoolContext(self, resource)

    async def release(self, resource: T) -> None:
        """Release a resource back to the pool."""
        self._total_released += 1

        if self._closed:
            await self._destroy_resource(resource)
            return

        try:
            self._pool.put_nowait((resource, time.monotonic()))
        except asyncio.QueueFull:
            await self._destroy_resource(resource)
            async with self._lock:
                self._size -= 1

    async def close(self) -> None:
        """Close the pool and all resources."""
        self._closed = True

        while not self._pool.empty():
            try:
                resource, _ = self._pool.get_nowait()
                await self._destroy_resource(resource)
            except asyncio.QueueEmpty:
                break

    @property
    def available(self) -> int:
        """Number of available resources."""
        return self._pool.qsize()

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        return {
            "size": self._size,
            "max_size": self.max_size,
            "available": self.available,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "total_created": self._total_created,
            "total_destroyed": self._total_destroyed,
            "health_check_failures": self._health_check_failures,
        }


class ResourcePoolContext(Generic[T]):
    """Context manager for resource pool."""

    def __init__(self, pool: ResourcePool[T], resource: T):
        self._pool = pool
        self._resource = resource

    async def __aenter__(self) -> T:
        return self._resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._pool.release(self._resource)


# =============================================================================
# STRUCTURED TASK GROUP (Python 3.11+ with fallback)
# =============================================================================

import sys

# Check Python version for native TaskGroup support
_NATIVE_TASKGROUP_AVAILABLE = sys.version_info >= (3, 11)


@dataclass
class TaskResult(Generic[T]):
    """Result from a task in a task group."""
    name: str
    result: Optional[T] = None
    exception: Optional[BaseException] = None
    duration_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled: bool = False

    @property
    def success(self) -> bool:
        return self.exception is None and not self.cancelled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "cancelled": self.cancelled,
            "exception": str(self.exception) if self.exception else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class TaskGroupError(Exception):
    """
    Exception raised when one or more tasks in a TaskGroup fail.

    Contains all exceptions from failed tasks.
    """
    def __init__(self, message: str, exceptions: List[BaseException]):
        super().__init__(message)
        self.exceptions = exceptions
        self.task_results: List[TaskResult] = []

    def __str__(self) -> str:
        exc_info = ", ".join(f"{type(e).__name__}: {e}" for e in self.exceptions[:5])
        if len(self.exceptions) > 5:
            exc_info += f", ... and {len(self.exceptions) - 5} more"
        return f"{self.args[0]} ({len(self.exceptions)} failures: {exc_info})"


class StructuredTaskGroup:
    """
    Structured concurrency TaskGroup with enhanced features.

    Provides Python 3.11+ TaskGroup semantics with:
    - Named tasks for debugging
    - Automatic cancellation on failure (configurable)
    - Task result collection
    - Metrics tracking
    - Graceful shutdown support
    - Fallback implementation for Python < 3.11

    Example:
        async with StructuredTaskGroup(name="data_processing") as tg:
            tg.create_task(fetch_data(), name="fetch")
            tg.create_task(process_data(), name="process")
            tg.create_task(save_results(), name="save")

        # All tasks completed (or failed with TaskGroupError)
        print(tg.get_results())

    With error handling:
        async with StructuredTaskGroup(
            name="batch_ops",
            cancel_on_error=False,  # Continue other tasks on failure
        ) as tg:
            for i, item in enumerate(items):
                tg.create_task(process(item), name=f"item_{i}")

        # Check individual results
        for result in tg.results:
            if result.success:
                print(f"{result.name}: {result.result}")
            else:
                print(f"{result.name} failed: {result.exception}")
    """

    def __init__(
        self,
        name: str = "unnamed",
        cancel_on_error: bool = True,
        max_concurrent: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        collect_results: bool = True,
    ):
        """
        Initialize StructuredTaskGroup.

        Args:
            name: Name of the task group for debugging/logging.
            cancel_on_error: If True, cancel all tasks when one fails.
            max_concurrent: Optional limit on concurrent tasks (uses semaphore).
            timeout_seconds: Optional timeout for the entire group.
            collect_results: Whether to collect results (disable for fire-and-forget).
        """
        self.name = name
        self.cancel_on_error = cancel_on_error
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.collect_results = collect_results

        self._tasks: List[asyncio.Task] = []
        self._task_names: Dict[asyncio.Task, str] = {}
        self._task_start_times: Dict[asyncio.Task, datetime] = {}
        self._results: List[TaskResult] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        self._started = False
        self._finished = False
        self._cancelled = False

        # For Python < 3.11 fallback
        self._exceptions: List[BaseException] = []

        # Metrics
        self._total_tasks_created = 0
        self._total_tasks_completed = 0
        self._total_tasks_failed = 0
        self._total_tasks_cancelled = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @property
    def results(self) -> List[TaskResult]:
        """Get collected task results."""
        return self._results.copy()

    @property
    def successful_results(self) -> List[TaskResult]:
        """Get only successful task results."""
        return [r for r in self._results if r.success]

    @property
    def failed_results(self) -> List[TaskResult]:
        """Get only failed task results."""
        return [r for r in self._results if not r.success]

    def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        name: Optional[str] = None,
    ) -> asyncio.Task[T]:
        """
        Create a task in the group.

        Args:
            coro: Coroutine to run.
            name: Optional name for the task.

        Returns:
            The created asyncio.Task.
        """
        if self._finished:
            raise RuntimeError(f"TaskGroup '{self.name}' has already finished")

        self._total_tasks_created += 1
        task_name = name or f"{self.name}_task_{self._total_tasks_created}"

        # Wrap coroutine with semaphore if limited concurrency
        if self._semaphore:
            wrapped_coro = self._run_with_semaphore(coro, task_name)
        else:
            wrapped_coro = self._run_task(coro, task_name)

        # Create the task
        task = asyncio.create_task(wrapped_coro, name=task_name)

        self._tasks.append(task)
        self._task_names[task] = task_name
        self._task_start_times[task] = datetime.now()

        # Add done callback for result collection
        task.add_done_callback(self._on_task_done)

        return task

    async def _run_with_semaphore(
        self,
        coro: Coroutine[Any, Any, T],
        task_name: str,
    ) -> T:
        """Run coroutine with semaphore for concurrency limiting."""
        async with self._semaphore:
            return await self._run_task(coro, task_name)

    async def _run_task(
        self,
        coro: Coroutine[Any, Any, T],
        task_name: str,
    ) -> T:
        """Run the actual coroutine and handle any exceptions."""
        try:
            return await coro
        except asyncio.CancelledError:
            raise  # Let cancellation propagate
        except BaseException as e:
            # Store exception for later
            async with self._lock:
                self._exceptions.append(e)

            # Cancel other tasks if configured
            if self.cancel_on_error and not self._cancelled:
                await self._cancel_all_tasks()

            raise

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Callback when a task completes."""
        task_name = self._task_names.get(task, "unknown")
        start_time = self._task_start_times.get(task)
        completed_at = datetime.now()
        duration_ms = (
            (completed_at - start_time).total_seconds() * 1000
            if start_time else 0.0
        )

        if self.collect_results:
            result = TaskResult(
                name=task_name,
                started_at=start_time,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

            if task.cancelled():
                result.cancelled = True
                self._total_tasks_cancelled += 1
            elif task.exception() is not None:
                result.exception = task.exception()
                self._total_tasks_failed += 1
            else:
                try:
                    result.result = task.result()
                except asyncio.CancelledError:
                    result.cancelled = True
                    self._total_tasks_cancelled += 1
                else:
                    self._total_tasks_completed += 1

            self._results.append(result)

    async def _cancel_all_tasks(self) -> None:
        """Cancel all pending tasks."""
        self._cancelled = True
        for task in self._tasks:
            if not task.done():
                task.cancel()

    async def __aenter__(self) -> "StructuredTaskGroup":
        """Enter the task group context."""
        self._started = True
        self._start_time = time.monotonic()
        logger.debug(f"TaskGroup '{self.name}' started")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit the task group, waiting for all tasks to complete."""
        try:
            if self._tasks:
                if self.timeout_seconds:
                    # Wait with timeout
                    try:
                        await asyncio.wait_for(
                            self._wait_for_tasks(),
                            timeout=self.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"TaskGroup '{self.name}' timed out after "
                            f"{self.timeout_seconds}s, cancelling remaining tasks"
                        )
                        await self._cancel_all_tasks()
                        # Wait a bit for cancellation to complete
                        await asyncio.gather(*self._tasks, return_exceptions=True)
                        raise TaskGroupError(
                            f"TaskGroup '{self.name}' timed out",
                            [TimeoutError(f"Timeout after {self.timeout_seconds}s")],
                        )
                else:
                    await self._wait_for_tasks()
        finally:
            self._finished = True
            self._end_time = time.monotonic()
            duration = (self._end_time - self._start_time) * 1000 if self._start_time else 0
            logger.debug(
                f"TaskGroup '{self.name}' finished in {duration:.1f}ms "
                f"(completed={self._total_tasks_completed}, "
                f"failed={self._total_tasks_failed}, "
                f"cancelled={self._total_tasks_cancelled})"
            )

        # If there was an incoming exception, don't suppress it
        if exc_val is not None:
            return False

        # Check for task exceptions
        if self._exceptions:
            error = TaskGroupError(
                f"TaskGroup '{self.name}' had {len(self._exceptions)} task failures",
                self._exceptions,
            )
            error.task_results = self._results
            raise error

        return False

    async def _wait_for_tasks(self) -> None:
        """Wait for all tasks to complete."""
        if not self._tasks:
            return

        # Use native TaskGroup on Python 3.11+ for better error handling
        if _NATIVE_TASKGROUP_AVAILABLE and not self.max_concurrent:
            # Tasks already created, just gather them
            await asyncio.gather(*self._tasks, return_exceptions=True)
        else:
            # Fallback: gather with exception collection
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def get_metrics(self) -> Dict[str, Any]:
        """Get task group metrics."""
        duration_ms = None
        if self._start_time:
            end = self._end_time or time.monotonic()
            duration_ms = (end - self._start_time) * 1000

        return {
            "name": self.name,
            "total_tasks": self._total_tasks_created,
            "completed": self._total_tasks_completed,
            "failed": self._total_tasks_failed,
            "cancelled": self._total_tasks_cancelled,
            "duration_ms": duration_ms,
            "started": self._started,
            "finished": self._finished,
            "success_rate": (
                self._total_tasks_completed / self._total_tasks_created
                if self._total_tasks_created > 0 else 0
            ),
        }

    def get_results(self) -> Dict[str, TaskResult]:
        """Get results as a dictionary keyed by task name."""
        return {r.name: r for r in self._results}


async def run_in_task_group(
    coros: List[Coroutine[Any, Any, T]],
    names: Optional[List[str]] = None,
    group_name: str = "batch",
    max_concurrent: Optional[int] = None,
    cancel_on_error: bool = True,
    timeout_seconds: Optional[float] = None,
) -> List[TaskResult[T]]:
    """
    Convenience function to run multiple coroutines in a task group.

    Args:
        coros: List of coroutines to run.
        names: Optional names for each coroutine.
        group_name: Name for the task group.
        max_concurrent: Optional limit on concurrent tasks.
        cancel_on_error: Whether to cancel all tasks on first error.
        timeout_seconds: Optional timeout for the entire group.

    Returns:
        List of TaskResults for each coroutine.

    Example:
        results = await run_in_task_group(
            [fetch(url) for url in urls],
            names=[f"fetch_{i}" for i, _ in enumerate(urls)],
            max_concurrent=5,
        )

        for result in results:
            if result.success:
                print(f"{result.name}: {result.result}")
    """
    async with StructuredTaskGroup(
        name=group_name,
        max_concurrent=max_concurrent,
        cancel_on_error=cancel_on_error,
        timeout_seconds=timeout_seconds,
    ) as tg:
        for i, coro in enumerate(coros):
            name = names[i] if names and i < len(names) else f"task_{i}"
            tg.create_task(coro, name=name)

    return tg.results


async def scatter_gather(
    func: Callable[..., Awaitable[T]],
    items: List[Any],
    max_concurrent: int = 10,
    cancel_on_error: bool = False,
    timeout_per_item: Optional[float] = None,
) -> List[TaskResult[T]]:
    """
    Apply an async function to multiple items with concurrency control.

    Like asyncio.gather but with:
    - Concurrency limiting
    - Named tasks for debugging
    - Individual timeouts
    - Result collection with success/failure tracking

    Args:
        func: Async function to apply to each item.
        items: Items to process.
        max_concurrent: Maximum concurrent executions.
        cancel_on_error: Whether to cancel remaining on first error.
        timeout_per_item: Optional timeout per item.

    Returns:
        List of TaskResults.

    Example:
        async def fetch_user(user_id: int) -> User:
            ...

        results = await scatter_gather(
            fetch_user,
            user_ids,
            max_concurrent=5,
            timeout_per_item=10.0,
        )
    """
    async def wrapped_call(item: Any, index: int) -> T:
        if timeout_per_item:
            return await asyncio.wait_for(
                func(item),
                timeout=timeout_per_item,
            )
        return await func(item)

    coros = [wrapped_call(item, i) for i, item in enumerate(items)]
    names = [f"item_{i}" for i in range(len(items))]

    return await run_in_task_group(
        coros,
        names=names,
        group_name="scatter_gather",
        max_concurrent=max_concurrent,
        cancel_on_error=cancel_on_error,
    )


# =============================================================================
# ASYNC CONTEXT MANAGEMENT
# =============================================================================

class AsyncContextGroup:
    """
    Manage multiple async context managers as a group.

    Ensures all contexts are properly entered and exited,
    even if some fail.

    Example:
        async with AsyncContextGroup() as group:
            db = await group.enter(get_db_connection())
            cache = await group.enter(get_cache_connection())
            queue = await group.enter(get_queue_connection())

            # Use db, cache, queue...

        # All connections properly closed
    """

    def __init__(self):
        self._contexts: List[Any] = []
        self._entered: List[Any] = []
        self._lock = asyncio.Lock()

    async def enter(self, context_manager: Any) -> Any:
        """Enter a context manager and track it."""
        async with self._lock:
            self._contexts.append(context_manager)

        result = await context_manager.__aenter__()

        async with self._lock:
            self._entered.append(context_manager)

        return result

    async def __aenter__(self) -> "AsyncContextGroup":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit all entered context managers in reverse order."""
        exceptions = []

        # Exit in reverse order
        for cm in reversed(self._entered):
            try:
                await cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                exceptions.append(e)
                logger.error(f"Error exiting context manager: {e}")

        if exceptions and exc_val is None:
            # Raise the first exception if no incoming exception
            raise exceptions[0]

        return False


# =============================================================================
# EVENT COORDINATION
# =============================================================================

class AsyncBarrier:
    """
    Async barrier for coordinating multiple tasks.

    All tasks must reach the barrier before any can proceed.

    Example:
        barrier = AsyncBarrier(3)  # Wait for 3 tasks

        async def worker(id: int):
            print(f"Worker {id} preparing")
            await barrier.wait()  # Wait for all workers
            print(f"Worker {id} proceeding")

        await asyncio.gather(
            worker(1),
            worker(2),
            worker(3),
        )
    """

    def __init__(self, parties: int):
        """
        Initialize barrier.

        Args:
            parties: Number of tasks that must wait.
        """
        if parties < 1:
            raise ValueError("parties must be >= 1")

        self._parties = parties
        self._count = 0
        self._generation = 0
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def wait(self) -> int:
        """
        Wait at the barrier.

        Returns:
            The arrival index (0 to parties-1).
        """
        async with self._lock:
            gen = self._generation
            index = self._count
            self._count += 1

            if self._count >= self._parties:
                # Last one to arrive - release everyone
                self._count = 0
                self._generation += 1
                self._event.set()
                self._event = asyncio.Event()  # Reset for next use
                return index

        # Wait for release
        while True:
            await self._event.wait()
            async with self._lock:
                if self._generation > gen:
                    return index

    async def reset(self) -> None:
        """Reset the barrier (releases waiting tasks with error)."""
        async with self._lock:
            self._count = 0
            self._generation += 1
            self._event.set()

    @property
    def parties(self) -> int:
        """Number of parties."""
        return self._parties

    @property
    def n_waiting(self) -> int:
        """Number currently waiting."""
        return self._count


class AsyncLatch:
    """
    Async countdown latch.

    Waiters block until the count reaches zero.

    Example:
        latch = AsyncLatch(5)  # Count down from 5

        async def worker(id: int):
            await do_work()
            latch.count_down()  # Signal completion

        async def main():
            # Start workers
            for i in range(5):
                asyncio.create_task(worker(i))

            # Wait for all to complete
            await latch.wait()
            print("All workers done!")
    """

    def __init__(self, count: int):
        """
        Initialize latch.

        Args:
            count: Initial count.
        """
        if count < 0:
            raise ValueError("count must be >= 0")

        self._count = count
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()

        if count == 0:
            self._event.set()

    def count_down(self, n: int = 1) -> None:
        """Decrement the count."""
        asyncio.create_task(self._async_count_down(n))

    async def _async_count_down(self, n: int) -> None:
        """Async count down implementation."""
        async with self._lock:
            self._count = max(0, self._count - n)
            if self._count == 0:
                self._event.set()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for count to reach zero.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if count reached zero, False if timed out.
        """
        if timeout is None:
            await self._event.wait()
            return True

        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def count(self) -> int:
        """Current count."""
        return self._count
