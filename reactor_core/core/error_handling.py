"""
Unified Error Handling and Resilience Patterns - v92.0
=======================================================

Provides centralized error handling with:
- Circuit breaker pattern for external services
- Retry with exponential backoff
- Bulkhead isolation
- Fallback strategies
- Error classification and routing
- Observability integration

This module is the single source of truth for resilience patterns.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
ExceptionT = TypeVar("ExceptionT", bound=Exception)


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for routing."""
    TRANSIENT = "transient"      # Retry-able errors
    PERMANENT = "permanent"       # Non-retry-able errors
    RESOURCE = "resource"         # Resource exhaustion
    NETWORK = "network"           # Network-related
    VALIDATION = "validation"     # Input validation
    CONFIGURATION = "configuration"  # Config errors
    EXTERNAL = "external"         # External service errors
    INTERNAL = "internal"         # Internal logic errors


@dataclass
class ClassifiedError:
    """Error with classification metadata."""
    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    is_retryable: bool
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    traceback_str: str = ""

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> "ClassifiedError":
        """Classify an exception."""
        # Determine category and severity
        category, severity, retryable = cls._classify(exception)

        return cls(
            exception=exception,
            category=category,
            severity=severity,
            is_retryable=retryable,
            context=context or {},
            traceback_str=traceback.format_exc(),
        )

    @staticmethod
    def _classify(exception: Exception) -> Tuple[ErrorCategory, ErrorSeverity, bool]:
        """Classify exception into category, severity, and retryability."""
        exc_type = type(exception).__name__
        exc_msg = str(exception).lower()

        # Network errors - usually transient
        if any(net in exc_type.lower() for net in ["connection", "timeout", "network", "socket"]):
            return ErrorCategory.NETWORK, ErrorSeverity.WARNING, True

        if "timeout" in exc_msg or "timed out" in exc_msg:
            return ErrorCategory.NETWORK, ErrorSeverity.WARNING, True

        # Resource errors
        if any(res in exc_msg for res in ["memory", "oom", "out of memory", "cuda"]):
            return ErrorCategory.RESOURCE, ErrorSeverity.ERROR, True

        if "disk" in exc_msg or "space" in exc_msg:
            return ErrorCategory.RESOURCE, ErrorSeverity.ERROR, False

        # Validation errors - not retryable
        if any(val in exc_type.lower() for val in ["validation", "value", "type", "attribute"]):
            return ErrorCategory.VALIDATION, ErrorSeverity.WARNING, False

        # Configuration errors
        if any(cfg in exc_msg for cfg in ["config", "setting", "environment"]):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.ERROR, False

        # External service errors
        if any(ext in exc_msg for ext in ["api", "service", "unavailable", "503", "502"]):
            return ErrorCategory.EXTERNAL, ErrorSeverity.WARNING, True

        # Rate limiting
        if "rate" in exc_msg and "limit" in exc_msg:
            return ErrorCategory.EXTERNAL, ErrorSeverity.WARNING, True

        # Default to internal error
        return ErrorCategory.INTERNAL, ErrorSeverity.ERROR, False


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_requests: int = 3
    error_categories: Set[ErrorCategory] = field(
        default_factory=lambda: {ErrorCategory.EXTERNAL, ErrorCategory.NETWORK}
    )


class CircuitBreaker:
    """
    Circuit breaker for external service protection.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Failures exceeded threshold, calls rejected
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Function arguments
            fallback: Optional fallback function
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result

        Raises:
            CircuitOpenError: If circuit is open and no fallback
        """
        async with self._lock:
            self._total_calls += 1

            # Check state transitions
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time > self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                    logger.info(f"Circuit breaker '{self.name}': OPEN -> HALF_OPEN")
                else:
                    self._total_rejections += 1
                    if fallback:
                        return fallback(*args, **kwargs) if not asyncio.iscoroutinefunction(fallback) else await fallback(*args, **kwargs)
                    raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")

        # Execute call
        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                self._success_count += 1

                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.config.half_open_requests:
                        self._state = CircuitState.CLOSED
                        self._failure_count = 0
                        logger.info(f"Circuit breaker '{self.name}': HALF_OPEN -> CLOSED")

                elif self._state == CircuitState.CLOSED:
                    # Decay failure count on success
                    self._failure_count = max(0, self._failure_count - 1)

            return result

        except Exception as e:
            classified = ClassifiedError.from_exception(e)

            # Only count errors in tracked categories
            if classified.category in self.config.error_categories:
                async with self._lock:
                    self._failure_count += 1
                    self._total_failures += 1
                    self._last_failure_time = time.time()

                    if self._failure_count >= self.config.failure_threshold:
                        self._state = CircuitState.OPEN
                        logger.warning(
                            f"Circuit breaker '{self.name}': CLOSED -> OPEN "
                            f"(failures: {self._failure_count})"
                        )

            if fallback:
                return fallback(*args, **kwargs) if not asyncio.iscoroutinefunction(fallback) else await fallback(*args, **kwargs)
            raise

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
        }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# RETRY WITH BACKOFF
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on: Set[ErrorCategory] = field(
        default_factory=lambda: {
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.EXTERNAL,
        }
    )


class RetryHandler:
    """
    Retry handler with exponential backoff.

    Features:
    - Configurable retry attempts
    - Exponential backoff with jitter
    - Error category filtering
    - Callback hooks
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
    ):
        self.config = config or RetryConfig()
        self.on_retry = on_retry

    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                classified = ClassifiedError.from_exception(e)

                # Check if error is retryable
                if not classified.is_retryable:
                    raise

                if classified.category not in self.config.retry_on:
                    raise

                if attempt >= self.config.max_retries:
                    raise

                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay,
                )
                jitter = delay * self.config.jitter * random.random()
                delay += jitter

                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} "
                    f"after {delay:.2f}s: {e}"
                )

                if self.on_retry:
                    self.on_retry(attempt + 1, e)

                await asyncio.sleep(delay)

        raise last_exception


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    retry_on: Optional[Set[ErrorCategory]] = None,
):
    """
    Decorator for retry with backoff.

    Usage:
        @with_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        retry_on=retry_on or {ErrorCategory.TRANSIENT, ErrorCategory.NETWORK},
    )
    handler = RetryHandler(config)

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await handler.execute(func, *args, **kwargs)
        return wrapper

    return decorator


# =============================================================================
# BULKHEAD ISOLATION
# =============================================================================


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent executions to prevent cascade failures.
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_waiting: int = 100,
        timeout: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self.timeout = timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_rejections = 0
        self._total_timeouts = 0

    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute function within bulkhead.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            BulkheadFullError: If bulkhead is at capacity
            asyncio.TimeoutError: If timeout exceeded
        """
        async with self._lock:
            self._total_calls += 1

            if self._waiting >= self.max_waiting:
                self._total_rejections += 1
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' is full: "
                    f"{self._waiting} waiting, max {self.max_waiting}"
                )

            self._waiting += 1

        try:
            # Wait for semaphore with timeout
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                self._total_timeouts += 1
                raise

            try:
                return await func(*args, **kwargs)
            finally:
                self._semaphore.release()

        finally:
            async with self._lock:
                self._waiting -= 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_concurrent": self.max_concurrent,
            "current_concurrent": self.max_concurrent - self._semaphore._value,
            "waiting": self._waiting,
            "total_calls": self._total_calls,
            "total_rejections": self._total_rejections,
            "total_timeouts": self._total_timeouts,
        }


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""
    pass


# =============================================================================
# FALLBACK STRATEGIES
# =============================================================================


class FallbackStrategy:
    """Base class for fallback strategies."""

    async def execute(
        self,
        exception: Exception,
        context: Dict[str, Any],
    ) -> Any:
        """Execute fallback logic."""
        raise NotImplementedError


class StaticFallback(FallbackStrategy):
    """Return a static value as fallback."""

    def __init__(self, value: Any):
        self.value = value

    async def execute(
        self,
        exception: Exception,
        context: Dict[str, Any],
    ) -> Any:
        return self.value


class CacheFallback(FallbackStrategy):
    """Return cached value as fallback."""

    def __init__(self, cache_key: str, cache: Dict[str, Any]):
        self.cache_key = cache_key
        self.cache = cache

    async def execute(
        self,
        exception: Exception,
        context: Dict[str, Any],
    ) -> Any:
        return self.cache.get(self.cache_key)


class CallableFallback(FallbackStrategy):
    """Execute a callable as fallback."""

    def __init__(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    async def execute(
        self,
        exception: Exception,
        context: Dict[str, Any],
    ) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(*self.args, **self.kwargs)
        return self.func(*self.args, **self.kwargs)


# =============================================================================
# ERROR REGISTRY
# =============================================================================


class ErrorRegistry:
    """
    Central registry for error tracking and routing.

    Provides:
    - Error collection and aggregation
    - Pattern detection
    - Alert generation
    """

    def __init__(
        self,
        max_errors: int = 1000,
        alert_threshold: int = 10,
        alert_window: float = 60.0,
    ):
        self.max_errors = max_errors
        self.alert_threshold = alert_threshold
        self.alert_window = alert_window

        self._errors: Deque[ClassifiedError] = deque(maxlen=max_errors)
        self._error_counts: Dict[ErrorCategory, int] = {}
        self._alert_callbacks: List[Callable[[str, Dict], None]] = []
        self._lock = asyncio.Lock()

    async def record(
        self,
        error: ClassifiedError,
        emit_alert: bool = True,
    ) -> None:
        """Record an error."""
        async with self._lock:
            self._errors.append(error)
            self._error_counts[error.category] = (
                self._error_counts.get(error.category, 0) + 1
            )

        # Check for alert condition
        if emit_alert and await self._should_alert(error.category):
            await self._emit_alert(error)

    async def _should_alert(self, category: ErrorCategory) -> bool:
        """Check if alert threshold exceeded."""
        cutoff = time.time() - self.alert_window

        async with self._lock:
            recent = sum(
                1 for e in self._errors
                if e.category == category and e.timestamp > cutoff
            )
            return recent >= self.alert_threshold

    async def _emit_alert(self, error: ClassifiedError) -> None:
        """Emit alert for error spike."""
        alert_data = {
            "category": error.category.value,
            "severity": error.severity.value,
            "count": self._error_counts.get(error.category, 0),
            "sample_error": str(error.exception),
            "timestamp": datetime.now().isoformat(),
        }

        logger.error(
            f"ERROR ALERT: {error.category.value} errors "
            f"exceeded threshold ({self.alert_threshold} in {self.alert_window}s)"
        )

        for callback in self._alert_callbacks:
            try:
                callback(f"error_spike_{error.category.value}", alert_data)
            except Exception as e:
                logger.debug(f"Alert callback failed: {e}")

    def on_alert(self, callback: Callable[[str, Dict], None]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_errors": len(self._errors),
            "error_counts": dict(self._error_counts),
            "recent_errors": [
                {
                    "category": e.category.value,
                    "severity": e.severity.value,
                    "message": str(e.exception)[:100],
                    "timestamp": e.timestamp,
                }
                for e in list(self._errors)[-10:]
            ],
        }


# =============================================================================
# RESILIENCE DECORATOR
# =============================================================================


def resilient(
    circuit_breaker: Optional[CircuitBreaker] = None,
    retry_config: Optional[RetryConfig] = None,
    bulkhead: Optional[Bulkhead] = None,
    fallback: Optional[FallbackStrategy] = None,
    error_registry: Optional[ErrorRegistry] = None,
):
    """
    Decorator combining multiple resilience patterns.

    Usage:
        @resilient(
            circuit_breaker=CircuitBreaker("api"),
            retry_config=RetryConfig(max_retries=3),
            fallback=StaticFallback(None),
        )
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        retry_handler = RetryHandler(retry_config) if retry_config else None

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async def execute():
                if retry_handler:
                    return await retry_handler.execute(func, *args, **kwargs)
                return await func(*args, **kwargs)

            try:
                # Apply bulkhead
                if bulkhead:
                    execute_func = execute
                    execute = lambda: bulkhead.execute(execute_func)

                # Apply circuit breaker
                if circuit_breaker:
                    fallback_func = None
                    if fallback:
                        async def fallback_wrapper():
                            return await fallback.execute(None, {})
                        fallback_func = fallback_wrapper

                    return await circuit_breaker.call(execute, fallback=fallback_func)

                return await execute()

            except Exception as e:
                # Record error
                if error_registry:
                    classified = ClassifiedError.from_exception(e)
                    await error_registry.record(classified)

                # Try fallback
                if fallback:
                    return await fallback.execute(e, {"args": args, "kwargs": kwargs})

                raise

        return wrapper

    return decorator


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Global error registry
_global_registry = ErrorRegistry()


def get_error_registry() -> ErrorRegistry:
    """Get global error registry."""
    return _global_registry


def record_error(exception: Exception, context: Optional[Dict] = None) -> None:
    """Record error to global registry."""
    classified = ClassifiedError.from_exception(exception, context)
    asyncio.create_task(_global_registry.record(classified))
