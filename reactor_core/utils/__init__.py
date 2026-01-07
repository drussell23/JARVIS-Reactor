"""
Utilities module for Night Shift Training Engine.

Provides:
- Environment detection (M1 Mac, GCP VM)
- Async helpers (rate limiting, batch processing)
- Structured logging configuration

ADVANCED PATTERNS (v76.0):
- CircuitBreaker: Fault tolerance with automatic recovery
- Backpressure: Adaptive load management
- Bulkhead: Failure isolation between components
- DeadLetterQueue: Failed operation tracking and retry
- HealthMonitor: Component health tracking
- MetricsCollector: Observability and performance tracking
- AdaptiveRateLimiter: Dynamic rate limiting based on success rates
- TimeoutPolicy: Configurable timeout with fallback strategies
- GracefulShutdown: Coordinated async shutdown
- ResourcePool: Pooled resource management with health checking
"""

from reactor_core.utils.environment import (
    detect_environment,
    EnvironmentType,
    EnvironmentInfo,
    get_recommended_config,
    get_quantization_config,
    print_environment_info,
)

from reactor_core.utils.async_helpers import (
    # Core async helpers
    AsyncSemaphore,
    TokenBucketRateLimiter,
    ParallelBatchProcessor,
    BatchResult,
    AsyncQueue,
    ProgressTracker,
    async_retry,
    gather_with_concurrency,
    run_with_timeout,
    # === ADVANCED PATTERNS (v76.0) ===
    # Circuit Breaker
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerOpenError,
    # Dead Letter Queue
    DeadLetterEntry,
    DeadLetterQueue,
    # Backpressure
    BackpressureStrategy,
    BackpressureConfig,
    BackpressureController,
    # Bulkhead
    Bulkhead,
    BulkheadFullError,
    # Adaptive Rate Limiting
    AdaptiveRateLimiter,
    # Health Monitoring
    HealthStatus,
    HealthCheck,
    HealthMonitor,
    # Metrics Collection
    MetricsCollector,
    # Timeout Policy
    TimeoutStrategy,
    TimeoutConfig,
    TimeoutPolicy,
    # Graceful Shutdown
    GracefulShutdown,
    # Resource Pool
    ResourcePool,
)

from reactor_core.utils.logging_config import (
    setup_logging,
    get_logger,
    LoggingConfig,
    LogContext,
    MetricsLogger,
    set_run_id,
    set_stage,
    set_context,
    clear_context,
    log_duration,
)

# Dependency management (v76.0)
from reactor_core.utils.dependencies import (
    DependencyStatus,
    Platform,
    AcceleratorType,
    DependencyInfo,
    DependencyChecker,
    LazyModule,
    TorchBackend,
    detect_platform,
    detect_accelerator,
    lazy_import,
    requires,
    optional_import,
    get_dependency_checker,
    get_torch_backend,
    validate_environment,
    print_environment_status,
)

__all__ = [
    # Environment
    "detect_environment",
    "EnvironmentType",
    "EnvironmentInfo",
    "get_recommended_config",
    "get_quantization_config",
    "print_environment_info",
    # Async helpers
    "AsyncSemaphore",
    "TokenBucketRateLimiter",
    "ParallelBatchProcessor",
    "BatchResult",
    "AsyncQueue",
    "ProgressTracker",
    "async_retry",
    "gather_with_concurrency",
    "run_with_timeout",
    # === ADVANCED PATTERNS (v76.0) ===
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    # Dead Letter Queue
    "DeadLetterEntry",
    "DeadLetterQueue",
    # Backpressure
    "BackpressureStrategy",
    "BackpressureConfig",
    "BackpressureController",
    # Bulkhead
    "Bulkhead",
    "BulkheadFullError",
    # Adaptive Rate Limiting
    "AdaptiveRateLimiter",
    # Health Monitoring
    "HealthStatus",
    "HealthCheck",
    "HealthMonitor",
    # Metrics Collection
    "MetricsCollector",
    # Timeout Policy
    "TimeoutStrategy",
    "TimeoutConfig",
    "TimeoutPolicy",
    # Graceful Shutdown
    "GracefulShutdown",
    # Resource Pool
    "ResourcePool",
    # Logging
    "setup_logging",
    "get_logger",
    "LoggingConfig",
    "LogContext",
    "MetricsLogger",
    "set_run_id",
    "set_stage",
    "set_context",
    "clear_context",
    "log_duration",
    # === DEPENDENCY MANAGEMENT (v76.0) ===
    "DependencyStatus",
    "Platform",
    "AcceleratorType",
    "DependencyInfo",
    "DependencyChecker",
    "LazyModule",
    "TorchBackend",
    "detect_platform",
    "detect_accelerator",
    "lazy_import",
    "requires",
    "optional_import",
    "get_dependency_checker",
    "get_torch_backend",
    "validate_environment",
    "print_environment_status",
]
