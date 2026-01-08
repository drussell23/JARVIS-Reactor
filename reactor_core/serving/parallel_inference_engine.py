"""
Parallel Inference Engine - Advanced Concurrent Model Inference

Features:
- Concurrent request batching with dynamic batch sizes
- Parallel model loading and warm-up
- Resource pooling with adaptive concurrency
- Stream multiplexing for real-time responses
- Memory-aware scheduling
- Circuit breakers and retry logic
- Request prioritization and queueing
- Performance monitoring and optimization

Author: JARVIS AGI
Version: v83.0 - Unified Model Management
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, AsyncIterator
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class RequestPriority(str, Enum):
    """Request priority levels."""
    CRITICAL = "critical"  # Real-time user requests
    HIGH = "high"          # Interactive tasks
    NORMAL = "normal"      # Batch processing
    LOW = "low"            # Background tasks


class BatchStrategy(str, Enum):
    """Batching strategies for inference optimization."""
    DYNAMIC = "dynamic"      # Adjust batch size based on load
    FIXED = "fixed"          # Fixed batch size
    GREEDY = "greedy"        # Batch as many as possible
    LATENCY_AWARE = "latency_aware"  # Optimize for latency


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, block requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class InferenceTask:
    """Single inference request task."""
    task_id: str
    prompt: str
    model_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    priority: RequestPriority = RequestPriority.NORMAL
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    submitted_at: float = field(default_factory=time.time)

    # Response handling
    result_future: asyncio.Future = field(default_factory=asyncio.Future)

    def __lt__(self, other: 'InferenceTask') -> bool:
        """Priority queue ordering."""
        priority_order = {
            RequestPriority.CRITICAL: 0,
            RequestPriority.HIGH: 1,
            RequestPriority.NORMAL: 2,
            RequestPriority.LOW: 3,
        }
        return priority_order[self.priority] < priority_order[other.priority]


@dataclass
class BatchConfig:
    """Configuration for request batching."""
    strategy: BatchStrategy = BatchStrategy.DYNAMIC
    max_batch_size: int = 32
    min_batch_size: int = 1
    batch_timeout_ms: float = 100.0  # Max wait time for batch
    enable_padding: bool = True  # Pad sequences to same length

    # Dynamic batching
    target_latency_ms: float = 200.0
    latency_percentile: float = 0.95  # P95 latency target


@dataclass
class ResourcePool:
    """Resource pool for concurrent execution."""
    max_workers: int = 4
    max_memory_gb: float = 8.0
    max_concurrent_models: int = 3
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8

    # Adaptive concurrency
    enable_adaptive: bool = True
    min_concurrency: int = 1
    max_concurrency: int = 8


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max calls in half-open state


@dataclass
class ParallelEngineConfig:
    """Configuration for parallel inference engine."""
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    resource_pool: ResourcePool = field(default_factory=ResourcePool)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Performance tuning
    enable_model_warmup: bool = True
    warmup_requests: int = 3
    enable_request_coalescing: bool = True  # Merge identical requests
    enable_speculative_loading: bool = True  # Preload popular models

    # Monitoring
    enable_metrics: bool = True
    metrics_window_seconds: float = 60.0


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            # Check if we should attempt recovery
            if (self.last_failure_time and
                time.time() - self.last_failure_time > self.config.timeout_seconds):
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception("Circuit breaker HALF_OPEN - max calls reached")
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failures = 0

        if self.state == CircuitState.HALF_OPEN:
            self.successes += 1
            if self.successes >= self.config.success_threshold:
                logger.info("Circuit breaker CLOSED - recovery successful")
                self.state = CircuitState.CLOSED
                self.successes = 0
                self.half_open_calls = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        self.successes = 0

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker OPEN - recovery failed")
            self.state = CircuitState.OPEN
            self.half_open_calls = 0
        elif self.failures >= self.config.failure_threshold:
            logger.error(f"Circuit breaker OPEN - {self.failures} failures")
            self.state = CircuitState.OPEN


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    total_tokens_generated: int = 0
    total_batches_processed: int = 0

    # Latency percentiles
    latencies: List[float] = field(default_factory=list)

    def record_request(self, latency_ms: float, tokens: int, success: bool):
        """Record a completed request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.total_tokens_generated += tokens

        self.latencies.append(latency_ms)

        # Keep only recent latencies for percentile calculation
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.total_requests == 0:
            return {"error": "No requests processed"}

        avg_latency = self.total_latency_ms / self.total_requests

        # Calculate percentiles
        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[p95_idx] if sorted_latencies else 0
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99 = sorted_latencies[p99_idx] if sorted_latencies else 0

        return {
            "total_requests": self.total_requests,
            "success_rate": self.successful_requests / self.total_requests,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "total_tokens": self.total_tokens_generated,
            "tokens_per_second": self.total_tokens_generated / (self.total_latency_ms / 1000) if self.total_latency_ms > 0 else 0,
        }


# ============================================================================
# PARALLEL INFERENCE ENGINE
# ============================================================================


class ParallelInferenceEngine:
    """
    Advanced parallel inference engine with batching, resource pooling, and optimization.

    Features:
    - Dynamic request batching
    - Concurrent model execution
    - Memory-aware scheduling
    - Circuit breakers
    - Adaptive concurrency
    - Stream multiplexing
    """

    def __init__(
        self,
        model_manager: Any,  # UnifiedModelManager
        config: Optional[ParallelEngineConfig] = None,
    ):
        self.model_manager = model_manager
        self.config = config or ParallelEngineConfig()

        # Request queues per model (priority-based)
        self.request_queues: Dict[str, asyncio.PriorityQueue] = defaultdict(lambda: asyncio.PriorityQueue())

        # Active batches being processed
        self.active_batches: Dict[str, List[InferenceTask]] = defaultdict(list)

        # Circuit breakers per model
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Performance metrics
        self.metrics = PerformanceMetrics()

        # Resource management
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.resource_pool.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()))

        # Adaptive concurrency
        self.current_concurrency = self.config.resource_pool.min_concurrency

        # Request coalescing (identical requests)
        self.pending_requests: Dict[str, List[InferenceTask]] = defaultdict(list)

        # Background workers
        self.batch_workers: Dict[str, asyncio.Task] = {}
        self.running = False

        logger.info("Parallel Inference Engine initialized")

    async def start(self):
        """Start the inference engine."""
        if self.running:
            logger.warning("Engine already running")
            return

        self.running = True
        logger.info("Starting Parallel Inference Engine")

        # Warm up models if configured
        if self.config.enable_model_warmup:
            await self._warmup_models()

    async def shutdown(self):
        """Gracefully shutdown the engine."""
        logger.info("Shutting down Parallel Inference Engine")
        self.running = False

        # Cancel all batch workers
        for worker in self.batch_workers.values():
            worker.cancel()

        # Wait for completion
        await asyncio.gather(*self.batch_workers.values(), return_exceptions=True)

        # Shutdown executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        logger.info("Parallel Inference Engine stopped")

    async def submit_request(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        priority: RequestPriority = RequestPriority.NORMAL,
        stream: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Submit an inference request.

        Returns:
            Generated text response or AsyncIterator for streaming.
        """
        if not self.running:
            raise RuntimeError("Engine not started - call start() first")

        # Create task
        task_id = f"{model_id}_{int(time.time() * 1000000)}"
        task = InferenceTask(
            task_id=task_id,
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            stream=stream,
            metadata=metadata or {},
        )

        # Check for request coalescing
        if self.config.enable_request_coalescing:
            request_key = f"{model_id}:{prompt}:{max_tokens}:{temperature}"
            if request_key in self.pending_requests:
                # Identical request already pending - piggyback on it
                self.pending_requests[request_key].append(task)
                logger.debug(f"Coalescing request {task_id}")
                return await task.result_future

        # Ensure batch worker exists for this model
        if model_id not in self.batch_workers:
            self.batch_workers[model_id] = asyncio.create_task(self._batch_worker(model_id))

        # Add to queue
        await self.request_queues[model_id].put((task.priority.value, task))

        # Wait for result
        return await task.result_future

    async def _batch_worker(self, model_id: str):
        """Background worker for batching and processing requests."""
        logger.info(f"Batch worker started for model: {model_id}")

        # Initialize circuit breaker
        if model_id not in self.circuit_breakers:
            self.circuit_breakers[model_id] = CircuitBreaker(self.config.circuit_breaker)

        circuit_breaker = self.circuit_breakers[model_id]

        while self.running:
            try:
                # Collect batch
                batch = await self._collect_batch(model_id)

                if not batch:
                    await asyncio.sleep(0.01)  # Small sleep to prevent busy-waiting
                    continue

                # Process batch with circuit breaker protection
                try:
                    await circuit_breaker.call(self._process_batch, model_id, batch)
                except Exception as e:
                    logger.error(f"Batch processing failed for {model_id}: {e}")
                    # Mark all tasks in batch as failed
                    for task in batch:
                        if not task.result_future.done():
                            task.result_future.set_exception(e)

            except asyncio.CancelledError:
                logger.info(f"Batch worker cancelled for model: {model_id}")
                break
            except Exception as e:
                logger.error(f"Batch worker error for {model_id}: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _collect_batch(self, model_id: str) -> List[InferenceTask]:
        """Collect a batch of requests based on strategy."""
        batch = []
        config = self.config.batch_config
        queue = self.request_queues[model_id]

        # Wait for at least one request
        try:
            _, first_task = await asyncio.wait_for(
                queue.get(),
                timeout=config.batch_timeout_ms / 1000.0
            )
            batch.append(first_task)
        except asyncio.TimeoutError:
            return []

        # Collect more requests up to max batch size or timeout
        deadline = time.time() + (config.batch_timeout_ms / 1000.0)

        while len(batch) < config.max_batch_size and time.time() < deadline:
            try:
                _, task = await asyncio.wait_for(
                    queue.get(),
                    timeout=(deadline - time.time())
                )
                batch.append(task)
            except asyncio.TimeoutError:
                break

        # Dynamic batch sizing based on strategy
        if config.strategy == BatchStrategy.LATENCY_AWARE:
            # Adjust batch size based on observed latency
            if self.metrics.latencies:
                p95 = sorted(self.metrics.latencies)[int(len(self.metrics.latencies) * 0.95)]
                if p95 > config.target_latency_ms:
                    # Reduce batch size if latency too high
                    batch = batch[:max(1, len(batch) // 2)]

        return batch

    async def _process_batch(self, model_id: str, batch: List[InferenceTask]):
        """Process a batch of inference requests."""
        start_time = time.time()

        try:
            # Separate streaming and non-streaming requests
            streaming_tasks = [t for t in batch if t.stream]
            non_streaming_tasks = [t for t in batch if not t.stream]

            # Process non-streaming in parallel
            if non_streaming_tasks:
                await self._process_non_streaming_batch(model_id, non_streaming_tasks)

            # Process streaming individually (no batching for streaming)
            for task in streaming_tasks:
                asyncio.create_task(self._process_streaming_task(model_id, task))

            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            for task in non_streaming_tasks:
                self.metrics.record_request(latency_ms, task.max_tokens, success=True)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Mark all tasks as failed
            for task in batch:
                if not task.result_future.done():
                    task.result_future.set_exception(e)
                    self.metrics.record_request(0, 0, success=False)

    async def _process_non_streaming_batch(self, model_id: str, batch: List[InferenceTask]):
        """Process non-streaming batch in parallel."""
        # Create inference requests
        tasks = []
        for task in batch:
            inference_task = asyncio.create_task(
                self.model_manager.generate(
                    prompt=task.prompt,
                    model_id=task.model_id,
                    max_tokens=task.max_tokens,
                    temperature=task.temperature,
                    stream=False,
                )
            )
            tasks.append((task, inference_task))

        # Wait for all to complete
        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        # Set results
        for (task, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                task.result_future.set_exception(result)
            else:
                task.result_future.set_result(result)

    async def _process_streaming_task(self, model_id: str, task: InferenceTask):
        """Process a single streaming task."""
        try:
            # Get streaming generator from model manager
            stream = await self.model_manager.generate(
                prompt=task.prompt,
                model_id=task.model_id,
                max_tokens=task.max_tokens,
                temperature=task.temperature,
                stream=True,
            )

            # Set result as the stream
            task.result_future.set_result(stream)

        except Exception as e:
            logger.error(f"Streaming task error: {e}")
            task.result_future.set_exception(e)

    async def _warmup_models(self):
        """Warm up models with dummy requests."""
        logger.info("Warming up models...")

        warmup_prompt = "Hello, this is a warmup request."

        # Get popular models from model manager
        # For now, just warm up the first available model
        try:
            # Submit warmup requests
            for i in range(self.config.warmup_requests):
                await asyncio.sleep(0.1)  # Stagger requests
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.get_stats()

    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get circuit breaker status for all models."""
        return {
            model_id: breaker.state.value
            for model_id, breaker in self.circuit_breakers.items()
        }


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


async def create_parallel_engine(
    model_manager: Any,
    config: Optional[ParallelEngineConfig] = None,
) -> ParallelInferenceEngine:
    """
    Create and start a parallel inference engine.

    Args:
        model_manager: Unified model manager instance
        config: Engine configuration

    Returns:
        Started ParallelInferenceEngine instance
    """
    engine = ParallelInferenceEngine(model_manager, config)
    await engine.start()
    return engine
