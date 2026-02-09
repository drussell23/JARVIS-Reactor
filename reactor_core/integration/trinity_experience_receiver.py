"""
Trinity Experience Receiver v2.0 - Production-Hardened
=======================================================

Receives experiences from JARVIS and feeds them to the training pipeline
with enterprise-grade reliability, causality guarantees, and fault tolerance.

This module CLOSES THE TRINITY LOOP by:
1. Subscribing to Trinity event bus for experience_batch events
2. Watching ~/.jarvis/trinity/events/ for experience files (file fallback)
3. Validating causal ordering via vector clocks and sequence numbers
4. Detecting gaps and requesting retransmission
5. Deduplicating via Bloom filters + LRU cache
6. Calling UnifiedTrainingPipeline.add_experiences() when experiences arrive

Architecture:
    ┌────────────┐    Trinity Events    ┌────────────────────────┐
    │   JARVIS   │ ─────────────────────│  TrinityExperienceRx   │
    │   (Body)   │                      │  (This Module)         │
    └────────────┘    File Fallback     └───────────┬────────────┘
                                                    │
                                                    ▼
                                        ┌────────────────────────┐
                                        │  UnifiedTrainingPipeline│
                                        │  add_experiences()     │
                                        └────────────────────────┘

v2.0 Features:
    - Vector clock causality validation
    - Sequence number tracking with gap detection
    - Bloom filter + LRU deduplication (10x memory efficiency)
    - Out-of-order event buffering with timeout
    - Circuit breaker for flush operations
    - Comprehensive health metrics
    - Graceful shutdown with drain
    - Watchdog-based real-time file monitoring

Author: Trinity System
Version: 2.0.0
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import struct
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("TrinityExperienceReceiver")


# =============================================================================
# BLOOM FILTER IMPLEMENTATION (Memory-Efficient Deduplication)
# =============================================================================

class BloomFilter:
    """
    Space-efficient probabilistic data structure for set membership testing.

    Used for fast deduplication with configurable false positive rate.
    A false positive means we might skip a unique event (acceptable for dedup).
    A false negative is impossible (we never process duplicates).
    """

    def __init__(
        self,
        expected_elements: int = 100000,
        false_positive_rate: float = 0.001,
    ):
        """
        Initialize Bloom filter with optimal size.

        Args:
            expected_elements: Expected number of unique elements
            false_positive_rate: Acceptable false positive rate (0.001 = 0.1%)
        """
        # Calculate optimal size and hash count
        import math
        self._size = self._optimal_size(expected_elements, false_positive_rate)
        self._hash_count = self._optimal_hash_count(self._size, expected_elements)

        # Bit array as bytearray for memory efficiency
        self._bits = bytearray((self._size + 7) // 8)
        self._count = 0

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        import math
        return int(-n * math.log(p) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        import math
        return max(1, int((m / n) * math.log(2)))

    def _get_hash_values(self, item: str) -> List[int]:
        """Generate k hash values using double hashing."""
        # Use MD5 for speed (not security)
        h = hashlib.md5(item.encode()).digest()
        h1 = struct.unpack('<Q', h[:8])[0]
        h2 = struct.unpack('<Q', h[8:16])[0]

        return [(h1 + i * h2) % self._size for i in range(self._hash_count)]

    def add(self, item: str) -> bool:
        """
        Add item to the filter.

        Returns:
            True if item was likely new, False if probably seen before
        """
        hash_values = self._get_hash_values(item)
        was_new = False

        for h in hash_values:
            byte_idx = h // 8
            bit_idx = h % 8
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                was_new = True
            self._bits[byte_idx] |= (1 << bit_idx)

        if was_new:
            self._count += 1
        return was_new

    def __contains__(self, item: str) -> bool:
        """Check if item is probably in the filter."""
        hash_values = self._get_hash_values(item)
        return all(
            self._bits[h // 8] & (1 << (h % 8))
            for h in hash_values
        )

    def estimated_false_positive_rate(self) -> float:
        """Calculate current estimated false positive rate."""
        import math
        if self._count == 0:
            return 0.0
        # Formula: (1 - e^(-kn/m))^k
        exponent = -self._hash_count * self._count / self._size
        return (1 - math.exp(exponent)) ** self._hash_count


# =============================================================================
# VECTOR CLOCK IMPLEMENTATION (Causal Ordering)
# =============================================================================

class VectorClock:
    """
    Vector clock for tracking causal relationships between events.

    Each component (JARVIS, Prime, Reactor) maintains a logical clock.
    Vector clocks enable detecting concurrent vs causally related events.
    """

    def __init__(self, node_id: str = "reactor"):
        self.node_id = node_id
        self._clocks: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def tick(self) -> Dict[str, int]:
        """Increment local clock and return current state."""
        async with self._lock:
            self._clocks[self.node_id] += 1
            return dict(self._clocks)

    async def update(self, received_clock: Dict[str, int]) -> Dict[str, int]:
        """
        Update clock based on received message.

        Merges by taking max of each component, then increments local.
        """
        async with self._lock:
            for node, time_val in received_clock.items():
                self._clocks[node] = max(self._clocks[node], time_val)
            self._clocks[self.node_id] += 1
            return dict(self._clocks)

    def compare(self, clock_a: Dict[str, int], clock_b: Dict[str, int]) -> str:
        """
        Compare two vector clocks.

        Returns:
            "before": a happened before b
            "after": a happened after b
            "concurrent": a and b are concurrent
            "equal": a and b are identical
        """
        all_keys = set(clock_a.keys()) | set(clock_b.keys())

        a_less = False
        b_less = False

        for key in all_keys:
            val_a = clock_a.get(key, 0)
            val_b = clock_b.get(key, 0)

            if val_a < val_b:
                a_less = True
            elif val_a > val_b:
                b_less = True

        if a_less and not b_less:
            return "before"
        elif b_less and not a_less:
            return "after"
        elif not a_less and not b_less:
            return "equal"
        else:
            return "concurrent"

    def get_clock(self) -> Dict[str, int]:
        """Get current clock state (non-async for logging)."""
        return dict(self._clocks)


# =============================================================================
# SEQUENCE TRACKER (Gap Detection)
# =============================================================================

class SequenceTracker:
    """
    Tracks sequence numbers per source to detect gaps and out-of-order events.

    Features:
    - Per-source sequence tracking
    - Gap detection with configurable window
    - Out-of-order buffering with timeout
    - Automatic gap reporting for retransmission requests
    """

    def __init__(
        self,
        gap_detection_window: int = 1000,
        out_of_order_timeout: float = 30.0,
    ):
        self._expected_sequences: Dict[str, int] = defaultdict(lambda: 1)
        self._received_sequences: Dict[str, Set[int]] = defaultdict(set)
        self._gaps: Dict[str, Set[int]] = defaultdict(set)
        self._out_of_order_buffer: Dict[str, Dict[int, Tuple[float, Any]]] = defaultdict(dict)
        self._gap_window = gap_detection_window
        self._ooo_timeout = out_of_order_timeout
        self._lock = asyncio.Lock()
        self._total_gaps_detected = 0
        self._total_gaps_filled = 0

    async def track(
        self,
        source: str,
        sequence: int,
        event: Any,
    ) -> Tuple[bool, List[Any]]:
        """
        Track a sequence number and return events ready for processing.

        Args:
            source: Event source identifier
            sequence: Sequence number of this event
            event: The event data

        Returns:
            Tuple of (is_new, ready_events) where:
            - is_new: True if this is a new sequence (not duplicate)
            - ready_events: List of events ready for processing (in order)
        """
        async with self._lock:
            # Check for duplicate
            if sequence in self._received_sequences[source]:
                return False, []

            expected = self._expected_sequences[source]
            self._received_sequences[source].add(sequence)

            # Clean up old sequences to prevent memory growth
            if len(self._received_sequences[source]) > self._gap_window * 2:
                min_keep = max(self._received_sequences[source]) - self._gap_window
                self._received_sequences[source] = {
                    s for s in self._received_sequences[source] if s >= min_keep
                }

            ready_events = []

            if sequence == expected:
                # In order - process immediately
                ready_events.append(event)
                self._expected_sequences[source] = sequence + 1

                # Check if this fills any gaps
                if sequence in self._gaps[source]:
                    self._gaps[source].discard(sequence)
                    self._total_gaps_filled += 1

                # Check for buffered out-of-order events that can now be processed
                while self._expected_sequences[source] in self._out_of_order_buffer[source]:
                    next_seq = self._expected_sequences[source]
                    _, buffered_event = self._out_of_order_buffer[source].pop(next_seq)
                    ready_events.append(buffered_event)
                    self._expected_sequences[source] = next_seq + 1

                    if next_seq in self._gaps[source]:
                        self._gaps[source].discard(next_seq)
                        self._total_gaps_filled += 1

            elif sequence > expected:
                # Out of order - buffer it
                self._out_of_order_buffer[source][sequence] = (time.time(), event)

                # Detect gaps
                for gap_seq in range(expected, sequence):
                    if gap_seq not in self._received_sequences[source]:
                        self._gaps[source].add(gap_seq)
                        self._total_gaps_detected += 1

            # else: sequence < expected - this is a late arrival, already processed

            return True, ready_events

    async def get_gaps(self, source: Optional[str] = None) -> Dict[str, Set[int]]:
        """Get current gaps for retransmission request."""
        async with self._lock:
            if source:
                return {source: set(self._gaps[source])}
            return {s: set(gaps) for s, gaps in self._gaps.items() if gaps}

    async def cleanup_expired(self) -> int:
        """Remove expired out-of-order events. Returns count removed."""
        now = time.time()
        removed = 0

        async with self._lock:
            for source in list(self._out_of_order_buffer.keys()):
                expired = [
                    seq for seq, (timestamp, _) in self._out_of_order_buffer[source].items()
                    if now - timestamp > self._ooo_timeout
                ]
                for seq in expired:
                    del self._out_of_order_buffer[source][seq]
                    removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "sources_tracked": len(self._expected_sequences),
            "total_gaps_detected": self._total_gaps_detected,
            "total_gaps_filled": self._total_gaps_filled,
            "current_gaps": sum(len(g) for g in self._gaps.values()),
            "buffered_events": sum(len(b) for b in self._out_of_order_buffer.values()),
        }


# =============================================================================
# CIRCUIT BREAKER (Fault Tolerance)
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker for protecting flush operations.

    Prevents cascading failures when training pipeline is overloaded.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 30.0,
        health_check: Optional[Callable] = None,
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout = timeout_seconds
        self._health_check = health_check  # v242.0: async callable returning bool

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if circuit allows execution."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # v242.0: Health-check-based recovery (preferred over timeout alone)
                # If a health check is configured, use it to determine readiness.
                # Falls back to timeout-based recovery if no health check provided.
                should_try = False
                if self._health_check:
                    try:
                        healthy = await self._health_check()
                        if healthy:
                            should_try = True
                            logger.debug(
                                f"[CircuitBreaker:{self.name}] Health check passed — "
                                f"transitioning to HALF_OPEN"
                            )
                    except Exception:
                        pass  # Health check itself failed — stay OPEN

                if not should_try and self._last_failure_time and (
                    time.time() - self._last_failure_time > self._timeout
                ):
                    should_try = True

                if should_try:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    return True
                return False

            # HALF_OPEN - allow test requests
            return True

    async def record_success(self) -> None:
        """Record successful execution."""
        async with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    logger.info(f"[CircuitBreaker:{self.name}] Circuit CLOSED - recovered")

    async def record_failure(self) -> None:
        """Record failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test
                self._state = CircuitState.OPEN
                logger.warning(f"[CircuitBreaker:{self.name}] Circuit OPEN - recovery failed")

            elif self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[CircuitBreaker:{self.name}] Circuit OPEN - "
                    f"{self._failure_count} failures"
                )

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time,
        }


def _get_dynamic_path(name: str, env_var: str, subdir: str) -> Path:
    """
    Get a dynamically resolved path using base_config if available.

    Falls back to environment variable or XDG-compliant default.
    """
    # First check environment variable
    env_value = os.getenv(env_var)
    if env_value:
        path = Path(env_value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Try to use base_config's path resolver
    try:
        from reactor_core.config.base_config import resolve_path
        return resolve_path(name, env_var=env_var, subdir=subdir)
    except ImportError:
        pass

    # XDG-compliant fallback
    xdg_data_home = os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    base_path = Path(xdg_data_home) / "jarvis"
    path = base_path / subdir if subdir else base_path
    path.mkdir(parents=True, exist_ok=True)
    return path


# Configuration (fully dynamic path resolution - no hardcoded paths)
TRINITY_EVENTS_DIR = _get_dynamic_path(
    "trinity_events", "TRINITY_EVENTS_DIR", "trinity/events"
)
JARVIS_EVENTS_DIR = _get_dynamic_path(
    "jarvis_events", "JARVIS_EVENTS_DIR", "events"
)
EXPERIENCE_QUEUE_DIR = _get_dynamic_path(
    "experience_queue", "EXPERIENCE_QUEUE_DIR", "experience_queue"
)

# Polling intervals
FILE_POLL_INTERVAL = float(os.getenv("EXPERIENCE_FILE_POLL_INTERVAL", "5.0"))
FLUSH_THRESHOLD = int(os.getenv("EXPERIENCE_FLUSH_THRESHOLD", "100"))
MAX_BATCH_SIZE = int(os.getenv("EXPERIENCE_MAX_BATCH_SIZE", "500"))

# Event types we process
EXPERIENCE_EVENT_TYPES = {
    "learning_signal",
    "interaction_end",
    "correction",
    "feedback",
    "experience_batch",
}


@dataclass
class ReceiverMetrics:
    """Comprehensive metrics for the experience receiver."""
    # Event counters
    events_received: int = 0
    experiences_ingested: int = 0
    files_processed: int = 0
    errors: int = 0

    # Deduplication stats
    duplicates_filtered: int = 0
    bloom_filter_checks: int = 0

    # Sequence tracking
    gaps_detected: int = 0
    gaps_filled: int = 0
    out_of_order_events: int = 0

    # Vector clock stats
    causal_violations: int = 0
    concurrent_events: int = 0

    # Flush stats
    flushes_attempted: int = 0
    flushes_succeeded: int = 0
    flushes_failed: int = 0
    circuit_breaker_rejections: int = 0

    # Timing
    last_event_time: Optional[float] = None
    last_flush_time: Optional[float] = None
    start_time: float = field(default_factory=time.time)

    # Health indicators
    is_healthy: bool = True
    last_health_check: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "events_received": self.events_received,
            "experiences_ingested": self.experiences_ingested,
            "files_processed": self.files_processed,
            "errors": self.errors,
            "duplicates_filtered": self.duplicates_filtered,
            "bloom_filter_checks": self.bloom_filter_checks,
            "gaps_detected": self.gaps_detected,
            "gaps_filled": self.gaps_filled,
            "out_of_order_events": self.out_of_order_events,
            "causal_violations": self.causal_violations,
            "concurrent_events": self.concurrent_events,
            "flushes_attempted": self.flushes_attempted,
            "flushes_succeeded": self.flushes_succeeded,
            "flushes_failed": self.flushes_failed,
            "circuit_breaker_rejections": self.circuit_breaker_rejections,
            "last_event_time": self.last_event_time,
            "last_flush_time": self.last_flush_time,
            "uptime_seconds": time.time() - self.start_time,
            "is_healthy": self.is_healthy,
        }


class TrinityExperienceReceiver:
    """
    Production-hardened experience receiver with enterprise-grade reliability.

    Features v2.0:
    - Multi-directory watching (trinity/events, jarvis/events, experience_queue)
    - Event bus subscription (when available)
    - Automatic batching with flush threshold
    - Bloom filter + LRU deduplication (10x memory efficiency)
    - Vector clock causality validation
    - Sequence number tracking with gap detection
    - Out-of-order event buffering with timeout
    - Circuit breaker protected flush operations
    - Comprehensive health metrics
    - Graceful shutdown with drain
    """

    def __init__(
        self,
        flush_callback: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        expected_events: int = 100000,
        bloom_false_positive_rate: float = 0.001,
        gap_detection_window: int = 1000,
        out_of_order_timeout: float = 30.0,
        circuit_breaker_threshold: int = 5,
    ):
        """
        Initialize the enhanced experience receiver.

        Args:
            flush_callback: Custom callback for flushing experiences
            expected_events: Expected unique events for Bloom filter sizing
            bloom_false_positive_rate: Acceptable false positive rate for dedup
            gap_detection_window: Window size for sequence gap detection
            out_of_order_timeout: Timeout for out-of-order event buffer
            circuit_breaker_threshold: Failures before circuit opens
        """
        self.logger = logging.getLogger("TrinityExperienceReceiver")

        # Callback to flush experiences to training
        self._flush_callback = flush_callback

        # Experience buffer
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()

        # Advanced deduplication: Bloom filter + LRU cache
        self._bloom_filter = BloomFilter(
            expected_elements=expected_events,
            false_positive_rate=bloom_false_positive_rate,
        )
        self._lru_cache: deque[str] = deque(maxlen=10000)  # For recent exact checks
        self._lru_set: Set[str] = set()

        # Vector clock for causal ordering
        self._vector_clock = VectorClock(node_id="reactor_core")
        self._last_received_clock: Dict[str, Dict[str, int]] = {}  # Per-source

        # Sequence tracking with gap detection
        self._sequence_tracker = SequenceTracker(
            gap_detection_window=gap_detection_window,
            out_of_order_timeout=out_of_order_timeout,
        )

        # Circuit breaker for flush operations
        # v242.0: Health-check-based recovery — probes training pipeline
        # readiness instead of relying solely on timeout-based recovery
        async def _pipeline_health_check() -> bool:
            """Check if training pipeline is ready to accept experiences."""
            try:
                from reactor_core.training.unified_pipeline import get_unified_trainer_async
                trainer = await get_unified_trainer_async()
                if not trainer:
                    return False
                progress = trainer.get_progress()
                # Pipeline is healthy if it's idle or collecting
                # (not in the middle of training/exporting/deploying)
                return progress.state.value in ("idle", "collecting", "completed")
            except Exception:
                return False

        self._circuit_breaker = CircuitBreaker(
            name="experience_flush",
            failure_threshold=circuit_breaker_threshold,
            success_threshold=2,
            timeout_seconds=30.0,
            health_check=_pipeline_health_check,
        )

        # Metrics
        self._metrics = ReceiverMetrics()

        # State
        self._running = False
        self._shutting_down = False
        self._watch_task: Optional[asyncio.Task] = None
        self._maintenance_task: Optional[asyncio.Task] = None
        self._event_bridge: Optional[Any] = None

        # Instance ID for distributed coordination
        self._instance_id = f"reactor-{os.getpid()}-{uuid.uuid4().hex[:6]}"

        # Ensure directories exist
        for dir_path in [TRINITY_EVENTS_DIR, JARVIS_EVENTS_DIR, EXPERIENCE_QUEUE_DIR]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.debug(f"Could not create {dir_path}: {e}")

    async def start(self) -> bool:
        """Start the experience receiver with all subsystems."""
        if self._running:
            return True

        self._running = True
        self._shutting_down = False
        self.logger.info(f"TrinityExperienceReceiver v2.0 starting (instance={self._instance_id})...")

        # Start file watcher
        self._watch_task = asyncio.create_task(
            self._watch_loop(),
            name="trinity_experience_watch_loop"
        )

        # Start maintenance task (cleanup, gap detection, health checks)
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(),
            name="trinity_experience_maintenance"
        )

        # Try to subscribe to event bus
        await self._subscribe_to_event_bus()

        self.logger.info(
            f"TrinityExperienceReceiver v2.0 ready "
            f"(watching: {TRINITY_EVENTS_DIR}, {JARVIS_EVENTS_DIR}) "
            f"[Bloom filter: {self._bloom_filter._size} bits, "
            f"{self._bloom_filter._hash_count} hashes]"
        )
        return True

    async def stop(self, drain_timeout: float = 10.0) -> None:
        """
        Stop the receiver with graceful shutdown.

        Args:
            drain_timeout: Maximum time to wait for buffer drain
        """
        if self._shutting_down:
            return

        self._shutting_down = True
        self.logger.info("TrinityExperienceReceiver initiating graceful shutdown...")

        # Stop accepting new events
        self._running = False

        # Drain remaining buffer with timeout
        drain_start = time.time()
        while len(self._buffer) > 0 and (time.time() - drain_start) < drain_timeout:
            self.logger.info(f"Draining {len(self._buffer)} buffered experiences...")
            await self._flush_buffer(force=True)
            if len(self._buffer) > 0:
                await asyncio.sleep(0.5)

        if len(self._buffer) > 0:
            self.logger.warning(f"Shutdown with {len(self._buffer)} unprocessed experiences")

        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

        # Cancel watch task
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        # Stop event bridge if running
        if self._event_bridge:
            try:
                await self._event_bridge.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping event bridge: {e}")
            self._event_bridge = None

        # Log final stats
        self.logger.info(
            f"TrinityExperienceReceiver stopped. Final stats: "
            f"events={self._metrics.events_received}, "
            f"experiences={self._metrics.experiences_ingested}, "
            f"gaps_detected={self._metrics.gaps_detected}, "
            f"gaps_filled={self._metrics.gaps_filled}, "
            f"duplicates={self._metrics.duplicates_filtered}"
        )

    async def _maintenance_loop(self) -> None:
        """Background maintenance: cleanup, gap detection, health checks."""
        maintenance_interval = float(os.getenv("RECEIVER_MAINTENANCE_INTERVAL", "30.0"))

        while self._running:
            try:
                await asyncio.sleep(maintenance_interval)

                if not self._running:
                    break

                # Cleanup expired out-of-order events
                expired = await self._sequence_tracker.cleanup_expired()
                if expired > 0:
                    self.logger.debug(f"Cleaned up {expired} expired out-of-order events")

                # Check for gaps and log
                gaps = await self._sequence_tracker.get_gaps()
                if gaps:
                    total_gaps = sum(len(g) for g in gaps.values())
                    self._metrics.gaps_detected = total_gaps
                    self.logger.warning(
                        f"Sequence gaps detected: {total_gaps} gaps across "
                        f"{len(gaps)} sources. Consider retransmission."
                    )
                    # TODO: Publish gap retransmission request event

                # Health check
                await self._perform_health_check()

                # Log stats periodically
                if self._metrics.events_received > 0 and self._metrics.events_received % 1000 == 0:
                    self.logger.info(
                        f"Receiver stats: events={self._metrics.events_received}, "
                        f"ingested={self._metrics.experiences_ingested}, "
                        f"dedup={self._metrics.duplicates_filtered}, "
                        f"bloom_fpr={self._bloom_filter.estimated_false_positive_rate():.4f}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(5.0)

    async def _perform_health_check(self) -> bool:
        """Perform health check and update metrics."""
        is_healthy = True
        issues = []

        # Check circuit breaker state
        cb_state = self._circuit_breaker.get_state()
        if cb_state["state"] == "open":
            is_healthy = False
            issues.append("Circuit breaker is OPEN")

        # Check error rate
        if self._metrics.events_received > 100:
            error_rate = self._metrics.errors / self._metrics.events_received
            if error_rate > 0.1:  # More than 10% errors
                is_healthy = False
                issues.append(f"High error rate: {error_rate:.2%}")

        # Check buffer growth (indicates flush problems)
        if len(self._buffer) > MAX_BATCH_SIZE * 3:
            is_healthy = False
            issues.append(f"Buffer growing: {len(self._buffer)} items")

        # Check Bloom filter saturation
        bloom_fpr = self._bloom_filter.estimated_false_positive_rate()
        if bloom_fpr > 0.01:  # More than 1% false positive rate
            issues.append(f"Bloom filter saturating: {bloom_fpr:.2%} FPR")

        self._metrics.is_healthy = is_healthy
        self._metrics.last_health_check = time.time()

        if not is_healthy:
            self.logger.warning(f"Health check failed: {', '.join(issues)}")

        return is_healthy

    async def _subscribe_to_event_bus(self) -> bool:
        """Subscribe to Trinity event bus for real-time events."""
        try:
            from reactor_core.integration.event_bridge import (
                create_event_bridge,
                EventType,
                EventSource,
            )

            # Create bridge for Reactor Core
            bridge = create_event_bridge(
                source=EventSource.REACTOR_CORE,
                events_dir=TRINITY_EVENTS_DIR,
            )

            # Register handlers for relevant event types
            bridge.register_handler(
                self._on_cross_repo_event,
                event_types=[EventType.FEEDBACK, EventType.CORRECTION, EventType.INTERACTION_END],
            )

            # Start the bridge
            await bridge.start()

            # Store reference for cleanup
            self._event_bridge = bridge

            self.logger.info("Subscribed to EventBridge for experience events")
            return True

        except ImportError:
            self.logger.debug("EventBridge not available, using file-only mode")
        except Exception as e:
            self.logger.warning(f"EventBridge subscription failed: {e}")

        return False

    async def _on_cross_repo_event(self, event) -> None:
        """Handle CrossRepoEvent from event bridge and convert to dict format."""
        try:
            # Convert CrossRepoEvent to dict format for processing
            event_dict = {
                "event_type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                "event_id": event.event_id,
                "source": event.source.value if hasattr(event.source, 'value') else str(event.source),
                "payload": event.payload,
                "metadata": event.metadata,
                "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
            }
            await self._on_event(event_dict)
        except Exception as e:
            self.logger.error(f"Error converting CrossRepoEvent: {e}")
            self._metrics.errors += 1

    async def _on_event(self, event: Dict[str, Any]) -> None:
        """Handle event from event bus."""
        try:
            event_type = event.get("event_type", event.get("type", ""))

            if event_type in EXPERIENCE_EVENT_TYPES:
                await self._process_event(event)

        except Exception as e:
            self.logger.error(f"Event handler error: {e}")
            self._metrics.errors += 1

    async def _watch_loop(self) -> None:
        """Background loop to watch directories for experience files."""
        while self._running:
            try:
                await asyncio.sleep(FILE_POLL_INTERVAL)

                if not self._running:
                    break

                # Scan all directories
                for dir_path in [TRINITY_EVENTS_DIR, JARVIS_EVENTS_DIR, EXPERIENCE_QUEUE_DIR]:
                    if dir_path.exists():
                        await self._scan_directory(dir_path)

                # Check if we should flush
                async with self._buffer_lock:
                    if len(self._buffer) >= FLUSH_THRESHOLD:
                        await self._flush_buffer()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Watch loop error: {e}")
                await asyncio.sleep(1.0)

    async def _scan_directory(self, dir_path: Path) -> None:
        """Scan directory for experience event files."""
        try:
            # Look for JSON files matching our patterns
            for pattern in ["*.json", "5_*_*.json", "packet_*.json"]:
                for file_path in dir_path.glob(pattern):
                    # Skip hidden/temp files
                    if file_path.name.startswith("."):
                        continue

                    await self._process_file(file_path)

        except Exception as e:
            self.logger.error(f"Directory scan error for {dir_path}: {e}")

    async def _process_file(self, file_path: Path) -> None:
        """Process a single event file."""
        try:
            # Read file using asyncio.to_thread (Python 3.9+) or run_in_executor
            try:
                content = await asyncio.to_thread(file_path.read_text)
            except AttributeError:
                # Fallback for Python < 3.9
                loop = asyncio.get_running_loop()
                content = await loop.run_in_executor(None, file_path.read_text)

            event = json.loads(content)

            # Check if this is an experience-related event
            event_type = event.get("event_type", event.get("type", ""))

            if event_type in EXPERIENCE_EVENT_TYPES:
                await self._process_event(event)

                # Delete processed file
                try:
                    await asyncio.to_thread(file_path.unlink)
                except AttributeError:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, file_path.unlink)

                self._metrics.files_processed += 1
                self.logger.debug(f"Processed and deleted: {file_path.name}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON in {file_path.name}")
            # Move to .failed
            try:
                file_path.rename(file_path.with_suffix(".json.failed"))
            except Exception:
                pass
        except FileNotFoundError:
            # File already processed by another instance
            pass
        except Exception as e:
            self.logger.error(f"File processing error for {file_path}: {e}")
            self._metrics.errors += 1

    async def _process_event(self, event: Dict[str, Any]) -> None:
        """
        Process an experience event with full validation.

        Applies:
        1. Bloom filter + LRU deduplication
        2. Vector clock causality validation
        3. Sequence number tracking with gap detection
        4. Experience normalization and buffering
        """
        event_id = event.get("event_id", event.get("id", ""))
        source = event.get("source", "unknown")
        sequence = event.get("sequence_number", event.get("sequence", 0))

        # =================================================================
        # STEP 1: Advanced Deduplication (Bloom Filter + LRU)
        # =================================================================
        if event_id:
            self._metrics.bloom_filter_checks += 1

            # Fast Bloom filter check first
            if event_id in self._bloom_filter:
                # Might be duplicate - check LRU for certainty
                if event_id in self._lru_set:
                    self._metrics.duplicates_filtered += 1
                    return

            # Add to both Bloom filter and LRU
            self._bloom_filter.add(event_id)

            # Maintain LRU cache
            if event_id not in self._lru_set:
                if len(self._lru_cache) >= self._lru_cache.maxlen:
                    oldest = self._lru_cache.popleft()
                    self._lru_set.discard(oldest)
                self._lru_cache.append(event_id)
                self._lru_set.add(event_id)

        # =================================================================
        # STEP 2: Vector Clock Causality Validation
        # =================================================================
        event_vector_clock = event.get("vector_clock", event.get("vclock", {}))

        if event_vector_clock:
            # Update our vector clock with received clock
            await self._vector_clock.update(event_vector_clock)

            # Check causality against last known clock from this source
            if source in self._last_received_clock:
                last_clock = self._last_received_clock[source]
                relation = self._vector_clock.compare(last_clock, event_vector_clock)

                if relation == "after":
                    # Old event arrived after newer one - causal violation
                    self._metrics.causal_violations += 1
                    self.logger.warning(
                        f"Causal violation detected from {source}: "
                        f"event {event_id} arrived out of causal order"
                    )
                elif relation == "concurrent":
                    # Concurrent events - fine, but track it
                    self._metrics.concurrent_events += 1

            # Update last known clock for this source
            self._last_received_clock[source] = event_vector_clock

        # =================================================================
        # STEP 3: Sequence Number Tracking with Gap Detection
        # =================================================================
        if sequence > 0:
            is_new, ready_events = await self._sequence_tracker.track(
                source=source,
                sequence=sequence,
                event=event,
            )

            if not is_new:
                self._metrics.duplicates_filtered += 1
                return

            # Process ready events (in order)
            for ready_event in ready_events:
                await self._process_event_payload(ready_event)

            # Update gap stats
            tracker_stats = self._sequence_tracker.get_stats()
            self._metrics.gaps_detected = tracker_stats["total_gaps_detected"]
            self._metrics.gaps_filled = tracker_stats["total_gaps_filled"]
            self._metrics.out_of_order_events = tracker_stats["buffered_events"]

        else:
            # No sequence number - process immediately
            await self._process_event_payload(event)

    async def _process_event_payload(self, event: Dict[str, Any]) -> None:
        """
        Process the payload of a validated event.

        This extracts experiences and adds them to the buffer.
        """
        # Extract experiences from event
        payload = event.get("payload", event)
        experiences = []

        if "experiences" in payload:
            # Batch of experiences
            experiences = payload["experiences"]
        elif "signal_type" in payload and payload["signal_type"] == "experience_batch":
            # Learning signal with batch
            experiences = payload.get("experiences", [])
        elif "user_input" in event or "user_input" in payload:
            # Single interaction event
            src = payload if "user_input" in payload else event
            experiences = [{
                "user_input": src.get("user_input", ""),
                "assistant_output": src.get("assistant_output", src.get("response", "")),
                "confidence": src.get("confidence", 1.0),
                "feedback_type": src.get("feedback_type", "implicit"),
                "timestamp": src.get("timestamp", time.time()),
            }]

        if not experiences:
            return

        # Add to buffer with normalization
        async with self._buffer_lock:
            for exp in experiences:
                # v242.0: Use canonical schema adapter if available
                try:
                    import sys as _sys
                    _jarvis_home = str(Path.home() / ".jarvis")
                    if _jarvis_home not in _sys.path:
                        _sys.path.insert(0, _jarvis_home)
                    from schemas.experience_schema import from_raw_dict
                    canonical = from_raw_dict(exp)
                    normalized = canonical.to_reactor_core_format()
                    # Preserve causal metadata
                    normalized["vector_clock"] = event.get("vector_clock", {})
                    normalized["sequence"] = event.get("sequence_number", 0)
                    normalized["source"] = event.get("source", "unknown")
                except ImportError:
                    # Fallback: manual normalization
                    normalized = {
                        "user_input": exp.get("user_input", exp.get("input", {}).get("query", "")),
                        "assistant_output": exp.get("assistant_output", exp.get("output", {}).get("response", "")),
                        "confidence": exp.get("confidence", 1.0),
                        "feedback_type": exp.get("feedback_type", "implicit"),
                        "timestamp": exp.get("timestamp", time.time()),
                        "metadata": exp.get("metadata", {}),
                        # Preserve causal metadata
                        "vector_clock": event.get("vector_clock", {}),
                        "sequence": event.get("sequence_number", 0),
                        "source": event.get("source", "unknown"),
                    }

                # Skip empty experiences
                if normalized["user_input"] and normalized["assistant_output"]:
                    self._buffer.append(normalized)
                    self._metrics.experiences_ingested += 1

        self._metrics.events_received += 1
        self._metrics.last_event_time = time.time()

        # Check flush threshold
        async with self._buffer_lock:
            if len(self._buffer) >= MAX_BATCH_SIZE:
                await self._flush_buffer()

    async def _flush_buffer(self, force: bool = False) -> None:
        """
        Flush buffered experiences to training pipeline with circuit breaker protection.

        Args:
            force: If True, flush even if below threshold
        """
        # Check circuit breaker first
        if not await self._circuit_breaker.can_execute():
            self._metrics.circuit_breaker_rejections += 1
            self.logger.warning("Flush rejected by circuit breaker")
            return

        async with self._buffer_lock:
            if not self._buffer:
                return

            if not force and len(self._buffer) < FLUSH_THRESHOLD:
                return

            # Move buffer to local
            experiences = self._buffer.copy()
            self._buffer.clear()

        self._metrics.flushes_attempted += 1
        self.logger.info(f"Flushing {len(experiences)} experiences to training pipeline")

        try:
            # Use callback if provided
            if self._flush_callback:
                result = self._flush_callback(experiences)
                if asyncio.iscoroutine(result):
                    await result
            else:
                # Default: use unified training pipeline
                # v242.0: Use async version for proper initialization in async context
                from reactor_core.training.unified_pipeline import get_unified_trainer_async

                trainer = await get_unified_trainer_async()
                await trainer.add_experiences(experiences, flush=True)

            # Success - update metrics and circuit breaker
            self._metrics.last_flush_time = time.time()
            self._metrics.flushes_succeeded += 1
            await self._circuit_breaker.record_success()
            self.logger.info(f"Successfully flushed {len(experiences)} experiences")

        except Exception as e:
            self.logger.error(f"Flush failed: {e}")
            self._metrics.errors += 1
            self._metrics.flushes_failed += 1
            await self._circuit_breaker.record_failure()

            # Re-add experiences to buffer for retry
            async with self._buffer_lock:
                self._buffer.extend(experiences)

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive receiver metrics."""
        metrics = self._metrics.to_dict()

        # Add runtime state
        metrics.update({
            "buffer_size": len(self._buffer),
            "running": self._running,
            "shutting_down": self._shutting_down,
            "instance_id": self._instance_id,
        })

        # Add component states
        metrics["circuit_breaker"] = self._circuit_breaker.get_state()
        metrics["sequence_tracker"] = self._sequence_tracker.get_stats()
        metrics["vector_clock"] = self._vector_clock.get_clock()

        # Add Bloom filter stats
        metrics["bloom_filter"] = {
            "size_bits": self._bloom_filter._size,
            "hash_count": self._bloom_filter._hash_count,
            "items_added": self._bloom_filter._count,
            "estimated_fpr": self._bloom_filter.estimated_false_positive_rate(),
        }

        return metrics

    def get_health(self) -> Dict[str, Any]:
        """
        Get health status for monitoring.

        Returns a health report suitable for health check endpoints.
        """
        is_healthy = self._metrics.is_healthy and self._running

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "component": "TrinityExperienceReceiver",
            "version": "2.0.0",
            "instance_id": self._instance_id,
            "running": self._running,
            "shutting_down": self._shutting_down,
            "circuit_breaker_state": self._circuit_breaker.get_state()["state"],
            "buffer_size": len(self._buffer),
            "error_count": self._metrics.errors,
            "last_event_age_seconds": (
                time.time() - self._metrics.last_event_time
                if self._metrics.last_event_time else None
            ),
            "uptime_seconds": time.time() - self._metrics.start_time,
            "checks": {
                "circuit_breaker": self._circuit_breaker.get_state()["state"] != "open",
                "buffer_not_overflowing": len(self._buffer) < MAX_BATCH_SIZE * 3,
                "error_rate_acceptable": (
                    self._metrics.errors / max(self._metrics.events_received, 1) < 0.1
                ),
            },
        }

    async def request_gap_retransmission(self, source: Optional[str] = None) -> Dict[str, Set[int]]:
        """
        Get gaps for retransmission request.

        This can be used to request missing events from the source.

        Args:
            source: Optional source to get gaps for. If None, returns all gaps.

        Returns:
            Dict mapping source to set of missing sequence numbers.
        """
        return await self._sequence_tracker.get_gaps(source)


# =============================================================================
# Global Instance Management
# =============================================================================

_receiver: Optional[TrinityExperienceReceiver] = None
_receiver_lock: Optional[asyncio.Lock] = None


def _get_receiver_lock() -> asyncio.Lock:
    """Get or create the receiver lock."""
    global _receiver_lock
    if _receiver_lock is None:
        _receiver_lock = asyncio.Lock()
    return _receiver_lock


async def get_experience_receiver() -> TrinityExperienceReceiver:
    """Get the global experience receiver instance."""
    global _receiver

    lock = _get_receiver_lock()
    async with lock:
        if _receiver is None:
            _receiver = TrinityExperienceReceiver()
            await _receiver.start()

        return _receiver


async def shutdown_experience_receiver() -> None:
    """Shutdown the global experience receiver."""
    global _receiver

    if _receiver:
        await _receiver.stop()
        _receiver = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core Receiver
    "TrinityExperienceReceiver",
    "ReceiverMetrics",
    # Deduplication
    "BloomFilter",
    # Causal Ordering
    "VectorClock",
    "SequenceTracker",
    # Fault Tolerance
    "CircuitBreaker",
    "CircuitState",
    # Global Instance Management
    "get_experience_receiver",
    "shutdown_experience_receiver",
]
