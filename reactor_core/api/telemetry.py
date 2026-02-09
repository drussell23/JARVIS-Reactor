"""
Advanced Telemetry Ingestion System for Reactor-Core.

Provides real-time streaming, metric aggregation, and event buffering
for the AGI OS ecosystem. This is the central nervous system's
sensory input layer.

Architecture:
    JARVIS → TelemetryCollector → EventBuffer → MetricsAggregator
                                              ↓
                                        WebSocket Stream → Dashboard
                                              ↓
                                        TimeSeries Store → Analytics

Features:
- Real-time WebSocket streaming
- Ring buffer for high-throughput ingestion
- Automatic metric aggregation (counters, gauges, histograms)
- Adaptive sampling for high-cardinality metrics
- Cross-repo event correlation
- Anomaly detection on streaming data
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import statistics
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Configuration
# ============================================================================

class TelemetryConfig:
    """Central telemetry configuration."""

    # Buffer settings
    RING_BUFFER_SIZE = int(os.getenv("TELEMETRY_BUFFER_SIZE", "10000"))
    FLUSH_INTERVAL_SECONDS = float(os.getenv("TELEMETRY_FLUSH_INTERVAL", "5.0"))
    BATCH_SIZE = int(os.getenv("TELEMETRY_BATCH_SIZE", "100"))

    # WebSocket settings
    WS_HEARTBEAT_INTERVAL = float(os.getenv("TELEMETRY_WS_HEARTBEAT", "30.0"))
    WS_MAX_CONNECTIONS = int(os.getenv("TELEMETRY_WS_MAX_CONN", "100"))

    # Sampling settings
    ADAPTIVE_SAMPLING_ENABLED = os.getenv("TELEMETRY_ADAPTIVE_SAMPLE", "true").lower() == "true"
    SAMPLING_THRESHOLD_PER_SECOND = int(os.getenv("TELEMETRY_SAMPLE_THRESHOLD", "1000"))
    MIN_SAMPLE_RATE = float(os.getenv("TELEMETRY_MIN_SAMPLE_RATE", "0.01"))

    # Anomaly detection
    ANOMALY_DETECTION_ENABLED = os.getenv("TELEMETRY_ANOMALY_DETECT", "true").lower() == "true"
    ANOMALY_ZSCORE_THRESHOLD = float(os.getenv("TELEMETRY_ANOMALY_ZSCORE", "3.0"))

    # Retention
    METRIC_RETENTION_HOURS = int(os.getenv("TELEMETRY_RETENTION_HOURS", "24"))

    # Storage
    STORAGE_PATH = Path(os.getenv("TELEMETRY_STORAGE_PATH", str(Path.home() / ".reactor_core" / "telemetry")))


# ============================================================================
# Data Models
# ============================================================================

class EventType(Enum):
    """Telemetry event types."""
    METRIC = auto()
    LOG = auto()
    TRACE = auto()
    SPAN = auto()
    HEARTBEAT = auto()
    ALERT = auto()
    CUSTOM = auto()


class MetricType(Enum):
    """Metric data types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class TelemetryEvent:
    """Base telemetry event structure."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    event_type: EventType = EventType.CUSTOM
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "source": self.source,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data,
            "labels": self.labels,
            "correlation_id": self.correlation_id,
        }


@dataclass
class MetricValue:
    """Single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    @property
    def label_hash(self) -> str:
        """Generate hash of labels for grouping."""
        sorted_labels = sorted(self.labels.items())
        return hashlib.md5(json.dumps(sorted_labels).encode()).hexdigest()[:8]


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time window."""
    name: str
    metric_type: MetricType
    labels: Dict[str, str]
    window_start: float
    window_end: float
    count: int = 0
    sum: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    values: List[float] = field(default_factory=list)  # For histogram/percentiles

    def add(self, value: float):
        """Add a value to the aggregation."""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        if self.metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
            self.values.append(value)

    @property
    def avg(self) -> float:
        """Calculate average."""
        return self.sum / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.metric_type.value,
            "labels": self.labels,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "count": self.count,
            "sum": self.sum,
            "avg": self.avg,
            "min": self.min if self.count > 0 else 0,
            "max": self.max if self.count > 0 else 0,
        }
        if self.metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
            result.update({
                "p50": self.percentile(50),
                "p90": self.percentile(90),
                "p95": self.percentile(95),
                "p99": self.percentile(99),
            })
        return result


@dataclass
class AnomalyAlert:
    """Anomaly detection alert."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metric_name: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    current_value: float = 0.0
    expected_value: float = 0.0
    zscore: float = 0.0
    severity: str = "warning"  # warning, critical
    timestamp: float = field(default_factory=time.time)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Ring Buffer for High-Throughput Ingestion
# ============================================================================

class RingBuffer(Generic[T]):
    """
    Lock-free ring buffer for high-throughput event ingestion.

    Uses a pre-allocated circular buffer with atomic-like operations
    to minimize allocation overhead during ingestion.
    """

    def __init__(self, capacity: int = TelemetryConfig.RING_BUFFER_SIZE):
        self._capacity = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._write_idx = 0
        self._read_idx = 0
        self._count = 0
        self._overflow_count = 0
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Current number of items in buffer."""
        return self._count

    @property
    def capacity(self) -> int:
        """Total buffer capacity."""
        return self._capacity

    @property
    def overflow_count(self) -> int:
        """Number of items dropped due to overflow."""
        return self._overflow_count

    async def push(self, item: T) -> bool:
        """
        Push item to buffer.

        Returns True if successful, False if buffer is full (oldest item dropped).
        """
        async with self._lock:
            self._buffer[self._write_idx] = item
            self._write_idx = (self._write_idx + 1) % self._capacity

            if self._count < self._capacity:
                self._count += 1
                return True
            else:
                # Buffer full - oldest item will be overwritten
                self._read_idx = (self._read_idx + 1) % self._capacity
                self._overflow_count += 1
                return False

    async def pop(self) -> Optional[T]:
        """Pop item from buffer."""
        async with self._lock:
            if self._count == 0:
                return None

            item = self._buffer[self._read_idx]
            self._buffer[self._read_idx] = None
            self._read_idx = (self._read_idx + 1) % self._capacity
            self._count -= 1
            return item

    async def pop_batch(self, max_items: int) -> List[T]:
        """Pop up to max_items from buffer."""
        async with self._lock:
            items = []
            while self._count > 0 and len(items) < max_items:
                item = self._buffer[self._read_idx]
                if item is not None:
                    items.append(item)
                self._buffer[self._read_idx] = None
                self._read_idx = (self._read_idx + 1) % self._capacity
                self._count -= 1
            return items

    async def peek(self) -> Optional[T]:
        """Peek at next item without removing it."""
        async with self._lock:
            if self._count == 0:
                return None
            return self._buffer[self._read_idx]

    async def clear(self) -> int:
        """Clear buffer and return number of items cleared."""
        async with self._lock:
            count = self._count
            self._buffer = [None] * self._capacity
            self._write_idx = 0
            self._read_idx = 0
            self._count = 0
            return count

    def stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": self._count,
            "capacity": self._capacity,
            "utilization": self._count / self._capacity if self._capacity > 0 else 0,
            "overflow_count": self._overflow_count,
        }


# ============================================================================
# Adaptive Sampler for High-Cardinality Metrics
# ============================================================================

class AdaptiveSampler:
    """
    Adaptive sampling for high-cardinality metrics.

    Automatically adjusts sample rate based on throughput to prevent
    system overload while maintaining statistical accuracy.
    """

    def __init__(
        self,
        threshold_per_second: int = TelemetryConfig.SAMPLING_THRESHOLD_PER_SECOND,
        min_sample_rate: float = TelemetryConfig.MIN_SAMPLE_RATE,
        window_seconds: float = 10.0,
    ):
        self._threshold = threshold_per_second
        self._min_rate = min_sample_rate
        self._window = window_seconds

        # Per-metric tracking
        self._counts: Dict[str, int] = defaultdict(int)
        self._sample_rates: Dict[str, float] = defaultdict(lambda: 1.0)
        self._window_start = time.time()
        self._lock = asyncio.Lock()

    async def should_sample(self, metric_name: str) -> Tuple[bool, float]:
        """
        Determine if a metric should be sampled.

        Returns:
            (should_sample, current_sample_rate)
        """
        async with self._lock:
            current_time = time.time()

            # Check if window expired
            if current_time - self._window_start > self._window:
                await self._recalculate_rates()
                self._window_start = current_time

            # Increment count
            self._counts[metric_name] += 1

            # Apply sampling
            rate = self._sample_rates[metric_name]
            if rate >= 1.0:
                return True, rate

            # Probabilistic sampling
            import random
            return random.random() < rate, rate

    async def _recalculate_rates(self):
        """Recalculate sample rates based on observed throughput."""
        for metric_name, count in self._counts.items():
            rate_per_second = count / self._window

            if rate_per_second > self._threshold:
                # Calculate new sample rate
                new_rate = self._threshold / rate_per_second
                new_rate = max(new_rate, self._min_rate)
                self._sample_rates[metric_name] = new_rate

                logger.debug(
                    f"[Sampler] Adjusted {metric_name}: "
                    f"rate={rate_per_second:.1f}/s, sample_rate={new_rate:.3f}"
                )
            else:
                # No sampling needed
                self._sample_rates[metric_name] = 1.0

        # Reset counts
        self._counts.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "enabled": TelemetryConfig.ADAPTIVE_SAMPLING_ENABLED,
            "threshold_per_second": self._threshold,
            "min_sample_rate": self._min_rate,
            "active_metrics": len(self._sample_rates),
            "sample_rates": dict(self._sample_rates),
        }


# ============================================================================
# Metrics Aggregator
# ============================================================================

class MetricsAggregator:
    """
    Time-windowed metrics aggregation.

    Aggregates raw metrics into time buckets for efficient storage
    and querying. Supports multiple aggregation windows.
    """

    def __init__(
        self,
        window_sizes: List[int] = None,  # seconds
    ):
        self._window_sizes = window_sizes or [60, 300, 3600]  # 1m, 5m, 1h
        self._aggregations: Dict[int, Dict[str, AggregatedMetric]] = {
            window: {} for window in self._window_sizes
        }
        self._current_windows: Dict[int, float] = {}
        self._lock = asyncio.Lock()

        # Initialize windows
        now = time.time()
        for window in self._window_sizes:
            self._current_windows[window] = (now // window) * window

    async def add_metric(self, metric: MetricValue):
        """Add a metric value to aggregation."""
        async with self._lock:
            now = time.time()

            for window_size in self._window_sizes:
                window_start = (now // window_size) * window_size

                # Check if we need to roll over to new window
                if window_start > self._current_windows.get(window_size, 0):
                    # Archive old window if needed
                    self._current_windows[window_size] = window_start
                    # Clear old aggregations for this window
                    self._aggregations[window_size].clear()

                # Create metric key
                key = f"{metric.name}:{metric.label_hash}"

                # Get or create aggregation
                if key not in self._aggregations[window_size]:
                    self._aggregations[window_size][key] = AggregatedMetric(
                        name=metric.name,
                        metric_type=metric.metric_type,
                        labels=metric.labels,
                        window_start=window_start,
                        window_end=window_start + window_size,
                    )

                # Add value
                self._aggregations[window_size][key].add(metric.value)

    async def get_metrics(
        self,
        window_size: int = 60,
        metric_names: Optional[List[str]] = None,
    ) -> List[AggregatedMetric]:
        """Get aggregated metrics for a window."""
        async with self._lock:
            if window_size not in self._aggregations:
                return []

            metrics = list(self._aggregations[window_size].values())

            if metric_names:
                metrics = [m for m in metrics if m.name in metric_names]

            return metrics

    async def get_latest(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[AggregatedMetric]:
        """Get latest aggregation for a specific metric."""
        async with self._lock:
            # Check smallest window first
            for window_size in sorted(self._window_sizes):
                for agg in self._aggregations[window_size].values():
                    if agg.name == metric_name:
                        if labels is None or agg.labels == labels:
                            return agg
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "window_sizes": self._window_sizes,
            "metrics_per_window": {
                w: len(self._aggregations[w]) for w in self._window_sizes
            },
        }


# ============================================================================
# Anomaly Detector
# ============================================================================

class AnomalyDetector:
    """
    Real-time anomaly detection on streaming metrics.

    Uses Z-score based detection with sliding windows.
    """

    def __init__(
        self,
        zscore_threshold: float = TelemetryConfig.ANOMALY_ZSCORE_THRESHOLD,
        window_size: int = 100,  # Number of samples
        min_samples: int = 20,   # Minimum samples before detection
    ):
        self._threshold = zscore_threshold
        self._window_size = window_size
        self._min_samples = min_samples

        # Per-metric history
        self._history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._alerts: Deque[AnomalyAlert] = deque(maxlen=1000)
        self._lock = asyncio.Lock()

        # Alert callbacks
        self._callbacks: List[Callable[[AnomalyAlert], Awaitable[None]]] = []

    def register_callback(self, callback: Callable[[AnomalyAlert], Awaitable[None]]):
        """Register callback for anomaly alerts."""
        self._callbacks.append(callback)

    async def check(self, metric: MetricValue) -> Optional[AnomalyAlert]:
        """
        Check metric value for anomalies.

        Returns alert if anomaly detected, None otherwise.
        """
        async with self._lock:
            key = f"{metric.name}:{metric.label_hash}"
            history = self._history[key]

            # Add to history
            history.append(metric.value)

            # Need minimum samples
            if len(history) < self._min_samples:
                return None

            # Calculate statistics
            mean = statistics.mean(history)
            stdev = statistics.stdev(history) if len(history) > 1 else 0

            if stdev == 0:
                return None

            # Calculate z-score
            zscore = (metric.value - mean) / stdev

            if abs(zscore) > self._threshold:
                severity = "critical" if abs(zscore) > self._threshold * 1.5 else "warning"

                alert = AnomalyAlert(
                    metric_name=metric.name,
                    labels=metric.labels,
                    current_value=metric.value,
                    expected_value=mean,
                    zscore=zscore,
                    severity=severity,
                    message=f"Metric {metric.name} is {abs(zscore):.1f} standard deviations from mean",
                )

                self._alerts.append(alert)

                # Fire callbacks
                for callback in self._callbacks:
                    try:
                        await callback(alert)
                    except Exception as e:
                        logger.error(f"[AnomalyDetector] Callback error: {e}")

                logger.warning(
                    f"[Anomaly] {metric.name}: value={metric.value:.2f}, "
                    f"expected={mean:.2f}, zscore={zscore:.2f}"
                )

                return alert

            return None

    async def get_alerts(
        self,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[AnomalyAlert]:
        """Get recent anomaly alerts."""
        async with self._lock:
            alerts = list(self._alerts)
            if since:
                alerts = [a for a in alerts if a.timestamp > since]
            return alerts[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "enabled": TelemetryConfig.ANOMALY_DETECTION_ENABLED,
            "zscore_threshold": self._threshold,
            "tracked_metrics": len(self._history),
            "total_alerts": len(self._alerts),
        }


# ============================================================================
# WebSocket Broadcaster
# ============================================================================

class WebSocketBroadcaster:
    """
    Real-time WebSocket broadcasting for telemetry events.

    Supports multiple subscribers with topic filtering.
    """

    def __init__(self, max_connections: int = TelemetryConfig.WS_MAX_CONNECTIONS):
        self._max_connections = max_connections
        self._connections: Dict[str, weakref.ref] = {}
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> connection_ids
        self._lock = asyncio.Lock()
        self._message_queue: Deque[Tuple[str, Dict[str, Any]]] = deque(maxlen=1000)

    async def register(self, connection_id: str, websocket: Any, topics: Optional[List[str]] = None):
        """Register a WebSocket connection."""
        async with self._lock:
            if len(self._connections) >= self._max_connections:
                raise RuntimeError(f"Max connections ({self._max_connections}) reached")

            self._connections[connection_id] = weakref.ref(websocket)

            # Subscribe to topics
            for topic in (topics or ["all"]):
                self._subscriptions[topic].add(connection_id)

            logger.info(f"[WS] Registered connection: {connection_id}, topics={topics or ['all']}")

    async def unregister(self, connection_id: str):
        """Unregister a WebSocket connection."""
        async with self._lock:
            self._connections.pop(connection_id, None)

            # Remove from all subscriptions
            for topic in list(self._subscriptions.keys()):
                self._subscriptions[topic].discard(connection_id)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]

            logger.info(f"[WS] Unregistered connection: {connection_id}")

    async def broadcast(self, topic: str, data: Dict[str, Any]):
        """Broadcast message to all subscribers of a topic."""
        async with self._lock:
            # Get subscribers for topic and "all"
            connection_ids = self._subscriptions.get(topic, set()) | self._subscriptions.get("all", set())

            if not connection_ids:
                return

            message = json.dumps({"topic": topic, "data": data, "timestamp": time.time()})

            dead_connections = []
            for conn_id in connection_ids:
                ws_ref = self._connections.get(conn_id)
                if ws_ref is None:
                    dead_connections.append(conn_id)
                    continue

                ws = ws_ref()
                if ws is None:
                    dead_connections.append(conn_id)
                    continue

                try:
                    await ws.send_text(message)
                except Exception:
                    dead_connections.append(conn_id)

            # Clean up dead connections
            for conn_id in dead_connections:
                await self.unregister(conn_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            "active_connections": len(self._connections),
            "max_connections": self._max_connections,
            "topics": list(self._subscriptions.keys()),
            "subscribers_per_topic": {t: len(s) for t, s in self._subscriptions.items()},
        }


# ============================================================================
# Telemetry Collector (Main Class)
# ============================================================================

class TelemetryCollector:
    """
    Central telemetry collection and processing system.

    Coordinates all telemetry components:
    - RingBuffer for high-throughput ingestion
    - AdaptiveSampler for rate control
    - MetricsAggregator for time-windowed aggregation
    - AnomalyDetector for real-time detection
    - WebSocketBroadcaster for streaming

    Usage:
        collector = TelemetryCollector()
        await collector.start()

        # Ingest metrics
        await collector.ingest_metric("request_latency", 42.5, MetricType.HISTOGRAM)

        # Ingest events
        await collector.ingest_event(TelemetryEvent(
            event_type=EventType.LOG,
            source="jarvis",
            data={"level": "info", "message": "Hello"},
        ))

        # Query aggregations
        metrics = await collector.get_aggregated_metrics(window_size=60)
    """

    def __init__(self):
        self._buffer = RingBuffer[TelemetryEvent]()
        self._sampler = AdaptiveSampler()
        self._aggregator = MetricsAggregator()
        self._anomaly_detector = AnomalyDetector()
        self._broadcaster = WebSocketBroadcaster()

        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self._ingested_count = 0
        self._sampled_count = 0
        self._processed_count = 0
        self._start_time = time.time()

        # Event handlers
        self._event_handlers: Dict[EventType, List[Callable[[TelemetryEvent], Awaitable[None]]]] = defaultdict(list)

    async def start(self):
        """Start the telemetry collector."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        # Start background processors
        self._processor_task = asyncio.create_task(self._process_loop())
        self._flush_task = asyncio.create_task(self._flush_loop())

        logger.info("[Telemetry] Collector started")

    async def flush_pending(self):
        """Drain all remaining events from the buffer and process them.

        Call this BEFORE stop() during shutdown to avoid data loss.
        The ring buffer is in-memory only — anything not processed is lost
        when the process exits.
        """
        flushed = 0
        while True:
            events = await self._buffer.pop_batch(TelemetryConfig.BATCH_SIZE)
            if not events:
                break
            for event in events:
                self._processed_count += 1
                flushed += 1
                # Broadcast event (handlers may persist to disk/DB)
                topic = f"events:{event.event_type.name.lower()}"
                try:
                    await self._broadcaster.broadcast(topic, event.to_dict())
                except Exception as e:
                    logger.debug(f"[Telemetry] Flush broadcast failed: {e}")

                # Call registered handlers
                handlers = self._event_handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.debug(f"[Telemetry] Flush handler error: {e}")

        # Final system metrics broadcast
        try:
            await self._broadcast_system_metrics()
        except Exception:
            pass

        if flushed > 0:
            logger.info(f"[Telemetry] Flushed {flushed} pending events on shutdown")

    async def stop(self):
        """Stop the telemetry collector."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        logger.info("[Telemetry] Collector stopped")

    def register_handler(self, event_type: EventType, handler: Callable[[TelemetryEvent], Awaitable[None]]):
        """Register event handler."""
        self._event_handlers[event_type].append(handler)

    async def ingest_event(self, event: TelemetryEvent) -> bool:
        """
        Ingest a telemetry event.

        Returns True if event was accepted, False if dropped (sampling).
        """
        self._ingested_count += 1

        # Apply sampling if enabled
        if TelemetryConfig.ADAPTIVE_SAMPLING_ENABLED:
            should_sample, rate = await self._sampler.should_sample(f"event:{event.source}")
            if not should_sample:
                return False

        self._sampled_count += 1

        # Add to buffer
        await self._buffer.push(event)
        return True

    async def ingest_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "",
        source: str = "api",
    ) -> bool:
        """
        Ingest a metric value.

        Returns True if metric was accepted.
        """
        self._ingested_count += 1

        # Apply sampling
        if TelemetryConfig.ADAPTIVE_SAMPLING_ENABLED:
            should_sample, rate = await self._sampler.should_sample(name)
            if not should_sample:
                return False

        self._sampled_count += 1

        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            unit=unit,
        )

        # Add to aggregator
        await self._aggregator.add_metric(metric)

        # Check for anomalies
        if TelemetryConfig.ANOMALY_DETECTION_ENABLED:
            alert = await self._anomaly_detector.check(metric)
            if alert:
                # Broadcast anomaly alert
                await self._broadcaster.broadcast("alerts", alert.to_dict())

        # Broadcast metric
        await self._broadcaster.broadcast("metrics", {
            "name": name,
            "value": value,
            "type": metric_type.value,
            "labels": labels or {},
            "timestamp": time.time(),
        })

        return True

    async def ingest_batch(self, events: List[TelemetryEvent]) -> Tuple[int, int]:
        """
        Ingest batch of events.

        Returns (accepted_count, dropped_count).
        """
        accepted = 0
        dropped = 0

        for event in events:
            if await self.ingest_event(event):
                accepted += 1
            else:
                dropped += 1

        return accepted, dropped

    async def _process_loop(self):
        """Background event processing loop."""
        while self._running:
            try:
                # Process batch from buffer
                events = await self._buffer.pop_batch(TelemetryConfig.BATCH_SIZE)

                for event in events:
                    self._processed_count += 1

                    # Broadcast event
                    topic = f"events:{event.event_type.name.lower()}"
                    await self._broadcaster.broadcast(topic, event.to_dict())

                    # Call handlers
                    handlers = self._event_handlers.get(event.event_type, [])
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"[Telemetry] Handler error: {e}")

                if not events:
                    await asyncio.sleep(0.01)  # Small sleep when buffer empty

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Telemetry] Process loop error: {e}")
                await asyncio.sleep(0.1)

    async def _flush_loop(self):
        """Periodic flush for metrics."""
        while self._running:
            try:
                await asyncio.sleep(TelemetryConfig.FLUSH_INTERVAL_SECONDS)

                # Broadcast system metrics
                await self._broadcast_system_metrics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Telemetry] Flush loop error: {e}")

    async def _broadcast_system_metrics(self):
        """Broadcast telemetry system metrics."""
        stats = self.get_stats()
        await self._broadcaster.broadcast("system", stats)

    async def get_aggregated_metrics(
        self,
        window_size: int = 60,
        metric_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get aggregated metrics."""
        metrics = await self._aggregator.get_metrics(window_size, metric_names)
        return [m.to_dict() for m in metrics]

    async def get_alerts(self, since: Optional[float] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent anomaly alerts."""
        alerts = await self._anomaly_detector.get_alerts(since, limit)
        return [a.to_dict() for a in alerts]

    def register_anomaly_callback(self, callback: Callable[[AnomalyAlert], Awaitable[None]]):
        """Register callback for anomaly alerts."""
        self._anomaly_detector.register_callback(callback)

    async def register_websocket(
        self,
        connection_id: str,
        websocket: Any,
        topics: Optional[List[str]] = None,
    ):
        """Register WebSocket for streaming."""
        await self._broadcaster.register(connection_id, websocket, topics)

    async def unregister_websocket(self, connection_id: str):
        """Unregister WebSocket."""
        await self._broadcaster.unregister(connection_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive telemetry statistics."""
        uptime = time.time() - self._start_time
        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "ingested_total": self._ingested_count,
            "sampled_total": self._sampled_count,
            "processed_total": self._processed_count,
            "sampling_rate": self._sampled_count / self._ingested_count if self._ingested_count > 0 else 1.0,
            "events_per_second": self._ingested_count / uptime if uptime > 0 else 0,
            "buffer": self._buffer.stats(),
            "sampler": self._sampler.get_stats(),
            "aggregator": self._aggregator.get_stats(),
            "anomaly_detector": self._anomaly_detector.get_stats(),
            "broadcaster": self._broadcaster.get_stats(),
        }


# ============================================================================
# Cross-Repo Event Correlator
# ============================================================================

class EventCorrelator:
    """
    Correlate events across JARVIS, Prime, and Reactor-Core.

    Uses correlation IDs to track requests/events across the
    AGI OS ecosystem.
    """

    def __init__(self, max_correlations: int = 10000, ttl_seconds: float = 3600):
        self._max_correlations = max_correlations
        self._ttl = ttl_seconds
        self._correlations: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate new correlation ID."""
        return f"corr-{uuid.uuid4().hex[:16]}"

    async def track(
        self,
        correlation_id: str,
        source: str,
        event_type: str,
        data: Dict[str, Any],
    ):
        """Track an event in a correlation chain."""
        async with self._lock:
            now = time.time()

            if correlation_id not in self._correlations:
                if len(self._correlations) >= self._max_correlations:
                    # Evict oldest
                    oldest = min(self._correlations.items(), key=lambda x: x[1]["created_at"])
                    del self._correlations[oldest[0]]

                self._correlations[correlation_id] = {
                    "correlation_id": correlation_id,
                    "created_at": now,
                    "events": [],
                }

            self._correlations[correlation_id]["events"].append({
                "source": source,
                "event_type": event_type,
                "timestamp": now,
                "data": data,
            })
            self._correlations[correlation_id]["last_updated"] = now

    async def get_chain(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get full correlation chain."""
        async with self._lock:
            return self._correlations.get(correlation_id)

    async def cleanup_expired(self):
        """Remove expired correlations."""
        async with self._lock:
            now = time.time()
            expired = [
                cid for cid, data in self._correlations.items()
                if now - data.get("last_updated", data["created_at"]) > self._ttl
            ]
            for cid in expired:
                del self._correlations[cid]
            return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get correlator statistics."""
        return {
            "active_correlations": len(self._correlations),
            "max_correlations": self._max_correlations,
            "ttl_seconds": self._ttl,
        }


# ============================================================================
# Global Telemetry Instance
# ============================================================================

_telemetry_collector: Optional[TelemetryCollector] = None
_event_correlator: Optional[EventCorrelator] = None


def get_telemetry() -> TelemetryCollector:
    """Get global telemetry collector instance."""
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector


def get_correlator() -> EventCorrelator:
    """Get global event correlator instance."""
    global _event_correlator
    if _event_correlator is None:
        _event_correlator = EventCorrelator()
    return _event_correlator


@asynccontextmanager
async def telemetry_context():
    """Context manager for telemetry lifecycle."""
    collector = get_telemetry()
    await collector.start()
    try:
        yield collector
    finally:
        await collector.stop()
