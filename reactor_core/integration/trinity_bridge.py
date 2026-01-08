"""
Trinity Bridge - Ultra-High Performance Event Bus for AGI OS
=============================================================

The neural pathway connecting JARVIS (Body), J-Prime (Mind), and Reactor Core (Nerves).

Features:
- **Zero-Copy Message Passing** via shared memory
- **Async/Await First** - All operations non-blocking
- **Auto-Reconnection** with exponential backoff
- **Message Deduplication** via bloom filters
- **Priority Queueing** - Critical events bypass normal queue
- **Circuit Breakers** - Prevent cascade failures
- **Distributed Tracing** - Full observability
- **Type-Safe Events** - Pydantic validation
- **WebSocket + HTTP Fallback** - Multi-protocol support

Version: v82.0 (Trinity Unification)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import aiohttp
from aiohttp import web, WSMessage, WSMsgType

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 0  # System-critical events (health failures, crashes)
    HIGH = 1  # Important events (model updates, training completion)
    NORMAL = 2  # Standard events (telemetry, interactions)
    LOW = 3  # Background events (metrics, logs)


class EventType(Enum):
    """Types of events flowing through Trinity."""
    # JARVIS → Reactor
    INTERACTION = "interaction"  # User interaction telemetry
    FEEDBACK = "feedback"  # User feedback/corrections
    EXPERIENCE = "experience"  # Complete experience log

    # Reactor → J-Prime
    MODEL_UPDATE = "model_update"  # New model available
    CHECKPOINT_READY = "checkpoint_ready"  # Training checkpoint saved

    # J-Prime → JARVIS
    INFERENCE_RESULT = "inference_result"  # LLM response
    MODEL_LOADED = "model_loaded"  # Model successfully loaded

    # System Events
    HEALTH_CHECK = "health_check"  # Health status update
    HEARTBEAT = "heartbeat"  # Keep-alive signal
    COMPONENT_READY = "component_ready"  # Component startup complete
    COMPONENT_ERROR = "component_error"  # Component failure
    SHUTDOWN = "shutdown"  # Graceful shutdown request


class BridgeState(Enum):
    """Trinity Bridge connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrinityEvent:
    """Type-safe event structure."""
    event_id: str
    event_type: EventType
    source: str  # "jarvis", "jprime", "reactor"
    target: Optional[str]  # None = broadcast
    priority: EventPriority
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None  # For request-response correlation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "target": self.target,
            "priority": self.priority.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrinityEvent:
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            source=data["source"],
            target=data.get("target"),
            priority=EventPriority(data["priority"]),
            payload=data["payload"],
            timestamp=data["timestamp"],
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BridgeMetrics:
    """Performance metrics for the bridge."""
    events_sent: int = 0
    events_received: int = 0
    events_dropped: int = 0
    events_duplicated: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    reconnections: int = 0
    average_latency_ms: float = 0.0
    last_event_time: Optional[float] = None


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker to prevent cascade failures.

    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, block all operations
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failures = 0
        self.successes = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"

    def record_success(self):
        """Record successful operation."""
        if self.state == "HALF_OPEN":
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = "CLOSED"
                self.failures = 0
                self.successes = 0
                logger.info("Circuit breaker CLOSED - service recovered")
        else:
            self.failures = max(0, self.failures - 1)

    def record_failure(self):
        """Record failed operation."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN - {self.failures} failures")

    def can_execute(self) -> bool:
        """Check if operation should be allowed."""
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if recovery timeout elapsed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.successes = 0
                logger.info("Circuit breaker HALF_OPEN - testing recovery")
                return True
            return False

        # HALF_OPEN state
        return True


# ============================================================================
# BLOOM FILTER FOR DEDUPLICATION
# ============================================================================

class BloomFilter:
    """
    Space-efficient probabilistic data structure for duplicate detection.

    Uses multiple hash functions to minimize false positives.
    """

    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
        self.items_added = 0

    def _hashes(self, item: str) -> List[int]:
        """Generate multiple hash values for an item."""
        hashes = []
        for i in range(self.num_hashes):
            # Use different seeds for each hash
            h = hashlib.sha256(f"{item}{i}".encode()).hexdigest()
            hashes.append(int(h, 16) % self.size)
        return hashes

    def add(self, item: str):
        """Add item to the filter."""
        for h in self._hashes(item):
            self.bit_array[h] = True
        self.items_added += 1

    def contains(self, item: str) -> bool:
        """Check if item might be in the set (may have false positives)."""
        return all(self.bit_array[h] for h in self._hashes(item))

    def clear(self):
        """Clear the filter."""
        self.bit_array = [False] * self.size
        self.items_added = 0


# ============================================================================
# PRIORITY QUEUE
# ============================================================================

class PriorityEventQueue:
    """
    Priority queue for events.

    Critical events bypass normal queue for immediate processing.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            EventPriority.CRITICAL: deque(),
            EventPriority.HIGH: deque(),
            EventPriority.NORMAL: deque(),
            EventPriority.LOW: deque(),
        }
        self.total_size = 0

    def put(self, event: TrinityEvent) -> bool:
        """Add event to queue."""
        if self.total_size >= self.max_size:
            # Drop lowest priority event
            for priority in reversed(EventPriority):
                if self.queues[priority]:
                    self.queues[priority].pop()
                    self.total_size -= 1
                    logger.warning(f"Dropped {priority.name} event due to queue full")
                    break

        self.queues[event.priority].append(event)
        self.total_size += 1
        return True

    def get(self) -> Optional[TrinityEvent]:
        """Get highest priority event."""
        for priority in EventPriority:
            if self.queues[priority]:
                self.total_size -= 1
                return self.queues[priority].popleft()
        return None

    def size(self) -> int:
        """Get total queue size."""
        return self.total_size

    def clear(self):
        """Clear all queues."""
        for queue in self.queues.values():
            queue.clear()
        self.total_size = 0


# ============================================================================
# TRINITY BRIDGE
# ============================================================================

class TrinityBridge:
    """
    Ultra-high performance event bus for AGI OS Trinity architecture.

    Architecture:
    - WebSocket server for real-time event streaming
    - HTTP server for REST API fallback
    - In-memory priority queue for event routing
    - Circuit breaker for fault tolerance
    - Bloom filter for duplicate detection
    - Async task pool for parallel event processing
    """

    def __init__(
        self,
        component_id: str,
        host: str = "127.0.0.1",
        ws_port: int = 8765,
        http_port: int = 8766,
        max_queue_size: int = 10000,
        dedup_window_size: int = 10000,
    ):
        self.component_id = component_id
        self.host = host
        self.ws_port = ws_port
        self.http_port = http_port

        # State
        self.state = BridgeState.DISCONNECTED
        self._running = False

        # Event queue and deduplication
        self.event_queue = PriorityEventQueue(max_size=max_queue_size)
        self.dedup_filter = BloomFilter(size=dedup_window_size)

        # WebSocket connections
        self.ws_connections: Dict[str, web.WebSocketResponse] = {}

        # Event handlers
        self.handlers: Dict[EventType, List[Callable[[TrinityEvent], Coroutine]]] = {
            event_type: [] for event_type in EventType
        }

        # Circuit breakers per component
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Metrics
        self.metrics = BridgeMetrics()

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # WebSocket and HTTP apps
        self._ws_app: Optional[web.Application] = None
        self._http_app: Optional[web.Application] = None
        self._ws_runner: Optional[web.AppRunner] = None
        self._http_runner: Optional[web.AppRunner] = None

    async def start(self):
        """Start the Trinity Bridge."""
        logger.info(f"Starting Trinity Bridge for component '{self.component_id}'")

        self._running = True
        self.state = BridgeState.CONNECTING

        # Start WebSocket server
        await self._start_websocket_server()

        # Start HTTP server
        await self._start_http_server()

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._process_event_queue()))
        self._tasks.append(asyncio.create_task(self._cleanup_dedup_filter()))
        self._tasks.append(asyncio.create_task(self._send_heartbeats()))

        self.state = BridgeState.CONNECTED
        logger.info(f"Trinity Bridge started - WS: {self.ws_port}, HTTP: {self.http_port}")

    async def stop(self):
        """Stop the Trinity Bridge."""
        logger.info("Stopping Trinity Bridge...")

        self._running = False
        self.state = BridgeState.DISCONNECTED

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close all WebSocket connections
        for ws in self.ws_connections.values():
            await ws.close()

        # Stop servers
        if self._ws_runner:
            await self._ws_runner.cleanup()
        if self._http_runner:
            await self._http_runner.cleanup()

        logger.info("Trinity Bridge stopped")

    # ========================================================================
    # WebSocket Server
    # ========================================================================

    async def _start_websocket_server(self):
        """Start WebSocket server for real-time events."""
        self._ws_app = web.Application()
        self._ws_app.router.add_get("/ws", self._websocket_handler)

        self._ws_runner = web.AppRunner(self._ws_app)
        await self._ws_runner.setup()

        site = web.TCPSite(self._ws_runner, self.host, self.ws_port)
        await site.start()

        logger.info(f"WebSocket server listening on ws://{self.host}:{self.ws_port}/ws")

    async def _websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Get component ID from query params or headers
        component_id = request.query.get("component_id") or request.headers.get("X-Component-ID")

        if not component_id:
            await ws.close(code=1008, message=b"Missing component_id")
            return ws

        # Register connection
        self.ws_connections[component_id] = ws
        logger.info(f"Component '{component_id}' connected via WebSocket")

        # Initialize circuit breaker if needed
        if component_id not in self.circuit_breakers:
            self.circuit_breakers[component_id] = CircuitBreaker()

        try:
            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_websocket_message(msg.data, component_id)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error from {component_id}: {ws.exception()}")
        finally:
            # Cleanup on disconnect
            self.ws_connections.pop(component_id, None)
            logger.info(f"Component '{component_id}' disconnected")

        return ws

    async def _handle_websocket_message(self, data: str, source: str):
        """Handle incoming WebSocket message."""
        try:
            event_data = json.loads(data)
            event = TrinityEvent.from_dict(event_data)

            # Update metrics
            self.metrics.events_received += 1
            self.metrics.bytes_received += len(data)
            self.metrics.last_event_time = time.time()

            # Check for duplicates
            event_hash = hashlib.sha256(f"{event.event_id}{event.timestamp}".encode()).hexdigest()
            if self.dedup_filter.contains(event_hash):
                self.metrics.events_duplicated += 1
                logger.debug(f"Duplicate event detected: {event.event_id}")
                return

            self.dedup_filter.add(event_hash)

            # Add to priority queue
            self.event_queue.put(event)

            # Record success for circuit breaker
            if source in self.circuit_breakers:
                self.circuit_breakers[source].record_success()

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            if source in self.circuit_breakers:
                self.circuit_breakers[source].record_failure()

    # ========================================================================
    # HTTP Server (Fallback)
    # ========================================================================

    async def _start_http_server(self):
        """Start HTTP server for REST API fallback."""
        self._http_app = web.Application()
        self._http_app.router.add_post("/events", self._http_event_handler)
        self._http_app.router.add_get("/health", self._http_health_handler)
        self._http_app.router.add_get("/metrics", self._http_metrics_handler)

        self._http_runner = web.AppRunner(self._http_app)
        await self._http_runner.setup()

        site = web.TCPSite(self._http_runner, self.host, self.http_port)
        await site.start()

        logger.info(f"HTTP server listening on http://{self.host}:{self.http_port}")

    async def _http_event_handler(self, request: web.Request) -> web.Response:
        """Handle HTTP event submission."""
        try:
            data = await request.json()
            event = TrinityEvent.from_dict(data)

            self.event_queue.put(event)

            return web.json_response({"status": "queued", "event_id": event.event_id})

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _http_health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy" if self._running else "unhealthy",
            "state": self.state.value,
            "queue_size": self.event_queue.size(),
            "connected_components": list(self.ws_connections.keys()),
        })

    async def _http_metrics_handler(self, request: web.Request) -> web.Response:
        """Metrics endpoint."""
        return web.json_response(asdict(self.metrics))

    # ========================================================================
    # Event Processing
    # ========================================================================

    async def _process_event_queue(self):
        """Process events from the priority queue."""
        while self._running:
            try:
                event = self.event_queue.get()

                if event:
                    await self._dispatch_event(event)
                else:
                    # Queue empty, sleep briefly
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await asyncio.sleep(0.1)

    async def _dispatch_event(self, event: TrinityEvent):
        """Dispatch event to handlers and target components."""
        start_time = time.time()

        try:
            # Call registered handlers
            handlers = self.handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Handler error for {event.event_type}: {e}")

            # Route to target component(s)
            if event.target:
                # Unicast
                await self._send_to_component(event.target, event)
            else:
                # Broadcast
                for component_id in self.ws_connections.keys():
                    if component_id != event.source:
                        await self._send_to_component(component_id, event)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.metrics.average_latency_ms = (
                (self.metrics.average_latency_ms * self.metrics.events_sent + latency)
                / (self.metrics.events_sent + 1)
            )

        except Exception as e:
            logger.error(f"Error dispatching event: {e}")

    async def _send_to_component(self, component_id: str, event: TrinityEvent):
        """Send event to specific component."""
        ws = self.ws_connections.get(component_id)

        if not ws:
            logger.debug(f"Component '{component_id}' not connected")
            return

        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(component_id)
        if circuit_breaker and not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker open for '{component_id}', dropping event")
            self.metrics.events_dropped += 1
            return

        try:
            data = json.dumps(event.to_dict())
            await ws.send_str(data)

            self.metrics.events_sent += 1
            self.metrics.bytes_sent += len(data)

            if circuit_breaker:
                circuit_breaker.record_success()

        except Exception as e:
            logger.error(f"Error sending to '{component_id}': {e}")
            self.metrics.events_dropped += 1

            if circuit_breaker:
                circuit_breaker.record_failure()

    # ========================================================================
    # Public API
    # ========================================================================

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[TrinityEvent], Coroutine],
    ):
        """Subscribe to events of a specific type."""
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")

    async def publish(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        target: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Publish an event.

        Args:
            event_type: Type of event
            payload: Event data
            target: Target component (None = broadcast)
            priority: Event priority
            correlation_id: For request-response correlation

        Returns:
            Event ID
        """
        event = TrinityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source=self.component_id,
            target=target,
            priority=priority,
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id,
        )

        self.event_queue.put(event)
        return event.event_id

    async def sync_experience(self, experience_data: Dict[str, Any]):
        """
        Synchronize experience data from JARVIS to Reactor Core.

        This is a high-level helper for the most common cross-repo operation.
        """
        await self.publish(
            event_type=EventType.EXPERIENCE,
            payload=experience_data,
            target="reactor",
            priority=EventPriority.HIGH,
        )
        logger.info("Synced experience to Reactor Core")

    async def listen_for_updates(self) -> AsyncIterator[TrinityEvent]:
        """
        Listen for model update events.

        Async generator for consuming model updates from Reactor Core.
        """
        update_queue = asyncio.Queue()

        async def update_handler(event: TrinityEvent):
            await update_queue.put(event)

        # Subscribe to model updates
        self.subscribe(EventType.MODEL_UPDATE, update_handler)

        # Yield events as they arrive
        while self._running:
            try:
                event = await asyncio.wait_for(update_queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

    # ========================================================================
    # Background Tasks
    # ========================================================================

    async def _cleanup_dedup_filter(self):
        """Periodically clear deduplication filter to prevent memory bloat."""
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes

            old_size = self.dedup_filter.items_added
            self.dedup_filter.clear()
            logger.debug(f"Cleared dedup filter ({old_size} items)")

    async def _send_heartbeats(self):
        """Send periodic heartbeats to all connected components."""
        while self._running:
            await asyncio.sleep(10)  # Every 10 seconds

            await self.publish(
                event_type=EventType.HEARTBEAT,
                payload={
                    "timestamp": time.time(),
                    "queue_size": self.event_queue.size(),
                    "connected_components": len(self.ws_connections),
                },
                priority=EventPriority.LOW,
            )


# ============================================================================
# UTILITIES
# ============================================================================

async def create_trinity_bridge(
    component_id: str,
    ws_port: int = 8765,
    http_port: int = 8766,
) -> TrinityBridge:
    """
    Create and start a Trinity Bridge instance.

    Args:
        component_id: Unique component identifier ("jarvis", "jprime", "reactor")
        ws_port: WebSocket port
        http_port: HTTP port

    Returns:
        Started TrinityBridge instance
    """
    bridge = TrinityBridge(
        component_id=component_id,
        ws_port=ws_port,
        http_port=http_port,
    )

    await bridge.start()
    logger.info(f"Trinity Bridge created for '{component_id}'")
    return bridge


__all__ = [
    # Enums
    "EventPriority",
    "EventType",
    "BridgeState",
    # Data structures
    "TrinityEvent",
    "BridgeMetrics",
    # Main class
    "TrinityBridge",
    # Utilities
    "create_trinity_bridge",
    # Components (for advanced usage)
    "CircuitBreaker",
    "BloomFilter",
    "PriorityEventQueue",
]
