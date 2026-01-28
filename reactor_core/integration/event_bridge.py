"""
Cross-Repository Event Bridge.

This module provides real-time event synchronization between:
- JARVIS-AI-Agent
- JARVIS Prime
- Reactor Core (Night Shift)

Features:
- WebSocket-based real-time event streaming
- File-based event watching (fallback)
- Redis pub/sub integration (optional)
- Automatic reconnection with backoff
- Event filtering and routing
- Event deduplication
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


class EventSource(Enum):
    """Source of events."""
    JARVIS_AGENT = "jarvis_agent"
    JARVIS_PRIME = "jarvis_prime"
    REACTOR_CORE = "reactor_core"
    SCOUT = "scout"
    USER = "user"
    SYSTEM = "system"


class EventType(Enum):
    """Types of cross-repo events."""
    # Interaction events
    INTERACTION_START = "interaction_start"
    INTERACTION_END = "interaction_end"
    CORRECTION = "correction"
    FEEDBACK = "feedback"

    # Training events
    TRAINING_START = "training_start"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"

    # Scout events
    SCOUT_TOPIC_ADDED = "scout_topic_added"
    SCOUT_PAGE_FETCHED = "scout_page_fetched"
    SCOUT_SYNTHESIS_COMPLETE = "scout_synthesis_complete"

    # System events
    SERVICE_UP = "service_up"
    SERVICE_DOWN = "service_down"
    CONFIG_CHANGED = "config_changed"
    ERROR = "error"

    # Learning events
    NEW_KNOWLEDGE = "new_knowledge"
    MODEL_UPDATED = "model_updated"

    # Cost tracking events (v10.0)
    COST_UPDATE = "cost_update"
    COST_ALERT = "cost_alert"
    COST_REPORT = "cost_report"
    INFERENCE_METRICS = "inference_metrics"

    # Infrastructure events (v10.0)
    RESOURCE_CREATED = "resource_created"
    RESOURCE_DESTROYED = "resource_destroyed"
    ORPHAN_DETECTED = "orphan_detected"
    ORPHAN_CLEANED = "orphan_cleaned"
    ARTIFACT_CLEANED = "artifact_cleaned"
    SQL_STOPPED = "sql_stopped"
    SQL_STARTED = "sql_started"

    # Safety events (v10.3 - Vision Safety Integration)
    SAFETY_AUDIT = "safety_audit"           # Plan was audited for safety
    SAFETY_BLOCKED = "safety_blocked"       # Action was blocked by safety
    SAFETY_CONFIRMED = "safety_confirmed"   # User confirmed risky action
    SAFETY_DENIED = "safety_denied"         # User denied risky action
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"  # Dead man's switch activated
    VISUAL_CLICK_PREVIEW = "visual_click_preview"    # Click preview shown
    VISUAL_CLICK_VETOED = "visual_click_vetoed"      # Click was vetoed during preview

    # Repository Intelligence events (v11.0 - Codebase Brain)
    REPO_MAP_GENERATED = "repo_map_generated"           # Repository map was generated
    REPO_MAP_CACHED = "repo_map_cached"                 # Repository map was cached
    REPO_ANALYSIS_COMPLETE = "repo_analysis_complete"   # Cross-repo analysis finished
    REPO_SYMBOL_FOUND = "repo_symbol_found"             # Symbol was located
    REPO_DEPENDENCY_DETECTED = "repo_dependency_detected"  # New dependency detected
    REPO_CONTEXT_ENRICHED = "repo_context_enriched"     # Context was enriched with repo info

    # Docker Infrastructure Events (v12.0 - Intelligent Self-Healing Integration)
    DOCKER_STARTING = "docker_starting"
    DOCKER_STARTED = "docker_started"
    DOCKER_STOPPING = "docker_stopping"
    DOCKER_STOPPED = "docker_stopped"
    DOCKER_HEALTHY = "docker_healthy"
    DOCKER_UNHEALTHY = "docker_unhealthy"
    DOCKER_RECOVERING = "docker_recovering"
    DOCKER_RECOVERED = "docker_recovered"
    DOCKER_FAILED = "docker_failed"
    DOCKER_TIMEOUT = "docker_timeout"
    DOCKER_REQUEST_START = "docker_request_start"
    DOCKER_REQUEST_STOP = "docker_request_stop"
    DOCKER_HEALTH_CHECK = "docker_health_check"

    # PROJECT TRINITY events - Unified Cognitive Architecture
    # Surveillance commands (JARVIS Ghost Monitor Protocol)
    TRINITY_START_SURVEILLANCE = "trinity_start_surveillance"
    TRINITY_STOP_SURVEILLANCE = "trinity_stop_surveillance"
    TRINITY_SURVEILLANCE_TRIGGERED = "trinity_surveillance_triggered"

    # Window management (Ghost Display operations)
    TRINITY_EXILE_WINDOW = "trinity_exile_window"
    TRINITY_BRING_BACK_WINDOW = "trinity_bring_back_window"
    TRINITY_TELEPORT_WINDOW = "trinity_teleport_window"

    # Cryostasis (Process freeze/thaw)
    TRINITY_FREEZE_APP = "trinity_freeze_app"
    TRINITY_THAW_APP = "trinity_thaw_app"

    # Phantom Hardware (Ghost Display management)
    TRINITY_CREATE_GHOST_DISPLAY = "trinity_create_ghost_display"
    TRINITY_DESTROY_GHOST_DISPLAY = "trinity_destroy_ghost_display"

    # Trinity system events
    TRINITY_HEARTBEAT = "trinity_heartbeat"
    TRINITY_COMMAND_ACK = "trinity_command_ack"
    TRINITY_COMMAND_NACK = "trinity_command_nack"
    TRINITY_STATUS_UPDATE = "trinity_status_update"

    # Cognitive commands from J-Prime
    TRINITY_EXECUTE_PLAN = "trinity_execute_plan"
    TRINITY_ABORT_PLAN = "trinity_abort_plan"
    TRINITY_PLAN_COMPLETE = "trinity_plan_complete"


@dataclass
class CrossRepoEvent:
    """An event that can be shared across repositories."""
    event_id: str
    event_type: EventType
    source: EventSource
    timestamp: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Routing
    target_sources: Set[EventSource] = field(default_factory=set)  # Empty = broadcast
    priority: int = 5  # 1-10, lower is higher priority

    # Deduplication
    _hash: str = ""

    def __post_init__(self):
        if not self._hash:
            self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash for deduplication."""
        content = f"{self.event_type.value}:{self.source.value}:{json.dumps(self.payload, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata,
            "target_sources": [s.value for s in self.target_sources],
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossRepoEvent":
        targets = {EventSource(s) for s in data.get("target_sources", [])}
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            source=EventSource(data["source"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            target_sources=targets,
            priority=data.get("priority", 5),
        )


class EventTransport(ABC):
    """Abstract base class for event transports."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the transport."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the transport."""
        pass

    @abstractmethod
    async def publish(self, event: CrossRepoEvent) -> bool:
        """Publish an event."""
        pass

    @abstractmethod
    async def subscribe(self) -> AsyncIterator[CrossRepoEvent]:
        """Subscribe to events."""
        pass


class FileTransport(EventTransport):
    """
    File-based event transport using a shared directory.

    Uses file watching to detect new events.
    """

    def __init__(
        self,
        events_dir: Path,
        source: EventSource,
        cleanup_hours: int = 24,
    ):
        self.events_dir = events_dir
        self.source = source
        self.cleanup_hours = cleanup_hours
        self._running = False
        self._processed_files: Set[str] = set()

    async def connect(self) -> None:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        logger.info(f"FileTransport connected: {self.events_dir}")

    async def disconnect(self) -> None:
        self._running = False

    async def publish(self, event: CrossRepoEvent) -> bool:
        try:
            filename = f"{event.timestamp.strftime('%Y%m%d_%H%M%S')}_{event.event_id}.json"
            filepath = self.events_dir / filename

            with open(filepath, "w") as f:
                json.dump(event.to_dict(), f)

            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False

    async def subscribe(self) -> AsyncIterator[CrossRepoEvent]:
        while self._running:
            try:
                # Scan for new event files
                for filepath in sorted(self.events_dir.glob("*.json")):
                    if filepath.name in self._processed_files:
                        continue

                    try:
                        with open(filepath) as f:
                            data = json.load(f)

                        event = CrossRepoEvent.from_dict(data)

                        # Skip own events
                        if event.source == self.source:
                            self._processed_files.add(filepath.name)
                            continue

                        # Check if targeted
                        if event.target_sources and self.source not in event.target_sources:
                            self._processed_files.add(filepath.name)
                            continue

                        self._processed_files.add(filepath.name)
                        yield event

                    except Exception as e:
                        logger.warning(f"Error reading event file {filepath}: {e}")
                        self._processed_files.add(filepath.name)

                # Cleanup old files
                await self._cleanup_old_files()

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in file transport subscribe: {e}")
                await asyncio.sleep(5.0)

    async def _cleanup_old_files(self) -> None:
        """Remove old event files."""
        cutoff = datetime.now() - timedelta(hours=self.cleanup_hours)

        for filepath in self.events_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    self._processed_files.discard(filepath.name)
            except Exception:
                pass


class WebSocketTransport(EventTransport):
    """
    WebSocket-based event transport for real-time sync.
    """

    def __init__(
        self,
        url: str,
        source: EventSource,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: int = 10,
    ):
        self.url = url
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._ws = None
        self._session = None
        self._running = False
        self._reconnect_count = 0

    async def connect(self) -> None:
        import aiohttp

        self._running = True
        self._session = aiohttp.ClientSession()

        try:
            self._ws = await self._session.ws_connect(self.url)
            self._reconnect_count = 0

            # Send identity
            await self._ws.send_json({
                "type": "identity",
                "source": self.source.value,
            })

            logger.info(f"WebSocketTransport connected: {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise

    async def disconnect(self) -> None:
        self._running = False

        if self._ws and not self._ws.closed:
            await self._ws.close()

        if self._session and not self._session.closed:
            await self._session.close()

    async def publish(self, event: CrossRepoEvent) -> bool:
        if not self._ws or self._ws.closed:
            return False

        try:
            await self._ws.send_json(event.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to publish event via WebSocket: {e}")
            return False

    async def subscribe(self) -> AsyncIterator[CrossRepoEvent]:
        import aiohttp

        while self._running:
            try:
                if not self._ws or self._ws.closed:
                    await self._reconnect()
                    if not self._ws:
                        await asyncio.sleep(self.reconnect_delay)
                        continue

                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            event = CrossRepoEvent.from_dict(data)

                            # Skip own events
                            if event.source == self.source:
                                continue

                            yield event

                        except Exception as e:
                            logger.warning(f"Error parsing WebSocket message: {e}")

                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        if self._reconnect_count >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self._reconnect_count += 1
        logger.info(f"Reconnecting ({self._reconnect_count}/{self.max_reconnect_attempts})...")

        try:
            await self.connect()
        except Exception as e:
            logger.warning(f"Reconnection failed: {e}")


class EventBridge:
    """
    Main event bridge for cross-repository communication.

    Provides:
    - Multi-transport support (file, WebSocket, Redis)
    - Event routing and filtering
    - Deduplication
    - Callback registration
    """

    def __init__(
        self,
        source: EventSource,
        transports: Optional[List[EventTransport]] = None,
    ):
        self.source = source
        self.transports = transports or []
        self._callbacks: Dict[EventType, List[Callable]] = {}
        self._global_callbacks: List[Callable] = []
        self._running = False
        self._seen_hashes: deque = deque(maxlen=1000)  # Deduplication
        self._tasks: List[asyncio.Task] = []

    def add_transport(self, transport: EventTransport) -> None:
        """Add an event transport."""
        self.transports.append(transport)

    def on_event(
        self,
        event_type: Optional[EventType] = None,
    ) -> Callable:
        """Decorator to register event handler."""
        def decorator(func: Callable) -> Callable:
            if event_type:
                if event_type not in self._callbacks:
                    self._callbacks[event_type] = []
                self._callbacks[event_type].append(func)
            else:
                self._global_callbacks.append(func)
            return func
        return decorator

    def register_handler(
        self,
        handler: Callable,
        event_types: Optional[List[EventType]] = None,
    ) -> None:
        """Register an event handler."""
        if event_types:
            for event_type in event_types:
                if event_type not in self._callbacks:
                    self._callbacks[event_type] = []
                self._callbacks[event_type].append(handler)
        else:
            self._global_callbacks.append(handler)

    async def start(self) -> None:
        """Start the event bridge."""
        self._running = True

        # Connect all transports
        for transport in self.transports:
            await transport.connect()

        # Start subscriber tasks
        for transport in self.transports:
            task = asyncio.create_task(self._handle_transport(transport))
            self._tasks.append(task)

        logger.info(f"EventBridge started with {len(self.transports)} transports")

    async def stop(self) -> None:
        """Stop the event bridge."""
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Disconnect transports
        for transport in self.transports:
            await transport.disconnect()

        logger.info("EventBridge stopped")

    async def _handle_transport(self, transport: EventTransport) -> None:
        """Handle events from a transport."""
        try:
            async for event in transport.subscribe():
                if not self._running:
                    break

                # Deduplication
                if event._hash in self._seen_hashes:
                    continue
                self._seen_hashes.append(event._hash)

                # Dispatch event
                await self._dispatch_event(event)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Transport handler error: {e}")

    async def _dispatch_event(self, event: CrossRepoEvent) -> None:
        """Dispatch event to handlers."""
        # Type-specific handlers
        if event.event_type in self._callbacks:
            for handler in self._callbacks[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

        # Global handlers
        for handler in self._global_callbacks:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")

    async def publish(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        targets: Optional[Set[EventSource]] = None,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish an event to all transports."""
        import uuid

        event = CrossRepoEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            source=self.source,
            timestamp=datetime.now(),
            payload=payload,
            metadata=metadata or {},
            target_sources=targets or set(),
            priority=priority,
        )

        success = True
        for transport in self.transports:
            try:
                result = await transport.publish(event)
                success = success and result
            except Exception as e:
                logger.error(f"Failed to publish to transport: {e}")
                success = False

        return success

    async def emit_interaction(
        self,
        user_input: str,
        response: str,
        success: bool = True,
        confidence: float = 1.0,
    ) -> bool:
        """Convenience method to emit an interaction event."""
        return await self.publish(
            EventType.INTERACTION_END,
            {
                "user_input": user_input,
                "response": response,
                "success": success,
                "confidence": confidence,
            },
        )

    async def emit_correction(
        self,
        original: str,
        corrected: str,
        user_input: str,
    ) -> bool:
        """Convenience method to emit a correction event."""
        return await self.publish(
            EventType.CORRECTION,
            {
                "original_response": original,
                "corrected_response": corrected,
                "user_input": user_input,
            },
            priority=2,  # High priority for learning
        )

    async def emit_training_progress(
        self,
        step: int,
        total_steps: int,
        loss: float,
        metrics: Dict[str, float],
    ) -> bool:
        """Convenience method to emit training progress."""
        return await self.publish(
            EventType.TRAINING_PROGRESS,
            {
                "step": step,
                "total_steps": total_steps,
                "loss": loss,
                "metrics": metrics,
            },
        )

    # =========================================================================
    # Safety Event Convenience Methods (v10.3 - Vision Safety Integration)
    # =========================================================================

    async def emit_safety_audit(
        self,
        goal: str,
        plan_steps: int,
        verdict: str,
        risk_level: str,
        risky_steps: List[Dict[str, Any]],
        confirmation_required: bool,
    ) -> bool:
        """Emit a safety audit event when a plan is audited."""
        return await self.publish(
            EventType.SAFETY_AUDIT,
            {
                "goal": goal,
                "plan_steps": plan_steps,
                "verdict": verdict,
                "risk_level": risk_level,
                "risky_steps": risky_steps,
                "confirmation_required": confirmation_required,
            },
            priority=2,  # High priority for training
        )

    async def emit_safety_blocked(
        self,
        action: str,
        reason: str,
        safety_tier: str,
        auto_blocked: bool = True,
    ) -> bool:
        """Emit when an action is blocked by safety systems."""
        return await self.publish(
            EventType.SAFETY_BLOCKED,
            {
                "action": action,
                "reason": reason,
                "safety_tier": safety_tier,
                "auto_blocked": auto_blocked,
            },
            priority=1,  # Highest priority
        )

    async def emit_safety_confirmation(
        self,
        action: str,
        risk_level: str,
        confirmed: bool,
        confirmation_method: str,  # "voice", "text", "timeout"
        user_response: Optional[str] = None,
    ) -> bool:
        """Emit when user confirms or denies a risky action."""
        event_type = EventType.SAFETY_CONFIRMED if confirmed else EventType.SAFETY_DENIED
        return await self.publish(
            event_type,
            {
                "action": action,
                "risk_level": risk_level,
                "confirmed": confirmed,
                "confirmation_method": confirmation_method,
                "user_response": user_response,
            },
            priority=2,
        )

    async def emit_kill_switch_triggered(
        self,
        trigger_method: str,  # "mouse_corner", "voice", "keyboard"
        halted_action: Optional[str] = None,
        response_time_ms: float = 0.0,
    ) -> bool:
        """Emit when the dead man's switch is triggered."""
        return await self.publish(
            EventType.KILL_SWITCH_TRIGGERED,
            {
                "trigger_method": trigger_method,
                "halted_action": halted_action,
                "response_time_ms": response_time_ms,
            },
            priority=1,  # Highest priority
        )

    async def emit_visual_click_event(
        self,
        x: int,
        y: int,
        button: str,
        vetoed: bool,
        preview_duration_ms: float,
        veto_reason: Optional[str] = None,
    ) -> bool:
        """Emit visual click preview or veto event."""
        event_type = EventType.VISUAL_CLICK_VETOED if vetoed else EventType.VISUAL_CLICK_PREVIEW
        return await self.publish(
            event_type,
            {
                "x": x,
                "y": y,
                "button": button,
                "vetoed": vetoed,
                "preview_duration_ms": preview_duration_ms,
                "veto_reason": veto_reason,
            },
            priority=3,
        )


# =============================================================================
# Docker State Integration for Training Jobs (v12.0)
# =============================================================================

# Docker state directory (shared via Trinity Protocol)
DOCKER_STATE_DIR = Path.home() / ".jarvis" / "trinity" / "docker"
DOCKER_STATE_FILE = DOCKER_STATE_DIR / "state.json"
DOCKER_EVENTS_FILE = DOCKER_STATE_DIR / "events.json"
DOCKER_CHECK_INTERVAL = float(os.getenv("JARVIS_DOCKER_CHECK_INTERVAL", "15.0"))


@dataclass
class DockerStateSnapshot:
    """Snapshot of Docker state from Trinity Protocol shared state."""
    available: bool = False
    status: str = "unknown"
    health_score: float = 0.0
    last_check: str = ""
    recovery_in_progress: bool = False
    current_level: str = "none"
    error_message: str = ""

    @classmethod
    def from_state_file(cls) -> "DockerStateSnapshot":
        """Load Docker state from shared Trinity state file."""
        if not DOCKER_STATE_FILE.exists():
            return cls()

        try:
            with open(DOCKER_STATE_FILE, "r") as f:
                data = json.load(f)

            return cls(
                available=data.get("docker_available", False),
                status=data.get("status", "unknown"),
                health_score=data.get("health_score", 0.0),
                last_check=data.get("last_update", ""),
                recovery_in_progress=data.get("recovery_in_progress", False),
                current_level=data.get("current_recovery_level", "none"),
                error_message=data.get("error", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to read Docker state file: {e}")
            return cls()


class DockerAwareTrainingScheduler:
    """
    Docker-aware training job scheduler for Reactor Core.

    Features:
    - Monitors Docker state via Trinity Protocol
    - Pauses training when Docker becomes unavailable
    - Resumes training when Docker recovers
    - Adjusts resource allocation based on Docker health
    - Emits events for training state changes
    """

    def __init__(
        self,
        event_bridge: EventBridge,
        min_health_score: float = 0.7,
        check_interval: float = DOCKER_CHECK_INTERVAL,
    ):
        self.event_bridge = event_bridge
        self.min_health_score = min_health_score
        self.check_interval = check_interval

        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._docker_available = False
        self._last_state: Optional[DockerStateSnapshot] = None
        self._paused_jobs: Set[str] = set()
        self._state_change_callbacks: List[Callable[[DockerStateSnapshot, bool], Any]] = []

    def register_state_change_callback(
        self,
        callback: Callable[[DockerStateSnapshot, bool], Any],
    ) -> None:
        """Register callback for Docker state changes.

        Args:
            callback: Function(state, is_available) called on state change
        """
        self._state_change_callbacks.append(callback)

    async def start(self) -> None:
        """Start Docker state monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("DockerAwareTrainingScheduler started")

    async def stop(self) -> None:
        """Stop Docker state monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("DockerAwareTrainingScheduler stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                state = DockerStateSnapshot.from_state_file()
                is_available = self._evaluate_availability(state)

                # Check for state changes
                if self._last_state is None or is_available != self._docker_available:
                    await self._handle_state_change(state, is_available)

                self._last_state = state
                self._docker_available = is_available

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Docker monitor loop: {e}")
                await asyncio.sleep(self.check_interval * 2)

    def _evaluate_availability(self, state: DockerStateSnapshot) -> bool:
        """Evaluate if Docker is available for training jobs."""
        if not state.available:
            return False

        if state.recovery_in_progress:
            return False

        if state.health_score < self.min_health_score:
            return False

        return state.status in ("running", "healthy")

    async def _handle_state_change(
        self,
        state: DockerStateSnapshot,
        is_available: bool,
    ) -> None:
        """Handle Docker state change."""
        logger.info(
            f"Docker state changed: available={is_available}, "
            f"status={state.status}, health={state.health_score:.2f}"
        )

        # Emit event
        event_type = EventType.DOCKER_HEALTHY if is_available else EventType.DOCKER_UNHEALTHY
        await self.event_bridge.publish(
            event_type,
            {
                "available": is_available,
                "status": state.status,
                "health_score": state.health_score,
                "recovery_in_progress": state.recovery_in_progress,
            },
        )

        # Call registered callbacks
        for callback in self._state_change_callbacks:
            try:
                result = callback(state, is_available)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    def is_docker_available(self) -> bool:
        """Check if Docker is currently available."""
        return self._docker_available

    def get_current_state(self) -> Optional[DockerStateSnapshot]:
        """Get the current Docker state snapshot."""
        return self._last_state

    async def wait_for_docker(
        self,
        timeout: float = 300.0,
        check_interval: float = 5.0,
    ) -> bool:
        """Wait for Docker to become available.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check

        Returns:
            True if Docker became available, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            state = DockerStateSnapshot.from_state_file()
            if self._evaluate_availability(state):
                self._docker_available = True
                self._last_state = state
                return True

            await asyncio.sleep(check_interval)

        return False

    async def request_docker_start(self) -> bool:
        """Request Docker to be started via Trinity Protocol."""
        try:
            DOCKER_STATE_DIR.mkdir(parents=True, exist_ok=True)

            request_file = DOCKER_STATE_DIR / "start_request.json"
            request_data = {
                "requester": "reactor_core",
                "timestamp": datetime.now().isoformat(),
                "reason": "training_job_pending",
            }

            # Atomic write
            temp_file = request_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(request_data, f)
            temp_file.rename(request_file)

            # Emit event
            await self.event_bridge.publish(
                EventType.DOCKER_REQUEST_START,
                request_data,
            )

            logger.info("Docker start request sent via Trinity Protocol")
            return True

        except Exception as e:
            logger.error(f"Failed to request Docker start: {e}")
            return False

    def mark_job_paused(self, job_id: str) -> None:
        """Mark a training job as paused due to Docker unavailability."""
        self._paused_jobs.add(job_id)
        logger.info(f"Training job {job_id} marked as paused (Docker unavailable)")

    def mark_job_resumed(self, job_id: str) -> None:
        """Mark a training job as resumed after Docker recovery."""
        self._paused_jobs.discard(job_id)
        logger.info(f"Training job {job_id} marked as resumed (Docker available)")

    def get_paused_jobs(self) -> Set[str]:
        """Get set of paused job IDs."""
        return self._paused_jobs.copy()

    async def emit_training_docker_status(
        self,
        job_id: str,
        status: str,  # "paused", "resumed", "waiting", "failed"
        reason: str,
    ) -> bool:
        """Emit training job Docker status event."""
        return await self.event_bridge.publish(
            EventType.TRAINING_PROGRESS,
            {
                "job_id": job_id,
                "docker_status": status,
                "reason": reason,
                "docker_available": self._docker_available,
                "docker_health": self._last_state.health_score if self._last_state else 0.0,
            },
            metadata={"docker_related": True},
        )


class DockerEventWatcher:
    """
    Watches Docker events from Trinity Protocol for training job coordination.

    Processes events from the shared Docker events file and dispatches
    them to registered handlers.
    """

    def __init__(self, event_bridge: EventBridge):
        self.event_bridge = event_bridge
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._last_event_id: str = ""
        self._handlers: Dict[str, List[Callable]] = {}

    def on_docker_event(self, event_type: str) -> Callable:
        """Decorator to register Docker event handler."""
        def decorator(func: Callable) -> Callable:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(func)
            return func
        return decorator

    async def start(self) -> None:
        """Start watching Docker events."""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info("DockerEventWatcher started")

    async def stop(self) -> None:
        """Stop watching Docker events."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        logger.info("DockerEventWatcher stopped")

    async def _watch_loop(self) -> None:
        """Main event watching loop."""
        while self._running:
            try:
                if DOCKER_EVENTS_FILE.exists():
                    events = await self._read_events()
                    for event in events:
                        await self._dispatch_event(event)

                await asyncio.sleep(2.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Docker event watch loop: {e}")
                await asyncio.sleep(5.0)

    async def _read_events(self) -> List[Dict[str, Any]]:
        """Read new events from events file."""
        try:
            with open(DOCKER_EVENTS_FILE, "r") as f:
                data = json.load(f)

            events = data.get("events", [])

            # Filter to new events only
            new_events = []
            found_last = self._last_event_id == ""

            for event in events:
                event_id = event.get("event_id", "")

                if found_last:
                    new_events.append(event)
                elif event_id == self._last_event_id:
                    found_last = True

            if new_events and new_events[-1].get("event_id"):
                self._last_event_id = new_events[-1]["event_id"]

            return new_events

        except Exception as e:
            logger.warning(f"Failed to read Docker events: {e}")
            return []

    async def _dispatch_event(self, event: Dict[str, Any]) -> None:
        """Dispatch a Docker event to handlers."""
        event_type = event.get("event_type", "")

        # Call type-specific handlers
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error in Docker event handler: {e}")

        # Forward to event bridge as CrossRepoEvent
        try:
            bridge_event_type = self._map_docker_event_type(event_type)
            if bridge_event_type:
                await self.event_bridge.publish(
                    bridge_event_type,
                    event.get("payload", {}),
                    metadata={"docker_event": True, "original_type": event_type},
                )
        except Exception as e:
            logger.warning(f"Failed to forward Docker event to bridge: {e}")

    def _map_docker_event_type(self, docker_type: str) -> Optional[EventType]:
        """Map Docker event type string to EventType enum."""
        mapping = {
            "starting": EventType.DOCKER_STARTING,
            "started": EventType.DOCKER_STARTED,
            "stopping": EventType.DOCKER_STOPPING,
            "stopped": EventType.DOCKER_STOPPED,
            "healthy": EventType.DOCKER_HEALTHY,
            "unhealthy": EventType.DOCKER_UNHEALTHY,
            "recovering": EventType.DOCKER_RECOVERING,
            "recovered": EventType.DOCKER_RECOVERED,
            "failed": EventType.DOCKER_FAILED,
            "timeout": EventType.DOCKER_TIMEOUT,
            "health_check": EventType.DOCKER_HEALTH_CHECK,
        }
        return mapping.get(docker_type.lower())


def _get_default_events_dir() -> Path:
    """
    Get the default events directory using dynamic path resolution.

    Priority:
    1. JARVIS_EVENTS_DIR environment variable
    2. base_config's resolve_path (if available)
    3. XDG_DATA_HOME/jarvis/events
    """
    # Check environment variable first
    env_value = os.getenv("JARVIS_EVENTS_DIR")
    if env_value:
        path = Path(env_value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Try base_config's path resolver
    try:
        from reactor_core.config.base_config import resolve_path
        return resolve_path("jarvis_events", env_var="JARVIS_EVENTS_DIR", subdir="events")
    except ImportError:
        pass

    # XDG-compliant fallback
    xdg_data_home = os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    path = Path(xdg_data_home) / "jarvis" / "events"
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_event_bridge(
    source: EventSource,
    events_dir: Optional[Path] = None,
    websocket_url: Optional[str] = None,
) -> EventBridge:
    """
    Factory function to create an event bridge with default transports.

    Args:
        source: The source identifier for this service
        events_dir: Directory for file-based events (uses dynamic resolution if None)
        websocket_url: Optional WebSocket URL for real-time sync

    Returns:
        Configured EventBridge instance
    """
    transports = []

    # File transport (always enabled as fallback)
    if events_dir is None:
        events_dir = _get_default_events_dir()

    transports.append(FileTransport(events_dir, source))

    # WebSocket transport (if URL provided)
    if websocket_url:
        transports.append(WebSocketTransport(websocket_url, source))

    return EventBridge(source, transports)


# =============================================================================
# Cross-Repo Readiness Integration (v13.0 - Inline Readiness Tracking)
# =============================================================================

# Trinity readiness state directory
TRINITY_STATE_DIR = Path.home() / ".jarvis" / "trinity" / "state"
JARVIS_BODY_READINESS_FILE = TRINITY_STATE_DIR / "jarvis-body_readiness.json"
READINESS_STALENESS_THRESHOLD = float(os.getenv("JARVIS_READINESS_STALENESS", "120.0"))  # 2 minutes


@dataclass
class JarvisBodyReadinessState:
    """
    Snapshot of JARVIS-AI-Agent (jarvis-body) readiness state from Trinity Protocol.

    This allows Reactor Core to wait for jarvis-body to be fully ready before
    starting training jobs or other operations that depend on it.
    """
    phase: str = "NOT_STARTED"
    is_ready: bool = False
    is_healthy: bool = False
    ready_at: Optional[float] = None
    started_at: Optional[float] = None
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    critical_components_ready: int = 0
    total_critical_components: int = 0
    timestamp: float = 0.0

    @property
    def is_stale(self) -> bool:
        """Check if state data is stale (older than threshold)."""
        if self.timestamp == 0.0:
            return True
        return (time.time() - self.timestamp) > READINESS_STALENESS_THRESHOLD

    @property
    def startup_duration(self) -> Optional[float]:
        """Get time from start to ready, if available."""
        if self.ready_at and self.started_at:
            return self.ready_at - self.started_at
        return None

    @classmethod
    def from_state_file(cls) -> "JarvisBodyReadinessState":
        """Load readiness state from Trinity Protocol shared state file."""
        if not JARVIS_BODY_READINESS_FILE.exists():
            return cls()

        try:
            with open(JARVIS_BODY_READINESS_FILE, "r") as f:
                data = json.load(f)

            return cls(
                phase=data.get("phase", "NOT_STARTED"),
                is_ready=data.get("is_ready", False),
                is_healthy=data.get("is_healthy", False),
                ready_at=data.get("ready_at"),
                started_at=data.get("started_at"),
                components=data.get("components", {}),
                critical_components_ready=data.get("critical_components_ready", 0),
                total_critical_components=data.get("total_critical_components", 0),
                timestamp=data.get("timestamp", 0.0),
            )
        except Exception as e:
            logger.warning(f"Failed to read jarvis-body readiness state: {e}")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase,
            "is_ready": self.is_ready,
            "is_healthy": self.is_healthy,
            "ready_at": self.ready_at,
            "started_at": self.started_at,
            "components": self.components,
            "critical_components_ready": self.critical_components_ready,
            "total_critical_components": self.total_critical_components,
            "timestamp": self.timestamp,
            "is_stale": self.is_stale,
            "startup_duration": self.startup_duration,
        }


class JarvisBodyReadinessWatcher:
    """
    Watches jarvis-body readiness state for cross-repo coordination.

    This enables Reactor Core training jobs and other operations to:
    - Wait for jarvis-body to be ready before starting
    - React to jarvis-body becoming unavailable
    - Coordinate startup sequences across repos

    Features:
    - Async polling with configurable interval
    - Callback registration for state changes
    - Staleness detection (auto-mark unavailable if no updates)
    - Integration with EventBridge for cross-repo events
    """

    def __init__(
        self,
        event_bridge: Optional[EventBridge] = None,
        check_interval: float = 5.0,
    ):
        self.event_bridge = event_bridge
        self.check_interval = check_interval

        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._last_state: Optional[JarvisBodyReadinessState] = None
        self._was_ready: bool = False

        # Callbacks: (state, became_ready: bool, became_unavailable: bool)
        self._state_change_callbacks: List[Callable[[JarvisBodyReadinessState, bool, bool], Any]] = []

    def register_state_change_callback(
        self,
        callback: Callable[[JarvisBodyReadinessState, bool, bool], Any],
    ) -> None:
        """
        Register callback for jarvis-body readiness state changes.

        Args:
            callback: Function(state, became_ready, became_unavailable) called on change
        """
        self._state_change_callbacks.append(callback)

    async def start(self) -> None:
        """Start watching jarvis-body readiness."""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info("JarvisBodyReadinessWatcher started")

    async def stop(self) -> None:
        """Stop watching jarvis-body readiness."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        logger.info("JarvisBodyReadinessWatcher stopped")

    async def _watch_loop(self) -> None:
        """Main watching loop."""
        while self._running:
            try:
                state = JarvisBodyReadinessState.from_state_file()

                # Determine if state changed
                is_now_ready = state.is_ready and not state.is_stale
                became_ready = is_now_ready and not self._was_ready
                became_unavailable = not is_now_ready and self._was_ready

                if became_ready or became_unavailable:
                    await self._handle_state_change(state, became_ready, became_unavailable)

                self._last_state = state
                self._was_ready = is_now_ready

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in jarvis-body readiness watch loop: {e}")
                await asyncio.sleep(self.check_interval * 2)

    async def _handle_state_change(
        self,
        state: JarvisBodyReadinessState,
        became_ready: bool,
        became_unavailable: bool,
    ) -> None:
        """Handle jarvis-body readiness state change."""
        if became_ready:
            logger.info(
                f"jarvis-body became READY: phase={state.phase}, "
                f"components={state.critical_components_ready}/{state.total_critical_components}"
            )
        elif became_unavailable:
            reason = "stale" if state.is_stale else f"phase={state.phase}"
            logger.warning(f"jarvis-body became UNAVAILABLE: {reason}")

        # Emit event via EventBridge
        if self.event_bridge:
            try:
                await self.event_bridge.publish(
                    EventType.SERVICE_UP if became_ready else EventType.SERVICE_DOWN,
                    {
                        "service": "jarvis-body",
                        "ready": state.is_ready,
                        "healthy": state.is_healthy,
                        "phase": state.phase,
                        "stale": state.is_stale,
                        "components_ready": state.critical_components_ready,
                        "components_total": state.total_critical_components,
                    },
                    priority=2,
                )
            except Exception as e:
                logger.warning(f"Failed to emit jarvis-body state event: {e}")

        # Call registered callbacks
        for callback in self._state_change_callbacks:
            try:
                result = callback(state, became_ready, became_unavailable)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Error in jarvis-body state change callback: {e}")

    def is_jarvis_body_ready(self) -> bool:
        """
        Quick synchronous check if jarvis-body is ready.

        Returns:
            True if jarvis-body is ready and state is not stale
        """
        state = JarvisBodyReadinessState.from_state_file()
        return state.is_ready and not state.is_stale

    def get_current_state(self) -> JarvisBodyReadinessState:
        """Get the current jarvis-body readiness state."""
        return JarvisBodyReadinessState.from_state_file()

    async def wait_for_jarvis_body(
        self,
        timeout: float = 120.0,
        check_interval: float = 2.0,
        require_healthy: bool = False,
    ) -> bool:
        """
        Wait for jarvis-body to become ready.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check
            require_healthy: If True, wait for HEALTHY phase (not just READY)

        Returns:
            True if jarvis-body became ready within timeout, False otherwise
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            state = JarvisBodyReadinessState.from_state_file()

            if require_healthy:
                if state.is_healthy and not state.is_stale:
                    logger.info(f"jarvis-body is HEALTHY after {time.time() - start_time:.1f}s")
                    return True
            else:
                if state.is_ready and not state.is_stale:
                    logger.info(f"jarvis-body is READY after {time.time() - start_time:.1f}s")
                    return True

            await asyncio.sleep(check_interval)

        logger.warning(f"Timeout waiting for jarvis-body readiness after {timeout}s")
        return False

    async def get_component_status(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific jarvis-body component.

        Args:
            component_name: Name of component (e.g., "websocket", "voice_biometric")

        Returns:
            Component status dict or None if not found
        """
        state = JarvisBodyReadinessState.from_state_file()
        return state.components.get(component_name)

    async def wait_for_component(
        self,
        component_name: str,
        timeout: float = 60.0,
        check_interval: float = 1.0,
    ) -> bool:
        """
        Wait for a specific jarvis-body component to become ready.

        Args:
            component_name: Name of component to wait for
            timeout: Maximum time to wait
            check_interval: How often to check

        Returns:
            True if component became ready within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            state = JarvisBodyReadinessState.from_state_file()

            if state.is_stale:
                await asyncio.sleep(check_interval)
                continue

            component = state.components.get(component_name)
            if component and component.get("is_ready", False):
                logger.info(f"jarvis-body component '{component_name}' is ready")
                return True

            await asyncio.sleep(check_interval)

        logger.warning(f"Timeout waiting for jarvis-body component '{component_name}'")
        return False


# =============================================================================
# Cloud Offload State Integration (v14.0)
# =============================================================================

CLOUD_OFFLOAD_STATE_FILE = TRINITY_STATE_DIR / "cloud_offload_state.json"
CLOUD_OFFLOAD_STALENESS_THRESHOLD = 60.0  # Cloud state stale after 60s


@dataclass
class CloudOffloadState:
    """
    State of jarvis-body cloud offloading.

    This reflects when jarvis-body activates cloud offloading due to
    memory/CPU pressure, enabling other repos to route ML inference
    to cloud endpoints.
    """

    cloud_offload_active: bool = False
    cloud_ip: str = ""
    reason: str = ""
    triggered_at: float = 0.0
    timestamp: float = 0.0
    vm_name: str = ""
    vm_zone: str = ""
    prefer_cloud_run: bool = False
    use_cloud_ml: bool = False
    spot_vm_enabled: bool = False

    @property
    def is_stale(self) -> bool:
        """Check if state is stale (no updates for threshold period)."""
        if self.timestamp == 0.0:
            return True
        return (time.time() - self.timestamp) > CLOUD_OFFLOAD_STALENESS_THRESHOLD

    @property
    def age_seconds(self) -> float:
        """Get age of state in seconds."""
        if self.timestamp == 0.0:
            return float("inf")
        return time.time() - self.timestamp

    @property
    def cloud_ml_endpoint(self) -> Optional[str]:
        """Get cloud ML endpoint URL if available."""
        if self.cloud_offload_active and self.cloud_ip:
            return f"http://{self.cloud_ip}:8080"
        return None

    @classmethod
    def from_state_file(cls) -> "CloudOffloadState":
        """
        Load cloud offload state from Trinity Protocol state file.

        Returns:
            CloudOffloadState (may have default values if file missing/invalid)
        """
        try:
            if not CLOUD_OFFLOAD_STATE_FILE.exists():
                return cls()

            data = json.loads(CLOUD_OFFLOAD_STATE_FILE.read_text())
            return cls(
                cloud_offload_active=data.get("cloud_offload_active", False),
                cloud_ip=data.get("cloud_ip", ""),
                reason=data.get("reason", ""),
                triggered_at=data.get("triggered_at", 0.0),
                timestamp=data.get("timestamp", 0.0),
                vm_name=data.get("vm_name", ""),
                vm_zone=data.get("vm_zone", ""),
                prefer_cloud_run=data.get("prefer_cloud_run", False),
                use_cloud_ml=data.get("use_cloud_ml", False),
                spot_vm_enabled=data.get("spot_vm_enabled", False),
            )
        except Exception as e:
            logger.debug(f"Failed to load cloud offload state: {e}")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cloud_offload_active": self.cloud_offload_active,
            "cloud_ip": self.cloud_ip,
            "reason": self.reason,
            "triggered_at": self.triggered_at,
            "timestamp": self.timestamp,
            "vm_name": self.vm_name,
            "vm_zone": self.vm_zone,
            "prefer_cloud_run": self.prefer_cloud_run,
            "use_cloud_ml": self.use_cloud_ml,
            "spot_vm_enabled": self.spot_vm_enabled,
            "is_stale": self.is_stale,
            "age_seconds": self.age_seconds,
            "cloud_ml_endpoint": self.cloud_ml_endpoint,
        }


class CloudOffloadWatcher:
    """
    Watches cloud offload state for training orchestration.

    This enables Reactor Core to:
    - Use cloud ML endpoints when jarvis-body is under pressure
    - Route training data collection to cloud services
    - Coordinate with cloud VM lifecycle

    Features:
    - Async polling with configurable interval
    - Callback registration for state changes
    - Automatic staleness detection
    - Integration with EventBridge
    """

    def __init__(
        self,
        event_bridge: Optional[EventBridge] = None,
        check_interval: float = 10.0,
    ):
        self.event_bridge = event_bridge
        self.check_interval = check_interval

        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._last_state: Optional[CloudOffloadState] = None
        self._was_active: bool = False

        # Callbacks: (state, became_active: bool, became_inactive: bool)
        self._state_change_callbacks: List[Callable[[CloudOffloadState, bool, bool], Any]] = []

    def register_state_change_callback(
        self,
        callback: Callable[[CloudOffloadState, bool, bool], Any],
    ) -> None:
        """
        Register callback for cloud offload state changes.

        Args:
            callback: Function(state, became_active, became_inactive) called on change
        """
        self._state_change_callbacks.append(callback)

    async def start(self) -> None:
        """Start watching cloud offload state."""
        if self._running:
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info("[v14.0] CloudOffloadWatcher started")

    async def stop(self) -> None:
        """Stop watching cloud offload state."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        logger.info("[v14.0] CloudOffloadWatcher stopped")

    async def _watch_loop(self) -> None:
        """Main watching loop."""
        while self._running:
            try:
                state = CloudOffloadState.from_state_file()

                # Determine if state changed
                is_now_active = state.cloud_offload_active and not state.is_stale
                became_active = is_now_active and not self._was_active
                became_inactive = not is_now_active and self._was_active

                if became_active or became_inactive:
                    await self._handle_state_change(state, became_active, became_inactive)

                self._last_state = state
                self._was_active = is_now_active

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[v14.0] Error in cloud offload watch loop: {e}")
                await asyncio.sleep(self.check_interval * 2)

    async def _handle_state_change(
        self,
        state: CloudOffloadState,
        became_active: bool,
        became_inactive: bool,
    ) -> None:
        """Handle cloud offload state change."""
        if became_active:
            logger.info(
                f"[v14.0] Cloud offloading ACTIVATED: reason='{state.reason}', "
                f"endpoint={state.cloud_ml_endpoint}"
            )
        elif became_inactive:
            logger.info("[v14.0] Cloud offloading DEACTIVATED - returning to local processing")

        # Emit event via EventBridge
        if self.event_bridge:
            try:
                event_type = EventType.RESOURCE_CREATED if became_active else EventType.RESOURCE_DESTROYED
                await self.event_bridge.publish(
                    event_type,
                    {
                        "resource_type": "cloud_offload",
                        "active": state.cloud_offload_active,
                        "cloud_ip": state.cloud_ip,
                        "reason": state.reason,
                        "vm_name": state.vm_name,
                        "endpoint": state.cloud_ml_endpoint,
                    },
                    priority=2,
                )
            except Exception as e:
                logger.warning(f"[v14.0] Failed to emit cloud offload event: {e}")

        # Call registered callbacks
        for callback in self._state_change_callbacks:
            try:
                result = callback(state, became_active, became_inactive)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[v14.0] Error in cloud offload state change callback: {e}")

    def is_cloud_offload_active(self) -> bool:
        """
        Quick synchronous check if cloud offloading is active.

        Returns:
            True if cloud offloading is active and state is not stale
        """
        state = CloudOffloadState.from_state_file()
        return state.cloud_offload_active and not state.is_stale

    def get_current_state(self) -> CloudOffloadState:
        """Get the current cloud offload state."""
        return CloudOffloadState.from_state_file()

    def get_cloud_ml_endpoint(self) -> Optional[str]:
        """
        Get cloud ML endpoint if offloading is active.

        Returns:
            Cloud ML endpoint URL or None
        """
        state = CloudOffloadState.from_state_file()
        if state.cloud_offload_active and not state.is_stale:
            return state.cloud_ml_endpoint
        return None

    async def wait_for_cloud_ready(
        self,
        timeout: float = 120.0,
        check_interval: float = 5.0,
    ) -> bool:
        """
        Wait for cloud offloading to have a ready endpoint.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check

        Returns:
            True if cloud endpoint became available within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            state = CloudOffloadState.from_state_file()

            if state.cloud_offload_active and state.cloud_ip and not state.is_stale:
                logger.info(f"[v14.0] Cloud offload ready: {state.cloud_ml_endpoint}")
                return True

            await asyncio.sleep(check_interval)

        logger.warning(f"[v14.0] Timeout waiting for cloud offload to be ready ({timeout}s)")
        return False

    async def should_use_cloud_inference(self) -> bool:
        """
        Determine if ML inference should use cloud endpoints.

        Combines:
        - Cloud offload state from jarvis-body
        - Environment variable overrides

        Returns:
            True if cloud inference should be used
        """
        # Check environment override first
        if os.environ.get("JARVIS_PREFER_CLOUD_RUN", "").lower() == "true":
            return True

        if os.environ.get("JARVIS_USE_CLOUD_ML", "").lower() == "true":
            return True

        # Check cloud offload state
        return self.is_cloud_offload_active()


async def check_cloud_offload_state() -> CloudOffloadState:
    """
    Convenience function to check cloud offload state.

    Returns:
        Current CloudOffloadState
    """
    return CloudOffloadState.from_state_file()


def is_cloud_offload_active_sync() -> bool:
    """
    Synchronous convenience function to check if cloud offloading is active.

    Returns:
        True if cloud offloading is active and state is not stale
    """
    state = CloudOffloadState.from_state_file()
    return state.cloud_offload_active and not state.is_stale


def get_cloud_ml_endpoint_sync() -> Optional[str]:
    """
    Synchronous convenience function to get cloud ML endpoint.

    Returns:
        Cloud ML endpoint URL or None
    """
    state = CloudOffloadState.from_state_file()
    if state.cloud_offload_active and not state.is_stale:
        return state.cloud_ml_endpoint
    return None


async def check_jarvis_body_readiness() -> JarvisBodyReadinessState:
    """
    Convenience function to check jarvis-body readiness state.

    Returns:
        Current JarvisBodyReadinessState
    """
    return JarvisBodyReadinessState.from_state_file()


def is_jarvis_body_ready_sync() -> bool:
    """
    Synchronous convenience function to check if jarvis-body is ready.

    Returns:
        True if jarvis-body is ready and state is not stale
    """
    state = JarvisBodyReadinessState.from_state_file()
    return state.is_ready and not state.is_stale


# Convenience exports
__all__ = [
    "EventBridge",
    "EventTransport",
    "FileTransport",
    "WebSocketTransport",
    "CrossRepoEvent",
    "EventSource",
    "EventType",
    "create_event_bridge",
    # Docker Integration (v12.0)
    "DockerStateSnapshot",
    "DockerAwareTrainingScheduler",
    "DockerEventWatcher",
    "DOCKER_STATE_DIR",
    "DOCKER_STATE_FILE",
    "DOCKER_EVENTS_FILE",
    # Cross-Repo Readiness Integration (v13.0)
    "JarvisBodyReadinessState",
    "JarvisBodyReadinessWatcher",
    "check_jarvis_body_readiness",
    "is_jarvis_body_ready_sync",
    "TRINITY_STATE_DIR",
    "JARVIS_BODY_READINESS_FILE",
    # Cloud Offload State Integration (v14.0)
    "CloudOffloadState",
    "CloudOffloadWatcher",
    "check_cloud_offload_state",
    "is_cloud_offload_active_sync",
    "get_cloud_ml_endpoint_sync",
    "CLOUD_OFFLOAD_STATE_FILE",
    "CLOUD_OFFLOAD_STALENESS_THRESHOLD",
]
