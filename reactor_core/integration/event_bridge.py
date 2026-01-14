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
]
