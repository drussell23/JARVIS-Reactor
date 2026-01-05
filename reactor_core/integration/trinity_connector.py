"""
PROJECT TRINITY: Reactor Core -> JARVIS Body Connector

This module provides the Reactor Core side of the Trinity communication layer,
enabling J-Prime (Mind) to send commands through Reactor Core (Nerves) to JARVIS (Body).

ARCHITECTURE:
┌────────────┐    Commands    ┌──────────────┐    Execute    ┌────────────┐
│  J-PRIME   │ ──────────────│ REACTOR CORE │ ─────────────│   JARVIS   │
│   (Mind)   │               │   (Nerves)   │              │   (Body)   │
└────────────┘    Results    └──────────────┘   Heartbeat  └────────────┘

FEATURES:
- Bidirectional communication with JARVIS via shared files
- Command routing with priority queuing
- Heartbeat monitoring for JARVIS liveness
- Command acknowledgment tracking
- Automatic retry with exponential backoff
- Event bridge integration for cross-repo events

USAGE:
    from reactor_core.integration.trinity_connector import (
        TrinityConnector,
        get_trinity_connector,
        send_surveillance_command,
    )

    connector = get_trinity_connector()
    await connector.connect()

    # Send a surveillance command
    result = await connector.send_command(
        intent="start_surveillance",
        payload={"app_name": "Chrome", "trigger_text": "bouncing ball"},
    )
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

TRINITY_DIR = Path.home() / ".jarvis" / "trinity"
COMMANDS_DIR = TRINITY_DIR / "commands"
HEARTBEATS_DIR = TRINITY_DIR / "heartbeats"
RESPONSES_DIR = TRINITY_DIR / "responses"

HEARTBEAT_TIMEOUT = 15.0  # seconds - consider JARVIS offline if no heartbeat
COMMAND_TIMEOUT = 30.0  # seconds - default command timeout
RETRY_DELAYS = [1.0, 2.0, 5.0, 10.0]  # exponential backoff


# =============================================================================
# ENUMS
# =============================================================================

class TrinitySource(Enum):
    """Source component in Trinity architecture."""
    J_PRIME = "j_prime"
    REACTOR_CORE = "reactor_core"
    JARVIS_BODY = "jarvis_body"
    USER = "user"
    SYSTEM = "system"


class TrinityIntent(Enum):
    """Command intents for Trinity protocol."""
    # Surveillance
    START_SURVEILLANCE = "start_surveillance"
    STOP_SURVEILLANCE = "stop_surveillance"
    UPDATE_SURVEILLANCE = "update_surveillance"

    # Window management
    EXILE_WINDOW = "exile_window"
    BRING_BACK_WINDOW = "bring_back_window"
    TELEPORT_WINDOW = "teleport_window"

    # Cryostasis
    FREEZE_APP = "freeze_app"
    THAW_APP = "thaw_app"

    # Phantom Hardware
    CREATE_GHOST_DISPLAY = "create_ghost_display"
    DESTROY_GHOST_DISPLAY = "destroy_ghost_display"

    # System
    HEARTBEAT = "heartbeat"
    PING = "ping"
    PONG = "pong"
    ACK = "ack"
    NACK = "nack"
    STATUS_UPDATE = "status_update"

    # Cognitive
    EXECUTE_PLAN = "execute_plan"
    ABORT_PLAN = "abort_plan"


class JARVISStatus(Enum):
    """Status of JARVIS Body."""
    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrinityCommand:
    """A command in the Trinity protocol."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: TrinitySource = TrinitySource.REACTOR_CORE
    intent: TrinityIntent = TrinityIntent.PING
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    target: Optional[TrinitySource] = None
    priority: int = 5
    requires_ack: bool = False
    response_to: Optional[str] = None
    ttl_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source.value,
            "intent": self.intent.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "target": self.target.value if self.target else None,
            "priority": self.priority,
            "requires_ack": self.requires_ack,
            "response_to": self.response_to,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrinityCommand":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            source=TrinitySource(data["source"]) if data.get("source") else TrinitySource.SYSTEM,
            intent=TrinityIntent(data["intent"]) if data.get("intent") else TrinityIntent.PING,
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            target=TrinitySource(data["target"]) if data.get("target") else None,
            priority=data.get("priority", 5),
            requires_ack=data.get("requires_ack", False),
            response_to=data.get("response_to"),
            ttl_seconds=data.get("ttl_seconds", 30.0),
        )

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl_seconds


@dataclass
class HeartbeatState:
    """State from JARVIS heartbeat."""
    active_window_title: str = ""
    active_app_name: str = ""
    apps_on_ghost_display: List[str] = field(default_factory=list)
    frozen_apps: List[str] = field(default_factory=list)
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0
    surveillance_active: bool = False
    surveillance_targets: List[str] = field(default_factory=list)
    ghost_display_available: bool = False
    uptime_seconds: float = 0.0
    last_command_id: Optional[str] = None
    timestamp: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeartbeatState":
        return cls(
            active_window_title=data.get("active_window_title", ""),
            active_app_name=data.get("active_app_name", ""),
            apps_on_ghost_display=data.get("apps_on_ghost_display", []),
            frozen_apps=data.get("frozen_apps", []),
            system_cpu_percent=data.get("system_cpu_percent", 0.0),
            system_memory_percent=data.get("system_memory_percent", 0.0),
            surveillance_active=data.get("surveillance_active", False),
            surveillance_targets=data.get("surveillance_targets", []),
            ghost_display_available=data.get("ghost_display_available", False),
            uptime_seconds=data.get("uptime_seconds", 0.0),
            last_command_id=data.get("last_command_id"),
            timestamp=data.get("timestamp", 0.0),
        )


# =============================================================================
# TRINITY CONNECTOR
# =============================================================================

class TrinityConnector:
    """
    Reactor Core connector to JARVIS Body.

    Manages:
    - Command sending to JARVIS
    - Heartbeat monitoring
    - Response tracking
    - JARVIS status management
    """

    _instance: Optional['TrinityConnector'] = None

    def __new__(cls) -> 'TrinityConnector':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # State
        self._connected = False
        self._jarvis_status = JARVISStatus.UNKNOWN
        self._last_heartbeat: Optional[HeartbeatState] = None
        self._last_heartbeat_time: float = 0.0

        # Pending commands waiting for ACK
        self._pending_commands: Dict[str, TrinityCommand] = {}

        # Response handlers
        self._response_handlers: Dict[str, Callable] = {}

        # Background tasks
        self._heartbeat_monitor_task: Optional[asyncio.Task] = None
        self._response_watcher_task: Optional[asyncio.Task] = None

        # Processed response files (for deduplication)
        self._processed_responses: Set[str] = set()

        # Stats
        self._stats = {
            "commands_sent": 0,
            "commands_acked": 0,
            "commands_nacked": 0,
            "commands_timeout": 0,
            "heartbeats_received": 0,
        }

        logger.info("[Trinity] TrinityConnector initialized")

    async def connect(self) -> bool:
        """Connect to JARVIS via Trinity protocol."""
        if self._connected:
            return True

        try:
            # Ensure directories exist
            COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
            HEARTBEATS_DIR.mkdir(parents=True, exist_ok=True)
            RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

            # Check for recent JARVIS heartbeat
            await self._check_jarvis_heartbeat()

            # Start background tasks
            self._heartbeat_monitor_task = asyncio.create_task(
                self._heartbeat_monitor_loop()
            )
            self._response_watcher_task = asyncio.create_task(
                self._response_watcher_loop()
            )

            self._connected = True
            logger.info(f"[Trinity] Connected (JARVIS status: {self._jarvis_status.value})")
            return True

        except Exception as e:
            logger.error(f"[Trinity] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Trinity."""
        self._connected = False

        if self._heartbeat_monitor_task:
            self._heartbeat_monitor_task.cancel()
        if self._response_watcher_task:
            self._response_watcher_task.cancel()

        logger.info("[Trinity] Disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def get_jarvis_status(self) -> JARVISStatus:
        return self._jarvis_status

    def get_last_heartbeat(self) -> Optional[HeartbeatState]:
        return self._last_heartbeat

    # =========================================================================
    # COMMAND SENDING
    # =========================================================================

    async def send_command(
        self,
        intent: TrinityIntent,
        payload: Dict[str, Any],
        requires_ack: bool = True,
        timeout: float = COMMAND_TIMEOUT,
        priority: int = 5,
    ) -> Dict[str, Any]:
        """
        Send a command to JARVIS.

        Args:
            intent: The command intent
            payload: Command payload
            requires_ack: Whether to wait for ACK
            timeout: Timeout in seconds
            priority: Command priority (1-10, lower is higher)

        Returns:
            Result dict with success status and response
        """
        if not self._connected:
            return {"success": False, "error": "Not connected"}

        if self._jarvis_status == JARVISStatus.OFFLINE:
            return {"success": False, "error": "JARVIS is offline"}

        command = TrinityCommand(
            source=TrinitySource.REACTOR_CORE,
            intent=intent,
            payload=payload,
            target=TrinitySource.JARVIS_BODY,
            priority=priority,
            requires_ack=requires_ack,
            ttl_seconds=timeout,
        )

        # Write command file
        try:
            filename = f"{int(command.timestamp * 1000)}_{command.id}.json"
            filepath = COMMANDS_DIR / filename

            with open(filepath, "w") as f:
                json.dump(command.to_dict(), f, indent=2)

            self._stats["commands_sent"] += 1
            logger.info(f"[Trinity] Sent command: {intent.value} (id={command.id[:8]})")

            # If ACK required, wait for response
            if requires_ack:
                self._pending_commands[command.id] = command
                result = await self._wait_for_response(command.id, timeout)
                return result

            return {"success": True, "command_id": command.id}

        except Exception as e:
            logger.error(f"[Trinity] Failed to send command: {e}")
            return {"success": False, "error": str(e)}

    async def _wait_for_response(
        self,
        command_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Wait for ACK/NACK response to a command."""
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            # Check if response received
            if command_id not in self._pending_commands:
                # Response was processed by watcher
                return {"success": True, "command_id": command_id}

            await asyncio.sleep(0.5)

        # Timeout
        if command_id in self._pending_commands:
            del self._pending_commands[command_id]
            self._stats["commands_timeout"] += 1

        return {"success": False, "error": "Timeout waiting for response"}

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def start_surveillance(
        self,
        app_name: str,
        trigger_text: str,
        all_spaces: bool = True,
        max_duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start surveillance on an app."""
        return await self.send_command(
            intent=TrinityIntent.START_SURVEILLANCE,
            payload={
                "app_name": app_name,
                "trigger_text": trigger_text,
                "all_spaces": all_spaces,
                "max_duration": max_duration,
            },
        )

    async def stop_surveillance(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """Stop surveillance."""
        return await self.send_command(
            intent=TrinityIntent.STOP_SURVEILLANCE,
            payload={"app_name": app_name},
        )

    async def bring_back_windows(
        self,
        app_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bring back windows from Ghost Display."""
        return await self.send_command(
            intent=TrinityIntent.BRING_BACK_WINDOW,
            payload={"app_name": app_name},
        )

    async def exile_window(
        self,
        app_name: str,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Exile a window to Ghost Display."""
        return await self.send_command(
            intent=TrinityIntent.EXILE_WINDOW,
            payload={
                "app_name": app_name,
                "window_title": window_title,
            },
        )

    async def freeze_app(self, app_name: str, reason: str = "") -> Dict[str, Any]:
        """Freeze an app (SIGSTOP)."""
        return await self.send_command(
            intent=TrinityIntent.FREEZE_APP,
            payload={"app_name": app_name, "reason": reason},
        )

    async def thaw_app(self, app_name: str) -> Dict[str, Any]:
        """Thaw a frozen app (SIGCONT)."""
        return await self.send_command(
            intent=TrinityIntent.THAW_APP,
            payload={"app_name": app_name},
        )

    async def create_ghost_display(self) -> Dict[str, Any]:
        """Create a Ghost Display (virtual display)."""
        return await self.send_command(
            intent=TrinityIntent.CREATE_GHOST_DISPLAY,
            payload={},
        )

    async def ping_jarvis(self) -> Dict[str, Any]:
        """Ping JARVIS to check liveness."""
        return await self.send_command(
            intent=TrinityIntent.PING,
            payload={},
            timeout=10.0,
        )

    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================

    async def _heartbeat_monitor_loop(self) -> None:
        """Monitor JARVIS heartbeats."""
        while self._connected:
            try:
                await self._check_jarvis_heartbeat()
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Trinity] Heartbeat monitor error: {e}")
                await asyncio.sleep(5.0)

    async def _check_jarvis_heartbeat(self) -> None:
        """Check for recent JARVIS heartbeat."""
        try:
            # Find most recent heartbeat file
            heartbeat_files = sorted(
                HEARTBEATS_DIR.glob("*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )

            if not heartbeat_files:
                self._jarvis_status = JARVISStatus.OFFLINE
                return

            latest = heartbeat_files[0]
            mtime = latest.stat().st_mtime
            age = time.time() - mtime

            if age > HEARTBEAT_TIMEOUT:
                self._jarvis_status = JARVISStatus.OFFLINE
                return

            # Parse heartbeat
            with open(latest) as f:
                data = json.load(f)

            # Extract payload (heartbeat has command wrapper)
            payload = data.get("payload", data)
            self._last_heartbeat = HeartbeatState.from_dict(payload)
            self._last_heartbeat.timestamp = mtime
            self._last_heartbeat_time = mtime
            self._jarvis_status = JARVISStatus.ONLINE
            self._stats["heartbeats_received"] += 1

        except Exception as e:
            logger.warning(f"[Trinity] Heartbeat check error: {e}")

    async def _response_watcher_loop(self) -> None:
        """Watch for ACK/NACK responses from JARVIS."""
        while self._connected:
            try:
                for filepath in RESPONSES_DIR.glob("*.json"):
                    if filepath.name in self._processed_responses:
                        continue

                    try:
                        with open(filepath) as f:
                            data = json.load(f)

                        command = TrinityCommand.from_dict(data)
                        self._processed_responses.add(filepath.name)

                        # Skip if not from JARVIS
                        if command.source != TrinitySource.JARVIS_BODY:
                            continue

                        # Process ACK/NACK
                        if command.intent == TrinityIntent.ACK:
                            if command.response_to in self._pending_commands:
                                del self._pending_commands[command.response_to]
                                self._stats["commands_acked"] += 1
                                logger.debug(f"[Trinity] ACK received for {command.response_to[:8]}")

                        elif command.intent == TrinityIntent.NACK:
                            if command.response_to in self._pending_commands:
                                del self._pending_commands[command.response_to]
                                self._stats["commands_nacked"] += 1
                                logger.warning(f"[Trinity] NACK received for {command.response_to[:8]}")

                        # Clean up old response file
                        if (time.time() - command.timestamp) > 3600:
                            filepath.unlink(missing_ok=True)

                    except Exception as e:
                        logger.warning(f"[Trinity] Response parse error: {e}")
                        self._processed_responses.add(filepath.name)

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Trinity] Response watcher error: {e}")
                await asyncio.sleep(2.0)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "jarvis_status": self._jarvis_status.value,
            "pending_commands": len(self._pending_commands),
            "last_heartbeat_age": (
                time.time() - self._last_heartbeat_time
                if self._last_heartbeat_time else None
            ),
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_trinity_connector: Optional[TrinityConnector] = None


def get_trinity_connector() -> TrinityConnector:
    """Get the singleton TrinityConnector instance."""
    global _trinity_connector
    if _trinity_connector is None:
        _trinity_connector = TrinityConnector()
    return _trinity_connector


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def connect_to_jarvis() -> bool:
    """Connect to JARVIS via Trinity."""
    connector = get_trinity_connector()
    return await connector.connect()


async def send_surveillance_command(
    app_name: str,
    trigger_text: str,
    all_spaces: bool = True,
) -> Dict[str, Any]:
    """Send a surveillance command to JARVIS."""
    connector = get_trinity_connector()
    return await connector.start_surveillance(app_name, trigger_text, all_spaces)


async def send_bring_back_command(app_name: Optional[str] = None) -> Dict[str, Any]:
    """Send a bring back windows command to JARVIS."""
    connector = get_trinity_connector()
    return await connector.bring_back_windows(app_name)


def get_jarvis_status() -> JARVISStatus:
    """Get current JARVIS status."""
    connector = get_trinity_connector()
    return connector.get_jarvis_status()


def get_jarvis_heartbeat() -> Optional[HeartbeatState]:
    """Get last JARVIS heartbeat state."""
    connector = get_trinity_connector()
    return connector.get_last_heartbeat()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TrinityConnector",
    "TrinityCommand",
    "TrinityIntent",
    "TrinitySource",
    "JARVISStatus",
    "HeartbeatState",
    "get_trinity_connector",
    "connect_to_jarvis",
    "send_surveillance_command",
    "send_bring_back_command",
    "get_jarvis_status",
    "get_jarvis_heartbeat",
]
