"""
JARVIS Prime Connector for cross-system experience ingestion.

Provides:
- Real-time connection to JARVIS Prime via WebSocket
- REST API fallback for batch queries
- Interaction history retrieval
- Event streaming for live learning
- Cross-system event correlation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)


def _parse_path_list(value: str, default_paths: List[str]) -> List[str]:
    """Parse comma-separated endpoint paths and normalize to '/path' format."""
    raw_items = value.split(",") if value else list(default_paths)
    normalized: List[str] = []
    seen: set = set()
    for item in raw_items:
        path = item.strip()
        if not path:
            continue
        if "://" in path:
            # Path list is path-only; ignore fully qualified URLs.
            continue
        if not path.startswith("/"):
            path = f"/{path}"
        if path in seen:
            continue
        normalized.append(path)
        seen.add(path)
    return normalized or list(default_paths)


class PrimeEventType(Enum):
    """Types of events from JARVIS Prime."""
    INTERACTION = "interaction"
    COMMAND = "command"
    RESPONSE = "response"
    CORRECTION = "correction"
    FEEDBACK = "feedback"
    SYSTEM = "system"
    ERROR = "error"
    HEALTH = "health"


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class PrimeEvent:
    """An event from JARVIS Prime."""
    event_id: str
    event_type: PrimeEventType
    timestamp: datetime

    # Interaction data
    user_input: str = ""
    assistant_response: str = ""
    system_context: str = ""

    # Quality signals
    success: bool = True
    confidence: float = 1.0
    latency_ms: float = 0.0

    # Correction data
    is_correction: bool = False
    original_response: str = ""
    corrected_response: str = ""

    # Metadata
    session_id: str = ""
    model_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "assistant_response": self.assistant_response,
            "system_context": self.system_context,
            "success": self.success,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "is_correction": self.is_correction,
            "original_response": self.original_response,
            "corrected_response": self.corrected_response,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimeEvent":
        raw_type = str(data.get("event_type", "interaction")).lower()
        try:
            event_type = PrimeEventType(raw_type)
        except ValueError:
            # Normalize Prime websocket envelope types to connector semantics.
            if raw_type in {"heartbeat", "health", "health_check"}:
                event_type = PrimeEventType.HEALTH
            elif raw_type in {"response", "inference_complete"}:
                event_type = PrimeEventType.RESPONSE
            elif raw_type in {"command", "inference_request"}:
                event_type = PrimeEventType.COMMAND
            else:
                event_type = PrimeEventType.SYSTEM

        timestamp_value: Any = data.get("timestamp", datetime.now().isoformat())
        if isinstance(timestamp_value, str):
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp_value)
            except ValueError:
                parsed_timestamp = datetime.now()
        elif isinstance(timestamp_value, datetime):
            parsed_timestamp = timestamp_value
        else:
            parsed_timestamp = datetime.now()

        payload = data.get("data", {})

        metadata: Dict[str, Any]
        raw_metadata = data.get("metadata", {})
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        else:
            metadata = {"value": raw_metadata}
        if payload is not None:
            metadata["payload"] = payload

        success = data.get("success")
        if success is None and isinstance(payload, dict):
            payload_status = str(payload.get("status", "")).lower()
            if payload_status:
                success = payload_status in {"healthy", "ready", "ok", "starting"}
        if success is None:
            success = True

        return cls(
            event_id=str(data.get("event_id", f"evt_{int(parsed_timestamp.timestamp() * 1000)}")),
            event_type=event_type,
            timestamp=parsed_timestamp,
            user_input=data.get("user_input", ""),
            assistant_response=data.get("assistant_response", ""),
            system_context=data.get("system_context", ""),
            success=bool(success),
            confidence=data.get("confidence", 1.0),
            latency_ms=data.get("latency_ms", 0.0),
            is_correction=data.get("is_correction", False),
            original_response=data.get("original_response", ""),
            corrected_response=data.get("corrected_response", ""),
            session_id=data.get("session_id", ""),
            model_id=data.get("model_id", ""),
            metadata=metadata,
        )


@dataclass
class PrimeConnectorConfig:
    """Configuration for JARVIS Prime connector."""
    # Connection settings
    host: str = field(
        default_factory=lambda: os.getenv("JARVIS_PRIME_HOST", "localhost")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_PRIME_PORT", "8002"))
    )
    use_ssl: bool = field(
        default_factory=lambda: os.getenv("JARVIS_PRIME_SSL", "false").lower() == "true"
    )

    # API settings
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("JARVIS_PRIME_API_KEY")
    )
    api_version: str = "v1"

    # WebSocket settings
    enable_websocket: bool = True
    websocket_path: str = "/ws/events"
    websocket_paths: List[str] = field(
        default_factory=lambda: _parse_path_list(
            os.getenv("JARVIS_PRIME_WEBSOCKET_PATHS", ""),
            ["/ws/events"],
        )
    )
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    ping_interval: float = 30.0

    # Health polling fallback settings
    health_paths: List[str] = field(
        default_factory=lambda: _parse_path_list(
            os.getenv("JARVIS_PRIME_HEALTH_PATHS", ""),
            ["/health"],
        )
    )
    enable_health_poll_fallback: bool = field(
        default_factory=lambda: os.getenv("PRIME_HEALTH_POLL_FALLBACK", "true").lower() == "true"
    )
    health_poll_interval: float = field(
        default_factory=lambda: float(os.getenv("PRIME_HEALTH_POLL_INTERVAL", "5.0"))
    )

    # Request settings
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

    # Filtering
    event_types: List[PrimeEventType] = field(
        default_factory=lambda: list(PrimeEventType)
    )
    min_confidence: float = 0.0

    @property
    def base_url(self) -> str:
        protocol = "https" if self.use_ssl else "http"
        return f"{protocol}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.host}:{self.port}{self.websocket_path}"

    @property
    def ws_urls(self) -> List[str]:
        protocol = "wss" if self.use_ssl else "ws"
        return [f"{protocol}://{self.host}:{self.port}{path}" for path in self.websocket_paths]


class PrimeConnector:
    """
    Connects to JARVIS Prime for experience ingestion.

    Supports both REST API and WebSocket streaming for
    real-time event collection.
    """

    def __init__(
        self,
        config: Optional[PrimeConnectorConfig] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize the Prime connector.

        Args:
            config: Full configuration object
            host: Override host (convenience parameter)
            port: Override port (convenience parameter)
        """
        self.config = config or PrimeConnectorConfig()

        if host:
            self.config.host = host
        if port:
            self.config.port = port

        # Keep backwards compatibility with single-path configuration.
        if self.config.websocket_path and self.config.websocket_path not in self.config.websocket_paths:
            self.config.websocket_paths.insert(0, self.config.websocket_path)

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_state = ConnectionState.DISCONNECTED
        self._ws_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._callbacks: List[Callable[[PrimeEvent], None]] = []
        self._reconnect_count = 0
        self._ws_path_index = 0

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws_state == ConnectionState.CONNECTED

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
        return self._session

    async def check_health(self) -> Dict[str, Any]:
        """
        Check JARVIS Prime health status.

        Returns:
            Health status dict
        """
        session = await self._get_session()

        last_error: Optional[Exception] = None
        for health_path in self.config.health_paths:
            try:
                url = urljoin(self.config.base_url, health_path)
                async with session.get(url) as response:
                    # 503 often means "starting" (service alive, not ready yet).
                    if response.status in (200, 503):
                        payload = await response.json(content_type=None)
                        if isinstance(payload, dict):
                            payload.setdefault("_health_path", health_path)
                            payload.setdefault("_http_status", response.status)
                            return payload
                        return {
                            "status": "healthy" if response.status == 200 else "starting",
                            "value": payload,
                            "_health_path": health_path,
                            "_http_status": response.status,
                        }
                    last_error = RuntimeError(f"HTTP {response.status} on {health_path}")
            except Exception as e:
                last_error = e

        if last_error is not None:
            return {"status": "error", "error": str(last_error)}
        return {"status": "error", "error": "No health endpoints configured"}

    async def get_recent_interactions(
        self,
        hours: int = 168,
        limit: int = 1000,
        include_corrections: bool = True,
    ) -> List[PrimeEvent]:
        """
        Get recent interactions from JARVIS Prime.

        Args:
            hours: Lookback period in hours
            limit: Maximum events to return
            include_corrections: Include correction events

        Returns:
            List of PrimeEvent objects
        """
        session = await self._get_session()

        params = {
            "since": (datetime.now() - timedelta(hours=hours)).isoformat(),
            "limit": limit,
            "include_corrections": str(include_corrections).lower(),
        }

        try:
            url = urljoin(self.config.base_url, f"/api/{self.config.api_version}/interactions")
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        PrimeEvent.from_dict(item)
                        for item in data.get("interactions", [])
                    ]
                else:
                    logger.error(f"Failed to get interactions: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching interactions: {e}")
            return []

    async def get_corrections(
        self,
        hours: int = 168,
        limit: int = 500,
    ) -> List[PrimeEvent]:
        """Get correction events only."""
        session = await self._get_session()

        params = {
            "since": (datetime.now() - timedelta(hours=hours)).isoformat(),
            "limit": limit,
        }

        try:
            url = urljoin(self.config.base_url, f"/api/{self.config.api_version}/corrections")
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        PrimeEvent.from_dict(item)
                        for item in data.get("corrections", [])
                    ]
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching corrections: {e}")
            return []

    async def get_session_history(
        self,
        session_id: str,
    ) -> List[PrimeEvent]:
        """Get all events from a specific session."""
        session = await self._get_session()

        try:
            url = urljoin(
                self.config.base_url,
                f"/api/{self.config.api_version}/sessions/{session_id}/events"
            )
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        PrimeEvent.from_dict(item)
                        for item in data.get("events", [])
                    ]
                else:
                    return []
        except Exception as e:
            logger.error(f"Error fetching session history: {e}")
            return []

    @staticmethod
    def _is_ws_contract_error(exc: Exception) -> bool:
        """Return True when the error suggests endpoint mismatch/forbidden route."""
        msg = str(exc).lower()
        exc_type = type(exc).__name__
        return (
            "404" in msg
            or "403" in msg
            or "not found" in msg
            or "forbidden" in msg
            or exc_type in ("WSServerHandshakeError", "InvalidStatusCode", "InvalidStatus")
        )

    def _rotated_ws_urls(self) -> List[str]:
        base_urls = self.config.ws_urls
        if not base_urls:
            return []
        idx = self._ws_path_index % len(base_urls)
        return base_urls[idx:] + base_urls[:idx]

    async def _build_health_event(self) -> Optional[PrimeEvent]:
        """Create a synthetic HEALTH event from Prime's health endpoint."""
        health = await self.check_health()
        status = str(health.get("status", "unknown")).lower()
        success = status in ("healthy", "ready", "ok", "starting")
        return PrimeEvent(
            event_id=f"health_{int(datetime.now().timestamp() * 1000)}",
            event_type=PrimeEventType.HEALTH,
            timestamp=datetime.now(),
            success=success,
            confidence=1.0 if success else 0.0,
            metadata={"health": health},
        )

    async def connect_websocket(self) -> None:
        """
        Connect to JARVIS Prime WebSocket for real-time events.
        """
        if not self.config.enable_websocket:
            raise RuntimeError("WebSocket is disabled in configuration")

        self._ws_state = ConnectionState.CONNECTING
        session = await self._get_session()

        last_error: Optional[Exception] = None
        for candidate_url in self._rotated_ws_urls():
            try:
                self._ws = await session.ws_connect(
                    candidate_url,
                    heartbeat=self.config.ping_interval,
                )
                self._ws_state = ConnectionState.CONNECTED
                self._reconnect_count = 0

                # Persist chosen path for next connection.
                for idx, known_url in enumerate(self.config.ws_urls):
                    if known_url == candidate_url:
                        self._ws_path_index = idx
                        self.config.websocket_path = self.config.websocket_paths[idx]
                        break

                logger.info(f"Connected to JARVIS Prime WebSocket at {candidate_url}")

                # Start listening
                self._ws_task = asyncio.create_task(self._ws_listener())
                return
            except Exception as e:
                last_error = e
                if self._is_ws_contract_error(e):
                    self._ws_path_index += 1
                continue

        self._ws_state = ConnectionState.FAILED
        logger.error(f"Failed to connect WebSocket using all configured endpoints: {last_error}")
        if last_error is not None:
            raise last_error
        raise RuntimeError("No websocket endpoints configured")

    async def _ws_listener(self) -> None:
        """Listen for WebSocket messages."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        event = PrimeEvent.from_dict(data)

                        # Filter by event type
                        if event.event_type not in self.config.event_types:
                            continue

                        # Filter by confidence
                        if event.confidence < self.config.min_confidence:
                            continue

                        # Queue event
                        await self._event_queue.put(event)

                        # Notify callbacks
                        for callback in self._callbacks:
                            try:
                                callback(event)
                            except Exception as e:
                                logger.warning(f"Callback error: {e}")

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from WebSocket: {msg.data[:100]}")
                    except Exception as e:
                        logger.warning(f"Error processing WebSocket message: {e}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break

        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")

        finally:
            self._ws_state = ConnectionState.DISCONNECTED
            await self._handle_disconnect()

    async def _handle_disconnect(self) -> None:
        """Handle WebSocket disconnection with reconnection logic."""
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            self._ws_state = ConnectionState.FAILED
            logger.error("Max reconnection attempts reached")
            return

        self._ws_state = ConnectionState.RECONNECTING
        self._reconnect_count += 1

        logger.info(
            f"Reconnecting to WebSocket (attempt {self._reconnect_count}/"
            f"{self.config.max_reconnect_attempts})..."
        )

        await asyncio.sleep(self.config.reconnect_interval)

        try:
            await self.connect_websocket()
        except Exception as e:
            logger.warning(f"Reconnection failed: {e}")
            await self._handle_disconnect()

    async def stream_events(self) -> AsyncIterator[PrimeEvent]:
        """
        Stream events from WebSocket.

        Yields:
            PrimeEvent objects as they arrive
        """
        connector_error: Optional[Exception] = None

        if self.config.enable_websocket:
            try:
                if self._ws_state != ConnectionState.CONNECTED:
                    await self.connect_websocket()

                while self._ws_state in (ConnectionState.CONNECTED, ConnectionState.RECONNECTING):
                    try:
                        event = await asyncio.wait_for(
                            self._event_queue.get(),
                            timeout=max(5.0, self.config.health_poll_interval),
                        )
                        yield event
                    except asyncio.TimeoutError:
                        # Keep stream alive and optionally emit health snapshots when idle.
                        if self.config.enable_health_poll_fallback:
                            health_event = await self._build_health_event()
                            if health_event:
                                yield health_event
                        continue
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                connector_error = exc
                logger.warning(
                    "WebSocket event stream unavailable; switching to health-poll fallback: %s",
                    exc,
                )

        if not self.config.enable_health_poll_fallback:
            if connector_error is not None:
                raise connector_error
            return

        # Deterministic fallback stream.
        failures = 0
        while True:
            try:
                health_event = await self._build_health_event()
                if health_event is not None:
                    failures = 0
                    yield health_event
                else:
                    failures += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                if failures <= 3 or failures % 10 == 0:
                    logger.debug("Health fallback poll failed: %s", exc)

            interval = max(0.5, self.config.health_poll_interval)
            backoff = min(interval * (1.0 + 0.25 * min(failures, 8)), interval * 4.0)
            await asyncio.sleep(backoff)

    def add_event_callback(
        self,
        callback: Callable[[PrimeEvent], None],
    ) -> None:
        """Add a callback for real-time events."""
        self._callbacks.append(callback)

    def remove_event_callback(
        self,
        callback: Callable[[PrimeEvent], None],
    ) -> None:
        """Remove an event callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def disconnect(self) -> None:
        """Disconnect WebSocket and cleanup."""
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass

        if self._ws and not self._ws.closed:
            await self._ws.close()

        self._ws_state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from JARVIS Prime")

    async def close(self) -> None:
        """Close all connections."""
        await self.disconnect()

        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "PrimeConnector":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


# Convenience exports
__all__ = [
    "PrimeConnector",
    "PrimeConnectorConfig",
    "PrimeEvent",
    "PrimeEventType",
    "ConnectionState",
]
