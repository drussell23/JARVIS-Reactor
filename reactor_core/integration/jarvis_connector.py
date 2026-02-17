"""
JARVIS-AI-Agent connector for experience ingestion.

Provides:
- Real-time log streaming from JARVIS
- Experience event parsing
- Correction detection and extraction
- Context memory integration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of JARVIS events."""
    COMMAND = "command"
    RESPONSE = "response"
    CORRECTION = "correction"
    FEEDBACK = "feedback"
    SYSTEM_ACTION = "system_action"
    ERROR = "error"
    CONTEXT_UPDATE = "context_update"
    AUTHENTICATION = "authentication"


class CorrectionType(Enum):
    """Types of user corrections."""
    EXPLICIT = "explicit"      # User explicitly corrected JARVIS
    IMPLICIT = "implicit"      # User rephrased and JARVIS understood
    COMMAND_RETRY = "retry"    # User retried with different command
    EDIT = "edit"              # User edited JARVIS output


@dataclass
class JARVISEvent:
    """An event from the JARVIS system."""
    event_id: str
    event_type: EventType
    timestamp: datetime

    # Interaction data
    user_input: str = ""
    jarvis_response: str = ""
    system_context: str = ""

    # Outcome
    success: bool = True
    confidence: float = 1.0
    latency_ms: float = 0.0

    # Correction data
    is_correction: bool = False
    correction_type: Optional[CorrectionType] = None
    original_response: str = ""
    corrected_response: str = ""

    # Metadata
    session_id: str = ""
    component: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "jarvis_response": self.jarvis_response,
            "system_context": self.system_context,
            "success": self.success,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "is_correction": self.is_correction,
            "correction_type": self.correction_type.value if self.correction_type else None,
            "original_response": self.original_response,
            "corrected_response": self.corrected_response,
            "session_id": self.session_id,
            "component": self.component,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JARVISEvent":
        correction_type = None
        if data.get("correction_type"):
            correction_type = CorrectionType(data["correction_type"])

        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_input=data.get("user_input", ""),
            jarvis_response=data.get("jarvis_response", ""),
            system_context=data.get("system_context", ""),
            success=data.get("success", True),
            confidence=data.get("confidence", 1.0),
            latency_ms=data.get("latency_ms", 0.0),
            is_correction=data.get("is_correction", False),
            correction_type=correction_type,
            original_response=data.get("original_response", ""),
            corrected_response=data.get("corrected_response", ""),
            session_id=data.get("session_id", ""),
            component=data.get("component", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class JARVISConnectorConfig:
    """Configuration for JARVIS connector."""
    # Paths
    jarvis_repo_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "JARVIS_REPO_PATH",
                "~/Documents/repos/JARVIS-AI-Agent"
            )
        ).expanduser()
    )

    log_directory: str = "backend/logs"
    state_file: str = "backend/data/autonomous_engine_state.json"

    # Event filtering
    include_event_types: List[EventType] = field(
        default_factory=lambda: list(EventType)
    )
    min_confidence: float = 0.0  # Include all by default
    only_corrections: bool = False

    # Time window
    lookback_hours: int = 168  # One week by default

    # Real-time streaming
    enable_file_watching: bool = False
    watch_debounce_seconds: float = 1.0


class LogFileHandler(FileSystemEventHandler):
    """Watches log files for changes."""

    def __init__(
        self,
        callback: Callable[[str], None],
        debounce_seconds: float = 1.0,
    ):
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._last_modified: Dict[str, datetime] = {}

    def on_modified(self, event):
        if not isinstance(event, FileModifiedEvent):
            return

        if not event.src_path.endswith(".log") and not event.src_path.endswith(".jsonl"):
            return

        now = datetime.now()
        last = self._last_modified.get(event.src_path)

        if last and (now - last).total_seconds() < self.debounce_seconds:
            return

        self._last_modified[event.src_path] = now
        self.callback(event.src_path)


class JARVISConnector:
    """
    Connects to JARVIS-AI-Agent for experience ingestion.

    Reads logs, parses events, and detects corrections
    for training data generation.
    """

    def __init__(
        self,
        config: Optional[JARVISConnectorConfig] = None,
    ):
        self.config = config or JARVISConnectorConfig()

        # Validate paths
        if not self.config.jarvis_repo_path.exists():
            logger.warning(
                f"JARVIS repo not found at {self.config.jarvis_repo_path}"
            )

        self._observer: Optional[Observer] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()

        # v258.0: Store event loop reference for thread-safe task scheduling
        # (watchdog Observer callbacks run in a separate thread)
        try:
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = None

    @property
    def log_path(self) -> Path:
        """Get full path to log directory."""
        return self.config.jarvis_repo_path / self.config.log_directory

    @property
    def state_path(self) -> Path:
        """Get full path to state file."""
        return self.config.jarvis_repo_path / self.config.state_file

    async def get_events(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[JARVISEvent]:
        """
        Get JARVIS events from logs.

        Args:
            since: Start time (default: lookback_hours ago)
            until: End time (default: now)
            limit: Maximum events to return

        Returns:
            List of JARVISEvent objects
        """
        if since is None:
            since = datetime.now() - timedelta(hours=self.config.lookback_hours)
        if until is None:
            until = datetime.now()

        events = []

        # Scan log files
        if self.log_path.exists():
            log_files = list(self.log_path.glob("*.log")) + \
                        list(self.log_path.glob("*.jsonl"))

            for log_file in sorted(log_files, reverse=True):
                file_events = await self._parse_log_file(log_file, since, until)
                events.extend(file_events)

                if len(events) >= limit:
                    break

        # Apply filters
        filtered = self._filter_events(events)

        return filtered[:limit]

    async def get_corrections(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[JARVISEvent]:
        """Get correction events only."""
        events = await self.get_events(since, until)
        return [e for e in events if e.is_correction]

    async def get_successful_interactions(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        min_confidence: float = 0.85,
    ) -> List[JARVISEvent]:
        """Get high-confidence successful interactions."""
        events = await self.get_events(since, until)
        return [
            e for e in events
            if e.success and e.confidence >= min_confidence
        ]

    async def _parse_log_file(
        self,
        file_path: Path,
        since: datetime,
        until: datetime,
    ) -> List[JARVISEvent]:
        """Parse a log file for events."""
        events = []

        try:
            content = file_path.read_text()

            # Try to parse as JSONL
            if file_path.suffix == ".jsonl":
                for line in content.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        event = self._parse_json_event(data)
                        if event and since <= event.timestamp <= until:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue

            # Parse as structured log
            else:
                events.extend(self._parse_structured_log(content, since, until))

        except Exception as e:
            logger.error(f"Error parsing log file {file_path}: {e}")

        return events

    def _parse_json_event(self, data: Dict[str, Any]) -> Optional[JARVISEvent]:
        """Parse a JSON log entry into a JARVISEvent."""
        try:
            # Handle different log formats
            timestamp = None
            if "timestamp" in data:
                timestamp = datetime.fromisoformat(
                    data["timestamp"].replace("Z", "+00:00")
                )
            elif "time" in data:
                timestamp = datetime.fromisoformat(
                    data["time"].replace("Z", "+00:00")
                )
            else:
                timestamp = datetime.now()

            # Detect event type
            event_type = EventType.COMMAND
            if "event_type" in data:
                try:
                    event_type = EventType(data["event_type"])
                except ValueError:
                    pass
            elif data.get("type") == "correction":
                event_type = EventType.CORRECTION
            elif data.get("type") == "error":
                event_type = EventType.ERROR

            # Detect corrections
            is_correction = False
            correction_type = None

            if data.get("is_correction") or data.get("type") == "correction":
                is_correction = True
                if data.get("correction_type"):
                    correction_type = CorrectionType(data["correction_type"])
                else:
                    correction_type = CorrectionType.EXPLICIT

            # Build event
            event_id = data.get("event_id") or data.get("id") or \
                       f"{timestamp.isoformat()}-{hash(str(data)) % 10000}"

            return JARVISEvent(
                event_id=str(event_id),
                event_type=event_type,
                timestamp=timestamp,
                user_input=data.get("user_input") or data.get("query") or "",
                jarvis_response=data.get("jarvis_response") or data.get("response") or "",
                system_context=data.get("system_context") or data.get("context") or "",
                success=data.get("success", True),
                confidence=data.get("confidence", 1.0),
                latency_ms=data.get("latency_ms") or data.get("duration_ms", 0.0),
                is_correction=is_correction,
                correction_type=correction_type,
                original_response=data.get("original_response", ""),
                corrected_response=data.get("corrected_response", ""),
                session_id=data.get("session_id", ""),
                component=data.get("component") or data.get("source", ""),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.debug(f"Failed to parse JSON event: {e}")
            return None

    def _parse_structured_log(
        self,
        content: str,
        since: datetime,
        until: datetime,
    ) -> List[JARVISEvent]:
        """Parse a structured text log."""
        events = []
        import re

        # Common log patterns
        patterns = [
            # ISO timestamp with level
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*(?:Z|[+-]\d{2}:?\d{2})?)\s+'
            r'(?P<level>\w+)\s+'
            r'(?P<component>[\w.]+)\s*[-:]\s*'
            r'(?P<message>.*)',

            # Simple timestamp
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
            r'(?P<message>.*)',
        ]

        for line in content.split("\n"):
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    try:
                        timestamp_str = match.group("timestamp")
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00").replace(" ", "T")
                        )

                        if since <= timestamp <= until:
                            message = match.group("message")
                            component = match.group("component") if "component" in match.groupdict() else ""

                            # Detect corrections in message
                            is_correction = any(
                                kw in message.lower()
                                for kw in ["correction", "corrected", "retry", "redo"]
                            )

                            events.append(JARVISEvent(
                                event_id=f"{timestamp.isoformat()}-{hash(message) % 10000}",
                                event_type=EventType.CORRECTION if is_correction else EventType.COMMAND,
                                timestamp=timestamp,
                                jarvis_response=message,
                                is_correction=is_correction,
                                component=component,
                            ))

                    except Exception:
                        continue
                    break

        return events

    def _filter_events(self, events: List[JARVISEvent]) -> List[JARVISEvent]:
        """Apply configured filters to events."""
        filtered = []

        for event in events:
            # Filter by event type
            if event.event_type not in self.config.include_event_types:
                continue

            # Filter by confidence
            if event.confidence < self.config.min_confidence:
                continue

            # Filter corrections only
            if self.config.only_corrections and not event.is_correction:
                continue

            filtered.append(event)

        return filtered

    async def stream_events(self) -> AsyncIterator[JARVISEvent]:
        """
        Stream events in real-time (requires file watching).

        Yields:
            New JARVISEvent objects as they occur
        """
        if not self.config.enable_file_watching:
            raise RuntimeError(
                "File watching not enabled. Set enable_file_watching=True"
            )

        # Start file watcher if not running
        if self._observer is None:
            self._start_file_watcher()

        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=60.0,
                )
                yield event
            except asyncio.TimeoutError:
                continue

    def _start_file_watcher(self) -> None:
        """Start watching log files for changes."""
        # v258.0: Capture the running event loop for thread-safe scheduling.
        # watchdog Observer runs callbacks in a background thread where
        # asyncio.create_task() would fail (no running event loop).
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = None

        def on_file_change(file_path: str):
            # v258.0: Use thread-safe scheduling since watchdog runs in a thread
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_file_change(file_path),
                    self._loop,
                )
            else:
                logger.debug(
                    f"Cannot process file change for {file_path}: no running event loop"
                )

        handler = LogFileHandler(
            on_file_change,
            self.config.watch_debounce_seconds,
        )

        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(self.log_path),
            recursive=False,
        )
        self._observer.start()
        logger.info(f"Started watching {self.log_path}")

    async def _process_file_change(self, file_path: str) -> None:
        """Process a file change event."""
        events = await self._parse_log_file(
            Path(file_path),
            datetime.now() - timedelta(minutes=5),
            datetime.now(),
        )

        for event in events:
            await self._event_queue.put(event)

    def stop_watching(self) -> None:
        """Stop file watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

    async def get_context_state(self) -> Dict[str, Any]:
        """Get current JARVIS context state."""
        if not self.state_path.exists():
            return {}

        try:
            content = self.state_path.read_text()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error reading state file: {e}")
            return {}


# Convenience exports
__all__ = [
    "JARVISConnector",
    "JARVISConnectorConfig",
    "JARVISEvent",
    "EventType",
    "CorrectionType",
]
