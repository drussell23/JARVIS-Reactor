"""
Trinity Experience Receiver v1.0
================================

Receives experiences from JARVIS and feeds them to the training pipeline.

This module CLOSES THE TRINITY LOOP by:
1. Subscribing to Trinity event bus for experience_batch events
2. Watching ~/.jarvis/trinity/events/ for experience files (file fallback)
3. Calling UnifiedTrainingPipeline.add_experiences() when experiences arrive

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

Author: Trinity System
Version: 1.0.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("TrinityExperienceReceiver")

# Configuration (environment-driven, no hardcoding)
TRINITY_EVENTS_DIR = Path(os.getenv(
    "TRINITY_EVENTS_DIR",
    str(Path.home() / ".jarvis" / "trinity" / "events")
))
JARVIS_EVENTS_DIR = Path(os.getenv(
    "JARVIS_EVENTS_DIR",
    str(Path.home() / ".jarvis" / "events")
))
EXPERIENCE_QUEUE_DIR = Path(os.getenv(
    "EXPERIENCE_QUEUE_DIR",
    str(Path.home() / ".jarvis" / "experience_queue")
))

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
    """Metrics for the experience receiver."""
    events_received: int = 0
    experiences_ingested: int = 0
    files_processed: int = 0
    errors: int = 0
    last_event_time: Optional[float] = None
    last_flush_time: Optional[float] = None


class TrinityExperienceReceiver:
    """
    Receives experiences from JARVIS and feeds them to the training pipeline.

    Features:
    - Multi-directory watching (trinity/events, jarvis/events, experience_queue)
    - Event bus subscription (when available)
    - Automatic batching with flush threshold
    - Deduplication via event ID tracking
    - Graceful error handling with retry
    """

    def __init__(
        self,
        flush_callback: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    ):
        self.logger = logging.getLogger("TrinityExperienceReceiver")

        # Callback to flush experiences to training
        self._flush_callback = flush_callback

        # Experience buffer
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()

        # Deduplication
        self._seen_event_ids: Set[str] = set()
        self._seen_ids_queue: deque[str] = deque(maxlen=10000)

        # Metrics
        self._metrics = ReceiverMetrics()

        # State
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._event_bridge: Optional[Any] = None  # EventBridge instance (if available)

        # Ensure directories exist
        for dir_path in [TRINITY_EVENTS_DIR, JARVIS_EVENTS_DIR, EXPERIENCE_QUEUE_DIR]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.debug(f"Could not create {dir_path}: {e}")

    async def start(self) -> bool:
        """Start the experience receiver."""
        if self._running:
            return True

        self._running = True
        self.logger.info("TrinityExperienceReceiver starting...")

        # Start file watcher
        self._watch_task = asyncio.create_task(
            self._watch_loop(),
            name="trinity_experience_watch_loop"
        )

        # Try to subscribe to event bus
        await self._subscribe_to_event_bus()

        self.logger.info(
            f"TrinityExperienceReceiver ready "
            f"(watching: {TRINITY_EVENTS_DIR}, {JARVIS_EVENTS_DIR})"
        )
        return True

    async def stop(self) -> None:
        """Stop the receiver and flush remaining experiences."""
        self._running = False

        # Final flush
        await self._flush_buffer(force=True)

        # Cancel watch task
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        # Stop event bridge if running
        if self._event_bridge:
            try:
                await self._event_bridge.stop()
            except Exception as e:
                self.logger.debug(f"Error stopping event bridge: {e}")
            self._event_bridge = None

        self.logger.info("TrinityExperienceReceiver stopped")

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
            # Read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, file_path.read_text)
            event = json.loads(content)

            # Check if this is an experience-related event
            event_type = event.get("event_type", event.get("type", ""))

            if event_type in EXPERIENCE_EVENT_TYPES:
                await self._process_event(event)

                # Delete processed file
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
        """Process an experience event and add to buffer."""
        event_id = event.get("event_id", event.get("id", ""))

        # Deduplication check
        if event_id and event_id in self._seen_event_ids:
            return

        # Track seen IDs
        if event_id:
            if len(self._seen_ids_queue) >= 10000:
                oldest = self._seen_ids_queue.popleft()
                self._seen_event_ids.discard(oldest)
            self._seen_ids_queue.append(event_id)
            self._seen_event_ids.add(event_id)

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
            source = payload if "user_input" in payload else event
            experiences = [{
                "user_input": source.get("user_input", ""),
                "assistant_output": source.get("assistant_output", source.get("response", "")),
                "confidence": source.get("confidence", 1.0),
                "feedback_type": source.get("feedback_type", "implicit"),
                "timestamp": source.get("timestamp", time.time()),
            }]

        if not experiences:
            return

        # Add to buffer
        async with self._buffer_lock:
            for exp in experiences:
                # Normalize experience format
                normalized = {
                    "user_input": exp.get("user_input", exp.get("input", {}).get("query", "")),
                    "assistant_output": exp.get("assistant_output", exp.get("output", {}).get("response", "")),
                    "confidence": exp.get("confidence", 1.0),
                    "feedback_type": exp.get("feedback_type", "implicit"),
                    "timestamp": exp.get("timestamp", time.time()),
                    "metadata": exp.get("metadata", {}),
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
        """Flush buffered experiences to training pipeline."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            if not force and len(self._buffer) < FLUSH_THRESHOLD:
                return

            # Move buffer to local
            experiences = self._buffer.copy()
            self._buffer.clear()

        self.logger.info(f"Flushing {len(experiences)} experiences to training pipeline")

        try:
            # Use callback if provided
            if self._flush_callback:
                result = self._flush_callback(experiences)
                if asyncio.iscoroutine(result):
                    await result
            else:
                # Default: use unified training pipeline
                from reactor_core.training.unified_pipeline import get_unified_trainer

                trainer = get_unified_trainer()
                await trainer.add_experiences(experiences, flush=True)

            self._metrics.last_flush_time = time.time()
            self.logger.info(f"Successfully flushed {len(experiences)} experiences")

        except Exception as e:
            self.logger.error(f"Flush failed: {e}")
            self._metrics.errors += 1

            # Re-add experiences to buffer for retry
            async with self._buffer_lock:
                self._buffer.extend(experiences)

    def get_metrics(self) -> Dict[str, Any]:
        """Get receiver metrics."""
        return {
            "events_received": self._metrics.events_received,
            "experiences_ingested": self._metrics.experiences_ingested,
            "files_processed": self._metrics.files_processed,
            "errors": self._metrics.errors,
            "buffer_size": len(self._buffer),
            "last_event_time": self._metrics.last_event_time,
            "last_flush_time": self._metrics.last_flush_time,
            "running": self._running,
        }


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
    "TrinityExperienceReceiver",
    "ReceiverMetrics",
    "get_experience_receiver",
    "shutdown_experience_receiver",
]
