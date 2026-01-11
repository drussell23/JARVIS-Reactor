"""
Trinity Event Publisher for Reactor-Core v1.0
==============================================

This module enables Reactor-Core to publish events to the Trinity ecosystem.
It writes events to the shared event directory that the Trinity Bridge Adapter watches.

THE LOOP:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  Reactor trains model                                                   │
    │         │                                                               │
    │         ▼                                                               │
    │  trinity_publisher.publish_training_complete()  ◄── YOU ARE HERE        │
    │         │                                                               │
    │         ▼                                                               │
    │  Writes to ~/.jarvis/reactor/events/                                    │
    │         │                                                               │
    │         ▼                                                               │
    │  TrinityBridgeAdapter picks up event                                    │
    │         │                                                               │
    │         ▼                                                               │
    │  Prime's TrinityOrchestrator receives MODEL_READY                       │
    │         │                                                               │
    │         ▼                                                               │
    │  _trigger_model_hot_swap() activates new model                         │
    │         │                                                               │
    │         ▼                                                               │
    │  JARVIS routes requests to new model                                   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    from reactor_core.integration.trinity_publisher import (
        publish_training_complete,
        publish_training_started,
        publish_model_ready,
    )

    # At training start
    await publish_training_started(
        model_name="prime-v2",
        config={"epochs": 3, "lr": 2e-5},
    )

    # At training end
    await publish_training_complete(
        model_name="prime-v2",
        model_path="/models/prime-v2",
        metrics={"loss": 0.023, "accuracy": 0.95},
    )

    # When model is validated and ready
    await publish_model_ready(
        model_name="prime-v2",
        model_path="/models/prime-v2/gguf/prime-v2-Q4_K_M.gguf",
        capabilities=["text_generation", "code_generation"],
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_events_dir() -> Path:
    """Get the events directory, creating if needed."""
    events_dir = Path(os.environ.get(
        "REACTOR_EVENTS_DIR",
        str(Path.home() / ".jarvis" / "reactor" / "events")
    ))
    events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir


def _is_publishing_enabled() -> bool:
    """Check if event publishing is enabled."""
    return os.environ.get("REACTOR_EVENTS_ENABLED", "true").lower() in ("true", "1", "yes")


# =============================================================================
# EVENT TYPES
# =============================================================================

class ReactorEventType(Enum):
    """Event types published by Reactor-Core."""
    TRAINING_START = "training_start"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"
    MODEL_UPDATED = "model_updated"  # Maps to MODEL_READY


# =============================================================================
# EVENT PUBLISHER
# =============================================================================

@dataclass
class ReactorEvent:
    """An event from Reactor-Core."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: ReactorEventType = ReactorEventType.TRAINING_PROGRESS
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "reactor_core"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata,
            "source": self.source,
        }


async def _publish_event(event: ReactorEvent) -> bool:
    """
    Publish an event to the Trinity ecosystem.

    Events are written as JSON files to ~/.jarvis/reactor/events/
    The TrinityBridgeAdapter watches this directory and forwards events.
    """
    if not _is_publishing_enabled():
        logger.debug("Event publishing disabled")
        return False

    try:
        events_dir = _get_events_dir()

        # Use timestamp prefix for ordering
        timestamp_prefix = f"{int(time.time() * 1000):015d}"
        event_file = events_dir / f"{timestamp_prefix}_{event.event_id}.json"

        # Atomic write: write to temp then rename
        temp_file = event_file.with_suffix(".tmp")

        with open(temp_file, "w") as f:
            json.dump(event.to_dict(), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

        temp_file.rename(event_file)

        logger.info(f"[TRINITY] Published {event.event_type.value}: {event.payload.get('model_name', 'N/A')}")
        return True

    except Exception as e:
        logger.error(f"[TRINITY] Failed to publish event: {e}")
        return False


# =============================================================================
# PUBLIC API - Convenience Functions
# =============================================================================

async def publish_training_started(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> bool:
    """
    Publish TRAINING_STARTED event.

    Call this at the beginning of a training run.

    Args:
        model_name: Name of the model being trained
        config: Training configuration (epochs, lr, etc.)
        run_id: Optional unique run identifier

    Returns:
        True if published successfully
    """
    event = ReactorEvent(
        event_type=ReactorEventType.TRAINING_START,
        payload={
            "model_name": model_name,
            "config": config or {},
            "run_id": run_id or uuid.uuid4().hex[:8],
            "started_at": datetime.now().isoformat(),
        },
    )
    return await _publish_event(event)


async def publish_training_progress(
    model_name: str,
    step: int,
    total_steps: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
) -> bool:
    """
    Publish TRAINING_PROGRESS event.

    Call this periodically during training to report progress.

    Args:
        model_name: Name of the model being trained
        step: Current training step
        total_steps: Total number of steps
        loss: Current loss value
        metrics: Additional metrics (accuracy, etc.)

    Returns:
        True if published successfully
    """
    event = ReactorEvent(
        event_type=ReactorEventType.TRAINING_PROGRESS,
        payload={
            "model_name": model_name,
            "step": step,
            "total_steps": total_steps,
            "progress_percent": round(step / max(total_steps, 1) * 100, 2),
            "loss": loss,
            "metrics": metrics or {},
        },
    )
    return await _publish_event(event)


async def publish_training_complete(
    model_name: str,
    model_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    total_steps: Optional[int] = None,
    training_time_seconds: Optional[float] = None,
) -> bool:
    """
    Publish TRAINING_COMPLETE event.

    Call this when training finishes successfully.

    Args:
        model_name: Name of the trained model
        model_path: Path to the trained model files
        metrics: Training metrics (final loss, accuracy, etc.)
        total_steps: Total training steps completed
        training_time_seconds: Total training duration

    Returns:
        True if published successfully
    """
    event = ReactorEvent(
        event_type=ReactorEventType.TRAINING_COMPLETE,
        payload={
            "model_name": model_name,
            "model_path": str(model_path),
            "metrics": metrics or {},
            "total_steps": total_steps,
            "training_time_seconds": training_time_seconds,
            "completed_at": datetime.now().isoformat(),
        },
    )
    return await _publish_event(event)


async def publish_training_failed(
    model_name: str,
    error_message: str,
    step: Optional[int] = None,
    traceback: Optional[str] = None,
) -> bool:
    """
    Publish TRAINING_FAILED event.

    Call this when training fails.

    Args:
        model_name: Name of the model that failed
        error_message: Error description
        step: Step where failure occurred
        traceback: Full traceback if available

    Returns:
        True if published successfully
    """
    event = ReactorEvent(
        event_type=ReactorEventType.TRAINING_FAILED,
        payload={
            "model_name": model_name,
            "error_message": error_message,
            "step": step,
            "traceback": traceback,
            "failed_at": datetime.now().isoformat(),
        },
    )
    return await _publish_event(event)


async def publish_model_ready(
    model_name: str,
    model_path: str,
    capabilities: Optional[List[str]] = None,
    model_type: str = "llm",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Publish MODEL_READY event (maps to MODEL_UPDATED).

    Call this when a model is validated and ready for deployment.
    This is THE KEY EVENT that triggers hot-swap in Prime.

    Args:
        model_name: Human-readable model name
        model_path: Path to model files (GGUF, safetensors, etc.)
        capabilities: List of capabilities (text_generation, code_generation, etc.)
        model_type: Type of model (llm, embedding, etc.)
        metadata: Additional metadata

    Returns:
        True if published successfully
    """
    event = ReactorEvent(
        event_type=ReactorEventType.MODEL_UPDATED,
        payload={
            "model_name": model_name,
            "model_path": str(model_path),
            "model_type": model_type,
            "capabilities": capabilities or ["text_generation"],
            "metadata": metadata or {},
            "ready_at": datetime.now().isoformat(),
        },
    )
    return await _publish_event(event)


# =============================================================================
# SYNC WRAPPERS (for non-async code)
# =============================================================================

def publish_training_started_sync(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> bool:
    """Sync wrapper for publish_training_started."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule coroutine without blocking
            asyncio.ensure_future(publish_training_started(model_name, config, run_id))
            return True
        else:
            return loop.run_until_complete(publish_training_started(model_name, config, run_id))
    except RuntimeError:
        # No event loop - create one
        return asyncio.run(publish_training_started(model_name, config, run_id))


def publish_training_complete_sync(
    model_name: str,
    model_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    total_steps: Optional[int] = None,
    training_time_seconds: Optional[float] = None,
) -> bool:
    """Sync wrapper for publish_training_complete."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(publish_training_complete(
                model_name, model_path, metrics, total_steps, training_time_seconds
            ))
            return True
        else:
            return loop.run_until_complete(publish_training_complete(
                model_name, model_path, metrics, total_steps, training_time_seconds
            ))
    except RuntimeError:
        return asyncio.run(publish_training_complete(
            model_name, model_path, metrics, total_steps, training_time_seconds
        ))


def publish_model_ready_sync(
    model_name: str,
    model_path: str,
    capabilities: Optional[List[str]] = None,
    model_type: str = "llm",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Sync wrapper for publish_model_ready."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(publish_model_ready(
                model_name, model_path, capabilities, model_type, metadata
            ))
            return True
        else:
            return loop.run_until_complete(publish_model_ready(
                model_name, model_path, capabilities, model_type, metadata
            ))
    except RuntimeError:
        return asyncio.run(publish_model_ready(
            model_name, model_path, capabilities, model_type, metadata
        ))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "ReactorEventType",
    "ReactorEvent",
    # Async functions
    "publish_training_started",
    "publish_training_progress",
    "publish_training_complete",
    "publish_training_failed",
    "publish_model_ready",
    # Sync wrappers
    "publish_training_started_sync",
    "publish_training_complete_sync",
    "publish_model_ready_sync",
]
