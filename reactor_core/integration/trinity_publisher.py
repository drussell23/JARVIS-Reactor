"""
Trinity Event Publisher for Reactor-Core v2.0
==============================================

This module enables Reactor-Core to publish events to the Trinity ecosystem.
It writes events to the shared event directory that the Trinity Bridge Adapter watches.

v2.0 FEATURES:
- Dead Letter Queue (DLQ) for failed event retry
- Pydantic schema validation for all events
- Event versioning for schema evolution
- Distributed tracing with trace IDs
- Circuit breaker for fault tolerance

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
import traceback as tb
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# v258.0: Strong references for fire-and-forget background tasks (prevents GC collection)
_trinity_pub_background_tasks: set = set()

# Event schema version for evolution
EVENT_SCHEMA_VERSION = "2.0"

# =============================================================================
# PYDANTIC VALIDATION (with graceful fallback)
# =============================================================================

try:
    from pydantic import BaseModel, Field, validator, ValidationError

    class ModelReadyEventSchema(BaseModel):
        """Pydantic schema for MODEL_READY events."""
        model_name: str = Field(..., min_length=1, max_length=200)
        model_path: str = Field(..., description="Absolute path to model")
        model_type: str = Field(default="llm")
        capabilities: List[str] = Field(default_factory=lambda: ["text_generation"])
        metadata: Dict[str, Any] = Field(default_factory=dict)
        ready_at: str = Field(..., description="ISO timestamp")

        @validator("model_path")
        def validate_path(cls, v):
            path = Path(v)
            if not path.is_absolute():
                raise ValueError(f"Model path must be absolute: {v}")
            return str(path.resolve())

        @validator("capabilities")
        def validate_capabilities(cls, v):
            valid = {"text_generation", "code_generation", "embeddings", "classification", "chat"}
            # Allow any capabilities but log warning for unknown ones
            unknown = set(v) - valid
            if unknown:
                logger.debug(f"Unknown capabilities (allowed): {unknown}")
            return v

    class TrainingEventSchema(BaseModel):
        """Pydantic schema for training events."""
        model_name: str = Field(..., min_length=1, max_length=200)
        config: Dict[str, Any] = Field(default_factory=dict)
        run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
        started_at: Optional[str] = None
        completed_at: Optional[str] = None
        metrics: Dict[str, Any] = Field(default_factory=dict)

    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    ValidationError = Exception
    logger.debug("Pydantic not available, schema validation disabled")

# =============================================================================
# DLQ AND CIRCUIT BREAKER
# =============================================================================

_dlq = None
_circuit_breaker = None
_dlq_lock = asyncio.Lock()


async def _get_dlq():
    """Get or create Dead Letter Queue for failed events."""
    global _dlq
    if _dlq is not None:
        return _dlq

    async with _dlq_lock:
        if _dlq is not None:
            return _dlq

        try:
            from reactor_core.utils.async_helpers import DeadLetterQueue
            from reactor_core.config.base_config import resolve_path

            dlq_path = resolve_path("dlq", env_var="REACTOR_DLQ_PATH", subdir="dlq")
            _dlq = DeadLetterQueue(
                name="trinity_publisher",
                persist_path=dlq_path / "trinity_publisher_dlq.json",
                auto_retry_interval=60.0,
            )

            # Register retry operations
            _dlq.register_operation("publish_event", _retry_publish_event)

            # Start auto-retry
            await _dlq.start_auto_retry()
            logger.info("[TRINITY] Dead Letter Queue initialized")

        except ImportError:
            logger.debug("DLQ not available")
        except Exception as e:
            logger.debug(f"DLQ initialization failed: {e}")

    return _dlq


async def _get_circuit_breaker():
    """Get or create circuit breaker for event publishing."""
    global _circuit_breaker
    if _circuit_breaker is not None:
        return _circuit_breaker

    try:
        from reactor_core.utils.async_helpers import CircuitBreaker, CircuitBreakerConfig

        _circuit_breaker = CircuitBreaker(
            "trinity_publisher",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout_seconds=30.0,
            )
        )
        logger.debug("[TRINITY] Circuit breaker initialized")

    except ImportError:
        logger.debug("Circuit breaker not available")

    return _circuit_breaker


async def _retry_publish_event(event_dict: Dict[str, Any]) -> bool:
    """Retry function for DLQ - republish failed event."""
    event = ReactorEvent(
        event_id=event_dict.get("event_id", uuid.uuid4().hex[:12]),
        event_type=ReactorEventType(event_dict["event_type"]),
        timestamp=event_dict.get("timestamp", datetime.now().isoformat()),
        payload=event_dict.get("payload", {}),
        metadata=event_dict.get("metadata", {}),
    )
    return await _publish_event_internal(event)


# =============================================================================
# DISTRIBUTED TRACING
# =============================================================================

_current_trace_id: Optional[str] = None


def set_trace_id(trace_id: str) -> None:
    """Set the current trace ID for distributed tracing."""
    global _current_trace_id
    _current_trace_id = trace_id


def get_trace_id() -> str:
    """Get current trace ID or generate a new one."""
    global _current_trace_id
    if _current_trace_id is None:
        _current_trace_id = f"reactor-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    return _current_trace_id


@asynccontextmanager
async def trace_context(trace_id: Optional[str] = None):
    """Context manager for distributed tracing."""
    global _current_trace_id
    old_trace_id = _current_trace_id
    _current_trace_id = trace_id or get_trace_id()
    try:
        yield _current_trace_id
    finally:
        _current_trace_id = old_trace_id


# =============================================================================
# CONFIGURATION - DYNAMIC PATH RESOLUTION
# =============================================================================

def _get_events_dir() -> Path:
    """Get the events directory using dynamic path resolution."""
    try:
        from reactor_core.config.base_config import resolve_path
        events_dir = resolve_path(
            "reactor_events",
            env_var="REACTOR_EVENTS_DIR",
            xdg_type="data",
            subdir="reactor/events"
        )
    except ImportError:
        # Fallback if config not available
        events_dir = Path(os.environ.get(
            "REACTOR_EVENTS_DIR",
            str(Path.home() / ".jarvis" / "reactor" / "events")
        ))
        events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir


def _is_publishing_enabled() -> bool:
    """Check if event publishing is enabled."""
    return os.environ.get("REACTOR_EVENTS_ENABLED", "true").lower() in ("true", "1", "yes")


def _validate_model_path(model_path: str) -> Path:
    """
    Validate and resolve model path.

    Args:
        model_path: Path string to validate

    Returns:
        Resolved absolute Path

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If path is invalid
    """
    path = Path(model_path).expanduser().resolve()

    # If not absolute, try to resolve relative to common model directories
    if not path.is_absolute():
        candidates = [
            Path.home() / ".jarvis" / "models" / model_path,
            Path.home() / ".jarvis" / "training" / "output" / model_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        logger.warning(f"[TRINITY] Model path does not exist: {path}")
        # Don't raise - path might be created later

    return path


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
    """An event from Reactor-Core with versioning and tracing."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: ReactorEventType = ReactorEventType.TRAINING_PROGRESS
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "reactor_core"
    # v2.0: Add versioning and tracing
    schema_version: str = EVENT_SCHEMA_VERSION
    trace_id: str = field(default_factory=get_trace_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata,
            "source": self.source,
            # v2.0 fields
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
        }


async def _publish_event_internal(event: ReactorEvent) -> bool:
    """
    Internal publish function (without DLQ fallback).

    Events are written as JSON files to ~/.jarvis/reactor/events/
    The TrinityBridgeAdapter watches this directory and forwards events.
    """
    events_dir = _get_events_dir()

    # Use timestamp prefix for ordering
    timestamp_prefix = f"{int(time.time() * 1000):015d}"
    event_file = events_dir / f"{timestamp_prefix}_{event.event_id}.json"

    # Atomic write: write to temp then rename
    temp_file = event_file.with_suffix(".tmp")

    # Use asyncio.to_thread for file I/O
    def write_event():
        with open(temp_file, "w") as f:
            json.dump(event.to_dict(), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        temp_file.rename(event_file)

    try:
        await asyncio.to_thread(write_event)
    except AttributeError:
        # Python < 3.9 fallback
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, write_event)

    logger.info(
        f"[TRINITY] Published {event.event_type.value}: "
        f"{event.payload.get('model_name', 'N/A')} (trace={event.trace_id[:8]})"
    )
    return True


async def _publish_event(event: ReactorEvent) -> bool:
    """
    Publish an event to the Trinity ecosystem with fault tolerance.

    Features:
    - Circuit breaker to prevent cascading failures
    - Dead Letter Queue for retry on failure
    - Distributed tracing support
    """
    if not _is_publishing_enabled():
        logger.debug("Event publishing disabled")
        return False

    # Use circuit breaker if available
    circuit_breaker = await _get_circuit_breaker()

    try:
        if circuit_breaker:
            # Execute through circuit breaker
            async def publish_with_cb():
                return await _publish_event_internal(event)

            return await circuit_breaker.execute(publish_with_cb)
        else:
            # Direct execution
            return await _publish_event_internal(event)

    except Exception as e:
        logger.error(f"[TRINITY] Failed to publish event: {e}")

        # Add to DLQ for retry
        dlq = await _get_dlq()
        if dlq:
            try:
                await dlq.add(
                    operation="publish_event",
                    args=(),
                    kwargs={"event_dict": event.to_dict()},
                    exception=e,
                    metadata={
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "trace_id": event.trace_id,
                    },
                )
                logger.info(f"[TRINITY] Event {event.event_id} added to DLQ for retry")
            except Exception as dlq_error:
                logger.error(f"[TRINITY] Failed to add to DLQ: {dlq_error}")

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
    model_path: Union[str, Path],
    capabilities: Optional[List[str]] = None,
    model_type: str = "llm",
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
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
        trace_id: Optional trace ID for distributed tracing

    Returns:
        True if published successfully
    """
    # Validate and resolve path
    validated_path = _validate_model_path(str(model_path))
    ready_at = datetime.now().isoformat()

    # v2.0: Pydantic validation if available
    if HAS_PYDANTIC:
        try:
            validated = ModelReadyEventSchema(
                model_name=model_name,
                model_path=str(validated_path),
                model_type=model_type,
                capabilities=capabilities or ["text_generation"],
                metadata=metadata or {},
                ready_at=ready_at,
            )
            # Use validated data
            payload = validated.dict()
        except ValidationError as e:
            logger.error(f"[TRINITY] Invalid MODEL_READY event: {e}")
            return False
    else:
        # Fallback without validation
        payload = {
            "model_name": model_name,
            "model_path": str(validated_path),
            "model_type": model_type,
            "capabilities": capabilities or ["text_generation"],
            "metadata": metadata or {},
            "ready_at": ready_at,
        }

    event = ReactorEvent(
        event_type=ReactorEventType.MODEL_UPDATED,
        payload=payload,
        trace_id=trace_id or get_trace_id(),
    )
    return await _publish_event(event)


# =============================================================================
# SYNC WRAPPERS (for non-async code) - FIXED: No deprecated asyncio patterns
# =============================================================================

def _run_async_in_sync(coro) -> Any:
    """
    Safely run async coroutine from sync context.

    Handles all edge cases:
    - No running loop: use asyncio.run()
    - Running loop in same thread: schedule and return True (fire-and-forget)
    - Running loop in different thread: use run_coroutine_threadsafe()
    """
    import threading

    try:
        # Try to get the running loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)

    # There's a running loop - check if we're in the same thread
    if threading.current_thread() is threading.main_thread():
        # Same thread - schedule as task (v258.0: store strong reference)
        _task = asyncio.create_task(coro, name="trinity_pub_sync_wrapper")
        _trinity_pub_background_tasks.add(_task)
        _task.add_done_callback(_trinity_pub_background_tasks.discard)
        return True
    else:
        # Different thread - use thread-safe method
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=30.0)
        except Exception as e:
            logger.error(f"[TRINITY] Sync wrapper error: {e}")
            return False


def publish_training_started_sync(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> bool:
    """Sync wrapper for publish_training_started."""
    return _run_async_in_sync(publish_training_started(model_name, config, run_id))


def publish_training_complete_sync(
    model_name: str,
    model_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    total_steps: Optional[int] = None,
    training_time_seconds: Optional[float] = None,
) -> bool:
    """Sync wrapper for publish_training_complete."""
    return _run_async_in_sync(publish_training_complete(
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
    return _run_async_in_sync(publish_model_ready(
        model_name, model_path, capabilities, model_type, metadata
    ))


# =============================================================================
# LIFECYCLE MANAGEMENT
# =============================================================================

async def shutdown_publisher() -> None:
    """Shutdown the publisher and cleanup resources."""
    global _dlq, _circuit_breaker

    if _dlq:
        try:
            await _dlq.stop_auto_retry()
            logger.info("[TRINITY] Publisher DLQ stopped")
        except Exception as e:
            logger.debug(f"DLQ stop error: {e}")
        _dlq = None

    _circuit_breaker = None


async def get_publisher_stats() -> Dict[str, Any]:
    """Get publisher statistics."""
    stats = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "trace_id": get_trace_id(),
        "dlq_available": _dlq is not None,
        "circuit_breaker_available": _circuit_breaker is not None,
        "pydantic_validation": HAS_PYDANTIC,
    }

    if _dlq:
        dlq_stats = _dlq.get_stats()
        stats["dlq"] = dlq_stats

    if _circuit_breaker:
        cb_stats = _circuit_breaker.get_stats()
        stats["circuit_breaker"] = cb_stats

    return stats


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "ReactorEventType",
    "ReactorEvent",
    "EVENT_SCHEMA_VERSION",
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
    # Tracing
    "set_trace_id",
    "get_trace_id",
    "trace_context",
    # Lifecycle
    "shutdown_publisher",
    "get_publisher_stats",
]
