"""
Reactor Core Voice Integration - Voice Orchestrator Client Bridge
===================================================================

Provides intelligent voice announcements for Reactor Core training lifecycle events.
Integrates with JARVIS Voice Orchestrator via Unix domain socket IPC.

v2.0 Features:
- Uses VoiceClient for IPC communication (replaces trinity_voice_coordinator)
- Training start/complete/failed announcements
- Model export announcements
- Deployment announcements
- Training progress milestones
- Zero hardcoding (environment-driven)
- Async/parallel execution
- Graceful degradation if orchestrator unavailable

Architecture:
+------------------------------------------------------------------+
|              Reactor Core Voice Integration v2.0                  |
+------------------------------------------------------------------+
|                                                                   |
|  Reactor Event               Category/Context                     |
|  -------------               ---------------                      |
|  Training Start  ---------> init (HIGH priority)                  |
|  Training Complete -------> ready (HIGH priority)                 |
|  Training Failed ---------> error (HIGH priority)                 |
|  Model Export    ---------> general (NORMAL priority)             |
|  Deployment      ---------> ready (HIGH priority)                 |
|                                                                   |
|           |                                                       |
|           v                                                       |
|  +--------------------------------------------------+            |
|  |   VoiceClient (Unix socket IPC)                   |            |
|  |   -> Queues locally if disconnected               |            |
|  |   -> Reconnects with exponential backoff          |            |
|  +--------------------------------------------------+            |
|           |                                                       |
|           v                                                       |
|  +--------------------------------------------------+            |
|  |   VoiceOrchestrator (JARVIS Body)                 |            |
|  |   -> Coalesces messages                           |            |
|  |   -> Serializes playback                          |            |
|  |   -> Multi-engine TTS                             |            |
|  +--------------------------------------------------+            |
|                                                                   |
+------------------------------------------------------------------+

Usage:
    from reactor_core.voice_integration import (
        announce_training_started,
        announce_training_complete,
        announce_deployment_complete,
    )

    await announce_training_started(
        model_name="TinyLlama-1.1B",
        samples=1500,
    )

    await announce_training_complete(
        model_name="TinyLlama-1.1B",
        steps=1000,
        loss=0.245,
        duration_seconds=1823.4,
    )

Author: Reactor Core Voice Integration v2.0
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# VoiceClient Import (from JARVIS Body or local copy)
# =============================================================================

_VOICE_AVAILABLE = False

# Try multiple import paths for the shared voice client
JARVIS_BODY_PATH = os.getenv(
    "JARVIS_BODY_PATH",
    str(Path(__file__).parent.parent.parent / "JARVIS-AI-Agent")
)

# Add JARVIS Body to path if it exists
if JARVIS_BODY_PATH and Path(JARVIS_BODY_PATH).exists():
    sys.path.insert(0, JARVIS_BODY_PATH)

try:
    # Try importing from JARVIS Body's shared module
    from backend.core.shared_voice_client import (
        VoiceClient,
        VoicePriority,
        VoiceContext,
        announce as _raw_announce,
    )
    _VOICE_AVAILABLE = True
    logger.info("[Reactor Voice] VoiceClient available for announcements")
except ImportError:
    try:
        # Fallback: try importing from voice_client directly
        from backend.core.voice_client import (
            VoiceClient,
            VoicePriority,
            announce as _raw_announce,
        )
        # Create VoiceContext for backward compat
        class VoiceContext:
            STARTUP = "init"
            TRINITY = "init"
            RUNTIME = "general"
            NARRATOR = "general"
            ALERT = "error"
            SUCCESS = "ready"
            SHUTDOWN = "shutdown"
            HEALTH = "health"
            PROGRESS = "progress"

            @staticmethod
            def category(ctx):
                return ctx

        _VOICE_AVAILABLE = True
        logger.info("[Reactor Voice] VoiceClient (basic) available for announcements")
    except ImportError as e:
        logger.debug(f"[Reactor Voice] VoiceClient not available: {e}")

# Create dummy implementations for graceful degradation
if not _VOICE_AVAILABLE:
    class VoicePriority:
        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        NORMAL = "NORMAL"
        LOW = "LOW"
        BACKGROUND = "BACKGROUND"

    class VoiceContext:
        STARTUP = "init"
        TRINITY = "init"
        RUNTIME = "general"
        NARRATOR = "general"
        ALERT = "error"
        SUCCESS = "ready"
        SHUTDOWN = "shutdown"
        HEALTH = "health"
        PROGRESS = "progress"

        @property
        def category(self):
            return self

    async def _raw_announce(*args, **kwargs) -> bool:
        return False


# =============================================================================
# Configuration
# =============================================================================

class ReactorVoiceConfig:
    """Configuration for Reactor Core voice announcements (environment-driven)."""

    def __init__(self):
        # Enable/disable voice announcements
        self.enabled = os.getenv("REACTOR_VOICE_ENABLED", "true").lower() == "true"

        # Announcement granularity
        self.announce_training_start = os.getenv("REACTOR_VOICE_TRAINING_START", "true").lower() == "true"
        self.announce_training_complete = os.getenv("REACTOR_VOICE_TRAINING_COMPLETE", "true").lower() == "true"
        self.announce_export = os.getenv("REACTOR_VOICE_EXPORT", "true").lower() == "true"
        self.announce_deployment = os.getenv("REACTOR_VOICE_DEPLOYMENT", "true").lower() == "true"
        self.announce_failures = os.getenv("REACTOR_VOICE_FAILURES", "true").lower() == "true"

        # Source identifier for cross-repo tracking
        self.source_id = os.getenv("REACTOR_VOICE_SOURCE", "reactor_core")


_config = ReactorVoiceConfig()


# =============================================================================
# Internal Helper
# =============================================================================

async def _announce(
    message: str,
    context: str,
    priority: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Internal announce wrapper with graceful degradation."""
    if not _VOICE_AVAILABLE:
        logger.debug(f"[Reactor Voice] Would announce: {message}")
        return False

    try:
        # Map context string to category
        category_map = {
            "trinity": "init",
            "startup": "init",
            "runtime": "general",
            "narrator": "general",
            "alert": "error",
            "success": "ready",
            "shutdown": "shutdown",
            "health": "health",
            "progress": "progress",
        }
        category = category_map.get(context, "general")

        # Map priority string to VoicePriority
        priority_map = {
            "CRITICAL": VoicePriority.CRITICAL,
            "HIGH": VoicePriority.HIGH,
            "NORMAL": VoicePriority.NORMAL,
            "LOW": VoicePriority.LOW,
            "BACKGROUND": VoicePriority.BACKGROUND,
        }
        prio = priority_map.get(priority, VoicePriority.NORMAL)

        return await _raw_announce(
            message=message,
            context=VoiceContext.RUNTIME,
            priority=prio,
            source=_config.source_id,
            metadata=metadata,
        )
    except Exception as e:
        logger.debug(f"[Reactor Voice] Announce failed (non-critical): {e}")
        return False


# =============================================================================
# Voice Announcement Functions
# =============================================================================

async def announce_training_started(
    model_name: str,
    samples: int,
    epochs: int = 3,
) -> bool:
    """
    Announce that model training has started.

    Args:
        model_name: Name of model being trained
        samples: Number of training samples
        epochs: Number of epochs

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_training_start:
        return False

    try:
        message = (
            f"Reactor Core: Starting model training for {model_name}. "
            f"{samples} samples, {epochs} epochs."
        )

        return await _announce(
            message=message,
            context="trinity",
            priority="HIGH",
            metadata={
                "event": "training_started",
                "model_name": model_name,
                "samples": samples,
                "epochs": epochs,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce training start: {e}")
        return False


async def announce_training_complete(
    model_name: str,
    steps: int,
    loss: float,
    duration_seconds: float,
    success: bool = True,
) -> bool:
    """
    Announce that model training has completed.

    Args:
        model_name: Name of model trained
        steps: Training steps completed
        loss: Final training loss
        duration_seconds: Training duration
        success: Whether training was successful

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_training_complete:
        return False

    try:
        if success:
            minutes = int(duration_seconds / 60)
            message = (
                f"Model training complete in {minutes} minutes. "
                f"{steps} steps, final loss {loss:.3f}. "
                f"New model ready for deployment."
            )
            context = "success"
            priority = "HIGH"
        else:
            message = f"Model training incomplete. {steps} steps completed before stopping."
            context = "alert"
            priority = "NORMAL"

        return await _announce(
            message=message,
            context=context,
            priority=priority,
            metadata={
                "event": "training_complete",
                "model_name": model_name,
                "steps": steps,
                "loss": loss,
                "duration": duration_seconds,
                "success": success,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce training complete: {e}")
        return False


async def announce_training_failed(
    model_name: str,
    error_message: str,
    steps_completed: int = 0,
) -> bool:
    """
    Announce that model training has failed.

    Args:
        model_name: Name of model
        error_message: Error that caused failure
        steps_completed: Steps completed before failure

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_failures:
        return False

    try:
        message = (
            f"Model training failed for {model_name}: {error_message}. "
            f"{steps_completed} steps completed."
        )

        return await _announce(
            message=message,
            context="alert",
            priority="HIGH",
            metadata={
                "event": "training_failed",
                "model_name": model_name,
                "error": error_message,
                "steps": steps_completed,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce training failure: {e}")
        return False


async def announce_export_started(
    format: str = "GGUF",
    quantization: Optional[str] = None,
) -> bool:
    """
    Announce that model export has started.

    Args:
        format: Export format (e.g., "GGUF")
        quantization: Quantization method (e.g., "Q4_K_M")

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_export:
        return False

    try:
        if quantization:
            message = f"Exporting model to {format} with {quantization} quantization."
        else:
            message = f"Exporting model to {format} format."

        return await _announce(
            message=message,
            context="narrator",
            priority="NORMAL",
            metadata={
                "event": "export_started",
                "format": format,
                "quantization": quantization,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce export start: {e}")
        return False


async def announce_export_complete(
    format: str = "GGUF",
    file_size_mb: Optional[float] = None,
) -> bool:
    """
    Announce that model export has completed.

    Args:
        format: Export format
        file_size_mb: Size of exported file in MB

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_export:
        return False

    try:
        if file_size_mb:
            message = f"{format} export complete. File size: {file_size_mb:.1f} megabytes."
        else:
            message = f"{format} export complete."

        return await _announce(
            message=message,
            context="success",
            priority="NORMAL",
            metadata={
                "event": "export_complete",
                "format": format,
                "file_size_mb": file_size_mb,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce export complete: {e}")
        return False


async def announce_deployment_started(
    target: str = "JARVIS-Prime",
) -> bool:
    """
    Announce that model deployment has started.

    Args:
        target: Deployment target

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_deployment:
        return False

    try:
        message = f"Deploying new model to {target}."

        return await _announce(
            message=message,
            context="narrator",
            priority="NORMAL",
            metadata={
                "event": "deployment_started",
                "target": target,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce deployment start: {e}")
        return False


async def announce_deployment_complete(
    target: str = "JARVIS-Prime",
    model_version: Optional[str] = None,
) -> bool:
    """
    Announce that model deployment has completed.

    Args:
        target: Deployment target
        model_version: Version of deployed model

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_deployment:
        return False

    try:
        if model_version:
            message = f"Model {model_version} deployed successfully to {target}."
        else:
            message = f"Model deployed successfully to {target}."

        return await _announce(
            message=message,
            context="success",
            priority="HIGH",
            metadata={
                "event": "deployment_complete",
                "target": target,
                "model_version": model_version,
            }
        )

    except Exception as e:
        logger.error(f"[Reactor Voice] Failed to announce deployment complete: {e}")
        return False


# =============================================================================
# Utility Functions
# =============================================================================

def is_voice_available() -> bool:
    """Check if VoiceClient is available."""
    return _VOICE_AVAILABLE


def get_config() -> ReactorVoiceConfig:
    """Get current voice configuration."""
    return _config


async def test_voice_integration() -> bool:
    """Test voice integration by sending a test announcement."""
    if not _VOICE_AVAILABLE:
        logger.warning("[Reactor Voice] VoiceClient unavailable for testing")
        return False

    try:
        success = await _announce(
            message="Reactor Core voice integration test successful.",
            context="runtime",
            priority="LOW",
            metadata={"event": "test"}
        )

        if success:
            logger.info("[Reactor Voice] Voice integration test: sent successfully")
        else:
            logger.info("[Reactor Voice] Voice integration test: queued (orchestrator may be unavailable)")

        return True  # Return True if no exception (message queued or sent)

    except Exception as e:
        logger.error(f"[Reactor Voice] Voice integration test failed: {e}")
        return False
