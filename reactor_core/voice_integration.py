"""
Reactor Core Voice Integration - Trinity Voice Coordinator Bridge
===================================================================

Provides intelligent voice announcements for Reactor Core training lifecycle events.
Integrates with Trinity Voice Coordinator (JARVIS Body repo) for cross-repo coordination.

v1.0 Features:
- Training start/complete/failed announcements
- Model export announcements
- Deployment announcements
- Training progress milestones
- Zero hardcoding (environment-driven)
- Async/parallel execution
- Graceful degradation if voice unavailable

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│              Reactor Core Voice Integration                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Reactor Event            Trinity Voice Context                │
│  ──────────────            ──────────────────                   │
│  • Training Start ──────▶  TRINITY context                     │
│  • Training Complete ───▶  SUCCESS context                     │
│  • Training Failed ─────▶  ALERT context                       │
│  • Model Export ────────▶  NARRATOR context                    │
│  • Deployment ──────────▶  SUCCESS context                     │
│                                                                 │
│           ▼                                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │   Trinity Voice Coordinator              │                   │
│  │   (backend.core.trinity_voice_coordinator)│                  │
│  └─────────────────────────────────────────┘                   │
│           │                                                     │
│           ▼                                                     │
│  Multi-engine TTS (MacOS Say → pyttsx3 → Edge TTS)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from reactor_core.voice_integration import (
        announce_training_started,
        announce_training_complete,
        announce_deployment_complete,
    )

    # Before training
    await announce_training_started(
        model_name="TinyLlama-1.1B",
        samples=1500,
    )

    # After training
    await announce_training_complete(
        model_name="TinyLlama-1.1B",
        steps=1000,
        loss=0.245,
        duration_seconds=1823.4,
    )

Author: Reactor Core Trinity v1.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Add JARVIS body repo to path for Trinity Voice Coordinator import
JARVIS_BODY_PATH = os.getenv(
    "JARVIS_BODY_PATH",
    str(Path(__file__).parent.parent.parent / "JARVIS-AI-Agent")
)
if JARVIS_BODY_PATH and Path(JARVIS_BODY_PATH).exists():
    sys.path.insert(0, JARVIS_BODY_PATH)


# =============================================================================
# Trinity Voice Coordinator Import (with graceful fallback)
# =============================================================================

_VOICE_AVAILABLE = False
_VOICE_COORDINATOR = None

try:
    from backend.core.trinity_voice_coordinator import (
        announce as trinity_announce,
        get_voice_coordinator,
        VoiceContext,
        VoicePriority,
    )
    _VOICE_AVAILABLE = True
    logger.info("✅ Trinity Voice Coordinator available for Reactor Core announcements")
except ImportError as e:
    logger.debug(f"Trinity Voice Coordinator not available: {e}")
    # Create dummy implementations for graceful degradation
    class VoiceContext:
        STARTUP = "startup"
        TRINITY = "trinity"
        RUNTIME = "runtime"
        NARRATOR = "narrator"
        ALERT = "alert"
        SUCCESS = "success"

    class VoicePriority:
        CRITICAL = 0
        HIGH = 1
        NORMAL = 2
        LOW = 3
        BACKGROUND = 4

    async def trinity_announce(*args, **kwargs):
        return False

    async def get_voice_coordinator():
        return None


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

    if not _VOICE_AVAILABLE:
        logger.debug("[Reactor Voice] Trinity coordinator unavailable, skipping announcement")
        return False

    try:
        message = (
            f"Reactor Core: Starting model training for {model_name}. "
            f"{samples} samples, {epochs} epochs."
        )

        return await trinity_announce(
            message=message,
            context=VoiceContext.TRINITY,
            priority=VoicePriority.HIGH,
            source=_config.source_id,
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

    if not _VOICE_AVAILABLE:
        return False

    try:
        if success:
            minutes = int(duration_seconds / 60)
            message = (
                f"Model training complete in {minutes} minutes. "
                f"{steps} steps, final loss {loss:.3f}. "
                f"New model ready for deployment."
            )
            context = VoiceContext.SUCCESS
            priority = VoicePriority.HIGH
        else:
            message = f"Model training incomplete. {steps} steps completed before stopping."
            context = VoiceContext.ALERT
            priority = VoicePriority.NORMAL

        return await trinity_announce(
            message=message,
            context=context,
            priority=priority,
            source=_config.source_id,
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

    if not _VOICE_AVAILABLE:
        return False

    try:
        message = (
            f"Model training failed for {model_name}: {error_message}. "
            f"{steps_completed} steps completed."
        )

        return await trinity_announce(
            message=message,
            context=VoiceContext.ALERT,
            priority=VoicePriority.HIGH,
            source=_config.source_id,
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

    if not _VOICE_AVAILABLE:
        return False

    try:
        if quantization:
            message = f"Exporting model to {format} with {quantization} quantization."
        else:
            message = f"Exporting model to {format} format."

        return await trinity_announce(
            message=message,
            context=VoiceContext.NARRATOR,
            priority=VoicePriority.NORMAL,
            source=_config.source_id,
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

    if not _VOICE_AVAILABLE:
        return False

    try:
        if file_size_mb:
            message = f"{format} export complete. File size: {file_size_mb:.1f} megabytes."
        else:
            message = f"{format} export complete."

        return await trinity_announce(
            message=message,
            context=VoiceContext.SUCCESS,
            priority=VoicePriority.NORMAL,
            source=_config.source_id,
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

    if not _VOICE_AVAILABLE:
        return False

    try:
        message = f"Deploying new model to {target}."

        return await trinity_announce(
            message=message,
            context=VoiceContext.NARRATOR,
            priority=VoicePriority.NORMAL,
            source=_config.source_id,
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

    if not _VOICE_AVAILABLE:
        return False

    try:
        if model_version:
            message = f"Model {model_version} deployed successfully to {target}."
        else:
            message = f"Model deployed successfully to {target}."

        return await trinity_announce(
            message=message,
            context=VoiceContext.SUCCESS,
            priority=VoicePriority.HIGH,
            source=_config.source_id,
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
    """Check if Trinity Voice Coordinator is available."""
    return _VOICE_AVAILABLE


def get_config() -> ReactorVoiceConfig:
    """Get current voice configuration."""
    return _config


async def test_voice_integration() -> bool:
    """Test voice integration by sending a test announcement."""
    if not _VOICE_AVAILABLE:
        logger.warning("[Reactor Voice] Trinity coordinator unavailable for testing")
        return False

    try:
        success = await trinity_announce(
            message="Reactor Core voice integration test successful.",
            context=VoiceContext.RUNTIME,
            priority=VoicePriority.LOW,
            source=_config.source_id,
            metadata={"event": "test"}
        )

        if success:
            logger.info("[Reactor Voice] ✅ Voice integration test successful")
        else:
            logger.warning("[Reactor Voice] ⚠️  Voice integration test returned False")

        return success

    except Exception as e:
        logger.error(f"[Reactor Voice] ❌ Voice integration test failed: {e}")
        return False
