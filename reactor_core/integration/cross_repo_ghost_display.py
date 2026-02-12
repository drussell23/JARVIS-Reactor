"""
Cross-Repo Ghost Display State Reader for Reactor Core.

Reads ghost display state from ~/.jarvis/trinity/state/ghost_display_state.json,
the canonical state file published by the Unified Supervisor in JARVIS Body.

This follows the same cross-repo file-based communication pattern as
TrinityConnector (heartbeats, commands) and jarvis-body_readiness.json.

ARCHITECTURE:
    JARVIS Body (Supervisor) --writes--> ghost_display_state.json
    Reactor Core            --reads-->  ghost_display_state.json
    J-Prime                 --reads-->  ghost_display_state.json

USAGE:
    from reactor_core.integration.cross_repo_ghost_display import (
        get_ghost_display_state,
        is_ghost_display_active,
        get_visual_pipeline_state,
        is_visual_pipeline_active,
        GhostDisplayStateMonitor,
    )

    # One-shot query
    state = get_ghost_display_state()
    if state and state["is_ready"]:
        print("Ghost display is active")

    # Visual pipeline query (v250.0)
    vp_state = get_visual_pipeline_state()
    if vp_state and vp_state["is_ready"]:
        print("Visual pipeline is active")

    # Continuous monitoring
    monitor = GhostDisplayStateMonitor()
    await monitor.start()
    ...
    await monitor.stop()
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

_GHOST_STATE_FILE = (
    Path.home() / ".jarvis" / "trinity" / "state" / "ghost_display_state.json"
)

# Supported schema versions — add new versions here for backward compat
_SUPPORTED_SCHEMA_VERSIONS = {1}

_DEFAULT_CHECK_INTERVAL = float(os.environ.get(
    "JARVIS_GHOST_STATE_CHECK_INTERVAL", "30.0"
))
_DEFAULT_STALE_THRESHOLD = float(os.environ.get(
    "JARVIS_GHOST_STATE_STALE_THRESHOLD", "120.0"
))


# =============================================================================
# ONE-SHOT API
# =============================================================================

def get_ghost_display_state() -> Optional[Dict[str, Any]]:
    """
    Read ghost display state from the canonical state file.

    Returns:
        Parsed state dict if file exists, is valid JSON, and has a supported
        schema version. Returns None on any failure (missing file, parse error,
        unsupported schema).
    """
    try:
        if not _GHOST_STATE_FILE.exists():
            return None

        raw = _GHOST_STATE_FILE.read_text()
        state = json.loads(raw)

        # Schema version validation (backward compat: version 0 = no field)
        schema_version = state.get("schema_version", 0)
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS and schema_version != 0:
            logger.warning(
                f"[GhostDisplayReader] Unsupported schema version {schema_version} "
                f"(supported: {_SUPPORTED_SCHEMA_VERSIONS})"
            )
            return None

        return state

    except (json.JSONDecodeError, OSError, TypeError) as e:
        logger.debug(f"[GhostDisplayReader] Failed to read state: {e}")
        return None


def is_ghost_display_active(stale_threshold: Optional[float] = None) -> bool:
    """
    Convenience check: is the ghost display currently active and fresh?

    Args:
        stale_threshold: Seconds after which the state is considered stale.
                        Defaults to JARVIS_GHOST_STATE_STALE_THRESHOLD (120s).

    Returns:
        True only if state file exists, is_ready=True, and timestamp is fresh.
    """
    if stale_threshold is None:
        stale_threshold = _DEFAULT_STALE_THRESHOLD

    state = get_ghost_display_state()
    if not state:
        return False

    if not state.get("is_ready", False):
        return False

    timestamp = state.get("timestamp", 0)
    age = time.time() - timestamp
    if age > stale_threshold:
        logger.debug(
            f"[GhostDisplayReader] State is stale ({age:.0f}s > {stale_threshold:.0f}s)"
        )
        return False

    return True


# =============================================================================
# v250.0: VISUAL PIPELINE STATE READER
# =============================================================================
# Reads visual pipeline state from ~/.jarvis/trinity/state/visual_pipeline_state.json,
# the canonical state file published by the Unified Supervisor (Phase 6.8).
# Follows the same pattern as the ghost display functions above.
# =============================================================================

_VISUAL_PIPELINE_STATE_FILE = (
    Path.home() / ".jarvis" / "trinity" / "state" / "visual_pipeline_state.json"
)


def get_visual_pipeline_state() -> Optional[Dict[str, Any]]:
    """
    Read visual pipeline state from the canonical state file.

    Returns:
        Parsed state dict if file exists, is valid JSON, and has a supported
        schema version. Returns None on any failure.
    """
    try:
        if not _VISUAL_PIPELINE_STATE_FILE.exists():
            return None

        raw = _VISUAL_PIPELINE_STATE_FILE.read_text()
        state = json.loads(raw)

        # Schema version validation
        schema_version = state.get("schema_version", 0)
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS and schema_version != 0:
            logger.warning(
                f"[VisualPipelineReader] Unsupported schema version {schema_version} "
                f"(supported: {_SUPPORTED_SCHEMA_VERSIONS})"
            )
            return None

        return state

    except (json.JSONDecodeError, OSError, TypeError) as e:
        logger.debug(f"[VisualPipelineReader] Failed to read state: {e}")
        return None


def is_visual_pipeline_active(stale_threshold: Optional[float] = None) -> bool:
    """
    Convenience check: is the visual pipeline currently active and fresh?

    Args:
        stale_threshold: Seconds after which the state is considered stale.
                        Defaults to JARVIS_GHOST_STATE_STALE_THRESHOLD (120s).

    Returns:
        True only if state file exists, is_ready=True, and timestamp is fresh.
    """
    if stale_threshold is None:
        stale_threshold = _DEFAULT_STALE_THRESHOLD

    state = get_visual_pipeline_state()
    if not state:
        return False

    if not state.get("is_ready", False):
        return False

    timestamp = state.get("timestamp", 0)
    age = time.time() - timestamp
    if age > stale_threshold:
        logger.debug(
            f"[VisualPipelineReader] State is stale ({age:.0f}s > {stale_threshold:.0f}s)"
        )
        return False

    return True


# =============================================================================
# CONTINUOUS MONITOR
# =============================================================================

class GhostDisplayStateMonitor:
    """
    Background monitor for ghost display state changes.

    Periodically reads the state file and notifies registered callbacks
    when the ghost display becomes active, inactive, or stale.

    Env vars:
        JARVIS_GHOST_STATE_CHECK_INTERVAL: Check interval in seconds (default: 30)
        JARVIS_GHOST_STATE_STALE_THRESHOLD: Staleness threshold in seconds (default: 120)
    """

    def __init__(
        self,
        check_interval: Optional[float] = None,
        stale_threshold: Optional[float] = None,
    ):
        self._check_interval = check_interval or _DEFAULT_CHECK_INTERVAL
        self._stale_threshold = stale_threshold or _DEFAULT_STALE_THRESHOLD
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._async_callbacks: List[Callable[[Dict[str, Any]], Any]] = []
        self._task: Optional[asyncio.Task] = None
        self._last_state: Optional[Dict[str, Any]] = None
        self._last_active: Optional[bool] = None

    def on_change(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a sync callback for state changes."""
        self._callbacks.append(callback)

    def on_change_async(self, callback: Callable[[Dict[str, Any]], Any]) -> None:
        """Register an async callback for state changes."""
        self._async_callbacks.append(callback)

    async def start(self) -> None:
        """Start the background monitoring loop."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"[GhostDisplayMonitor] Started (interval={self._check_interval}s, "
            f"stale_threshold={self._stale_threshold}s)"
        )

    async def stop(self) -> None:
        """Stop the background monitoring loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._task = None
            logger.info("[GhostDisplayMonitor] Stopped")

    @property
    def is_active(self) -> bool:
        """Whether the ghost display is currently active (from last check)."""
        return self._last_active is True

    @property
    def last_state(self) -> Optional[Dict[str, Any]]:
        """Last read state dict."""
        return self._last_state

    async def _monitor_loop(self) -> None:
        """Background loop that checks state file periodically."""
        try:
            while True:
                try:
                    state = get_ghost_display_state()
                    now_active = False

                    if state:
                        self._last_state = state
                        timestamp = state.get("timestamp", 0)
                        age = time.time() - timestamp
                        now_active = (
                            state.get("is_ready", False)
                            and age <= self._stale_threshold
                        )

                    # Detect transition
                    if now_active != self._last_active:
                        event = {
                            "previous_active": self._last_active,
                            "current_active": now_active,
                            "state": state,
                            "timestamp": time.time(),
                        }

                        transition = (
                            "activated" if now_active else "deactivated"
                        )
                        logger.info(
                            f"[GhostDisplayMonitor] Ghost display {transition}"
                        )

                        # Notify callbacks
                        for cb in self._callbacks:
                            try:
                                cb(event)
                            except Exception as e:
                                logger.warning(
                                    f"[GhostDisplayMonitor] Sync callback error: {e}"
                                )

                        for cb in self._async_callbacks:
                            try:
                                await cb(event)
                            except Exception as e:
                                logger.warning(
                                    f"[GhostDisplayMonitor] Async callback error: {e}"
                                )

                        self._last_active = now_active

                except Exception as e:
                    logger.debug(f"[GhostDisplayMonitor] Check error: {e}")

                await asyncio.sleep(self._check_interval)

        except asyncio.CancelledError:
            pass
