"""
Cloud Mode Detector v152.0 - Cross-Repo Cloud State Awareness
==============================================================

This module provides cloud mode detection for Reactor Core, ensuring
it respects the JARVIS ecosystem's cloud-only mode when active.

When cloud mode is active:
- Skip attempting connections to localhost services
- Route to GCP endpoints if available
- Prevent circuit breaker failures for intentionally disabled services

Integration Points:
- Reads JARVIS_GCP_OFFLOAD_ACTIVE environment variable (set by supervisor)
- Reads ~/.jarvis/trinity/cloud_lock.json (persistent state)
- Provides unified API for all Reactor Core components

Author: Trinity System
Version: 152.0.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("reactor_core.cloud_mode")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Shared Trinity directory for cross-repo state
TRINITY_DIR = Path.home() / ".jarvis" / "trinity"
CLOUD_LOCK_FILE = TRINITY_DIR / "cloud_lock.json"

# v258.4: CPU pressure signal from supervisor
CPU_PRESSURE_SIGNAL_FILE = TRINITY_DIR.parent / "signals" / "cpu_pressure.json"
CPU_PRESSURE_SIGNAL_TTL = float(os.environ.get("JARVIS_CPU_PRESSURE_SIGNAL_TTL", "60.0"))

# Cache TTL to avoid excessive file I/O
CLOUD_STATE_CACHE_TTL = 5.0  # seconds


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CloudModeState:
    """Current cloud mode state with metadata."""
    is_active: bool
    reason: Optional[str] = None
    source: str = "unknown"  # "env_var", "cloud_lock", "hollow_client"
    timestamp: float = field(default_factory=time.time)
    gcp_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # v258.4: CPU pressure signal from supervisor
    cpu_pressure_active: bool = False
    cpu_pressure_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_active": self.is_active,
            "reason": self.reason,
            "source": self.source,
            "timestamp": self.timestamp,
            "gcp_endpoint": self.gcp_endpoint,
            "metadata": self.metadata,
            "cpu_pressure_active": self.cpu_pressure_active,
            "cpu_pressure_percent": self.cpu_pressure_percent,
        }


# =============================================================================
# CLOUD MODE DETECTOR
# =============================================================================

class CloudModeDetector:
    """
    Singleton detector for cloud mode state.

    Thread-safe and cache-aware to minimize file I/O overhead.
    """

    _instance: Optional["CloudModeDetector"] = None

    def __new__(cls) -> "CloudModeDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache: Optional[CloudModeState] = None
        self._cache_time: float = 0.0
        self._initialized = True

        logger.debug("[v152.0] CloudModeDetector initialized")

    def is_cloud_mode_active(self, force_refresh: bool = False) -> bool:
        """
        Check if cloud mode is active.

        This is the primary method all Reactor Core components should use
        before attempting to connect to local services.

        Args:
            force_refresh: Bypass cache and check current state

        Returns:
            True if cloud mode is active and local services should be skipped
        """
        state = self.get_cloud_state(force_refresh)
        return state.is_active

    def get_cloud_state(self, force_refresh: bool = False) -> CloudModeState:
        """
        Get detailed cloud mode state.

        Args:
            force_refresh: Bypass cache and check current state

        Returns:
            CloudModeState with full details
        """
        now = time.time()

        # Return cached state if valid
        if (not force_refresh and
            self._cache is not None and
            (now - self._cache_time) < CLOUD_STATE_CACHE_TTL):
            return self._cache

        # Check all cloud mode indicators
        state = self._detect_cloud_mode()

        # v258.4: Enrich state with CPU pressure info
        cpu_active, cpu_pct = self.check_cpu_pressure()
        state.cpu_pressure_active = cpu_active
        state.cpu_pressure_percent = cpu_pct

        # Update cache
        self._cache = state
        self._cache_time = now

        return state

    def _detect_cloud_mode(self) -> CloudModeState:
        """
        Detect cloud mode from all available sources.

        Priority order:
        1. JARVIS_GCP_OFFLOAD_ACTIVE environment variable
        2. JARVIS_HOLLOW_CLIENT environment variable
        3. cloud_lock.json persistent state
        """
        # Check 1: GCP Offload environment variable (highest priority)
        if os.getenv("JARVIS_GCP_OFFLOAD_ACTIVE", "").lower() == "true":
            return CloudModeState(
                is_active=True,
                reason="GCP offload active via environment",
                source="env_var",
                gcp_endpoint=self._get_gcp_endpoint(),
            )

        # Check 2: Hollow Client mode
        if os.getenv("JARVIS_HOLLOW_CLIENT", "").lower() == "true":
            return CloudModeState(
                is_active=True,
                reason="Hollow Client mode active",
                source="hollow_client",
                gcp_endpoint=self._get_gcp_endpoint(),
            )

        # Check 3: Persistent cloud lock file
        try:
            if CLOUD_LOCK_FILE.exists():
                lock_data = json.loads(CLOUD_LOCK_FILE.read_text())
                if lock_data.get("locked", False):
                    return CloudModeState(
                        is_active=True,
                        reason=lock_data.get("reason", "Cloud lock active"),
                        source="cloud_lock",
                        gcp_endpoint=self._get_gcp_endpoint(),
                        metadata={
                            "oom_count": lock_data.get("oom_count", 0),
                            "consecutive_ooms": lock_data.get("consecutive_ooms", 0),
                            "hardware_ram_gb": lock_data.get("hardware_ram_gb"),
                            "lock_timestamp": lock_data.get("timestamp"),
                        },
                    )
        except Exception as e:
            logger.debug(f"[v152.0] Error reading cloud lock: {e}")

        # Not in cloud mode
        return CloudModeState(
            is_active=False,
            reason=None,
            source="none",
        )

    def _get_gcp_endpoint(self) -> Optional[str]:
        """Get the GCP endpoint URL if available."""
        # Priority 1: Direct Cloud Run URL
        cloud_run_url = os.getenv("JARVIS_PRIME_CLOUD_RUN_URL")
        if cloud_run_url:
            return cloud_run_url

        # Priority 2: Construct from GCP project
        gcp_project = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
        gcp_region = os.getenv("GCP_REGION", "us-central1")
        if gcp_project:
            return f"https://jarvis-prime-{gcp_region}-{gcp_project}.a.run.app"

        return None

    def get_effective_jarvis_url(self, default_local: str = "http://localhost:8000") -> str:
        """
        Get the effective JARVIS URL based on cloud mode.

        This is a convenience method for components that need to decide
        which endpoint to use.

        Args:
            default_local: Default local URL when not in cloud mode

        Returns:
            GCP endpoint if cloud mode active, otherwise default_local
        """
        state = self.get_cloud_state()

        if state.is_active and state.gcp_endpoint:
            logger.info(
                f"[v152.0] Cloud mode active ({state.reason}) - "
                f"using GCP endpoint: {state.gcp_endpoint}"
            )
            return state.gcp_endpoint

        if state.is_active:
            # Cloud mode but no GCP endpoint - return None to indicate skip
            logger.warning(
                f"[v152.0] Cloud mode active ({state.reason}) but no GCP endpoint - "
                f"service calls should be skipped"
            )
            return ""

        return default_local

    def should_skip_local_service(self, service_name: str = "jarvis") -> Tuple[bool, Optional[str]]:
        """
        Check if local service calls should be skipped.

        Returns:
            Tuple of (should_skip, reason)
        """
        state = self.get_cloud_state()

        if state.is_active:
            return True, state.reason

        return False, None

    def check_cpu_pressure(self) -> Tuple[bool, float]:
        """v258.4: Check supervisor CPU pressure signal.

        Reads the signal file written by the JARVIS supervisor when CPU
        exceeds 95% sustained. The signal has a TTL (default 60s) after
        which it is considered expired.

        Returns:
            Tuple of (is_active, cpu_percent). Non-blocking with file cache.
            is_active is True only when the signal exists, is not expired,
            and cpu_percent >= 95.0.
        """
        try:
            if not CPU_PRESSURE_SIGNAL_FILE.exists():
                return False, 0.0

            content = CPU_PRESSURE_SIGNAL_FILE.read_text()
            data = json.loads(content)

            timestamp = data.get("timestamp", 0)
            if time.time() - timestamp > CPU_PRESSURE_SIGNAL_TTL:
                return False, 0.0  # Expired signal

            cpu_pct = data.get("cpu_percent", 0.0)
            return cpu_pct >= 95.0, cpu_pct

        except Exception:
            return False, 0.0  # Non-fatal

    def check_supervisor_phase(self) -> Optional[Dict[str, Any]]:
        """v258.4: Check supervisor's current startup/runtime phase.

        Reads from ~/.jarvis/trinity/state/system_phase.json.
        """
        try:
            _phase_file = TRINITY_DIR / "state" / "system_phase.json"
            if not _phase_file.exists():
                return None

            content = _phase_file.read_text()
            phase_data = json.loads(content)

            timestamp = phase_data.get("timestamp", 0)
            if time.time() - timestamp > 600:  # 10 min TTL
                return None

            return phase_data

        except Exception:
            return None


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_detector: Optional[CloudModeDetector] = None


def get_cloud_mode_detector() -> CloudModeDetector:
    """Get the global CloudModeDetector instance."""
    global _detector
    if _detector is None:
        _detector = CloudModeDetector()
    return _detector


def is_cloud_mode_active() -> bool:
    """
    Quick check if cloud mode is active.

    Use this before attempting any local service connections.
    """
    return get_cloud_mode_detector().is_cloud_mode_active()


def get_cloud_state() -> CloudModeState:
    """Get detailed cloud mode state."""
    return get_cloud_mode_detector().get_cloud_state()


def get_effective_jarvis_url(default: str = "http://localhost:8000") -> str:
    """Get effective JARVIS URL based on cloud mode."""
    return get_cloud_mode_detector().get_effective_jarvis_url(default)


def should_skip_local_service(service_name: str = "jarvis") -> Tuple[bool, Optional[str]]:
    """Check if local service calls should be skipped."""
    return get_cloud_mode_detector().should_skip_local_service(service_name)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CloudModeDetector",
    "CloudModeState",
    "get_cloud_mode_detector",
    "is_cloud_mode_active",
    "get_cloud_state",
    "get_effective_jarvis_url",
    "should_skip_local_service",
]
