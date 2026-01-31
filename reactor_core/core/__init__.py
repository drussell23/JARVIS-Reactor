"""
Reactor Core - Core Module v152.0
==================================

Core utilities and detectors for Reactor Core.

v152.0: Added cloud mode detection for JARVIS ecosystem awareness.
"""

from __future__ import annotations

# v152.0: Cloud mode detection
from reactor_core.core.cloud_mode_detector import (
    CloudModeDetector,
    CloudModeState,
    get_cloud_mode_detector,
    is_cloud_mode_active,
    get_cloud_state,
    get_effective_jarvis_url,
    should_skip_local_service,
)

__all__ = [
    # Cloud mode detection
    "CloudModeDetector",
    "CloudModeState",
    "get_cloud_mode_detector",
    "is_cloud_mode_active",
    "get_cloud_state",
    "get_effective_jarvis_url",
    "should_skip_local_service",
]
