"""
Trinity Configuration Module for Reactor Core.

Provides centralized configuration for Trinity Orchestrator with:
- Environment variable support
- Sensible defaults
- Type-safe configuration objects
- Helper utilities for retry logic and timing

This is the local Reactor Core implementation. If JARVIS-AI-Agent's
backend.core.trinity_config is available, it will be used instead
for cross-repo consistency.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthConfig:
    """Health monitoring configuration."""

    heartbeat_timeout: float = 15.0
    health_check_interval: float = 5.0
    stale_threshold: float = 30.0

    def __post_init__(self):
        """Load from environment variables."""
        self.heartbeat_timeout = float(
            os.getenv("TRINITY_HEARTBEAT_TIMEOUT", str(self.heartbeat_timeout))
        )
        self.health_check_interval = float(
            os.getenv("TRINITY_HEALTH_CHECK_INTERVAL", str(self.health_check_interval))
        )
        self.stale_threshold = float(
            os.getenv("TRINITY_STALE_THRESHOLD", str(self.stale_threshold))
        )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_timeout: float = 10.0

    def __post_init__(self):
        """Load from environment variables."""
        self.failure_threshold = int(
            os.getenv("TRINITY_CIRCUIT_BREAKER_THRESHOLD", str(self.failure_threshold))
        )
        self.timeout_seconds = float(
            os.getenv("TRINITY_CIRCUIT_BREAKER_RESET", str(self.timeout_seconds))
        )
        self.half_open_timeout = float(
            os.getenv("TRINITY_CIRCUIT_BREAKER_HALF_OPEN", str(self.half_open_timeout))
        )


@dataclass
class CommandConfig:
    """Command routing configuration."""

    command_timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 5.0
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0

    def __post_init__(self):
        """Load from environment variables."""
        self.command_timeout = float(
            os.getenv("TRINITY_COMMAND_TIMEOUT", str(self.command_timeout))
        )
        self.max_retries = int(
            os.getenv("TRINITY_MAX_RETRIES", str(self.max_retries))
        )
        self.retry_delay = float(
            os.getenv("TRINITY_RETRY_DELAY", str(self.retry_delay))
        )
        self.max_retry_delay = float(
            os.getenv("TRINITY_MAX_RETRY_DELAY", str(self.max_retry_delay))
        )


@dataclass
class DeadLetterQueueConfig:
    """Dead Letter Queue configuration."""

    max_retries: int = 3
    retry_delay: float = 5.0
    exponential_backoff: bool = True
    max_age_hours: float = 24.0

    def __post_init__(self):
        """Load from environment variables."""
        self.max_retries = int(
            os.getenv("TRINITY_DLQ_MAX_RETRIES", str(self.max_retries))
        )
        self.retry_delay = float(
            os.getenv("TRINITY_DLQ_RETRY_DELAY", str(self.retry_delay))
        )
        self.max_age_hours = float(
            os.getenv("TRINITY_DLQ_MAX_AGE_HOURS", str(self.max_age_hours))
        )


@dataclass
class TrinityConfig:
    """
    Unified Trinity Orchestrator configuration.

    This provides all configuration values for the Trinity system with:
    - Environment variable overrides
    - Sensible defaults
    - Cross-repo consistency

    Attributes:
        trinity_dir: Base directory for Trinity state files
        health: Health monitoring configuration
        circuit_breaker: Circuit breaker configuration
        command: Command routing configuration
        dlq: Dead Letter Queue configuration
    """

    trinity_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")
    health: HealthConfig = field(default_factory=HealthConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    command: CommandConfig = field(default_factory=CommandConfig)
    dlq: DeadLetterQueueConfig = field(default_factory=DeadLetterQueueConfig)

    def __post_init__(self):
        """Load configuration from environment variables."""
        # Override trinity_dir from environment
        env_trinity_dir = os.getenv("TRINITY_DIR")
        if env_trinity_dir:
            self.trinity_dir = Path(env_trinity_dir)

        # Ensure directory exists
        self.trinity_dir.mkdir(parents=True, exist_ok=True)

    @property
    def orchestrator_state_file(self) -> Path:
        """Path to orchestrator state file."""
        return self.trinity_dir / "orchestrator_state.json"

    @property
    def components_dir(self) -> Path:
        """Directory for component heartbeat files."""
        path = self.trinity_dir / "components"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dlq_dir(self) -> Path:
        """Directory for dead letter queue."""
        path = self.trinity_dir / "dlq"
        path.mkdir(parents=True, exist_ok=True)
        return path


# Singleton instance
_config: Optional[TrinityConfig] = None


def get_config() -> TrinityConfig:
    """
    Get the global Trinity configuration instance.

    Returns:
        TrinityConfig: The singleton configuration instance
    """
    global _config
    if _config is None:
        _config = TrinityConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sleep_with_jitter(base_delay: float, max_jitter: float = 1.0) -> float:
    """
    Sleep for a duration with added jitter to prevent thundering herd.

    Args:
        base_delay: Base sleep duration in seconds
        max_jitter: Maximum random jitter to add (default 1.0 second)

    Returns:
        float: Actual sleep duration used
    """
    jitter = random.uniform(0, max_jitter)
    actual_delay = base_delay + jitter
    time.sleep(actual_delay)
    return actual_delay


async def async_sleep_with_jitter(base_delay: float, max_jitter: float = 1.0) -> float:
    """
    Async sleep for a duration with added jitter.

    Args:
        base_delay: Base sleep duration in seconds
        max_jitter: Maximum random jitter to add (default 1.0 second)

    Returns:
        float: Actual sleep duration used
    """
    jitter = random.uniform(0, max_jitter)
    actual_delay = base_delay + jitter
    await asyncio.sleep(actual_delay)
    return actual_delay


def get_retry_delay(
    attempt: int,
    base_delay: float = 5.0,
    exponential: bool = True,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """
    Calculate retry delay with exponential backoff and jitter.

    Args:
        attempt: Current retry attempt number (0-indexed)
        base_delay: Base delay in seconds (default 5.0)
        exponential: Whether to use exponential backoff (default True)
        max_delay: Maximum delay in seconds (default 60.0)
        jitter: Whether to add random jitter (default True)

    Returns:
        float: Calculated retry delay in seconds

    Examples:
        >>> get_retry_delay(0)  # First retry
        5.0  # (base_delay + jitter)

        >>> get_retry_delay(1)  # Second retry
        10.0  # (base_delay * 2 + jitter)

        >>> get_retry_delay(5)  # Sixth retry
        60.0  # (capped at max_delay + jitter)
    """
    if exponential:
        # Exponential backoff: base * 2^attempt
        delay = base_delay * (2 ** attempt)
    else:
        # Linear backoff: base * (attempt + 1)
        delay = base_delay * (attempt + 1)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter if requested
    if jitter:
        jitter_amount = random.uniform(0, base_delay * 0.1)  # 10% jitter
        delay += jitter_amount

    return delay


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "TrinityConfig",
    "HealthConfig",
    "CircuitBreakerConfig",
    "CommandConfig",
    "DeadLetterQueueConfig",
    "get_config",
    "reset_config",
    "sleep_with_jitter",
    "async_sleep_with_jitter",
    "get_retry_delay",
]
