"""
Training Pipeline Shutdown Integration v1.0
=============================================

Integrates the training subsystem with the async lifecycle coordinator
for graceful shutdown support.

Features:
- Coordinates experience receiver shutdown with training pipeline
- Ensures in-flight training completes or checkpoints
- Drains experience buffer before shutdown
- Publishes final MODEL_READY events
- Releases distributed locks cleanly

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   Training Shutdown Coordinator                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  Signal (SIGTERM/SIGINT)                                             │
    │         │                                                            │
    │         ▼                                                            │
    │  ┌────────────────────────────────────────────────────────────────┐ │
    │  │  Async Lifecycle Coordinator                                   │ │
    │  │  (Orchestrates shutdown phases)                                │ │
    │  └────────────────────┬───────────────────────────────────────────┘ │
    │                       │                                              │
    │         ┌─────────────┼─────────────┬─────────────┐                  │
    │         ▼             ▼             ▼             ▼                  │
    │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │
    │  │ Experience │ │  Training  │ │   Model    │ │   Dist.    │        │
    │  │  Receiver  │ │  Pipeline  │ │ Publisher  │ │   Locks    │        │
    │  │  (Drain)   │ │(Checkpoint)│ │  (Final)   │ │ (Release)  │        │
    │  └────────────┘ └────────────┘ └────────────┘ └────────────┘        │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    from reactor_core.training.shutdown_integration import (
        init_training_shutdown,
        register_training_shutdown_handlers,
    )

    # During initialization
    await init_training_shutdown()

    # The system will now gracefully shutdown on SIGTERM/SIGINT

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

SHUTDOWN_TIMEOUT = float(os.getenv("TRAINING_SHUTDOWN_TIMEOUT", "60.0"))
EXPERIENCE_DRAIN_TIMEOUT = float(os.getenv("EXPERIENCE_DRAIN_TIMEOUT", "30.0"))
CHECKPOINT_TIMEOUT = float(os.getenv("TRAINING_CHECKPOINT_TIMEOUT", "120.0"))


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ShutdownState:
    """Tracks shutdown progress."""
    initiated: bool = False
    initiated_at: Optional[float] = None
    reason: str = ""
    phases_completed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    completed: bool = False
    completed_at: Optional[float] = None


# =============================================================================
# TRAINING SHUTDOWN COORDINATOR
# =============================================================================

class TrainingShutdownCoordinator:
    """
    Coordinates graceful shutdown of the training subsystem.

    Shutdown phases (in order):
    1. Stop accepting new experiences
    2. Drain experience buffer
    3. Checkpoint in-flight training
    4. Publish final model events
    5. Release distributed locks
    6. Stop health monitoring
    """

    def __init__(self):
        self._state = ShutdownState()
        self._shutdown_callbacks: List[Callable[[], Any]] = []
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Component references (set during init)
        self._experience_receiver = None
        self._training_pipeline = None
        self._trinity_publisher = None
        self._health_monitor = None

    async def initialize(self) -> None:
        """Initialize and register signal handlers."""
        self._running = True
        self._setup_signal_handlers()
        logger.info("Training shutdown coordinator initialized")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            loop = asyncio.get_running_loop()

            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(
                        self._handle_signal(s)
                    ),
                )
            logger.info("Signal handlers registered (SIGTERM, SIGINT)")

        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Signal handlers not available on this platform")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        sig_name = sig.name if hasattr(sig, 'name') else str(sig)
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        await self.shutdown(reason=f"signal:{sig_name}")

    def register_callback(self, callback: Callable[[], Any]) -> None:
        """Register a callback to be called during shutdown."""
        self._shutdown_callbacks.append(callback)

    async def shutdown(self, reason: str = "requested") -> bool:
        """
        Initiate graceful shutdown.

        Args:
            reason: Reason for shutdown

        Returns:
            True if shutdown completed successfully
        """
        if self._state.initiated:
            logger.warning("Shutdown already in progress")
            return False

        self._state.initiated = True
        self._state.initiated_at = time.time()
        self._state.reason = reason

        logger.info(f"Training shutdown initiated: {reason}")

        try:
            # Phase 1: Stop accepting new experiences
            await self._phase_stop_accepting()

            # Phase 2: Drain experience buffer
            await self._phase_drain_experiences()

            # Phase 3: Checkpoint training
            await self._phase_checkpoint_training()

            # Phase 4: Publish final events
            await self._phase_publish_final_events()

            # Phase 5: Release locks
            await self._phase_release_locks()

            # Phase 6: Stop health monitoring
            await self._phase_stop_health_monitoring()

            # Run registered callbacks
            await self._run_callbacks()

            self._state.completed = True
            self._state.completed_at = time.time()

            duration = self._state.completed_at - self._state.initiated_at
            logger.info(
                f"Training shutdown completed in {duration:.2f}s "
                f"(phases: {', '.join(self._state.phases_completed)})"
            )

            # Signal completion
            self._shutdown_event.set()
            return True

        except Exception as e:
            self._state.errors.append(str(e))
            logger.error(f"Shutdown error: {e}")
            self._shutdown_event.set()
            return False

    async def _phase_stop_accepting(self) -> None:
        """Phase 1: Stop accepting new experiences."""
        logger.info("Phase 1: Stopping experience ingestion...")

        try:
            from reactor_core.integration.trinity_experience_receiver import (
                get_experience_receiver,
            )
            receiver = await get_experience_receiver()
            receiver._running = False  # Stop the watch loop
            self._experience_receiver = receiver
            self._state.phases_completed.append("stop_accepting")
            logger.info("Phase 1 complete: Experience ingestion stopped")

        except Exception as e:
            logger.warning(f"Phase 1 error (non-fatal): {e}")

    async def _phase_drain_experiences(self) -> None:
        """Phase 2: Drain remaining experiences to training pipeline."""
        logger.info("Phase 2: Draining experience buffer...")

        try:
            if self._experience_receiver:
                await asyncio.wait_for(
                    self._experience_receiver.stop(drain_timeout=EXPERIENCE_DRAIN_TIMEOUT),
                    timeout=EXPERIENCE_DRAIN_TIMEOUT + 5,
                )
                self._state.phases_completed.append("drain_experiences")
                logger.info("Phase 2 complete: Experience buffer drained")

        except asyncio.TimeoutError:
            logger.warning("Phase 2 timeout: Some experiences may be lost")
            self._state.errors.append("Experience drain timeout")

        except Exception as e:
            logger.warning(f"Phase 2 error (non-fatal): {e}")

    async def _phase_checkpoint_training(self) -> None:
        """Phase 3: Checkpoint in-flight training."""
        logger.info("Phase 3: Checkpointing training state...")

        try:
            from reactor_core.training.unified_pipeline import get_unified_trainer

            trainer = get_unified_trainer()
            self._training_pipeline = trainer

            if hasattr(trainer, 'checkpoint'):
                await asyncio.wait_for(
                    trainer.checkpoint(reason="shutdown"),
                    timeout=CHECKPOINT_TIMEOUT,
                )
                logger.info("Phase 3 complete: Training checkpointed")
            elif hasattr(trainer, 'stop'):
                await asyncio.wait_for(
                    trainer.stop(),
                    timeout=CHECKPOINT_TIMEOUT,
                )
                logger.info("Phase 3 complete: Training stopped")

            self._state.phases_completed.append("checkpoint_training")

        except asyncio.TimeoutError:
            logger.warning("Phase 3 timeout: Checkpoint may be incomplete")
            self._state.errors.append("Checkpoint timeout")

        except ImportError:
            logger.debug("Training pipeline not available")

        except Exception as e:
            logger.warning(f"Phase 3 error (non-fatal): {e}")

    async def _phase_publish_final_events(self) -> None:
        """Phase 4: Publish final model events."""
        logger.info("Phase 4: Publishing final events...")

        try:
            from reactor_core.integration.trinity_publisher import (
                get_trinity_publisher,
                shutdown_trinity_publisher,
            )

            publisher = await get_trinity_publisher()
            self._trinity_publisher = publisher

            # Publish shutdown event
            if hasattr(publisher, 'publish_event'):
                await publisher.publish_event(
                    event_type="training_shutdown",
                    payload={
                        "reason": self._state.reason,
                        "timestamp": time.time(),
                    },
                )

            # Shutdown publisher (flushes DLQ)
            await shutdown_trinity_publisher()

            self._state.phases_completed.append("publish_final")
            logger.info("Phase 4 complete: Final events published")

        except ImportError:
            logger.debug("Trinity publisher not available")

        except Exception as e:
            logger.warning(f"Phase 4 error (non-fatal): {e}")

    async def _phase_release_locks(self) -> None:
        """Phase 5: Release all distributed locks."""
        logger.info("Phase 5: Releasing distributed locks...")

        try:
            from reactor_core.utils.distributed_lock import shutdown_distributed_locks

            await shutdown_distributed_locks()

            self._state.phases_completed.append("release_locks")
            logger.info("Phase 5 complete: Distributed locks released")

        except ImportError:
            logger.debug("Distributed locks not available")

        except Exception as e:
            logger.warning(f"Phase 5 error (non-fatal): {e}")

    async def _phase_stop_health_monitoring(self) -> None:
        """Phase 6: Stop health monitoring."""
        logger.info("Phase 6: Stopping health monitoring...")

        try:
            from reactor_core.api.training_health import shutdown_training_health_monitor

            await shutdown_training_health_monitor()

            self._state.phases_completed.append("stop_health")
            logger.info("Phase 6 complete: Health monitoring stopped")

        except ImportError:
            logger.debug("Health monitor not available")

        except Exception as e:
            logger.warning(f"Phase 6 error (non-fatal): {e}")

    async def _run_callbacks(self) -> None:
        """Run registered shutdown callbacks."""
        for callback in self._shutdown_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")
                self._state.errors.append(f"Callback error: {e}")

    async def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for shutdown to complete.

        Args:
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            True if shutdown completed, False if timeout
        """
        try:
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=timeout,
            )
            return self._state.completed
        except asyncio.TimeoutError:
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get current shutdown state."""
        return {
            "initiated": self._state.initiated,
            "initiated_at": self._state.initiated_at,
            "reason": self._state.reason,
            "phases_completed": self._state.phases_completed,
            "errors": self._state.errors,
            "completed": self._state.completed,
            "completed_at": self._state.completed_at,
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_coordinator: Optional[TrainingShutdownCoordinator] = None


def get_shutdown_coordinator() -> TrainingShutdownCoordinator:
    """Get global shutdown coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = TrainingShutdownCoordinator()
    return _coordinator


async def init_training_shutdown() -> TrainingShutdownCoordinator:
    """Initialize training shutdown coordinator."""
    coordinator = get_shutdown_coordinator()
    await coordinator.initialize()
    return coordinator


async def request_shutdown(reason: str = "requested") -> bool:
    """Request graceful shutdown."""
    coordinator = get_shutdown_coordinator()
    return await coordinator.shutdown(reason=reason)


async def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """Wait for shutdown to complete."""
    coordinator = get_shutdown_coordinator()
    return await coordinator.wait_for_shutdown(timeout=timeout)


def register_shutdown_callback(callback: Callable[[], Any]) -> None:
    """Register a callback to be called during shutdown."""
    coordinator = get_shutdown_coordinator()
    coordinator.register_callback(callback)


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class TrainingShutdownContext:
    """
    Context manager for training with graceful shutdown support.

    Usage:
        async with TrainingShutdownContext():
            await run_training()
            # Automatically handles shutdown on exit or signal
    """

    def __init__(self, shutdown_timeout: float = SHUTDOWN_TIMEOUT):
        self._timeout = shutdown_timeout
        self._coordinator: Optional[TrainingShutdownCoordinator] = None

    async def __aenter__(self) -> TrainingShutdownCoordinator:
        self._coordinator = await init_training_shutdown()
        return self._coordinator

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._coordinator and not self._coordinator._state.initiated:
            reason = "context_exit"
            if exc_type is not None:
                reason = f"exception:{exc_type.__name__}"

            await self._coordinator.shutdown(reason=reason)
            await self._coordinator.wait_for_shutdown(timeout=self._timeout)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "TrainingShutdownCoordinator",
    "ShutdownState",
    # Global functions
    "get_shutdown_coordinator",
    "init_training_shutdown",
    "request_shutdown",
    "wait_for_shutdown",
    "register_shutdown_callback",
    # Context manager
    "TrainingShutdownContext",
]
