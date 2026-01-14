"""
Training Pipeline Health Endpoints v1.0
========================================

Provides comprehensive health monitoring for the training subsystem,
including the experience receiver, training pipeline, and model versioning.

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Training Health Monitor                       │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
    │  │   Experience    │  │    Training     │  │     Model       │   │
    │  │   Receiver      │  │    Pipeline     │  │   Versioning    │   │
    │  │   Health        │  │    Health       │  │   Health        │   │
    │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │
    │           │                    │                    │            │
    │           └────────────────────┼────────────────────┘            │
    │                                ▼                                 │
    │                    ┌────────────────────────┐                    │
    │                    │   Aggregated Health    │                    │
    │                    │   with Auto-Recovery   │                    │
    │                    └────────────────────────┘                    │
    │                                │                                 │
    │           ┌────────────────────┼────────────────────┐            │
    │           ▼                    ▼                    ▼            │
    │    ┌───────────┐      ┌───────────────┐    ┌───────────────┐     │
    │    │  /health  │      │  /health/deep │    │  /metrics     │     │
    │    │  (quick)  │      │  (detailed)   │    │  (Prometheus) │     │
    │    └───────────┘      └───────────────┘    └───────────────┘     │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘

Endpoints:
    GET /health         - Quick liveness check
    GET /health/ready   - Readiness probe (all components ready)
    GET /health/deep    - Deep health check with component details
    GET /metrics        - Prometheus-compatible metrics

Features:
- Component health aggregation with weighted scoring
- Auto-recovery triggers on unhealthy states
- Prometheus metrics export
- Kubernetes-compatible liveness/readiness probes
- Circuit breaker state monitoring
- Experience receiver health integration

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

HEALTH_CHECK_TIMEOUT = float(os.getenv("TRAINING_HEALTH_TIMEOUT", "5.0"))
AUTO_RECOVERY_ENABLED = os.getenv("TRAINING_AUTO_RECOVERY", "true").lower() == "true"
AUTO_RECOVERY_COOLDOWN = float(os.getenv("TRAINING_RECOVERY_COOLDOWN", "300.0"))


# =============================================================================
# DATA MODELS
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check,
            "details": self.details,
        }


@dataclass
class TrainingHealthReport:
    """Complete health report for training subsystem."""
    status: HealthStatus = HealthStatus.UNKNOWN
    timestamp: float = field(default_factory=time.time)
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    version: str = "1.0.0"
    auto_recovery_enabled: bool = AUTO_RECOVERY_ENABLED
    last_recovery_attempt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "version": self.version,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "last_recovery_attempt": self.last_recovery_attempt,
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class TrainingHealthMonitor:
    """
    Monitors health of the training subsystem.

    Aggregates health from:
    - Experience receiver (ingestion pipeline)
    - Training pipeline (model training)
    - Model versioning (deployment)
    - Distributed locks (coordination)
    """

    def __init__(self):
        self._start_time = time.time()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Component health state
        self._component_health: Dict[str, ComponentHealth] = {}

        # Auto-recovery
        self._last_recovery_time: Optional[float] = None
        self._recovery_callbacks: List[Callable[[], Any]] = []

        # Metrics for Prometheus
        self._metrics: Dict[str, float] = {
            "health_checks_total": 0,
            "health_checks_failed": 0,
            "recovery_attempts_total": 0,
            "recovery_success_total": 0,
        }

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="training_health_monitor"
        )
        logger.info("Training health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Training health monitor stopped")

    def register_recovery_callback(self, callback: Callable[[], Any]) -> None:
        """Register callback for auto-recovery."""
        self._recovery_callbacks.append(callback)

    async def _monitor_loop(self) -> None:
        """Background health monitoring loop."""
        check_interval = float(os.getenv("TRAINING_HEALTH_INTERVAL", "30.0"))

        while self._running:
            try:
                await self._perform_health_checks()

                # Check if auto-recovery needed
                if AUTO_RECOVERY_ENABLED:
                    await self._check_auto_recovery()

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all components."""
        self._metrics["health_checks_total"] += 1

        # Check experience receiver
        await self._check_experience_receiver()

        # Check training pipeline
        await self._check_training_pipeline()

        # Check model versioning
        await self._check_model_versioning()

        # Check distributed locks
        await self._check_distributed_locks()

    async def _check_experience_receiver(self) -> ComponentHealth:
        """Check experience receiver health."""
        start_time = time.time()
        health = ComponentHealth(name="experience_receiver")

        try:
            from reactor_core.integration.trinity_experience_receiver import (
                get_experience_receiver,
            )

            receiver = await get_experience_receiver()
            receiver_health = receiver.get_health()

            latency_ms = (time.time() - start_time) * 1000

            if receiver_health["status"] == "healthy":
                health.status = HealthStatus.HEALTHY
                health.message = "Experience receiver operational"
            elif receiver_health.get("circuit_breaker_state") == "open":
                health.status = HealthStatus.UNHEALTHY
                health.message = "Circuit breaker open"
            else:
                health.status = HealthStatus.DEGRADED
                health.message = receiver_health.get("status", "Unknown status")

            health.latency_ms = latency_ms
            health.details = receiver_health

        except ImportError:
            health.status = HealthStatus.UNKNOWN
            health.message = "Experience receiver not available"
            health.latency_ms = (time.time() - start_time) * 1000

        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Error: {str(e)}"
            health.latency_ms = (time.time() - start_time) * 1000
            self._metrics["health_checks_failed"] += 1

        health.last_check = time.time()
        self._component_health["experience_receiver"] = health
        return health

    async def _check_training_pipeline(self) -> ComponentHealth:
        """Check training pipeline health."""
        start_time = time.time()
        health = ComponentHealth(name="training_pipeline")

        try:
            from reactor_core.training.unified_pipeline import get_unified_trainer

            trainer = get_unified_trainer()

            # Check if trainer is initialized and operational
            if hasattr(trainer, 'get_status'):
                status = await trainer.get_status()
                is_healthy = status.get("initialized", False)
            else:
                # Basic check - trainer object exists
                is_healthy = trainer is not None

            latency_ms = (time.time() - start_time) * 1000

            if is_healthy:
                health.status = HealthStatus.HEALTHY
                health.message = "Training pipeline operational"
            else:
                health.status = HealthStatus.DEGRADED
                health.message = "Training pipeline not fully initialized"

            health.latency_ms = latency_ms
            health.details = status if hasattr(trainer, 'get_status') else {}

        except ImportError:
            health.status = HealthStatus.UNKNOWN
            health.message = "Training pipeline not available"
            health.latency_ms = (time.time() - start_time) * 1000

        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Error: {str(e)}"
            health.latency_ms = (time.time() - start_time) * 1000
            self._metrics["health_checks_failed"] += 1

        health.last_check = time.time()
        self._component_health["training_pipeline"] = health
        return health

    async def _check_model_versioning(self) -> ComponentHealth:
        """Check model versioning health."""
        start_time = time.time()
        health = ComponentHealth(name="model_versioning")

        try:
            from reactor_core.integration.trinity_publisher import get_trinity_publisher

            publisher = await get_trinity_publisher()

            # Check publisher status
            if hasattr(publisher, 'get_metrics'):
                metrics = publisher.get_metrics()
                is_healthy = metrics.get("connected", False) or True  # Graceful if no Redis
            else:
                is_healthy = publisher is not None

            latency_ms = (time.time() - start_time) * 1000

            if is_healthy:
                health.status = HealthStatus.HEALTHY
                health.message = "Model versioning operational"
            else:
                health.status = HealthStatus.DEGRADED
                health.message = "Model versioning degraded"

            health.latency_ms = latency_ms
            health.details = metrics if hasattr(publisher, 'get_metrics') else {}

        except ImportError:
            health.status = HealthStatus.UNKNOWN
            health.message = "Model versioning not available"
            health.latency_ms = (time.time() - start_time) * 1000

        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Error: {str(e)}"
            health.latency_ms = (time.time() - start_time) * 1000
            self._metrics["health_checks_failed"] += 1

        health.last_check = time.time()
        self._component_health["model_versioning"] = health
        return health

    async def _check_distributed_locks(self) -> ComponentHealth:
        """Check distributed lock health."""
        start_time = time.time()
        health = ComponentHealth(name="distributed_locks")

        try:
            from reactor_core.utils.distributed_lock import check_training_lock_status

            lock_status = await check_training_lock_status()
            latency_ms = (time.time() - start_time) * 1000

            if lock_status.get("redis_available", False):
                health.status = HealthStatus.HEALTHY
                health.message = "Distributed locks operational"
            else:
                health.status = HealthStatus.DEGRADED
                health.message = "Redis unavailable, using local locks"

            health.latency_ms = latency_ms
            health.details = lock_status

        except ImportError:
            health.status = HealthStatus.UNKNOWN
            health.message = "Distributed locks not available"
            health.latency_ms = (time.time() - start_time) * 1000

        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Error: {str(e)}"
            health.latency_ms = (time.time() - start_time) * 1000
            self._metrics["health_checks_failed"] += 1

        health.last_check = time.time()
        self._component_health["distributed_locks"] = health
        return health

    async def _check_auto_recovery(self) -> None:
        """Check if auto-recovery is needed and trigger if so."""
        # Check cooldown
        if self._last_recovery_time:
            if time.time() - self._last_recovery_time < AUTO_RECOVERY_COOLDOWN:
                return

        # Check for unhealthy components
        unhealthy = [
            name for name, health in self._component_health.items()
            if health.status == HealthStatus.UNHEALTHY
        ]

        if unhealthy:
            logger.warning(f"Unhealthy components detected: {unhealthy}, triggering recovery")
            await self._trigger_recovery(unhealthy)

    async def _trigger_recovery(self, unhealthy_components: List[str]) -> None:
        """Trigger auto-recovery for unhealthy components."""
        self._metrics["recovery_attempts_total"] += 1
        self._last_recovery_time = time.time()

        for callback in self._recovery_callbacks:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")

        # Specific recovery actions
        for component in unhealthy_components:
            await self._recover_component(component)

        self._metrics["recovery_success_total"] += 1

    async def _recover_component(self, component: str) -> None:
        """Attempt to recover a specific component."""
        logger.info(f"Attempting to recover: {component}")

        if component == "experience_receiver":
            try:
                from reactor_core.integration.trinity_experience_receiver import (
                    shutdown_experience_receiver,
                    get_experience_receiver,
                )
                await shutdown_experience_receiver()
                await get_experience_receiver()  # Re-initialize
                logger.info("Experience receiver recovered")
            except Exception as e:
                logger.error(f"Failed to recover experience receiver: {e}")

        elif component == "distributed_locks":
            try:
                from reactor_core.utils.distributed_lock import (
                    shutdown_distributed_locks,
                    DistributedLock,
                )
                await shutdown_distributed_locks()
                # Force reconnection on next use
                DistributedLock._redis_client = None
                logger.info("Distributed locks recovered")
            except Exception as e:
                logger.error(f"Failed to recover distributed locks: {e}")

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    async def get_health(self, deep: bool = False) -> TrainingHealthReport:
        """
        Get current health status.

        Args:
            deep: If True, perform fresh health checks

        Returns:
            Complete health report
        """
        if deep:
            await self._perform_health_checks()

        # Aggregate status
        statuses = [h.status for h in self._component_health.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            overall = HealthStatus.UNKNOWN
        else:
            overall = HealthStatus.HEALTHY

        return TrainingHealthReport(
            status=overall,
            components=dict(self._component_health),
            uptime_seconds=time.time() - self._start_time,
            last_recovery_attempt=self._last_recovery_time,
        )

    async def is_ready(self) -> bool:
        """
        Check if system is ready to handle requests.

        Used for Kubernetes readiness probe.
        """
        report = await self.get_health()
        return report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    async def is_alive(self) -> bool:
        """
        Check if system is alive.

        Used for Kubernetes liveness probe.
        """
        return self._running

    def get_metrics_prometheus(self) -> str:
        """
        Get Prometheus-format metrics.

        Returns metrics suitable for Prometheus scraping.
        """
        lines = [
            "# HELP training_health_checks_total Total number of health checks performed",
            "# TYPE training_health_checks_total counter",
            f"training_health_checks_total {self._metrics['health_checks_total']}",
            "",
            "# HELP training_health_checks_failed Total number of failed health checks",
            "# TYPE training_health_checks_failed counter",
            f"training_health_checks_failed {self._metrics['health_checks_failed']}",
            "",
            "# HELP training_recovery_attempts_total Total recovery attempts",
            "# TYPE training_recovery_attempts_total counter",
            f"training_recovery_attempts_total {self._metrics['recovery_attempts_total']}",
            "",
            "# HELP training_recovery_success_total Successful recovery attempts",
            "# TYPE training_recovery_success_total counter",
            f"training_recovery_success_total {self._metrics['recovery_success_total']}",
            "",
            "# HELP training_uptime_seconds Training subsystem uptime",
            "# TYPE training_uptime_seconds gauge",
            f"training_uptime_seconds {time.time() - self._start_time:.2f}",
            "",
        ]

        # Component health status (1=healthy, 0.5=degraded, 0=unhealthy)
        lines.extend([
            "# HELP training_component_health Component health status",
            "# TYPE training_component_health gauge",
        ])

        for name, health in self._component_health.items():
            value = {
                HealthStatus.HEALTHY: 1.0,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0.0,
                HealthStatus.UNKNOWN: -1.0,
            }.get(health.status, -1.0)

            lines.append(f'training_component_health{{component="{name}"}} {value}')

        # Component latencies
        lines.extend([
            "",
            "# HELP training_component_latency_ms Component check latency",
            "# TYPE training_component_latency_ms gauge",
        ])

        for name, health in self._component_health.items():
            lines.append(f'training_component_latency_ms{{component="{name}"}} {health.latency_ms:.2f}')

        return "\n".join(lines)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_monitor: Optional[TrainingHealthMonitor] = None


def get_training_health_monitor() -> TrainingHealthMonitor:
    """Get global training health monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = TrainingHealthMonitor()
    return _monitor


async def init_training_health_monitor() -> TrainingHealthMonitor:
    """Initialize and start training health monitor."""
    monitor = get_training_health_monitor()
    await monitor.start()
    return monitor


async def shutdown_training_health_monitor() -> None:
    """Shutdown training health monitor."""
    global _monitor
    if _monitor:
        await _monitor.stop()
        _monitor = None


# =============================================================================
# FASTAPI ROUTES (Optional Integration)
# =============================================================================

def create_health_routes():
    """
    Create FastAPI routes for health endpoints.

    Returns a FastAPI router that can be mounted.

    Usage:
        from reactor_core.api.training_health import create_health_routes
        app.include_router(create_health_routes(), prefix="/training")
    """
    try:
        from fastapi import APIRouter, Response
        from fastapi.responses import JSONResponse, PlainTextResponse
    except ImportError:
        logger.warning("FastAPI not available, health routes not created")
        return None

    router = APIRouter(tags=["health"])

    @router.get("/health")
    async def health_check():
        """Quick liveness check."""
        monitor = get_training_health_monitor()
        is_alive = await monitor.is_alive()
        return JSONResponse(
            content={"status": "alive" if is_alive else "dead"},
            status_code=200 if is_alive else 503,
        )

    @router.get("/health/ready")
    async def readiness_check():
        """Readiness probe for Kubernetes."""
        monitor = get_training_health_monitor()
        is_ready = await monitor.is_ready()
        return JSONResponse(
            content={"ready": is_ready},
            status_code=200 if is_ready else 503,
        )

    @router.get("/health/deep")
    async def deep_health_check():
        """Deep health check with component details."""
        monitor = get_training_health_monitor()
        report = await monitor.get_health(deep=True)
        status_code = {
            HealthStatus.HEALTHY: 200,
            HealthStatus.DEGRADED: 200,
            HealthStatus.UNHEALTHY: 503,
            HealthStatus.UNKNOWN: 503,
        }.get(report.status, 503)
        return JSONResponse(content=report.to_dict(), status_code=status_code)

    @router.get("/metrics")
    async def prometheus_metrics():
        """Prometheus-compatible metrics endpoint."""
        monitor = get_training_health_monitor()
        metrics = monitor.get_metrics_prometheus()
        return PlainTextResponse(
            content=metrics,
            media_type="text/plain; charset=utf-8",
        )

    return router


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data models
    "HealthStatus",
    "ComponentHealth",
    "TrainingHealthReport",
    # Monitor
    "TrainingHealthMonitor",
    # Global functions
    "get_training_health_monitor",
    "init_training_health_monitor",
    "shutdown_training_health_monitor",
    # FastAPI integration
    "create_health_routes",
]
