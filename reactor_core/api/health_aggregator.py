"""
Unified Health Aggregation Dashboard for AGI OS.

Provides cross-repo health monitoring, alerting, and dashboard
data for the JARVIS, Prime, and Reactor-Core ecosystem.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Health Aggregator                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
    │  │    JARVIS    │  │    Prime     │  │   Reactor-Core   │   │
    │  │   Checker    │  │   Checker    │  │     Checker      │   │
    │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
    │         │                 │                   │              │
    │         └─────────────────┼───────────────────┘              │
    │                           ▼                                  │
    │               ┌────────────────────────┐                     │
    │               │   Health Aggregation   │                     │
    │               │   (status, trends,     │                     │
    │               │    SLA tracking)       │                     │
    │               └───────────┬────────────┘                     │
    │                           │                                  │
    │         ┌─────────────────┼─────────────────┐                │
    │         ▼                 ▼                 ▼                │
    │  ┌─────────────┐  ┌─────────────────┐  ┌───────────────┐    │
    │  │   Alerting  │  │   Dashboard     │  │   SLA Report  │    │
    │  │   System    │  │   API           │  │   Generator   │    │
    │  └─────────────┘  └─────────────────┘  └───────────────┘    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

Features:
- Parallel health checks across all AGI OS components
- Historical health data with trend analysis
- SLA tracking and uptime calculation
- Multi-channel alerting (webhook, email templates)
- Circuit breaker for failing health endpoints
- Dashboard data aggregation with caching
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import statistics
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

# v152.0: Import cloud mode detector for JARVIS ecosystem awareness
try:
    from reactor_core.core.cloud_mode_detector import (
        is_cloud_mode_active,
        get_effective_jarvis_url,
        should_skip_local_service,
    )
    CLOUD_MODE_DETECTOR_AVAILABLE = True
except ImportError:
    CLOUD_MODE_DETECTOR_AVAILABLE = False
    def is_cloud_mode_active() -> bool:
        return os.getenv("JARVIS_GCP_OFFLOAD_ACTIVE", "false").lower() == "true"
    def get_effective_jarvis_url(default: str = "http://localhost:8000") -> str:
        if is_cloud_mode_active():
            return os.getenv("JARVIS_PRIME_CLOUD_RUN_URL", "")
        return default
    def should_skip_local_service(service_name: str = "jarvis") -> tuple:
        if is_cloud_mode_active():
            return True, "Cloud mode active"
        return False, None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class HealthConfig:
    """Health aggregator configuration."""

    # v152.0: Cloud-aware endpoint resolution
    # When cloud mode is active, use GCP endpoints or skip local checks entirely

    @staticmethod
    def _get_jarvis_health_url() -> str:
        """v152.0: Get JARVIS health URL, respecting cloud mode."""
        env_url = os.getenv("JARVIS_HEALTH_URL")
        if env_url:
            return env_url
        base_url = get_effective_jarvis_url("http://localhost:8000")
        return f"{base_url}/health" if base_url else ""

    @staticmethod
    def _get_prime_health_url() -> str:
        """v152.0: Get Prime health URL, respecting cloud mode."""
        env_url = os.getenv("PRIME_HEALTH_URL")
        if env_url:
            return env_url
        # Prime is same as JARVIS in cloud mode (unified endpoint)
        if is_cloud_mode_active():
            base_url = get_effective_jarvis_url("")
            return f"{base_url}/health" if base_url else ""
        return "http://localhost:8001/health"

    # Endpoints - use property-like access via class methods
    JARVIS_HEALTH_URL = os.getenv("JARVIS_HEALTH_URL", "http://localhost:8000/health")
    PRIME_HEALTH_URL = os.getenv("PRIME_HEALTH_URL", "http://localhost:8001/health")
    REACTOR_HEALTH_URL = os.getenv("REACTOR_HEALTH_URL", "http://localhost:8003/health")

    # Check intervals
    CHECK_INTERVAL_SECONDS = float(os.getenv("HEALTH_CHECK_INTERVAL", "30.0"))
    TIMEOUT_SECONDS = float(os.getenv("HEALTH_CHECK_TIMEOUT", "10.0"))

    # Thresholds
    UNHEALTHY_THRESHOLD = int(os.getenv("HEALTH_UNHEALTHY_THRESHOLD", "3"))  # consecutive failures
    DEGRADED_LATENCY_MS = float(os.getenv("HEALTH_DEGRADED_LATENCY", "500.0"))

    # History retention
    HISTORY_RETENTION_HOURS = int(os.getenv("HEALTH_HISTORY_HOURS", "24"))
    MAX_HISTORY_POINTS = int(os.getenv("HEALTH_MAX_HISTORY", "2880"))  # 24h at 30s intervals

    # Alerting
    ALERT_COOLDOWN_SECONDS = float(os.getenv("HEALTH_ALERT_COOLDOWN", "300.0"))  # 5 minutes

    # SLA
    SLA_TARGET_UPTIME = float(os.getenv("HEALTH_SLA_TARGET", "99.9"))  # 99.9%

    # Startup-aware configuration (v93.4)
    STARTUP_GRACE_PERIOD = float(os.getenv("TRINITY_STARTUP_GRACE_PERIOD", "120.0"))
    STARTUP_UNHEALTHY_MULTIPLIER = float(os.getenv("HEALTH_STARTUP_THRESHOLD_MULTIPLIER", "5.0"))
    STARTUP_CHECK_INTERVAL = float(os.getenv("HEALTH_STARTUP_CHECK_INTERVAL", "10.0"))
    CIRCUIT_HALF_OPEN_SECONDS = float(os.getenv("HEALTH_CIRCUIT_HALF_OPEN", "30.0"))
    STARTUP_CIRCUIT_HALF_OPEN_SECONDS = float(os.getenv("HEALTH_STARTUP_CIRCUIT_HALF_OPEN", "10.0"))


# ============================================================================
# Data Models
# ============================================================================

class HealthStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Single health check result."""
    component: str
    status: HealthStatus
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "message": self.message,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class ComponentHealth:
    """Aggregated health for a component."""
    component: str
    current_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_check: Optional[HealthCheck] = None
    last_healthy: Optional[float] = None
    last_unhealthy: Optional[float] = None
    uptime_percent: float = 100.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    check_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "current_status": self.current_status.value,
            "consecutive_failures": self.consecutive_failures,
            "last_check": self.last_check.to_dict() if self.last_check else None,
            "last_healthy": self.last_healthy,
            "last_unhealthy": self.last_unhealthy,
            "uptime_percent": round(self.uptime_percent, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "check_count": self.check_count,
            "failure_count": self.failure_count,
        }


@dataclass
class HealthAlert:
    """Health alert notification."""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    component: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    title: str = ""
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "severity": self.severity.value,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }


@dataclass
class SLAReport:
    """SLA compliance report."""
    component: str
    period_start: float
    period_end: float
    uptime_percent: float
    target_uptime: float
    is_compliant: bool
    total_downtime_seconds: float
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


@dataclass
class DashboardData:
    """Aggregated dashboard data."""
    timestamp: float = field(default_factory=time.time)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    active_alerts: List[HealthAlert] = field(default_factory=list)
    recent_incidents: List[Dict[str, Any]] = field(default_factory=list)
    uptime_24h: Dict[str, float] = field(default_factory=dict)
    latency_trend: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "overall_status": self.overall_status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "recent_incidents": self.recent_incidents,
            "uptime_24h": self.uptime_24h,
            "latency_trend": self.latency_trend,
        }


# ============================================================================
# Component Health Checker
# ============================================================================

class ComponentChecker:
    """
    Health checker for a single component.

    Supports HTTP health endpoints with configurable timeout
    and circuit breaker pattern.

    v93.4: Startup-aware circuit breaker and failure thresholds.
    """

    def __init__(
        self,
        component: str,
        health_url: str,
        timeout: float = HealthConfig.TIMEOUT_SECONDS,
        aggregator_start_time: Optional[float] = None,
    ):
        self.component = component
        self.health_url = health_url
        self.timeout = timeout
        self._aggregator_start_time = aggregator_start_time or time.time()

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_opened_at: float = 0

        # Statistics
        self._consecutive_failures = 0
        self._latencies: Deque[float] = deque(maxlen=100)

    def is_in_startup_phase(self) -> bool:
        """Check if we're still in the startup grace period."""
        return (time.time() - self._aggregator_start_time) < HealthConfig.STARTUP_GRACE_PERIOD

    def get_circuit_half_open_seconds(self) -> float:
        """Get circuit half-open timeout, shorter during startup for faster recovery."""
        if self.is_in_startup_phase():
            return HealthConfig.STARTUP_CIRCUIT_HALF_OPEN_SECONDS
        return HealthConfig.CIRCUIT_HALF_OPEN_SECONDS

    def get_unhealthy_threshold(self) -> int:
        """Get unhealthy threshold, more lenient during startup."""
        base_threshold = HealthConfig.UNHEALTHY_THRESHOLD
        if self.is_in_startup_phase():
            return int(base_threshold * HealthConfig.STARTUP_UNHEALTHY_MULTIPLIER)
        return base_threshold

    def update_aggregator_start_time(self, start_time: float):
        """Update the aggregator start time for startup phase detection."""
        self._aggregator_start_time = start_time

    async def check(self) -> HealthCheck:
        """Perform health check."""
        start_time = time.time()

        # v152.0: Check cloud mode before attempting local service checks
        # When cloud mode is active, local JARVIS/Prime are intentionally offline
        if self.component in ("jarvis", "prime"):
            skip_local, reason = should_skip_local_service(self.component)
            if skip_local:
                logger.debug(
                    f"[Health] [v152.0] Skipping {self.component} local check: {reason}"
                )
                # Return a special "expected offline" status instead of unhealthy
                # This prevents circuit breaker from recording failures for intentionally disabled services
                return HealthCheck(
                    component=self.component,
                    status=HealthStatus.DEGRADED,  # Not unhealthy - it's expected
                    latency_ms=0,
                    message=f"Cloud mode active - {self.component} running in cloud",
                    details={
                        "cloud_mode": True,
                        "reason": reason,
                        "skip_local": True,
                    },
                )

        # Check circuit breaker (v93.4: startup-aware half-open timeout)
        if self._circuit_open:
            half_open_timeout = self.get_circuit_half_open_seconds()
            if time.time() - self._circuit_opened_at > half_open_timeout:
                # Half-open: try one request
                logger.debug(
                    f"[Health] {self.component} circuit breaker half-open after {half_open_timeout}s"
                    f" (startup_phase={self.is_in_startup_phase()})"
                )
            else:
                return HealthCheck(
                    component=self.component,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    error=f"Circuit breaker open (retry in {half_open_timeout - (time.time() - self._circuit_opened_at):.1f}s)",
                )

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.health_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    self._latencies.append(latency_ms)

                    if response.status == 200:
                        self._consecutive_failures = 0
                        self._circuit_open = False

                        try:
                            data = await response.json()
                        except Exception:
                            data = {}

                        # Determine status based on response and latency
                        if latency_ms > HealthConfig.DEGRADED_LATENCY_MS:
                            status = HealthStatus.DEGRADED
                        else:
                            status = HealthStatus.HEALTHY

                        return HealthCheck(
                            component=self.component,
                            status=status,
                            latency_ms=latency_ms,
                            message=data.get("status", "OK"),
                            details=data,
                        )
                    else:
                        return self._handle_failure(
                            latency_ms, f"HTTP {response.status}"
                        )

        except asyncio.TimeoutError:
            latency_ms = self.timeout * 1000
            return self._handle_failure(latency_ms, "Timeout")

        except ImportError:
            # No aiohttp - try basic check with socket
            return await self._basic_check()

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return self._handle_failure(latency_ms, str(e))

    def _handle_failure(self, latency_ms: float, error: str) -> HealthCheck:
        """Handle check failure (v93.4: startup-aware thresholds)."""
        self._consecutive_failures += 1

        # During startup, use more lenient threshold before opening circuit
        threshold = self.get_unhealthy_threshold()
        if self._consecutive_failures >= threshold:
            self._circuit_open = True
            self._circuit_opened_at = time.time()
            logger.warning(
                f"[Health] {self.component} circuit breaker opened after {self._consecutive_failures} failures "
                f"(threshold={threshold}, startup_phase={self.is_in_startup_phase()})"
            )

        # During startup, mark as DEGRADED instead of UNHEALTHY for transient failures
        if self.is_in_startup_phase() and self._consecutive_failures < threshold:
            status = HealthStatus.DEGRADED
            logger.debug(
                f"[Health] {self.component} startup degraded: {error} "
                f"(failures={self._consecutive_failures}/{threshold})"
            )
        else:
            status = HealthStatus.UNHEALTHY

        return HealthCheck(
            component=self.component,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )

    async def _basic_check(self) -> HealthCheck:
        """Basic check without aiohttp."""
        import socket
        from urllib.parse import urlparse

        start_time = time.time()

        try:
            parsed = urlparse(self.health_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((host, port))
            sock.close()

            latency_ms = (time.time() - start_time) * 1000
            self._consecutive_failures = 0

            return HealthCheck(
                component=self.component,
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Port reachable (basic check)",
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return self._handle_failure(latency_ms, str(e))

    def get_stats(self) -> Dict[str, Any]:
        """Get checker statistics."""
        latencies = list(self._latencies)
        return {
            "component": self.component,
            "health_url": self.health_url,
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0,
        }

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


# ============================================================================
# Health History Manager
# ============================================================================

class HealthHistoryManager:
    """
    Manage historical health data for trend analysis.

    Stores time-series health data with configurable retention.
    """

    def __init__(self, max_points: int = HealthConfig.MAX_HISTORY_POINTS):
        self._max_points = max_points
        self._history: Dict[str, Deque[HealthCheck]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )
        self._incidents: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record(self, check: HealthCheck):
        """Record a health check."""
        async with self._lock:
            self._history[check.component].append(check)

            # Track incidents (transitions to unhealthy)
            history = self._history[check.component]
            if len(history) >= 2:
                prev = history[-2]
                if prev.status == HealthStatus.HEALTHY and check.status == HealthStatus.UNHEALTHY:
                    self._incidents[check.component].append({
                        "started_at": check.timestamp,
                        "error": check.error,
                        "resolved_at": None,
                    })
                elif prev.status == HealthStatus.UNHEALTHY and check.status == HealthStatus.HEALTHY:
                    # Resolve last incident
                    if self._incidents[check.component]:
                        last_incident = self._incidents[check.component][-1]
                        if last_incident.get("resolved_at") is None:
                            last_incident["resolved_at"] = check.timestamp
                            last_incident["duration_seconds"] = (
                                check.timestamp - last_incident["started_at"]
                            )

    async def get_history(
        self,
        component: str,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[HealthCheck]:
        """Get health check history."""
        async with self._lock:
            history = list(self._history.get(component, []))
            if since:
                history = [h for h in history if h.timestamp > since]
            return history[-limit:]

    async def get_uptime(
        self,
        component: str,
        period_seconds: float = 86400,  # 24 hours
    ) -> float:
        """Calculate uptime percentage for a period."""
        async with self._lock:
            now = time.time()
            cutoff = now - period_seconds
            history = [h for h in self._history.get(component, []) if h.timestamp > cutoff]

            if not history:
                return 100.0

            healthy_count = sum(1 for h in history if h.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED))
            return (healthy_count / len(history)) * 100

    async def get_latency_trend(
        self,
        component: str,
        points: int = 60,
    ) -> List[float]:
        """Get latency trend (last N points)."""
        async with self._lock:
            history = list(self._history.get(component, []))
            return [h.latency_ms for h in history[-points:]]

    async def get_incidents(
        self,
        component: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get incident history."""
        async with self._lock:
            if component:
                incidents = self._incidents.get(component, [])
            else:
                incidents = [
                    {**inc, "component": comp}
                    for comp, incs in self._incidents.items()
                    for inc in incs
                ]

            if since:
                incidents = [i for i in incidents if i["started_at"] > since]

            incidents.sort(key=lambda i: i["started_at"], reverse=True)
            return incidents[:limit]

    async def calculate_sla(
        self,
        component: str,
        period_seconds: float = 86400 * 30,  # 30 days
    ) -> SLAReport:
        """Calculate SLA report for a component."""
        async with self._lock:
            now = time.time()
            period_start = now - period_seconds
            history = [h for h in self._history.get(component, []) if h.timestamp > period_start]
            incidents = [
                i for i in self._incidents.get(component, [])
                if i["started_at"] > period_start
            ]

            if not history:
                return SLAReport(
                    component=component,
                    period_start=period_start,
                    period_end=now,
                    uptime_percent=100.0,
                    target_uptime=HealthConfig.SLA_TARGET_UPTIME,
                    is_compliant=True,
                    total_downtime_seconds=0,
                )

            # Calculate uptime
            healthy_count = sum(1 for h in history if h.status != HealthStatus.UNHEALTHY)
            uptime_percent = (healthy_count / len(history)) * 100

            # Calculate downtime from incidents
            total_downtime = sum(
                i.get("duration_seconds", 0) for i in incidents if i.get("resolved_at")
            )
            # Add ongoing incidents
            for incident in incidents:
                if incident.get("resolved_at") is None:
                    total_downtime += now - incident["started_at"]

            # Latency stats
            latencies = [h.latency_ms for h in history]
            avg_latency = statistics.mean(latencies) if latencies else 0
            p99_latency = self._percentile(latencies, 99) if latencies else 0

            return SLAReport(
                component=component,
                period_start=period_start,
                period_end=now,
                uptime_percent=uptime_percent,
                target_uptime=HealthConfig.SLA_TARGET_UPTIME,
                is_compliant=uptime_percent >= HealthConfig.SLA_TARGET_UPTIME,
                total_downtime_seconds=total_downtime,
                incidents=[asdict(i) if hasattr(i, "__dataclass_fields__") else i for i in incidents],
                avg_latency_ms=avg_latency,
                p99_latency_ms=p99_latency,
            )

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """
    Manage health alerts with deduplication and cooldown.

    Supports multiple notification channels.
    """

    def __init__(self, cooldown_seconds: float = HealthConfig.ALERT_COOLDOWN_SECONDS):
        self._cooldown = cooldown_seconds
        self._alerts: Dict[str, HealthAlert] = {}
        self._last_alert_time: Dict[str, float] = {}  # component -> timestamp
        self._callbacks: List[Callable[[HealthAlert], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

    def register_callback(self, callback: Callable[[HealthAlert], Awaitable[None]]):
        """Register alert callback."""
        self._callbacks.append(callback)

    async def check_and_alert(self, check: HealthCheck, prev_status: HealthStatus) -> Optional[HealthAlert]:
        """Check if alert should be raised."""
        async with self._lock:
            # Only alert on status changes or initial unhealthy
            if check.status == prev_status and prev_status != HealthStatus.UNKNOWN:
                return None

            # Check cooldown
            last_alert = self._last_alert_time.get(check.component, 0)
            if time.time() - last_alert < self._cooldown:
                return None

            # Determine severity
            if check.status == HealthStatus.UNHEALTHY:
                severity = AlertSeverity.ERROR
                title = f"{check.component} is UNHEALTHY"
            elif check.status == HealthStatus.DEGRADED:
                severity = AlertSeverity.WARNING
                title = f"{check.component} is DEGRADED"
            elif prev_status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
                severity = AlertSeverity.INFO
                title = f"{check.component} recovered"
            else:
                return None

            alert = HealthAlert(
                component=check.component,
                severity=severity,
                title=title,
                message=check.error or check.message or f"Status: {check.status.value}",
                metadata={
                    "latency_ms": check.latency_ms,
                    "prev_status": prev_status.value,
                    "new_status": check.status.value,
                },
            )

            self._alerts[alert.alert_id] = alert
            self._last_alert_time[check.component] = time.time()

            # Fire callbacks
            for callback in self._callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    logger.error(f"[Alert] Callback error: {e}")

            logger.warning(f"[Alert] {title}: {alert.message}")
            return alert

    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        async with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.resolved = True
                alert.resolved_at = time.time()

    async def get_active_alerts(self) -> List[HealthAlert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self._alerts.values() if not a.resolved]

    async def get_alert(self, alert_id: str) -> Optional[HealthAlert]:
        """Get alert by ID."""
        return self._alerts.get(alert_id)

    async def get_alerts(
        self,
        component: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[HealthAlert]:
        """Get alerts with optional filters."""
        alerts = list(self._alerts.values())

        if component:
            alerts = [a for a in alerts if a.component == component]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]


# ============================================================================
# Health Aggregator (Main Class)
# ============================================================================

class HealthAggregator:
    """
    Central health aggregation system for AGI OS.

    Coordinates health checks across all components and provides
    unified dashboard data.

    v93.4: Startup-aware health checking with configurable grace periods.
    """

    def __init__(self):
        self._start_time = time.time()

        # Component checkers (v93.4: pass start time for startup-aware checking)
        self._checkers: Dict[str, ComponentChecker] = {
            "jarvis": ComponentChecker(
                "jarvis",
                HealthConfig.JARVIS_HEALTH_URL,
                aggregator_start_time=self._start_time,
            ),
            "prime": ComponentChecker(
                "prime",
                HealthConfig.PRIME_HEALTH_URL,
                aggregator_start_time=self._start_time,
            ),
            "reactor-core": ComponentChecker(
                "reactor-core",
                HealthConfig.REACTOR_HEALTH_URL,
                aggregator_start_time=self._start_time,
            ),
        }

        # Component health state
        self._components: Dict[str, ComponentHealth] = {
            name: ComponentHealth(component=name)
            for name in self._checkers
        }

        # Sub-managers
        self._history = HealthHistoryManager()
        self._alerts = AlertManager()

        # State
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        # Dashboard cache
        self._dashboard_cache: Optional[DashboardData] = None
        self._dashboard_cache_ttl: float = 5.0  # seconds
        self._dashboard_cache_time: float = 0

    def is_in_startup_phase(self) -> bool:
        """Check if the aggregator is still in the startup grace period."""
        return (time.time() - self._start_time) < HealthConfig.STARTUP_GRACE_PERIOD

    def get_check_interval(self) -> float:
        """Get check interval, faster during startup for quicker recovery detection."""
        if self.is_in_startup_phase():
            return HealthConfig.STARTUP_CHECK_INTERVAL
        return HealthConfig.CHECK_INTERVAL_SECONDS

    async def start(self):
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        # Update all checkers with the new start time
        for checker in self._checkers.values():
            checker.update_aggregator_start_time(self._start_time)

        self._check_task = asyncio.create_task(self._check_loop())

        logger.info(
            f"[Health] Aggregator started (startup_grace={HealthConfig.STARTUP_GRACE_PERIOD}s, "
            f"startup_check_interval={HealthConfig.STARTUP_CHECK_INTERVAL}s)"
        )

    async def stop(self):
        """Stop health monitoring."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("[Health] Aggregator stopped")

    def add_component(self, name: str, health_url: str):
        """Add a component to monitor."""
        self._checkers[name] = ComponentChecker(
            name,
            health_url,
            aggregator_start_time=self._start_time,
        )
        self._components[name] = ComponentHealth(component=name)

    def remove_component(self, name: str):
        """Remove a component from monitoring."""
        self._checkers.pop(name, None)
        self._components.pop(name, None)

    def register_alert_callback(self, callback: Callable[[HealthAlert], Awaitable[None]]):
        """Register callback for alerts."""
        self._alerts.register_callback(callback)

    async def _check_loop(self):
        """Main health check loop (v93.4: startup-aware dynamic intervals)."""
        last_interval_log = 0

        while self._running:
            try:
                await self._perform_checks()

                # Use faster interval during startup for quicker recovery detection
                interval = self.get_check_interval()

                # Log interval changes (at most once per minute)
                now = time.time()
                if now - last_interval_log > 60:
                    startup_status = "startup" if self.is_in_startup_phase() else "normal"
                    logger.debug(f"[Health] Check interval: {interval}s ({startup_status} mode)")
                    last_interval_log = now

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Health] Check loop error: {e}")
                await asyncio.sleep(5)

    async def _perform_checks(self):
        """Perform parallel health checks on all components."""
        # Run all checks in parallel
        tasks = [
            self._check_component(name, checker)
            for name, checker in self._checkers.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Invalidate dashboard cache
        self._dashboard_cache = None

    async def _check_component(self, name: str, checker: ComponentChecker):
        """Check a single component."""
        try:
            prev_status = self._components[name].current_status
            check = await checker.check()

            # Update component health
            health = self._components[name]
            health.last_check = check
            health.check_count += 1

            if check.status == HealthStatus.UNHEALTHY:
                health.consecutive_failures += 1
                health.failure_count += 1
                health.last_unhealthy = check.timestamp
            else:
                health.consecutive_failures = 0
                health.last_healthy = check.timestamp

            health.current_status = check.status

            # Record history
            await self._history.record(check)

            # Update uptime and latency stats
            health.uptime_percent = await self._history.get_uptime(name)
            latencies = await self._history.get_latency_trend(name, points=100)
            if latencies:
                health.avg_latency_ms = statistics.mean(latencies)
                health.p95_latency_ms = self._percentile(latencies, 95)

            # Check for alerts
            await self._alerts.check_and_alert(check, prev_status)

        except Exception as e:
            logger.error(f"[Health] Error checking {name}: {e}")

    def _percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)

    async def get_dashboard(self, force_refresh: bool = False) -> DashboardData:
        """Get aggregated dashboard data."""
        # Check cache
        if not force_refresh and self._dashboard_cache:
            if time.time() - self._dashboard_cache_time < self._dashboard_cache_ttl:
                return self._dashboard_cache

        # Calculate overall status
        statuses = [h.current_status for h in self._components.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in statuses):
            overall = HealthStatus.UNKNOWN
        else:
            overall = HealthStatus.HEALTHY

        # Get uptime for last 24h
        uptime_24h = {}
        latency_trend = {}
        for name in self._components:
            uptime_24h[name] = await self._history.get_uptime(name, 86400)
            latency_trend[name] = await self._history.get_latency_trend(name, 60)

        # Get recent incidents
        incidents = await self._history.get_incidents(limit=10)

        dashboard = DashboardData(
            overall_status=overall,
            components=dict(self._components),
            active_alerts=await self._alerts.get_active_alerts(),
            recent_incidents=incidents,
            uptime_24h=uptime_24h,
            latency_trend=latency_trend,
        )

        self._dashboard_cache = dashboard
        self._dashboard_cache_time = time.time()

        return dashboard

    async def get_component_health(self, component: str) -> Optional[ComponentHealth]:
        """Get health for a specific component."""
        return self._components.get(component)

    async def get_sla_report(
        self,
        component: str,
        period_seconds: float = 86400 * 30,
    ) -> SLAReport:
        """Get SLA report for a component."""
        return await self._history.calculate_sla(component, period_seconds)

    async def get_all_sla_reports(self, period_seconds: float = 86400 * 30) -> Dict[str, SLAReport]:
        """Get SLA reports for all components."""
        reports = {}
        for name in self._components:
            reports[name] = await self._history.calculate_sla(name, period_seconds)
        return reports

    async def check_now(self, component: Optional[str] = None) -> Dict[str, HealthCheck]:
        """Trigger immediate health check."""
        results = {}

        if component:
            if component in self._checkers:
                await self._check_component(component, self._checkers[component])
                if self._components[component].last_check:
                    results[component] = self._components[component].last_check
        else:
            await self._perform_checks()
            for name, health in self._components.items():
                if health.last_check:
                    results[name] = health.last_check

        return results

    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics (v93.4: includes startup phase info)."""
        uptime = time.time() - self._start_time
        in_startup = self.is_in_startup_phase()

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "components_monitored": len(self._components),
            "check_interval_seconds": self.get_check_interval(),
            "active_alerts": len(await self._alerts.get_active_alerts()),
            "startup_phase": in_startup,
            "startup_remaining_seconds": max(0, HealthConfig.STARTUP_GRACE_PERIOD - uptime) if in_startup else 0,
            "startup_grace_period": HealthConfig.STARTUP_GRACE_PERIOD,
            "checkers": {
                name: {
                    **checker.get_stats(),
                    "startup_phase": checker.is_in_startup_phase(),
                    "effective_unhealthy_threshold": checker.get_unhealthy_threshold(),
                }
                for name, checker in self._checkers.items()
            },
        }


# ============================================================================
# Webhook Notifier
# ============================================================================

class WebhookNotifier:
    """Send alerts to webhook endpoints."""

    def __init__(self, webhook_urls: Optional[List[str]] = None):
        self._urls = webhook_urls or []

    def add_url(self, url: str):
        """Add webhook URL."""
        if url not in self._urls:
            self._urls.append(url)

    async def notify(self, alert: HealthAlert):
        """Send alert to all webhooks."""
        if not self._urls:
            return

        try:
            import aiohttp

            payload = alert.to_dict()

            async with aiohttp.ClientSession() as session:
                for url in self._urls:
                    try:
                        async with session.post(
                            url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status == 200:
                                logger.debug(f"[Webhook] Sent alert to {url}")
                            else:
                                logger.warning(f"[Webhook] Failed to send to {url}: {response.status}")
                    except Exception as e:
                        logger.warning(f"[Webhook] Error sending to {url}: {e}")

        except ImportError:
            logger.warning("[Webhook] aiohttp not available")


# ============================================================================
# Global Health Aggregator Instance
# ============================================================================

_aggregator: Optional[HealthAggregator] = None


def get_health_aggregator() -> HealthAggregator:
    """Get global health aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = HealthAggregator()
    return _aggregator


async def init_health_aggregator() -> HealthAggregator:
    """Initialize and start health aggregator."""
    aggregator = get_health_aggregator()
    await aggregator.start()
    return aggregator
