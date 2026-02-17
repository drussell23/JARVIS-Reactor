"""
Cross-Repo Cost Bridge.

Provides unified cost tracking across JARVIS, JARVIS-Prime, and Reactor-Core.
Reads bridge state files and emits cost events for unified tracking.

Features:
- Read cost data from JARVIS infrastructure orchestrator
- Read inference metrics from JARVIS-Prime
- Aggregate costs across all repos
- Emit alerts when budgets are approached
- Generate cost reports for the training pipeline

Usage:
    from reactor_core.integration.cost_bridge import (
        CostBridge,
        get_aggregated_costs,
        emit_cost_event,
    )

    # Get unified cost view
    costs = await get_aggregated_costs()

    # Check budget
    if costs["total_usd"] > 50.0:
        await emit_cost_event("budget_warning", costs)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

from reactor_core.integration.event_bridge import (
    CrossRepoEvent,
    EventSource,
    EventType,
    FileTransport,
)

logger = logging.getLogger(__name__)

# Shared state directory
BRIDGE_STATE_DIR = Path.home() / ".jarvis" / "cross_repo"
JARVIS_STATE_FILE = BRIDGE_STATE_DIR / "bridge_state.json"
PRIME_STATE_FILE = BRIDGE_STATE_DIR / "jarvis_prime_state.json"
REACTOR_STATE_FILE = BRIDGE_STATE_DIR / "reactor_core_state.json"
PRIME_EVENTS_FILE = BRIDGE_STATE_DIR / "prime_events.json"


@dataclass
class CostSummary:
    """Aggregated cost summary across all repos."""
    jarvis_costs: Dict[str, float] = field(default_factory=dict)
    prime_costs: Dict[str, float] = field(default_factory=dict)
    reactor_costs: Dict[str, float] = field(default_factory=dict)

    # Totals
    total_usd: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0

    # Savings from local inference
    cloud_equivalent_usd: float = 0.0
    savings_usd: float = 0.0

    # Budget tracking
    budget_limit_usd: float = 100.0
    budget_used_percent: float = 0.0

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jarvis_costs": self.jarvis_costs,
            "prime_costs": self.prime_costs,
            "reactor_costs": self.reactor_costs,
            "total_usd": self.total_usd,
            "total_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "cloud_equivalent_usd": self.cloud_equivalent_usd,
            "savings_usd": self.savings_usd,
            "budget_limit_usd": self.budget_limit_usd,
            "budget_used_percent": self.budget_used_percent,
            "timestamp": self.timestamp.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }


class CostBridge:
    """
    Unified cost tracking bridge for cross-repo integration.

    Reads cost data from JARVIS and JARVIS-Prime state files,
    tracks Reactor-Core training costs, and provides aggregated views.
    """

    def __init__(
        self,
        budget_limit_usd: float = 100.0,
        alert_threshold: float = 0.8,  # Alert at 80% of budget
    ):
        self.budget_limit_usd = budget_limit_usd
        self.alert_threshold = alert_threshold

        # Reactor-Core internal cost tracking
        self.reactor_costs: Dict[str, float] = {
            "distillation_api_costs": 0.0,
            "training_compute_costs": 0.0,
            "storage_costs": 0.0,
        }
        self.reactor_tokens_used = 0
        self.reactor_requests = 0

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

        # Event transport
        self._transport: Optional[FileTransport] = None

        # v258.0: Strong references for fire-and-forget background tasks
        self._background_tasks: set = set()

    async def initialize(self) -> None:
        """Initialize the cost bridge."""
        BRIDGE_STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize file transport for events
        self._transport = FileTransport(
            events_dir=BRIDGE_STATE_DIR / "events",
            source=EventSource.REACTOR_CORE,
            cleanup_hours=168,  # 1 week
        )
        await self._transport.connect()

        # Write initial state
        await self._write_reactor_state()

        logger.info("CostBridge initialized")

    async def shutdown(self) -> None:
        """Shutdown the cost bridge."""
        if self._transport:
            await self._transport.disconnect()

        # Write final state
        await self._write_reactor_state()

    def on_budget_alert(self, callback: Callable) -> None:
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)

    def _spawn_budget_check(self) -> None:
        """v258.0: Spawn budget check with strong task reference."""
        try:
            _task = asyncio.create_task(
                self._check_budget(),
                name="cost_bridge_budget_check",
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
        except RuntimeError:
            # No running event loop — skip budget check
            pass

    def record_distillation_cost(
        self,
        provider: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        """Record cost from distillation API calls."""
        self.reactor_costs["distillation_api_costs"] += cost_usd
        self.reactor_tokens_used += tokens_in + tokens_out
        self.reactor_requests += 1

        # Check budget
        self._spawn_budget_check()

    def record_training_cost(self, cost_usd: float) -> None:
        """Record cost from training compute."""
        self.reactor_costs["training_compute_costs"] += cost_usd
        self._spawn_budget_check()

    def record_storage_cost(self, cost_usd: float) -> None:
        """Record cost from storage usage."""
        self.reactor_costs["storage_costs"] += cost_usd

    async def get_jarvis_costs(self) -> Dict[str, Any]:
        """Read JARVIS cost data from bridge state."""
        try:
            if JARVIS_STATE_FILE.exists():
                content = JARVIS_STATE_FILE.read_text()
                data = json.loads(content)
                return data.get("cost_tracking", {})
        except Exception as e:
            logger.warning(f"Failed to read JARVIS costs: {e}")
        return {}

    async def get_prime_costs(self) -> Dict[str, Any]:
        """Read JARVIS-Prime cost data from bridge state."""
        try:
            if PRIME_STATE_FILE.exists():
                content = PRIME_STATE_FILE.read_text()
                data = json.loads(content)
                metrics = data.get("metrics", {})
                return {
                    "total_requests": metrics.get("total_requests", 0),
                    "total_tokens": metrics.get("total_tokens_in", 0) + metrics.get("total_tokens_out", 0),
                    "local_cost_usd": 0.0,  # Local inference is free
                    "cloud_equivalent_usd": metrics.get("estimated_cost_usd", 0.0),
                    "savings_usd": metrics.get("savings_vs_cloud_usd", 0.0),
                }
        except Exception as e:
            logger.warning(f"Failed to read Prime costs: {e}")
        return {}

    async def get_aggregated_costs(self) -> CostSummary:
        """Get unified cost view across all repos."""
        jarvis = await self.get_jarvis_costs()
        prime = await self.get_prime_costs()

        # Calculate totals
        jarvis_total = sum(
            v for k, v in jarvis.items()
            if isinstance(v, (int, float)) and "cost" in k.lower()
        )
        reactor_total = sum(self.reactor_costs.values())
        prime_total = prime.get("local_cost_usd", 0.0)

        total = jarvis_total + reactor_total + prime_total

        # Cloud equivalent
        cloud_equiv = prime.get("cloud_equivalent_usd", 0.0)
        savings = prime.get("savings_usd", 0.0)

        summary = CostSummary(
            jarvis_costs=jarvis,
            prime_costs=prime,
            reactor_costs=self.reactor_costs.copy(),
            total_usd=total,
            total_tokens=self.reactor_tokens_used + prime.get("total_tokens", 0),
            total_requests=self.reactor_requests + prime.get("total_requests", 0),
            cloud_equivalent_usd=cloud_equiv,
            savings_usd=savings,
            budget_limit_usd=self.budget_limit_usd,
            budget_used_percent=(total / self.budget_limit_usd * 100) if self.budget_limit_usd > 0 else 0,
        )

        return summary

    async def emit_cost_event(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        priority: int = 5,
    ) -> bool:
        """Emit a cost-related event to the bridge."""
        if not self._transport:
            return False

        event = CrossRepoEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type,
            source=EventSource.REACTOR_CORE,
            timestamp=datetime.now(),
            payload=payload,
            priority=priority,
        )

        return await self._transport.publish(event)

    async def generate_cost_report(self) -> Dict[str, Any]:
        """Generate a detailed cost report."""
        summary = await self.get_aggregated_costs()

        report = {
            "report_id": str(uuid.uuid4())[:8],
            "generated_at": datetime.now().isoformat(),
            "period": "current_session",
            "summary": summary.to_dict(),
            "breakdown": {
                "jarvis_agent": summary.jarvis_costs,
                "jarvis_prime": summary.prime_costs,
                "reactor_core": summary.reactor_costs,
            },
            "optimization": {
                "local_inference_savings": summary.savings_usd,
                "savings_vs_cloud": f"{(summary.savings_usd / max(summary.cloud_equivalent_usd, 0.01)) * 100:.1f}%",
            },
            "budget_status": {
                "limit": self.budget_limit_usd,
                "used": summary.total_usd,
                "remaining": max(0, self.budget_limit_usd - summary.total_usd),
                "percentage": summary.budget_used_percent,
                "status": "ok" if summary.budget_used_percent < self.alert_threshold * 100 else "warning",
            },
        }

        # Emit report event
        await self.emit_cost_event(EventType.COST_REPORT, report, priority=3)

        return report

    async def _check_budget(self) -> None:
        """Check if budget threshold is approached and emit alerts."""
        summary = await self.get_aggregated_costs()

        if summary.budget_used_percent >= self.alert_threshold * 100:
            # Emit alert
            alert_payload = {
                "alert_type": "budget_warning",
                "current_usage": summary.total_usd,
                "budget_limit": self.budget_limit_usd,
                "percentage_used": summary.budget_used_percent,
                "message": f"Budget usage at {summary.budget_used_percent:.1f}%",
            }

            await self.emit_cost_event(EventType.COST_ALERT, alert_payload, priority=1)

            # Run callbacks
            for callback in self._alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert_payload)
                    else:
                        callback(alert_payload)
                except Exception as e:
                    logger.warning(f"Alert callback error: {e}")

    async def _write_reactor_state(self) -> None:
        """Write Reactor-Core state to bridge file."""
        try:
            state = {
                "source": "reactor_core",
                "last_update": datetime.now().isoformat(),
                "costs": self.reactor_costs,
                "metrics": {
                    "tokens_used": self.reactor_tokens_used,
                    "requests": self.reactor_requests,
                },
            }
            REACTOR_STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write reactor state: {e}")


# ============================================================================
# Global Instance & Helpers
# ============================================================================

_cost_bridge: Optional[CostBridge] = None


def get_cost_bridge() -> Optional[CostBridge]:
    """Get the global cost bridge instance."""
    return _cost_bridge


async def initialize_cost_bridge(
    budget_limit_usd: float = 100.0,
) -> CostBridge:
    """Initialize the global cost bridge."""
    global _cost_bridge

    if _cost_bridge is None:
        _cost_bridge = CostBridge(budget_limit_usd=budget_limit_usd)
        await _cost_bridge.initialize()

    return _cost_bridge


async def shutdown_cost_bridge() -> None:
    """Shutdown the global cost bridge."""
    global _cost_bridge

    if _cost_bridge:
        await _cost_bridge.shutdown()
        _cost_bridge = None


async def get_aggregated_costs() -> CostSummary:
    """Get unified cost view across all repos."""
    if _cost_bridge:
        return await _cost_bridge.get_aggregated_costs()
    return CostSummary()


async def emit_cost_event(
    event_type: str,
    payload: Dict[str, Any],
) -> bool:
    """Emit a cost-related event."""
    if _cost_bridge:
        try:
            etype = EventType(event_type) if isinstance(event_type, str) else event_type
        except ValueError:
            etype = EventType.COST_UPDATE
        return await _cost_bridge.emit_cost_event(etype, payload)
    return False


def record_distillation_cost(
    provider: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
) -> None:
    """Record distillation API cost."""
    if _cost_bridge:
        _cost_bridge.record_distillation_cost(provider, tokens_in, tokens_out, cost_usd)
