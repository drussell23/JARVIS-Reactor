"""
Orchestration module for Night Shift Training Engine.

Provides:
- End-to-end pipeline orchestration
- Cron-based scheduling
- Pipeline state management
- Slack/webhook notifications
"""

from reactor_core.orchestration.pipeline import (
    NightShiftPipeline,
    PipelineConfig,
    PipelineStage,
    PipelineState,
    PipelineResult,
    DataSource,
)

from reactor_core.orchestration.scheduler import (
    PipelineScheduler,
    ScheduleConfig,
    ScheduledRun,
)

from reactor_core.orchestration.notifications import (
    NotificationManager,
    NotificationConfig,
    NotificationType,
    SlackNotifier,
    WebhookNotifier,
)

# PROJECT TRINITY: Central Orchestrator
from reactor_core.orchestration.trinity_orchestrator import (
    TrinityOrchestrator,
    ComponentType,
    ComponentHealth,
    ComponentState,
    AggregatedState,
    get_orchestrator,
    initialize_orchestrator,
    shutdown_orchestrator,
    dispatch_to_jarvis,
    dispatch_to_jprime,
    update_jarvis_heartbeat,
    update_jprime_heartbeat,
)

__all__ = [
    # Pipeline
    "NightShiftPipeline",
    "PipelineConfig",
    "PipelineStage",
    "PipelineState",
    "PipelineResult",
    "DataSource",
    # Scheduler
    "PipelineScheduler",
    "ScheduleConfig",
    "ScheduledRun",
    # Notifications
    "NotificationManager",
    "NotificationConfig",
    "NotificationType",
    "SlackNotifier",
    "WebhookNotifier",
    # PROJECT TRINITY: Orchestrator
    "TrinityOrchestrator",
    "ComponentType",
    "ComponentHealth",
    "ComponentState",
    "AggregatedState",
    "get_orchestrator",
    "initialize_orchestrator",
    "shutdown_orchestrator",
    "dispatch_to_jarvis",
    "dispatch_to_jprime",
    "update_jarvis_heartbeat",
    "update_jprime_heartbeat",
]
