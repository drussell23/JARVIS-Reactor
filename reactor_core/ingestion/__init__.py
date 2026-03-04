"""
Data ingestion module for Night Shift Training Engine.

Provides ingestors for various JARVIS data sources:
- Telemetry events
- User feedback (corrections)
- Authentication records
- Raw log files
"""

from reactor_core.ingestion.base_ingestor import (
    AbstractIngestor,
    BaseIngestor,
    RawInteraction,
    InteractionOutcome,
    SourceType,
)
from reactor_core.ingestion.batch_processor import (
    BatchIngestionProcessor,
    IngestionResult,
    IngestionStats,
)
from reactor_core.ingestion.telemetry_ingestor import TelemetryIngestor
from reactor_core.ingestion.feedback_ingestor import (
    FeedbackIngestor,
    AuthRecordIngestor,
)
from reactor_core.ingestion.autonomy_classifier import AutonomyEventClassifier
from reactor_core.ingestion.autonomy_event_ingestor import AutonomyEventIngestor

__all__ = [
    # Base types
    "AbstractIngestor",
    "BaseIngestor",
    "RawInteraction",
    "InteractionOutcome",
    "SourceType",
    # Batch processing
    "BatchIngestionProcessor",
    "IngestionResult",
    "IngestionStats",
    # Ingestors
    "TelemetryIngestor",
    "FeedbackIngestor",
    "AuthRecordIngestor",
    # Autonomy (Phase 2)
    "AutonomyEventClassifier",
    "AutonomyEventIngestor",
]
