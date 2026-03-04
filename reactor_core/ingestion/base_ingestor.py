"""
Base ingestor protocol and core data structures for Night Shift.

Defines:
- RawInteraction: Unified format for all ingested data
- BaseIngestor: Abstract protocol for data ingestors
- SourceType: Enum of supported data sources
- InteractionOutcome: Enum of possible outcomes
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)


class SourceType(Enum):
    """Supported data source types."""
    TELEMETRY = "telemetry"
    FEEDBACK = "feedback"
    AUTH_RECORD = "auth_record"
    FUSION_RESULT = "fusion_result"
    RAW_LOG = "raw_log"
    STATE_FILE = "state_file"
    UNKNOWN = "unknown"


class InteractionOutcome(Enum):
    """Possible interaction outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    NEGATIVE_FEEDBACK = "negative_feedback"
    POSITIVE_FEEDBACK = "positive_feedback"
    ENGAGED = "engaged"
    DISMISSED = "dismissed"
    DEFERRED = "deferred"
    CHALLENGED = "challenged"
    INFRASTRUCTURE = "infrastructure"  # v300.0: Policy/infra events, not model quality
    UNKNOWN = "unknown"


@dataclass
class RawInteraction:
    """
    Unified format for all ingested JARVIS interactions.

    This is the core data structure that all ingestors produce.
    It captures:
    - The original user input/request
    - The system's response/output
    - Quality signals (outcome, confidence)
    - Correction data (if user corrected the system)
    - Rich metadata for filtering and formatting
    """

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Source information
    source_type: SourceType = SourceType.UNKNOWN
    source_id: str = ""  # Original ID from source system
    source_file: Optional[str] = None  # Path to source file

    # Conversation content
    user_input: Optional[str] = None
    assistant_output: Optional[str] = None
    system_context: Optional[str] = None

    # Multi-turn context (for conversation threads)
    conversation_id: Optional[str] = None
    turn_number: int = 0
    previous_turns: List[Dict[str, str]] = field(default_factory=list)

    # Quality signals
    outcome: InteractionOutcome = InteractionOutcome.UNKNOWN
    confidence: float = 0.0
    quality_score: float = 0.0  # Computed or from distillation

    # Correction data (when user corrected JARVIS)
    is_correction: bool = False
    correction_original: Optional[str] = None  # What JARVIS originally said
    correction_improved: Optional[str] = None  # What user wanted instead
    correction_reason: Optional[str] = None  # Why it was wrong

    # Rich metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Authentication context (for auth-related interactions)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    auth_confidence: Optional[float] = None

    # Environment context
    environment: Dict[str, Any] = field(default_factory=dict)

    # Processing flags
    processed: bool = False
    formatted: bool = False
    distilled: bool = False

    def __post_init__(self):
        """Compute derived fields."""
        if not self.id:
            self.id = str(uuid.uuid4())

        # Compute quality score if not set
        if self.quality_score == 0.0 and self.confidence > 0:
            self.quality_score = self._compute_quality_score()

    def _compute_quality_score(self) -> float:
        """Compute initial quality score from available signals."""
        score = self.confidence

        # Boost for successful outcomes
        if self.outcome == InteractionOutcome.SUCCESS:
            score *= 1.2
        elif self.outcome == InteractionOutcome.POSITIVE_FEEDBACK:
            score *= 1.3
        elif self.outcome == InteractionOutcome.ENGAGED:
            score *= 1.1

        # Penalty for failures
        if self.outcome == InteractionOutcome.FAILURE:
            score *= 0.5
        elif self.outcome == InteractionOutcome.NEGATIVE_FEEDBACK:
            # But corrections are valuable for training!
            score *= 0.8 if self.is_correction else 0.6

        # Boost for having user input and output
        if self.user_input and self.assistant_output:
            score *= 1.1

        # Normalize
        return min(1.0, max(0.0, score))

    def content_hash(self) -> str:
        """Generate hash of content for deduplication."""
        content = f"{self.user_input or ''}{self.assistant_output or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_trainable(self, min_confidence: float = 0.5) -> bool:
        """Check if this interaction is suitable for training."""
        # Must have both input and output
        if not self.user_input or not self.assistant_output:
            return False

        # Must meet minimum quality
        if self.quality_score < min_confidence:
            return False

        # Corrections are always valuable
        if self.is_correction:
            return True

        # Exclude failures without corrections
        if self.outcome == InteractionOutcome.FAILURE and not self.is_correction:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "source_file": self.source_file,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "system_context": self.system_context,
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "previous_turns": self.previous_turns,
            "outcome": self.outcome.value,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "is_correction": self.is_correction,
            "correction_original": self.correction_original,
            "correction_improved": self.correction_improved,
            "correction_reason": self.correction_reason,
            "properties": self.properties,
            "tags": self.tags,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "auth_confidence": self.auth_confidence,
            "environment": self.environment,
            "processed": self.processed,
            "formatted": self.formatted,
            "distilled": self.distilled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawInteraction":
        """Create from dictionary."""
        # Parse enum fields
        source_type = SourceType(data.get("source_type", "unknown"))
        outcome = InteractionOutcome(data.get("outcome", "unknown"))

        # Parse timestamp
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=timestamp,
            source_type=source_type,
            source_id=data.get("source_id", ""),
            source_file=data.get("source_file"),
            user_input=data.get("user_input"),
            assistant_output=data.get("assistant_output"),
            system_context=data.get("system_context"),
            conversation_id=data.get("conversation_id"),
            turn_number=data.get("turn_number", 0),
            previous_turns=data.get("previous_turns", []),
            outcome=outcome,
            confidence=data.get("confidence", 0.0),
            quality_score=data.get("quality_score", 0.0),
            is_correction=data.get("is_correction", False),
            correction_original=data.get("correction_original"),
            correction_improved=data.get("correction_improved"),
            correction_reason=data.get("correction_reason"),
            properties=data.get("properties", {}),
            tags=data.get("tags", []),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            auth_confidence=data.get("auth_confidence"),
            environment=data.get("environment", {}),
            processed=data.get("processed", False),
            formatted=data.get("formatted", False),
            distilled=data.get("distilled", False),
        )


@runtime_checkable
class BaseIngestor(Protocol):
    """
    Protocol for data ingestors.

    All ingestors must implement these methods:
    - ingest(): Async iterator yielding RawInteraction
    - ingest_batch(): Async iterator yielding batches
    - supports_streaming(): Whether real-time ingestion is supported
    """

    @property
    def source_type(self) -> SourceType:
        """The type of data source this ingestor handles."""
        ...

    async def ingest(
        self,
        source_path: Union[str, Path],
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[RawInteraction]:
        """
        Ingest data from source, yielding individual interactions.

        Args:
            source_path: Path to data source (file or directory)
            since: Only ingest data after this timestamp
            until: Only ingest data before this timestamp

        Yields:
            RawInteraction for each discovered interaction
        """
        ...

    async def ingest_batch(
        self,
        source_path: Union[str, Path],
        batch_size: int = 100,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[List[RawInteraction]]:
        """
        Ingest data in batches for efficient parallel processing.

        Args:
            source_path: Path to data source
            batch_size: Number of interactions per batch
            since: Only ingest data after this timestamp
            until: Only ingest data before this timestamp

        Yields:
            List of RawInteraction (batch)
        """
        ...

    def supports_streaming(self) -> bool:
        """Whether this ingestor supports real-time streaming."""
        ...


class AbstractIngestor(ABC):
    """
    Abstract base class for ingestors with common functionality.

    Subclasses only need to implement:
    - _parse_item(): Parse a single item from the source
    - _iter_items(): Iterate over items in the source
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        include_failures: bool = True,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize ingestor.

        Args:
            min_confidence: Minimum confidence threshold
            include_failures: Whether to include failed interactions
            tags: Tags to add to all ingested interactions
        """
        self.min_confidence = min_confidence
        self.include_failures = include_failures
        self.default_tags = tags or []
        self._ingested_count = 0
        self._filtered_count = 0

    @property
    @abstractmethod
    def source_type(self) -> SourceType:
        """The type of data source this ingestor handles."""
        ...

    @abstractmethod
    async def _iter_items(
        self,
        source_path: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Iterate over raw items in the source.

        Args:
            source_path: Path to data source
            since: Only include items after this timestamp
            until: Only include items before this timestamp

        Yields:
            Raw item dictionaries
        """
        ...

    @abstractmethod
    async def _parse_item(
        self,
        item: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> Optional[RawInteraction]:
        """
        Parse a single item into RawInteraction.

        Args:
            item: Raw item dictionary
            source_file: Path to source file

        Returns:
            RawInteraction or None if item should be skipped
        """
        ...

    def _should_include(self, interaction: RawInteraction) -> bool:
        """Check if interaction should be included based on filters."""
        # Confidence filter
        if interaction.confidence < self.min_confidence:
            return False

        # Failure filter
        if not self.include_failures:
            if interaction.outcome == InteractionOutcome.FAILURE:
                return False

        return True

    async def ingest(
        self,
        source_path: Union[str, Path],
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[RawInteraction]:
        """Ingest data yielding individual interactions."""
        source_path = Path(source_path)

        async for item in self._iter_items(source_path, since, until):
            interaction = await self._parse_item(
                item,
                source_file=str(source_path),
            )

            if interaction is None:
                self._filtered_count += 1
                continue

            if not self._should_include(interaction):
                self._filtered_count += 1
                continue

            # Add default tags
            interaction.tags.extend(self.default_tags)
            self._ingested_count += 1

            yield interaction

    async def ingest_batch(
        self,
        source_path: Union[str, Path],
        batch_size: int = 100,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[List[RawInteraction]]:
        """Ingest data in batches."""
        batch: List[RawInteraction] = []

        async for interaction in self.ingest(source_path, since, until):
            batch.append(interaction)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    def supports_streaming(self) -> bool:
        """Default: no streaming support."""
        return False

    def get_stats(self) -> Dict[str, int]:
        """Get ingestion statistics."""
        return {
            "ingested": self._ingested_count,
            "filtered": self._filtered_count,
            "total_processed": self._ingested_count + self._filtered_count,
        }

    def reset_stats(self) -> None:
        """Reset ingestion statistics."""
        self._ingested_count = 0
        self._filtered_count = 0
