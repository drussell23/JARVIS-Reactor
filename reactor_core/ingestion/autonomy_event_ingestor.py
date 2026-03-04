"""
Autonomy Event Ingestor (v300.0 — Phase 2 Trinity Autonomy Wiring)
===================================================================

Ingests structured autonomy lifecycle events emitted by JARVIS Body.
Validates the strict metadata schema, classifies events via the
centralized ``AutonomyEventClassifier``, and quarantines malformed events
rather than silently coercing them.

Events arrive as ``ExperiencePacket`` items via the ``TrinityExperienceReceiver``
filesystem watch or the Trinity event bus.  This ingestor is registered as
a handler for ``experience_type`` values prefixed with ``"autonomy:"``.

Quarantine Policy:
    Location:   ``~/.jarvis/reactor/quarantine/autonomy/``
    Retention:  7 days (env: ``REACTOR_QUARANTINE_RETENTION_DAYS``)
    Max size:   100 MB (env: ``REACTOR_QUARANTINE_MAX_SIZE_MB``)
    Alert:      >10 quarantined/hour triggers warning log
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from reactor_core.ingestion.base_ingestor import (
    AbstractIngestor,
    InteractionOutcome,
    RawInteraction,
    SourceType,
)
from reactor_core.ingestion.autonomy_classifier import (
    ALL_KNOWN_TYPES,
    SUPPORTED_SCHEMA_VERSIONS,
    AutonomyEventClassifier,
)

logger = logging.getLogger("AutonomyEventIngestor")

# ---------------------------------------------------------------------------
# Configuration (all via environment variables — no hardcoding)
# ---------------------------------------------------------------------------

QUARANTINE_DIR = Path(
    os.getenv(
        "REACTOR_QUARANTINE_DIR",
        str(Path.home() / ".jarvis" / "reactor" / "quarantine" / "autonomy"),
    )
)
QUARANTINE_RETENTION_DAYS = int(os.getenv("REACTOR_QUARANTINE_RETENTION_DAYS", "7"))
QUARANTINE_MAX_SIZE_MB = int(os.getenv("REACTOR_QUARANTINE_MAX_SIZE_MB", "100"))
QUARANTINE_ALERT_THRESHOLD = int(os.getenv("REACTOR_QUARANTINE_ALERT_THRESHOLD", "10"))

# 7 required metadata keys — must match JARVIS Body constants
AUTONOMY_REQUIRED_KEYS = frozenset({
    "autonomy_event_type",
    "autonomy_schema_version",
    "idempotency_key",
    "trace_id",
    "correlation_id",
    "action",
    "request_kind",
})


class AutonomyEventIngestor(AbstractIngestor):
    """Ingestor for autonomy lifecycle events from JARVIS Body.

    Extends ``AbstractIngestor`` with strict validation, quarantine for
    malformed events, and integration with ``AutonomyEventClassifier``
    for training label assignment.
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        include_failures: bool = True,
        tags: Optional[List[str]] = None,
    ):
        super().__init__(
            min_confidence=min_confidence,
            include_failures=include_failures,
            tags=tags,
        )
        self._classifier = AutonomyEventClassifier()
        self._quarantine_count_window: deque = deque(maxlen=1000)
        self._quarantine_total = 0
        self._dedup_keys: set = set()
        self._dedup_max = 50_000

        # Ensure quarantine directory exists
        try:
            QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Cannot create quarantine dir %s: %s", QUARANTINE_DIR, exc)

    @property
    def source_type(self) -> SourceType:
        return SourceType.TELEMETRY

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_event(self, event: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate an autonomy event against the strict schema.

        Returns ``(True, "")`` if valid, ``(False, reason)`` otherwise.
        """
        metadata = event.get("metadata", {})

        # 1. Check all 7 required keys present
        missing = AUTONOMY_REQUIRED_KEYS - set(metadata.keys())
        if missing:
            return False, f"missing_required_keys:{','.join(sorted(missing))}"

        # 2. Hard enum validation for autonomy_event_type
        event_type = metadata.get("autonomy_event_type", "")
        if not self._classifier.is_known_type(event_type):
            return False, f"unknown_event_type:{event_type}"

        # 3. Schema version check
        schema_version = metadata.get("autonomy_schema_version", "")
        if not self._classifier.is_schema_supported(schema_version):
            return False, f"unsupported_schema:{schema_version}"

        return True, ""

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _dedup_key(self, metadata: Dict[str, Any]) -> str:
        """Composite dedup key: (idempotency_key, autonomy_event_type, trace_id)."""
        return "|".join([
            metadata.get("idempotency_key", ""),
            metadata.get("autonomy_event_type", ""),
            metadata.get("trace_id", ""),
        ])

    def _is_duplicate(self, metadata: Dict[str, Any]) -> bool:
        """Check and register the event for deduplication."""
        key = self._dedup_key(metadata)
        if key in self._dedup_keys:
            return True
        # Evict oldest if over limit
        if len(self._dedup_keys) >= self._dedup_max:
            # Simple eviction — discard all and rebuild
            # (bloom filter would be better at scale, but this is
            # adequate for the default 50K window)
            self._dedup_keys.clear()
        self._dedup_keys.add(key)
        return False

    # ------------------------------------------------------------------
    # Quarantine
    # ------------------------------------------------------------------

    def _quarantine(self, event: Dict[str, Any], reason: str) -> None:
        """Quarantine a malformed event to disk for ops review."""
        try:
            self._quarantine_total += 1
            now = time.time()
            self._quarantine_count_window.append(now)

            # Alert if threshold exceeded in the last hour
            one_hour_ago = now - 3600
            recent = sum(1 for t in self._quarantine_count_window if t > one_hour_ago)
            if recent >= QUARANTINE_ALERT_THRESHOLD:
                logger.warning(
                    "[AutonomyIngestor] Quarantine alert: %d events quarantined in last hour "
                    "(threshold: %d)",
                    recent,
                    QUARANTINE_ALERT_THRESHOLD,
                )

            # Check disk budget
            if not self._quarantine_within_budget():
                logger.warning(
                    "[AutonomyIngestor] Quarantine full (>%dMB). Dropping event.",
                    QUARANTINE_MAX_SIZE_MB,
                )
                return

            filename = f"quarantine_{int(now * 1000)}_{uuid.uuid4().hex[:8]}.json"
            filepath = QUARANTINE_DIR / filename
            payload = {
                "quarantine_reason": reason,
                "quarantined_at": datetime.utcnow().isoformat(),
                "event": event,
            }
            filepath.write_text(json.dumps(payload, default=str), encoding="utf-8")

        except Exception as exc:
            logger.debug("[AutonomyIngestor] Failed to quarantine event: %s", exc)

    def _quarantine_within_budget(self) -> bool:
        """Check if quarantine disk usage is within budget."""
        try:
            total = sum(f.stat().st_size for f in QUARANTINE_DIR.iterdir() if f.is_file())
            return total < QUARANTINE_MAX_SIZE_MB * 1024 * 1024
        except OSError:
            return True  # Assume OK if we can't check

    def cleanup_quarantine(self) -> int:
        """Remove quarantine files older than retention period. Returns count removed."""
        removed = 0
        cutoff = time.time() - (QUARANTINE_RETENTION_DAYS * 86400)
        try:
            for f in QUARANTINE_DIR.iterdir():
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
        except OSError as exc:
            logger.debug("[AutonomyIngestor] Quarantine cleanup error: %s", exc)
        return removed

    # ------------------------------------------------------------------
    # AbstractIngestor implementation
    # ------------------------------------------------------------------

    async def _iter_items(
        self,
        source_path: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Iterate over autonomy events from JSONL source files.

        Scans ``source_path`` for JSONL files containing autonomy events
        (identified by ``experience_type`` starting with ``"autonomy:"``).
        """
        if source_path.is_file():
            paths = [source_path]
        elif source_path.is_dir():
            paths = sorted(source_path.glob("*.jsonl"))
        else:
            return

        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Only process autonomy events
                        exp_type = item.get("type", "")
                        if not exp_type.startswith("autonomy:"):
                            continue

                        # Timestamp filter
                        ts = item.get("timestamp")
                        if ts and since and ts < since.timestamp():
                            continue
                        if ts and until and ts > until.timestamp():
                            continue

                        yield item

            except OSError as exc:
                logger.warning("[AutonomyIngestor] Failed to read %s: %s", path, exc)

    async def _parse_item(
        self,
        item: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> Optional[RawInteraction]:
        """Parse a single autonomy event into a RawInteraction.

        Performs strict validation and quarantines invalid events.
        """
        # Validate schema
        valid, reason = self.validate_event(item)
        if not valid:
            self._quarantine(item, reason)
            return None

        metadata = item.get("metadata", {})

        # Dedup check
        if self._is_duplicate(metadata):
            return None

        # Classify
        event_type = metadata["autonomy_event_type"]
        outcome, should_train = self._classifier.classify(event_type)

        # Build RawInteraction
        return RawInteraction(
            id=item.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromtimestamp(item.get("timestamp", time.time())),
            source_type=SourceType.TELEMETRY,
            source_id=f"autonomy:{event_type}",
            source_file=source_file,
            user_input=json.dumps(item.get("input", {})),
            assistant_output=json.dumps(item.get("output", {})),
            system_context=json.dumps(metadata),
            outcome=outcome,
            confidence=item.get("confidence", 1.0),
            quality_score=1.0 if should_train else 0.0,
            properties={
                "autonomy_event_type": event_type,
                "autonomy_schema_version": metadata.get("autonomy_schema_version", ""),
                "idempotency_key": metadata.get("idempotency_key", ""),
                "action": metadata.get("action", ""),
                "should_train": should_train,
            },
            tags=["autonomy", f"autonomy:{event_type}"],
        )

    def supports_streaming(self) -> bool:
        """Autonomy events support streaming via filesystem watch."""
        return True

    # ------------------------------------------------------------------
    # Convenience: process a single event dict (for TrinityExperienceReceiver)
    # ------------------------------------------------------------------

    async def process_single_event(
        self, event: Dict[str, Any]
    ) -> Optional[RawInteraction]:
        """Validate, classify, and convert a single autonomy event.

        Used by ``TrinityExperienceReceiver`` when an autonomy event
        arrives via the filesystem watch or event bus (not via JSONL scan).
        """
        return await self._parse_item(event)
