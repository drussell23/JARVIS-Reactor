"""
Telemetry event ingestor for JARVIS TelemetryEvent data.

Parses telemetry events from JSONL files and converts to RawInteraction format.

Expected format (from JARVIS backend/core/telemetry/events.py):
{
    "event_type": "INTENT_DETECTED",
    "timestamp": "2024-12-22T02:15:30.123456Z",
    "event_id": "uuid",
    "session_id": "session-uuid",
    "user_id": "user-id",
    "properties": {...},
    "metrics": {...},
    "tags": [...]
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiofiles

from reactor_core.ingestion.base_ingestor import (
    AbstractIngestor,
    InteractionOutcome,
    RawInteraction,
    SourceType,
)

logger = logging.getLogger(__name__)


# Map JARVIS event types to outcomes
EVENT_OUTCOME_MAP = {
    # Success events
    "INTENT_DETECTED": InteractionOutcome.SUCCESS,
    "ROUTE_MATCHED": InteractionOutcome.SUCCESS,
    "VISION_ANALYSIS_COMPLETED": InteractionOutcome.SUCCESS,
    "SEMANTIC_MATCH_FOUND": InteractionOutcome.SUCCESS,
    "FOLLOWUP_RESOLVED": InteractionOutcome.SUCCESS,
    "CONTEXT_CREATED": InteractionOutcome.SUCCESS,

    # Failure events
    "INTENT_CLASSIFICATION_FAILED": InteractionOutcome.FAILURE,
    "ROUTE_FAILED": InteractionOutcome.FAILURE,
    "ROUTE_NO_MATCH": InteractionOutcome.FAILURE,
    "SEMANTIC_MATCH_NONE": InteractionOutcome.FAILURE,
    "FOLLOWUP_CONTEXT_MISSING": InteractionOutcome.FAILURE,
    "ERROR_OCCURRED": InteractionOutcome.FAILURE,

    # Partial/neutral events
    "CONTEXT_ACCESSED": InteractionOutcome.PARTIAL,
    "CONTEXT_CONSUMED": InteractionOutcome.PARTIAL,
    "CONTEXT_EXPIRED": InteractionOutcome.PARTIAL,
    "VISION_SNAPSHOT_TAKEN": InteractionOutcome.PARTIAL,
    "VISION_OCR_EXECUTED": InteractionOutcome.PARTIAL,
    "FOLLOWUP_INITIATED": InteractionOutcome.PARTIAL,
    "LATENCY_RECORDED": InteractionOutcome.PARTIAL,
}


class TelemetryIngestor(AbstractIngestor):
    """
    Ingestor for JARVIS telemetry events.

    Extracts:
    - User intents and detected commands
    - Successful task completions
    - Error events with context
    - Performance metrics
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        include_failures: bool = True,
        include_partial: bool = False,
        event_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize telemetry ingestor.

        Args:
            min_confidence: Minimum confidence threshold
            include_failures: Include failed events
            include_partial: Include partial/neutral events
            event_types: Filter to specific event types (None = all)
            tags: Tags to add to all interactions
        """
        super().__init__(
            min_confidence=min_confidence,
            include_failures=include_failures,
            tags=tags,
        )
        self.include_partial = include_partial
        self.event_types = set(event_types) if event_types else None

    @property
    def source_type(self) -> SourceType:
        return SourceType.TELEMETRY

    async def _iter_items(
        self,
        source_path: Path,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Iterate over JSONL telemetry events."""
        if not source_path.exists():
            logger.warning(f"Telemetry file not found: {source_path}")
            return

        try:
            async with aiofiles.open(source_path, "r") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Parse timestamp
                    timestamp_str = event.get("timestamp")
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )

                            # Time range filter
                            if since and timestamp < since:
                                continue
                            if until and timestamp > until:
                                continue

                            event["_parsed_timestamp"] = timestamp
                        except ValueError:
                            pass

                    # Event type filter
                    if self.event_types:
                        event_type = event.get("event_type", "")
                        if event_type not in self.event_types:
                            continue

                    yield event

        except Exception as e:
            logger.error(f"Error reading telemetry file {source_path}: {e}")

    async def _parse_item(
        self,
        item: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> Optional[RawInteraction]:
        """Parse a telemetry event into RawInteraction."""
        event_type = item.get("event_type", "")
        properties = item.get("properties", {})
        metrics = item.get("metrics", {})

        # Determine outcome
        outcome = EVENT_OUTCOME_MAP.get(event_type, InteractionOutcome.UNKNOWN)

        # Skip partial events if not requested
        if outcome == InteractionOutcome.PARTIAL and not self.include_partial:
            return None

        # Extract user input and assistant output
        user_input = self._extract_user_input(properties)
        assistant_output = self._extract_assistant_output(properties)

        # Skip if no meaningful content
        if not user_input and not assistant_output:
            # Some events are still valuable for context
            if event_type not in ("ERROR_OCCURRED", "INTENT_DETECTED"):
                return None

        # Calculate confidence from metrics
        confidence = self._calculate_confidence(metrics, properties)

        # v242.0: Parse Trinity classification metadata
        routing = item.get("x_jarvis_routing", {})
        if routing:
            properties["phi_intent"] = routing.get("intent", "")
            properties["phi_domain"] = routing.get("domain", "")
            properties["phi_complexity"] = routing.get("complexity", "")
            properties["phi_confidence"] = routing.get("confidence", 0.0)
            properties["escalated_to_claude"] = routing.get("escalated", False)
            properties["classifier_model"] = routing.get("classifier_model", "")
            properties["generator_model"] = routing.get("generator_model", "")
            properties["classification_ms"] = routing.get("classification_ms", 0)
            properties["generation_ms"] = routing.get("generation_ms", 0)

            # Override confidence with Phi's classification confidence
            if routing.get("confidence"):
                confidence = routing["confidence"]

        # Parse timestamp
        timestamp = item.get("_parsed_timestamp")
        if not timestamp:
            timestamp_str = item.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(
                    timestamp_str.replace("Z", "+00:00")
                )
            except ValueError:
                timestamp = datetime.now()

        # Build tags list
        interaction_tags = list(item.get("tags", [])) + [f"event:{event_type}"]

        # v242.0: Tag escalated queries as DPO training candidates
        if routing.get("escalated"):
            interaction_tags.append("escalated")
            interaction_tags.append(
                f"escalation_domain:{routing.get('domain', 'unknown')}"
            )
            # This is a DPO training signal: local model couldn't handle it
            properties["dpo_candidate"] = True

        return RawInteraction(
            timestamp=timestamp,
            source_type=SourceType.TELEMETRY,
            source_id=item.get("event_id", ""),
            source_file=source_file,
            user_input=user_input,
            assistant_output=assistant_output,
            system_context=self._extract_context(properties),
            session_id=item.get("session_id"),
            user_id=item.get("user_id"),
            outcome=outcome,
            confidence=confidence,
            properties=properties,
            tags=interaction_tags,
            environment=self._extract_environment(properties),
        )

    def _extract_user_input(self, properties: Dict[str, Any]) -> Optional[str]:
        """Extract user input from event properties."""
        # Try common property names
        for key in ("user_input", "input", "query", "command", "text", "message", "intent"):
            if key in properties:
                value = properties[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, dict) and "text" in value:
                    return value["text"]

        return None

    def _extract_assistant_output(self, properties: Dict[str, Any]) -> Optional[str]:
        """Extract assistant output from event properties."""
        for key in ("output", "response", "result", "assistant_output", "reply"):
            if key in properties:
                value = properties[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                elif isinstance(value, dict) and "text" in value:
                    return value["text"]

        return None

    def _extract_context(self, properties: Dict[str, Any]) -> Optional[str]:
        """Extract system context from event properties."""
        context_parts = []

        for key in ("context", "system_context", "background", "state"):
            if key in properties:
                value = properties[key]
                if isinstance(value, str):
                    context_parts.append(value)
                elif isinstance(value, dict):
                    context_parts.append(json.dumps(value))

        return "\n".join(context_parts) if context_parts else None

    def _extract_environment(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environment context."""
        env = {}

        for key in ("environment", "env", "device", "location", "network"):
            if key in properties:
                env[key] = properties[key]

        return env

    def _calculate_confidence(
        self,
        metrics: Dict[str, float],
        properties: Dict[str, Any],
    ) -> float:
        """Calculate confidence score from metrics and properties."""
        confidence = 0.5  # Default

        # Check metrics for confidence values
        for key in ("confidence", "score", "probability", "certainty"):
            if key in metrics:
                confidence = float(metrics[key])
                break

        # Check properties for confidence
        if confidence == 0.5:
            for key in ("confidence", "score", "probability"):
                if key in properties:
                    try:
                        confidence = float(properties[key])
                        break
                    except (ValueError, TypeError):
                        pass

        # Normalize to 0-1
        if confidence > 1:
            confidence = confidence / 100 if confidence <= 100 else 1.0

        return min(1.0, max(0.0, confidence))

    def supports_streaming(self) -> bool:
        return True
