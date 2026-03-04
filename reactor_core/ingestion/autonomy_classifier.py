"""
Centralized Autonomy Event Classifier (v300.0 — Phase 2 Trinity Autonomy Wiring)
=================================================================================

Single source of truth for training label classification of autonomy events.
Every pipeline stage — ingestor, training pipeline, deployment gate — inherits
the same exclusion policy from this module.

Event Type → (InteractionOutcome, should_train)
─────────────────────────────────────────────────
committed          → (SUCCESS, True)
failed             → (FAILURE, True)
policy_denied      → (INFRASTRUCTURE, False)
no_journal_lease   → (INFRASTRUCTURE, False)
superseded         → (DEFERRED, False)    # until reconciled
deduplicated       → (UNKNOWN, False)     # exclude
intent_written     → (UNKNOWN, False)     # exclude
"""

from __future__ import annotations

from typing import Tuple

from reactor_core.ingestion.base_ingestor import InteractionOutcome


# ---------------------------------------------------------------------------
# Canonical autonomy event type sets — must stay in sync with JARVIS Body
# constants (google_workspace_agent.py → AUTONOMY_EVENT_TYPES).
# ---------------------------------------------------------------------------

TRAINABLE: frozenset = frozenset({"committed", "failed"})
INFRASTRUCTURE: frozenset = frozenset({"policy_denied", "no_journal_lease"})
EXCLUDE: frozenset = frozenset({"deduplicated", "intent_written"})
RECONCILE_ONLY: frozenset = frozenset({"superseded"})

ALL_KNOWN_TYPES: frozenset = TRAINABLE | INFRASTRUCTURE | EXCLUDE | RECONCILE_ONLY

# Schema version this classifier is compatible with
SUPPORTED_SCHEMA_VERSIONS: frozenset = frozenset({"1.0"})


class AutonomyEventClassifier:
    """Centralized training label classifier for autonomy events.

    Every pipeline stage inherits the same exclusion policy.
    Instantiate once per pipeline run or use as a stateless utility.
    """

    __slots__ = ()

    # Expose sets as class attributes for external introspection
    TRAINABLE = TRAINABLE
    INFRASTRUCTURE = INFRASTRUCTURE
    EXCLUDE = EXCLUDE
    RECONCILE_ONLY = RECONCILE_ONLY

    @staticmethod
    def classify(event_type: str) -> Tuple[InteractionOutcome, bool]:
        """Classify an autonomy event type into (outcome, should_train).

        Parameters
        ----------
        event_type:
            One of the 7 canonical autonomy event types.

        Returns
        -------
        (InteractionOutcome, bool)
            The mapped outcome and whether the event should enter the
            training pipeline.  Unknown event types default to
            ``(UNKNOWN, False)`` — safe fallback.
        """
        if event_type in TRAINABLE:
            outcome = (
                InteractionOutcome.SUCCESS
                if event_type == "committed"
                else InteractionOutcome.FAILURE
            )
            return outcome, True

        if event_type in INFRASTRUCTURE:
            return InteractionOutcome.INFRASTRUCTURE, False

        if event_type in RECONCILE_ONLY:
            return InteractionOutcome.DEFERRED, False

        # EXCLUDE and truly unknown — never train
        return InteractionOutcome.UNKNOWN, False

    @staticmethod
    def is_known_type(event_type: str) -> bool:
        """Return True if *event_type* is a recognized autonomy event."""
        return event_type in ALL_KNOWN_TYPES

    @staticmethod
    def is_schema_supported(version: str) -> bool:
        """Return True if *version* is in the supported set."""
        return version in SUPPORTED_SCHEMA_VERSIONS
