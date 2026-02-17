# tests/test_pipeline_events.py
"""Tests for the pipeline event logger (v3.1).

Verifies that emit_pipeline_event writes structured JSONL events
with correct fields and handles errors gracefully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def events_dir(tmp_path):
    """Provide a temporary directory for pipeline events."""
    events_path = tmp_path / "events"
    events_path.mkdir()
    return events_path


@pytest.fixture
def emit_fn(events_dir):
    """Return emit_pipeline_event patched to use the temp directory."""
    with patch("reactor_core.api.server._PIPELINE_EVENTS_DIR", events_dir), \
         patch("reactor_core.api.server._PIPELINE_EVENTS_FILE", events_dir / "pipeline_events.jsonl"):
        from reactor_core.api.server import emit_pipeline_event
        yield emit_pipeline_event


# =========================================================================
# Tests
# =========================================================================

class TestEmitPipelineEvent:
    """Verify emit_pipeline_event writes correct JSONL events."""

    def test_writes_event_to_file(self, emit_fn, events_dir):
        """Should write a JSONL line to the events file."""
        event_id = emit_fn(
            topic="training.started",
            payload={"job_id": "test-123"},
            correlation_id="test-123",
        )

        assert event_id is not None
        events_file = events_dir / "pipeline_events.jsonl"
        assert events_file.exists()

        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["topic"] == "training.started"
        assert event["source"] == "reactor"
        assert event["correlation_id"] == "test-123"
        assert event["payload"]["job_id"] == "test-123"
        assert event["event_id"] == event_id

    def test_multiple_events_append(self, emit_fn, events_dir):
        """Multiple calls should append lines to the same file."""
        emit_fn(topic="training.started", payload={"job_id": "j1"})
        emit_fn(topic="training.completed", payload={"job_id": "j1"})
        emit_fn(topic="training.failed", payload={"job_id": "j2"})

        events_file = events_dir / "pipeline_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

        topics = [json.loads(line)["topic"] for line in lines]
        assert topics == ["training.started", "training.completed", "training.failed"]

    def test_event_has_required_fields(self, emit_fn, events_dir):
        """Each event should have all required fields."""
        emit_fn(
            topic="gate.evaluated",
            payload={"tier": "ABUNDANT"},
            correlation_id="corr-1",
            causation_id="cause-1",
        )

        events_file = events_dir / "pipeline_events.jsonl"
        event = json.loads(events_file.read_text().strip())

        required_fields = ["event_id", "topic", "source", "timestamp",
                           "correlation_id", "causation_id", "payload"]
        for field in required_fields:
            assert field in event, f"Missing field: {field}"

        assert event["causation_id"] == "cause-1"

    def test_returns_event_id(self, emit_fn):
        """Should return a non-empty event_id string."""
        event_id = emit_fn(topic="training.started")
        assert event_id is not None
        assert len(event_id) > 0

    def test_default_correlation_id_is_event_id(self, emit_fn, events_dir):
        """When no correlation_id is provided, it should default to event_id."""
        event_id = emit_fn(topic="training.started")

        events_file = events_dir / "pipeline_events.jsonl"
        event = json.loads(events_file.read_text().strip())
        assert event["correlation_id"] == event_id

    def test_survives_write_error(self, events_dir):
        """Should return None (not crash) when write fails."""
        # Patch to a non-writable path
        bad_dir = Path("/nonexistent/path/that/does/not/exist")
        with patch("reactor_core.api.server._PIPELINE_EVENTS_DIR", bad_dir), \
             patch("reactor_core.api.server._PIPELINE_EVENTS_FILE", bad_dir / "events.jsonl"):
            from reactor_core.api.server import emit_pipeline_event
            result = emit_pipeline_event(topic="training.started")
            # Should not crash, returns None
            assert result is None
