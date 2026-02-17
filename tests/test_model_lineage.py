"""Tests for model lineage tracking.

Verifies that lineage records are correctly written after training
completes and the deployment gate validates the GGUF model.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reactor_core.data.lineage import (
    LineageRecord,
    write_lineage_record,
    read_lineage_records,
    update_lineage_record,
    LINEAGE_FILE_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lineage_dir(tmp_path):
    """Provide a temporary lineage directory."""
    d = tmp_path / "reactor" / "models"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def lineage_file(lineage_dir):
    """Provide the path where lineage.jsonl will be written."""
    return lineage_dir / "lineage.jsonl"


@pytest.fixture
def sample_gguf(tmp_path):
    """Create a small fake GGUF file so DataHash.from_file works."""
    gguf = tmp_path / "test_model.gguf"
    gguf.write_bytes(b"\x00" * 2048)
    return gguf


@pytest.fixture
def sample_record():
    """A minimal valid lineage record dict."""
    return {
        "model_id": "jarvis-trained-v0.1.1",
        "model_hash": "abc123def456",
        "parent_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "training_method": "lora_sft",
        "training_job_id": "job-1234",
        "dataset": {
            "hash": "dataset_sha256_abc",
            "size": 847,
            "date_range": ["2026-02-10", "2026-02-15"],
            "source_distribution": {"jarvis_body": 612, "corrections": 37},
            "weighted_score": 142.5,
        },
        "eval_scores": {"overall_quality": 0.82, "safety": 0.98},
        "gate_decision": "APPROVED",
        "deployed_at": None,
        "probation_result": None,
        "transformation_steps": [],
    }


# ---------------------------------------------------------------------------
# LineageRecord tests
# ---------------------------------------------------------------------------

class TestLineageRecord:
    """Tests for the LineageRecord dataclass."""

    def test_from_dict_roundtrip(self, sample_record):
        """A record should survive dict -> LineageRecord -> dict roundtrip."""
        rec = LineageRecord.from_dict(sample_record)
        out = rec.to_dict()
        assert out["model_id"] == sample_record["model_id"]
        assert out["model_hash"] == sample_record["model_hash"]
        assert out["gate_decision"] == "APPROVED"
        assert out["deployed_at"] is None
        assert out["probation_result"] is None

    def test_to_dict_includes_timestamp(self, sample_record):
        """The dict output should include a created_at timestamp."""
        rec = LineageRecord.from_dict(sample_record)
        out = rec.to_dict()
        assert "created_at" in out
        # Should be a valid ISO timestamp
        datetime.fromisoformat(out["created_at"])

    def test_from_dict_with_extra_fields(self, sample_record):
        """Extra unknown fields should be silently ignored."""
        sample_record["unknown_field"] = "should_be_ignored"
        rec = LineageRecord.from_dict(sample_record)
        assert rec.model_id == "jarvis-trained-v0.1.1"

    def test_minimal_record(self):
        """A record with only model_id should still be valid."""
        rec = LineageRecord(model_id="minimal-model")
        out = rec.to_dict()
        assert out["model_id"] == "minimal-model"
        assert out["model_hash"] is None
        assert out["dataset"] is None


# ---------------------------------------------------------------------------
# write_lineage_record tests
# ---------------------------------------------------------------------------

class TestWriteLineageRecord:
    """Tests for the write_lineage_record function."""

    def test_creates_directory_and_file(self, lineage_dir, sample_record):
        """Should create the lineage.jsonl file if it doesn't exist."""
        target = lineage_dir / "lineage.jsonl"
        assert not target.exists()

        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        assert target.exists()
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["model_id"] == "jarvis-trained-v0.1.1"
        assert data["gate_decision"] == "APPROVED"

    def test_appends_multiple_records(self, lineage_dir, sample_record):
        """Multiple writes should append lines, not overwrite."""
        record1 = LineageRecord.from_dict(sample_record)
        sample_record["model_id"] = "second-model"
        record2 = LineageRecord.from_dict(sample_record)

        write_lineage_record(record1, lineage_dir=lineage_dir)
        write_lineage_record(record2, lineage_dir=lineage_dir)

        target = lineage_dir / "lineage.jsonl"
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["model_id"] == "jarvis-trained-v0.1.1"
        assert json.loads(lines[1])["model_id"] == "second-model"

    def test_each_line_is_valid_json(self, lineage_dir, sample_record):
        """Every line in the JSONL file should be parseable as JSON."""
        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        target = lineage_dir / "lineage.jsonl"
        for line in target.read_text().strip().split("\n"):
            json.loads(line)  # Should not raise

    def test_default_lineage_dir(self, sample_record, tmp_path):
        """When lineage_dir is None, should use LINEAGE_FILE_PATH parent."""
        record = LineageRecord.from_dict(sample_record)
        # Patch LINEAGE_FILE_PATH to a temp location
        custom_path = tmp_path / "reactor" / "models" / "lineage.jsonl"
        with patch("reactor_core.data.lineage.LINEAGE_FILE_PATH", custom_path):
            write_lineage_record(record)
        assert custom_path.exists()

    def test_record_contains_created_at(self, lineage_dir, sample_record):
        """Written record should have a created_at timestamp."""
        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        target = lineage_dir / "lineage.jsonl"
        data = json.loads(target.read_text().strip())
        assert "created_at" in data
        datetime.fromisoformat(data["created_at"])


# ---------------------------------------------------------------------------
# read_lineage_records tests
# ---------------------------------------------------------------------------

class TestReadLineageRecords:
    """Tests for reading lineage records back."""

    def test_read_empty_file(self, lineage_dir):
        """Reading from non-existent file should return empty list."""
        records = read_lineage_records(lineage_dir=lineage_dir)
        assert records == []

    def test_read_written_records(self, lineage_dir, sample_record):
        """Should read back what was written."""
        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        records = read_lineage_records(lineage_dir=lineage_dir)
        assert len(records) == 1
        assert records[0]["model_id"] == "jarvis-trained-v0.1.1"

    def test_read_multiple_records(self, lineage_dir, sample_record):
        """Should read all appended records."""
        for i in range(5):
            sample_record["model_id"] = f"model-{i}"
            record = LineageRecord.from_dict(sample_record)
            write_lineage_record(record, lineage_dir=lineage_dir)

        records = read_lineage_records(lineage_dir=lineage_dir)
        assert len(records) == 5
        assert records[4]["model_id"] == "model-4"


# ---------------------------------------------------------------------------
# update_lineage_record tests
# ---------------------------------------------------------------------------

class TestUpdateLineageRecord:
    """Tests for updating an existing lineage record."""

    def test_update_deployed_at(self, lineage_dir, sample_record):
        """Should update the deployed_at field of an existing record."""
        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        updated = update_lineage_record(
            model_id="jarvis-trained-v0.1.1",
            updates={"deployed_at": "2026-02-17T12:00:00"},
            lineage_dir=lineage_dir,
        )
        assert updated is True

        records = read_lineage_records(lineage_dir=lineage_dir)
        assert len(records) == 1
        assert records[0]["deployed_at"] == "2026-02-17T12:00:00"

    def test_update_probation_result(self, lineage_dir, sample_record):
        """Should update the probation_result field."""
        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        updated = update_lineage_record(
            model_id="jarvis-trained-v0.1.1",
            updates={"probation_result": "COMMITTED"},
            lineage_dir=lineage_dir,
        )
        assert updated is True

        records = read_lineage_records(lineage_dir=lineage_dir)
        assert records[0]["probation_result"] == "COMMITTED"

    def test_update_nonexistent_record(self, lineage_dir, sample_record):
        """Updating a non-existent model_id should return False."""
        record = LineageRecord.from_dict(sample_record)
        write_lineage_record(record, lineage_dir=lineage_dir)

        updated = update_lineage_record(
            model_id="nonexistent-model",
            updates={"deployed_at": "2026-02-17T12:00:00"},
            lineage_dir=lineage_dir,
        )
        assert updated is False

    def test_update_preserves_other_records(self, lineage_dir, sample_record):
        """Updating one record should not affect others."""
        for i in range(3):
            sample_record["model_id"] = f"model-{i}"
            record = LineageRecord.from_dict(sample_record)
            write_lineage_record(record, lineage_dir=lineage_dir)

        update_lineage_record(
            model_id="model-1",
            updates={"probation_result": "ROLLED_BACK"},
            lineage_dir=lineage_dir,
        )

        records = read_lineage_records(lineage_dir=lineage_dir)
        assert len(records) == 3
        assert records[0]["probation_result"] is None
        assert records[1]["probation_result"] == "ROLLED_BACK"
        assert records[2]["probation_result"] is None
