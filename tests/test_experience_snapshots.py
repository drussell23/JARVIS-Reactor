"""Tests for atomic experience snapshot functionality.

Verifies that the training pipeline:
1. Atomically drains the experience buffer under lock
2. Writes a JSONL snapshot file to ~/.jarvis/reactor/training_data/
3. Computes a DataHash of the snapshot for dataset versioning
4. Stores the hash in the job's metadata
5. Handles empty buffers gracefully (skip snapshot, no crash)
"""

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def tmp_persist_dir(tmp_path):
    """Temp directory for job persistence."""
    return tmp_path / "reactor_state"


@pytest.fixture
def tmp_snapshot_dir(tmp_path):
    """Temp directory for training data snapshots."""
    return tmp_path / "training_data"


@pytest.fixture
def job_manager(tmp_persist_dir):
    """Fresh TrainingJobManager with temp persistence dir."""
    from reactor_core.api.server import TrainingJobManager
    return TrainingJobManager(persist_dir=tmp_persist_dir)


@pytest.fixture
def sample_experiences():
    """Sample experiences for testing."""
    return [
        {
            "user_input": "What is Python?",
            "assistant_output": "Python is a programming language.",
            "quality_score": 0.9,
            "source": "jarvis",
            "ingested_at": "2026-02-17T10:00:00",
        },
        {
            "user_input": "Tell me about ML",
            "assistant_output": "ML is a subset of AI.",
            "quality_score": 0.85,
            "source": "jarvis",
            "ingested_at": "2026-02-17T10:01:00",
        },
        {
            "user_input": "How do I use asyncio?",
            "assistant_output": "asyncio provides async/await.",
            "quality_score": 0.95,
            "source": "scout",
            "ingested_at": "2026-02-17T10:02:00",
        },
    ]


class TestDrainExperienceBuffer:
    """Test atomic drain of the experience buffer."""

    def test_drain_returns_copy_and_clears_buffer(self, job_manager, sample_experiences):
        """drain_experience_buffer should return all experiences and clear the buffer."""
        from reactor_core.api.server import drain_experience_buffer

        job_manager.experiences = list(sample_experiences)
        loop = asyncio.get_event_loop()
        drained = loop.run_until_complete(drain_experience_buffer(job_manager))

        # Should return all experiences
        assert len(drained) == 3
        assert drained[0]["user_input"] == "What is Python?"
        # Buffer should be empty now
        assert len(job_manager.experiences) == 0

    def test_drain_empty_buffer_returns_empty_list(self, job_manager):
        """drain_experience_buffer on empty buffer should return empty list."""
        from reactor_core.api.server import drain_experience_buffer

        loop = asyncio.get_event_loop()
        drained = loop.run_until_complete(drain_experience_buffer(job_manager))
        assert drained == []
        assert job_manager.experiences == []

    def test_drain_is_atomic(self, job_manager, sample_experiences):
        """After drain, no experiences should remain even if buffer was large."""
        from reactor_core.api.server import drain_experience_buffer

        job_manager.experiences = list(sample_experiences) * 100  # 300 experiences
        loop = asyncio.get_event_loop()
        drained = loop.run_until_complete(drain_experience_buffer(job_manager))

        assert len(drained) == 300
        assert len(job_manager.experiences) == 0


class TestWriteSnapshot:
    """Test writing experience snapshots to JSONL files."""

    def test_write_snapshot_creates_jsonl_file(self, tmp_snapshot_dir, sample_experiences):
        """write_experience_snapshot should create a .jsonl file."""
        from reactor_core.api.server import write_experience_snapshot

        path = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-001",
            snapshot_dir=tmp_snapshot_dir,
        )

        assert path.exists()
        assert path.name == "snapshot_test-001.jsonl"
        assert path.parent == tmp_snapshot_dir

    def test_snapshot_contains_one_json_per_line(self, tmp_snapshot_dir, sample_experiences):
        """Each line in the snapshot should be valid JSON, one experience per line."""
        from reactor_core.api.server import write_experience_snapshot

        path = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-002",
            snapshot_dir=tmp_snapshot_dir,
        )

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            parsed = json.loads(line)
            assert "user_input" in parsed

    def test_snapshot_dir_created_automatically(self, tmp_path, sample_experiences):
        """Snapshot directory should be created if it doesn't exist."""
        from reactor_core.api.server import write_experience_snapshot

        nested = tmp_path / "deep" / "nested" / "training_data"
        path = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-003",
            snapshot_dir=nested,
        )
        assert nested.exists()
        assert path.exists()

    def test_snapshot_atomic_write(self, tmp_snapshot_dir, sample_experiences):
        """Snapshot should not leave .tmp files on success."""
        from reactor_core.api.server import write_experience_snapshot

        write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-004",
            snapshot_dir=tmp_snapshot_dir,
        )

        tmp_files = list(tmp_snapshot_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_snapshot_content_matches_experiences(self, tmp_snapshot_dir, sample_experiences):
        """Parsed JSONL content should match input experiences."""
        from reactor_core.api.server import write_experience_snapshot

        path = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-005",
            snapshot_dir=tmp_snapshot_dir,
        )

        lines = path.read_text().strip().split("\n")
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["user_input"] == sample_experiences[i]["user_input"]
            assert parsed["assistant_output"] == sample_experiences[i]["assistant_output"]


class TestSnapshotHash:
    """Test DataHash computation for snapshot files."""

    def test_snapshot_hash_computed(self, tmp_snapshot_dir, sample_experiences):
        """write_experience_snapshot should return a DataHash."""
        from reactor_core.api.server import write_experience_snapshot
        from reactor_core.data.versioning import DataHash

        path = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-hash-1",
            snapshot_dir=tmp_snapshot_dir,
        )

        data_hash = DataHash.from_file(path)
        assert data_hash.digest != ""
        assert data_hash.algorithm == "sha256"
        assert data_hash.size_bytes > 0

    def test_same_data_produces_same_hash(self, tmp_snapshot_dir, sample_experiences):
        """Identical experiences should produce identical hashes."""
        from reactor_core.api.server import write_experience_snapshot
        from reactor_core.data.versioning import DataHash

        path1 = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-hash-2a",
            snapshot_dir=tmp_snapshot_dir,
        )
        path2 = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-hash-2b",
            snapshot_dir=tmp_snapshot_dir,
        )

        hash1 = DataHash.from_file(path1)
        hash2 = DataHash.from_file(path2)
        assert hash1.digest == hash2.digest

    def test_different_data_produces_different_hash(self, tmp_snapshot_dir, sample_experiences):
        """Different experiences should produce different hashes."""
        from reactor_core.api.server import write_experience_snapshot
        from reactor_core.data.versioning import DataHash

        path1 = write_experience_snapshot(
            experiences=sample_experiences,
            job_id="test-hash-3a",
            snapshot_dir=tmp_snapshot_dir,
        )

        modified = sample_experiences.copy()
        modified.append({"user_input": "extra", "assistant_output": "data"})

        path2 = write_experience_snapshot(
            experiences=modified,
            job_id="test-hash-3b",
            snapshot_dir=tmp_snapshot_dir,
        )

        hash1 = DataHash.from_file(path1)
        hash2 = DataHash.from_file(path2)
        assert hash1.digest != hash2.digest


class TestJobMetadataHashStorage:
    """Test that snapshot hash gets stored in job metadata."""

    def test_hash_stored_in_job_metadata(self, job_manager, sample_experiences, tmp_snapshot_dir):
        """After snapshot, the job metadata should contain dataset_hash."""
        from reactor_core.api.server import drain_experience_buffer, write_experience_snapshot
        from reactor_core.data.versioning import DataHash

        loop = asyncio.get_event_loop()

        # Create a job
        job = loop.run_until_complete(
            job_manager.create_job(
                experience_count=3, priority="normal",
                sources=["test"], metadata={}, triggered_by="test",
            )
        )
        job_id = job["job_id"]

        # Add experiences and drain
        job_manager.experiences = list(sample_experiences)
        drained = loop.run_until_complete(drain_experience_buffer(job_manager))

        # Write snapshot
        snapshot_path = write_experience_snapshot(
            experiences=drained,
            job_id=job_id,
            snapshot_dir=tmp_snapshot_dir,
        )

        # Compute hash and store
        data_hash = DataHash.from_file(snapshot_path)
        loop.run_until_complete(
            job_manager.update_job(
                job_id,
                metadata={
                    **job["metadata"],
                    "dataset_hash": str(data_hash),
                    "dataset_hash_digest": data_hash.digest,
                    "snapshot_path": str(snapshot_path),
                    "snapshot_size_bytes": data_hash.size_bytes,
                },
            )
        )

        # Verify
        updated_job = loop.run_until_complete(job_manager.get_job(job_id))
        assert "dataset_hash" in updated_job["metadata"]
        assert "dataset_hash_digest" in updated_job["metadata"]
        assert updated_job["metadata"]["dataset_hash_digest"] == data_hash.digest
        assert "snapshot_path" in updated_job["metadata"]


class TestEmptyBufferHandling:
    """Test graceful handling of empty experience buffers."""

    def test_empty_buffer_skips_snapshot(self, job_manager, tmp_snapshot_dir):
        """When buffer is empty, no snapshot file should be created."""
        from reactor_core.api.server import drain_experience_buffer, write_experience_snapshot

        loop = asyncio.get_event_loop()
        drained = loop.run_until_complete(drain_experience_buffer(job_manager))

        assert drained == []

        # write_experience_snapshot should return None for empty data
        result = write_experience_snapshot(
            experiences=[],
            job_id="test-empty",
            snapshot_dir=tmp_snapshot_dir,
        )
        assert result is None

        # No file should exist
        snapshot_files = list(tmp_snapshot_dir.glob("snapshot_*.jsonl"))
        assert len(snapshot_files) == 0
