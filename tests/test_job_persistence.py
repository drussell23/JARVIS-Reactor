"""Tests for training job persistence across restarts."""

import json
import pytest
from pathlib import Path
from datetime import datetime


@pytest.fixture
def tmp_jobs_dir(tmp_path):
    """Temporary directory for job persistence."""
    return tmp_path / "reactor_state"


class TestJobPersistence:
    """Verify jobs survive process restarts via file persistence."""

    def test_create_job_persists_to_file(self, tmp_jobs_dir):
        """Creating a job should write to jobs.json."""
        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)

        import asyncio
        job = asyncio.get_event_loop().run_until_complete(
            mgr.create_job(experience_count=100, priority="normal", sources=["jarvis"], metadata={}, triggered_by="test")
        )

        jobs_file = tmp_jobs_dir / "jobs.json"
        assert jobs_file.exists()
        data = json.loads(jobs_file.read_text())
        assert job["job_id"] in data

    def test_load_jobs_on_init(self, tmp_jobs_dir):
        """New TrainingJobManager should load existing jobs from disk."""
        tmp_jobs_dir.mkdir(parents=True, exist_ok=True)
        jobs_file = tmp_jobs_dir / "jobs.json"
        jobs_file.write_text(json.dumps({
            "restored-job-1": {
                "job_id": "restored-job-1",
                "status": "running",
                "created_at": datetime.now().isoformat(),
            }
        }))

        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)
        assert "restored-job-1" in mgr.jobs
        assert mgr.jobs["restored-job-1"]["status"] == "running"

    def test_update_job_persists(self, tmp_jobs_dir):
        """Updating job status should persist the change."""
        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)

        import asyncio
        loop = asyncio.get_event_loop()
        job = loop.run_until_complete(
            mgr.create_job(experience_count=50, priority="high", sources=[], metadata={}, triggered_by="test")
        )

        loop.run_until_complete(mgr.update_job(job["job_id"], status="completed", metrics={"loss": 0.42}))

        # Re-read from disk
        data = json.loads((tmp_jobs_dir / "jobs.json").read_text())
        assert data[job["job_id"]]["status"] == "completed"

    def test_persist_dir_created_automatically(self, tmp_path):
        """persist_dir should be created if it does not exist."""
        from reactor_core.api.server import TrainingJobManager

        nested = tmp_path / "deep" / "nested" / "dir"
        mgr = TrainingJobManager(persist_dir=nested)
        assert nested.exists()

    def test_corrupt_jobs_file_handled_gracefully(self, tmp_jobs_dir):
        """Corrupt jobs.json should not crash init."""
        from reactor_core.api.server import TrainingJobManager

        tmp_jobs_dir.mkdir(parents=True, exist_ok=True)
        (tmp_jobs_dir / "jobs.json").write_text("NOT VALID JSON {{{")

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)
        assert mgr.jobs == {}

    def test_atomic_write_leaves_no_tmp_on_success(self, tmp_jobs_dir):
        """After successful persist, no .tmp file should remain."""
        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mgr.create_job(experience_count=10, priority="low", sources=[], metadata={}, triggered_by="test")
        )

        tmp_file = tmp_jobs_dir / "jobs.tmp"
        assert not tmp_file.exists()

    def test_multiple_jobs_persisted(self, tmp_jobs_dir):
        """Multiple jobs should all be persisted."""
        from reactor_core.api.server import TrainingJobManager

        mgr = TrainingJobManager(persist_dir=tmp_jobs_dir)

        import asyncio
        loop = asyncio.get_event_loop()
        job1 = loop.run_until_complete(
            mgr.create_job(experience_count=10, priority="low", sources=[], metadata={}, triggered_by="test")
        )
        job2 = loop.run_until_complete(
            mgr.create_job(experience_count=20, priority="high", sources=[], metadata={}, triggered_by="test")
        )

        data = json.loads((tmp_jobs_dir / "jobs.json").read_text())
        assert job1["job_id"] in data
        assert job2["job_id"] in data
        assert len(data) == 2

    def test_restart_simulation(self, tmp_jobs_dir):
        """Simulate restart: create manager, add jobs, create new manager, verify jobs loaded."""
        from reactor_core.api.server import TrainingJobManager

        import asyncio
        loop = asyncio.get_event_loop()

        # First "process"
        mgr1 = TrainingJobManager(persist_dir=tmp_jobs_dir)
        job = loop.run_until_complete(
            mgr1.create_job(experience_count=100, priority="urgent", sources=["test"], metadata={"key": "value"}, triggered_by="test")
        )
        del mgr1  # "Process dies"

        # Second "process" (restart)
        mgr2 = TrainingJobManager(persist_dir=tmp_jobs_dir)
        assert job["job_id"] in mgr2.jobs
        assert mgr2.jobs[job["job_id"]]["priority"] == "urgent"
        assert mgr2.jobs[job["job_id"]]["experience_count"] == 100
