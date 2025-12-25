"""
Reactor-Core API Server.

Provides REST API endpoints for:
- Training pipeline triggering and management
- Experience log ingestion
- Pipeline status monitoring
- Scout topic management
- Cross-repo integration

This is the API that JARVIS connects to via the "Ignition Key" (ReactorCoreClient).

Usage:
    python -m reactor_core.api.server
    # or
    uvicorn reactor_core.api.server:app --reload --port 8003
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

API_HOST = os.getenv("REACTOR_CORE_HOST", "0.0.0.0")
API_PORT = int(os.getenv("REACTOR_CORE_PORT", "8003"))
JARVIS_API_URL = os.getenv("JARVIS_API_URL", "http://localhost:8000")


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "2.1.0"
    timestamp: str
    pipeline_active: bool = False
    current_stage: Optional[str] = None


class TrainingTriggerRequest(BaseModel):
    """Training trigger request."""
    experience_count: int = Field(default=0, ge=0)
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent)$")
    sources: List[str] = Field(default=["jarvis_experience", "scout"])
    metadata: Dict[str, Any] = Field(default_factory=dict)
    triggered_by: str = Field(default="api")
    trigger_time: Optional[str] = None


class TrainingJobResponse(BaseModel):
    """Training job response."""
    job_id: str
    status: str
    stage: str
    progress: float = 0.0
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    experience_count: int = 0
    priority: str = "normal"
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PipelineStateResponse(BaseModel):
    """Pipeline state response."""
    run_id: str
    stage: str
    started_at: str
    last_updated: str
    progress: float = 0.0
    ingestion_count: int = 0
    formatted_count: int = 0
    distilled_count: int = 0
    training_step: int = 0
    error: Optional[str] = None


class ExperienceStreamRequest(BaseModel):
    """Experience stream request."""
    experience: Dict[str, Any]
    timestamp: Optional[str] = None
    source: str = Field(default="jarvis_agent")


class ExperienceCountResponse(BaseModel):
    """Experience count response."""
    count: int
    last_ingested: Optional[str] = None


class TopicAddRequest(BaseModel):
    """Add learning topic request."""
    topic: str
    category: str = Field(default="general")
    priority: str = Field(default="normal")
    urls: List[str] = Field(default_factory=list)
    added_by: str = Field(default="api")


class StatusResponse(BaseModel):
    """Status response."""
    healthy: bool = True
    pipeline_active: bool = False
    current_job_id: Optional[str] = None
    pending_experiences: int = 0
    last_training: Optional[str] = None
    uptime_seconds: float = 0.0


# ============================================================================
# Training Job Manager (In-Memory for now)
# ============================================================================

class TrainingJobManager:
    """Manages training jobs and pipeline execution."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.current_job_id: Optional[str] = None
        self.experiences: List[Dict[str, Any]] = []
        self.last_training: Optional[datetime] = None
        self.start_time = datetime.now()

    def create_job(
        self,
        experience_count: int,
        priority: str,
        sources: List[str],
        metadata: Dict[str, Any],
        triggered_by: str,
    ) -> Dict[str, Any]:
        """Create a new training job."""
        job_id = str(uuid.uuid4())[:8]
        job = {
            "job_id": job_id,
            "status": "queued",
            "stage": "idle",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "experience_count": experience_count,
            "priority": priority,
            "sources": sources,
            "metadata": metadata,
            "triggered_by": triggered_by,
            "error": None,
            "metrics": {},
        }
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_history(self, limit: int = 10, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get job history."""
        jobs = list(self.jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j["status"] == status_filter]
        # Sort by created_at descending
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return jobs[:limit]

    def start_job(self, job_id: str) -> bool:
        """Start a job."""
        job = self.jobs.get(job_id)
        if job:
            job["status"] = "running"
            job["started_at"] = datetime.now().isoformat()
            self.current_job_id = job_id
            return True
        return False

    def complete_job(self, job_id: str, metrics: Dict[str, Any]) -> bool:
        """Complete a job."""
        job = self.jobs.get(job_id)
        if job:
            job["status"] = "completed"
            job["stage"] = "completed"
            job["progress"] = 100.0
            job["completed_at"] = datetime.now().isoformat()
            job["metrics"] = metrics
            self.current_job_id = None
            self.last_training = datetime.now()
            return True
        return False

    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark a job as failed."""
        job = self.jobs.get(job_id)
        if job:
            job["status"] = "failed"
            job["stage"] = "failed"
            job["error"] = error
            job["completed_at"] = datetime.now().isoformat()
            self.current_job_id = None
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self.jobs.get(job_id)
        if job and job["status"] in ("queued", "running"):
            job["status"] = "cancelled"
            job["completed_at"] = datetime.now().isoformat()
            if self.current_job_id == job_id:
                self.current_job_id = None
            return True
        return False

    def add_experience(self, experience: Dict[str, Any]) -> bool:
        """Add an experience to the pending queue."""
        self.experiences.append({
            **experience,
            "ingested_at": datetime.now().isoformat(),
        })
        return True

    def get_experience_count(self) -> int:
        """Get pending experience count."""
        return len(self.experiences)

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "healthy": True,
            "pipeline_active": self.current_job_id is not None,
            "current_job_id": self.current_job_id,
            "pending_experiences": len(self.experiences),
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


# Global job manager instance
job_manager = TrainingJobManager()


# ============================================================================
# JARVIS Status Broadcaster - The Feedback Loop
# ============================================================================

class JARVISStatusBroadcaster:
    """
    Broadcasts training status updates to JARVIS in real-time.

    This is the "Feedback Loop" that keeps JARVIS informed about
    training progress, enabling voice announcements and UI updates.

    Architecture:
        Reactor-Core Pipeline → Broadcaster → JARVIS /reactor-core/training/status
                                           ↓
                                     JARVIS logs + TTS + WebSocket broadcast
    """

    def __init__(self):
        self._jarvis_url = JARVIS_API_URL
        self._enabled = os.getenv("JARVIS_FEEDBACK_ENABLED", "true").lower() == "true"
        self._timeout = float(os.getenv("JARVIS_FEEDBACK_TIMEOUT", "5.0"))
        self._max_retries = int(os.getenv("JARVIS_FEEDBACK_RETRIES", "2"))
        self._retry_delay = 0.5

        # Stats
        self._notifications_sent = 0
        self._notifications_failed = 0
        self._last_notification: Optional[datetime] = None

        # Session (created lazily)
        self._session = None

        logger.info(
            f"[Broadcaster] Initialized (enabled={self._enabled}, "
            f"jarvis_url={self._jarvis_url})"
        )

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                )
            except ImportError:
                logger.warning("[Broadcaster] aiohttp not installed - notifications disabled")
                self._enabled = False
                return None
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def notify(
        self,
        job_id: str,
        status: str,
        progress: float,
        stage: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        output_model_path: Optional[str] = None,
    ) -> bool:
        """
        Send a status notification to JARVIS.

        Args:
            job_id: Training job identifier
            status: Current status (queued, running, completed, failed)
            progress: Progress percentage 0-100
            stage: Current pipeline stage
            message: Human-readable status message
            metrics: Optional training metrics
            output_model_path: Optional path to output model file

        Returns:
            True if notification was sent successfully
        """
        if not self._enabled:
            return False

        session = await self._get_session()
        if not session:
            return False

        payload = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "stage": stage,
            "message": message,
            "metrics": metrics or {},
            "started_at": datetime.now().isoformat(),
            "experience_count": job_manager.get_job(job_id).get("experience_count", 0) if job_manager.get_job(job_id) else 0,
        }

        if output_model_path:
            payload["output_model_path"] = output_model_path

        endpoint = f"{self._jarvis_url}/reactor-core/training/status"

        for attempt in range(self._max_retries):
            try:
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        self._notifications_sent += 1
                        self._last_notification = datetime.now()
                        result = await response.json()

                        logger.debug(
                            f"[Broadcaster] Notified JARVIS: {stage} {progress:.0f}% "
                            f"(announced={result.get('announced', False)})"
                        )
                        return True

                    elif response.status == 404:
                        # Endpoint not found - JARVIS might not have the router yet
                        logger.debug(f"[Broadcaster] JARVIS endpoint not found (404)")
                        return False

                    else:
                        text = await response.text()
                        logger.warning(
                            f"[Broadcaster] JARVIS notification failed: "
                            f"{response.status} - {text[:100]}"
                        )

            except asyncio.TimeoutError:
                logger.debug(f"[Broadcaster] Timeout (attempt {attempt + 1}/{self._max_retries})")
            except Exception as e:
                logger.debug(f"[Broadcaster] Error: {e} (attempt {attempt + 1}/{self._max_retries})")

            if attempt < self._max_retries - 1:
                await asyncio.sleep(self._retry_delay * (attempt + 1))

        self._notifications_failed += 1
        return False

    async def notify_stage_started(
        self,
        job_id: str,
        stage: str,
        progress: float,
        message: Optional[str] = None,
    ) -> bool:
        """Convenience method for stage start notifications."""
        stage_messages = {
            "data_prep": "Data preparation started",
            "ingesting": "Ingesting training data",
            "formatting": "Formatting data for training",
            "distilling": "Distilling knowledge",
            "fine_tuning": "Fine-tuning model started",
            "training": "Model training in progress",
            "evaluation": "Evaluating model performance",
            "evaluating": "Running evaluation metrics",
            "quantizing": "Quantizing model for deployment",
            "exporting": "Exporting trained model",
            "deploying": "Deploying model",
            "completed": "Training complete!",
        }

        msg = message or stage_messages.get(stage, f"Stage: {stage}")

        return await self.notify(
            job_id=job_id,
            status="running",
            progress=progress,
            stage=stage,
            message=msg,
        )

    async def notify_completed(
        self,
        job_id: str,
        metrics: Dict[str, Any],
        output_model_path: Optional[str] = None,
    ) -> bool:
        """Notify JARVIS that training completed successfully."""
        return await self.notify(
            job_id=job_id,
            status="completed",
            progress=100.0,
            stage="completed",
            message="Training completed successfully",
            metrics=metrics,
            output_model_path=output_model_path,
        )

    async def notify_failed(
        self,
        job_id: str,
        error: str,
        stage: str = "failed",
    ) -> bool:
        """Notify JARVIS that training failed."""
        return await self.notify(
            job_id=job_id,
            status="failed",
            progress=0.0,
            stage=stage,
            message=f"Training failed: {error}",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            "enabled": self._enabled,
            "jarvis_url": self._jarvis_url,
            "notifications_sent": self._notifications_sent,
            "notifications_failed": self._notifications_failed,
            "last_notification": self._last_notification.isoformat() if self._last_notification else None,
        }


# Global broadcaster instance
jarvis_broadcaster = JARVISStatusBroadcaster()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Reactor-Core API starting...")
    logger.info(f"[Feedback Loop] JARVIS URL: {JARVIS_API_URL}")
    yield
    logger.info("Reactor-Core API shutting down...")
    # Close broadcaster session
    await jarvis_broadcaster.close()


app = FastAPI(
    title="Reactor-Core API",
    description="Training Pipeline API for JARVIS Continuous Learning",
    version="2.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    status = job_manager.get_status()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        pipeline_active=status["pipeline_active"],
        current_stage=job_manager.jobs.get(status["current_job_id"], {}).get("stage") if status["current_job_id"] else None,
    )


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get overall service status."""
    return StatusResponse(**job_manager.get_status())


@app.get("/api/broadcaster/status")
async def get_broadcaster_status():
    """Get JARVIS broadcaster status (Feedback Loop)."""
    return jarvis_broadcaster.get_stats()


@app.get("/api/pipeline/state", response_model=Optional[PipelineStateResponse])
async def get_pipeline_state():
    """Get current pipeline execution state."""
    if not job_manager.current_job_id:
        return None

    job = job_manager.get_job(job_manager.current_job_id)
    if not job:
        return None

    return PipelineStateResponse(
        run_id=job["job_id"],
        stage=job["stage"],
        started_at=job["started_at"] or job["created_at"],
        last_updated=datetime.now().isoformat(),
        progress=job["progress"],
    )


# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/training/trigger", response_model=TrainingJobResponse)
async def trigger_training(
    request: TrainingTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger a training run.

    This is the main endpoint called by JARVIS via the ReactorCoreClient.
    """
    # Check if a job is already running
    if job_manager.current_job_id:
        current = job_manager.get_job(job_manager.current_job_id)
        if current and current["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {job_manager.current_job_id}"
            )

    # Create the job
    job = job_manager.create_job(
        experience_count=request.experience_count,
        priority=request.priority,
        sources=request.sources,
        metadata=request.metadata,
        triggered_by=request.triggered_by,
    )

    logger.info(
        f"Training triggered: job_id={job['job_id']}, "
        f"experiences={request.experience_count}, priority={request.priority}"
    )

    # Start the job in background
    background_tasks.add_task(run_training_pipeline, job["job_id"])

    return TrainingJobResponse(**job)


@app.post("/api/training/cancel/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a running training job."""
    if job_manager.cancel_job(job_id):
        return {"cancelled": True, "job_id": job_id}
    raise HTTPException(status_code=404, detail=f"Job not found or cannot be cancelled: {job_id}")


@app.get("/api/training/job/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Get status of a training job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return TrainingJobResponse(**job)


@app.get("/api/training/history", response_model=List[TrainingJobResponse])
async def get_training_history(limit: int = 10, status: Optional[str] = None):
    """Get training job history."""
    jobs = job_manager.get_history(limit=limit, status_filter=status)
    return [TrainingJobResponse(**job) for job in jobs]


# ============================================================================
# Experience Endpoints
# ============================================================================

@app.post("/api/experiences/stream")
async def stream_experience(request: ExperienceStreamRequest):
    """Stream an experience for future training."""
    job_manager.add_experience(request.experience)
    return {"accepted": True, "count": job_manager.get_experience_count()}


@app.get("/api/experiences/count", response_model=ExperienceCountResponse)
async def get_experience_count():
    """Get count of pending experiences."""
    count = job_manager.get_experience_count()
    last = job_manager.experiences[-1]["ingested_at"] if job_manager.experiences else None
    return ExperienceCountResponse(count=count, last_ingested=last)


# ============================================================================
# Scout Endpoints
# ============================================================================

@app.post("/api/scout/topics")
async def add_learning_topic(request: TopicAddRequest):
    """Add a new learning topic for the Scout."""
    # For now, just acknowledge - actual implementation would queue the topic
    logger.info(f"Topic added: {request.topic} (category={request.category})")
    return {
        "added": True,
        "topic": request.topic,
        "category": request.category,
    }


# ============================================================================
# Background Training Task
# ============================================================================

async def run_training_pipeline(job_id: str):
    """
    Run the training pipeline in the background.

    This is a simplified version - the real implementation would:
    1. Run the NightShiftPipeline from reactor_core.orchestration
    2. Update job status as it progresses
    3. Handle errors and recovery

    Feedback Loop Integration:
    - Notifies JARVIS at key stages: 0%, 20%, 50%, 80%, 100%
    - Enables voice announcements and real-time UI updates
    """
    job_manager.start_job(job_id)
    job = job_manager.get_job(job_id)

    # Define pipeline stages with progress percentages and descriptions
    pipeline_stages = [
        ("data_prep", 0, "Preparing training data"),
        ("ingesting", 10, "Ingesting experiences from database"),
        ("formatting", 20, "Formatting data for training"),
        ("distilling", 35, "Distilling knowledge patterns"),
        ("fine_tuning", 50, "Fine-tuning model weights"),
        ("training", 65, "Training model on new data"),
        ("evaluating", 80, "Evaluating model performance"),
        ("exporting", 90, "Exporting trained model"),
    ]

    try:
        logger.info(f"[Pipeline] Starting training: job_id={job_id}")

        # Notify JARVIS: Training Started (0%)
        await jarvis_broadcaster.notify(
            job_id=job_id,
            status="running",
            progress=0.0,
            stage="data_prep",
            message="Training pipeline initiated",
        )

        for stage_name, progress, description in pipeline_stages:
            if job["status"] == "cancelled":
                logger.info(f"[Pipeline] Job cancelled: {job_id}")
                return

            # Update job state
            job["stage"] = stage_name
            job["progress"] = progress

            # Log stage start
            logger.info(f"[Pipeline] {stage_name.upper()}: {progress}% - {description}")

            # Notify JARVIS of stage change
            await jarvis_broadcaster.notify_stage_started(
                job_id=job_id,
                stage=stage_name,
                progress=progress,
                message=description,
            )

            # Simulate work (in real implementation, run actual stage)
            await asyncio.sleep(2)

        # Complete the job
        metrics = {
            "loss": 0.42,
            "eval_accuracy": 0.89,
            "examples_trained": job["experience_count"],
            "training_time_seconds": 16.0,  # 8 stages * 2 seconds
        }

        # Update job state
        job["stage"] = "completed"
        job["progress"] = 100.0

        # Determine output model path (for hot-swap integration)
        output_model_path = None
        output_dir = Path.home() / ".jarvis" / "models" / "trained"
        if output_dir.exists():
            gguf_files = list(output_dir.glob("*.gguf"))
            if gguf_files:
                output_model_path = str(max(gguf_files, key=lambda p: p.stat().st_mtime))
        metrics["output_model_path"] = output_model_path

        job_manager.complete_job(job_id, metrics)

        logger.info(f"[Pipeline] Training completed: job_id={job_id}")

        # Notify JARVIS: Training Complete (100%)
        await jarvis_broadcaster.notify_completed(
            job_id=job_id,
            metrics=metrics,
            output_model_path=output_model_path,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Pipeline] Training failed: job_id={job_id}, error={error_msg}")

        job_manager.fail_job(job_id, error_msg)

        # Notify JARVIS: Training Failed
        await jarvis_broadcaster.notify_failed(
            job_id=job_id,
            error=error_msg,
            stage=job.get("stage", "unknown"),
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Reactor-Core API Server")
    logger.info("=" * 60)
    logger.info(f"Listening: http://{API_HOST}:{API_PORT}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  GET  http://localhost:{API_PORT}/health")
    logger.info(f"  GET  http://localhost:{API_PORT}/api/status")
    logger.info(f"  POST http://localhost:{API_PORT}/api/training/trigger")
    logger.info(f"  GET  http://localhost:{API_PORT}/api/training/job/{{job_id}}")
    logger.info(f"  POST http://localhost:{API_PORT}/api/experiences/stream")
    logger.info("=" * 60)

    uvicorn.run(
        "reactor_core.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
