"""
Reactor-Core API Server v3.0

Advanced REST API server for the AGI OS ecosystem providing:
- Training pipeline triggering and management
- Real-time telemetry ingestion with WebSocket streaming
- Night Shift automated training scheduler
- Model versioning and A/B testing
- Cross-repo health aggregation
- JARVIS/Prime feedback loop integration

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Reactor-Core API Server v3.0                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
    │  │   REST API       │  │   WebSocket      │  │   Health         │   │
    │  │   Endpoints      │  │   Streaming      │  │   Dashboard      │   │
    │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
    │           │                     │                     │              │
    │           └─────────────────────┼─────────────────────┘              │
    │                                 ▼                                    │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │                    Core Services                              │   │
    │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
    │  │  │  Telemetry  │  │  Night      │  │  Model              │   │   │
    │  │  │  Collector  │  │  Scheduler  │  │  Registry           │   │   │
    │  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                 │                                    │
    │                                 ▼                                    │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │              JARVIS / Prime Integration                       │   │
    │  │   • Training notifications     • Model deployment             │   │
    │  │   • Experience ingestion       • Health reporting             │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    python -m reactor_core.api.server
    # or
    uvicorn reactor_core.api.server:app --reload --port 8003
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import advanced systems
from reactor_core.api.telemetry import (
    TelemetryCollector,
    TelemetryEvent,
    EventType,
    MetricType,
    get_telemetry,
    telemetry_context,
)
from reactor_core.api.scheduler import (
    NightShiftScheduler,
    ScheduleRule,
    ScheduleType,
    JobPriority,
    JobStatus,
    ScheduleTemplates,
    get_scheduler,
    init_scheduler,
)
from reactor_core.api.model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelStatus as RegistryModelStatus,
    ModelMetrics,
    DeploymentTarget,
    SemanticVersion,
    get_registry,
)
from reactor_core.api.health_aggregator import (
    HealthAggregator,
    HealthStatus,
    HealthCheck,
    HealthAlert,
    AlertSeverity,
    get_health_aggregator,
    init_health_aggregator,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """API Server configuration."""
    HOST = os.getenv("REACTOR_CORE_HOST", "0.0.0.0")
    PORT = int(os.getenv("REACTOR_CORE_PORT", "8003"))
    DEBUG = os.getenv("REACTOR_CORE_DEBUG", "false").lower() == "true"
    VERSION = "3.0.0"

    # Integration URLs
    JARVIS_API_URL = os.getenv("JARVIS_API_URL", "http://localhost:8000")
    PRIME_API_URL = os.getenv("PRIME_API_URL", "http://localhost:8001")

    # Features
    TELEMETRY_ENABLED = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"
    HEALTH_AGGREGATOR_ENABLED = os.getenv("HEALTH_AGGREGATOR_ENABLED", "true").lower() == "true"


# ============================================================================
# Request/Response Models
# ============================================================================

# --- Health & Status ---

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = ServerConfig.VERSION
    timestamp: str
    services: Dict[str, str] = Field(default_factory=dict)
    pipeline_active: bool = False
    current_stage: Optional[str] = None


class StatusResponse(BaseModel):
    """Overall status response."""
    healthy: bool = True
    version: str = ServerConfig.VERSION
    uptime_seconds: float = 0.0
    pipeline_active: bool = False
    current_job_id: Optional[str] = None
    pending_experiences: int = 0
    last_training: Optional[str] = None
    telemetry_running: bool = False
    scheduler_running: bool = False
    health_aggregator_running: bool = False


# --- Training ---

class TrainingTriggerRequest(BaseModel):
    """Training trigger request."""
    experience_count: int = Field(default=0, ge=0)
    priority: str = Field(default="normal", pattern="^(low|normal|high|urgent|critical)$")
    sources: List[str] = Field(default=["jarvis_experience", "scout"])
    metadata: Dict[str, Any] = Field(default_factory=dict)
    triggered_by: str = Field(default="api")


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


# --- Telemetry ---

class TelemetryEventRequest(BaseModel):
    """Telemetry event request."""
    event_type: str = "custom"
    source: str = "api"
    data: Dict[str, Any] = Field(default_factory=dict)
    labels: Dict[str, str] = Field(default_factory=dict)
    correlation_id: Optional[str] = None


class MetricRequest(BaseModel):
    """Metric ingestion request."""
    name: str
    value: float
    metric_type: str = "gauge"
    labels: Dict[str, str] = Field(default_factory=dict)
    unit: str = ""


class MetricBatchRequest(BaseModel):
    """Batch metric request."""
    metrics: List[MetricRequest]


# --- Scheduler ---

class ScheduleRuleRequest(BaseModel):
    """Schedule rule creation request."""
    name: str
    schedule_type: str = "cron"  # cron, interval, threshold
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    threshold_value: Optional[int] = None
    priority: str = "normal"
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScheduleRuleResponse(BaseModel):
    """Schedule rule response."""
    rule_id: str
    name: str
    schedule_type: str
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    priority: str
    enabled: bool
    next_scheduled: Optional[str] = None
    last_triggered: Optional[str] = None


# --- Model Registry ---

class ModelVersionRequest(BaseModel):
    """Model version creation request."""
    model_name: str
    artifact_path: Optional[str] = None
    parent_version_id: Optional[str] = None
    training_job_id: Optional[str] = None
    increment: str = "patch"  # major, minor, patch
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelVersionResponse(BaseModel):
    """Model version response."""
    version_id: str
    model_name: str
    version: str
    status: str
    artifact_path: Optional[str] = None
    created_at: str
    deployed_at: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class DeploymentRequest(BaseModel):
    """Model deployment request."""
    version_id: str
    target: str = "jarvis"  # jarvis, prime, both
    notify: bool = True


class ABTestRequest(BaseModel):
    """A/B test creation request."""
    name: str
    control_version_id: str
    treatment_version_id: str
    traffic_split: float = 0.5
    min_sample_size: int = 100


# --- Experience ---

class ExperienceStreamRequest(BaseModel):
    """Experience stream request."""
    experience: Dict[str, Any]
    timestamp: Optional[str] = None
    source: str = Field(default="jarvis_agent")


class ExperienceCountResponse(BaseModel):
    """Experience count response."""
    count: int
    last_ingested: Optional[str] = None


# ============================================================================
# Training Job Manager
# ============================================================================

class TrainingJobManager:
    """Manages training jobs and pipeline execution."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.current_job_id: Optional[str] = None
        self.experiences: List[Dict[str, Any]] = []
        self.last_training: Optional[datetime] = None
        self.start_time = datetime.now()
        self._lock = asyncio.Lock()

    async def create_job(
        self,
        experience_count: int,
        priority: str,
        sources: List[str],
        metadata: Dict[str, Any],
        triggered_by: str,
    ) -> Dict[str, Any]:
        """Create a new training job."""
        async with self._lock:
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

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    async def get_history(self, limit: int = 10, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get job history."""
        jobs = list(self.jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j["status"] == status_filter]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return jobs[:limit]

    async def start_job(self, job_id: str) -> bool:
        """Start a job."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job["status"] = "running"
                job["started_at"] = datetime.now().isoformat()
                self.current_job_id = job_id
                return True
            return False

    async def update_progress(self, job_id: str, stage: str, progress: float) -> bool:
        """Update job progress."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job["stage"] = stage
                job["progress"] = progress
                return True
            return False

    async def complete_job(self, job_id: str, metrics: Dict[str, Any]) -> bool:
        """Complete a job."""
        async with self._lock:
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

    async def fail_job(self, job_id: str, error: str) -> bool:
        """Mark a job as failed."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["stage"] = "failed"
                job["error"] = error
                job["completed_at"] = datetime.now().isoformat()
                self.current_job_id = None
                return True
            return False

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if job and job["status"] in ("queued", "running"):
                job["status"] = "cancelled"
                job["completed_at"] = datetime.now().isoformat()
                if self.current_job_id == job_id:
                    self.current_job_id = None
                return True
            return False

    async def add_experience(self, experience: Dict[str, Any]) -> int:
        """Add an experience to the pending queue."""
        async with self._lock:
            self.experiences.append({
                **experience,
                "ingested_at": datetime.now().isoformat(),
            })
            return len(self.experiences)

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


# Global instances
job_manager = TrainingJobManager()


# ============================================================================
# JARVIS Status Broadcaster
# ============================================================================

class JARVISBroadcaster:
    """Broadcasts status updates to JARVIS and Prime."""

    def __init__(self):
        self._jarvis_url = ServerConfig.JARVIS_API_URL
        self._prime_url = ServerConfig.PRIME_API_URL
        self._enabled = os.getenv("JARVIS_FEEDBACK_ENABLED", "true").lower() == "true"
        self._timeout = float(os.getenv("JARVIS_FEEDBACK_TIMEOUT", "5.0"))
        self._session = None
        self._notifications_sent = 0
        self._notifications_failed = 0

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                )
            except ImportError:
                logger.warning("[Broadcaster] aiohttp not installed")
                self._enabled = False
                return None
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def notify_training_status(
        self,
        job_id: str,
        status: str,
        progress: float,
        stage: str,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        output_model_path: Optional[str] = None,
    ) -> bool:
        """Send training status notification."""
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
            "timestamp": datetime.now().isoformat(),
        }

        if output_model_path:
            payload["output_model_path"] = output_model_path

        endpoint = f"{self._jarvis_url}/reactor-core/training/status"

        try:
            async with session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    self._notifications_sent += 1
                    return True
                else:
                    self._notifications_failed += 1
                    return False
        except Exception as e:
            logger.debug(f"[Broadcaster] Error: {e}")
            self._notifications_failed += 1
            return False

    async def notify_model_deployed(
        self,
        version_id: str,
        model_name: str,
        version: str,
        artifact_path: Optional[str] = None,
    ) -> bool:
        """Notify about model deployment."""
        if not self._enabled:
            return False

        session = await self._get_session()
        if not session:
            return False

        payload = {
            "event": "model_deployed",
            "version_id": version_id,
            "model_name": model_name,
            "version": version,
            "artifact_path": artifact_path,
            "deployed_at": datetime.now().isoformat(),
        }

        success = True
        for url in [self._jarvis_url, self._prime_url]:
            try:
                async with session.post(f"{url}/reactor-core/model/deployed", json=payload) as response:
                    if response.status != 200:
                        success = False
            except Exception:
                success = False

        return success

    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return {
            "enabled": self._enabled,
            "jarvis_url": self._jarvis_url,
            "prime_url": self._prime_url,
            "notifications_sent": self._notifications_sent,
            "notifications_failed": self._notifications_failed,
        }


# Global broadcaster
broadcaster = JARVISBroadcaster()


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class WebSocketManager:
    """Manage WebSocket connections for real-time streaming."""

    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._subscriptions: Dict[str, set] = {}  # topic -> connection_ids
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, connection_id: str, topics: Optional[List[str]] = None):
        """Connect a WebSocket client."""
        await websocket.accept()

        async with self._lock:
            self._connections[connection_id] = websocket

            for topic in (topics or ["all"]):
                if topic not in self._subscriptions:
                    self._subscriptions[topic] = set()
                self._subscriptions[topic].add(connection_id)

        logger.info(f"[WebSocket] Connected: {connection_id}, topics: {topics}")

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        async with self._lock:
            self._connections.pop(connection_id, None)

            for topic in list(self._subscriptions.keys()):
                self._subscriptions[topic].discard(connection_id)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]

        logger.info(f"[WebSocket] Disconnected: {connection_id}")

    async def broadcast(self, topic: str, data: Dict[str, Any]):
        """Broadcast message to all subscribers of a topic."""
        import json

        async with self._lock:
            connection_ids = self._subscriptions.get(topic, set()) | self._subscriptions.get("all", set())

            dead = []
            for conn_id in connection_ids:
                ws = self._connections.get(conn_id)
                if ws:
                    try:
                        await ws.send_json({"topic": topic, "data": data, "timestamp": time.time()})
                    except Exception:
                        dead.append(conn_id)

            for conn_id in dead:
                await self.disconnect(conn_id)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


ws_manager = WebSocketManager()


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("=" * 60)
    logger.info("Reactor-Core API Server v3.0 Starting...")
    logger.info("=" * 60)

    # Initialize services
    telemetry = None
    scheduler = None
    health_aggregator = None

    try:
        # Start telemetry collector
        if ServerConfig.TELEMETRY_ENABLED:
            telemetry = get_telemetry()
            await telemetry.start()
            logger.info("[✓] Telemetry collector started")

        # Start scheduler with training callback
        if ServerConfig.SCHEDULER_ENABLED:
            async def training_callback(**kwargs):
                """Callback for scheduled training."""
                job = await job_manager.create_job(
                    experience_count=job_manager.get_experience_count(),
                    priority=kwargs.get("priority", "normal"),
                    sources=["scheduled"],
                    metadata=kwargs.get("metadata", {}),
                    triggered_by="scheduler",
                )
                asyncio.create_task(run_training_pipeline(job["job_id"]))
                return job

            scheduler = await init_scheduler(training_callback)

            # Add default schedules
            scheduler.add_rule(ScheduleTemplates.nightly())
            logger.info("[✓] Night Shift scheduler started")

        # Start health aggregator
        if ServerConfig.HEALTH_AGGREGATOR_ENABLED:
            health_aggregator = await init_health_aggregator()
            logger.info("[✓] Health aggregator started")

        logger.info("")
        logger.info(f"Server running at http://{ServerConfig.HOST}:{ServerConfig.PORT}")
        logger.info("=" * 60)

        yield

    finally:
        logger.info("Shutting down services...")

        # Stop services
        if telemetry:
            await telemetry.stop()
        if scheduler:
            await scheduler.stop()
        if health_aggregator:
            await health_aggregator.stop()

        await broadcaster.close()
        logger.info("Reactor-Core API Server stopped")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Reactor-Core API",
    description="Advanced Training Pipeline API for JARVIS AGI OS",
    version=ServerConfig.VERSION,
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

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers and orchestrators."""
    status = job_manager.get_status()
    services = {}

    # Check services
    telemetry = get_telemetry()
    scheduler = get_scheduler()
    health_agg = get_health_aggregator()

    services["telemetry"] = "running" if telemetry._running else "stopped"
    services["scheduler"] = "running" if scheduler._running else "stopped"
    services["health_aggregator"] = "running" if health_agg._running else "stopped"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        services=services,
        pipeline_active=status["pipeline_active"],
        current_stage=job_manager.jobs.get(status["current_job_id"], {}).get("stage") if status["current_job_id"] else None,
    )


@app.get("/api/v1/status", response_model=StatusResponse, tags=["Health"])
async def get_status():
    """Get overall service status."""
    status = job_manager.get_status()
    telemetry = get_telemetry()
    scheduler = get_scheduler()
    health_agg = get_health_aggregator()

    return StatusResponse(
        healthy=True,
        version=ServerConfig.VERSION,
        uptime_seconds=status["uptime_seconds"],
        pipeline_active=status["pipeline_active"],
        current_job_id=status["current_job_id"],
        pending_experiences=status["pending_experiences"],
        last_training=status["last_training"],
        telemetry_running=telemetry._running,
        scheduler_running=scheduler._running,
        health_aggregator_running=health_agg._running,
    )


@app.get("/api/v1/broadcaster/status", tags=["Health"])
async def get_broadcaster_status():
    """Get JARVIS broadcaster status."""
    return broadcaster.get_stats()


# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/v1/train", response_model=TrainingJobResponse, tags=["Training"])
async def trigger_training(
    request: TrainingTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger a training run.

    This is the main endpoint called by JARVIS via the ReactorCoreClient
    or by the Night Shift scheduler.
    """
    # Check if a job is already running
    if job_manager.current_job_id:
        current = await job_manager.get_job(job_manager.current_job_id)
        if current and current["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {job_manager.current_job_id}"
            )

    # Create the job
    job = await job_manager.create_job(
        experience_count=request.experience_count or job_manager.get_experience_count(),
        priority=request.priority,
        sources=request.sources,
        metadata=request.metadata,
        triggered_by=request.triggered_by,
    )

    logger.info(
        f"Training triggered: job_id={job['job_id']}, "
        f"experiences={job['experience_count']}, priority={request.priority}"
    )

    # Start the job in background
    background_tasks.add_task(run_training_pipeline, job["job_id"])

    # Ingest telemetry event
    if ServerConfig.TELEMETRY_ENABLED:
        telemetry = get_telemetry()
        await telemetry.ingest_event(TelemetryEvent(
            event_type=EventType.CUSTOM,
            source="training",
            data={"action": "job_created", "job_id": job["job_id"], "priority": request.priority},
        ))

    return TrainingJobResponse(**job)


@app.post("/api/v1/training/cancel/{job_id}", tags=["Training"])
async def cancel_training(job_id: str):
    """Cancel a running training job."""
    if await job_manager.cancel_job(job_id):
        return {"cancelled": True, "job_id": job_id}
    raise HTTPException(status_code=404, detail=f"Job not found or cannot be cancelled: {job_id}")


@app.get("/api/v1/training/job/{job_id}", response_model=TrainingJobResponse, tags=["Training"])
async def get_training_job(job_id: str):
    """Get status of a training job."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return TrainingJobResponse(**job)


@app.get("/api/v1/training/history", response_model=List[TrainingJobResponse], tags=["Training"])
async def get_training_history(limit: int = 10, status: Optional[str] = None):
    """Get training job history."""
    jobs = await job_manager.get_history(limit=limit, status_filter=status)
    return [TrainingJobResponse(**job) for job in jobs]


@app.get("/api/v1/pipeline/state", tags=["Training"])
async def get_pipeline_state():
    """Get current pipeline execution state."""
    if not job_manager.current_job_id:
        return None

    job = await job_manager.get_job(job_manager.current_job_id)
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
# Telemetry Endpoints
# ============================================================================

@app.post("/api/v1/telemetry/ingest", tags=["Telemetry"])
async def ingest_telemetry_event(request: TelemetryEventRequest):
    """Ingest a telemetry event."""
    if not ServerConfig.TELEMETRY_ENABLED:
        raise HTTPException(status_code=503, detail="Telemetry disabled")

    telemetry = get_telemetry()

    try:
        event_type = EventType[request.event_type.upper()]
    except KeyError:
        event_type = EventType.CUSTOM

    event = TelemetryEvent(
        event_type=event_type,
        source=request.source,
        data=request.data,
        labels=request.labels,
        correlation_id=request.correlation_id,
    )

    accepted = await telemetry.ingest_event(event)
    return {"accepted": accepted, "event_id": event.event_id}


@app.post("/api/v1/telemetry/metrics", tags=["Telemetry"])
async def ingest_metric(request: MetricRequest):
    """Ingest a single metric."""
    if not ServerConfig.TELEMETRY_ENABLED:
        raise HTTPException(status_code=503, detail="Telemetry disabled")

    telemetry = get_telemetry()

    try:
        metric_type = MetricType[request.metric_type.upper()]
    except KeyError:
        metric_type = MetricType.GAUGE

    accepted = await telemetry.ingest_metric(
        name=request.name,
        value=request.value,
        metric_type=metric_type,
        labels=request.labels,
        unit=request.unit,
    )

    return {"accepted": accepted}


@app.post("/api/v1/telemetry/metrics/batch", tags=["Telemetry"])
async def ingest_metrics_batch(request: MetricBatchRequest):
    """Ingest a batch of metrics."""
    if not ServerConfig.TELEMETRY_ENABLED:
        raise HTTPException(status_code=503, detail="Telemetry disabled")

    telemetry = get_telemetry()
    accepted = 0

    for metric in request.metrics:
        try:
            metric_type = MetricType[metric.metric_type.upper()]
        except KeyError:
            metric_type = MetricType.GAUGE

        if await telemetry.ingest_metric(
            name=metric.name,
            value=metric.value,
            metric_type=metric_type,
            labels=metric.labels,
            unit=metric.unit,
        ):
            accepted += 1

    return {"accepted": accepted, "total": len(request.metrics)}


@app.get("/api/v1/telemetry/metrics", tags=["Telemetry"])
async def get_aggregated_metrics(window_size: int = 60, metric_names: Optional[str] = None):
    """Get aggregated metrics."""
    if not ServerConfig.TELEMETRY_ENABLED:
        raise HTTPException(status_code=503, detail="Telemetry disabled")

    telemetry = get_telemetry()
    names = metric_names.split(",") if metric_names else None
    metrics = await telemetry.get_aggregated_metrics(window_size=window_size, metric_names=names)
    return {"metrics": metrics, "window_size": window_size}


@app.get("/api/v1/telemetry/alerts", tags=["Telemetry"])
async def get_telemetry_alerts(since: Optional[float] = None, limit: int = 100):
    """Get anomaly alerts."""
    if not ServerConfig.TELEMETRY_ENABLED:
        raise HTTPException(status_code=503, detail="Telemetry disabled")

    telemetry = get_telemetry()
    alerts = await telemetry.get_alerts(since=since, limit=limit)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/api/v1/telemetry/stats", tags=["Telemetry"])
async def get_telemetry_stats():
    """Get telemetry system statistics."""
    if not ServerConfig.TELEMETRY_ENABLED:
        return {"enabled": False}

    telemetry = get_telemetry()
    return telemetry.get_stats()


# ============================================================================
# Scheduler Endpoints
# ============================================================================

@app.get("/api/v1/scheduler/status", tags=["Scheduler"])
async def get_scheduler_status():
    """Get scheduler status."""
    if not ServerConfig.SCHEDULER_ENABLED:
        return {"enabled": False}

    scheduler = get_scheduler()
    return await scheduler.get_status()


@app.post("/api/v1/scheduler/rules", response_model=ScheduleRuleResponse, tags=["Scheduler"])
async def create_schedule_rule(request: ScheduleRuleRequest):
    """Create a new schedule rule."""
    if not ServerConfig.SCHEDULER_ENABLED:
        raise HTTPException(status_code=503, detail="Scheduler disabled")

    scheduler = get_scheduler()

    try:
        schedule_type = ScheduleType[request.schedule_type.upper()]
    except KeyError:
        schedule_type = ScheduleType.CRON

    try:
        priority = JobPriority[request.priority.upper()]
    except KeyError:
        priority = JobPriority.NORMAL

    rule = ScheduleRule(
        name=request.name,
        schedule_type=schedule_type,
        cron_expression=request.cron_expression,
        interval_seconds=request.interval_seconds,
        threshold_value=request.threshold_value,
        priority=priority,
        enabled=request.enabled,
        metadata=request.metadata,
    )

    scheduler.add_rule(rule)

    return ScheduleRuleResponse(
        rule_id=rule.rule_id,
        name=rule.name,
        schedule_type=rule.schedule_type.name,
        cron_expression=rule.cron_expression,
        interval_seconds=rule.interval_seconds,
        priority=rule.priority.name,
        enabled=rule.enabled,
        next_scheduled=datetime.fromtimestamp(rule.next_scheduled).isoformat() if rule.next_scheduled else None,
        last_triggered=datetime.fromtimestamp(rule.last_triggered).isoformat() if rule.last_triggered else None,
    )


@app.get("/api/v1/scheduler/rules", tags=["Scheduler"])
async def list_schedule_rules():
    """List all schedule rules."""
    if not ServerConfig.SCHEDULER_ENABLED:
        return {"enabled": False, "rules": []}

    scheduler = get_scheduler()
    rules = scheduler.get_rules()

    return {
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "schedule_type": r.schedule_type.name,
                "cron_expression": r.cron_expression,
                "priority": r.priority.name,
                "enabled": r.enabled,
                "next_scheduled": datetime.fromtimestamp(r.next_scheduled).isoformat() if r.next_scheduled else None,
            }
            for r in rules
        ]
    }


@app.delete("/api/v1/scheduler/rules/{rule_id}", tags=["Scheduler"])
async def delete_schedule_rule(rule_id: str):
    """Delete a schedule rule."""
    if not ServerConfig.SCHEDULER_ENABLED:
        raise HTTPException(status_code=503, detail="Scheduler disabled")

    scheduler = get_scheduler()
    if scheduler.remove_rule(rule_id):
        return {"deleted": True, "rule_id": rule_id}
    raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")


@app.post("/api/v1/scheduler/trigger", tags=["Scheduler"])
async def trigger_scheduled_training(priority: str = "normal"):
    """Manually trigger scheduled training now."""
    if not ServerConfig.SCHEDULER_ENABLED:
        raise HTTPException(status_code=503, detail="Scheduler disabled")

    scheduler = get_scheduler()

    try:
        job_priority = JobPriority[priority.upper()]
    except KeyError:
        job_priority = JobPriority.NORMAL

    job = await scheduler.trigger_now(priority=job_priority)
    return {"triggered": True, "job_id": job.job_id, "status": job.status.value}


@app.get("/api/v1/scheduler/jobs", tags=["Scheduler"])
async def list_scheduled_jobs(status: Optional[str] = None, limit: int = 50):
    """List scheduled jobs."""
    if not ServerConfig.SCHEDULER_ENABLED:
        return {"enabled": False, "jobs": []}

    scheduler = get_scheduler()

    job_status = None
    if status:
        try:
            job_status = JobStatus[status.upper()]
        except KeyError:
            pass

    jobs = scheduler.get_jobs(status=job_status, limit=limit)
    return {
        "jobs": [j.to_dict() for j in jobs]
    }


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@app.post("/api/v1/models/versions", response_model=ModelVersionResponse, tags=["Model Registry"])
async def create_model_version(request: ModelVersionRequest):
    """Create a new model version."""
    registry = get_registry()

    version = await registry.versions.create_version(
        model_name=request.model_name,
        artifact_path=request.artifact_path,
        parent_version_id=request.parent_version_id,
        training_job_id=request.training_job_id,
        increment=request.increment,
        tags=request.tags,
        metadata=request.metadata,
    )

    return ModelVersionResponse(
        version_id=version.version_id,
        model_name=version.model_name,
        version=str(version.version),
        status=version.status.value,
        artifact_path=version.artifact_path,
        created_at=datetime.fromtimestamp(version.created_at).isoformat(),
        deployed_at=datetime.fromtimestamp(version.deployed_at).isoformat() if version.deployed_at else None,
        metrics=version.metrics.to_dict(),
        tags=version.tags,
    )


@app.get("/api/v1/models/versions", tags=["Model Registry"])
async def list_model_versions(model_name: Optional[str] = None, status: Optional[str] = None, limit: int = 50):
    """List model versions."""
    registry = get_registry()

    model_status = None
    if status:
        try:
            model_status = RegistryModelStatus[status.upper()]
        except KeyError:
            pass

    versions = await registry.versions.list_versions(
        model_name=model_name,
        status=model_status,
        limit=limit,
    )

    return {
        "versions": [v.to_dict() for v in versions]
    }


@app.get("/api/v1/models/versions/{version_id}", response_model=ModelVersionResponse, tags=["Model Registry"])
async def get_model_version(version_id: str):
    """Get a specific model version."""
    registry = get_registry()
    version = await registry.versions.get_version(version_id)

    if not version:
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")

    return ModelVersionResponse(
        version_id=version.version_id,
        model_name=version.model_name,
        version=str(version.version),
        status=version.status.value,
        artifact_path=version.artifact_path,
        created_at=datetime.fromtimestamp(version.created_at).isoformat(),
        deployed_at=datetime.fromtimestamp(version.deployed_at).isoformat() if version.deployed_at else None,
        metrics=version.metrics.to_dict(),
        tags=version.tags,
    )


@app.post("/api/v1/models/deploy", tags=["Model Registry"])
async def deploy_model(request: DeploymentRequest):
    """Deploy a model version."""
    registry = get_registry()

    try:
        target = DeploymentTarget[request.target.upper()]
    except KeyError:
        target = DeploymentTarget.JARVIS

    deployment = await registry.deployments.deploy(
        version_id=request.version_id,
        target=target,
        notify=request.notify,
    )

    # Broadcast to JARVIS/Prime
    if request.notify:
        version = await registry.versions.get_version(request.version_id)
        if version:
            await broadcaster.notify_model_deployed(
                version_id=version.version_id,
                model_name=version.model_name,
                version=str(version.version),
                artifact_path=version.artifact_path,
            )

    return {
        "deployment_id": deployment.deployment_id,
        "version_id": deployment.version_id,
        "target": deployment.target.value,
        "status": deployment.status,
    }


@app.post("/api/v1/models/rollback", tags=["Model Registry"])
async def rollback_model(target: str = "jarvis", to_version_id: Optional[str] = None, reason: str = "manual"):
    """Rollback to a previous model version."""
    registry = get_registry()

    try:
        deployment_target = DeploymentTarget[target.upper()]
    except KeyError:
        deployment_target = DeploymentTarget.JARVIS

    deployment = await registry.deployments.rollback(
        target=deployment_target,
        to_version_id=to_version_id,
        reason=reason,
    )

    return {
        "deployment_id": deployment.deployment_id,
        "version_id": deployment.version_id,
        "status": deployment.status,
        "rollback_of": deployment.rollback_of,
    }


@app.get("/api/v1/models/registry/status", tags=["Model Registry"])
async def get_registry_status():
    """Get model registry status."""
    registry = get_registry()
    return await registry.get_status()


# --- A/B Testing ---

@app.post("/api/v1/models/ab-tests", tags=["A/B Testing"])
async def create_ab_test(request: ABTestRequest):
    """Create a new A/B test."""
    registry = get_registry()

    test = await registry.ab_tests.create_test(
        name=request.name,
        control_version_id=request.control_version_id,
        treatment_version_id=request.treatment_version_id,
        traffic_split=request.traffic_split,
        min_sample_size=request.min_sample_size,
    )

    return {
        "test_id": test.test_id,
        "name": test.name,
        "control_version_id": test.control_version_id,
        "treatment_version_id": test.treatment_version_id,
        "traffic_split": test.traffic_split,
        "is_active": test.is_active,
    }


@app.get("/api/v1/models/ab-tests", tags=["A/B Testing"])
async def list_ab_tests():
    """List A/B tests."""
    registry = get_registry()
    tests = await registry.ab_tests.get_active_tests()
    return {"tests": [asdict(t) if hasattr(t, "__dataclass_fields__") else vars(t) for t in tests]}


@app.get("/api/v1/models/ab-tests/{test_id}/route", tags=["A/B Testing"])
async def route_ab_test_request(test_id: str):
    """Route a request to control or treatment version."""
    registry = get_registry()
    version_id, group = await registry.ab_tests.route_request(test_id)
    return {"version_id": version_id, "group": group}


@app.post("/api/v1/models/ab-tests/{test_id}/sample", tags=["A/B Testing"])
async def record_ab_test_sample(
    test_id: str,
    version_id: str,
    metrics: Dict[str, float],
):
    """Record a sample observation for an A/B test."""
    registry = get_registry()
    await registry.ab_tests.record_sample(test_id, version_id, metrics)
    return {"recorded": True}


@app.get("/api/v1/models/ab-tests/{test_id}/analyze", tags=["A/B Testing"])
async def analyze_ab_test(test_id: str):
    """Analyze A/B test results."""
    registry = get_registry()
    result = await registry.ab_tests.analyze_test(test_id)
    return {
        "test_id": result.test_id,
        "control_samples": result.control_samples,
        "treatment_samples": result.treatment_samples,
        "improvement": result.improvement,
        "is_significant": result.is_significant,
        "recommended_winner": result.recommended_winner,
        "confidence_level": result.confidence_level,
    }


@app.post("/api/v1/models/ab-tests/{test_id}/conclude", tags=["A/B Testing"])
async def conclude_ab_test(test_id: str, winner_version_id: Optional[str] = None):
    """Conclude an A/B test."""
    registry = get_registry()
    test = await registry.ab_tests.conclude_test(test_id, winner_version_id)
    return {
        "test_id": test.test_id,
        "is_active": test.is_active,
        "winner": test.winner,
    }


# ============================================================================
# Health Aggregator Endpoints
# ============================================================================

@app.get("/api/v1/health/dashboard", tags=["Health Dashboard"])
async def get_health_dashboard():
    """Get unified health dashboard data."""
    if not ServerConfig.HEALTH_AGGREGATOR_ENABLED:
        return {"enabled": False}

    aggregator = get_health_aggregator()
    dashboard = await aggregator.get_dashboard()
    return dashboard.to_dict()


@app.get("/api/v1/health/components", tags=["Health Dashboard"])
async def list_health_components():
    """List all monitored components."""
    if not ServerConfig.HEALTH_AGGREGATOR_ENABLED:
        return {"enabled": False, "components": []}

    aggregator = get_health_aggregator()
    dashboard = await aggregator.get_dashboard()
    return {
        "components": [h.to_dict() for h in dashboard.components.values()]
    }


@app.get("/api/v1/health/components/{component}", tags=["Health Dashboard"])
async def get_component_health(component: str):
    """Get health for a specific component."""
    if not ServerConfig.HEALTH_AGGREGATOR_ENABLED:
        raise HTTPException(status_code=503, detail="Health aggregator disabled")

    aggregator = get_health_aggregator()
    health = await aggregator.get_component_health(component)

    if not health:
        raise HTTPException(status_code=404, detail=f"Component not found: {component}")

    return health.to_dict()


@app.post("/api/v1/health/check", tags=["Health Dashboard"])
async def trigger_health_check(component: Optional[str] = None):
    """Trigger immediate health check."""
    if not ServerConfig.HEALTH_AGGREGATOR_ENABLED:
        raise HTTPException(status_code=503, detail="Health aggregator disabled")

    aggregator = get_health_aggregator()
    results = await aggregator.check_now(component)
    return {
        "checked": list(results.keys()),
        "results": {k: v.to_dict() for k, v in results.items()},
    }


@app.get("/api/v1/health/sla/{component}", tags=["Health Dashboard"])
async def get_sla_report(component: str, period_days: int = 30):
    """Get SLA report for a component."""
    if not ServerConfig.HEALTH_AGGREGATOR_ENABLED:
        raise HTTPException(status_code=503, detail="Health aggregator disabled")

    aggregator = get_health_aggregator()
    report = await aggregator.get_sla_report(component, period_seconds=period_days * 86400)

    return {
        "component": report.component,
        "period_days": period_days,
        "uptime_percent": round(report.uptime_percent, 3),
        "target_uptime": report.target_uptime,
        "is_compliant": report.is_compliant,
        "total_downtime_seconds": round(report.total_downtime_seconds, 1),
        "incident_count": len(report.incidents),
        "avg_latency_ms": round(report.avg_latency_ms, 2),
        "p99_latency_ms": round(report.p99_latency_ms, 2),
    }


@app.get("/api/v1/health/alerts", tags=["Health Dashboard"])
async def get_health_alerts(component: Optional[str] = None, limit: int = 100):
    """Get health alerts."""
    if not ServerConfig.HEALTH_AGGREGATOR_ENABLED:
        return {"enabled": False, "alerts": []}

    aggregator = get_health_aggregator()
    dashboard = await aggregator.get_dashboard()
    alerts = dashboard.active_alerts

    if component:
        alerts = [a for a in alerts if a.component == component]

    return {
        "alerts": [a.to_dict() for a in alerts[:limit]],
        "count": len(alerts),
    }


# ============================================================================
# Model Hot-Reload Notification Endpoints
# ============================================================================

class ModelReloadRequest(BaseModel):
    """Model reload notification request."""
    model_id: str
    model_path: Optional[str] = None
    backend: str = "auto"
    info: Dict[str, Any] = Field(default_factory=dict)


class ModelReloadResponse(BaseModel):
    """Model reload notification response."""
    acknowledged: bool
    model_id: str
    action: str
    timestamp: str


@app.post("/api/v1/models/reload", response_model=ModelReloadResponse, tags=["Model Hot-Reload"])
async def notify_model_reload(request: ModelReloadRequest):
    """
    Receive model hot-reload notification.

    This endpoint is called by the ModelServer when a model is hot-reloaded.
    It can be used by Prime to update its model cache.
    """
    logger.info(f"[Hot-Reload] Model reload notification: {request.model_id}")

    # Record telemetry
    if ServerConfig.TELEMETRY_ENABLED:
        telemetry = get_telemetry()
        await telemetry.ingest_event(TelemetryEvent(
            event_type=EventType.CUSTOM,
            source="model_server",
            data={
                "action": "model_hot_reload",
                "model_id": request.model_id,
                "backend": request.backend,
                "info": request.info,
            },
        ))

    # Broadcast via WebSocket
    await ws_manager.broadcast("model_updates", {
        "event": "model_reloaded",
        "model_id": request.model_id,
        "backend": request.backend,
        "info": request.info,
    })

    # Notify Prime if configured
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            await session.post(
                f"{ServerConfig.PRIME_API_URL}/api/v1/models/reload",
                json={
                    "model_id": request.model_id,
                    "model_path": request.model_path,
                    "backend": request.backend,
                    "info": request.info,
                },
            )
            logger.info(f"[Hot-Reload] Notified Prime about model {request.model_id}")
    except Exception as e:
        logger.debug(f"[Hot-Reload] Prime notification failed (non-critical): {e}")

    return ModelReloadResponse(
        acknowledged=True,
        model_id=request.model_id,
        action="reload_notification_received",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/v1/models/prime/notify-reload", tags=["Model Hot-Reload"])
async def prime_notify_reload(
    model_id: str,
    model_path: Optional[str] = None,
    backend: str = "auto",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Explicitly notify Prime about a model reload.

    This endpoint is for manual triggering of Prime model cache refresh.
    """
    try:
        import aiohttp

        payload = {
            "model_id": model_id,
            "model_path": model_path,
            "backend": backend,
            "metadata": metadata or {},
            "source": "reactor_core",
            "timestamp": datetime.now().isoformat(),
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(
                f"{ServerConfig.PRIME_API_URL}/api/v1/models/reload",
                json=payload,
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "notified": True,
                        "prime_response": result,
                        "model_id": model_id,
                    }
                else:
                    return {
                        "notified": False,
                        "error": f"Prime returned status {response.status}",
                        "model_id": model_id,
                    }

    except Exception as e:
        logger.warning(f"[Hot-Reload] Failed to notify Prime: {e}")
        return {
            "notified": False,
            "error": str(e),
            "model_id": model_id,
        }


@app.get("/api/v1/models/loaded", tags=["Model Hot-Reload"])
async def get_loaded_models():
    """
    Get list of currently loaded models from the model server.

    Returns information about models available for inference.
    """
    try:
        from reactor_core.serving.model_server import get_model_server

        server = get_model_server()
        if server and hasattr(server, 'get_loaded_models'):
            models = await server.get_loaded_models()
            return {
                "models": models,
                "count": len(models),
            }
        else:
            return {
                "models": [],
                "count": 0,
                "note": "Model server not initialized or not available",
            }
    except ImportError:
        return {
            "models": [],
            "count": 0,
            "error": "Model server module not available",
        }
    except Exception as e:
        return {
            "models": [],
            "count": 0,
            "error": str(e),
        }


# ============================================================================
# Experience Endpoints
# ============================================================================

@app.post("/api/v1/experiences/stream", tags=["Experiences"])
async def stream_experience(request: ExperienceStreamRequest):
    """Stream an experience for future training.

    v242.0: Accepts canonical ExperienceEvent format with proper field names.
    Normalizes legacy field names (response→assistant_output) via from_raw_dict().
    """
    experience = request.experience

    # v242.0: Normalize to canonical field names using the shared schema
    try:
        import sys as _sys
        _jarvis_home = str(Path.home() / ".jarvis")
        if _jarvis_home not in _sys.path:
            _sys.path.insert(0, _jarvis_home)
        from schemas.experience_schema import from_raw_dict
        canonical = from_raw_dict(experience)
        experience = canonical.to_reactor_core_format()
    except ImportError:
        # Fallback: manual normalization of critical field names
        if "response" in experience and "assistant_output" not in experience:
            experience["assistant_output"] = experience["response"]
        if "output" in experience and "assistant_output" not in experience:
            experience["assistant_output"] = experience["output"]

    count = await job_manager.add_experience(experience)

    # Check if experience threshold triggers training
    if ServerConfig.SCHEDULER_ENABLED:
        scheduler = get_scheduler()
        job = await scheduler.add_experiences(1)
        if job:
            return {
                "accepted": True,
                "count": count,
                "training_triggered": True,
                "job_id": job.job_id,
            }

    # Ingest telemetry
    if ServerConfig.TELEMETRY_ENABLED:
        telemetry = get_telemetry()
        await telemetry.ingest_metric(
            name="experiences_ingested",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={"source": request.source},
        )

    return {"accepted": True, "count": count}


@app.post("/api/v1/experiences/batch", tags=["Experiences"])
async def batch_experiences(request: Request):
    """
    v242.1: Batch experience ingestion endpoint.

    Accepts an array of experiences in a single HTTP request instead of
    requiring one request per experience. Dramatically reduces overhead
    for bulk uploads (e.g., ReactorCoreBridge sync of 500 conversations).

    Body: {"experiences": [{...}, {...}, ...], "source": "jarvis_body"}
    """
    body = await request.json()
    experiences = body.get("experiences", [])
    source = body.get("source", "unknown")

    if not experiences:
        return {"accepted": 0, "errors": 0}

    accepted = 0
    errors = 0

    # Normalize via canonical schema
    _from_raw_dict = None
    try:
        import sys as _sys
        _jarvis_home = str(Path.home() / ".jarvis")
        if _jarvis_home not in _sys.path:
            _sys.path.insert(0, _jarvis_home)
        from schemas.experience_schema import from_raw_dict as _from_raw_dict_fn
        _from_raw_dict = _from_raw_dict_fn
    except ImportError:
        pass

    for exp in experiences:
        try:
            if _from_raw_dict:
                canonical = _from_raw_dict(exp)
                exp = canonical.to_reactor_core_format()
            else:
                if "response" in exp and "assistant_output" not in exp:
                    exp["assistant_output"] = exp["response"]

            await job_manager.add_experience(exp)
            accepted += 1
        except Exception as e:
            errors += 1
            logger.debug(f"[Batch] Error ingesting experience: {e}")

    # Trigger scheduler check based on total accepted
    if accepted > 0 and ServerConfig.SCHEDULER_ENABLED:
        scheduler = get_scheduler()
        await scheduler.add_experiences(accepted)

    # Telemetry
    if accepted > 0 and ServerConfig.TELEMETRY_ENABLED:
        telemetry = get_telemetry()
        await telemetry.ingest_metric(
            name="experiences_ingested",
            value=accepted,
            metric_type=MetricType.COUNTER,
            labels={"source": source, "batch": "true"},
        )

    return {"accepted": accepted, "errors": errors, "total": len(experiences)}


@app.get("/api/v1/experiences/count", response_model=ExperienceCountResponse, tags=["Experiences"])
async def get_experience_count():
    """Get count of pending experiences."""
    count = job_manager.get_experience_count()
    last = job_manager.experiences[-1]["ingested_at"] if job_manager.experiences else None
    return ExperienceCountResponse(count=count, last_ingested=last)


# ============================================================================
# Corrections Endpoints (JARVIS-Prime → Reactor-Core)
# ============================================================================

class CorrectionData(BaseModel):
    """A single correction record."""
    correction_id: str
    original_prompt: str
    original_response: str
    corrected_response: str
    correction_type: str
    timestamp: float
    context: Optional[List[Dict[str, Any]]] = None
    user_feedback: Optional[str] = None
    quality_score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CorrectionsStreamRequest(BaseModel):
    """Request to stream corrections for training."""
    corrections: List[CorrectionData]
    source: str = "jarvis_prime"
    timestamp: float = Field(default_factory=time.time)


@app.post("/api/v1/corrections/stream", tags=["Corrections"])
async def stream_corrections(request: CorrectionsStreamRequest):
    """
    Stream corrections from JARVIS-Prime for training.

    Corrections are high-value training data because they represent
    direct user feedback on model errors.
    """
    corrections_added = 0

    for correction in request.corrections:
        # Convert correction to experience format for training
        experience = {
            "user_input": correction.original_prompt,
            "jarvis_response": correction.corrected_response,  # Use corrected as target
            "original_response": correction.original_response,
            "correction_type": correction.correction_type,
            "timestamp": datetime.fromtimestamp(correction.timestamp).isoformat(),
            "ingested_at": datetime.now().isoformat(),
            "source": request.source,
            "quality_score": correction.quality_score * 1.5,  # Boost corrections
            "is_correction": True,
            "user_feedback": correction.user_feedback,
            "metadata": correction.metadata,
        }

        await job_manager.add_experience(experience)
        corrections_added += 1

    # Ingest telemetry
    if ServerConfig.TELEMETRY_ENABLED:
        telemetry = get_telemetry()
        await telemetry.ingest_metric(
            name="corrections_ingested",
            value=corrections_added,
            metric_type=MetricType.COUNTER,
            labels={"source": request.source},
        )

    logger.info(f"[Corrections] Received {corrections_added} corrections from {request.source}")

    # Check if corrections trigger training (lower threshold for corrections)
    if ServerConfig.SCHEDULER_ENABLED and corrections_added >= 3:
        scheduler = get_scheduler()
        job = await scheduler.add_experiences(corrections_added)
        if job:
            return {
                "accepted": True,
                "count": corrections_added,
                "training_triggered": True,
                "job_id": job.job_id,
                "message": "Corrections received - training triggered due to high-value data",
            }

    return {
        "accepted": True,
        "count": corrections_added,
        "training_triggered": False,
        "message": f"Received {corrections_added} corrections for training",
    }


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket, topics: Optional[str] = Query(None)):
    """WebSocket endpoint for real-time telemetry streaming."""
    connection_id = str(uuid.uuid4())[:8]
    topic_list = topics.split(",") if topics else ["all"]

    await ws_manager.connect(websocket, connection_id, topic_list)

    try:
        # Register with telemetry collector
        if ServerConfig.TELEMETRY_ENABLED:
            telemetry = get_telemetry()
            await telemetry.register_websocket(connection_id, websocket, topic_list)

        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Handle client messages if needed

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(connection_id)
        if ServerConfig.TELEMETRY_ENABLED:
            telemetry = get_telemetry()
            await telemetry.unregister_websocket(connection_id)


@app.websocket("/ws/training/{job_id}")
async def websocket_training_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training progress."""
    connection_id = f"training-{job_id}-{str(uuid.uuid4())[:4]}"

    await ws_manager.connect(websocket, connection_id, [f"training:{job_id}"])

    try:
        while True:
            # Send current status periodically
            job = await job_manager.get_job(job_id)
            if job:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job["status"],
                    "stage": job["stage"],
                    "progress": job["progress"],
                    "timestamp": time.time(),
                })

                if job["status"] in ("completed", "failed", "cancelled"):
                    break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(connection_id)


# ============================================================================
# Server-Sent Events (SSE) Endpoint
# ============================================================================

@app.get("/api/v1/stream/events", tags=["Streaming"])
async def stream_events(topics: Optional[str] = Query(None)):
    """Server-Sent Events endpoint for real-time updates."""
    topic_list = topics.split(",") if topics else ["all"]

    async def event_generator():
        """Generate SSE events."""
        while True:
            # Get latest events from telemetry
            if ServerConfig.TELEMETRY_ENABLED:
                telemetry = get_telemetry()
                stats = telemetry.get_stats()
                yield f"data: {json.dumps({'type': 'stats', 'data': stats})}\n\n"

            # Get health status
            if ServerConfig.HEALTH_AGGREGATOR_ENABLED:
                aggregator = get_health_aggregator()
                dashboard = await aggregator.get_dashboard()
                yield f"data: {json.dumps({'type': 'health', 'status': dashboard.overall_status.value})}\n\n"

            await asyncio.sleep(5)

    import json
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================================
# Background Training Task
# ============================================================================

async def run_training_pipeline(job_id: str):
    """
    Run the training pipeline in the background.

    Uses the real unified training pipeline when available,
    with fallback to staged simulation for testing.
    """
    await job_manager.start_job(job_id)
    job = await job_manager.get_job(job_id)
    start_time = time.time()

    # Progress callback for real-time updates
    async def progress_callback(stage: str, progress: float, message: str) -> None:
        """Callback to update job progress and notify clients."""
        await job_manager.update_progress(job_id, stage, progress)

        await broadcaster.notify_training_status(
            job_id=job_id,
            status="running",
            progress=progress,
            stage=stage,
            message=message,
        )

        await ws_manager.broadcast(f"training:{job_id}", {
            "status": "running",
            "stage": stage,
            "progress": progress,
            "message": message,
        })

        if ServerConfig.TELEMETRY_ENABLED:
            telemetry = get_telemetry()
            await telemetry.ingest_metric(
                name="training_progress",
                value=progress,
                metric_type=MetricType.GAUGE,
                labels={"job_id": job_id, "stage": stage},
            )

    # v242.1: Mutual exclusion — prevent concurrent training runs from
    # multiple schedulers (NightShiftScheduler + PipelineScheduler) or
    # manual triggers overlapping with scheduled runs.
    lock_dir = Path.home() / ".reactor_core" / "scheduler" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "training_pipeline.lock"

    try:
        import fcntl
        _lock_fd = open(lock_file, "w")
        try:
            fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            logger.warning(f"[Pipeline] Training already in progress (lock held), skipping job {job_id}")
            await job_manager.fail_job(job_id, "Training already in progress — concurrent run blocked")
            _lock_fd.close()
            return
        _lock_fd.write(f"{os.getpid()}:{job_id}\n")
        _lock_fd.flush()
    except ImportError:
        _lock_fd = None  # fcntl not available (Windows) — skip locking

    try:
        logger.info(f"[Pipeline] Starting: job_id={job_id}")

        # Notify JARVIS
        await broadcaster.notify_training_status(
            job_id=job_id,
            status="running",
            progress=0.0,
            stage="initializing",
            message="Training pipeline initiated",
        )

        await ws_manager.broadcast(f"training:{job_id}", {
            "status": "running",
            "stage": "initializing",
            "progress": 0,
        })

        # Try to use the real unified training pipeline
        real_training_available = False
        output_model_path = None
        metrics = {}

        try:
            # v242.0: Fixed class name (UnifiedTrainingPipeline, not UnifiedTrainer)
            # Fixed method names (run_training_cycle, not train_async)
            # Fixed result field names to match PipelineResult dataclass
            from reactor_core.training.unified_pipeline import (
                get_unified_trainer_async, UnifiedTrainingPipeline, PipelineResult,
            )

            trainer = await get_unified_trainer_async()
            real_training_available = True
            logger.info(f"[Pipeline] Using real unified training pipeline")

            # Report initial progress
            await progress_callback("data_prep", 5, "Preparing training data")

            # Get collected experiences from job manager
            experiences = job_manager.experiences.copy()
            if experiences:
                await trainer.add_experiences(experiences)
                job_manager.experiences.clear()  # Clear after passing to trainer

            await progress_callback("training", 20, f"Training on {len(experiences)} experiences")

            # v242.0: Use run_training_cycle() — the actual method that exists
            training_result = await trainer.run_training_cycle()

            if training_result.success:
                metrics = {
                    "loss": training_result.final_loss,
                    "examples_trained": training_result.samples_used,
                    "training_time_seconds": training_result.training_time_seconds,
                    "training_steps": training_result.training_steps,
                    "total_time_seconds": training_result.total_time_seconds,
                }
                # Use gguf_path first (preferred), fall back to model_path
                output_model_path = str(training_result.gguf_path or training_result.model_path or "")
                output_model_path = output_model_path if output_model_path else None
            else:
                raise RuntimeError(training_result.error_message or "Training failed")

        except ImportError as e:
            logger.error(f"[Pipeline] Unified trainer not available: {e}")
            # v242.0: Fail loudly instead of silently falling back to fake simulation
            await job_manager.fail_job(job_id, str(e))
            return
        except Exception as e:
            logger.error(f"[Pipeline] Training pipeline error: {e}")
            # v242.0: On real failure, don't fake success — report the actual error
            if not real_training_available:
                await job_manager.fail_job(job_id, str(e))
                return

        # v242.0: If real training pipeline wasn't available, the job was already
        # failed above with an explicit error. No fake simulation fallback.
        # The old code returned fake metrics pretending training succeeded —
        # this masked the real issue (missing dependencies or broken imports).
        if not real_training_available:
            logger.error(
                f"[Pipeline] Job {job_id} has no real training pipeline. "
                f"Ensure reactor_core.training.unified_pipeline is importable."
            )
            return

        metrics["output_model_path"] = output_model_path
        await job_manager.complete_job(job_id, metrics)

        # Create model version in registry
        try:
            registry = get_registry()
            version = await registry.versions.create_version(
                model_name="jarvis-trained",
                artifact_path=output_model_path,
                training_job_id=job_id,
                increment="patch",
                metrics=ModelMetrics(
                    loss=metrics.get("loss", 0.0),
                    accuracy=metrics.get("eval_accuracy", 0.0),
                ),
            )
            logger.info(f"[Pipeline] Model version created: {version.version}")
        except Exception as e:
            logger.warning(f"[Pipeline] Could not create model version: {e}")

        logger.info(f"[Pipeline] Completed: job_id={job_id}")

        # Notify completion
        await broadcaster.notify_training_status(
            job_id=job_id,
            status="completed",
            progress=100.0,
            stage="completed",
            message="Training completed successfully",
            metrics=metrics,
            output_model_path=output_model_path,
        )

        await ws_manager.broadcast(f"training:{job_id}", {
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "metrics": metrics,
        })

        # Record telemetry
        if ServerConfig.TELEMETRY_ENABLED:
            telemetry = get_telemetry()
            await telemetry.ingest_event(TelemetryEvent(
                event_type=EventType.CUSTOM,
                source="training",
                data={
                    "action": "job_completed",
                    "job_id": job_id,
                    "metrics": metrics,
                    "training_time_seconds": time.time() - start_time,
                },
            ))

    except asyncio.CancelledError:
        logger.info(f"[Pipeline] Cancelled: job_id={job_id}")
        await job_manager.cancel_job(job_id)

        await broadcaster.notify_training_status(
            job_id=job_id,
            status="cancelled",
            progress=0.0,
            stage="cancelled",
            message="Training was cancelled",
        )

        await ws_manager.broadcast(f"training:{job_id}", {
            "status": "cancelled",
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Pipeline] Failed: job_id={job_id}, error={error_msg}")

        await job_manager.fail_job(job_id, error_msg)

        await broadcaster.notify_training_status(
            job_id=job_id,
            status="failed",
            progress=0.0,
            stage="failed",
            message=f"Training failed: {error_msg}",
        )

        await ws_manager.broadcast(f"training:{job_id}", {
            "status": "failed",
            "error": error_msg,
        })

        # Record failure telemetry
        if ServerConfig.TELEMETRY_ENABLED:
            telemetry = get_telemetry()
            await telemetry.ingest_event(TelemetryEvent(
                event_type=EventType.CUSTOM,
                source="training",
                data={
                    "action": "job_failed",
                    "job_id": job_id,
                    "error": error_msg,
                },
            ))

    finally:
        # v242.1: Release training pipeline lock so other schedulers can proceed
        if _lock_fd:
            try:
                import fcntl
                fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                _lock_fd.close()
                _lock_fd = None


# ============================================================================
# Background Server Startup
# ============================================================================

_server_task: Optional[asyncio.Task] = None
_server_shutdown_event: Optional[asyncio.Event] = None


async def start_server_background(
    host: str = None,
    port: int = None,
    log_level: str = "info",
) -> bool:
    """
    Start the API server in the background.

    Called by run_supervisor.py to start the API server as part
    of the unified supervisor startup process.

    Args:
        host: Server host (defaults to ServerConfig.HOST)
        port: Server port (defaults to ServerConfig.PORT)
        log_level: Logging level

    Returns:
        True if server started successfully
    """
    global _server_task, _server_shutdown_event

    host = host or ServerConfig.HOST
    port = port or ServerConfig.PORT

    if _server_task is not None and not _server_task.done():
        logger.warning("[Server] Background server already running")
        return True

    _server_shutdown_event = asyncio.Event()

    async def run_server():
        """Run uvicorn server in background."""
        try:
            import uvicorn
            from uvicorn import Config, Server

            config = Config(
                app=app,
                host=host,
                port=port,
                log_level=log_level,
                access_log=False,
            )
            server = Server(config)

            # Run server until shutdown event
            logger.info(f"[Server] Starting background API server on http://{host}:{port}")
            await server.serve()

        except Exception as e:
            logger.error(f"[Server] Background server error: {e}")

    # Start server in background task
    _server_task = asyncio.create_task(run_server())

    # Wait briefly to ensure server starts
    await asyncio.sleep(0.5)

    if _server_task.done():
        # Server failed to start
        try:
            _server_task.result()
        except Exception as e:
            logger.error(f"[Server] Failed to start: {e}")
            return False

    logger.info(f"[Server] Background API server started on port {port}")
    return True


async def stop_server_background() -> None:
    """Stop the background API server."""
    global _server_task, _server_shutdown_event

    if _server_shutdown_event:
        _server_shutdown_event.set()

    if _server_task and not _server_task.done():
        _server_task.cancel()
        try:
            await _server_task
        except asyncio.CancelledError:
            pass

    _server_task = None
    _server_shutdown_event = None
    logger.info("[Server] Background server stopped")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn

    logging.basicConfig(
        level=logging.DEBUG if ServerConfig.DEBUG else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    print("=" * 70)
    print("  Reactor-Core API Server v3.0")
    print("  AGI OS Nervous System - Training Pipeline & Telemetry")
    print("=" * 70)
    print()
    print(f"  Listening: http://{ServerConfig.HOST}:{ServerConfig.PORT}")
    print()
    print("  Endpoints:")
    print(f"    GET  /health                     - Health check")
    print(f"    GET  /api/v1/status              - Service status")
    print(f"    POST /api/v1/train               - Trigger training")
    print(f"    POST /api/v1/telemetry/ingest    - Ingest telemetry")
    print(f"    GET  /api/v1/scheduler/status    - Scheduler status")
    print(f"    GET  /api/v1/models/versions     - List model versions")
    print(f"    GET  /api/v1/health/dashboard    - Health dashboard")
    print(f"    WS   /ws/telemetry               - Real-time streaming")
    print()
    print("  Features:")
    print(f"    Telemetry:         {'Enabled' if ServerConfig.TELEMETRY_ENABLED else 'Disabled'}")
    print(f"    Scheduler:         {'Enabled' if ServerConfig.SCHEDULER_ENABLED else 'Disabled'}")
    print(f"    Health Aggregator: {'Enabled' if ServerConfig.HEALTH_AGGREGATOR_ENABLED else 'Disabled'}")
    print()
    print("=" * 70)

    uvicorn.run(
        "reactor_core.api.server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=ServerConfig.DEBUG,
        log_level="debug" if ServerConfig.DEBUG else "info",
    )


if __name__ == "__main__":
    main()
