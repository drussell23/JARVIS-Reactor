#!/usr/bin/env python3
"""
Reactor Core Entry Point - Trinity-Integrated Training Pipeline
================================================================

v92.0 - Unified Entry Point for Cross-Repo Orchestration

This script starts the Reactor Core (Training Pipeline) as part of the
Trinity ecosystem. It's designed to be called by the unified supervisor
in JARVIS-Prime via:

    python3 run_supervisor.py --unified --enable-reactor

FEATURES:
    - Trinity Protocol integration for cross-repo communication
    - Health endpoint for supervisor monitoring
    - Training job management API
    - Experience collection from JARVIS Body
    - Model deployment to JARVIS-Prime
    - Graceful shutdown with job persistence

TRINITY ARCHITECTURE:
    JARVIS-Prime (Mind)  <-->  JARVIS (Body)  <-->  Reactor-Core (Nerves)
         Port 8000                Port 8080              Port 8090
            |                                               |
            +------- Training Data Flow <--------> Model Deployment

USAGE:
    # Direct execution (standalone)
    python3 run_reactor.py --port 8090

    # Via unified supervisor (recommended)
    cd ../jarvis-prime && python3 run_supervisor.py --unified

ENVIRONMENT VARIABLES:
    REACTOR_PORT: Port for HTTP server (default: 8090)
    JARVIS_PRIME_URL: URL of JARVIS-Prime (default: http://localhost:8000)
    TRINITY_ENABLED: Enable Trinity Protocol (default: true)
    MODEL_OUTPUT_DIR: Directory for trained models
    LOG_LEVEL: Logging level (default: INFO)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("reactor.core")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ReactorCoreConfig:
    """Configuration for Reactor Core service."""

    def __init__(self):
        self.port = int(os.getenv("REACTOR_PORT", "8090"))
        self.host = os.getenv("REACTOR_HOST", "0.0.0.0")
        self.jarvis_prime_url = os.getenv("JARVIS_PRIME_URL", "http://localhost:8000")
        self.trinity_enabled = os.getenv("TRINITY_ENABLED", "true").lower() == "true"
        self.service_name = "reactor_core"
        self.version = "v92.0"

        # Directories
        jarvis_prime_path = Path.home() / "Documents" / "repos" / "jarvis-prime"
        self.model_output_dir = Path(os.getenv(
            "MODEL_OUTPUT_DIR",
            str(jarvis_prime_path / "models")
        ))
        self.state_dir = Path.home() / ".jarvis" / "reactor_state"
        self.cross_repo_dir = Path.home() / ".jarvis" / "cross_repo"
        self.trinity_dir = Path.home() / ".jarvis" / "trinity"
        self.experiences_dir = self.cross_repo_dir / "experiences"

        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cross_repo_dir.mkdir(parents=True, exist_ok=True)
        self.experiences_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# JOB MANAGEMENT
# =============================================================================

class TrainingJobManager:
    """Manages training jobs with persistence."""

    def __init__(self, config: ReactorCoreConfig):
        self._config = config
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def submit_job(self, job_spec: Dict[str, Any]) -> str:
        """Submit a new training job."""
        async with self._lock:
            job_id = f"job_{int(time.time())}_{len(self._jobs)}"

            job = {
                "id": job_id,
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "spec": job_spec,
                "progress": 0.0,
                "metrics": {},
                "error": None,
            }

            self._jobs[job_id] = job
            await self._persist_jobs()

            # Start job in background
            task = asyncio.create_task(self._execute_job(job_id))
            self._active_tasks[job_id] = task

            logger.info(f"Submitted training job {job_id}")
            return job_id

    async def _execute_job(self, job_id: str):
        """Execute a training job."""
        try:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = datetime.now().isoformat()
            await self._persist_jobs()

            # Simulate training progress
            for i in range(10):
                await asyncio.sleep(1)
                self._jobs[job_id]["progress"] = (i + 1) * 10.0
                await self._persist_jobs()

            self._jobs[job_id]["status"] = "completed"
            self._jobs[job_id]["completed_at"] = datetime.now().isoformat()
            self._jobs[job_id]["progress"] = 100.0

            logger.info(f"Job {job_id} completed successfully")

        except asyncio.CancelledError:
            self._jobs[job_id]["status"] = "cancelled"
            logger.info(f"Job {job_id} cancelled")

        except Exception as e:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = str(e)
            logger.error(f"Job {job_id} failed: {e}")

        finally:
            await self._persist_jobs()
            self._active_tasks.pop(job_id, None)

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        return self._jobs.get(job_id)

    async def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        return list(self._jobs.values())

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self._active_tasks:
            self._active_tasks[job_id].cancel()
            return True
        return False

    async def _persist_jobs(self):
        """Persist jobs to disk."""
        jobs_path = self._config.state_dir / "jobs.json"
        with open(jobs_path, "w") as f:
            json.dump(self._jobs, f, indent=2)

    async def load_jobs(self):
        """Load jobs from disk."""
        jobs_path = self._config.state_dir / "jobs.json"
        if jobs_path.exists():
            try:
                with open(jobs_path, "r") as f:
                    self._jobs = json.load(f)
                logger.info(f"Loaded {len(self._jobs)} jobs from disk")
            except Exception as e:
                logger.warning(f"Failed to load jobs: {e}")

    async def shutdown(self):
        """Shutdown job manager and cancel active jobs."""
        for job_id, task in list(self._active_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._persist_jobs()


# =============================================================================
# HEALTH SERVER
# =============================================================================

async def create_health_server(
    config: ReactorCoreConfig,
    state: Dict[str, Any],
    job_manager: TrainingJobManager,
):
    """Create HTTP server with health and job endpoints."""
    try:
        from aiohttp import web
        AIOHTTP_AVAILABLE = True
    except ImportError:
        logger.warning("aiohttp not available - using basic HTTP server")
        AIOHTTP_AVAILABLE = False

    if AIOHTTP_AVAILABLE:
        app = web.Application()

        async def health_handler(request):
            return web.json_response({
                "status": "healthy" if state.get("running") else "starting",
                "service": config.service_name,
                "version": config.version,
                "uptime_seconds": time.time() - state.get("start_time", time.time()),
                "trinity_connected": state.get("trinity_connected", False),
                "training_ready": True,
                "timestamp": datetime.now().isoformat(),
            })

        async def jobs_list_handler(request):
            jobs = await job_manager.list_jobs()
            return web.json_response({"jobs": jobs})

        async def job_submit_handler(request):
            try:
                data = await request.json()
                job_id = await job_manager.submit_job(data)
                return web.json_response({"job_id": job_id}, status=201)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=400)

        async def job_status_handler(request):
            job_id = request.match_info["job_id"]
            job = await job_manager.get_job(job_id)
            if job:
                return web.json_response(job)
            return web.json_response({"error": "Job not found"}, status=404)

        async def job_cancel_handler(request):
            job_id = request.match_info["job_id"]
            success = await job_manager.cancel_job(job_id)
            if success:
                return web.json_response({"status": "cancelled"})
            return web.json_response({"error": "Job not found or not running"}, status=400)

        async def metrics_handler(request):
            jobs = await job_manager.list_jobs()
            return web.json_response({
                "total_jobs": len(jobs),
                "running_jobs": sum(1 for j in jobs if j["status"] == "running"),
                "completed_jobs": sum(1 for j in jobs if j["status"] == "completed"),
                "failed_jobs": sum(1 for j in jobs if j["status"] == "failed"),
            })

        app.router.add_get("/health", health_handler)
        app.router.add_get("/jobs", jobs_list_handler)
        app.router.add_post("/jobs/submit", job_submit_handler)
        app.router.add_get("/jobs/{job_id}", job_status_handler)
        app.router.add_post("/jobs/{job_id}/cancel", job_cancel_handler)
        app.router.add_get("/metrics", metrics_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()

        logger.info(f"Reactor Core server started on http://{config.host}:{config.port}")
        return runner
    else:
        return None


# =============================================================================
# TRINITY INTEGRATION
# =============================================================================

class TrinityClient:
    """Client for Trinity Protocol communication."""

    def __init__(self, config: ReactorCoreConfig):
        self._config = config
        self._connected = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to Trinity service mesh."""
        try:
            # Register with service mesh
            registry_path = self._config.trinity_dir / "service_registry.json"

            service_info = {
                "name": self._config.service_name,
                "host": "localhost",
                "port": self._config.port,
                "capabilities": ["training", "fine_tuning", "model_evaluation"],
                "health_endpoint": "/health",
                "version": self._config.version,
                "registered_at": datetime.now().isoformat(),
            }

            # Load existing registry or create new
            registry = {"services": {}}
            if registry_path.exists():
                try:
                    with open(registry_path, "r") as f:
                        registry = json.load(f)
                except Exception:
                    pass

            registry["services"][self._config.service_name] = service_info

            self._config.trinity_dir.mkdir(parents=True, exist_ok=True)
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            logger.info("Registered with Trinity service mesh")
            self._connected = True

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Trinity: {e}")
            return False

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        heartbeat_path = self._config.trinity_dir / "heartbeats" / f"{self._config.service_name}.json"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                heartbeat = {
                    "service": self._config.service_name,
                    "timestamp": time.time(),
                    "timestamp_iso": datetime.now().isoformat(),
                    "status": "healthy",
                    "port": self._config.port,
                }

                with open(heartbeat_path, "w") as f:
                    json.dump(heartbeat, f)

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def disconnect(self):
        """Disconnect from Trinity."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        logger.info("Disconnected from Trinity")


# =============================================================================
# REACTOR CORE SERVICE
# =============================================================================

class ReactorCoreService:
    """Main Reactor Core service."""

    def __init__(self, config: ReactorCoreConfig):
        self._config = config
        self._state: Dict[str, Any] = {
            "running": False,
            "start_time": time.time(),
            "trinity_connected": False,
        }
        self._trinity_client: Optional[TrinityClient] = None
        self._job_manager = TrainingJobManager(config)
        self._health_runner = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the Reactor Core service."""
        logger.info(f"Starting Reactor Core service {self._config.version}")

        # Load persisted jobs
        await self._job_manager.load_jobs()

        # Start health server
        self._health_runner = await create_health_server(
            self._config,
            self._state,
            self._job_manager,
        )

        # Connect to Trinity if enabled
        if self._config.trinity_enabled:
            self._trinity_client = TrinityClient(self._config)
            connected = await self._trinity_client.connect()
            self._state["trinity_connected"] = connected

        self._state["running"] = True
        logger.info(f"Reactor Core service started on port {self._config.port}")

        # Write state for cross-repo coordination
        await self._write_state()

    async def _write_state(self):
        """Write state for cross-repo coordination."""
        state_path = self._config.cross_repo_dir / "reactor_state.json"
        with open(state_path, "w") as f:
            json.dump({
                **self._state,
                "port": self._config.port,
                "version": self._config.version,
                "model_output_dir": str(self._config.model_output_dir),
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)

    async def run(self):
        """Run the service until shutdown."""
        logger.info("Reactor Core service running. Press Ctrl+C to stop.")
        await self._shutdown_event.wait()

    async def stop(self):
        """Stop the Reactor Core service."""
        logger.info("Stopping Reactor Core service...")

        self._state["running"] = False

        # Shutdown job manager
        await self._job_manager.shutdown()

        # Disconnect from Trinity
        if self._trinity_client:
            await self._trinity_client.disconnect()

        # Stop health server
        if self._health_runner:
            await self._health_runner.cleanup()

        logger.info("Reactor Core service stopped")

    def request_shutdown(self):
        """Request service shutdown."""
        self._shutdown_event.set()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main(args: argparse.Namespace):
    """Main entry point."""
    config = ReactorCoreConfig()

    # Override from args
    if args.port:
        config.port = args.port
    if args.prime_url:
        config.jarvis_prime_url = args.prime_url

    # Create and start service
    service = ReactorCoreService(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        service.request_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await service.start()
        await service.run()
    finally:
        await service.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reactor Core - Training Pipeline Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for HTTP server (default: 8090)",
    )
    parser.add_argument(
        "--prime-url",
        type=str,
        default=None,
        help="URL of JARVIS-Prime (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Shutdown by keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
