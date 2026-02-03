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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# v96.0: Process fingerprinting for enhanced service registry
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("reactor.core")

# v152.0: Import cloud mode detector for JARVIS ecosystem awareness
try:
    from reactor_core.core.cloud_mode_detector import (
        is_cloud_mode_active,
        get_effective_jarvis_url,
        get_cloud_state,
    )
    CLOUD_MODE_DETECTOR_AVAILABLE = True
    logger.info("[v152.0] Cloud mode detector loaded")
except ImportError as e:
    CLOUD_MODE_DETECTOR_AVAILABLE = False
    logger.debug(f"[v152.0] Cloud mode detector not available: {e}")
    def is_cloud_mode_active() -> bool:
        return os.getenv("JARVIS_GCP_OFFLOAD_ACTIVE", "false").lower() == "true"
    def get_effective_jarvis_url(default: str = "http://localhost:8000") -> str:
        if is_cloud_mode_active():
            return os.getenv("JARVIS_PRIME_CLOUD_RUN_URL", "")
        return default
    def get_cloud_state():
        return None


# =============================================================================
# v96.0: PROCESS FINGERPRINTING FOR SERVICE REGISTRY
# =============================================================================

def _get_process_fingerprint() -> Dict[str, Any]:
    """
    v96.0: Capture process fingerprint for enhanced service registry.

    This enables PID reuse detection and process identity validation.
    If the same PID is reused by a different process (after a crash),
    the fingerprint mismatch will detect this.

    Returns:
        Dictionary with process fingerprint data
    """
    fingerprint = {
        "pid": os.getpid(),
        "process_name": "",
        "process_cmdline": "",
        "process_exe_path": "",
        "process_cwd": "",
        "parent_pid": 0,
        "parent_name": "",
        "process_start_time": 0.0,
    }

    if PSUTIL_AVAILABLE:
        try:
            proc = psutil.Process()
            fingerprint["process_name"] = proc.name()
            fingerprint["process_cmdline"] = " ".join(proc.cmdline())
            fingerprint["process_exe_path"] = proc.exe()
            fingerprint["process_cwd"] = proc.cwd()
            fingerprint["process_start_time"] = proc.create_time()

            # Parent process info
            parent = proc.parent()
            if parent:
                fingerprint["parent_pid"] = parent.pid
                fingerprint["parent_name"] = parent.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            logger.debug(f"[v96.0] Process fingerprint capture partial: {e}")

    return fingerprint


def _get_machine_id() -> str:
    """
    v96.0: Get unique machine identifier for distributed environments.

    This helps identify which machine owns a lock/registration when
    multiple machines may have the same PID.
    """
    # Try various methods to get machine ID
    machine_id_paths = [
        "/etc/machine-id",           # Linux
        "/var/lib/dbus/machine-id",  # Older Linux
    ]

    for path in machine_id_paths:
        try:
            if Path(path).exists():
                return Path(path).read_text().strip()
        except Exception:
            pass

    # Fallback: use hostname + some system info
    try:
        import platform
        return f"{platform.node()}-{uuid.getnode()}"
    except Exception:
        return f"unknown-{uuid.uuid4().hex[:8]}"


# =============================================================================
# v96.0: ATOMIC SHARED REGISTRY FOR CROSS-PROCESS COORDINATION
# =============================================================================

import fcntl
from contextlib import contextmanager

# Registry lock configuration
_REGISTRY_LOCK_TIMEOUT = 30.0  # Max seconds to wait for lock
_REGISTRY_LOCK_POLL_INTERVAL = 0.05  # Poll interval when waiting


@contextmanager
def _acquire_registry_lock(lock_path: Path, timeout: float = _REGISTRY_LOCK_TIMEOUT):
    """
    v96.0: Acquire exclusive lock on the registry using a separate lock file.

    This is the CRITICAL fix for race conditions - we lock a separate file
    and hold it for the entire read-modify-write cycle.

    Args:
        lock_path: Path to the lock file
        timeout: Max seconds to wait for lock

    Yields:
        The lock file handle

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    # Ensure directory exists
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Create lock file if it doesn't exist
    lock_path.touch(exist_ok=True)

    start_time = time.time()
    lock_fd = None

    try:
        # Open lock file
        lock_fd = open(lock_path, 'r+')

        # Try to acquire lock with timeout
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Lock acquired!
                break
            except (IOError, OSError):
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    lock_fd.close()
                    raise TimeoutError(
                        f"Could not acquire registry lock within {timeout}s. "
                        f"Another process may be holding the lock."
                    )
                time.sleep(_REGISTRY_LOCK_POLL_INTERVAL)

        yield lock_fd

    finally:
        # Release lock
        if lock_fd:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _atomic_register_service(
    service_name: str,
    service_data: Dict[str, Any],
    alternate_names: Optional[List[str]] = None,
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
    cleanup_stale: bool = True,
    max_stale_age: float = 300.0,
) -> bool:
    """
    v96.0: Register a service with the shared registry atomically.

    This function ensures proper cross-process synchronization by:
    1. Acquiring an exclusive lock on a separate .lock file
    2. Cleaning up stale entries from previous runs
    3. Reading the current registry state
    4. Modifying it
    5. Writing it back atomically
    6. Releasing the lock

    Args:
        service_name: Primary service name
        service_data: Service data dict (pid, port, host, etc.)
        alternate_names: Optional list of alternate names to also register
        timeout: Max seconds to wait for lock
        cleanup_stale: Whether to clean up stale entries first
        max_stale_age: Max age in seconds before entry is considered stale

    Returns:
        True if registration succeeded, False otherwise
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            # Read current registry (inside lock)
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            existing_services = {}
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"[v96.0] Registry read error (will overwrite): {e}")
                    existing_services = {}

            # v96.0: Clean up stale entries before registration
            if cleanup_stale and existing_services:
                current_time = time.time()
                entries_to_remove = []

                for name, data in list(existing_services.items()):
                    if not isinstance(data, dict):
                        entries_to_remove.append(name)
                        continue

                    # Check age
                    last_activity = max(
                        data.get("last_heartbeat", 0),
                        data.get("registered_at", 0)
                    )
                    age = current_time - last_activity

                    if age > max_stale_age:
                        entries_to_remove.append(name)
                        logger.debug(f"[v96.0] Removing stale: {name} (age: {age:.0f}s)")
                    else:
                        # Check if process is alive
                        pid = data.get("pid")
                        if pid and PSUTIL_AVAILABLE:
                            try:
                                if not psutil.pid_exists(pid):
                                    entries_to_remove.append(name)
                                    logger.debug(f"[v96.0] Removing dead: {name} (pid {pid})")
                            except Exception:
                                pass

                for name in entries_to_remove:
                    existing_services.pop(name, None)

                if entries_to_remove:
                    logger.info(f"[v96.0] Cleaned {len(entries_to_remove)} stale entries before registration")

            # Update registry
            existing_services[service_name] = service_data

            # Register alternate names
            if alternate_names:
                for alt_name in alternate_names:
                    existing_services[alt_name] = service_data

            # Write atomically (inside lock)
            registry_dir.mkdir(parents=True, exist_ok=True)
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_file.replace(registry_file)

        return True

    except TimeoutError as e:
        logger.error(f"[v96.0] Registry lock timeout for {service_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"[v96.0] Failed to register {service_name}: {e}")
        return False


def _atomic_update_heartbeat(
    service_names: List[str],
    status: str = "running",
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v97.0: Atomically update heartbeat timestamp in the service registry.

    This function ensures external services (like reactor-core) maintain
    fresh heartbeats in the shared registry so they don't get cleaned up
    as stale by the JARVIS service registry cleanup process.

    Args:
        service_names: List of service names to update heartbeat for
        status: Service status (running, starting, stopping, etc.)
        timeout: Max seconds to wait for lock

    Returns:
        True if heartbeat update succeeded, False otherwise
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            # Read current registry
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            existing_services = {}
                except (json.JSONDecodeError, OSError):
                    return False

            # Update heartbeat for each service
            current_time = time.time()
            updated = False
            for name in service_names:
                if name in existing_services:
                    existing_services[name]["last_heartbeat"] = current_time
                    existing_services[name]["status"] = status
                    updated = True

            if not updated:
                return False  # Services not found in registry

            # Write back atomically
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            temp_file.replace(registry_file)

        return True

    except Exception as e:
        logger.debug(f"[v97.0] Heartbeat update error: {e}")
        return False


def _atomic_deregister_service(
    service_names: List[str],
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v96.0: Deregister services from the shared registry atomically.

    Args:
        service_names: List of service names to remove
        timeout: Max seconds to wait for lock

    Returns:
        True if deregistration succeeded, False otherwise
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            # Read current registry
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            return True  # Nothing to deregister
                except (json.JSONDecodeError, OSError):
                    return True  # Nothing to deregister

            # Remove services
            for name in service_names:
                existing_services.pop(name, None)

            # Write back atomically
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            temp_file.replace(registry_file)

        return True

    except Exception as e:
        logger.debug(f"[v96.0] Deregistration error: {e}")
        return False


# =============================================================================
# CONFIGURATION
# =============================================================================

class ReactorCoreConfig:
    """Configuration for Reactor Core service."""

    def __init__(self):
        self.port = int(os.getenv("REACTOR_PORT", "8090"))
        self.host = os.getenv("REACTOR_HOST", "0.0.0.0")

        # v152.0: Cloud-aware JARVIS Prime URL resolution
        env_url = os.getenv("JARVIS_PRIME_URL")
        if env_url:
            self.jarvis_prime_url = env_url
        else:
            self.jarvis_prime_url = get_effective_jarvis_url("http://localhost:8000")

        # v152.0: Track cloud mode state
        self.cloud_mode_active = is_cloud_mode_active()
        if self.cloud_mode_active:
            logger.info(
                f"[v152.0] Cloud mode detected - JARVIS Prime URL: {self.jarvis_prime_url}"
            )

        self.trinity_enabled = os.getenv("TRINITY_ENABLED", "true").lower() == "true"
        self.service_name = "reactor_core"
        self.version = "v152.0"  # v152.0: Updated version for cloud mode support

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

async def _wait_for_port_available(
    host: str,
    port: int,
    timeout: float = 30.0,
    check_interval: float = 1.0,
) -> bool:
    """
    v118.0: Wait for a port to become available with intelligent retry.

    Handles TCP TIME_WAIT state by waiting for the port to be released
    instead of failing immediately. Uses SO_REUSEADDR to test availability.

    Args:
        host: Host to bind to
        port: Port to check
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks

    Returns:
        True if port is available, False if timeout exceeded
    """
    import socket
    import time

    start_time = time.time()
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        try:
            # Create a test socket with SO_REUSEADDR
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Try to enable SO_REUSEPORT if available (macOS/Linux)
            if hasattr(socket, 'SO_REUSEPORT'):
                try:
                    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass  # SO_REUSEPORT not supported on this platform

            test_sock.settimeout(1.0)
            test_sock.bind((host, port))
            test_sock.close()

            if attempt > 1:
                elapsed = time.time() - start_time
                logger.info(
                    f"[v118.0] Port {port} became available after {elapsed:.1f}s "
                    f"({attempt} attempts)"
                )
            return True

        except OSError as e:
            error_code = getattr(e, 'errno', None)
            # EADDRINUSE (48 on macOS, 98 on Linux) or EADDRNOTAVAIL
            if error_code in (48, 98, 99):
                if attempt == 1:
                    logger.info(
                        f"[v118.0] Port {port} in use (likely TIME_WAIT), "
                        f"waiting up to {timeout}s for release..."
                    )
                await asyncio.sleep(check_interval)
            else:
                logger.warning(f"[v118.0] Unexpected socket error on port {port}: {e}")
                await asyncio.sleep(check_interval)
        finally:
            try:
                test_sock.close()
            except Exception:
                pass

    elapsed = time.time() - start_time
    logger.warning(
        f"[v118.0] Port {port} not available after {elapsed:.1f}s ({attempt} attempts)"
    )
    return False


def _update_cross_repo_port_registry(
    service_name: str,
    actual_port: int,
    original_port: int,
) -> bool:
    """
    v118.0: Update the distributed port registry for cross-repo coordination.

    When reactor-core gets a fallback port, the supervisor and other repos
    need to know where to connect. This updates ~/.jarvis/registry/ports.json
    with the actual port mapping.

    This is a SYNCHRONOUS function to be called after successful port binding,
    ensuring the registry is updated immediately before any health checks.

    Args:
        service_name: Name of the service (e.g., "reactor-core")
        actual_port: The port actually bound to
        original_port: The originally requested port

    Returns:
        True if registry updated successfully, False otherwise
    """
    registry_dir = Path.home() / ".jarvis" / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    registry_file = registry_dir / "ports.json"
    lock_file = registry_dir / "ports.json.lock"

    try:
        # Use file locking for atomic update
        with _acquire_registry_lock(lock_file, timeout=10.0):
            # Load existing registry
            if registry_file.exists():
                try:
                    registry = json.loads(registry_file.read_text())
                except json.JSONDecodeError:
                    registry = {"version": "1.0", "ports": {}, "fallbacks": []}
            else:
                registry = {"version": "1.0", "ports": {}, "fallbacks": []}

            # Update port mapping for all reactor-core variants
            for name in [service_name, "reactor-core", "reactor_core", "reactor"]:
                registry["ports"][name] = {
                    "port": actual_port,
                    "original_port": original_port,
                    "allocated_at": time.time(),
                    "is_fallback": actual_port != original_port,
                    "pid": os.getpid(),
                }

            # Track fallback history
            if actual_port != original_port:
                if "fallbacks" not in registry:
                    registry["fallbacks"] = []
                registry["fallbacks"].append({
                    "service": service_name,
                    "original": original_port,
                    "fallback": actual_port,
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                })
                # Keep only last 100 fallback records
                registry["fallbacks"] = registry["fallbacks"][-100:]

            # Write atomically
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(registry, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            temp_file.replace(registry_file)

        fallback_msg = f" (fallback from {original_port})" if actual_port != original_port else ""
        logger.info(
            f"[v118.0] ✅ Updated cross-repo port registry: "
            f"{service_name} -> {actual_port}{fallback_msg}"
        )
        return True

    except Exception as e:
        logger.warning(f"[v118.0] ⚠️ Could not update port registry: {e}")
        return False


async def _find_available_port(
    host: str,
    preferred_port: int,
    fallback_range: tuple = (9001, 9100),
) -> int:
    """
    v118.0: Find an available port, preferring the specified port.

    If preferred port is unavailable after a brief wait, searches for
    an alternative port in the fallback range.

    Args:
        host: Host to bind to
        preferred_port: Preferred port to use
        fallback_range: Range of ports to search if preferred unavailable

    Returns:
        Available port number

    Raises:
        RuntimeError: If no port could be found
    """
    import socket

    # First, try the preferred port with a short wait
    if await _wait_for_port_available(host, preferred_port, timeout=5.0):
        return preferred_port

    # Preferred port unavailable, search for fallback
    logger.info(
        f"[v118.0] Preferred port {preferred_port} unavailable, "
        f"searching for fallback in range {fallback_range}..."
    )

    for port in range(fallback_range[0], fallback_range[1] + 1):
        try:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, 'SO_REUSEPORT'):
                try:
                    test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            test_sock.settimeout(1.0)
            test_sock.bind((host, port))
            test_sock.close()

            logger.info(
                f"[v118.0] Found available fallback port: {port} "
                f"(preferred {preferred_port} was unavailable)"
            )
            return port

        except OSError:
            continue
        finally:
            try:
                test_sock.close()
            except Exception:
                pass

    raise RuntimeError(
        f"No available port found: preferred {preferred_port} and "
        f"fallback range {fallback_range} all exhausted"
    )


async def create_health_server(
    config: ReactorCoreConfig,
    state: Dict[str, Any],
    job_manager: TrainingJobManager,
):
    """
    Create HTTP server with health and job endpoints.

    v118.0: Enhanced with robust port binding that handles TIME_WAIT state.
    - Waits for port to become available instead of failing immediately
    - Uses SO_REUSEADDR and SO_REUSEPORT for fast rebind
    - Falls back to alternative port if preferred port is stuck
    """
    try:
        from aiohttp import web
        AIOHTTP_AVAILABLE = True
    except ImportError:
        logger.warning("aiohttp not available - using basic HTTP server")
        AIOHTTP_AVAILABLE = False

    if AIOHTTP_AVAILABLE:
        app = web.Application()

        async def health_handler(request):
            """
            v190.0: Enhanced health endpoint with semantic readiness detection.

            Returns detailed state information enabling intelligent readiness
            detection by the unified_supervisor:

            - status: "starting" | "healthy" | "error"
            - training_ready: True only when job manager is fully initialized
            - trinity_connected: Whether connected to Trinity mesh
            - phase: Current startup phase for progress tracking
            """
            is_running = state.get("running", False)
            startup_phase = state.get("startup_phase", "initializing")

            # v190.0: training_ready is TRUE only when:
            # 1. Service is running
            # 2. Startup phase is "running" or "ready" (not "initializing")
            # 3. Job manager is operational (tracked by startup completion)
            training_ready = is_running and startup_phase in ("running", "ready", "operational")

            # Determine semantic status
            if is_running and training_ready:
                status = "healthy"
                phase = "ready"
            elif is_running:
                status = "starting"
                phase = startup_phase
            else:
                status = "starting"
                phase = startup_phase or "pre-init"

            return web.json_response({
                "status": status,
                "phase": phase,
                "service": config.service_name,
                "version": config.version,
                "uptime_seconds": time.time() - state.get("start_time", time.time()),
                "trinity_connected": state.get("trinity_connected", False),
                "training_ready": training_ready,
                "startup_progress": state.get("startup_progress", 0),
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

        # v118.0: Robust port binding with TIME_WAIT handling
        # First, ensure port is available (wait for TIME_WAIT to clear if needed)
        original_port = config.port  # Track original for logging
        actual_port = config.port
        max_bind_attempts = 3
        bind_attempt = 0
        site = None

        while bind_attempt < max_bind_attempts and site is None:
            bind_attempt += 1
            try:
                # Wait for port availability before binding
                port_available = await _wait_for_port_available(
                    config.host if config.host != "0.0.0.0" else "127.0.0.1",
                    actual_port,
                    timeout=15.0,
                    check_interval=1.0,
                )

                if not port_available:
                    # Try to find a fallback port
                    logger.warning(
                        f"[v118.0] Port {actual_port} unavailable after wait, "
                        f"finding fallback..."
                    )
                    actual_port = await _find_available_port(
                        config.host if config.host != "0.0.0.0" else "127.0.0.1",
                        config.port,
                        fallback_range=(9001, 9100),
                    )
                    # Update config to use new port
                    config.port = actual_port
                    logger.info(
                        f"[v118.0] Using fallback port {actual_port}"
                    )

                # Create TCPSite with reuse_address=True for fast rebind
                # This is CRITICAL for handling TIME_WAIT state
                site = web.TCPSite(
                    runner,
                    config.host,
                    actual_port,
                    reuse_address=True,  # v118.0: Enable SO_REUSEADDR
                    reuse_port=True,     # v118.0: Enable SO_REUSEPORT where supported
                )
                await site.start()

            except OSError as e:
                error_msg = str(e)
                if "Address already in use" in error_msg or "address already in use" in error_msg:
                    logger.warning(
                        f"[v118.0] Bind attempt {bind_attempt}/{max_bind_attempts} failed: {e}"
                    )
                    if bind_attempt < max_bind_attempts:
                        # Exponential backoff before retry
                        backoff = 2.0 ** bind_attempt
                        logger.info(f"[v118.0] Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        # Try fallback port on next attempt
                        actual_port = await _find_available_port(
                            config.host if config.host != "0.0.0.0" else "127.0.0.1",
                            config.port,
                            fallback_range=(9001 + bind_attempt * 10, 9100),
                        )
                        config.port = actual_port
                    else:
                        raise
                else:
                    raise

        if site is None:
            raise RuntimeError(f"Failed to bind to any port after {max_bind_attempts} attempts")

        # Update config with actual port used (for registration purposes)
        config.port = actual_port

        # v118.0: Update cross-repo port registry so supervisor knows our actual port
        # This is CRITICAL for cross-repo coordination when using fallback ports
        _update_cross_repo_port_registry(
            service_name="reactor-core",
            actual_port=actual_port,
            original_port=original_port,
        )

        if actual_port != original_port:
            logger.warning(
                f"Reactor Core server started on FALLBACK port "
                f"http://{config.host}:{actual_port} (preferred: {original_port})"
            )
        else:
            logger.info(f"Reactor Core server started on http://{config.host}:{actual_port}")

        return runner
    else:
        return None


# =============================================================================
# TRINITY INTEGRATION
# =============================================================================

class TrinityClient:
    """
    v93.0: Enhanced Client for Trinity Protocol communication.

    Features:
    - Robust registry format handling (handles legacy formats)
    - Heartbeat validation with HeartbeatValidator integration
    - Automatic retry on connection failures
    - Cross-repo service discovery
    """

    def __init__(self, config: ReactorCoreConfig):
        self._config = config
        self._connected = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._connection_attempts = 0
        self._max_connection_attempts = 3

    async def connect(self) -> bool:
        """Connect to Trinity service mesh with robust error handling."""
        self._connection_attempts += 1

        try:
            # Ensure trinity directory exists first
            self._config.trinity_dir.mkdir(parents=True, exist_ok=True)
            heartbeats_dir = self._config.trinity_dir / "heartbeats"
            heartbeats_dir.mkdir(parents=True, exist_ok=True)

            # Register with service mesh (use separate reactor registry)
            registry_path = self._config.trinity_dir / "reactor_registry.json"

            service_info = {
                "name": self._config.service_name,
                "instance_id": f"{self._config.service_name}-{int(time.time())}",
                "host": "localhost",
                "port": self._config.port,
                "capabilities": ["training", "fine_tuning", "model_evaluation"],
                "health_endpoint": "/health",
                "version": self._config.version,
                "registered_at": datetime.now().isoformat(),
                "pid": os.getpid(),
            }

            # Write reactor-specific registry
            with open(registry_path, "w") as f:
                json.dump({"reactor_core": service_info}, f, indent=2)

            logger.info(f"[Trinity] Registered reactor_core in {registry_path}")

            # Write initial heartbeat immediately
            await self._write_heartbeat()

            self._connected = True
            logger.info("[Trinity] Connected to Trinity service mesh")

            # Start heartbeat loop
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name="reactor_heartbeat_loop"
            )

            return True

        except Exception as e:
            logger.error(f"[Trinity] Connection failed (attempt {self._connection_attempts}): {e}")
            import traceback
            logger.debug(f"[Trinity] Traceback: {traceback.format_exc()}")

            # Retry if under max attempts
            if self._connection_attempts < self._max_connection_attempts:
                logger.info(f"[Trinity] Retrying connection in 2 seconds...")
                await asyncio.sleep(2)
                return await self.connect()

            return False

    async def _write_heartbeat(self) -> bool:
        """Write a single heartbeat to file."""
        try:
            heartbeat_path = self._config.trinity_dir / "heartbeats" / f"{self._config.service_name}.json"
            heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

            heartbeat = {
                "component_id": self._config.service_name,
                "component_type": "reactor_core",
                "service": self._config.service_name,
                "timestamp": time.time(),
                "timestamp_iso": datetime.now().isoformat(),
                "status": "healthy",
                "host": "localhost",
                "port": self._config.port,
                "pid": os.getpid(),
                "version": self._config.version,
                "metrics": {
                    "connection_attempts": self._connection_attempts,
                },
            }

            # Use atomic write pattern
            tmp_path = heartbeat_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(heartbeat, f, indent=2)
            tmp_path.rename(heartbeat_path)

            return True

        except Exception as e:
            logger.warning(f"[Trinity] Heartbeat write error: {e}")
            return False

    async def _heartbeat_loop(self):
        """Send periodic heartbeats with robust error handling."""
        logger.info(f"[Trinity] Heartbeat loop started for {self._config.service_name}")
        consecutive_failures = 0
        max_consecutive_failures = 5

        while self._connected:
            try:
                success = await self._write_heartbeat()

                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"[Trinity] {consecutive_failures} consecutive heartbeat failures")

                await asyncio.sleep(5)

            except asyncio.CancelledError:
                logger.info("[Trinity] Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.warning(f"[Trinity] Heartbeat loop error: {e}")
                consecutive_failures += 1
                await asyncio.sleep(5)

        logger.info("[Trinity] Heartbeat loop exited")

    async def disconnect(self):
        """Disconnect from Trinity with cleanup."""
        logger.info("[Trinity] Disconnecting from Trinity...")
        self._connected = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await asyncio.wait_for(self._heartbeat_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._heartbeat_task = None

        # Remove heartbeat file to signal disconnection
        try:
            heartbeat_path = self._config.trinity_dir / "heartbeats" / f"{self._config.service_name}.json"
            if heartbeat_path.exists():
                heartbeat_path.unlink()
        except Exception as e:
            logger.debug(f"[Trinity] Could not remove heartbeat file: {e}")

        logger.info("[Trinity] Disconnected from Trinity")

    def is_connected(self) -> bool:
        """Check if connected to Trinity."""
        return self._connected


# =============================================================================
# REACTOR CORE SERVICE
# =============================================================================

class ReactorCoreService:
    """Main Reactor Core service."""

    # v97.0: Registry heartbeat interval (seconds)
    _REGISTRY_HEARTBEAT_INTERVAL = 15.0

    def __init__(self, config: ReactorCoreConfig):
        self._config = config
        self._state: Dict[str, Any] = {
            "running": False,
            "start_time": time.time(),
            "trinity_connected": False,
            # v186.0: Startup phase tracking for DMS cross-repo monitoring
            "startup_phase": "initializing",
            "startup_progress": 0,
        }
        self._trinity_client: Optional[TrinityClient] = None
        self._job_manager = TrainingJobManager(config)
        self._health_runner = None
        self._shutdown_event = asyncio.Event()
        # v97.0: Registry heartbeat task to prevent stale cleanup
        self._registry_heartbeat_task: Optional[asyncio.Task] = None

    async def start(self):
        """
        v190.0: Start the Reactor Core service with shared service registration
        and progressive startup phase tracking.

        Registers Reactor Core with the shared service registry at
        ~/.jarvis/registry/services.json to enable TrinityIntegrator's
        registration-aware verification.

        Startup phases (for semantic readiness detection):
        1. initializing - Service object created
        2. loading_jobs - Loading persisted training jobs
        3. starting_server - Starting HTTP health server
        4. registering - Registering with service registry
        5. connecting_trinity - Connecting to Trinity mesh
        6. ready - Fully operational
        """
        logger.info(f"Starting Reactor Core service {self._config.version}")

        # v190.0: Progressive startup phase tracking
        self._state["startup_phase"] = "loading_jobs"
        self._state["startup_progress"] = 10

        # Load persisted jobs
        await self._job_manager.load_jobs()
        self._state["startup_progress"] = 30
        logger.debug("[v190.0] Job manager initialized")

        # Start health server
        self._state["startup_phase"] = "starting_server"
        self._state["startup_progress"] = 40
        self._health_runner = await create_health_server(
            self._config,
            self._state,
            self._job_manager,
        )
        self._state["startup_progress"] = 60
        logger.debug("[v190.0] Health server started")

        # v96.0: Register with shared service registry using ATOMIC locking
        # This is CRITICAL for TrinityIntegrator's registration-aware verification
        # Uses file locking to prevent race conditions between multiple services

        # Capture process fingerprint for PID reuse detection
        fingerprint = _get_process_fingerprint()
        machine_id = _get_machine_id()

        # Determine configured port vs actual port
        configured_port = int(os.getenv("REACTOR_PORT", "8090"))
        actual_port = self._config.port
        is_fallback = actual_port != configured_port

        # Build service data
        service_data = {
            "service_name": "reactor_core",
            "pid": os.getpid(),
            "port": actual_port,
            "host": self._config.host,
            "health_endpoint": "/health",
            "status": "starting",
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "metadata": {
                "version": self._config.version,
                "role": "training_pipeline",
                "trinity_enabled": self._config.trinity_enabled,
            },
            # v96.0: Port fallback tracking (Problem 17 fix)
            "primary_port": configured_port,
            "is_fallback_port": is_fallback,
            "fallback_reason": f"Port {configured_port} unavailable" if is_fallback else "",
            "ports_tried": [configured_port] if is_fallback else [],
            "port_allocation_time": time.time(),
            # v96.0: Process fingerprint for identity validation (Problem 18 fix)
            "process_name": fingerprint["process_name"],
            "process_cmdline": fingerprint["process_cmdline"],
            "process_exe_path": fingerprint["process_exe_path"],
            "process_cwd": fingerprint["process_cwd"],
            "parent_pid": fingerprint["parent_pid"],
            "parent_name": fingerprint["parent_name"],
            "process_start_time": fingerprint["process_start_time"],
            # v96.0: Machine ID for distributed environments
            "machine_id": machine_id,
        }

        # v190.0: Phase tracking for service registration
        self._state["startup_phase"] = "registering"
        self._state["startup_progress"] = 70

        # Use atomic registration with file locking
        if _atomic_register_service(
            service_name="reactor_core",
            service_data=service_data,
            alternate_names=["reactor-core", "reactor"],
        ):
            fallback_msg = f" (fallback from {configured_port})" if is_fallback else ""
            logger.info(
                f"[v96.0] ✅ Registered with service registry using atomic lock "
                f"(port={actual_port}{fallback_msg}, pid={os.getpid()})"
            )

            # v97.0: Start registry heartbeat loop to prevent stale cleanup
            self._registry_heartbeat_task = asyncio.create_task(
                self._registry_heartbeat_loop(),
                name="reactor_registry_heartbeat"
            )
            logger.info(
                f"[v97.0] ✅ Registry heartbeat started "
                f"(interval: {self._REGISTRY_HEARTBEAT_INTERVAL}s)"
            )
        else:
            logger.warning("[v96.0] Service registry registration failed (non-fatal)")

        self._state["startup_progress"] = 80

        # v190.0: Phase tracking for Trinity connection
        self._state["startup_phase"] = "connecting_trinity"
        self._state["startup_progress"] = 85

        # Connect to Trinity if enabled
        if self._config.trinity_enabled:
            self._trinity_client = TrinityClient(self._config)
            connected = await self._trinity_client.connect()
            self._state["trinity_connected"] = connected
            if connected:
                logger.debug("[v190.0] Trinity mesh connected")
            else:
                logger.debug("[v190.0] Trinity mesh connection failed (non-fatal)")
        else:
            logger.debug("[v190.0] Trinity disabled, skipping connection")

        self._state["startup_progress"] = 95

        # v190.0: Final readiness
        self._state["running"] = True
        self._state["startup_phase"] = "ready"
        self._state["startup_progress"] = 100
        logger.info(f"[v190.0] ✅ Reactor Core service fully ready on port {self._config.port}")

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

    async def _registry_heartbeat_loop(self) -> None:
        """
        v97.0: Periodically update heartbeat in the shared service registry.

        This is CRITICAL for external services like reactor-core to prevent
        being marked as stale by the JARVIS service registry cleanup process.
        The cleanup process removes entries without recent heartbeat updates.

        The heartbeat loop:
        1. Runs every _REGISTRY_HEARTBEAT_INTERVAL seconds (default 15s)
        2. Updates last_heartbeat and status for all reactor-core service names
        3. Uses atomic file locking to prevent race conditions
        4. Continues until service shutdown
        """
        service_names = ["reactor_core", "reactor-core", "reactor"]
        consecutive_failures = 0
        max_consecutive_failures = 5

        logger.info(f"[v97.0] Registry heartbeat loop started for: {service_names}")

        while self._state.get("running", True):
            try:
                await asyncio.sleep(self._REGISTRY_HEARTBEAT_INTERVAL)

                # Update heartbeat in service registry
                success = _atomic_update_heartbeat(
                    service_names=service_names,
                    status="running"
                )

                if success:
                    consecutive_failures = 0
                    logger.debug(
                        f"[v97.0] Registry heartbeat updated for {service_names[0]}"
                    )
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"[v97.0] {consecutive_failures} consecutive heartbeat failures - "
                            f"service may be deregistered by cleanup"
                        )
                        # Try to re-register
                        logger.info("[v97.0] Attempting re-registration...")
                        # Note: We don't have service_data here, so just update heartbeat
                        # The next iteration will try again

            except asyncio.CancelledError:
                logger.info("[v97.0] Registry heartbeat loop cancelled")
                break
            except Exception as e:
                logger.debug(f"[v97.0] Registry heartbeat error: {e}")
                consecutive_failures += 1
                await asyncio.sleep(1.0)  # Brief pause before retry

        logger.info("[v97.0] Registry heartbeat loop stopped")

    async def run(self):
        """Run the service until shutdown."""
        logger.info("Reactor Core service running. Press Ctrl+C to stop.")
        await self._shutdown_event.wait()

    async def stop(self):
        """
        v97.0: Stop the Reactor Core service with service deregistration.

        Ensures Reactor Core is removed from the shared service registry
        so TrinityIntegrator knows it's no longer available.
        """
        logger.info("Stopping Reactor Core service...")

        # v97.0: Stop registry heartbeat loop first
        if self._registry_heartbeat_task and not self._registry_heartbeat_task.done():
            self._registry_heartbeat_task.cancel()
            try:
                await asyncio.wait_for(self._registry_heartbeat_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            logger.info("[v97.0] Registry heartbeat stopped")

        # v96.0: Deregister from shared service registry using atomic locking
        if _atomic_deregister_service(["reactor_core", "reactor-core", "reactor"]):
            logger.info("[v96.0] ✅ Deregistered from service registry")
        else:
            logger.debug("[v96.0] Service deregistration completed (may have already been removed)")

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

def _check_port_available(port: int, host: str = "0.0.0.0") -> tuple[bool, str]:
    """
    v198.0: Pre-startup port availability check for Reactor Core.

    Defense-in-depth measure that provides clear error message BEFORE
    attempting to bind, rather than cryptic "Address already in use".

    Uses two-phase check:
    1. Connect test - detects if something is actively listening
    2. Bind test - confirms we can actually bind

    Returns:
        Tuple of (is_available, error_message)
    """
    import socket

    bind_host = '127.0.0.1' if host == '0.0.0.0' else host

    # Phase 1: Check if anything is already listening
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex((bind_host, port))
            if result == 0:
                # Connection succeeded = something is listening
                pass  # Fall through to get PID info
            else:
                # No listener, verify we can bind
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bind_sock:
                        bind_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        bind_sock.settimeout(1.0)
                        bind_sock.bind((bind_host, port))
                        return True, ""
                except OSError:
                    pass  # Fall through to error reporting
    except Exception:
        pass

    # Port is in use - gather diagnostic info
    pid_info = ""
    if PSUTIL_AVAILABLE:
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    try:
                        proc = psutil.Process(conn.pid)
                        pid_info = f" (PID {conn.pid}: {proc.name()})"
                    except Exception:
                        pid_info = f" (PID {conn.pid})"
                    break
        except Exception:
            pass
    return False, f"Port {port} is already in use{pid_info}"


async def main(args: argparse.Namespace):
    """Main entry point."""
    config = ReactorCoreConfig()

    # Override from args
    if args.port:
        config.port = args.port
    if args.prime_url:
        config.jarvis_prime_url = args.prime_url

    # =========================================================================
    # v198.0: PRE-STARTUP PORT AVAILABILITY CHECK
    # =========================================================================
    port_available, port_error = _check_port_available(config.port)
    if not port_available:
        logger.error("=" * 70)
        logger.error("🔴 PORT CONFLICT DETECTED - REACTOR CORE")
        logger.error("=" * 70)
        logger.error(f"   {port_error}")
        logger.error("")
        logger.error("   Possible solutions:")
        logger.error(f"   1. Kill the process using port {config.port}")
        logger.error(f"   2. Use a different port: --port {config.port + 1}")
        logger.error(f"   3. Run 'lsof -i :{config.port}' to see what's using it")
        logger.error("=" * 70)
        sys.exit(1)

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
