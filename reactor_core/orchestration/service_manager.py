"""
Ultra-Robust Service Manager for Trinity Architecture
======================================================

Handles the three critical landmines:
1. **Dependency Hell** - Intelligent venv detection per repo
2. **Zombie Processes** - Aggressive process cleanup with signal handling
3. **Race Conditions** - Health-check gating with exponential backoff

Version: v82.0 (Trinity Unification)
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import psutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# VENV DETECTOR - Solves Landmine #1: Dependency Hell
# ============================================================================

class VenvDetector:
    """
    Intelligent virtual environment detector.

    Finds the correct Python executable for each repo to avoid dependency conflicts.

    Detection strategies (in order):
    1. Activated venv in current shell
    2. `.venv` directory in repo root
    3. `venv` directory in repo root
    4. `env` directory in repo root
    5. Poetry virtualenv
    6. Pipenv virtualenv
    7. Conda environment
    8. System Python (fallback)
    """

    @staticmethod
    def detect_venv(repo_path: Path) -> Path:
        """
        Detect the Python executable for a repository.

        Args:
            repo_path: Path to repository root

        Returns:
            Path to Python executable
        """
        logger.debug(f"Detecting venv for {repo_path}")

        # Strategy 1: Check if venv is already activated in current shell
        if "VIRTUAL_ENV" in os.environ:
            venv_path = Path(os.environ["VIRTUAL_ENV"])
            python_exe = VenvDetector._get_python_executable(venv_path)
            if python_exe.exists():
                logger.info(f"Using activated venv: {python_exe}")
                return python_exe

        # Strategy 2-4: Check common venv directory names
        venv_names = [".venv", "venv", "env"]
        for venv_name in venv_names:
            venv_path = repo_path / venv_name
            if venv_path.exists() and venv_path.is_dir():
                python_exe = VenvDetector._get_python_executable(venv_path)
                if python_exe.exists():
                    logger.info(f"Found venv at {venv_name}: {python_exe}")
                    return python_exe

        # Strategy 5: Poetry virtualenv
        poetry_python = VenvDetector._detect_poetry_venv(repo_path)
        if poetry_python:
            logger.info(f"Found Poetry venv: {poetry_python}")
            return poetry_python

        # Strategy 6: Pipenv virtualenv
        pipenv_python = VenvDetector._detect_pipenv_venv(repo_path)
        if pipenv_python:
            logger.info(f"Found Pipenv venv: {pipenv_python}")
            return pipenv_python

        # Strategy 7: Conda environment
        conda_python = VenvDetector._detect_conda_env(repo_path)
        if conda_python:
            logger.info(f"Found Conda env: {conda_python}")
            return conda_python

        # Strategy 8: Fallback to system Python
        system_python = Path(sys.executable)
        logger.warning(
            f"No venv found for {repo_path.name}, using system Python: {system_python}"
        )
        return system_python

    @staticmethod
    def _get_python_executable(venv_path: Path) -> Path:
        """Get Python executable path from venv directory."""
        if platform.system() == "Windows":
            return venv_path / "Scripts" / "python.exe"
        else:
            return venv_path / "bin" / "python"

    @staticmethod
    def _detect_poetry_venv(repo_path: Path) -> Optional[Path]:
        """Detect Poetry virtualenv."""
        try:
            # Check for pyproject.toml
            if not (repo_path / "pyproject.toml").exists():
                return None

            # Run poetry env info to get venv path
            result = subprocess.run(
                ["poetry", "env", "info", "--path"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                venv_path = Path(result.stdout.strip())
                python_exe = VenvDetector._get_python_executable(venv_path)
                if python_exe.exists():
                    return python_exe

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    @staticmethod
    def _detect_pipenv_venv(repo_path: Path) -> Optional[Path]:
        """Detect Pipenv virtualenv."""
        try:
            # Check for Pipfile
            if not (repo_path / "Pipfile").exists():
                return None

            # Run pipenv --venv to get venv path
            result = subprocess.run(
                ["pipenv", "--venv"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                venv_path = Path(result.stdout.strip())
                python_exe = VenvDetector._get_python_executable(venv_path)
                if python_exe.exists():
                    return python_exe

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None

    @staticmethod
    def _detect_conda_env(repo_path: Path) -> Optional[Path]:
        """Detect Conda environment."""
        try:
            # Check for environment.yml
            if not (repo_path / "environment.yml").exists():
                return None

            # Check if conda is available
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Look for environment matching repo name
                repo_name = repo_path.name
                for line in result.stdout.split("\n"):
                    if repo_name in line and not line.startswith("#"):
                        parts = line.split()
                        if len(parts) >= 2:
                            env_path = Path(parts[-1])
                            python_exe = env_path / "bin" / "python"
                            if python_exe.exists():
                                return python_exe

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None


# ============================================================================
# PROCESS MANAGER - Solves Landmine #2: Zombie Processes
# ============================================================================

class ProcessManager:
    """
    Manages subprocess lifecycle with aggressive zombie prevention.

    Features:
    - Async subprocess creation
    - Process group management
    - Signal propagation
    - Graceful + forceful termination
    - Process tree cleanup
    """

    def __init__(self):
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.process_groups: Dict[str, int] = {}  # service_id -> pgid
        self._shutdown_handlers: List[callable] = []

        # Register signal handlers
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Register signal handlers for clean shutdown."""
        loop = asyncio.get_event_loop()

        def shutdown_handler(signum):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown_all())

        # Handle SIGTERM and SIGINT
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: shutdown_handler(s))

        logger.debug("Signal handlers registered")

    async def start_process(
        self,
        service_id: str,
        command: List[str],
        cwd: Path,
        env: Optional[Dict[str, str]] = None,
        stdout_callback: Optional[callable] = None,
    ) -> asyncio.subprocess.Process:
        """
        Start a subprocess with proper isolation.

        Args:
            service_id: Unique service identifier
            command: Command to execute
            cwd: Working directory
            env: Environment variables
            stdout_callback: Optional callback for stdout lines

        Returns:
            Started subprocess
        """
        logger.info(f"Starting process for '{service_id}': {' '.join(command)}")

        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Create process in new process group for better control
        # This prevents signals from parent affecting the child
        if platform.system() != "Windows":
            # Unix: Use preexec_fn to create new process group
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=os.setsid,  # Create new session
            )

            # Store process group ID
            self.process_groups[service_id] = os.getpgid(process.pid)
        else:
            # Windows: CREATE_NEW_PROCESS_GROUP
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )

        # Store process
        self.processes[service_id] = process

        # Start stdout/stderr readers
        if stdout_callback:
            asyncio.create_task(self._read_stream(process.stdout, stdout_callback))
            asyncio.create_task(self._read_stream(process.stderr, stdout_callback))

        logger.info(f"Process '{service_id}' started with PID {process.pid}")
        return process

    async def _read_stream(self, stream, callback):
        """Read from stream and call callback for each line."""
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                await callback(line.decode().rstrip())
        except Exception as e:
            logger.error(f"Error reading stream: {e}")

    async def stop_process(
        self,
        service_id: str,
        graceful_timeout: float = 10.0,
    ) -> bool:
        """
        Stop a process gracefully, then forcefully if needed.

        Args:
            service_id: Service identifier
            graceful_timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully
        """
        process = self.processes.get(service_id)
        if not process:
            logger.warning(f"Process '{service_id}' not found")
            return False

        logger.info(f"Stopping process '{service_id}' (PID {process.pid})...")

        try:
            # Step 1: Send SIGTERM for graceful shutdown
            process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(process.wait(), timeout=graceful_timeout)
                logger.info(f"Process '{service_id}' terminated gracefully")
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    f"Process '{service_id}' did not terminate gracefully, "
                    f"sending SIGKILL..."
                )

            # Step 2: Force kill if graceful failed
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=5.0)
            logger.info(f"Process '{service_id}' killed forcefully")

            # Step 3: Clean up entire process group (zombie prevention)
            if service_id in self.process_groups:
                await self._kill_process_group(self.process_groups[service_id])

            return True

        except Exception as e:
            logger.error(f"Error stopping process '{service_id}': {e}")
            return False
        finally:
            # Cleanup
            self.processes.pop(service_id, None)
            self.process_groups.pop(service_id, None)

    async def _kill_process_group(self, pgid: int):
        """Kill entire process group to prevent zombies."""
        if platform.system() == "Windows":
            # Windows doesn't have process groups the same way
            return

        try:
            # Send SIGKILL to entire process group
            os.killpg(pgid, signal.SIGKILL)
            logger.debug(f"Killed process group {pgid}")

            # Give processes time to die
            await asyncio.sleep(0.5)

            # Verify all processes in group are dead
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if os.getpgid(proc.pid) == pgid:
                        logger.warning(f"Zombie process detected: {proc.pid} ({proc.name})")
                        proc.kill()
                except (ProcessLookupError, psutil.NoSuchProcess):
                    pass

        except ProcessLookupError:
            # Process group already dead
            pass
        except Exception as e:
            logger.error(f"Error killing process group {pgid}: {e}")

    async def shutdown_all(self):
        """Shutdown all managed processes."""
        logger.info("Shutting down all managed processes...")

        # Stop all processes in parallel
        tasks = [
            self.stop_process(service_id)
            for service_id in list(self.processes.keys())
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("All processes stopped")

    def is_running(self, service_id: str) -> bool:
        """Check if process is running."""
        process = self.processes.get(service_id)
        if not process:
            return False

        return process.returncode is None


# ============================================================================
# HEALTH CHECKER - Solves Landmine #3: Race Conditions
# ============================================================================

@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    url: str
    timeout: float = 5.0
    max_retries: int = 30
    retry_delay: float = 2.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 1.5
    max_backoff: float = 30.0


class HealthChecker:
    """
    Robust health checker with exponential backoff.

    Prevents race conditions by ensuring services are fully ready before
    allowing dependent services to start.
    """

    def __init__(self):
        self._check_cache: Dict[str, Tuple[bool, float]] = {}  # url -> (healthy, timestamp)
        self._cache_ttl = 5.0  # Cache health status for 5 seconds

    async def wait_for_healthy(
        self,
        config: HealthCheckConfig,
        service_name: str = "service",
    ) -> bool:
        """
        Wait for service to become healthy with exponential backoff.

        Args:
            config: Health check configuration
            service_name: Service name for logging

        Returns:
            True if service became healthy, False if max retries exceeded
        """
        logger.info(f"Waiting for '{service_name}' to become healthy...")
        logger.info(f"  Health URL: {config.url}")
        logger.info(f"  Max retries: {config.max_retries}")

        retry_delay = config.retry_delay

        for attempt in range(1, config.max_retries + 1):
            # Check if service is healthy
            is_healthy = await self._check_health(config.url, config.timeout)

            if is_healthy:
                logger.info(
                    f"✅ '{service_name}' is healthy (attempt {attempt}/{config.max_retries})"
                )
                return True

            # Not healthy yet, calculate next retry delay
            if attempt < config.max_retries:
                if config.exponential_backoff:
                    retry_delay = min(
                        retry_delay * config.backoff_multiplier,
                        config.max_backoff,
                    )

                logger.debug(
                    f"⏳ '{service_name}' not ready yet "
                    f"(attempt {attempt}/{config.max_retries}), "
                    f"retrying in {retry_delay:.1f}s..."
                )

                await asyncio.sleep(retry_delay)

        logger.error(
            f"❌ '{service_name}' failed to become healthy after "
            f"{config.max_retries} attempts"
        )
        return False

    async def _check_health(self, url: str, timeout: float) -> bool:
        """
        Check if a service is healthy.

        Args:
            url: Health check URL
            timeout: Request timeout

        Returns:
            True if service returned 200 OK
        """
        # Check cache first
        if url in self._check_cache:
            cached_healthy, cached_time = self._check_cache[url]
            if time.time() - cached_time < self._cache_ttl:
                return cached_healthy

        try:
            # Import aiohttp only when needed
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    is_healthy = response.status == 200

                    # Cache result
                    self._check_cache[url] = (is_healthy, time.time())

                    return is_healthy

        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            self._check_cache[url] = (False, time.time())
            return False


# ============================================================================
# SERVICE MANAGER - Brings It All Together
# ============================================================================

class ServiceStatus(Enum):
    """Service status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    service_id: str
    repo_path: Path
    start_command: List[str]  # Command without python executable (will be prepended)
    health_check: Optional[HealthCheckConfig] = None
    dependencies: List[str] = field(default_factory=list)  # Services that must start first
    env: Dict[str, str] = field(default_factory=dict)


class ServiceManager:
    """
    Ultra-robust service manager for Trinity architecture.

    Solves all three landmines:
    1. Venv detection for dependency isolation
    2. Zombie process prevention
    3. Health-check gating for race condition prevention
    """

    def __init__(self):
        self.venv_detector = VenvDetector()
        self.process_manager = ProcessManager()
        self.health_checker = HealthChecker()

        self.services: Dict[str, ServiceConfig] = {}
        self.statuses: Dict[str, ServiceStatus] = {}
        self.python_executables: Dict[str, Path] = {}

    async def register_service(self, config: ServiceConfig):
        """Register a service."""
        logger.info(f"Registering service '{config.service_id}'")

        # Detect venv
        python_exe = self.venv_detector.detect_venv(config.repo_path)
        self.python_executables[config.service_id] = python_exe

        # Store config
        self.services[config.service_id] = config
        self.statuses[config.service_id] = ServiceStatus.STOPPED

        logger.info(f"  Python: {python_exe}")
        logger.info(f"  Command: {' '.join(config.start_command)}")

    async def start_service(self, service_id: str) -> bool:
        """
        Start a service with dependency checking and health gating.

        Args:
            service_id: Service to start

        Returns:
            True if started successfully
        """
        config = self.services.get(service_id)
        if not config:
            logger.error(f"Service '{service_id}' not registered")
            return False

        if self.statuses[service_id] == ServiceStatus.RUNNING:
            logger.info(f"Service '{service_id}' already running")
            return True

        logger.info(f"Starting service '{service_id}'...")
        self.statuses[service_id] = ServiceStatus.STARTING

        try:
            # Step 1: Start dependencies first
            for dep_id in config.dependencies:
                if not await self.start_service(dep_id):
                    logger.error(f"Failed to start dependency '{dep_id}'")
                    self.statuses[service_id] = ServiceStatus.FAILED
                    return False

            # Step 2: Prepare command with correct Python executable
            python_exe = self.python_executables[service_id]
            full_command = [str(python_exe)] + config.start_command

            # Step 3: Start process
            def stdout_callback(line):
                logger.info(f"[{service_id}] {line}")

            await self.process_manager.start_process(
                service_id=service_id,
                command=full_command,
                cwd=config.repo_path,
                env=config.env,
                stdout_callback=lambda line: asyncio.create_task(asyncio.coroutine(lambda: stdout_callback(line))()),
            )

            # Step 4: Wait for health check (if configured)
            if config.health_check:
                is_healthy = await self.health_checker.wait_for_healthy(
                    config=config.health_check,
                    service_name=service_id,
                )

                if not is_healthy:
                    logger.error(f"Service '{service_id}' failed health check")
                    await self.stop_service(service_id)
                    self.statuses[service_id] = ServiceStatus.FAILED
                    return False

            self.statuses[service_id] = ServiceStatus.RUNNING
            logger.info(f"✅ Service '{service_id}' started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting service '{service_id}': {e}")
            self.statuses[service_id] = ServiceStatus.FAILED
            return False

    async def stop_service(self, service_id: str) -> bool:
        """Stop a service."""
        if service_id not in self.services:
            logger.error(f"Service '{service_id}' not registered")
            return False

        logger.info(f"Stopping service '{service_id}'...")
        self.statuses[service_id] = ServiceStatus.STOPPING

        success = await self.process_manager.stop_process(service_id)

        if success:
            self.statuses[service_id] = ServiceStatus.STOPPED
            logger.info(f"Service '{service_id}' stopped")
        else:
            self.statuses[service_id] = ServiceStatus.FAILED
            logger.error(f"Failed to stop service '{service_id}'")

        return success

    async def stop_all(self):
        """Stop all services."""
        await self.process_manager.shutdown_all()

        for service_id in self.services:
            self.statuses[service_id] = ServiceStatus.STOPPED


__all__ = [
    # Venv Detection
    "VenvDetector",
    # Process Management
    "ProcessManager",
    # Health Checking
    "HealthChecker",
    "HealthCheckConfig",
    # Service Management
    "ServiceManager",
    "ServiceConfig",
    "ServiceStatus",
]
