#!/usr/bin/env python3
"""
AGI OS Unified Supervisor - Project Trinity v92.0
==================================================

The central command that orchestrates the entire AGI ecosystem:
- JARVIS (Body) - macOS automation and interaction
- JARVIS Prime (Mind) - Local LLM inference and reasoning
- Reactor Core (Nervous System) - Training, learning, and model serving

RUN: python3 run_supervisor.py

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  AGI OS UNIFIED SUPERVISOR v92.0                   │
    │                    (Central Coordination Hub)                       │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
    ┌─────────┐               ┌─────────────┐               ┌─────────┐
    │ JARVIS  │◄────────────►│   TRINITY   │◄────────────►│ J-PRIME │
    │ (Body)  │     Events    │ ORCHESTRATOR│     Events    │ (Mind)  │
    │         │               │             │               │         │
    │ macOS   │               │ Heartbeats  │               │ LLM     │
    │ Actions │               │ Commands    │               │ Inference
    └─────────┘               │ State Sync  │               └─────────┘
                              └─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │REACTOR CORE │   │  ONLINE     │   │ DISTRIBUTED │
            │  (Nerves)   │   │  LEARNING   │   │  TRAINING   │
            │             │   │             │   │             │
            │ Training    │   │ Experience  │   │ Multi-VM    │
            │ Learning    │   │ Replay      │   │ Gradient    │
            │ Serving     │   │ EWC/Drift   │   │ Sync        │
            └─────────────┘   └─────────────┘   └─────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                              ┌─────────────┐
                              │   MLForge   │
                              │ C++ Bindings│
                              │ + GCP Spot  │
                              │ + Versioning│
                              └─────────────┘

v77.0 FEATURES:
- One-command startup for entire AGI ecosystem
- Automatic component discovery and health monitoring
- Graceful startup/shutdown sequence
- Real-time event streaming between components
- Continuous learning from JARVIS interactions
- Model serving for J-Prime inference
- Fault tolerance with automatic recovery

v91.0 ADVANCED FEATURES:
- Online/Incremental Learning: Prioritized experience replay with importance sampling
- Elastic Weight Consolidation (EWC): Prevents catastrophic forgetting during updates
- Concept Drift Detection: Page-Hinkley test for automatic model adaptation
- Data Versioning: Content-addressed storage with lineage tracking (DVC compatible)
- GCP Spot VM Checkpointing: Predictive preemption with multi-signal detection
- Distributed Training: Multi-VM coordination with gradient compression
- Dynamic Resource Allocation: Auto-scaling with cost-aware decisions
- MLForge C++ Bindings: High-performance matrix/neural ops with pybind11

v92.0 RELIABILITY FEATURES:
- Atomic File Writes: Prevents checkpoint corruption from partial writes
- Circuit Breaker Pattern: Protects external service calls with auto-recovery
- Backpressure Control: Prevents memory exhaustion under high load
- Proper Async Patterns: Deadlock-free async/await with timeouts
- Gradient Verification: Checksum validation for distributed training
- Memory Pressure Awareness: Adaptive behavior under resource constraints
- Unified Error Handling: Centralized error classification and routing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Add reactor_core to path
sys.path.insert(0, str(Path(__file__).parent))

from reactor_core.orchestration.trinity_orchestrator import (
    TrinityOrchestrator,
    ComponentType,
    ComponentHealth,
    initialize_orchestrator,
    shutdown_orchestrator,
    get_orchestrator,
)
from reactor_core.integration.event_bridge import (
    EventBridge,
    EventSource,
    EventType,
    create_event_bridge,
)

logger = logging.getLogger(__name__)

# v91.0 Advanced Module Imports
try:
    from reactor_core.training.online_learning import (
        OnlineLearningEngine,
        PrioritizedExperienceBuffer,
        FeedbackIntegrator,
        DriftDetector,
    )
    HAS_ONLINE_LEARNING = True
except ImportError as e:
    logger.debug(f"Online learning module not available: {e}")
    HAS_ONLINE_LEARNING = False

try:
    from reactor_core.training.distributed_coordinator import (
        DistributedCoordinator,
        DynamicResourceAllocator,
        WorkerManager,
        AutoScaler,
    )
    HAS_DISTRIBUTED_TRAINING = True
except ImportError as e:
    logger.debug(f"Distributed training module not available: {e}")
    HAS_DISTRIBUTED_TRAINING = False

try:
    from reactor_core.data.versioning import (
        DataVersionController,
        VersionStore,
        LineageTracker,
    )
    HAS_DATA_VERSIONING = True
except ImportError as e:
    logger.debug(f"Data versioning module not available: {e}")
    HAS_DATA_VERSIONING = False

try:
    from reactor_core.gcp.checkpointer import (
        SpotVMCheckpointer,
        PreemptionDetector,
        spot_vm_training_context,
    )
    HAS_SPOT_VM_CHECKPOINTING = True
except ImportError as e:
    logger.debug(f"Spot VM checkpointing module not available: {e}")
    HAS_SPOT_VM_CHECKPOINTING = False

try:
    from reactor_core.bindings import (
        MLForgeBridge,
        get_bridge,
        has_cpp_backend,
    )
    HAS_MLFORGE_BINDINGS = True
except ImportError as e:
    logger.debug(f"MLForge bindings not available: {e}")
    HAS_MLFORGE_BINDINGS = False

# v92.0: Unified Error Handling
try:
    from reactor_core.core.error_handling import (
        CircuitBreaker,
        CircuitBreakerConfig,
        ErrorRegistry,
        RetryHandler,
        RetryConfig,
        Bulkhead,
        resilient,
        get_error_registry,
        record_error,
        ErrorCategory,
        ClassifiedError,
    )
    HAS_ERROR_HANDLING = True
except ImportError as e:
    logger.debug(f"Error handling module not available: {e}")
    HAS_ERROR_HANDLING = False

# v77.0 API Integration
try:
    from reactor_core.api.telemetry import get_telemetry, TelemetryCollector
    from reactor_core.api.scheduler import get_scheduler, init_scheduler, ScheduleTemplates
    from reactor_core.api.model_registry import get_registry, ModelRegistry
    from reactor_core.api.health_aggregator import get_health_aggregator, init_health_aggregator
    from reactor_core.serving.model_server import get_model_server, ModelServer, ModelServerConfig
    HAS_V77_API = True
except ImportError as e:
    logger.warning(f"v77.0 API modules not available: {e}")
    HAS_V77_API = False


# =============================================================================
# ROBUST CROSS-REPO DISCOVERY (v76.0 Enhancement)
# =============================================================================

class RepoDiscovery:
    """
    Robust cross-repo discovery for AGI OS ecosystem.

    Features:
    - Multi-strategy repo detection (env vars, config files, git remotes)
    - Symlink resolution
    - Validation of repo contents
    - Caching of discovered paths
    - Platform-aware search paths
    """

    # Known repo names and their identifying files
    REPO_SIGNATURES = {
        "JARVIS-AI-Agent": ["jarvis/main.py", "jarvis/__init__.py", "setup.py"],
        "jarvis-prime": ["jarvis_prime/main.py", "jprime", "inference.py"],
        "reactor-core": ["reactor_core/__init__.py", "run_supervisor.py"],
    }

    # Environment variable mappings
    ENV_VAR_MAPPINGS = {
        "JARVIS-AI-Agent": ["JARVIS_PATH", "JARVIS_HOME", "JARVIS_DIR"],
        "jarvis-prime": ["JPRIME_PATH", "JARVIS_PRIME_PATH", "JPRIME_HOME"],
        "reactor-core": ["REACTOR_CORE_PATH", "REACTOR_PATH"],
    }

    # Common project directories by platform
    PROJECT_DIRS = {
        "darwin": [
            Path.home() / "Projects",
            Path.home() / "Developer",
            Path.home() / "dev",
            Path.home() / "code",
            Path.home() / "repos",
            Path.home() / "git",
            Path.home() / "src",
        ],
        "linux": [
            Path.home() / "projects",
            Path.home() / "dev",
            Path.home() / "code",
            Path.home() / "repos",
            Path.home() / "git",
            Path.home() / "src",
            Path("/opt"),
        ],
    }

    def __init__(self):
        self._cache: Dict[str, Path] = {}
        self._platform = sys.platform
        self._search_history: List[str] = []

    def find_repo(self, name: str, required: bool = False) -> Optional[Path]:
        """
        Find a repository by name using multiple strategies.

        Args:
            name: Repository name
            required: If True, raise error if not found

        Returns:
            Path to repository or None
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Strategy 1: Environment variables
        path = self._find_via_env_var(name)
        if path:
            self._cache[name] = path
            logger.info(f"Found {name} via environment variable: {path}")
            return path

        # Strategy 2: Config file
        path = self._find_via_config_file(name)
        if path:
            self._cache[name] = path
            logger.info(f"Found {name} via config file: {path}")
            return path

        # Strategy 3: Sibling directory (relative to current script)
        path = self._find_sibling_repo(name)
        if path:
            self._cache[name] = path
            logger.info(f"Found {name} as sibling directory: {path}")
            return path

        # Strategy 4: Search common project directories
        path = self._find_in_project_dirs(name)
        if path:
            self._cache[name] = path
            logger.info(f"Found {name} in project directories: {path}")
            return path

        # Strategy 5: Git remote search (if in a git repo)
        path = self._find_via_git_remote(name)
        if path:
            self._cache[name] = path
            logger.info(f"Found {name} via git remote reference: {path}")
            return path

        # Strategy 6: Home directory scan (last resort)
        path = self._scan_home_directory(name)
        if path:
            self._cache[name] = path
            logger.info(f"Found {name} via home directory scan: {path}")
            return path

        if required:
            raise FileNotFoundError(
                f"Required repository '{name}' not found. "
                f"Set {self.ENV_VAR_MAPPINGS.get(name, ['<repo>_PATH'])[0]} environment variable."
            )

        logger.warning(f"Repository '{name}' not found")
        return None

    def _find_via_env_var(self, name: str) -> Optional[Path]:
        """Find repo via environment variable."""
        env_vars = self.ENV_VAR_MAPPINGS.get(name, [])

        for var in env_vars:
            value = os.environ.get(var)
            if value:
                path = Path(value).resolve()
                if self._validate_repo(path, name):
                    return path

        return None

    def _find_via_config_file(self, name: str) -> Optional[Path]:
        """Find repo via AGI OS config file."""
        config_paths = [
            Path.home() / ".jarvis" / "agi_config.json",
            Path.home() / ".config" / "agi-os" / "config.json",
            Path.home() / ".agi-os.json",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    # Check various config key formats
                    repo_path = (
                        config.get("repos", {}).get(name) or
                        config.get("paths", {}).get(name) or
                        config.get(f"{name.lower().replace('-', '_')}_path")
                    )

                    if repo_path:
                        path = Path(repo_path).resolve()
                        if self._validate_repo(path, name):
                            return path

                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

        return None

    def _find_sibling_repo(self, name: str) -> Optional[Path]:
        """Find repo as sibling directory."""
        script_dir = Path(__file__).parent.resolve()
        parent_dir = script_dir.parent

        # Check direct sibling
        sibling = parent_dir / name
        if self._validate_repo(sibling, name):
            return sibling

        # Check with variations
        variations = [
            name,
            name.lower(),
            name.replace("-", "_"),
            name.replace("_", "-"),
        ]

        for variation in variations:
            sibling = parent_dir / variation
            if self._validate_repo(sibling, name):
                return sibling

        return None

    def _find_in_project_dirs(self, name: str) -> Optional[Path]:
        """Find repo in common project directories."""
        platform_key = "darwin" if self._platform == "darwin" else "linux"
        project_dirs = self.PROJECT_DIRS.get(platform_key, [])

        variations = [
            name,
            name.lower(),
            name.replace("-", "_"),
            name.replace("_", "-"),
        ]

        for project_dir in project_dirs:
            if not project_dir.exists():
                continue

            for variation in variations:
                repo_path = project_dir / variation
                if self._validate_repo(repo_path, name):
                    return repo_path

        return None

    def _find_via_git_remote(self, name: str) -> Optional[Path]:
        """Find repo via git remote configuration."""
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return None

            git_root = Path(result.stdout.strip())
            parent_dir = git_root.parent

            # Check siblings
            for sibling in parent_dir.iterdir():
                if sibling.is_dir() and self._validate_repo(sibling, name):
                    return sibling

        except (subprocess.SubprocessError, OSError):
            pass

        return None

    def _scan_home_directory(self, name: str, max_depth: int = 3) -> Optional[Path]:
        """
        Scan home directory for repo (limited depth).

        This is a last resort and is intentionally limited.
        """
        home = Path.home()
        variations = [
            name,
            name.lower(),
            name.replace("-", "_"),
            name.replace("_", "-"),
        ]

        def scan_dir(directory: Path, depth: int) -> Optional[Path]:
            if depth > max_depth:
                return None

            try:
                for item in directory.iterdir():
                    if not item.is_dir():
                        continue

                    # Skip hidden directories and common non-project dirs
                    if item.name.startswith("."):
                        continue
                    if item.name in ("Library", "Applications", "Documents", "Downloads"):
                        continue

                    # Check if this is the repo
                    if item.name in variations:
                        if self._validate_repo(item, name):
                            return item

                    # Recurse (but not too deep)
                    result = scan_dir(item, depth + 1)
                    if result:
                        return result

            except PermissionError:
                pass

            return None

        return scan_dir(home, 0)

    def _validate_repo(self, path: Path, name: str) -> bool:
        """Validate that path is a valid repo with expected contents."""
        if not path.exists():
            return False

        # Resolve symlinks
        path = path.resolve()

        # Check for signature files
        signatures = self.REPO_SIGNATURES.get(name, [])
        if not signatures:
            # No signatures known, just check if it's a directory with py files
            return path.is_dir() and any(path.glob("*.py"))

        # Check if any signature file exists
        for sig in signatures:
            if (path / sig).exists():
                return True

        # Also check for setup.py or pyproject.toml
        if (path / "setup.py").exists() or (path / "pyproject.toml").exists():
            return True

        return False

    def get_all_repos(self) -> Dict[str, Optional[Path]]:
        """Find all known AGI OS repositories."""
        return {
            name: self.find_repo(name)
            for name in self.REPO_SIGNATURES.keys()
        }

    def get_discovery_status(self) -> Dict[str, Any]:
        """Get status of repo discovery."""
        repos = self.get_all_repos()
        return {
            "discovered": {name: str(path) for name, path in repos.items() if path},
            "missing": [name for name, path in repos.items() if not path],
            "cached": list(self._cache.keys()),
            "platform": self._platform,
        }


class ComponentHealthChecker:
    """
    Advanced health checking for AGI OS components.

    Features:
    - Multiple health check strategies
    - Configurable thresholds
    - Health history tracking
    - Automatic recovery triggers
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        consecutive_failures_threshold: int = 3,
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.consecutive_failures_threshold = consecutive_failures_threshold

        self._health_history: Dict[str, List[bool]] = {}
        self._last_check_time: Dict[str, float] = {}
        self._failure_counts: Dict[str, int] = {}

    async def check_process_health(
        self,
        component_name: str,
        process: Optional[subprocess.Popen],
    ) -> Tuple[bool, str]:
        """Check if a process is healthy."""
        if process is None:
            return False, "No process"

        poll_result = process.poll()
        if poll_result is not None:
            return False, f"Process exited with code {poll_result}"

        return True, "Process running"

    async def check_heartbeat_health(
        self,
        component_name: str,
        last_heartbeat: float,
    ) -> Tuple[bool, str]:
        """Check if heartbeat is recent."""
        now = time.time()
        elapsed = now - last_heartbeat

        if last_heartbeat == 0:
            return False, "No heartbeat received"

        if elapsed > self.heartbeat_timeout:
            return False, f"Heartbeat timeout ({elapsed:.1f}s > {self.heartbeat_timeout}s)"

        return True, f"Heartbeat OK ({elapsed:.1f}s ago)"

    async def check_port_health(
        self,
        component_name: str,
        port: int,
        host: str = "127.0.0.1",
    ) -> Tuple[bool, str]:
        """Check if a service is responding on a port."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                result = s.connect_ex((host, port))
                if result == 0:
                    return True, f"Port {port} responding"
                return False, f"Port {port} not responding"
        except Exception as e:
            return False, f"Port check error: {e}"

    def record_health(self, component_name: str, is_healthy: bool) -> None:
        """Record health check result."""
        if component_name not in self._health_history:
            self._health_history[component_name] = []

        self._health_history[component_name].append(is_healthy)
        if len(self._health_history[component_name]) > 100:
            self._health_history[component_name] = self._health_history[component_name][-100:]

        if not is_healthy:
            self._failure_counts[component_name] = self._failure_counts.get(component_name, 0) + 1
        else:
            self._failure_counts[component_name] = 0

        self._last_check_time[component_name] = time.time()

    def should_restart(self, component_name: str) -> bool:
        """Check if component should be restarted."""
        failures = self._failure_counts.get(component_name, 0)
        return failures >= self.consecutive_failures_threshold

    def get_health_summary(self, component_name: str) -> Dict[str, Any]:
        """Get health summary for a component."""
        history = self._health_history.get(component_name, [])
        return {
            "recent_checks": len(history),
            "success_rate": sum(history) / len(history) if history else 0,
            "consecutive_failures": self._failure_counts.get(component_name, 0),
            "last_check": self._last_check_time.get(component_name, 0),
            "should_restart": self.should_restart(component_name),
        }


class SelfHealer:
    """
    Self-healing capabilities for AGI OS components.

    Features:
    - Automatic restart on failure
    - Exponential backoff
    - Max restart limits
    - Recovery validation
    """

    def __init__(
        self,
        max_restarts: int = 5,
        base_backoff: float = 5.0,
        max_backoff: float = 300.0,
        backoff_multiplier: float = 2.0,
    ):
        self.max_restarts = max_restarts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier

        self._restart_counts: Dict[str, int] = {}
        self._last_restart_time: Dict[str, float] = {}

    def should_restart(self, component_name: str) -> Tuple[bool, float]:
        """
        Check if component should be restarted.

        Returns:
            Tuple of (should_restart, delay_seconds)
        """
        count = self._restart_counts.get(component_name, 0)

        if count >= self.max_restarts:
            logger.error(f"Max restarts ({self.max_restarts}) exceeded for {component_name}")
            return False, 0

        delay = min(
            self.base_backoff * (self.backoff_multiplier ** count),
            self.max_backoff
        )

        return True, delay

    def record_restart(self, component_name: str) -> None:
        """Record a restart."""
        self._restart_counts[component_name] = self._restart_counts.get(component_name, 0) + 1
        self._last_restart_time[component_name] = time.time()
        logger.info(f"Restart recorded for {component_name} (count: {self._restart_counts[component_name]})")

    def record_recovery(self, component_name: str) -> None:
        """Record successful recovery (resets restart count)."""
        self._restart_counts[component_name] = 0
        logger.info(f"Recovery recorded for {component_name}, restart count reset")

    def get_status(self, component_name: str) -> Dict[str, Any]:
        """Get self-healing status for a component."""
        return {
            "restart_count": self._restart_counts.get(component_name, 0),
            "max_restarts": self.max_restarts,
            "last_restart": self._last_restart_time.get(component_name, 0),
            "can_restart": self._restart_counts.get(component_name, 0) < self.max_restarts,
        }


# Global instance
_repo_discovery: Optional[RepoDiscovery] = None


def get_repo_discovery() -> RepoDiscovery:
    """Get global repo discovery instance."""
    global _repo_discovery
    if _repo_discovery is None:
        _repo_discovery = RepoDiscovery()
    return _repo_discovery


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SupervisorConfig:
    """Configuration for AGI Supervisor."""
    # Component paths (auto-detected or manual)
    jarvis_path: Optional[Path] = None
    jprime_path: Optional[Path] = None
    reactor_core_path: Optional[Path] = None

    # Component enable flags
    enable_jarvis: bool = True
    enable_jprime: bool = True
    enable_reactor_core: bool = True

    # Services to start
    enable_training: bool = True
    enable_serving: bool = True
    enable_scout: bool = False
    enable_api: bool = True

    # Network ports
    api_port: int = 8003
    serving_port: int = 8001
    jprime_port: int = 8000

    # Trinity coordination
    trinity_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")
    heartbeat_interval: float = 5.0
    health_check_interval: float = 10.0

    # Continuous learning
    experience_collection: bool = True
    auto_training_threshold: int = 1000  # Train after N experiences

    # Startup behavior
    startup_timeout: float = 60.0  # Seconds to wait for components
    retry_on_failure: bool = True
    max_retries: int = 3

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    # v91.0 Advanced Features
    enable_online_learning: bool = True
    enable_distributed_training: bool = True
    enable_data_versioning: bool = True
    enable_spot_vm_checkpointing: bool = True
    enable_mlforge_bindings: bool = True

    # Online Learning Settings
    experience_buffer_size: int = 100000
    priority_alpha: float = 0.6
    ewc_lambda: float = 100.0
    drift_threshold: float = 0.1

    # Distributed Training Settings
    distributed_mode: str = "auto"  # auto, single, multi
    min_workers: int = 1
    max_workers: int = 8
    gradient_compression: bool = True

    # Data Versioning Settings
    version_store_path: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "data_versions")
    track_lineage: bool = True
    auto_detect_drift: bool = True

    # GCP Spot VM Settings
    gcp_project: Optional[str] = None
    checkpoint_bucket: Optional[str] = None
    preemption_prediction: bool = True
    checkpoint_interval: int = 300  # seconds

    # MLForge Settings
    mlforge_backend: str = "auto"  # auto, cpp, numpy, torch
    enable_cpp_optimizations: bool = True

    def __post_init__(self):
        # Use robust repo discovery (v76.0)
        discovery = get_repo_discovery()

        if self.jarvis_path is None:
            self.jarvis_path = discovery.find_repo("JARVIS-AI-Agent")
        if self.jprime_path is None:
            self.jprime_path = discovery.find_repo("jarvis-prime")
        if self.reactor_core_path is None:
            self.reactor_core_path = Path(__file__).parent

        # Log discovery status
        status = discovery.get_discovery_status()
        logger.info(f"Repo discovery status: {status}")

    @staticmethod
    def from_environment() -> "SupervisorConfig":
        """Create config from environment variables."""
        return SupervisorConfig(
            jarvis_path=Path(os.environ["JARVIS_PATH"]) if "JARVIS_PATH" in os.environ else None,
            jprime_path=Path(os.environ["JPRIME_PATH"]) if "JPRIME_PATH" in os.environ else None,
            api_port=int(os.environ.get("AGI_API_PORT", 8003)),
            serving_port=int(os.environ.get("AGI_SERVING_PORT", 8001)),
            log_level=os.environ.get("AGI_LOG_LEVEL", "INFO"),
        )


class ComponentStatus(Enum):
    """Status of a managed component."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ManagedComponent:
    """A component managed by the supervisor."""
    name: str
    component_type: ComponentType
    path: Optional[Path] = None
    process: Optional[subprocess.Popen] = None
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_heartbeat: float = 0.0
    start_time: float = 0.0
    restart_count: int = 0
    error_message: str = ""

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        if self.status not in (ComponentStatus.RUNNING, ComponentStatus.DEGRADED):
            return False
        if self.process and self.process.poll() is not None:
            return False
        return True


# =============================================================================
# AGI SUPERVISOR - Main Controller
# =============================================================================

class AGISupervisor:
    """
    Central AGI Supervisor orchestrating all components.

    Manages:
    - Component lifecycle (start, stop, restart)
    - Health monitoring and recovery
    - Event routing between components
    - Continuous learning pipeline
    - Model serving coordination
    """

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Components
        self._components: Dict[str, ManagedComponent] = {}

        # Trinity orchestrator
        self._orchestrator: Optional[TrinityOrchestrator] = None

        # Event bridge
        self._event_bridge: Optional[EventBridge] = None

        # v77.0 API Services
        self._telemetry: Optional[TelemetryCollector] = None
        self._scheduler = None
        self._model_registry: Optional[ModelRegistry] = None
        self._health_aggregator = None
        self._model_server: Optional[ModelServer] = None

        # v91.0 Advanced Services
        self._online_learning_engine: Optional[OnlineLearningEngine] = None
        self._distributed_coordinator: Optional[DistributedCoordinator] = None
        self._data_version_controller: Optional[DataVersionController] = None
        self._spot_vm_checkpointer: Optional[SpotVMCheckpointer] = None
        self._mlforge_bridge: Optional[MLForgeBridge] = None
        self._feedback_integrator: Optional[FeedbackIntegrator] = None
        self._drift_detector: Optional[DriftDetector] = None
        self._resource_allocator: Optional[DynamicResourceAllocator] = None

        # v101.0 Trinity Experience Receiver (closes the Trinity Loop)
        self._experience_receiver = None

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # Health checker and self-healer
        self._health_checker = ComponentHealthChecker()
        self._self_healer = SelfHealer()

        # Statistics
        self._start_time = 0.0
        self._stats = {
            "events_processed": 0,
            "experiences_collected": 0,
            "trainings_triggered": 0,
            "restarts": 0,
            "models_served": 0,
            "api_requests": 0,
            # v91.0 stats
            "online_updates": 0,
            "drift_detections": 0,
            "checkpoints_saved": 0,
            "preemptions_survived": 0,
            "distributed_syncs": 0,
            "data_versions_created": 0,
        }

        # Initialize logging
        self._setup_logging()

        logger.info("AGI Supervisor initialized")

    def _setup_logging(self) -> None:
        """Configure logging."""
        log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        handlers = [logging.StreamHandler(sys.stdout)]

        if self.config.log_file:
            handlers.append(logging.FileHandler(self.config.log_file))

        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )

    async def start(self) -> bool:
        """Start the AGI Supervisor and all components."""
        logger.info("=" * 70)
        logger.info("           AGI OS UNIFIED SUPERVISOR - PROJECT TRINITY")
        logger.info("=" * 70)
        logger.info("")

        self._start_time = time.time()
        self._running = True

        try:
            # Phase 1: Initialize Trinity Orchestrator
            logger.info("[Phase 1] Initializing Trinity Orchestrator...")
            self._orchestrator = await initialize_orchestrator()
            if not self._orchestrator.is_running():
                raise RuntimeError("Failed to start Trinity Orchestrator")
            logger.info("[OK] Trinity Orchestrator running")

            # Phase 2: Initialize Event Bridge
            logger.info("[Phase 2] Initializing Event Bridge...")
            self._event_bridge = create_event_bridge(
                source=EventSource.REACTOR_CORE,
                events_dir=self.config.trinity_dir / "events",
            )
            await self._event_bridge.start()
            logger.info("[OK] Event Bridge running")

            # Phase 3: Discover and register components
            logger.info("[Phase 3] Discovering components...")
            await self._discover_components()

            # Phase 4: Start Reactor Core services
            logger.info("[Phase 4] Starting Reactor Core services...")
            await self._start_reactor_core_services()

            # Phase 5: Initialize v91.0 Advanced Services
            logger.info("[Phase 5] Initializing v91.0 Advanced Services...")
            await self._initialize_v91_services()

            # Phase 6: Start external components
            if self.config.enable_jarvis and self._components.get("jarvis"):
                logger.info("[Phase 6] Starting JARVIS (Body)...")
                await self._start_component("jarvis")

            if self.config.enable_jprime and self._components.get("jprime"):
                logger.info("[Phase 7] Starting J-Prime (Mind)...")
                await self._start_component("jprime")

            # Phase 8: Start background tasks
            logger.info("[Phase 8] Starting background services...")
            await self._start_background_tasks()

            # Wait for components to become healthy
            logger.info("[Phase 9] Waiting for component health...")
            healthy = await self._wait_for_health()

            if healthy:
                logger.info("")
                logger.info("=" * 70)
                logger.info("            AGI OS READY - All Systems Operational")
                logger.info("=" * 70)
                self._print_status()
                return True
            else:
                logger.warning("Some components failed to start, running in degraded mode")
                return True

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """Gracefully stop all components."""
        logger.info("")
        logger.info("Initiating graceful shutdown...")
        self._running = False
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop v101.0 services first
        if self._experience_receiver:
            try:
                from reactor_core.integration.trinity_experience_receiver import shutdown_experience_receiver
                await shutdown_experience_receiver()
                logger.info("  [OK] Trinity Experience Receiver stopped")
            except Exception as e:
                logger.warning(f"  Trinity Experience Receiver stop failed: {e}")

        # Stop v91.0 services first (they depend on v77.0 services)
        if self._distributed_coordinator:
            try:
                await self._distributed_coordinator.stop()
                logger.info("  [OK] Distributed Coordinator stopped")
            except Exception as e:
                logger.warning(f"  Distributed Coordinator stop failed: {e}")

        if self._spot_vm_checkpointer:
            try:
                await self._spot_vm_checkpointer.stop()
                logger.info("  [OK] Spot VM Checkpointer stopped")
            except Exception as e:
                logger.warning(f"  Spot VM Checkpointer stop failed: {e}")

        if self._online_learning_engine:
            try:
                await self._online_learning_engine.stop()
                logger.info("  [OK] Online Learning Engine stopped")
            except Exception as e:
                logger.warning(f"  Online Learning Engine stop failed: {e}")

        if self._data_version_controller:
            try:
                await self._data_version_controller.close()
                logger.info("  [OK] Data Version Controller stopped")
            except Exception as e:
                logger.warning(f"  Data Version Controller stop failed: {e}")

        # Stop v77.0 services in reverse dependency order
        if self._model_server:
            try:
                await self._model_server.stop()
                logger.info("  [OK] Model Server stopped")
            except Exception as e:
                logger.warning(f"  Model Server stop failed: {e}")

        if self._scheduler:
            try:
                await self._scheduler.stop()
                logger.info("  [OK] Scheduler stopped")
            except Exception as e:
                logger.warning(f"  Scheduler stop failed: {e}")

        if self._health_aggregator:
            try:
                await self._health_aggregator.stop()
                logger.info("  [OK] Health Aggregator stopped")
            except Exception as e:
                logger.warning(f"  Health Aggregator stop failed: {e}")

        if self._telemetry:
            try:
                await self._telemetry.stop()
                logger.info("  [OK] Telemetry Collector stopped")
            except Exception as e:
                logger.warning(f"  Telemetry Collector stop failed: {e}")

        # Stop components in reverse order
        for name in reversed(list(self._components.keys())):
            await self._stop_component(name)

        # Stop event bridge
        if self._event_bridge:
            await self._event_bridge.stop()

        # Stop orchestrator
        if self._orchestrator:
            await shutdown_orchestrator()

        logger.info("AGI Supervisor shutdown complete")
        self._print_final_stats()

    async def _discover_components(self) -> None:
        """Discover available components."""
        # JARVIS (Body)
        if self.config.jarvis_path and self.config.jarvis_path.exists():
            self._components["jarvis"] = ManagedComponent(
                name="JARVIS",
                component_type=ComponentType.JARVIS_BODY,
                path=self.config.jarvis_path,
            )
            logger.info(f"  Found JARVIS at {self.config.jarvis_path}")
        elif self.config.enable_jarvis:
            logger.warning("  JARVIS path not found, will run without Body")

        # J-Prime (Mind)
        if self.config.jprime_path and self.config.jprime_path.exists():
            self._components["jprime"] = ManagedComponent(
                name="J-Prime",
                component_type=ComponentType.J_PRIME,
                path=self.config.jprime_path,
            )
            logger.info(f"  Found J-Prime at {self.config.jprime_path}")
        elif self.config.enable_jprime:
            logger.warning("  J-Prime path not found, will run without Mind")

        # Reactor Core (always present)
        self._components["reactor_core"] = ManagedComponent(
            name="Reactor Core",
            component_type=ComponentType.REACTOR_CORE,
            path=self.config.reactor_core_path,
            status=ComponentStatus.RUNNING,
        )
        logger.info(f"  Reactor Core at {self.config.reactor_core_path}")

    async def _start_reactor_core_services(self) -> None:
        """Start Reactor Core internal services with v77.0 integration."""
        # === v77.0 TELEMETRY COLLECTOR ===
        if HAS_V77_API:
            try:
                self._telemetry = get_telemetry()
                await self._telemetry.start()
                logger.info("  [OK] Telemetry Collector started")
            except Exception as e:
                logger.warning(f"  Telemetry Collector failed: {e}")

        # === v77.0 MODEL REGISTRY ===
        if HAS_V77_API:
            try:
                self._model_registry = get_registry()
                # Auto-discover models in common directories
                model_dirs = [
                    Path.home() / ".jarvis" / "models",
                    Path.home() / ".cache" / "huggingface",
                    self.config.reactor_core_path / "models" if self.config.reactor_core_path else None,
                ]
                for model_dir in model_dirs:
                    if model_dir and model_dir.exists():
                        await self._model_registry.scan_directory(model_dir)
                logger.info(f"  [OK] Model Registry initialized ({self._model_registry.model_count} models)")
            except Exception as e:
                logger.warning(f"  Model Registry failed: {e}")

        # === v77.0 HEALTH AGGREGATOR ===
        if HAS_V77_API:
            try:
                self._health_aggregator = await init_health_aggregator()
                # Register Trinity components for monitoring
                await self._health_aggregator.register_component("reactor_core", {
                    "type": "service",
                    "endpoint": f"http://localhost:{self.config.api_port}/health",
                    "interval": self.config.health_check_interval,
                })
                await self._health_aggregator.register_component("model_server", {
                    "type": "service",
                    "endpoint": f"http://localhost:{self.config.serving_port}/health",
                    "interval": self.config.health_check_interval,
                })
                if self.config.enable_jprime:
                    await self._health_aggregator.register_component("jprime", {
                        "type": "service",
                        "endpoint": f"http://localhost:{self.config.jprime_port}/health",
                        "interval": self.config.health_check_interval,
                    })
                await self._health_aggregator.start()
                logger.info("  [OK] Health Aggregator started (Trinity monitoring)")
            except Exception as e:
                logger.warning(f"  Health Aggregator failed: {e}")

        # === v77.0 SCHEDULER ===
        if HAS_V77_API:
            try:
                # Initialize scheduler with real training callback
                async def training_callback(job_id: str) -> None:
                    """Callback executed when scheduler triggers training."""
                    logger.info(f"Scheduler triggered training job: {job_id}")
                    self._stats["trainings_triggered"] += 1
                    if self._telemetry:
                        await self._telemetry.record_event("training_scheduled", {"job_id": job_id})
                    # Trigger actual training via unified pipeline
                    try:
                        from reactor_core.training.unified_pipeline import get_unified_trainer
                        trainer = await get_unified_trainer()
                        await trainer.train_async()
                    except Exception as e:
                        logger.error(f"Scheduled training failed: {e}")

                self._scheduler = await init_scheduler(training_callback=training_callback)

                # Add default training schedules
                await self._scheduler.add_job(
                    job_id="daily_incremental",
                    schedule=ScheduleTemplates.daily_at("02:00"),
                    callback=training_callback,
                    metadata={"type": "incremental", "description": "Daily incremental training"}
                )
                await self._scheduler.add_job(
                    job_id="weekly_full",
                    schedule=ScheduleTemplates.weekly_on("sunday", "03:00"),
                    callback=training_callback,
                    metadata={"type": "full", "description": "Weekly full training"}
                )
                await self._scheduler.start()
                logger.info("  [OK] Scheduler started (daily/weekly training)")
            except Exception as e:
                logger.warning(f"  Scheduler failed: {e}")

        # === v77.0 MODEL SERVER WITH HOT-RELOAD ===
        if HAS_V77_API and self.config.enable_serving:
            try:
                model_server_config = ModelServerConfig(
                    max_loaded_models=5,
                    memory_limit_gb=16.0,
                    enable_hot_reload=True,
                    watch_directories=[
                        Path.home() / ".jarvis" / "models",
                        self.config.reactor_core_path / "models" if self.config.reactor_core_path else Path.home() / "models",
                    ],
                    default_backend="auto",  # Auto-detect based on model format
                )
                self._model_server = get_model_server(config=model_server_config)

                # Register hot-reload callback to notify J-Prime
                @self._model_server.on_model_reload
                async def notify_prime_on_reload(model_id: str, model_info: dict) -> None:
                    """Notify J-Prime when models are hot-reloaded."""
                    logger.info(f"Model {model_id} reloaded, notifying J-Prime...")
                    if self._telemetry:
                        await self._telemetry.record_event("model_hot_reload", {
                            "model_id": model_id,
                            "info": model_info,
                        })
                    # Notify J-Prime via event bridge
                    if self._event_bridge:
                        await self._event_bridge.emit(EventType.MODEL_UPDATED, {
                            "model_id": model_id,
                            "info": model_info,
                            "action": "hot_reload",
                            "timestamp": datetime.now().isoformat(),
                        })
                    # Direct HTTP notification to J-Prime
                    if self._components.get("jprime") and self._components["jprime"].status == ComponentStatus.RUNNING:
                        try:
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                await session.post(
                                    f"http://localhost:{self.config.jprime_port}/api/v1/models/reload",
                                    json={"model_id": model_id, "info": model_info},
                                    timeout=aiohttp.ClientTimeout(total=5),
                                )
                        except Exception as e:
                            logger.debug(f"J-Prime notification failed (non-critical): {e}")

                await self._model_server.start()
                logger.info(f"  [OK] Model Server with Hot-Reload started")
            except Exception as e:
                logger.warning(f"  Model Server failed: {e}")

        # API Server
        if self.config.enable_api:
            try:
                from reactor_core.api.server import start_server_background
                await start_server_background(port=self.config.api_port)
                logger.info(f"  [OK] API Server on port {self.config.api_port}")
            except ImportError:
                logger.warning("  API Server module not available")
            except Exception as e:
                logger.warning(f"  API Server failed: {e}")

        # Legacy Inference Engine (fallback)
        if self.config.enable_serving and not self._model_server:
            try:
                from reactor_core.serving import InferenceEngine
                engine = InferenceEngine()
                await engine.start()
                logger.info(f"  [OK] Inference Engine started (legacy mode)")
            except ImportError:
                logger.warning("  Inference Engine module not available")
            except Exception as e:
                logger.warning(f"  Inference Engine failed: {e}")

        # Training Pipeline (background)
        if self.config.enable_training:
            logger.info("  [OK] Training Pipeline ready (on-demand + scheduled)")

        # Scout (optional)
        if self.config.enable_scout:
            logger.info("  [OK] Scout ingestion ready")

    async def _initialize_v91_services(self) -> None:
        """Initialize v91.0 Advanced Services for enhanced training and learning."""
        # === MLForge C++ Bindings ===
        if HAS_MLFORGE_BINDINGS and self.config.enable_mlforge_bindings:
            try:
                self._mlforge_bridge = get_bridge()
                backend = self._mlforge_bridge.get_backend_info()
                cpp_available = has_cpp_backend()
                logger.info(f"  [OK] MLForge Bridge initialized (backend: {backend['backend']}, C++ available: {cpp_available})")
            except Exception as e:
                logger.warning(f"  MLForge Bridge failed: {e}")

        # === Data Versioning Controller ===
        if HAS_DATA_VERSIONING and self.config.enable_data_versioning:
            try:
                # Ensure version store directory exists
                self.config.version_store_path.mkdir(parents=True, exist_ok=True)

                self._data_version_controller = DataVersionController(
                    store_path=self.config.version_store_path,
                    track_lineage=self.config.track_lineage,
                    auto_detect_drift=self.config.auto_detect_drift,
                )
                await self._data_version_controller.initialize()
                logger.info(f"  [OK] Data Version Controller initialized (path: {self.config.version_store_path})")
            except Exception as e:
                logger.warning(f"  Data Version Controller failed: {e}")

        # === Online Learning Engine ===
        if HAS_ONLINE_LEARNING and self.config.enable_online_learning:
            try:
                self._online_learning_engine = OnlineLearningEngine(
                    buffer_size=self.config.experience_buffer_size,
                    priority_alpha=self.config.priority_alpha,
                    ewc_lambda=self.config.ewc_lambda,
                )

                # Initialize drift detector
                self._drift_detector = DriftDetector(
                    threshold=self.config.drift_threshold,
                    window_size=1000,
                )

                # Initialize feedback integrator for JARVIS integration
                self._feedback_integrator = FeedbackIntegrator(
                    learning_engine=self._online_learning_engine,
                    event_bridge=self._event_bridge,
                )

                await self._online_learning_engine.start()
                logger.info(f"  [OK] Online Learning Engine initialized (buffer: {self.config.experience_buffer_size}, EWC λ: {self.config.ewc_lambda})")
            except Exception as e:
                logger.warning(f"  Online Learning Engine failed: {e}")

        # === GCP Spot VM Checkpointer ===
        if HAS_SPOT_VM_CHECKPOINTING and self.config.enable_spot_vm_checkpointing:
            try:
                # Auto-detect GCP project if not specified
                gcp_project = self.config.gcp_project
                if not gcp_project:
                    import subprocess
                    result = subprocess.run(
                        ["gcloud", "config", "get-value", "project"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        gcp_project = result.stdout.strip()

                if gcp_project:
                    self._spot_vm_checkpointer = SpotVMCheckpointer(
                        project_id=gcp_project,
                        bucket_name=self.config.checkpoint_bucket or f"{gcp_project}-checkpoints",
                        enable_prediction=self.config.preemption_prediction,
                        checkpoint_interval=self.config.checkpoint_interval,
                    )
                    await self._spot_vm_checkpointer.start()
                    logger.info(f"  [OK] Spot VM Checkpointer initialized (project: {gcp_project}, predictive: {self.config.preemption_prediction})")
                else:
                    logger.info("  [--] Spot VM Checkpointer skipped (no GCP project detected)")
            except Exception as e:
                logger.warning(f"  Spot VM Checkpointer failed: {e}")

        # === Distributed Training Coordinator ===
        if HAS_DISTRIBUTED_TRAINING and self.config.enable_distributed_training:
            try:
                # Determine distributed mode
                mode = self.config.distributed_mode
                if mode == "auto":
                    # Auto-detect based on environment
                    import torch
                    if torch.cuda.device_count() > 1:
                        mode = "multi"
                    else:
                        mode = "single"

                self._distributed_coordinator = DistributedCoordinator(
                    mode=mode,
                    min_workers=self.config.min_workers,
                    max_workers=self.config.max_workers,
                    enable_gradient_compression=self.config.gradient_compression,
                )

                # Initialize resource allocator
                self._resource_allocator = DynamicResourceAllocator(
                    coordinator=self._distributed_coordinator,
                    min_workers=self.config.min_workers,
                    max_workers=self.config.max_workers,
                )

                await self._distributed_coordinator.start()
                logger.info(f"  [OK] Distributed Coordinator initialized (mode: {mode}, workers: {self.config.min_workers}-{self.config.max_workers})")
            except Exception as e:
                logger.warning(f"  Distributed Coordinator failed: {e}")

        # === Connect components for integrated operation ===
        await self._connect_v91_services()

    async def _connect_v91_services(self) -> None:
        """Connect v91.0 services for integrated operation."""
        # Connect online learning to event bridge for JARVIS feedback
        if self._online_learning_engine and self._event_bridge:
            @self._event_bridge.on_event(EventType.FEEDBACK)
            async def handle_feedback_for_learning(event):
                """Route feedback to online learning engine."""
                try:
                    await self._online_learning_engine.add_experience({
                        "type": "feedback",
                        "payload": event.payload,
                        "timestamp": event.timestamp,
                        "source": event.source.value if hasattr(event.source, 'value') else str(event.source),
                    })
                    self._stats["online_updates"] += 1
                except Exception as e:
                    logger.debug(f"Failed to route feedback to online learning: {e}")

            @self._event_bridge.on_event(EventType.CORRECTION)
            async def handle_correction_for_learning(event):
                """Route corrections to online learning with high priority."""
                try:
                    await self._online_learning_engine.add_experience({
                        "type": "correction",
                        "payload": event.payload,
                        "timestamp": event.timestamp,
                        "priority": 1.0,  # Corrections get max priority
                    })
                    self._stats["online_updates"] += 1
                except Exception as e:
                    logger.debug(f"Failed to route correction to online learning: {e}")

        # Connect drift detector to trigger retraining
        if self._drift_detector and self._online_learning_engine:
            async def on_drift_detected(drift_info: dict):
                """Handle concept drift detection."""
                logger.warning(f"Concept drift detected: {drift_info}")
                self._stats["drift_detections"] += 1

                if self._telemetry:
                    await self._telemetry.record_event("concept_drift_detected", drift_info)

                # Trigger incremental retraining
                if self._scheduler:
                    await self._scheduler.trigger_job("daily_incremental", reason="drift_detected")

            self._drift_detector.on_drift(on_drift_detected)

        # Connect checkpointer to training events
        if self._spot_vm_checkpointer and self._event_bridge:
            @self._event_bridge.on_event(EventType.TRAINING_STARTED)
            async def handle_training_started(event):
                """Start checkpointing when training begins."""
                try:
                    await self._spot_vm_checkpointer.start_session(
                        training_id=event.payload.get("training_id", "default"),
                        model_config=event.payload.get("model_config", {}),
                    )
                except Exception as e:
                    logger.debug(f"Failed to start checkpoint session: {e}")

            @self._event_bridge.on_event(EventType.TRAINING_COMPLETE)
            async def handle_training_complete_checkpoint(event):
                """Finalize checkpointing when training completes."""
                try:
                    await self._spot_vm_checkpointer.finalize_session()
                    self._stats["checkpoints_saved"] += 1
                except Exception as e:
                    logger.debug(f"Failed to finalize checkpoint session: {e}")

        # Connect data versioning to training pipeline
        if self._data_version_controller and self._event_bridge:
            @self._event_bridge.on_event(EventType.TRAINING_STARTED)
            async def version_training_data(event):
                """Create version snapshot of training data."""
                try:
                    data_path = event.payload.get("data_path")
                    if data_path:
                        version = await self._data_version_controller.create_version(
                            data_path=data_path,
                            metadata={
                                "training_id": event.payload.get("training_id"),
                                "timestamp": event.timestamp,
                            }
                        )
                        self._stats["data_versions_created"] += 1
                        logger.debug(f"Created data version: {version.version_id}")
                except Exception as e:
                    logger.debug(f"Failed to create data version: {e}")

        logger.info("  [OK] v91.0 Services connected for integrated operation")

    async def _start_component(self, name: str) -> bool:
        """Start an external component."""
        component = self._components.get(name)
        if not component or not component.path:
            return False

        component.status = ComponentStatus.STARTING
        component.start_time = time.time()

        try:
            # Determine start command based on component
            if name == "jarvis":
                start_cmd = self._get_jarvis_start_command(component.path)
            elif name == "jprime":
                start_cmd = self._get_jprime_start_command(component.path)
            else:
                return False

            if start_cmd:
                # Start process
                component.process = subprocess.Popen(
                    start_cmd,
                    cwd=str(component.path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**os.environ, "TRINITY_ENABLED": "1"},
                )

                # Wait briefly for startup
                await asyncio.sleep(2.0)

                if component.process.poll() is None:
                    component.status = ComponentStatus.RUNNING
                    logger.info(f"  [OK] {component.name} started (PID: {component.process.pid})")
                    return True
                else:
                    component.status = ComponentStatus.FAILED
                    stderr = component.process.stderr.read().decode() if component.process.stderr else ""
                    component.error_message = stderr[:200]
                    logger.error(f"  [FAIL] {component.name} exited: {component.error_message}")
                    return False
            else:
                logger.warning(f"  No start command found for {name}")
                return False

        except Exception as e:
            component.status = ComponentStatus.FAILED
            component.error_message = str(e)
            logger.error(f"  [FAIL] Failed to start {component.name}: {e}")
            return False

    def _get_jarvis_start_command(self, path: Path) -> Optional[List[str]]:
        """Get command to start JARVIS."""
        # Check for various entry points
        candidates = [
            path / "main.py",
            path / "jarvis" / "main.py",
            path / "run.py",
            path / "jarvis.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                return [sys.executable, str(candidate), "--trinity"]

        # Try package execution
        if (path / "jarvis" / "__main__.py").exists():
            return [sys.executable, "-m", "jarvis", "--trinity"]

        return None

    def _get_jprime_start_command(self, path: Path) -> Optional[List[str]]:
        """Get command to start J-Prime."""
        candidates = [
            path / "run_server.py",
            path / "server.py",
            path / "main.py",
            path / "jprime" / "server.py",
        ]

        for candidate in candidates:
            if candidate.exists():
                return [
                    sys.executable, str(candidate),
                    "--port", str(self.config.jprime_port),
                    "--trinity",
                ]

        return None

    async def _stop_component(self, name: str) -> None:
        """Stop a component gracefully."""
        component = self._components.get(name)
        if not component:
            return

        if component.process and component.process.poll() is None:
            logger.info(f"  Stopping {component.name}...")

            # Try graceful shutdown first
            component.process.terminate()

            # Wait for termination
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, component.process.wait
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                # Force kill
                component.process.kill()
                component.process.wait()

            logger.info(f"  [OK] {component.name} stopped")

        component.status = ComponentStatus.STOPPED

    async def _wait_for_health(self) -> bool:
        """Wait for all components to become healthy."""
        deadline = time.time() + self.config.startup_timeout

        while time.time() < deadline:
            all_healthy = True

            for name, component in self._components.items():
                if component.status == ComponentStatus.STARTING:
                    all_healthy = False

                # Check process health
                if component.process:
                    if component.process.poll() is not None:
                        component.status = ComponentStatus.FAILED
                        all_healthy = False

                # Check heartbeat
                if component.status == ComponentStatus.RUNNING:
                    orchestrator_state = self._orchestrator.get_component_state(component.component_type)
                    if orchestrator_state.health == ComponentHealth.HEALTHY:
                        component.last_heartbeat = time.time()

            if all_healthy:
                return True

            await asyncio.sleep(1.0)

        return False

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and processing tasks."""
        # Health monitoring
        self._tasks.append(asyncio.create_task(self._health_monitor_loop()))

        # Experience collection (continuous learning)
        if self.config.experience_collection:
            self._tasks.append(asyncio.create_task(self._experience_collection_loop()))

        # Event processing
        self._tasks.append(asyncio.create_task(self._event_processing_loop()))

        # v101.0: Start Trinity Experience Receiver (closes the Trinity Loop)
        try:
            from reactor_core.integration.trinity_experience_receiver import get_experience_receiver
            self._experience_receiver = await get_experience_receiver()
            logger.info("[OK] Trinity Experience Receiver started (Trinity Loop CLOSED)")
        except ImportError:
            logger.debug("Trinity Experience Receiver not available")
        except Exception as e:
            logger.warning(f"Trinity Experience Receiver failed to start: {e}")

    async def _health_monitor_loop(self) -> None:
        """Monitor component health and handle failures."""
        while self._running:
            try:
                for name, component in self._components.items():
                    # Skip if not supposed to be running
                    if component.status in (ComponentStatus.STOPPED, ComponentStatus.UNKNOWN):
                        continue

                    # Check process
                    if component.process and component.process.poll() is not None:
                        logger.warning(f"{component.name} has died unexpectedly")
                        component.status = ComponentStatus.FAILED

                        if self.config.retry_on_failure and component.restart_count < self.config.max_retries:
                            logger.info(f"Attempting to restart {component.name}...")
                            component.restart_count += 1
                            self._stats["restarts"] += 1
                            await self._start_component(name)

                    # Check heartbeat freshness
                    if component.status == ComponentStatus.RUNNING:
                        orchestrator_state = self._orchestrator.get_component_state(component.component_type)

                        if orchestrator_state.health == ComponentHealth.OFFLINE:
                            if component.status != ComponentStatus.DEGRADED:
                                logger.warning(f"{component.name} missed heartbeat, marking as degraded")
                                component.status = ComponentStatus.DEGRADED

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)

    async def _experience_collection_loop(self) -> None:
        """Collect experiences from JARVIS for continuous learning with v77.0 integration."""
        processed_files: Set[str] = set()
        experience_buffer: List[dict] = []

        while self._running:
            try:
                # Check for new experiences in Trinity events
                events_dir = self.config.trinity_dir / "events"

                if events_dir.exists():
                    for event_file in events_dir.glob("*.json"):
                        file_key = str(event_file)
                        if file_key in processed_files:
                            continue

                        try:
                            with open(event_file) as f:
                                event_data = json.load(f)

                            event_type = event_data.get("event_type", "")

                            # Collect interaction events
                            if event_type in ("interaction_end", "correction", "feedback", "learning_signal"):
                                self._stats["experiences_collected"] += 1
                                experience_buffer.append(event_data)

                                # Record to telemetry
                                if self._telemetry:
                                    await self._telemetry.record_event(
                                        "experience_collected",
                                        {
                                            "type": event_type,
                                            "total_count": self._stats["experiences_collected"],
                                        }
                                    )

                                # Check if we should trigger training
                                if (
                                    self._stats["experiences_collected"] > 0 and
                                    self._stats["experiences_collected"] % self.config.auto_training_threshold == 0
                                ):
                                    logger.info(f"Experience threshold reached ({self._stats['experiences_collected']}), triggering training...")
                                    self._stats["trainings_triggered"] += 1

                                    if self._telemetry:
                                        await self._telemetry.record_event(
                                            "training_triggered",
                                            {
                                                "reason": "experience_threshold",
                                                "experience_count": self._stats["experiences_collected"],
                                            }
                                        )

                                    # Trigger real training via unified pipeline
                                    try:
                                        from reactor_core.training.unified_pipeline import get_unified_trainer
                                        trainer = await get_unified_trainer()
                                        # Pass collected experiences to trainer
                                        await trainer.add_experiences(experience_buffer)
                                        asyncio.create_task(trainer.train_async())
                                        experience_buffer.clear()
                                        logger.info("Training job queued with collected experiences")
                                    except ImportError:
                                        logger.warning("Unified trainer not available for auto-training")
                                    except Exception as e:
                                        logger.error(f"Auto-training failed: {e}")

                            processed_files.add(file_key)

                        except Exception as e:
                            logger.debug(f"Error processing event file: {e}")

                # Periodic telemetry flush
                if self._telemetry and self._stats["experiences_collected"] > 0:
                    await self._telemetry.flush()

                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Experience collection error: {e}")
                await asyncio.sleep(10.0)

    async def _event_processing_loop(self) -> None:
        """Process events from the event bridge."""
        if not self._event_bridge:
            return

        # Register event handlers
        @self._event_bridge.on_event(EventType.CORRECTION)
        async def handle_correction(event):
            logger.info(f"Received correction event: {event.payload.get('user_input', '')[:50]}...")
            self._stats["events_processed"] += 1

        @self._event_bridge.on_event(EventType.TRAINING_COMPLETE)
        async def handle_training_complete(event):
            logger.info(f"Training completed: {event.payload}")
            self._stats["events_processed"] += 1

        @self._event_bridge.on_event(EventType.SAFETY_BLOCKED)
        async def handle_safety_blocked(event):
            logger.warning(f"Safety block: {event.payload.get('action', '')} - {event.payload.get('reason', '')}")
            self._stats["events_processed"] += 1

        while self._running:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    def _print_status(self) -> None:
        """Print current system status."""
        logger.info("")
        logger.info("COMPONENT STATUS:")
        logger.info("-" * 50)

        for name, component in self._components.items():
            status_icon = {
                ComponentStatus.RUNNING: "[RUNNING]",
                ComponentStatus.DEGRADED: "[DEGRADED]",
                ComponentStatus.STOPPED: "[STOPPED]",
                ComponentStatus.FAILED: "[FAILED]",
                ComponentStatus.STARTING: "[STARTING]",
                ComponentStatus.UNKNOWN: "[UNKNOWN]",
            }.get(component.status, "?")

            pid_info = f" (PID: {component.process.pid})" if component.process else ""
            logger.info(f"  {status_icon} {component.name}{pid_info}")

        logger.info("-" * 50)

        # v77.0 Services Status
        if HAS_V77_API:
            logger.info("")
            logger.info("v77.0 SERVICES:")
            logger.info("-" * 50)
            logger.info(f"  [{'OK' if self._telemetry else '--'}] Telemetry Collector")
            logger.info(f"  [{'OK' if self._model_registry else '--'}] Model Registry ({self._model_registry.model_count if self._model_registry else 0} models)")
            logger.info(f"  [{'OK' if self._health_aggregator else '--'}] Health Aggregator")
            logger.info(f"  [{'OK' if self._scheduler else '--'}] Training Scheduler")
            logger.info(f"  [{'OK' if self._model_server else '--'}] Model Server (Hot-Reload)")
            logger.info("-" * 50)

        # v91.0 Advanced Services Status
        logger.info("")
        logger.info("v91.0 ADVANCED SERVICES:")
        logger.info("-" * 50)
        mlforge_info = f" (backend: {self._mlforge_bridge.get_backend_info()['backend']})" if self._mlforge_bridge else ""
        logger.info(f"  [{'OK' if self._mlforge_bridge else '--'}] MLForge Bridge{mlforge_info}")
        logger.info(f"  [{'OK' if self._online_learning_engine else '--'}] Online Learning Engine")
        logger.info(f"  [{'OK' if self._drift_detector else '--'}] Concept Drift Detector")
        logger.info(f"  [{'OK' if self._data_version_controller else '--'}] Data Version Controller")
        logger.info(f"  [{'OK' if self._spot_vm_checkpointer else '--'}] Spot VM Checkpointer")
        logger.info(f"  [{'OK' if self._distributed_coordinator else '--'}] Distributed Coordinator")
        logger.info(f"  [{'OK' if self._resource_allocator else '--'}] Dynamic Resource Allocator")
        logger.info("-" * 50)

        logger.info("")
        logger.info("ENDPOINTS:")
        if self.config.enable_api:
            logger.info(f"  API Server:      http://localhost:{self.config.api_port}")
            logger.info(f"  API Docs:        http://localhost:{self.config.api_port}/docs")
            logger.info(f"  Health Check:    http://localhost:{self.config.api_port}/health")
        if self.config.enable_serving:
            logger.info(f"  Model Serving:   http://localhost:{self.config.serving_port}")
        if self._components.get("jprime") and self._components["jprime"].status == ComponentStatus.RUNNING:
            logger.info(f"  J-Prime:         http://localhost:{self.config.jprime_port}")
        logger.info("")
        logger.info("FEATURES:")
        logger.info("  - Real-time model hot-reload with Prime notification")
        logger.info("  - Scheduled training (daily 2am, weekly Sunday 3am)")
        logger.info("  - Cross-component health monitoring")
        logger.info("  - Telemetry collection and event tracking")
        logger.info("  v91.0 ADVANCED:")
        logger.info("  - Online/incremental learning with experience replay")
        logger.info("  - Elastic Weight Consolidation (anti-forgetting)")
        logger.info("  - Concept drift detection with auto-retraining")
        logger.info("  - Content-addressed data versioning with lineage")
        logger.info("  - GCP Spot VM predictive checkpointing")
        logger.info("  - Distributed training with gradient compression")
        logger.info("  - Dynamic resource allocation with auto-scaling")
        logger.info("  - MLForge C++ bindings for accelerated ops")
        logger.info("")
        logger.info("Press Ctrl+C to shutdown")
        logger.info("")

    def _print_final_stats(self) -> None:
        """Print final statistics."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info("")
        logger.info("=" * 60)
        logger.info("                    FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"  Uptime:              {hours}h {minutes}m {seconds}s")
        logger.info(f"  Events Processed:    {self._stats['events_processed']}")
        logger.info(f"  Experiences:         {self._stats['experiences_collected']}")
        logger.info(f"  Trainings Triggered: {self._stats['trainings_triggered']}")
        logger.info(f"  Component Restarts:  {self._stats['restarts']}")
        logger.info("-" * 60)

        # v77.0 Service Stats
        if HAS_V77_API:
            logger.info("v77.0 SERVICE METRICS:")
            if self._telemetry:
                try:
                    telemetry_stats = self._telemetry.get_stats()
                    logger.info(f"  Telemetry Events:    {telemetry_stats.get('total_events', 'N/A')}")
                except Exception:
                    logger.info("  Telemetry Events:    N/A")
            if self._model_registry:
                logger.info(f"  Models Registered:   {self._model_registry.model_count}")
            if self._scheduler:
                try:
                    scheduler_stats = self._scheduler.get_stats()
                    logger.info(f"  Scheduled Jobs Run:  {scheduler_stats.get('jobs_executed', 'N/A')}")
                except Exception:
                    logger.info("  Scheduled Jobs Run:  N/A")
            if self._model_server:
                try:
                    server_stats = self._model_server.get_stats()
                    logger.info(f"  Inference Requests:  {server_stats.get('total_requests', 'N/A')}")
                    logger.info(f"  Models Hot-Reloaded: {server_stats.get('hot_reloads', 'N/A')}")
                except Exception:
                    logger.info("  Inference Requests:  N/A")
            if self._health_aggregator:
                try:
                    health_stats = self._health_aggregator.get_stats()
                    logger.info(f"  Health Checks:       {health_stats.get('total_checks', 'N/A')}")
                except Exception:
                    logger.info("  Health Checks:       N/A")

        # v91.0 Service Stats
        logger.info("")
        logger.info("v91.0 ADVANCED SERVICE METRICS:")
        logger.info(f"  Online Updates:      {self._stats['online_updates']}")
        logger.info(f"  Drift Detections:    {self._stats['drift_detections']}")
        logger.info(f"  Checkpoints Saved:   {self._stats['checkpoints_saved']}")
        logger.info(f"  Preemptions Survived:{self._stats['preemptions_survived']}")
        logger.info(f"  Distributed Syncs:   {self._stats['distributed_syncs']}")
        logger.info(f"  Data Versions:       {self._stats['data_versions_created']}")

        if self._online_learning_engine:
            try:
                ole_stats = self._online_learning_engine.get_stats()
                logger.info(f"  Experience Buffer:   {ole_stats.get('buffer_size', 'N/A')}/{ole_stats.get('buffer_capacity', 'N/A')}")
                logger.info(f"  Incremental Updates: {ole_stats.get('incremental_updates', 'N/A')}")
            except Exception:
                pass

        if self._distributed_coordinator:
            try:
                dc_stats = self._distributed_coordinator.get_stats()
                logger.info(f"  Active Workers:      {dc_stats.get('active_workers', 'N/A')}")
                logger.info(f"  Gradient Syncs:      {dc_stats.get('gradient_syncs', 'N/A')}")
            except Exception:
                pass

        if self._data_version_controller:
            try:
                dvc_stats = self._data_version_controller.get_stats()
                logger.info(f"  Total Versions:      {dvc_stats.get('total_versions', 'N/A')}")
                logger.info(f"  Lineage Edges:       {dvc_stats.get('lineage_edges', 'N/A')}")
            except Exception:
                pass

        logger.info("=" * 60)

    async def run_until_shutdown(self) -> None:
        """Run until shutdown signal received."""
        await self._shutdown_event.wait()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for AGI Supervisor."""
    parser = argparse.ArgumentParser(
        description="AGI OS Unified Supervisor - Project Trinity (v91.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_supervisor.py                    # Start all components
  python3 run_supervisor.py --no-jarvis        # Start without JARVIS
  python3 run_supervisor.py --no-training      # Disable training
  python3 run_supervisor.py --api-port 8080    # Custom API port
  python3 run_supervisor.py --distributed multi --max-workers 4   # Multi-GPU distributed
  python3 run_supervisor.py --gcp-project myproj --checkpoint-bucket mybucket  # GCP Spot VM
        """,
    )

    # Component flags
    parser.add_argument("--no-jarvis", action="store_true", help="Disable JARVIS (Body)")
    parser.add_argument("--no-jprime", action="store_true", help="Disable J-Prime (Mind)")
    parser.add_argument("--no-training", action="store_true", help="Disable training pipeline")
    parser.add_argument("--no-serving", action="store_true", help="Disable model serving")
    parser.add_argument("--enable-scout", action="store_true", help="Enable Scout web ingestion")

    # Paths
    parser.add_argument("--jarvis-path", type=Path, help="Path to JARVIS repository")
    parser.add_argument("--jprime-path", type=Path, help="Path to J-Prime repository")

    # Ports
    parser.add_argument("--api-port", type=int, default=8003, help="API server port")
    parser.add_argument("--serving-port", type=int, default=8001, help="Model serving port")
    parser.add_argument("--jprime-port", type=int, default=8000, help="J-Prime server port")

    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", type=Path, help="Log file path")

    # v91.0 Advanced Feature Flags
    v91_group = parser.add_argument_group("v91.0 Advanced Features")
    v91_group.add_argument("--no-online-learning", action="store_true", help="Disable online/incremental learning")
    v91_group.add_argument("--no-distributed", action="store_true", help="Disable distributed training")
    v91_group.add_argument("--no-versioning", action="store_true", help="Disable data versioning")
    v91_group.add_argument("--no-spot-checkpointing", action="store_true", help="Disable GCP Spot VM checkpointing")
    v91_group.add_argument("--no-mlforge", action="store_true", help="Disable MLForge C++ bindings")

    # Online Learning Settings
    ol_group = parser.add_argument_group("Online Learning")
    ol_group.add_argument("--experience-buffer-size", type=int, default=100000, help="Experience replay buffer size")
    ol_group.add_argument("--ewc-lambda", type=float, default=100.0, help="EWC regularization strength")
    ol_group.add_argument("--drift-threshold", type=float, default=0.1, help="Concept drift detection threshold")

    # Distributed Training Settings
    dist_group = parser.add_argument_group("Distributed Training")
    dist_group.add_argument("--distributed", choices=["auto", "single", "multi"], default="auto", help="Distributed mode")
    dist_group.add_argument("--min-workers", type=int, default=1, help="Minimum worker count")
    dist_group.add_argument("--max-workers", type=int, default=8, help="Maximum worker count")
    dist_group.add_argument("--no-gradient-compression", action="store_true", help="Disable gradient compression")

    # GCP Settings
    gcp_group = parser.add_argument_group("GCP Spot VM")
    gcp_group.add_argument("--gcp-project", type=str, help="GCP project ID")
    gcp_group.add_argument("--checkpoint-bucket", type=str, help="GCS bucket for checkpoints")
    gcp_group.add_argument("--checkpoint-interval", type=int, default=300, help="Checkpoint interval in seconds")
    gcp_group.add_argument("--no-preemption-prediction", action="store_true", help="Disable preemption prediction")

    # MLForge Settings
    mlforge_group = parser.add_argument_group("MLForge")
    mlforge_group.add_argument("--mlforge-backend", choices=["auto", "cpp", "numpy", "torch"], default="auto", help="MLForge backend")

    args = parser.parse_args()

    # Build config
    config = SupervisorConfig(
        jarvis_path=args.jarvis_path,
        jprime_path=args.jprime_path,
        enable_jarvis=not args.no_jarvis,
        enable_jprime=not args.no_jprime,
        enable_training=not args.no_training,
        enable_serving=not args.no_serving,
        enable_scout=args.enable_scout,
        api_port=args.api_port,
        serving_port=args.serving_port,
        jprime_port=args.jprime_port,
        log_level=args.log_level,
        log_file=args.log_file,
        # v91.0 settings
        enable_online_learning=not args.no_online_learning,
        enable_distributed_training=not args.no_distributed,
        enable_data_versioning=not args.no_versioning,
        enable_spot_vm_checkpointing=not args.no_spot_checkpointing,
        enable_mlforge_bindings=not args.no_mlforge,
        experience_buffer_size=args.experience_buffer_size,
        ewc_lambda=args.ewc_lambda,
        drift_threshold=args.drift_threshold,
        distributed_mode=args.distributed,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        gradient_compression=not args.no_gradient_compression,
        gcp_project=args.gcp_project,
        checkpoint_bucket=args.checkpoint_bucket,
        checkpoint_interval=args.checkpoint_interval,
        preemption_prediction=not args.no_preemption_prediction,
        mlforge_backend=args.mlforge_backend,
    )

    # Create supervisor
    supervisor = AGISupervisor(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\nReceived shutdown signal...")
        supervisor._shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # Start supervisor
    success = await supervisor.start()

    if success:
        # Run until shutdown
        await supervisor.run_until_shutdown()

    # Cleanup
    await supervisor.stop()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
