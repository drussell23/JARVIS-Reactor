"""
Model Versioning and Registry System for Reactor-Core.

Provides comprehensive model lifecycle management including:
- Version tracking with semantic versioning
- A/B testing support with traffic splitting
- Deployment notifications
- Rollback capabilities
- Catastrophic forgetting prevention through checkpointing

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Model Registry                            │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
    │  │ Version Manager │  │   A/B Testing   │  │ Checkpoint  │  │
    │  │                 │  │    Manager      │  │   Store     │  │
    │  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘  │
    │           │                    │                  │          │
    │           └────────────────────┼──────────────────┘          │
    │                                ▼                             │
    │                ┌──────────────────────────┐                  │
    │                │    Deployment Manager    │                  │
    │                │  (notifications, rollback│                  │
    │                │   hot-swap coordination) │                  │
    │                └──────────────────────────┘                  │
    └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │  JARVIS / Prime      │
                        │  (model consumers)   │
                        └──────────────────────┘

Features:
- Semantic versioning with automatic incrementing
- Model artifact storage with metadata
- A/B testing with configurable traffic split
- Deployment notifications to JARVIS
- Automatic rollback on performance degradation
- Checkpoint-based catastrophic forgetting prevention
- Model comparison and diff utilities
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# v152.0: Import cloud mode detector for JARVIS ecosystem awareness
try:
    from reactor_core.core.cloud_mode_detector import (
        is_cloud_mode_active,
        get_effective_jarvis_url,
        should_skip_local_service,
    )
    CLOUD_MODE_DETECTOR_AVAILABLE = True
except ImportError:
    CLOUD_MODE_DETECTOR_AVAILABLE = False
    def is_cloud_mode_active() -> bool:
        return os.getenv("JARVIS_GCP_OFFLOAD_ACTIVE", "false").lower() == "true"
    def get_effective_jarvis_url(default: str = "http://localhost:8000") -> str:
        if is_cloud_mode_active():
            return os.getenv("JARVIS_PRIME_CLOUD_RUN_URL", "")
        return default
    def should_skip_local_service(service_name: str = "jarvis") -> tuple:
        if is_cloud_mode_active():
            return True, "Cloud mode active"
        return False, None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class RegistryConfig:
    """Model registry configuration."""

    # Storage paths
    REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", str(Path.home() / ".reactor_core" / "models")))
    CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT_PATH", str(Path.home() / ".reactor_core" / "checkpoints")))

    # Versioning
    DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "jarvis-base")

    # A/B Testing
    AB_MIN_SAMPLE_SIZE = int(os.getenv("AB_MIN_SAMPLE_SIZE", "100"))
    AB_SIGNIFICANCE_THRESHOLD = float(os.getenv("AB_SIGNIFICANCE_THRESHOLD", "0.05"))

    # Deployment
    DEPLOYMENT_TIMEOUT = float(os.getenv("DEPLOYMENT_TIMEOUT", "300.0"))  # 5 minutes
    ROLLBACK_THRESHOLD = float(os.getenv("ROLLBACK_THRESHOLD", "0.15"))  # 15% error rate

    # Retention
    MAX_VERSIONS_KEPT = int(os.getenv("MAX_MODEL_VERSIONS", "10"))
    CHECKPOINT_RETENTION_DAYS = int(os.getenv("CHECKPOINT_RETENTION_DAYS", "30"))

    # Notification - v152.0: Use cloud-aware URL resolution
    JARVIS_API_URL = os.getenv("JARVIS_API_URL", "http://localhost:8000")
    PRIME_API_URL = os.getenv("PRIME_API_URL", "http://localhost:8001")

    @staticmethod
    def get_jarvis_api_url() -> str:
        """v152.0: Get JARVIS API URL, respecting cloud mode."""
        env_url = os.getenv("JARVIS_API_URL")
        if env_url:
            return env_url
        return get_effective_jarvis_url("http://localhost:8000")

    @staticmethod
    def get_prime_api_url() -> str:
        """v152.0: Get Prime API URL, respecting cloud mode."""
        env_url = os.getenv("PRIME_API_URL")
        if env_url:
            return env_url
        # In cloud mode, Prime and JARVIS share the same endpoint
        if is_cloud_mode_active():
            return get_effective_jarvis_url("")
        return "http://localhost:8001"


# ============================================================================
# Data Models
# ============================================================================

class ModelStatus(Enum):
    """Model version status."""
    DRAFT = "draft"
    TESTING = "testing"
    STAGED = "staged"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ROLLED_BACK = "rolled_back"
    ARCHIVED = "archived"


class DeploymentTarget(Enum):
    """Deployment targets."""
    JARVIS = "jarvis"
    PRIME = "prime"
    BOTH = "both"
    CANARY = "canary"


@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None  # e.g., "alpha", "beta", "rc1"
    build: Optional[str] = None  # e.g., build metadata

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string."""
        # Remove build metadata
        if "+" in version_str:
            version_str, build = version_str.split("+", 1)
        else:
            build = None

        # Remove prerelease
        if "-" in version_str:
            version_str, prerelease = version_str.split("-", 1)
        else:
            prerelease = None

        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
            prerelease=prerelease,
            build=build,
        )

    def increment_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)

    def increment_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)

    def increment_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    error_rate: Optional[float] = None
    memory_mb: Optional[float] = None
    custom: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class ModelVersion:
    """Model version metadata."""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    model_name: str = RegistryConfig.DEFAULT_MODEL_NAME
    version: SemanticVersion = field(default_factory=SemanticVersion)
    status: ModelStatus = ModelStatus.DRAFT
    artifact_path: Optional[str] = None
    artifact_hash: Optional[str] = None
    artifact_size_bytes: int = 0
    training_job_id: Optional[str] = None
    parent_version_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    deployed_at: Optional[float] = None
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "version": str(self.version),
            "status": self.status.value,
            "artifact_path": self.artifact_path,
            "artifact_hash": self.artifact_hash,
            "artifact_size_bytes": self.artifact_size_bytes,
            "training_job_id": self.training_job_id,
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at,
            "created_datetime": datetime.fromtimestamp(self.created_at).isoformat(),
            "deployed_at": self.deployed_at,
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
            "tags": self.tags,
        }


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    control_version_id: str = ""  # Version A (baseline)
    treatment_version_id: str = ""  # Version B (new)
    traffic_split: float = 0.5  # Fraction of traffic to treatment
    min_sample_size: int = RegistryConfig.AB_MIN_SAMPLE_SIZE
    significance_threshold: float = RegistryConfig.AB_SIGNIFICANCE_THRESHOLD
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    is_active: bool = True
    winner: Optional[str] = None  # version_id of winner
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """A/B test results."""
    test_id: str = ""
    control_samples: int = 0
    treatment_samples: int = 0
    control_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    treatment_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    improvement: Dict[str, float] = field(default_factory=dict)  # metric -> % improvement
    p_values: Dict[str, float] = field(default_factory=dict)
    is_significant: bool = False
    recommended_winner: Optional[str] = None
    confidence_level: float = 0.0


@dataclass
class Checkpoint:
    """Model checkpoint for catastrophic forgetting prevention."""
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    model_name: str = ""
    version_id: str = ""
    checkpoint_path: str = ""
    checkpoint_hash: str = ""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentRecord:
    """Deployment record."""
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    version_id: str = ""
    target: DeploymentTarget = DeploymentTarget.JARVIS
    status: str = "pending"  # pending, deploying, deployed, failed, rolled_back
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    notification_sent: bool = False
    rollback_of: Optional[str] = None  # deployment_id this rolled back from


# ============================================================================
# Version Manager
# ============================================================================

class VersionManager:
    """
    Manage model versions with semantic versioning.

    Handles version creation, storage, retrieval, and cleanup.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self._registry_path = registry_path or RegistryConfig.REGISTRY_PATH
        self._registry_path.mkdir(parents=True, exist_ok=True)

        self._versions: Dict[str, ModelVersion] = {}  # version_id -> ModelVersion
        self._by_model: Dict[str, List[str]] = defaultdict(list)  # model_name -> [version_ids]
        self._lock = asyncio.Lock()

        # Load existing versions
        self._load_registry()

    def _load_registry(self):
        """Load version registry from disk."""
        registry_file = self._registry_path / "registry.json"
        if registry_file.exists():
            try:
                data = json.loads(registry_file.read_text())
                for v_data in data.get("versions", []):
                    version = ModelVersion(
                        version_id=v_data["version_id"],
                        model_name=v_data["model_name"],
                        version=SemanticVersion.parse(v_data["version"]),
                        status=ModelStatus(v_data["status"]),
                        artifact_path=v_data.get("artifact_path"),
                        artifact_hash=v_data.get("artifact_hash"),
                        artifact_size_bytes=v_data.get("artifact_size_bytes", 0),
                        training_job_id=v_data.get("training_job_id"),
                        parent_version_id=v_data.get("parent_version_id"),
                        created_at=v_data.get("created_at", time.time()),
                        deployed_at=v_data.get("deployed_at"),
                        tags=v_data.get("tags", []),
                        metadata=v_data.get("metadata", {}),
                    )
                    self._versions[version.version_id] = version
                    self._by_model[version.model_name].append(version.version_id)
                logger.info(f"[Registry] Loaded {len(self._versions)} versions")
            except Exception as e:
                logger.error(f"[Registry] Failed to load registry: {e}")

    async def _save_registry(self):
        """Save version registry to disk."""
        async with self._lock:
            registry_file = self._registry_path / "registry.json"
            data = {
                "versions": [v.to_dict() for v in self._versions.values()],
                "updated_at": time.time(),
            }
            registry_file.write_text(json.dumps(data, indent=2))

    async def create_version(
        self,
        model_name: str,
        artifact_path: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        training_job_id: Optional[str] = None,
        increment: str = "patch",  # major, minor, patch
        metrics: Optional[ModelMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelVersion:
        """Create a new model version."""
        async with self._lock:
            # Determine version number
            existing = self._by_model.get(model_name, [])
            if existing:
                latest = max(
                    (self._versions[vid] for vid in existing),
                    key=lambda v: (v.version.major, v.version.minor, v.version.patch),
                )
                if increment == "major":
                    new_version = latest.version.increment_major()
                elif increment == "minor":
                    new_version = latest.version.increment_minor()
                else:
                    new_version = latest.version.increment_patch()
            else:
                new_version = SemanticVersion(1, 0, 0)

            # Calculate artifact hash if path provided
            artifact_hash = None
            artifact_size = 0
            if artifact_path and Path(artifact_path).exists():
                artifact_hash = await self._compute_hash(artifact_path)
                artifact_size = Path(artifact_path).stat().st_size

            version = ModelVersion(
                model_name=model_name,
                version=new_version,
                artifact_path=artifact_path,
                artifact_hash=artifact_hash,
                artifact_size_bytes=artifact_size,
                training_job_id=training_job_id,
                parent_version_id=parent_version_id,
                metrics=metrics or ModelMetrics(),
                metadata=metadata or {},
                tags=tags or [],
            )

            self._versions[version.version_id] = version
            self._by_model[model_name].append(version.version_id)

            await self._save_registry()
            logger.info(f"[Registry] Created version: {model_name} v{new_version} ({version.version_id})")

            return version

    async def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version by ID."""
        return self._versions.get(version_id)

    async def get_latest(self, model_name: str, status: Optional[ModelStatus] = None) -> Optional[ModelVersion]:
        """Get latest version for a model."""
        version_ids = self._by_model.get(model_name, [])
        if not version_ids:
            return None

        versions = [self._versions[vid] for vid in version_ids]
        if status:
            versions = [v for v in versions if v.status == status]

        if not versions:
            return None

        return max(versions, key=lambda v: (v.version.major, v.version.minor, v.version.patch))

    async def get_deployed(self, model_name: str) -> Optional[ModelVersion]:
        """Get currently deployed version."""
        return await self.get_latest(model_name, ModelStatus.DEPLOYED)

    async def list_versions(
        self,
        model_name: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 50,
    ) -> List[ModelVersion]:
        """List versions with optional filters."""
        versions = list(self._versions.values())

        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by version descending
        versions.sort(key=lambda v: (v.version.major, v.version.minor, v.version.patch), reverse=True)

        return versions[:limit]

    async def update_status(self, version_id: str, status: ModelStatus) -> bool:
        """Update version status."""
        async with self._lock:
            version = self._versions.get(version_id)
            if not version:
                return False

            version.status = status
            if status == ModelStatus.DEPLOYED:
                version.deployed_at = time.time()

            await self._save_registry()
            return True

    async def update_metrics(self, version_id: str, metrics: ModelMetrics) -> bool:
        """Update version metrics."""
        async with self._lock:
            version = self._versions.get(version_id)
            if not version:
                return False

            version.metrics = metrics
            await self._save_registry()
            return True

    async def cleanup_old_versions(self, model_name: str, keep: int = RegistryConfig.MAX_VERSIONS_KEPT) -> int:
        """Clean up old versions, keeping the most recent N."""
        async with self._lock:
            version_ids = self._by_model.get(model_name, [])
            if len(version_ids) <= keep:
                return 0

            versions = [self._versions[vid] for vid in version_ids]
            versions.sort(key=lambda v: v.created_at, reverse=True)

            to_remove = versions[keep:]
            removed = 0

            for version in to_remove:
                # Don't remove deployed or staged versions
                if version.status in (ModelStatus.DEPLOYED, ModelStatus.STAGED):
                    continue

                # Remove artifact if exists
                if version.artifact_path and Path(version.artifact_path).exists():
                    try:
                        Path(version.artifact_path).unlink()
                    except Exception as e:
                        logger.warning(f"[Registry] Failed to remove artifact: {e}")

                # Remove from registry
                del self._versions[version.version_id]
                self._by_model[model_name].remove(version.version_id)
                removed += 1

            if removed > 0:
                await self._save_registry()
                logger.info(f"[Registry] Cleaned up {removed} old versions of {model_name}")

            return removed


# ============================================================================
# A/B Testing Manager
# ============================================================================

class ABTestManager:
    """
    Manage A/B testing between model versions.

    Supports traffic splitting, statistical significance testing,
    and automatic winner selection.
    """

    def __init__(self, version_manager: VersionManager):
        self._version_manager = version_manager
        self._tests: Dict[str, ABTestConfig] = {}
        self._results: Dict[str, ABTestResult] = {}
        self._samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)  # test_id -> samples
        self._lock = asyncio.Lock()

    async def create_test(
        self,
        name: str,
        control_version_id: str,
        treatment_version_id: str,
        traffic_split: float = 0.5,
        min_sample_size: int = RegistryConfig.AB_MIN_SAMPLE_SIZE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ABTestConfig:
        """Create a new A/B test."""
        async with self._lock:
            # Validate versions exist
            control = await self._version_manager.get_version(control_version_id)
            treatment = await self._version_manager.get_version(treatment_version_id)

            if not control or not treatment:
                raise ValueError("Control or treatment version not found")

            test = ABTestConfig(
                name=name,
                control_version_id=control_version_id,
                treatment_version_id=treatment_version_id,
                traffic_split=traffic_split,
                min_sample_size=min_sample_size,
                metadata=metadata or {},
            )

            self._tests[test.test_id] = test
            self._results[test.test_id] = ABTestResult(test_id=test.test_id)

            logger.info(
                f"[A/B Test] Created test: {name} "
                f"(control={control_version_id}, treatment={treatment_version_id})"
            )

            return test

    async def route_request(self, test_id: str) -> Tuple[str, str]:
        """
        Route a request to control or treatment.

        Returns:
            (version_id, group) where group is "control" or "treatment"
        """
        import random

        test = self._tests.get(test_id)
        if not test or not test.is_active:
            raise ValueError(f"Test {test_id} not found or not active")

        if random.random() < test.traffic_split:
            return test.treatment_version_id, "treatment"
        else:
            return test.control_version_id, "control"

    async def record_sample(
        self,
        test_id: str,
        version_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a sample observation."""
        async with self._lock:
            test = self._tests.get(test_id)
            if not test:
                return

            group = "treatment" if version_id == test.treatment_version_id else "control"

            self._samples[test_id].append({
                "version_id": version_id,
                "group": group,
                "metrics": metrics,
                "metadata": metadata or {},
                "timestamp": time.time(),
            })

            # Update result counts
            result = self._results.get(test_id)
            if result:
                if group == "control":
                    result.control_samples += 1
                else:
                    result.treatment_samples += 1

    async def analyze_test(self, test_id: str) -> ABTestResult:
        """Analyze A/B test results."""
        async with self._lock:
            test = self._tests.get(test_id)
            if not test:
                raise ValueError(f"Test {test_id} not found")

            samples = self._samples.get(test_id, [])
            result = self._results.get(test_id, ABTestResult(test_id=test_id))

            # Separate samples by group
            control_samples = [s for s in samples if s["group"] == "control"]
            treatment_samples = [s for s in samples if s["group"] == "treatment"]

            result.control_samples = len(control_samples)
            result.treatment_samples = len(treatment_samples)

            if not control_samples or not treatment_samples:
                return result

            # Aggregate metrics
            control_metrics = self._aggregate_metrics(control_samples)
            treatment_metrics = self._aggregate_metrics(treatment_samples)

            result.control_metrics = control_metrics
            result.treatment_metrics = treatment_metrics

            # Calculate improvements
            for metric in ["accuracy", "latency_p50_ms", "error_rate"]:
                control_val = getattr(control_metrics, metric, None)
                treatment_val = getattr(treatment_metrics, metric, None)

                if control_val and treatment_val and control_val > 0:
                    if metric in ["latency_p50_ms", "error_rate"]:
                        # Lower is better
                        improvement = (control_val - treatment_val) / control_val * 100
                    else:
                        # Higher is better
                        improvement = (treatment_val - control_val) / control_val * 100
                    result.improvement[metric] = improvement

            # Statistical significance (simplified t-test approximation)
            if min(result.control_samples, result.treatment_samples) >= test.min_sample_size:
                result.is_significant = self._check_significance(
                    control_samples, treatment_samples, test.significance_threshold
                )

                if result.is_significant:
                    # Determine winner based on primary metric (accuracy)
                    if result.improvement.get("accuracy", 0) > 0:
                        result.recommended_winner = test.treatment_version_id
                    else:
                        result.recommended_winner = test.control_version_id

                    result.confidence_level = 1 - test.significance_threshold

            self._results[test_id] = result
            return result

    def _aggregate_metrics(self, samples: List[Dict[str, Any]]) -> ModelMetrics:
        """Aggregate metrics from samples."""
        metrics = ModelMetrics()

        # Collect values for each metric
        metric_values: Dict[str, List[float]] = defaultdict(list)
        for sample in samples:
            for key, value in sample.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    metric_values[key].append(float(value))

        # Calculate averages
        import statistics as stats

        if "accuracy" in metric_values:
            metrics.accuracy = stats.mean(metric_values["accuracy"])
        if "latency" in metric_values:
            values = sorted(metric_values["latency"])
            metrics.latency_p50_ms = self._percentile(values, 50)
            metrics.latency_p95_ms = self._percentile(values, 95)
            metrics.latency_p99_ms = self._percentile(values, 99)
        if "error_rate" in metric_values:
            metrics.error_rate = stats.mean(metric_values["error_rate"])

        return metrics

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        k = (len(values) - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return values[f]
        return values[f] * (c - k) + values[c] * (k - f)

    def _check_significance(
        self,
        control: List[Dict],
        treatment: List[Dict],
        threshold: float,
    ) -> bool:
        """Check statistical significance (simplified)."""
        # Extract primary metric (accuracy) for both groups
        control_values = [s["metrics"].get("accuracy", 0) for s in control if "accuracy" in s.get("metrics", {})]
        treatment_values = [s["metrics"].get("accuracy", 0) for s in treatment if "accuracy" in s.get("metrics", {})]

        if len(control_values) < 10 or len(treatment_values) < 10:
            return False

        import statistics as stats

        # Simple t-test approximation
        try:
            control_mean = stats.mean(control_values)
            treatment_mean = stats.mean(treatment_values)
            control_std = stats.stdev(control_values) if len(control_values) > 1 else 0
            treatment_std = stats.stdev(treatment_values) if len(treatment_values) > 1 else 0

            if control_std == 0 and treatment_std == 0:
                return abs(control_mean - treatment_mean) > 0.01

            # Pooled standard error
            se = ((control_std ** 2 / len(control_values)) + (treatment_std ** 2 / len(treatment_values))) ** 0.5

            if se == 0:
                return False

            t_stat = abs(treatment_mean - control_mean) / se

            # Approximate p-value (simplified)
            # t > 2 roughly corresponds to p < 0.05
            return t_stat > 2.0

        except Exception:
            return False

    async def conclude_test(self, test_id: str, winner_version_id: Optional[str] = None) -> ABTestConfig:
        """Conclude an A/B test."""
        async with self._lock:
            test = self._tests.get(test_id)
            if not test:
                raise ValueError(f"Test {test_id} not found")

            test.is_active = False
            test.ended_at = time.time()
            test.winner = winner_version_id

            logger.info(f"[A/B Test] Concluded test: {test.name}, winner: {winner_version_id}")

            return test

    async def get_active_tests(self) -> List[ABTestConfig]:
        """Get all active tests."""
        return [t for t in self._tests.values() if t.is_active]

    async def get_test(self, test_id: str) -> Optional[ABTestConfig]:
        """Get test by ID."""
        return self._tests.get(test_id)


# ============================================================================
# Checkpoint Manager (Catastrophic Forgetting Prevention)
# ============================================================================

class CheckpointManager:
    """
    Manage model checkpoints for catastrophic forgetting prevention.

    Stores periodic checkpoints during training and enables
    rollback to previous states if performance degrades.
    """

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self._checkpoint_path = checkpoint_path or RegistryConfig.CHECKPOINT_PATH
        self._checkpoint_path.mkdir(parents=True, exist_ok=True)

        self._checkpoints: Dict[str, Checkpoint] = {}
        self._by_version: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def save_checkpoint(
        self,
        version_id: str,
        model_state: Any,  # Model state dict or path
        step: int,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """Save a training checkpoint."""
        async with self._lock:
            checkpoint = Checkpoint(
                version_id=version_id,
                step=step,
                epoch=epoch,
                loss=loss,
                metrics=metrics or {},
                metadata=metadata or {},
            )

            # Save checkpoint to disk
            checkpoint_dir = self._checkpoint_path / version_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint.checkpoint_id}.pt"

            try:
                import torch
                if isinstance(model_state, (dict, type(None))):
                    torch.save(model_state, checkpoint_file)
                elif isinstance(model_state, (str, Path)):
                    shutil.copy(model_state, checkpoint_file)
                else:
                    # Try to get state_dict
                    torch.save(model_state.state_dict(), checkpoint_file)
            except ImportError:
                # Fallback to pickle
                import pickle
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(model_state, f)

            checkpoint.checkpoint_path = str(checkpoint_file)
            checkpoint.checkpoint_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()

            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._by_version[version_id].append(checkpoint.checkpoint_id)

            logger.info(
                f"[Checkpoint] Saved checkpoint: {checkpoint.checkpoint_id} "
                f"(step={step}, epoch={epoch}, loss={loss:.4f})"
            )

            return checkpoint

    async def load_checkpoint(self, checkpoint_id: str) -> Tuple[Optional[Any], Optional[Checkpoint]]:
        """Load a checkpoint."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint or not checkpoint.checkpoint_path:
            return None, None

        checkpoint_file = Path(checkpoint.checkpoint_path)
        if not checkpoint_file.exists():
            return None, checkpoint

        try:
            import torch
            state = torch.load(checkpoint_file, map_location="cpu")
            return state, checkpoint
        except ImportError:
            import pickle
            with open(checkpoint_file, "rb") as f:
                return pickle.load(f), checkpoint

    async def get_best_checkpoint(
        self,
        version_id: str,
        metric: str = "loss",
        minimize: bool = True,
    ) -> Optional[Checkpoint]:
        """Get the best checkpoint for a version."""
        checkpoint_ids = self._by_version.get(version_id, [])
        if not checkpoint_ids:
            return None

        checkpoints = [self._checkpoints[cid] for cid in checkpoint_ids]

        if metric == "loss":
            key_func = lambda c: c.loss
        else:
            key_func = lambda c: c.metrics.get(metric, float("inf") if minimize else float("-inf"))

        if minimize:
            return min(checkpoints, key=key_func)
        else:
            return max(checkpoints, key=key_func)

    async def list_checkpoints(self, version_id: str) -> List[Checkpoint]:
        """List all checkpoints for a version."""
        checkpoint_ids = self._by_version.get(version_id, [])
        return [self._checkpoints[cid] for cid in checkpoint_ids]

    async def cleanup_old_checkpoints(
        self,
        version_id: str,
        keep_best: int = 3,
        keep_latest: int = 2,
    ) -> int:
        """Clean up old checkpoints, keeping best and latest."""
        async with self._lock:
            checkpoint_ids = self._by_version.get(version_id, [])
            if not checkpoint_ids:
                return 0

            checkpoints = [self._checkpoints[cid] for cid in checkpoint_ids]

            # Sort by loss (ascending) and created_at (descending)
            by_loss = sorted(checkpoints, key=lambda c: c.loss)[:keep_best]
            by_time = sorted(checkpoints, key=lambda c: c.created_at, reverse=True)[:keep_latest]

            to_keep = set(c.checkpoint_id for c in by_loss + by_time)
            to_remove = [c for c in checkpoints if c.checkpoint_id not in to_keep]

            removed = 0
            for checkpoint in to_remove:
                try:
                    if checkpoint.checkpoint_path:
                        Path(checkpoint.checkpoint_path).unlink(missing_ok=True)
                    del self._checkpoints[checkpoint.checkpoint_id]
                    self._by_version[version_id].remove(checkpoint.checkpoint_id)
                    removed += 1
                except Exception as e:
                    logger.warning(f"[Checkpoint] Failed to remove {checkpoint.checkpoint_id}: {e}")

            return removed


# ============================================================================
# Deployment Manager
# ============================================================================

class DeploymentManager:
    """
    Manage model deployment lifecycle.

    Handles deployment notifications to JARVIS/Prime,
    rollback operations, and deployment status tracking.
    """

    def __init__(
        self,
        version_manager: VersionManager,
        checkpoint_manager: CheckpointManager,
    ):
        self._version_manager = version_manager
        self._checkpoint_manager = checkpoint_manager

        self._deployments: Dict[str, DeploymentRecord] = {}
        self._active_deployments: Dict[str, str] = {}  # target -> deployment_id
        self._lock = asyncio.Lock()

        # Notification callbacks
        self._on_deploy: List[Callable[[ModelVersion, DeploymentTarget], Awaitable[None]]] = []
        self._on_rollback: List[Callable[[ModelVersion, str], Awaitable[None]]] = []

    def on_deploy(self, callback: Callable[[ModelVersion, DeploymentTarget], Awaitable[None]]):
        """Register deployment callback."""
        self._on_deploy.append(callback)

    def on_rollback(self, callback: Callable[[ModelVersion, str], Awaitable[None]]):
        """Register rollback callback."""
        self._on_rollback.append(callback)

    async def deploy(
        self,
        version_id: str,
        target: DeploymentTarget = DeploymentTarget.JARVIS,
        notify: bool = True,
    ) -> DeploymentRecord:
        """Deploy a model version."""
        async with self._lock:
            version = await self._version_manager.get_version(version_id)
            if not version:
                raise ValueError(f"Version {version_id} not found")

            # Create deployment record
            deployment = DeploymentRecord(
                version_id=version_id,
                target=target,
                status="deploying",
            )

            self._deployments[deployment.deployment_id] = deployment

            try:
                # Update version status
                await self._version_manager.update_status(version_id, ModelStatus.DEPLOYED)

                # Mark previous deployment as deprecated
                prev_deployment_id = self._active_deployments.get(target.value)
                if prev_deployment_id:
                    prev = self._deployments.get(prev_deployment_id)
                    if prev:
                        prev_version = await self._version_manager.get_version(prev.version_id)
                        if prev_version:
                            await self._version_manager.update_status(
                                prev.version_id, ModelStatus.DEPRECATED
                            )

                self._active_deployments[target.value] = deployment.deployment_id

                # Send notifications
                if notify:
                    await self._notify_deployment(version, target)
                    deployment.notification_sent = True

                deployment.status = "deployed"
                deployment.completed_at = time.time()

                # Fire callbacks
                for callback in self._on_deploy:
                    try:
                        await callback(version, target)
                    except Exception as e:
                        logger.error(f"[Deploy] Callback error: {e}")

                logger.info(
                    f"[Deploy] Deployed {version.model_name} v{version.version} "
                    f"to {target.value}"
                )

            except Exception as e:
                deployment.status = "failed"
                deployment.error = str(e)
                deployment.completed_at = time.time()
                raise

            return deployment

    async def rollback(
        self,
        target: DeploymentTarget = DeploymentTarget.JARVIS,
        to_version_id: Optional[str] = None,
        reason: str = "manual",
    ) -> DeploymentRecord:
        """Rollback to a previous version."""
        async with self._lock:
            current_deployment_id = self._active_deployments.get(target.value)
            if not current_deployment_id:
                raise ValueError(f"No active deployment for {target.value}")

            current = self._deployments[current_deployment_id]

            # Find version to rollback to
            if to_version_id:
                rollback_version = await self._version_manager.get_version(to_version_id)
            else:
                # Find previous deployed version
                versions = await self._version_manager.list_versions(status=ModelStatus.DEPRECATED)
                if not versions:
                    raise ValueError("No previous version to rollback to")
                rollback_version = versions[0]

            if not rollback_version:
                raise ValueError("Rollback version not found")

            # Create new deployment for rollback
            deployment = DeploymentRecord(
                version_id=rollback_version.version_id,
                target=target,
                status="deploying",
                rollback_of=current_deployment_id,
            )

            self._deployments[deployment.deployment_id] = deployment

            try:
                # Update statuses
                await self._version_manager.update_status(
                    current.version_id, ModelStatus.ROLLED_BACK
                )
                await self._version_manager.update_status(
                    rollback_version.version_id, ModelStatus.DEPLOYED
                )

                self._active_deployments[target.value] = deployment.deployment_id

                # Notify
                await self._notify_rollback(rollback_version, target, reason)
                deployment.notification_sent = True

                deployment.status = "deployed"
                deployment.completed_at = time.time()

                # Fire callbacks
                for callback in self._on_rollback:
                    try:
                        await callback(rollback_version, reason)
                    except Exception as e:
                        logger.error(f"[Rollback] Callback error: {e}")

                logger.warning(
                    f"[Rollback] Rolled back to {rollback_version.model_name} "
                    f"v{rollback_version.version} (reason: {reason})"
                )

            except Exception as e:
                deployment.status = "failed"
                deployment.error = str(e)
                deployment.completed_at = time.time()
                raise

            return deployment

    async def _notify_deployment(self, version: ModelVersion, target: DeploymentTarget):
        """Send deployment notification to target systems."""
        # v152.0: Check cloud mode before attempting local notifications
        if is_cloud_mode_active():
            logger.info(
                f"[Deploy] [v152.0] Cloud mode active - skipping local notifications "
                f"for {version.model_name} v{version.version}"
            )
            # In cloud mode, we could optionally notify the cloud endpoint
            cloud_url = get_effective_jarvis_url("")
            if cloud_url:
                try:
                    import aiohttp
                    payload = {
                        "event": "model_deployed",
                        "version_id": version.version_id,
                        "model_name": version.model_name,
                        "version": str(version.version),
                        "artifact_path": version.artifact_path,
                        "deployed_at": time.time(),
                        "source": "reactor-core",
                        "cloud_mode": True,
                    }
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{cloud_url}/reactor-core/model/deployed",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status == 200:
                                logger.info(f"[Deploy] [v152.0] Notified cloud endpoint")
                except Exception as e:
                    logger.debug(f"[Deploy] [v152.0] Cloud notification failed (non-fatal): {e}")
            return

        # Standard local notification
        targets = []
        if target in (DeploymentTarget.JARVIS, DeploymentTarget.BOTH):
            targets.append(("jarvis", RegistryConfig.get_jarvis_api_url()))
        if target in (DeploymentTarget.PRIME, DeploymentTarget.BOTH):
            targets.append(("prime", RegistryConfig.get_prime_api_url()))

        try:
            import aiohttp

            payload = {
                "event": "model_deployed",
                "version_id": version.version_id,
                "model_name": version.model_name,
                "version": str(version.version),
                "artifact_path": version.artifact_path,
                "deployed_at": time.time(),
            }

            for name, base_url in targets:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{base_url}/reactor-core/model/deployed",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as response:
                            if response.status == 200:
                                logger.info(f"[Deploy] Notified {name}")
                            else:
                                logger.warning(f"[Deploy] {name} notification failed: {response.status}")
                except Exception as e:
                    logger.warning(f"[Deploy] Failed to notify {name}: {e}")

        except ImportError:
            logger.warning("[Deploy] aiohttp not available, skipping notifications")

    async def _notify_rollback(self, version: ModelVersion, target: DeploymentTarget, reason: str):
        """Send rollback notification."""
        # v152.0: Check cloud mode before attempting local notifications
        if is_cloud_mode_active():
            logger.info(
                f"[Rollback] [v152.0] Cloud mode active - using cloud endpoint for "
                f"{version.model_name} v{version.version} rollback notification"
            )
            base_url = get_effective_jarvis_url("")
            if not base_url:
                logger.debug("[Rollback] [v152.0] No cloud endpoint available, skipping notification")
                return
        else:
            base_url = (
                RegistryConfig.get_jarvis_api_url() if target == DeploymentTarget.JARVIS
                else RegistryConfig.get_prime_api_url()
            )

        try:
            import aiohttp

            payload = {
                "event": "model_rollback",
                "version_id": version.version_id,
                "model_name": version.model_name,
                "version": str(version.version),
                "reason": reason,
                "rolled_back_at": time.time(),
                "cloud_mode": is_cloud_mode_active(),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/reactor-core/model/rollback",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        logger.info(f"[Rollback] Notified {target.value}")

        except Exception as e:
            logger.warning(f"[Rollback] Failed to send notification: {e}")

    async def get_deployment(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    async def get_active_deployment(self, target: DeploymentTarget) -> Optional[DeploymentRecord]:
        """Get active deployment for target."""
        deployment_id = self._active_deployments.get(target.value)
        if deployment_id:
            return self._deployments.get(deployment_id)
        return None

    async def list_deployments(self, limit: int = 50) -> List[DeploymentRecord]:
        """List recent deployments."""
        deployments = list(self._deployments.values())
        deployments.sort(key=lambda d: d.started_at, reverse=True)
        return deployments[:limit]


# ============================================================================
# Model Registry (Main Class)
# ============================================================================

class ModelRegistry:
    """
    Unified model registry system.

    Provides comprehensive model lifecycle management including
    versioning, A/B testing, checkpointing, and deployment.
    """

    def __init__(self):
        self._version_manager = VersionManager()
        self._checkpoint_manager = CheckpointManager()
        self._ab_manager = ABTestManager(self._version_manager)
        self._deployment_manager = DeploymentManager(
            self._version_manager, self._checkpoint_manager
        )

    @property
    def versions(self) -> VersionManager:
        return self._version_manager

    @property
    def checkpoints(self) -> CheckpointManager:
        return self._checkpoint_manager

    @property
    def ab_tests(self) -> ABTestManager:
        return self._ab_manager

    @property
    def deployments(self) -> DeploymentManager:
        return self._deployment_manager

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status."""
        versions = await self._version_manager.list_versions()
        active_tests = await self._ab_manager.get_active_tests()
        active_deployments = {}

        for target in DeploymentTarget:
            if target == DeploymentTarget.BOTH:
                continue
            dep = await self._deployment_manager.get_active_deployment(target)
            if dep:
                version = await self._version_manager.get_version(dep.version_id)
                active_deployments[target.value] = {
                    "deployment_id": dep.deployment_id,
                    "version_id": dep.version_id,
                    "version": str(version.version) if version else "unknown",
                    "deployed_at": dep.completed_at,
                }

        return {
            "total_versions": len(versions),
            "deployed_versions": sum(1 for v in versions if v.status == ModelStatus.DEPLOYED),
            "draft_versions": sum(1 for v in versions if v.status == ModelStatus.DRAFT),
            "active_ab_tests": len(active_tests),
            "active_deployments": active_deployments,
        }


# ============================================================================
# Global Registry Instance
# ============================================================================

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
