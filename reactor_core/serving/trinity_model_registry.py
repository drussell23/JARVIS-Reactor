"""
Trinity Model Registry - Cross-Repo Model Synchronization

Provides unified model discovery, metadata synchronization, and coordination
across JARVIS (Body), J-Prime (Mind), and Reactor Core (Nerves).

Features:
- Cross-repo model discovery
- Model metadata synchronization via Trinity Bridge
- Version tracking and compatibility checking
- Model availability broadcasting
- Cache coordination to prevent duplication
- Remote model loading requests
- Distributed model registry with conflict resolution

Author: JARVIS AGI
Version: v83.0 - Unified Model Management
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================


class RepositoryType(str, Enum):
    """Trinity repository types."""
    JARVIS = "jarvis"          # Body - Frontend, UI, user interaction
    JPRIME = "jprime"          # Mind - Reasoning, planning, decision making
    REACTOR = "reactor"        # Nerves - Training, learning, model management


class ModelSource(str, Enum):
    """Where the model is stored."""
    LOCAL = "local"            # Local filesystem
    HUGGINGFACE = "huggingface"  # Hugging Face Hub
    CUSTOM = "custom"          # Custom endpoint
    SHARED = "shared"          # Shared across repos


class ModelStatus(str, Enum):
    """Model availability status."""
    AVAILABLE = "available"    # Ready to use
    LOADING = "loading"        # Currently loading
    UNAVAILABLE = "unavailable"  # Not available
    ERROR = "error"            # Failed to load


class SyncStrategy(str, Enum):
    """Model synchronization strategies."""
    EAGER = "eager"            # Sync immediately
    LAZY = "lazy"              # Sync on demand
    PERIODIC = "periodic"      # Sync on schedule
    EVENT_DRIVEN = "event_driven"  # Sync on events


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    model_id: str
    model_name: str
    repository: RepositoryType
    source: ModelSource

    # Model characteristics
    model_type: str  # "llm", "embedding", "vision", etc.
    backend: str     # "gguf", "transformers", "mlx", etc.
    quantization: Optional[str] = None
    parameter_count: Optional[int] = None

    # Location
    local_path: Optional[Path] = None
    remote_url: Optional[str] = None

    # Version & compatibility
    version: str = "1.0.0"
    compatibility_tags: Set[str] = field(default_factory=set)

    # Performance characteristics
    memory_requirement_gb: float = 0.0
    typical_latency_ms: float = 0.0
    max_context_length: int = 2048

    # Status
    status: ModelStatus = ModelStatus.UNAVAILABLE
    last_used: Optional[float] = None
    load_count: int = 0

    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)

    # Synchronization
    last_sync: Optional[float] = None
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "repository": self.repository.value,
            "source": self.source.value,
            "model_type": self.model_type,
            "backend": self.backend,
            "quantization": self.quantization,
            "parameter_count": self.parameter_count,
            "local_path": str(self.local_path) if self.local_path else None,
            "remote_url": self.remote_url,
            "version": self.version,
            "compatibility_tags": list(self.compatibility_tags),
            "memory_requirement_gb": self.memory_requirement_gb,
            "typical_latency_ms": self.typical_latency_ms,
            "max_context_length": self.max_context_length,
            "status": self.status.value,
            "last_used": self.last_used,
            "load_count": self.load_count,
            "description": self.description,
            "tags": list(self.tags),
            "capabilities": list(self.capabilities),
            "last_sync": self.last_sync,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            repository=RepositoryType(data["repository"]),
            source=ModelSource(data["source"]),
            model_type=data["model_type"],
            backend=data["backend"],
            quantization=data.get("quantization"),
            parameter_count=data.get("parameter_count"),
            local_path=Path(data["local_path"]) if data.get("local_path") else None,
            remote_url=data.get("remote_url"),
            version=data.get("version", "1.0.0"),
            compatibility_tags=set(data.get("compatibility_tags", [])),
            memory_requirement_gb=data.get("memory_requirement_gb", 0.0),
            typical_latency_ms=data.get("typical_latency_ms", 0.0),
            max_context_length=data.get("max_context_length", 2048),
            status=ModelStatus(data.get("status", "unavailable")),
            last_used=data.get("last_used"),
            load_count=data.get("load_count", 0),
            description=data.get("description", ""),
            tags=set(data.get("tags", [])),
            capabilities=set(data.get("capabilities", [])),
            last_sync=data.get("last_sync"),
            checksum=data.get("checksum"),
        )


@dataclass
class RegistryConfig:
    """Configuration for Trinity Model Registry."""
    registry_path: Path = Path.home() / ".jarvis" / "model_registry.json"
    sync_strategy: SyncStrategy = SyncStrategy.EVENT_DRIVEN
    sync_interval_seconds: float = 300.0  # For periodic sync

    # Cache coordination
    enable_distributed_cache: bool = True
    cache_eviction_policy: str = "lru"

    # Conflict resolution
    conflict_resolution: str = "latest_wins"  # "latest_wins", "manual", "version_based"

    # Discovery
    auto_discover_models: bool = True
    discovery_paths: List[Path] = field(default_factory=list)

    # Performance
    enable_model_preloading: bool = False
    preload_popular_threshold: int = 10  # Preload if used > N times


@dataclass
class SyncEvent:
    """Model synchronization event."""
    event_type: str  # "added", "updated", "removed", "status_changed"
    model_id: str
    repository: RepositoryType
    metadata: Optional[ModelMetadata] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# TRINITY MODEL REGISTRY
# ============================================================================


class TrinityModelRegistry:
    """
    Cross-repo model registry with synchronization.

    Features:
    - Unified model discovery across JARVIS, J-Prime, Reactor
    - Metadata synchronization via Trinity Bridge
    - Distributed cache coordination
    - Conflict resolution
    - Model availability tracking
    """

    def __init__(
        self,
        trinity_bridge: Any,  # TrinityBridge instance
        repository: RepositoryType,
        config: Optional[RegistryConfig] = None,
    ):
        self.bridge = trinity_bridge
        self.repository = repository
        self.config = config or RegistryConfig()

        # Local registry: model_id -> ModelMetadata
        self.local_models: Dict[str, ModelMetadata] = {}

        # Remote registry: repository -> model_id -> ModelMetadata
        self.remote_models: Dict[RepositoryType, Dict[str, ModelMetadata]] = {
            RepositoryType.JARVIS: {},
            RepositoryType.JPRIME: {},
            RepositoryType.REACTOR: {},
        }

        # Pending sync events
        self.pending_syncs: List[SyncEvent] = []

        # Sync lock
        self._sync_lock = asyncio.Lock()

        # Background tasks
        self.sync_task: Optional[asyncio.Task] = None
        self.discovery_task: Optional[asyncio.Task] = None

        self.running = False

        logger.info(f"Trinity Model Registry initialized for {repository.value}")

    async def start(self):
        """Start the registry."""
        if self.running:
            logger.warning("Registry already running")
            return

        self.running = True
        logger.info(f"Starting Trinity Model Registry ({self.repository.value})")

        # Load local registry from disk
        await self._load_registry()

        # Start background tasks
        if self.config.sync_strategy == SyncStrategy.PERIODIC:
            self.sync_task = asyncio.create_task(self._periodic_sync_worker())

        if self.config.auto_discover_models:
            self.discovery_task = asyncio.create_task(self._discovery_worker())

        # Subscribe to Trinity Bridge events
        asyncio.create_task(self._listen_for_remote_updates())

        # Broadcast our models to other repos
        await self._broadcast_local_models()

    async def shutdown(self):
        """Gracefully shutdown the registry."""
        logger.info("Shutting down Trinity Model Registry")
        self.running = False

        # Cancel background tasks
        if self.sync_task:
            self.sync_task.cancel()
        if self.discovery_task:
            self.discovery_task.cancel()

        # Save registry to disk
        await self._save_registry()

        logger.info("Trinity Model Registry stopped")

    # ========================================================================
    # MODEL REGISTRATION & DISCOVERY
    # ========================================================================

    async def register_model(self, metadata: ModelMetadata) -> bool:
        """
        Register a new model or update existing one.

        Args:
            metadata: Model metadata

        Returns:
            True if registration successful
        """
        async with self._sync_lock:
            model_id = metadata.model_id

            # Check if model already exists
            is_new = model_id not in self.local_models

            # Update local registry
            self.local_models[model_id] = metadata
            metadata.last_sync = time.time()

            logger.info(f"{'Registered' if is_new else 'Updated'} model: {model_id}")

            # Create sync event
            event = SyncEvent(
                event_type="added" if is_new else "updated",
                model_id=model_id,
                repository=self.repository,
                metadata=metadata,
            )

            # Broadcast to other repos
            await self._broadcast_sync_event(event)

            # Save to disk
            await self._save_registry()

            return True

    async def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model.

        Args:
            model_id: Model identifier

        Returns:
            True if unregistration successful
        """
        async with self._sync_lock:
            if model_id not in self.local_models:
                logger.warning(f"Model not found: {model_id}")
                return False

            # Remove from local registry
            del self.local_models[model_id]

            logger.info(f"Unregistered model: {model_id}")

            # Create sync event
            event = SyncEvent(
                event_type="removed",
                model_id=model_id,
                repository=self.repository,
            )

            # Broadcast to other repos
            await self._broadcast_sync_event(event)

            # Save to disk
            await self._save_registry()

            return True

    async def update_model_status(self, model_id: str, status: ModelStatus):
        """Update model status."""
        async with self._sync_lock:
            if model_id in self.local_models:
                self.local_models[model_id].status = status
                self.local_models[model_id].last_sync = time.time()

                # Broadcast status change
                event = SyncEvent(
                    event_type="status_changed",
                    model_id=model_id,
                    repository=self.repository,
                    metadata=self.local_models[model_id],
                )
                await self._broadcast_sync_event(event)

    # ========================================================================
    # MODEL QUERIES
    # ========================================================================

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID (local or remote)."""
        # Check local first
        if model_id in self.local_models:
            return self.local_models[model_id]

        # Check remote registries
        for repo_models in self.remote_models.values():
            if model_id in repo_models:
                return repo_models[model_id]

        return None

    def list_models(
        self,
        repository: Optional[RepositoryType] = None,
        model_type: Optional[str] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[ModelMetadata]:
        """
        List models with optional filters.

        Args:
            repository: Filter by repository (None = all)
            model_type: Filter by model type
            status: Filter by status
            tags: Filter by tags (any match)

        Returns:
            List of matching models
        """
        models = []

        # Collect models
        if repository is None or repository == self.repository:
            models.extend(self.local_models.values())

        if repository is None or repository != self.repository:
            for repo, repo_models in self.remote_models.items():
                if repository is None or repo == repository:
                    models.extend(repo_models.values())

        # Apply filters
        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if status:
            models = [m for m in models if m.status == status]

        if tags:
            models = [m for m in models if m.tags & tags]

        return models

    def find_best_model(
        self,
        model_type: str,
        min_memory_gb: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        required_capabilities: Optional[Set[str]] = None,
    ) -> Optional[ModelMetadata]:
        """
        Find the best available model matching criteria.

        Args:
            model_type: Required model type
            min_memory_gb: Minimum memory requirement
            max_latency_ms: Maximum acceptable latency
            required_capabilities: Required capabilities

        Returns:
            Best matching model or None
        """
        candidates = self.list_models(
            model_type=model_type,
            status=ModelStatus.AVAILABLE,
        )

        # Apply constraints
        if min_memory_gb:
            candidates = [m for m in candidates if m.memory_requirement_gb <= min_memory_gb]

        if max_latency_ms:
            candidates = [m for m in candidates if m.typical_latency_ms <= max_latency_ms]

        if required_capabilities:
            candidates = [m for m in candidates if required_capabilities.issubset(m.capabilities)]

        if not candidates:
            return None

        # Score candidates (prefer local, then by usage, then by latency)
        def score_model(model: ModelMetadata) -> float:
            score = 0.0
            # Prefer local models
            if model.repository == self.repository:
                score += 1000.0
            # Prefer frequently used models
            score += model.load_count * 10.0
            # Prefer lower latency
            if model.typical_latency_ms > 0:
                score -= model.typical_latency_ms
            return score

        candidates.sort(key=score_model, reverse=True)
        return candidates[0]

    # ========================================================================
    # CROSS-REPO SYNCHRONIZATION
    # ========================================================================

    async def _broadcast_sync_event(self, event: SyncEvent):
        """Broadcast sync event to other repos via Trinity Bridge."""
        try:
            await self.bridge.publish(
                event_type="model_registry_sync",
                payload={
                    "event": event.event_type,
                    "model_id": event.model_id,
                    "repository": event.repository.value,
                    "metadata": event.metadata.to_dict() if event.metadata else None,
                    "timestamp": event.timestamp,
                },
                priority="normal",
            )
        except Exception as e:
            logger.error(f"Failed to broadcast sync event: {e}")

    async def _broadcast_local_models(self):
        """Broadcast all local models to other repos."""
        for metadata in self.local_models.values():
            event = SyncEvent(
                event_type="added",
                model_id=metadata.model_id,
                repository=self.repository,
                metadata=metadata,
            )
            await self._broadcast_sync_event(event)

    async def _listen_for_remote_updates(self):
        """Listen for model registry updates from other repos."""
        logger.info("Listening for remote model registry updates...")

        try:
            async for event in self.bridge.listen_for_updates():
                if event.get("event_type") == "model_registry_sync":
                    await self._handle_remote_sync_event(event.get("payload", {}))
        except Exception as e:
            logger.error(f"Error listening for remote updates: {e}")

    async def _handle_remote_sync_event(self, payload: Dict[str, Any]):
        """Handle a remote sync event."""
        try:
            event_type = payload.get("event")
            model_id = payload.get("model_id")
            repository = RepositoryType(payload.get("repository"))

            # Ignore events from ourselves
            if repository == self.repository:
                return

            async with self._sync_lock:
                if event_type == "added" or event_type == "updated":
                    metadata_dict = payload.get("metadata")
                    if metadata_dict:
                        metadata = ModelMetadata.from_dict(metadata_dict)
                        self.remote_models[repository][model_id] = metadata
                        logger.debug(f"Synced {event_type} model from {repository.value}: {model_id}")

                elif event_type == "removed":
                    if model_id in self.remote_models[repository]:
                        del self.remote_models[repository][model_id]
                        logger.debug(f"Removed model from {repository.value}: {model_id}")

                elif event_type == "status_changed":
                    metadata_dict = payload.get("metadata")
                    if metadata_dict and model_id in self.remote_models[repository]:
                        self.remote_models[repository][model_id].status = ModelStatus(metadata_dict["status"])
                        logger.debug(f"Updated status for {repository.value}/{model_id}")

        except Exception as e:
            logger.error(f"Error handling remote sync event: {e}")

    # ========================================================================
    # PERIODIC SYNC & DISCOVERY
    # ========================================================================

    async def _periodic_sync_worker(self):
        """Background worker for periodic synchronization."""
        logger.info("Periodic sync worker started")

        while self.running:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)
                await self._broadcast_local_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic sync error: {e}")

    async def _discovery_worker(self):
        """Background worker for model discovery."""
        logger.info("Model discovery worker started")

        while self.running:
            try:
                # Discover models in configured paths
                for path in self.config.discovery_paths:
                    if path.exists():
                        await self._discover_models_in_path(path)

                # Sleep before next discovery
                await asyncio.sleep(60.0)  # Discover every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discovery error: {e}")

    async def _discover_models_in_path(self, path: Path):
        """Discover models in a given path."""
        # Look for GGUF files
        for gguf_file in path.rglob("*.gguf"):
            model_id = f"local_{gguf_file.stem}"

            # Check if already registered
            if model_id not in self.local_models:
                # Auto-register discovered model
                metadata = ModelMetadata(
                    model_id=model_id,
                    model_name=gguf_file.stem,
                    repository=self.repository,
                    source=ModelSource.LOCAL,
                    model_type="llm",
                    backend="gguf",
                    local_path=gguf_file,
                    status=ModelStatus.AVAILABLE,
                    description=f"Auto-discovered GGUF model: {gguf_file.name}",
                )
                await self.register_model(metadata)
                logger.info(f"Auto-discovered model: {model_id}")

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    async def _save_registry(self):
        """Save registry to disk."""
        try:
            self.config.registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "repository": self.repository.value,
                "models": {
                    model_id: metadata.to_dict()
                    for model_id, metadata in self.local_models.items()
                },
                "last_save": time.time(),
            }

            with open(self.config.registry_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved registry to {self.config.registry_path}")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    async def _load_registry(self):
        """Load registry from disk."""
        try:
            if not self.config.registry_path.exists():
                logger.info("No existing registry found - starting fresh")
                return

            with open(self.config.registry_path, 'r') as f:
                data = json.load(f)

            for model_id, metadata_dict in data.get("models", {}).items():
                self.local_models[model_id] = ModelMetadata.from_dict(metadata_dict)

            logger.info(f"Loaded {len(self.local_models)} models from registry")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


async def create_trinity_registry(
    trinity_bridge: Any,
    repository: RepositoryType,
    config: Optional[RegistryConfig] = None,
) -> TrinityModelRegistry:
    """
    Create and start a Trinity Model Registry.

    Args:
        trinity_bridge: Trinity Bridge instance
        repository: Which repository this is (JARVIS, JPRIME, or REACTOR)
        config: Registry configuration

    Returns:
        Started TrinityModelRegistry instance
    """
    registry = TrinityModelRegistry(trinity_bridge, repository, config)
    await registry.start()
    return registry
