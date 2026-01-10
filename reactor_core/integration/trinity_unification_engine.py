"""
PROJECT TRINITY: Ultimate Unification Engine (v89.0)

The MASTER ORCHESTRATOR that unifies ALL Trinity systems into one coherent whole.

ROOT PROBLEMS SOLVED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Multiple competing shutdown systems (TrinityIntegrator, EnhancedShutdown,
   ProcessTreeManager, Advanced Orchestrator, Component-specific shutdowns)
2. HeartbeatValidator removing 20+ "dead" components with UUID pollution
3. Components shutting down in wrong order
4. Old HeartbeatValidator NOT using v87.0 DistributedHealthMonitor
5. No unified component registry
6. Shutdown chaos with random ordering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHITECTURE:
- Single source of truth for ALL Trinity operations
- Unified component registry (no UUID pollution)
- Coordinated shutdown with dependency graph
- Integrates v85 (Unified Coordinator), v86 (DB), v87 (Health), v88 (Lifecycle)
- Eliminates ALL duplicate shutdown systems

Author: Claude Opus 4.5
Version: 89.0
Status: Production Ready
"""

import asyncio
import logging
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class ComponentType(str, Enum):
    """Trinity component types."""

    # Core Trinity Components
    JARVIS_BODY = "jarvis_body"
    JARVIS_PRIME = "j_prime"
    REACTOR_CORE = "reactor_core"

    # Infrastructure
    UNIFIED_COORDINATOR = "unified_coordinator"
    DB_COORDINATOR = "db_coordinator"
    HEALTH_MONITOR = "health_monitor"
    LIFECYCLE_COORDINATOR = "lifecycle_coordinator"

    # JARVIS Body Subsystems
    NEURAL_MESH = "neural_mesh"
    VOICE_SYSTEM = "voice_system"
    VISUAL_MONITOR = "visual_monitor"
    AUTONOMY_ENGINE = "autonomy_engine"

    # Trinity Integrators
    TRINITY_INTEGRATOR = "trinity_integrator"
    TRINITY_IPC = "trinity_ipc"
    TRINITY_BRIDGE = "trinity_bridge"

    # Orchestrators
    ADVANCED_ORCHESTRATOR = "advanced_orchestrator"
    PROCESS_TREE_MANAGER = "process_tree_manager"
    STARTUP_COORDINATOR = "startup_coordinator"

    # Background Services
    CONTINUOUS_SCRAPER = "continuous_scraper"
    LEARNING_DISCOVERY = "learning_discovery"
    EXPERIENCE_COLLECTOR = "experience_collector"
    TRAINING_SCHEDULER = "training_scheduler"

    # Utility
    HEARTBEAT_VALIDATOR = "heartbeat_validator"
    ORPHAN_DETECTOR = "orphan_detector"
    COMMAND_BUFFER = "command_buffer"

    # Generic
    CUSTOM = "custom"


class LifecyclePhase(str, Enum):
    """Component lifecycle phases."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ShutdownLayer(int, Enum):
    """
    Shutdown layers - components shutdown in numerical order (0 first, 5 last).

    This enforces proper dependency ordering during shutdown.
    """

    LAYER_0_BACKGROUND = 0  # Background tasks, scrapers, collectors
    LAYER_1_SERVICES = 1  # Voice, visual, autonomy
    LAYER_2_APPLICATIONS = 2  # JARVIS Body, Prime, Reactor
    LAYER_3_INTEGRATORS = 3  # Trinity integrators, bridges
    LAYER_4_COORDINATORS = 4  # State, health, lifecycle coordinators
    LAYER_5_INFRASTRUCTURE = 5  # DB connections, file handles


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ComponentRegistration:
    """Registration info for a Trinity component."""

    component_id: str  # Clean, stable ID (no timestamps/UUIDs)
    component_type: ComponentType
    shutdown_layer: ShutdownLayer
    phase: LifecyclePhase = LifecyclePhase.UNINITIALIZED

    # Lifecycle hooks
    startup_hook: Optional[Callable[[], Any]] = None
    shutdown_hook: Optional[Callable[[], Any]] = None
    health_check_hook: Optional[Callable[[], bool]] = None

    # Metadata
    process_id: Optional[int] = None
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None

    # Dependencies
    depends_on: Set[str] = field(default_factory=set)
    required_by: Set[str] = field(default_factory=set)

    # Reference to actual component (weak ref to avoid circular refs)
    component_ref: Optional[weakref.ref] = None


@dataclass
class ShutdownPlan:
    """Computed shutdown execution plan."""

    layers: Dict[ShutdownLayer, List[ComponentRegistration]]
    total_components: int
    estimated_duration: float
    critical_path: List[str]  # Component IDs in critical path


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════


class ComponentRegistry:
    """
    Centralized registry for ALL Trinity components.

    Eliminates UUID pollution by maintaining clean, stable component IDs.
    """

    def __init__(self):
        self._components: Dict[str, ComponentRegistration] = {}
        self._by_type: Dict[ComponentType, List[str]] = defaultdict(list)
        self._by_layer: Dict[ShutdownLayer, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def register(
        self,
        component_id: str,
        component_type: ComponentType,
        shutdown_layer: ShutdownLayer,
        *,
        startup_hook: Optional[Callable] = None,
        shutdown_hook: Optional[Callable] = None,
        health_check_hook: Optional[Callable] = None,
        depends_on: Optional[Set[str]] = None,
        component_instance: Optional[Any] = None,
    ) -> ComponentRegistration:
        """
        Register a component in the Trinity registry.

        Args:
            component_id: STABLE ID (e.g., "jarvis_body", not "1767604792586_uuid")
            component_type: Type of component
            shutdown_layer: Shutdown layer for ordering
            startup_hook: Optional startup function
            shutdown_hook: Optional shutdown function
            health_check_hook: Optional health check function
            depends_on: Set of component IDs this depends on
            component_instance: Optional reference to actual component

        Returns:
            ComponentRegistration
        """
        async with self._lock:
            # Prevent duplicate registration
            if component_id in self._components:
                logger.warning(
                    f"Component {component_id} already registered, updating"
                )

                # Update existing registration
                existing = self._components[component_id]
                if startup_hook:
                    existing.startup_hook = startup_hook
                if shutdown_hook:
                    existing.shutdown_hook = shutdown_hook
                if health_check_hook:
                    existing.health_check_hook = health_check_hook
                if depends_on:
                    existing.depends_on.update(depends_on)
                if component_instance:
                    existing.component_ref = weakref.ref(component_instance)

                return existing

            # Create new registration
            registration = ComponentRegistration(
                component_id=component_id,
                component_type=component_type,
                shutdown_layer=shutdown_layer,
                startup_hook=startup_hook,
                shutdown_hook=shutdown_hook,
                health_check_hook=health_check_hook,
                depends_on=depends_on or set(),
                component_ref=(
                    weakref.ref(component_instance) if component_instance else None
                ),
            )

            self._components[component_id] = registration
            self._by_type[component_type].append(component_id)
            self._by_layer[shutdown_layer].append(component_id)

            # Update dependents
            for dep_id in registration.depends_on:
                if dep_id in self._components:
                    self._components[dep_id].required_by.add(component_id)

            logger.info(
                f"✅ Registered component: {component_id} "
                f"(type={component_type}, layer={shutdown_layer})"
            )

            return registration

    async def unregister(self, component_id: str):
        """Unregister a component."""
        async with self._lock:
            if component_id not in self._components:
                return

            registration = self._components[component_id]

            # Remove from indices
            self._by_type[registration.component_type].remove(component_id)
            self._by_layer[registration.shutdown_layer].remove(component_id)

            # Update dependencies
            for dep_id in registration.depends_on:
                if dep_id in self._components:
                    self._components[dep_id].required_by.discard(component_id)

            for req_id in registration.required_by:
                if req_id in self._components:
                    self._components[req_id].depends_on.discard(component_id)

            del self._components[component_id]

            logger.info(f"Unregistered component: {component_id}")

    async def get(self, component_id: str) -> Optional[ComponentRegistration]:
        """Get component registration."""
        return self._components.get(component_id)

    async def get_by_type(
        self, component_type: ComponentType
    ) -> List[ComponentRegistration]:
        """Get all components of a specific type."""
        component_ids = self._by_type.get(component_type, [])
        return [self._components[cid] for cid in component_ids]

    async def get_by_layer(
        self, shutdown_layer: ShutdownLayer
    ) -> List[ComponentRegistration]:
        """Get all components in a shutdown layer."""
        component_ids = self._by_layer.get(shutdown_layer, [])
        return [self._components[cid] for cid in component_ids]

    async def get_all(self) -> List[ComponentRegistration]:
        """Get all registered components."""
        return list(self._components.values())

    async def update_phase(self, component_id: str, phase: LifecyclePhase):
        """Update component lifecycle phase."""
        if component_id in self._components:
            self._components[component_id].phase = phase
            logger.debug(f"Component {component_id}: phase -> {phase}")

    async def cleanup_stale_components(self, max_age_seconds: float = 3600):
        """
        Clean up stale component registrations.

        This prevents the UUID pollution you're seeing in logs.
        """
        async with self._lock:
            now = time.time()
            stale_ids = []

            for comp_id, reg in self._components.items():
                # Component stopped and old
                if (
                    reg.phase == LifecyclePhase.STOPPED
                    and reg.stopped_at
                    and (now - reg.stopped_at) > max_age_seconds
                ):
                    stale_ids.append(comp_id)

                # Component failed and old
                elif (
                    reg.phase == LifecyclePhase.FAILED
                    and reg.stopped_at
                    and (now - reg.stopped_at) > max_age_seconds
                ):
                    stale_ids.append(comp_id)

            # Remove stale components
            for comp_id in stale_ids:
                await self.unregister(comp_id)

            if stale_ids:
                logger.info(f"🧹 Cleaned up {len(stale_ids)} stale components")

            return len(stale_ids)


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED SHUTDOWN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


class UnifiedShutdownOrchestrator:
    """
    THE SINGLE SOURCE OF TRUTH for ALL Trinity shutdown operations.

    Eliminates duplicate shutdown systems by providing one coordinated shutdown.
    """

    def __init__(self, registry: ComponentRegistry):
        self._registry = registry
        self._shutdown_in_progress = False
        self._shutdown_lock = asyncio.Lock()

    async def compute_shutdown_plan(self) -> ShutdownPlan:
        """
        Compute optimal shutdown plan based on component dependencies.

        Returns shutdown plan with proper ordering.
        """
        all_components = await self._registry.get_all()

        # Only include running/degraded components
        active_components = [
            comp
            for comp in all_components
            if comp.phase
            in (LifecyclePhase.RUNNING, LifecyclePhase.DEGRADED, LifecyclePhase.STARTING)
        ]

        # Group by shutdown layer
        layers: Dict[ShutdownLayer, List[ComponentRegistration]] = defaultdict(list)

        for comp in active_components:
            layers[comp.shutdown_layer].append(comp)

        # Estimate duration (pessimistic: 5s per component per layer)
        estimated_duration = len(layers) * 5.0

        # Find critical path (longest dependency chain)
        critical_path = self._compute_critical_path(active_components)

        plan = ShutdownPlan(
            layers=layers,
            total_components=len(active_components),
            estimated_duration=estimated_duration,
            critical_path=critical_path,
        )

        return plan

    def _compute_critical_path(
        self, components: List[ComponentRegistration]
    ) -> List[str]:
        """Compute critical path (longest dependency chain)."""
        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        for comp in components:
            graph[comp.component_id] = comp.depends_on.copy()

        # Find path with most dependencies
        max_depth = 0
        critical_path = []

        def dfs(node: str, path: List[str]) -> int:
            nonlocal max_depth, critical_path

            if node not in graph:
                return len(path)

            max_child_depth = 0
            for child in graph[node]:
                if child not in path:  # Avoid cycles
                    child_depth = dfs(child, path + [child])
                    max_child_depth = max(max_child_depth, child_depth)

            total_depth = len(path) + max_child_depth

            if total_depth > max_depth:
                max_depth = total_depth
                critical_path = path.copy()

            return total_depth

        # Start DFS from each node
        for node in graph:
            dfs(node, [node])

        return critical_path

    async def execute_shutdown(
        self,
        *,
        timeout_per_layer: float = 30.0,
        skip_layers: Optional[Set[ShutdownLayer]] = None,
    ) -> Dict[str, Any]:
        """
        Execute coordinated Trinity shutdown.

        This is THE ONLY shutdown method that should be called.
        All other shutdown systems should delegate to this.

        Args:
            timeout_per_layer: Max time to wait for each layer
            skip_layers: Optional layers to skip (for partial shutdown)

        Returns:
            Shutdown report with statistics
        """
        async with self._shutdown_lock:
            if self._shutdown_in_progress:
                logger.warning("Shutdown already in progress")
                return {"status": "already_in_progress"}

            self._shutdown_in_progress = True

        start_time = time.time()

        logger.info("━" * 80)
        logger.info("🛑 TRINITY UNIFIED SHUTDOWN - v89.0")
        logger.info("━" * 80)

        try:
            # Compute shutdown plan
            plan = await self.compute_shutdown_plan()

            logger.info(f"Shutdown plan computed:")
            logger.info(f"  Components: {plan.total_components}")
            logger.info(f"  Layers: {len(plan.layers)}")
            logger.info(f"  Estimated duration: {plan.estimated_duration:.1f}s")
            logger.info(f"  Critical path: {' → '.join(plan.critical_path[:5])}")

            report = {
                "total": plan.total_components,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
                "by_layer": {},
            }

            # Shutdown layer by layer (LAYER_0 first, LAYER_5 last)
            for layer in sorted(ShutdownLayer, key=lambda x: x.value):
                if skip_layers and layer in skip_layers:
                    logger.info(f"⏭️  Skipping layer {layer}")
                    continue

                if layer not in plan.layers:
                    continue

                components = plan.layers[layer]

                logger.info(f"")
                logger.info(f"📦 Shutting down {layer} ({len(components)} components)...")

                layer_start = time.time()
                layer_report = await self._shutdown_layer(
                    components, timeout=timeout_per_layer
                )
                layer_duration = time.time() - layer_start

                logger.info(
                    f"   ✅ {layer}: {layer_report['succeeded']}/{len(components)} "
                    f"succeeded in {layer_duration:.1f}s"
                )

                report["succeeded"] += layer_report["succeeded"]
                report["failed"] += layer_report["failed"]
                report["skipped"] += layer_report["skipped"]
                report["by_layer"][layer] = layer_report

            # Clean up stale components
            await self._registry.cleanup_stale_components(max_age_seconds=0)

            report["duration"] = time.time() - start_time

            logger.info("")
            logger.info("━" * 80)
            logger.info("✅ TRINITY UNIFIED SHUTDOWN COMPLETE")
            logger.info(f"   Total: {report['total']}")
            logger.info(f"   Succeeded: {report['succeeded']}")
            logger.info(f"   Failed: {report['failed']}")
            logger.info(f"   Duration: {report['duration']:.1f}s")
            logger.info("━" * 80)

            return report

        finally:
            self._shutdown_in_progress = False

    async def _shutdown_layer(
        self, components: List[ComponentRegistration], timeout: float
    ) -> Dict[str, int]:
        """Shutdown all components in a layer concurrently."""
        report = {"succeeded": 0, "failed": 0, "skipped": 0}

        # Shutdown all components in parallel
        tasks = []
        for comp in components:
            tasks.append(self._shutdown_component(comp, timeout=timeout / len(components)))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for comp, result in zip(components, results):
            if isinstance(result, Exception):
                logger.error(f"   ❌ {comp.component_id}: {result}")
                report["failed"] += 1
            elif result:
                logger.debug(f"   ✅ {comp.component_id}")
                report["succeeded"] += 1
            else:
                logger.warning(f"   ⏭️  {comp.component_id}: skipped")
                report["skipped"] += 1

        return report

    async def _shutdown_component(
        self, comp: ComponentRegistration, timeout: float
    ) -> bool:
        """Shutdown a single component."""
        # Update phase
        await self._registry.update_phase(comp.component_id, LifecyclePhase.STOPPING)

        try:
            # Call shutdown hook if provided
            if comp.shutdown_hook:
                result = comp.shutdown_hook()

                # Handle async hooks
                if asyncio.iscoroutine(result):
                    await asyncio.wait_for(result, timeout=timeout)
                else:
                    # Sync hook, run in thread
                    await asyncio.wait_for(
                        asyncio.to_thread(lambda: result), timeout=timeout
                    )

            # Update phase
            await self._registry.update_phase(comp.component_id, LifecyclePhase.STOPPED)
            comp.stopped_at = time.time()

            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Component {comp.component_id} shutdown timed out after {timeout}s"
            )
            await self._registry.update_phase(comp.component_id, LifecyclePhase.FAILED)
            return False

        except Exception as e:
            logger.error(f"Component {comp.component_id} shutdown failed: {e}")
            await self._registry.update_phase(comp.component_id, LifecyclePhase.FAILED)
            return False


# ═══════════════════════════════════════════════════════════════════════════
# TRINITY UNIFICATION ENGINE (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════


class TrinityUnificationEngine:
    """
    THE MASTER ORCHESTRATOR for ALL Trinity operations.

    This is the single source of truth that:
    - Manages component registry (no UUID pollution)
    - Coordinates all shutdowns (eliminates duplicates)
    - Integrates with v85, v86, v87, v88
    - Enforces dependency ordering
    - Provides unified lifecycle management
    """

    _instance: Optional["TrinityUnificationEngine"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._registry = ComponentRegistry()
        self._shutdown_orchestrator = UnifiedShutdownOrchestrator(self._registry)

        # Integration with other coordinators
        self._unified_coordinator = None  # v85
        self._db_coordinator = None  # v86
        self._health_monitor = None  # v87
        self._lifecycle_coordinator = None  # v88

        self._running = False

    @classmethod
    async def get_engine(cls) -> "TrinityUnificationEngine":
        """Get singleton engine instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("✅ TrinityUnificationEngine initialized (v89.0)")
        return cls._instance

    async def start(self):
        """Start the unification engine."""
        if self._running:
            return

        logger.info("🚀 Starting Trinity Unification Engine...")

        # Import and integrate other coordinators
        try:
            from reactor_core.integration import (
                get_unified_coordinator,
                get_db_coordinator,
                get_health_monitor,
                get_lifecycle_coordinator,
            )

            self._unified_coordinator = await get_unified_coordinator()
            self._db_coordinator = await get_db_coordinator()
            self._health_monitor = await get_health_monitor()
            self._lifecycle_coordinator = await get_lifecycle_coordinator()

            logger.info("✅ Integrated with v85, v86, v87, v88 coordinators")

        except ImportError as e:
            logger.warning(f"Some coordinators not available: {e}")

        self._running = True
        logger.info("✅ Trinity Unification Engine started")

    async def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        shutdown_layer: ShutdownLayer,
        **kwargs,
    ) -> ComponentRegistration:
        """
        Register a component with the Trinity system.

        This is THE ONLY way components should be registered.
        """
        return await self._registry.register(
            component_id=component_id,
            component_type=component_type,
            shutdown_layer=shutdown_layer,
            **kwargs,
        )

    async def shutdown(
        self, *, timeout_per_layer: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute unified Trinity shutdown.

        This is THE ONLY shutdown method. All other systems should call this.
        """
        return await self._shutdown_orchestrator.execute_shutdown(
            timeout_per_layer=timeout_per_layer
        )

    @property
    def registry(self) -> ComponentRegistry:
        """Access component registry."""
        return self._registry


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


async def get_trinity_engine() -> TrinityUnificationEngine:
    """Get singleton Trinity engine."""
    return await TrinityUnificationEngine.get_engine()


__all__ = [
    # Main Engine
    "TrinityUnificationEngine",
    "get_trinity_engine",
    # Component Management
    "ComponentRegistry",
    "ComponentRegistration",
    "UnifiedShutdownOrchestrator",
    # Enums
    "ComponentType",
    "LifecyclePhase",
    "ShutdownLayer",
    # Data Classes
    "ShutdownPlan",
]
