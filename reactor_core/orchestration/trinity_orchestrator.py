"""
PROJECT TRINITY Phase 3: Central Orchestrator

The Trinity Orchestrator is the central coordination hub for the unified
JARVIS cognitive architecture. It manages:
- Component discovery and health monitoring
- State reconciliation across all three repos
- Intelligent command routing with load balancing
- Parallel command execution
- Fault tolerance with circuit breakers

ARCHITECTURE:
                    ┌──────────────────────────┐
                    │   TRINITY ORCHESTRATOR   │
                    │   (Central Coordinator)  │
                    └──────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    ┌────────────┐    ┌──────────────┐    ┌────────────┐
    │  J-PRIME   │    │ REACTOR CORE │    │   JARVIS   │
    │   (Mind)   │    │   (Nerves)   │    │   (Body)   │
    └────────────┘    └──────────────┘    └────────────┘

FEATURES:
- Auto-discovery of Trinity components via heartbeats
- State snapshot aggregation from all components
- Priority-based command queuing
- Circuit breaker pattern for fault tolerance
- Parallel command dispatch to multiple targets
- Event-driven architecture with async processing

USAGE:
    from reactor_core.orchestration.trinity_orchestrator import (
        TrinityOrchestrator,
        get_orchestrator,
        initialize_orchestrator,
    )

    orchestrator = await initialize_orchestrator()
    await orchestrator.dispatch_command(
        intent="start_surveillance",
        payload={"app_name": "Chrome", "trigger_text": "bouncing ball"},
    )
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

TRINITY_DIR = Path.home() / ".jarvis" / "trinity"
ORCHESTRATOR_STATE_FILE = TRINITY_DIR / "orchestrator_state.json"
COMPONENTS_DIR = TRINITY_DIR / "components"

# Health thresholds
HEARTBEAT_TIMEOUT = 15.0  # seconds
HEALTH_CHECK_INTERVAL = 5.0  # seconds
CIRCUIT_BREAKER_THRESHOLD = 3  # failures before opening
CIRCUIT_BREAKER_RESET = 30.0  # seconds to reset


# =============================================================================
# ENUMS
# =============================================================================

class ComponentType(Enum):
    """Types of Trinity components."""
    J_PRIME = "j_prime"
    REACTOR_CORE = "reactor_core"
    JARVIS_BODY = "jarvis_body"


class ComponentHealth(Enum):
    """Health states for components."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold
    HALF_OPEN = "half_open"  # Testing recovery


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComponentState:
    """State of a Trinity component."""
    component_type: ComponentType
    instance_id: str = ""
    health: ComponentHealth = ComponentHealth.UNKNOWN
    last_heartbeat: float = 0.0
    last_command_id: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Circuit breaker
    failure_count: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: float = 0.0

    def is_available(self) -> bool:
        """Check if component is available for commands."""
        if self.circuit_state == CircuitState.OPEN:
            # Check if circuit breaker should reset
            if (time.time() - self.circuit_opened_at) > CIRCUIT_BREAKER_RESET:
                self.circuit_state = CircuitState.HALF_OPEN
            else:
                return False

        return self.health in (ComponentHealth.HEALTHY, ComponentHealth.DEGRADED)

    def record_success(self) -> None:
        """Record successful command execution."""
        self.failure_count = 0
        if self.circuit_state == CircuitState.HALF_OPEN:
            self.circuit_state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record command failure."""
        self.failure_count += 1
        if self.failure_count >= CIRCUIT_BREAKER_THRESHOLD:
            self.circuit_state = CircuitState.OPEN
            self.circuit_opened_at = time.time()
            logger.warning(f"[Trinity] Circuit breaker OPEN for {self.component_type.value}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type.value,
            "instance_id": self.instance_id,
            "health": self.health.value,
            "last_heartbeat": self.last_heartbeat,
            "last_command_id": self.last_command_id,
            "capabilities": list(self.capabilities),
            "metrics": self.metrics,
            "circuit_state": self.circuit_state.value,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentState":
        return cls(
            component_type=ComponentType(data["component_type"]),
            instance_id=data.get("instance_id", ""),
            health=ComponentHealth(data.get("health", "unknown")),
            last_heartbeat=data.get("last_heartbeat", 0.0),
            last_command_id=data.get("last_command_id"),
            capabilities=set(data.get("capabilities", [])),
            metrics=data.get("metrics", {}),
            circuit_state=CircuitState(data.get("circuit_state", "closed")),
            failure_count=data.get("failure_count", 0),
        )


@dataclass
class AggregatedState:
    """Aggregated state from all Trinity components."""
    timestamp: float = field(default_factory=time.time)

    # JARVIS Body state
    surveillance_active: bool = False
    surveillance_targets: List[str] = field(default_factory=list)
    apps_on_ghost_display: List[str] = field(default_factory=list)
    frozen_apps: List[str] = field(default_factory=list)
    ghost_display_available: bool = False

    # J-Prime state
    model_loaded: bool = False
    model_name: str = ""
    active_plan_id: Optional[str] = None

    # Reactor Core state
    training_active: bool = False
    scout_topics: List[str] = field(default_factory=list)

    # System metrics
    total_commands_processed: int = 0
    commands_in_flight: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "surveillance_active": self.surveillance_active,
            "surveillance_targets": self.surveillance_targets,
            "apps_on_ghost_display": self.apps_on_ghost_display,
            "frozen_apps": self.frozen_apps,
            "ghost_display_available": self.ghost_display_available,
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "active_plan_id": self.active_plan_id,
            "training_active": self.training_active,
            "scout_topics": self.scout_topics,
            "total_commands_processed": self.total_commands_processed,
            "commands_in_flight": self.commands_in_flight,
            "last_error": self.last_error,
        }


@dataclass
class PendingCommand:
    """A command waiting to be executed."""
    id: str
    intent: str
    payload: Dict[str, Any]
    target: Optional[ComponentType]
    priority: int
    created_at: float
    timeout: float
    requires_ack: bool
    callback: Optional[Callable] = None

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.timeout


# =============================================================================
# TRINITY ORCHESTRATOR
# =============================================================================

class TrinityOrchestrator:
    """
    Central orchestrator for the Trinity architecture.

    Coordinates all three components (J-Prime, Reactor Core, JARVIS Body)
    with intelligent routing, state reconciliation, and fault tolerance.
    """

    _instance: Optional['TrinityOrchestrator'] = None

    def __new__(cls) -> 'TrinityOrchestrator':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Component states
        self._components: Dict[ComponentType, ComponentState] = {
            ComponentType.J_PRIME: ComponentState(ComponentType.J_PRIME),
            ComponentType.REACTOR_CORE: ComponentState(ComponentType.REACTOR_CORE),
            ComponentType.JARVIS_BODY: ComponentState(ComponentType.JARVIS_BODY),
        }

        # Command queues (priority-based)
        self._command_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._pending_commands: Dict[str, PendingCommand] = {}
        self._processed_commands: deque = deque(maxlen=1000)

        # State
        self._aggregated_state = AggregatedState()
        self._running = False
        self._start_time = time.time()

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._command_processor_task: Optional[asyncio.Task] = None
        self._state_reconciler_task: Optional[asyncio.Task] = None
        self._self_heartbeat_task: Optional[asyncio.Task] = None  # v72.0

        # Event handlers
        self._state_change_handlers: List[Callable] = []

        # Statistics
        self._stats = {
            "commands_dispatched": 0,
            "commands_succeeded": 0,
            "commands_failed": 0,
            "commands_timeout": 0,
            "state_reconciliations": 0,
            "circuit_breaker_trips": 0,
        }

        logger.info("[Trinity] Orchestrator initialized")

    async def start(self) -> bool:
        """Start the orchestrator."""
        if self._running:
            return True

        try:
            # Ensure directories exist
            TRINITY_DIR.mkdir(parents=True, exist_ok=True)
            COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

            # Load persisted state
            await self._load_state()

            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._command_processor_task = asyncio.create_task(self._command_processor_loop())
            self._state_reconciler_task = asyncio.create_task(self._state_reconciliation_loop())
            # v72.0: Start self-heartbeat task
            self._self_heartbeat_task = asyncio.create_task(self._self_heartbeat_loop())

            self._running = True
            logger.info("[Trinity] Orchestrator started")
            return True

        except Exception as e:
            logger.error(f"[Trinity] Orchestrator start failed: {e}")
            return False

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False

        # Cancel background tasks
        for task in [
            self._health_check_task,
            self._command_processor_task,
            self._state_reconciler_task,
            self._self_heartbeat_task,  # v72.0
        ]:
            if task:
                task.cancel()

        # Save state
        await self._save_state()

        logger.info("[Trinity] Orchestrator stopped")

    def is_running(self) -> bool:
        return self._running

    # =========================================================================
    # COMPONENT MANAGEMENT
    # =========================================================================

    def get_component_state(self, component: ComponentType) -> ComponentState:
        """Get state of a specific component."""
        return self._components[component]

    def get_all_component_states(self) -> Dict[ComponentType, ComponentState]:
        """Get states of all components."""
        return self._components.copy()

    def update_component_heartbeat(
        self,
        component: ComponentType,
        instance_id: str,
        metrics: Dict[str, Any],
        capabilities: Optional[Set[str]] = None,
    ) -> None:
        """Update component heartbeat and metrics."""
        state = self._components[component]
        state.instance_id = instance_id
        state.last_heartbeat = time.time()
        state.metrics = metrics

        if capabilities:
            state.capabilities = capabilities

        # Update health based on heartbeat
        state.health = ComponentHealth.HEALTHY

        logger.debug(f"[Trinity] Heartbeat from {component.value}: {instance_id}")

    def get_aggregated_state(self) -> AggregatedState:
        """Get the current aggregated state."""
        return self._aggregated_state

    # =========================================================================
    # COMMAND DISPATCH
    # =========================================================================

    async def dispatch_command(
        self,
        intent: str,
        payload: Dict[str, Any],
        target: Optional[ComponentType] = None,
        priority: int = 5,
        timeout: float = 30.0,
        requires_ack: bool = True,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a command to the appropriate component(s).

        Args:
            intent: Command intent (e.g., "start_surveillance")
            payload: Command payload
            target: Specific target component (None for auto-routing)
            priority: Priority (1-10, lower is higher)
            timeout: Command timeout in seconds
            requires_ack: Whether to wait for acknowledgment
            callback: Optional callback for async results

        Returns:
            Result dict with success status
        """
        command_id = str(uuid.uuid4())

        # Create pending command
        pending = PendingCommand(
            id=command_id,
            intent=intent,
            payload=payload,
            target=target,
            priority=priority,
            created_at=time.time(),
            timeout=timeout,
            requires_ack=requires_ack,
            callback=callback,
        )

        # Auto-route if no target specified
        if target is None:
            target = self._route_command(intent)
            pending.target = target

        # Check target availability
        if target and not self._components[target].is_available():
            return {
                "success": False,
                "error": f"Component {target.value} is not available",
                "command_id": command_id,
            }

        # Add to queue
        await self._command_queue.put((priority, time.time(), pending))
        self._pending_commands[command_id] = pending

        self._stats["commands_dispatched"] += 1
        logger.info(f"[Trinity] Dispatched command: {intent} -> {target.value if target else 'auto'}")

        # If requires ACK, wait for result
        if requires_ack:
            result = await self._wait_for_command_result(command_id, timeout)
            return result

        return {"success": True, "command_id": command_id}

    async def dispatch_parallel(
        self,
        commands: List[Dict[str, Any]],
        timeout: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """
        Dispatch multiple commands in parallel.

        Args:
            commands: List of command dicts with intent, payload, etc.
            timeout: Overall timeout for all commands

        Returns:
            List of results for each command
        """
        tasks = []

        for cmd in commands:
            task = self.dispatch_command(
                intent=cmd.get("intent", ""),
                payload=cmd.get("payload", {}),
                target=ComponentType(cmd["target"]) if cmd.get("target") else None,
                priority=cmd.get("priority", 5),
                timeout=cmd.get("timeout", timeout),
                requires_ack=cmd.get("requires_ack", True),
            )
            tasks.append(task)

        # Wait for all with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
            return [
                r if isinstance(r, dict) else {"success": False, "error": str(r)}
                for r in results
            ]
        except asyncio.TimeoutError:
            return [{"success": False, "error": "Parallel dispatch timeout"}] * len(commands)

    def _route_command(self, intent: str) -> ComponentType:
        """Auto-route command to appropriate component."""
        # Surveillance, window, cryostasis, phantom -> JARVIS Body
        jarvis_intents = {
            "start_surveillance", "stop_surveillance",
            "bring_back_window", "exile_window", "teleport_window",
            "freeze_app", "thaw_app",
            "create_ghost_display", "destroy_ghost_display",
            "ping",
        }

        # Cognitive, reasoning, planning -> J-Prime
        jprime_intents = {
            "reason", "plan", "analyze", "decide",
            "generate_response", "summarize",
        }

        # Training, learning -> Reactor Core
        reactor_intents = {
            "train", "distill", "scout",
            "record_experience", "store_memory",
        }

        if intent in jarvis_intents:
            return ComponentType.JARVIS_BODY
        elif intent in jprime_intents:
            return ComponentType.J_PRIME
        elif intent in reactor_intents:
            return ComponentType.REACTOR_CORE
        else:
            # Default to JARVIS Body for execution
            return ComponentType.JARVIS_BODY

    async def _wait_for_command_result(
        self,
        command_id: str,
        timeout: float,
    ) -> Dict[str, Any]:
        """Wait for a command result with timeout."""
        start = time.time()

        while (time.time() - start) < timeout:
            # Check if command is still pending
            if command_id not in self._pending_commands:
                # Command was processed
                return {"success": True, "command_id": command_id}

            # Check if expired
            pending = self._pending_commands.get(command_id)
            if pending and pending.is_expired():
                del self._pending_commands[command_id]
                self._stats["commands_timeout"] += 1
                return {"success": False, "error": "Command timeout", "command_id": command_id}

            await asyncio.sleep(0.1)

        # Final timeout
        if command_id in self._pending_commands:
            del self._pending_commands[command_id]
            self._stats["commands_timeout"] += 1

        return {"success": False, "error": "Timeout", "command_id": command_id}

    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Monitor component health via heartbeats."""
        while self._running:
            try:
                now = time.time()

                for component_type, state in self._components.items():
                    age = now - state.last_heartbeat

                    if state.last_heartbeat == 0:
                        state.health = ComponentHealth.UNKNOWN
                    elif age < HEARTBEAT_TIMEOUT:
                        if state.health != ComponentHealth.HEALTHY:
                            state.health = ComponentHealth.HEALTHY
                            logger.info(f"[Trinity] {component_type.value} is now HEALTHY")
                    elif age < HEARTBEAT_TIMEOUT * 2:
                        if state.health != ComponentHealth.DEGRADED:
                            state.health = ComponentHealth.DEGRADED
                            logger.warning(f"[Trinity] {component_type.value} is DEGRADED")
                    else:
                        if state.health != ComponentHealth.OFFLINE:
                            state.health = ComponentHealth.OFFLINE
                            logger.error(f"[Trinity] {component_type.value} is OFFLINE")

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Trinity] Health check error: {e}")
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _self_heartbeat_loop(self) -> None:
        """
        v72.0: Broadcast Reactor-Core's own heartbeat to Trinity.

        This writes the orchestrator's state to ~/.jarvis/trinity/components/reactor_core.json
        so that JARVIS Body can detect when Reactor-Core is online.
        """
        import os
        import uuid

        instance_id = f"reactor-core-{os.getpid()}-{int(self._start_time)}"

        while self._running:
            try:
                # Build state
                state = {
                    "component_type": "reactor_core",
                    "instance_id": instance_id,
                    "timestamp": time.time(),
                    "uptime_seconds": time.time() - self._start_time,
                    "metrics": {
                        "running": self._running,
                        "commands_dispatched": self._stats["commands_dispatched"],
                        "commands_succeeded": self._stats["commands_succeeded"],
                        "commands_failed": self._stats["commands_failed"],
                        "pending_commands": len(self._pending_commands),
                    },
                }

                # Write to components directory
                state_file = COMPONENTS_DIR / "reactor_core.json"
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)

                logger.debug(f"[Trinity] Reactor-Core heartbeat written (uptime: {state['uptime_seconds']:.1f}s)")

                await asyncio.sleep(5.0)  # Heartbeat every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Trinity] Self-heartbeat error: {e}")
                await asyncio.sleep(5.0)

    async def _command_processor_loop(self) -> None:
        """Process queued commands."""
        while self._running:
            try:
                # Get next command (blocks until available)
                try:
                    priority, timestamp, pending = await asyncio.wait_for(
                        self._command_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Skip expired commands
                if pending.is_expired():
                    if pending.id in self._pending_commands:
                        del self._pending_commands[pending.id]
                    self._stats["commands_timeout"] += 1
                    continue

                # Execute command
                success = await self._execute_command(pending)

                # Update stats
                if success:
                    self._stats["commands_succeeded"] += 1
                else:
                    self._stats["commands_failed"] += 1

                # Remove from pending
                if pending.id in self._pending_commands:
                    del self._pending_commands[pending.id]

                # Record in history
                self._processed_commands.append({
                    "id": pending.id,
                    "intent": pending.intent,
                    "target": pending.target.value if pending.target else None,
                    "success": success,
                    "timestamp": time.time(),
                })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Trinity] Command processor error: {e}")

    async def _execute_command(self, pending: PendingCommand) -> bool:
        """Execute a single command."""
        target = pending.target
        if not target:
            return False

        state = self._components[target]

        # Check availability
        if not state.is_available():
            logger.warning(f"[Trinity] Target {target.value} not available")
            return False

        try:
            # Write command file to Trinity directory
            command_data = {
                "id": pending.id,
                "timestamp": time.time(),
                "source": "reactor_core",
                "intent": pending.intent,
                "payload": pending.payload,
                "target": target.value,
                "priority": pending.priority,
                "requires_ack": pending.requires_ack,
                "ttl_seconds": pending.timeout,
            }

            # Write to component-specific directory
            commands_dir = TRINITY_DIR / "commands"
            commands_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{int(time.time() * 1000)}_{pending.id}.json"
            filepath = commands_dir / filename

            with open(filepath, "w") as f:
                json.dump(command_data, f, indent=2)

            logger.debug(f"[Trinity] Wrote command {pending.id[:8]} to {filepath.name}")

            state.record_success()
            state.last_command_id = pending.id
            return True

        except Exception as e:
            logger.error(f"[Trinity] Command execution error: {e}")
            state.record_failure()
            return False

    async def _state_reconciliation_loop(self) -> None:
        """Periodically reconcile state across components."""
        while self._running:
            try:
                await self._reconcile_state()
                self._stats["state_reconciliations"] += 1
                await asyncio.sleep(5.0)  # Reconcile every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Trinity] State reconciliation error: {e}")
                await asyncio.sleep(5.0)

    async def _reconcile_state(self) -> None:
        """Reconcile state from all component heartbeats."""
        # Read JARVIS heartbeats
        jarvis_state = self._components[ComponentType.JARVIS_BODY]
        if jarvis_state.metrics:
            self._aggregated_state.surveillance_active = jarvis_state.metrics.get(
                "surveillance_active", False
            )
            self._aggregated_state.surveillance_targets = jarvis_state.metrics.get(
                "surveillance_targets", []
            )
            self._aggregated_state.apps_on_ghost_display = jarvis_state.metrics.get(
                "apps_on_ghost_display", []
            )
            self._aggregated_state.frozen_apps = jarvis_state.metrics.get(
                "frozen_apps", []
            )
            self._aggregated_state.ghost_display_available = jarvis_state.metrics.get(
                "ghost_display_available", False
            )

        # Read J-Prime heartbeats
        jprime_state = self._components[ComponentType.J_PRIME]
        if jprime_state.metrics:
            self._aggregated_state.model_loaded = jprime_state.metrics.get(
                "model_loaded", False
            )
            self._aggregated_state.model_name = jprime_state.metrics.get(
                "model_path", ""
            )

        # Update timestamp
        self._aggregated_state.timestamp = time.time()
        self._aggregated_state.commands_in_flight = len(self._pending_commands)
        self._aggregated_state.total_commands_processed = (
            self._stats["commands_succeeded"] + self._stats["commands_failed"]
        )

        # Notify handlers
        for handler in self._state_change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self._aggregated_state)
                else:
                    handler(self._aggregated_state)
            except Exception as e:
                logger.warning(f"[Trinity] State handler error: {e}")

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_state_change(self, handler: Callable) -> None:
        """Register a state change handler."""
        self._state_change_handlers.append(handler)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def _load_state(self) -> None:
        """Load persisted orchestrator state."""
        try:
            if ORCHESTRATOR_STATE_FILE.exists():
                with open(ORCHESTRATOR_STATE_FILE) as f:
                    data = json.load(f)

                # Restore component states
                for comp_data in data.get("components", []):
                    comp_type = ComponentType(comp_data["component_type"])
                    self._components[comp_type] = ComponentState.from_dict(comp_data)

                logger.info("[Trinity] Restored orchestrator state")

        except Exception as e:
            logger.warning(f"[Trinity] Could not load state: {e}")

    async def _save_state(self) -> None:
        """Save orchestrator state."""
        try:
            data = {
                "timestamp": time.time(),
                "components": [s.to_dict() for s in self._components.values()],
                "stats": self._stats,
            }

            with open(ORCHESTRATOR_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("[Trinity] Saved orchestrator state")

        except Exception as e:
            logger.warning(f"[Trinity] Could not save state: {e}")

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self._stats,
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time,
            "pending_commands": len(self._pending_commands),
            "components": {
                c.value: {
                    "health": self._components[c].health.value,
                    "available": self._components[c].is_available(),
                }
                for c in ComponentType
            },
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_orchestrator: Optional[TrinityOrchestrator] = None


def get_orchestrator() -> TrinityOrchestrator:
    """Get the singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TrinityOrchestrator()
    return _orchestrator


async def initialize_orchestrator() -> TrinityOrchestrator:
    """Initialize and start the orchestrator."""
    orchestrator = get_orchestrator()
    await orchestrator.start()
    return orchestrator


async def shutdown_orchestrator() -> None:
    """Shutdown the orchestrator."""
    global _orchestrator
    if _orchestrator:
        await _orchestrator.stop()
        _orchestrator = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def dispatch_to_jarvis(
    intent: str,
    payload: Dict[str, Any],
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Dispatch a command to JARVIS Body."""
    orchestrator = get_orchestrator()
    return await orchestrator.dispatch_command(
        intent=intent,
        payload=payload,
        target=ComponentType.JARVIS_BODY,
        timeout=timeout,
    )


async def dispatch_to_jprime(
    intent: str,
    payload: Dict[str, Any],
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Dispatch a command to J-Prime."""
    orchestrator = get_orchestrator()
    return await orchestrator.dispatch_command(
        intent=intent,
        payload=payload,
        target=ComponentType.J_PRIME,
        timeout=timeout,
    )


def update_jarvis_heartbeat(
    instance_id: str,
    metrics: Dict[str, Any],
) -> None:
    """Update JARVIS heartbeat in orchestrator."""
    orchestrator = get_orchestrator()
    orchestrator.update_component_heartbeat(
        ComponentType.JARVIS_BODY,
        instance_id,
        metrics,
        capabilities={"surveillance", "window_management", "cryostasis", "phantom_hardware"},
    )


def update_jprime_heartbeat(
    instance_id: str,
    metrics: Dict[str, Any],
) -> None:
    """Update J-Prime heartbeat in orchestrator."""
    orchestrator = get_orchestrator()
    orchestrator.update_component_heartbeat(
        ComponentType.J_PRIME,
        instance_id,
        metrics,
        capabilities={"reasoning", "planning", "generation"},
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TrinityOrchestrator",
    "ComponentType",
    "ComponentHealth",
    "ComponentState",
    "AggregatedState",
    "get_orchestrator",
    "initialize_orchestrator",
    "shutdown_orchestrator",
    "dispatch_to_jarvis",
    "dispatch_to_jprime",
    "update_jarvis_heartbeat",
    "update_jprime_heartbeat",
]
