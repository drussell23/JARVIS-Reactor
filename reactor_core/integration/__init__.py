"""
Integration module for Night Shift Training Engine.

Provides:
- JARVIS-AI-Agent log ingestion
- JARVIS Prime WebSocket/REST integration
- Cross-repo event bridge for real-time sync
- Experience streaming and transformation
- Unified cost tracking across all repos (v10.0)
- Computer Use optimization tracking (v10.1)
"""

from reactor_core.integration.jarvis_connector import (
    JARVISConnector,
    JARVISConnectorConfig,
    JARVISEvent,
    EventType,
    CorrectionType,
)

from reactor_core.integration.prime_connector import (
    PrimeConnector,
    PrimeConnectorConfig,
    PrimeEvent,
    PrimeEventType,
    ConnectionState,
)

from reactor_core.integration.event_bridge import (
    EventBridge,
    EventTransport,
    FileTransport,
    WebSocketTransport,
    CrossRepoEvent,
    EventSource,
    EventType as BridgeEventType,
    create_event_bridge,
)

from reactor_core.integration.cost_bridge import (
    CostBridge,
    CostSummary,
    get_cost_bridge,
    initialize_cost_bridge,
    shutdown_cost_bridge,
    get_aggregated_costs,
    emit_cost_event,
    record_distillation_cost,
)

from reactor_core.integration.computer_use_connector import (
    ComputerUseConnector,
    ComputerUseConnectorConfig,
    ComputerUseEvent,
    ComputerUseEventType,
    ActionType,
)

# PROJECT TRINITY: JARVIS Body Connector
from reactor_core.integration.trinity_connector import (
    TrinityConnector,
    TrinityCommand,
    TrinityIntent,
    TrinitySource,
    JARVISStatus,
    HeartbeatState,
    get_trinity_connector,
    connect_to_jarvis,
    send_surveillance_command,
    send_bring_back_command,
    get_jarvis_status,
    get_jarvis_heartbeat,
)

# PROJECT TRINITY: Ultra-High Performance Event Bus (v82.0)
from reactor_core.integration.trinity_bridge import (
    EventPriority,
    EventType as TrinityEventType,
    BridgeState,
    TrinityEvent,
    BridgeMetrics,
    TrinityBridge,
    create_trinity_bridge,
    CircuitBreaker,
    BloomFilter,
    PriorityEventQueue,
)

# PROJECT TRINITY: Unified State Coordinator (v85.0)
from reactor_core.integration.unified_coordinator import (
    # Core Coordinator
    UnifiedStateCoordinator,
    get_unified_coordinator,
    cleanup_stale_state,
    # Enums
    ComponentType,
    EntryPoint,
    CoordinationState,
    EventType as CoordinatorEventType,
    # Data Classes
    ProcessSignature,
    ComponentOwnership,
    CoordinationEvent,
    # Coordination Layers
    SharedMemoryLayer,
    UnixSocketEventBus,
    ConsensusProtocol,
    # Detector
    TrinityEntryPointDetector,
)

# PROJECT TRINITY: Async Database Coordinator (v86.0)
from reactor_core.integration.async_db_coordinator import (
    # Core Coordinator
    DistributedDatabaseCoordinator,
    get_db_coordinator,
    # Connection Management
    SafeDatabaseConnection,
    SafeDatabaseCursor,
    ConnectionPool,
    # Context Managers
    CancellationSafeContextManager,
    safe_db_operation,
    # Enums
    TransactionState,
    IsolationLevel,
    RetryStrategy,
    # Data Classes
    RetryConfig,
    ConnectionMetrics,
)

__all__ = [
    # JARVIS-AI-Agent
    "JARVISConnector",
    "JARVISConnectorConfig",
    "JARVISEvent",
    "EventType",
    "CorrectionType",
    # JARVIS Prime
    "PrimeConnector",
    "PrimeConnectorConfig",
    "PrimeEvent",
    "PrimeEventType",
    "ConnectionState",
    # Event Bridge
    "EventBridge",
    "EventTransport",
    "FileTransport",
    "WebSocketTransport",
    "CrossRepoEvent",
    "EventSource",
    "BridgeEventType",
    "create_event_bridge",
    # Cost Bridge (v10.0)
    "CostBridge",
    "CostSummary",
    "get_cost_bridge",
    "initialize_cost_bridge",
    "shutdown_cost_bridge",
    "get_aggregated_costs",
    "emit_cost_event",
    "record_distillation_cost",
    # Computer Use Bridge (v10.1)
    "ComputerUseConnector",
    "ComputerUseConnectorConfig",
    "ComputerUseEvent",
    "ComputerUseEventType",
    "ActionType",
    # PROJECT TRINITY: JARVIS Body Connector
    "TrinityConnector",
    "TrinityCommand",
    "TrinityIntent",
    "TrinitySource",
    "JARVISStatus",
    "HeartbeatState",
    "get_trinity_connector",
    "connect_to_jarvis",
    "send_surveillance_command",
    "send_bring_back_command",
    "get_jarvis_status",
    "get_jarvis_heartbeat",
    # PROJECT TRINITY: Trinity Bridge (v82.0)
    "EventPriority",
    "TrinityEventType",
    "BridgeState",
    "TrinityEvent",
    "BridgeMetrics",
    "TrinityBridge",
    "create_trinity_bridge",
    "CircuitBreaker",
    "BloomFilter",
    "PriorityEventQueue",
    # PROJECT TRINITY: Unified State Coordinator (v85.0)
    "UnifiedStateCoordinator",
    "get_unified_coordinator",
    "cleanup_stale_state",
    "ComponentType",
    "EntryPoint",
    "CoordinationState",
    "CoordinatorEventType",
    "ProcessSignature",
    "ComponentOwnership",
    "CoordinationEvent",
    "SharedMemoryLayer",
    "UnixSocketEventBus",
    "ConsensusProtocol",
    "TrinityEntryPointDetector",
    # PROJECT TRINITY: Async Database Coordinator (v86.0)
    "DistributedDatabaseCoordinator",
    "get_db_coordinator",
    "SafeDatabaseConnection",
    "SafeDatabaseCursor",
    "ConnectionPool",
    "CancellationSafeContextManager",
    "safe_db_operation",
    "TransactionState",
    "IsolationLevel",
    "RetryStrategy",
    "RetryConfig",
    "ConnectionMetrics",
]
