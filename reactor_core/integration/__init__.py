"""
Integration module for Night Shift Training Engine.

Provides:
- JARVIS-AI-Agent log ingestion
- JARVIS Prime WebSocket/REST integration
- Cross-repo event bridge for real-time sync
- Experience streaming and transformation
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
]
