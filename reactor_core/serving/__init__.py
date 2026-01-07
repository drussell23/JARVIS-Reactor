"""
Model Serving Module - Reactor Core (Nervous System)
====================================================

Production-grade model serving infrastructure for AGI OS.

Components:
- ModelServer: Hot-reload capable model server
- InferenceEngine: Main entry point for model inference
- ModelRouter: Dynamic routing to appropriate models
- ResponseCache: Intelligent caching with LRU eviction
- ModelEnsemble: Multi-model aggregation
- Multiple backends: Transformers, vLLM, llama.cpp, MLX
"""

# Model Server with Hot-Reload (v77.0)
from reactor_core.serving.model_server import (
    ModelServer,
    ModelPool,
    ModelFileWatcher,
    SemanticCache,
    PriorityRequestQueue,
    CircuitBreaker,
    ModelBackend,
    LlamaCppBackend,
    TransformersBackend,
    MLXBackend,
    ModelServerConfig,
    ModelState,
    ModelInfo as ModelServerInfo,
    InferenceRequest as ServerInferenceRequest,
    InferenceResponse as ServerInferenceResponse,
    RequestPriority as ServerRequestPriority,
    get_model_server,
    model_server_context,
)

from reactor_core.serving.inference_engine import (
    # === AUTO-FALLBACK AND PORT DETECTION (v76.0) ===
    BackendFallbackChain,
    PortManager,
    AutoRetryInference,
    InferenceHealthMonitor,
    get_backend_fallback,
    get_port_manager,
    # Enums
    ModelBackend,
    QuantizationType,
    TaskType,
    RequestPriority,
    EnsembleStrategy,
    # Data structures
    GenerationConfig,
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    # Caching
    ResponseCache,
    # Backends
    ModelBackendInterface,
    TransformersBackend,
    VLLMBackend,
    LlamaCppBackend,
    # Routing
    RoutingRule,
    ModelRouter,
    # Engine
    InferenceEngineConfig,
    InferenceEngine,
    # Ensemble
    EnsembleConfig,
    ModelEnsemble,
)

__all__ = [
    # === MODEL SERVER WITH HOT-RELOAD (v77.0) ===
    "ModelServer",
    "ModelPool",
    "ModelFileWatcher",
    "SemanticCache",
    "PriorityRequestQueue",
    "CircuitBreaker",
    "ModelBackend",
    "LlamaCppBackend",
    "TransformersBackend",
    "MLXBackend",
    "ModelServerConfig",
    "ModelState",
    "ModelServerInfo",
    "ServerInferenceRequest",
    "ServerInferenceResponse",
    "ServerRequestPriority",
    "get_model_server",
    "model_server_context",
    # === AUTO-FALLBACK AND PORT DETECTION (v76.0) ===
    "BackendFallbackChain",
    "PortManager",
    "AutoRetryInference",
    "InferenceHealthMonitor",
    "get_backend_fallback",
    "get_port_manager",
    # Enums
    "ModelBackend",
    "QuantizationType",
    "TaskType",
    "RequestPriority",
    "EnsembleStrategy",
    # Data structures
    "GenerationConfig",
    "InferenceRequest",
    "InferenceResponse",
    "ModelInfo",
    # Caching
    "ResponseCache",
    # Backends
    "ModelBackendInterface",
    "TransformersBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    # Routing
    "RoutingRule",
    "ModelRouter",
    # Engine
    "InferenceEngineConfig",
    "InferenceEngine",
    # Ensemble
    "EnsembleConfig",
    "ModelEnsemble",
]
