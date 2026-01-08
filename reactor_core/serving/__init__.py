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

# === UNIFIED MODEL MANAGEMENT (v83.0) ===
from reactor_core.serving.unified_model_manager import (
    ModelBackend as UnifiedModelBackend,
    BackendDetector,
    ModelMetadata as UnifiedModelMetadata,
    ModelInstance,
    ModelPool as UnifiedModelPool,
    UnifiedModelManager,
    create_unified_manager,
)

from reactor_core.serving.hybrid_model_router import (
    TaskComplexity,
    ComplexityScore,
    ComplexityAnalyzer,
    RoutingStrategy as HybridRoutingStrategy,
    RoutingDecision,
    HybridModelRouter,
    create_hybrid_router,
)

from reactor_core.serving.parallel_inference_engine import (
    RequestPriority as ParallelRequestPriority,
    BatchStrategy,
    CircuitState,
    InferenceTask,
    BatchConfig,
    ResourcePool,
    CircuitBreakerConfig,
    ParallelEngineConfig,
    CircuitBreaker as ParallelCircuitBreaker,
    PerformanceMetrics,
    ParallelInferenceEngine,
    create_parallel_engine,
)

from reactor_core.serving.trinity_model_registry import (
    RepositoryType,
    ModelSource,
    ModelStatus,
    SyncStrategy,
    ModelMetadata as RegistryModelMetadata,
    RegistryConfig,
    SyncEvent,
    TrinityModelRegistry,
    create_trinity_registry,
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
    # === UNIFIED MODEL MANAGEMENT (v83.0) ===
    # Unified Model Manager
    "UnifiedModelBackend",
    "BackendDetector",
    "UnifiedModelMetadata",
    "ModelInstance",
    "UnifiedModelPool",
    "UnifiedModelManager",
    "create_unified_manager",
    # Hybrid Model Router
    "TaskComplexity",
    "ComplexityScore",
    "ComplexityAnalyzer",
    "HybridRoutingStrategy",
    "RoutingDecision",
    "HybridModelRouter",
    "create_hybrid_router",
    # Parallel Inference Engine
    "ParallelRequestPriority",
    "BatchStrategy",
    "CircuitState",
    "InferenceTask",
    "BatchConfig",
    "ResourcePool",
    "CircuitBreakerConfig",
    "ParallelEngineConfig",
    "ParallelCircuitBreaker",
    "PerformanceMetrics",
    "ParallelInferenceEngine",
    "create_parallel_engine",
    # Trinity Model Registry
    "RepositoryType",
    "ModelSource",
    "ModelStatus",
    "SyncStrategy",
    "RegistryModelMetadata",
    "RegistryConfig",
    "SyncEvent",
    "TrinityModelRegistry",
    "create_trinity_registry",
]
