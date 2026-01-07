"""
Configuration module for Night Shift Training Engine.

Provides dataclass-based configuration with:
- YAML file loading
- Environment variable interpolation
- Type validation
- Sensible defaults
- Unified cross-repo configuration
"""

from reactor_core.config.base_config import (
    BaseConfig,
    IngestionConfig,
    TrainingConfig,
    DistillationConfig,
    OrchestrationConfig,
    QuantizationConfig,
    EvalConfig,
    NightShiftConfig,
    load_config,
    get_config,
)

from reactor_core.config.unified_config import (
    UnifiedConfig,
    ServiceType,
    ServiceEndpoint,
    RepoConfig,
    Environment,
    get_config as get_unified_config,
    reset_config as reset_unified_config,
)

from reactor_core.config.trinity_config import (
    TrinityConfig,
    HealthConfig,
    CircuitBreakerConfig,
    CommandConfig,
    DeadLetterQueueConfig,
    get_config as get_trinity_config,
    reset_config as reset_trinity_config,
    sleep_with_jitter,
    async_sleep_with_jitter,
    get_retry_delay,
)

__all__ = [
    # Base configs
    "BaseConfig",
    "IngestionConfig",
    "TrainingConfig",
    "DistillationConfig",
    "OrchestrationConfig",
    "QuantizationConfig",
    "EvalConfig",
    "NightShiftConfig",
    "load_config",
    "get_config",
    # Unified cross-repo config
    "UnifiedConfig",
    "ServiceType",
    "ServiceEndpoint",
    "RepoConfig",
    "Environment",
    "get_unified_config",
    "reset_unified_config",
    # Trinity config
    "TrinityConfig",
    "HealthConfig",
    "CircuitBreakerConfig",
    "CommandConfig",
    "DeadLetterQueueConfig",
    "get_trinity_config",
    "reset_trinity_config",
    "sleep_with_jitter",
    "async_sleep_with_jitter",
    "get_retry_delay",
]
