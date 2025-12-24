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
]
