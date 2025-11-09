"""
Reactor Core - AI/ML Training Engine
"""

__version__ = "1.0.0"

from reactor_core.training import Trainer, TrainingConfig
from reactor_core.utils.environment import detect_environment, EnvironmentType

__all__ = [
    "Trainer",
    "TrainingConfig",
    "detect_environment",
    "EnvironmentType",
    "__version__",
]
