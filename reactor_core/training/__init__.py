"""
Training modules for Night Shift Training Engine - Reactor Core (Nervous System)
==================================================================================

Provides:
- AsyncTrainer with full HuggingFace/TRL integration
- LoRA/QLoRA configuration and application
- Training callbacks for progress tracking
- Gradient checkpointing and memory optimization

ADVANCED TRAINING (v76.0-v80.0):
- DPO (Direct Preference Optimization)
- RLHF (Reinforcement Learning from Human Feedback)
- Constitutional AI training
- FSDP (Fully Sharded Data Parallel)
- Curriculum Learning (v79.0)
- Meta-Learning: MAML, Reptile, Meta-SGD (v79.0)
- World Model Training (v80.0)
- Causal Reasoning (v80.0)
- Experience Buffer for continuous learning
"""

# Core trainer
from reactor_core.training.trainer import (
    Trainer,
    AsyncTrainer,
    TrainingConfig,
    TrainingProgress,
    TrainingResult,
    TrainingState,
)

# Advanced training methods
from reactor_core.training.advanced_training import (
    # === MEMORY OPTIMIZATION (v76.0) ===
    MemoryManager,
    ReferenceModelManager,
    DynamicBatchSizer,
    get_memory_manager,
    # Enums
    TrainingMethod,
    SafetyTier,
    CurriculumStrategy,
    ExperiencePriority,
    # Data structures
    PreferencePair,
    Experience,
    TrainingBatch,
    TrainingMetrics,
    # Experience Buffer
    ExperienceBuffer,
    # DPO
    DPOConfig,
    DPOTrainer,
    # RLHF
    RLHFConfig,
    RewardModel,
    PPOTrainer,
    RLHFPipeline,
    # Constitutional AI
    ConstitutionalPrinciple,
    ConstitutionalAITrainer,
    DEFAULT_CONSTITUTIONAL_PRINCIPLES,
    # Curriculum Learning
    CurriculumConfig,
    DifficultyScorer,
    CurriculumScheduler,
    # FSDP
    FSDPConfig,
    FSDPWrapper,
    # Unified Trainer
    AdvancedTrainingConfig,
    AdvancedTrainer,
)

# Callbacks
from reactor_core.training.callbacks import (
    TrainingEvent,
    BaseCallback,
    CallbackMixin,
    ProgressCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    GPUMemoryCallback,
    RichProgressCallback,
    WebhookCallback,
    CompositeCallback,
    create_default_callbacks,
)

# LoRA/QLoRA
from reactor_core.training.lora import (
    # Enums
    QuantizationType,
    TaskType,
    # Configs
    LoRAConfig,
    QLoRAConfig,
    # Core functions
    apply_lora,
    apply_qlora,
    prepare_model_for_qlora,
    load_model_for_qlora,
    # Merge/save/load
    merge_and_unload,
    save_adapter,
    load_adapter,
    # Adapter management
    set_adapter,
    disable_adapter,
    enable_adapter,
    get_adapter_info,
    # Utilities
    detect_model_architecture,
    get_trainable_parameters,
    format_params,
    create_optimal_lora_config,
    # Constants
    TARGET_MODULES_MAP,
)

# Curriculum Learning (v79.0)
from reactor_core.training.curriculum_learning import (
    DifficultyMetric,
    DifficultyScore,
    LossDifficultyScorer,
    LengthDifficultyScorer,
    CurriculumStage,
    CurriculumSampler,
    CurriculumLearner,
    create_default_curriculum,
)

# Meta-Learning (v79.0)
from reactor_core.training.meta_learning import (
    MetaAlgorithm,
    Task,
    TaskSampler,
    MAMLConfig,
    MAMLTrainer,
    ReptileConfig,
    ReptileTrainer,
    MetaSGDConfig,
    MetaSGDTrainer,
    create_n_way_k_shot_task,
)

# World Model Training (v80.0)
from reactor_core.training.world_model_training import (
    # Components
    LatentEncoder,
    LatentDecoder,
    TransitionModel,
    RewardModel,
    ValueModel,
    # World Model
    WorldModelConfig,
    WorldModel,
    # Training
    WorldModelTrainingConfig,
    WorldModelTrainer,
    # Reasoning
    CounterfactualReasoner,
)

# Causal Reasoning (v80.0)
from reactor_core.training.causal_reasoning import (
    # Causal Graph
    CausalEdge,
    CausalGraph,
    # Structural Causal Model
    StructuralCausalModel,
    # Causal Discovery
    CausalDiscovery,
    # Neural Causal Model
    NeuralCausalModel,
    # Causal Attention
    CausalAttention,
    # Evaluation
    CausalEvaluationMetrics,
    evaluate_causal_graph,
)

# FSDP Training (v81.0)
from reactor_core.training.fsdp_training import (
    # Enums
    FSDPShardingStrategy,
    FSDPMixedPrecisionPolicy,
    FSDPBackwardPrefetch,
    FSDPStateDictType,
    # Config
    FSDPConfig as FSDPTrainingConfig,
    # Trainer
    FSDPTrainer,
    # Utilities
    apply_fsdp_wrapping,
)

# Federated Learning (v81.0)
from reactor_core.training.federated_learning import (
    # Enums
    AggregationStrategy,
    ClientSelectionStrategy,
    # Data structures
    ClientUpdate,
    AggregationResult,
    FederatedMetrics,
    # Config
    FederatedConfig,
    # Server & Client
    FederatedServer,
    FederatedClient,
    # Utilities
    create_federated_setup,
)

# Cognitive Modules (v81.0)
from reactor_core.training.cognitive_modules import (
    # Enums
    CognitiveModuleType,
    PlanningStrategy,
    ReasoningType,
    MemoryType,
    # Data structures
    CognitiveState,
    CognitiveModuleConfig,
    TrainingBatch as CognitiveTrainingBatch,
    # Base module
    BaseCognitiveModule,
    CognitiveLayer,
    # Specialized modules
    PlanningModule,
    ReasoningModule,
    MemoryModule,
    PerceptionModule,
    # Orchestrator
    CognitiveOrchestrator,
    # Training
    CognitiveTrainingConfig,
    CognitiveModuleTrainer,
    # Utilities
    create_cognitive_system,
)

# Online Learning (v238.0 — activated from sleeping)
try:
    from reactor_core.training.online_learning import (
        OnlineLearningEngine,
        create_online_learning_engine,
        FeedbackIntegrator,
        Experience as OnlineExperience,
        LearningMetrics,
        LearningMode,
        FeedbackType,
        ForgetPreventionMethod,
        PrioritizedExperienceBuffer,
        DriftDetector,
        ElasticWeightConsolidation,
    )
    _ONLINE_LEARNING_AVAILABLE = True
except ImportError:
    _ONLINE_LEARNING_AVAILABLE = False

__all__ = [
    # Trainer
    "Trainer",
    "AsyncTrainer",
    "TrainingConfig",
    "TrainingProgress",
    "TrainingResult",
    "TrainingState",
    # Callbacks
    "TrainingEvent",
    "BaseCallback",
    "CallbackMixin",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "GPUMemoryCallback",
    "RichProgressCallback",
    "WebhookCallback",
    "CompositeCallback",
    "create_default_callbacks",
    # LoRA Enums
    "QuantizationType",
    "TaskType",
    # LoRA Configs
    "LoRAConfig",
    "QLoRAConfig",
    # LoRA Core
    "apply_lora",
    "apply_qlora",
    "prepare_model_for_qlora",
    "load_model_for_qlora",
    # LoRA Save/Load
    "merge_and_unload",
    "save_adapter",
    "load_adapter",
    # LoRA Adapter Management
    "set_adapter",
    "disable_adapter",
    "enable_adapter",
    "get_adapter_info",
    # LoRA Utilities
    "detect_model_architecture",
    "get_trainable_parameters",
    "format_params",
    "create_optimal_lora_config",
    "TARGET_MODULES_MAP",
    # === ADVANCED TRAINING (v76.0) ===
    # Memory Optimization
    "MemoryManager",
    "ReferenceModelManager",
    "DynamicBatchSizer",
    "get_memory_manager",
    # Training Method Enums
    "TrainingMethod",
    "SafetyTier",
    "CurriculumStrategy",
    "ExperiencePriority",
    # Data Structures
    "PreferencePair",
    "Experience",
    "TrainingBatch",
    "TrainingMetrics",
    # Experience Buffer
    "ExperienceBuffer",
    # DPO
    "DPOConfig",
    "DPOTrainer",
    # RLHF
    "RLHFConfig",
    "RewardModel",
    "PPOTrainer",
    "RLHFPipeline",
    # Constitutional AI
    "ConstitutionalPrinciple",
    "ConstitutionalAITrainer",
    "DEFAULT_CONSTITUTIONAL_PRINCIPLES",
    # Curriculum Learning
    "CurriculumConfig",
    "DifficultyScorer",
    "CurriculumScheduler",
    # FSDP
    "FSDPConfig",
    "FSDPWrapper",
    # Unified Trainer
    "AdvancedTrainingConfig",
    "AdvancedTrainer",
    # === CURRICULUM LEARNING (v79.0) ===
    "DifficultyMetric",
    "DifficultyScore",
    "LossDifficultyScorer",
    "LengthDifficultyScorer",
    "CurriculumStage",
    "CurriculumSampler",
    "CurriculumLearner",
    "create_default_curriculum",
    # === META-LEARNING (v79.0) ===
    "MetaAlgorithm",
    "Task",
    "TaskSampler",
    "MAMLConfig",
    "MAMLTrainer",
    "ReptileConfig",
    "ReptileTrainer",
    "MetaSGDConfig",
    "MetaSGDTrainer",
    "create_n_way_k_shot_task",
    # === WORLD MODEL TRAINING (v80.0) ===
    # Components
    "LatentEncoder",
    "LatentDecoder",
    "TransitionModel",
    "RewardModel",
    "ValueModel",
    # World Model
    "WorldModelConfig",
    "WorldModel",
    # Training
    "WorldModelTrainingConfig",
    "WorldModelTrainer",
    # Reasoning
    "CounterfactualReasoner",
    # === CAUSAL REASONING (v80.0) ===
    # Causal Graph
    "CausalEdge",
    "CausalGraph",
    # Structural Causal Model
    "StructuralCausalModel",
    # Causal Discovery
    "CausalDiscovery",
    # Neural Causal Model
    "NeuralCausalModel",
    # Causal Attention
    "CausalAttention",
    # Evaluation
    "CausalEvaluationMetrics",
    "evaluate_causal_graph",
    # === FSDP TRAINING (v81.0) ===
    # Enums
    "FSDPShardingStrategy",
    "FSDPMixedPrecisionPolicy",
    "FSDPBackwardPrefetch",
    "FSDPStateDictType",
    # Config
    "FSDPTrainingConfig",
    # Trainer
    "FSDPTrainer",
    # Utilities
    "apply_fsdp_wrapping",
    # === FEDERATED LEARNING (v81.0) ===
    # Enums
    "AggregationStrategy",
    "ClientSelectionStrategy",
    # Data structures
    "ClientUpdate",
    "AggregationResult",
    "FederatedMetrics",
    # Config
    "FederatedConfig",
    # Server & Client
    "FederatedServer",
    "FederatedClient",
    # Utilities
    "create_federated_setup",
    # === COGNITIVE MODULES (v81.0) ===
    # Enums
    "CognitiveModuleType",
    "PlanningStrategy",
    "ReasoningType",
    "MemoryType",
    # Data structures
    "CognitiveState",
    "CognitiveModuleConfig",
    "CognitiveTrainingBatch",
    # Base module
    "BaseCognitiveModule",
    "CognitiveLayer",
    # Specialized modules
    "PlanningModule",
    "ReasoningModule",
    "MemoryModule",
    "PerceptionModule",
    # Orchestrator
    "CognitiveOrchestrator",
    # Training
    "CognitiveTrainingConfig",
    "CognitiveModuleTrainer",
    # Utilities
    "create_cognitive_system",
    # === ONLINE LEARNING (v238.0 — activated from sleeping) ===
    "OnlineLearningEngine",
    "create_online_learning_engine",
    "FeedbackIntegrator",
    "OnlineExperience",
    "LearningMetrics",
    "LearningMode",
    "FeedbackType",
    "ForgetPreventionMethod",
    "PrioritizedExperienceBuffer",
    "DriftDetector",
    "ElasticWeightConsolidation",
]
