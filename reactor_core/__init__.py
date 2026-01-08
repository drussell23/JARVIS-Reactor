"""
JARVIS Reactor Core - AGI Training & Learning Engine

The Nervous System of JARVIS AGI - Continuous learning pipeline with advanced training methods.

CORE CAPABILITIES:
- Safe Scout: Defensive web ingestion with compliance checking
- JARVIS Integration: Experience log ingestion from JARVIS-AI-Agent
- Dataset formatting (ChatML, Alpaca, DPO preference pairs)
- Quality filtering and deduplication
- Async-first training with LoRA/QLoRA
- Knowledge distillation via teacher models (GPT-4o, Claude, Gemini)
- Model evaluation with gatekeeper approval
- GGUF quantization for M1 Mac deployment
- Cron-based pipeline orchestration with notifications

ADVANCED TRAINING (v76.0-v80.0):
- DPO (Direct Preference Optimization)
- RLHF/PPO (Reinforcement Learning from Human Feedback)
- Constitutional AI training
- Curriculum Learning (v79.0) - Easy → Medium → Hard progression
- Meta-Learning (v79.0) - MAML, Reptile, Meta-SGD for few-shot learning
- World Model Training (v80.0) - Planning, counterfactual reasoning
- Causal Reasoning (v80.0) - Understanding cause-effect relationships

ADVANCED DATA PROCESSING (v80.0):
- Multi-stage preprocessing with quality gates
- Synthetic data generation (3-10x augmentation)
- Active learning for efficient labeling (50-70% cost reduction)
- Deduplication (exact + semantic)
- Contamination detection
- Format normalization

TRINITY INTEGRATION:
- Cross-repo communication (JARVIS ↔ Prime ↔ Reactor)
- Event-driven coordination
- Service discovery and health monitoring
- Unified supervisor for single-command startup
"""

__version__ = "2.2.0"  # v80.0 - Advanced Data + World Models + Causal Reasoning

# Core training
from reactor_core.training import (
    Trainer,
    TrainingConfig,
    LoRAConfig,
    QLoRAConfig,
    apply_lora,
    apply_qlora,
    merge_and_unload,
    save_adapter,
    load_adapter,
    create_optimal_lora_config,
    ProgressCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    CompositeCallback,
    create_default_callbacks,
)

# Environment detection
from reactor_core.utils.environment import (
    detect_environment,
    EnvironmentType,
    EnvironmentInfo,
    get_quantization_config,
)

# Configuration
from reactor_core.config import (
    NightShiftConfig,
    TrainingConfig as NightShiftTrainingConfig,
    IngestionConfig,
    DistillationConfig,
    load_config,
)

# Data ingestion
from reactor_core.ingestion import (
    RawInteraction,
    SourceType,
    InteractionOutcome,
    BatchIngestionProcessor,
    TelemetryIngestor,
    FeedbackIngestor,
)

# Dataset formatting
from reactor_core.formatting import (
    FormattedExample,
    OutputFormat,
    ChatMLFormatter,
    AlpacaFormatter,
    QualityFilter,
    DatasetBuilder,
)

# Distillation
from reactor_core.distillation import (
    TeacherClient,
    OpenAIClient,
    AnthropicClient,
    GeminiClient,
    create_teacher_client,
    ScoringEngine,
    ScoringResult,
    QualityScore,
    RewriterEngine,
    RewriteStrategy,
    RewriteResult,
    TokenBucketRateLimiter as DistillationRateLimiter,
    MultiTierRateLimiter,
    CostTracker,
    BudgetEnforcer,
)

# Safe Scout (Web Ingestion)
from reactor_core.scout import (
    TopicQueue,
    TopicQueueConfig,
    LearningTopic,
    TopicStatus,
    TopicPriority,
    TopicCategory,
    URLValidator,
    URLValidatorConfig,
    URLSafetyLevel,
    BlockReason,
    ComplianceFilter,
    ComplianceResult,
    ComplianceStatus,
    SandboxExecutor,
    SandboxConfig,
    SandboxResult,
    ContentExtractor,
    ExtractedContent,
    KnowledgeSynthesizer,
    SynthesizedPair,
    SynthesisResult,
)

# JARVIS Integration
from reactor_core.integration import (
    JARVISConnector,
    JARVISConnectorConfig,
    JARVISEvent,
    EventType,
    CorrectionType,
)

# Evaluation
from reactor_core.eval import (
    BaseEvaluator,
    EvaluationResult,
    MetricResult,
    CompositeEvaluator,
    JARVISEvaluator,
    Gatekeeper,
    ApprovalDecision,
    ApprovalStatus,
)

# Quantization
from reactor_core.quantization import (
    GGUFConverter,
    GGUFConfig,
    QuantizationMethod,
    ConversionResult,
    LlamaCppBridge,
    LlamaCppConfig,
    InferenceResult,
    load_gguf_model,
    convert_to_gguf,
)

# Orchestration
from reactor_core.orchestration import (
    NightShiftPipeline,
    PipelineConfig,
    PipelineStage,
    PipelineState,
    PipelineResult,
    DataSource,
    PipelineScheduler,
    ScheduleConfig,
    ScheduledRun,
    NotificationManager,
    NotificationConfig,
    NotificationType,
    SlackNotifier,
    WebhookNotifier,
)

# Utilities
from reactor_core.utils import (
    setup_logging,
    get_logger,
    TokenBucketRateLimiter,
    ParallelBatchProcessor,
    async_retry,
)

# === PHASE 2 - ADVANCED FEATURES (v79.0-v80.0) ===

# Advanced Data Processing (v80.0)
from reactor_core.data import (
    # Preprocessing
    PreprocessingPipeline,
    PreprocessingConfig,
    QualityScorer,
    DeduplicationStrategy,
    # Synthetic Data
    SyntheticDataGenerator,
    SyntheticDataConfig,
    AugmentationStrategy,
    # Active Learning
    ActiveLearningLoop,
    ActiveLearningConfig,
    SamplingStrategy,
)

# Advanced Training (v79.0-v80.0)
from reactor_core.training import (
    # Curriculum Learning (v79.0)
    CurriculumLearner,
    CurriculumConfig,
    CurriculumStrategy,
    # Meta-Learning (v79.0)
    MAMLTrainer,
    MAMLConfig,
    ReptileTrainer,
    MetaSGDTrainer,
    # World Model Training (v80.0)
    WorldModel,
    WorldModelConfig,
    WorldModelTrainer,
    CounterfactualReasoner,
    # Causal Reasoning (v80.0)
    CausalGraph,
    StructuralCausalModel,
    NeuralCausalModel,
    CausalDiscovery,
    # DPO (v76.0)
    DPOTrainer,
    DPOConfig,
    # RLHF (v76.0)
    PPOTrainer,
    RLHFConfig,
)

# Trinity Integration
from reactor_core.integration import (
    TrinityConnector,
    get_trinity_connector,
    EventBridge,
    create_event_bridge,
)

__all__ = [
    # Version
    "__version__",
    # Core training
    "Trainer",
    "TrainingConfig",
    "LoRAConfig",
    "QLoRAConfig",
    "apply_lora",
    "apply_qlora",
    "merge_and_unload",
    "save_adapter",
    "load_adapter",
    "create_optimal_lora_config",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "CompositeCallback",
    "create_default_callbacks",
    # Environment
    "detect_environment",
    "EnvironmentType",
    "EnvironmentInfo",
    "get_quantization_config",
    # Configuration
    "NightShiftConfig",
    "NightShiftTrainingConfig",
    "IngestionConfig",
    "DistillationConfig",
    "load_config",
    # Ingestion
    "RawInteraction",
    "SourceType",
    "InteractionOutcome",
    "BatchIngestionProcessor",
    "TelemetryIngestor",
    "FeedbackIngestor",
    # Formatting
    "FormattedExample",
    "OutputFormat",
    "ChatMLFormatter",
    "AlpacaFormatter",
    "QualityFilter",
    "DatasetBuilder",
    # Distillation
    "TeacherClient",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "create_teacher_client",
    "ScoringEngine",
    "ScoringResult",
    "QualityScore",
    "RewriterEngine",
    "RewriteStrategy",
    "RewriteResult",
    "DistillationRateLimiter",
    "MultiTierRateLimiter",
    "CostTracker",
    "BudgetEnforcer",
    # Safe Scout
    "TopicQueue",
    "TopicQueueConfig",
    "LearningTopic",
    "TopicStatus",
    "TopicPriority",
    "TopicCategory",
    "URLValidator",
    "URLValidatorConfig",
    "URLSafetyLevel",
    "BlockReason",
    "ComplianceFilter",
    "ComplianceResult",
    "ComplianceStatus",
    "SandboxExecutor",
    "SandboxConfig",
    "SandboxResult",
    "ContentExtractor",
    "ExtractedContent",
    "KnowledgeSynthesizer",
    "SynthesizedPair",
    "SynthesisResult",
    # JARVIS Integration
    "JARVISConnector",
    "JARVISConnectorConfig",
    "JARVISEvent",
    "EventType",
    "CorrectionType",
    # Evaluation
    "BaseEvaluator",
    "EvaluationResult",
    "MetricResult",
    "CompositeEvaluator",
    "JARVISEvaluator",
    "Gatekeeper",
    "ApprovalDecision",
    "ApprovalStatus",
    # Quantization
    "GGUFConverter",
    "GGUFConfig",
    "QuantizationMethod",
    "ConversionResult",
    "LlamaCppBridge",
    "LlamaCppConfig",
    "InferenceResult",
    "load_gguf_model",
    "convert_to_gguf",
    # Orchestration
    "NightShiftPipeline",
    "PipelineConfig",
    "PipelineStage",
    "PipelineState",
    "PipelineResult",
    "DataSource",
    "PipelineScheduler",
    "ScheduleConfig",
    "ScheduledRun",
    "NotificationManager",
    "NotificationConfig",
    "NotificationType",
    "SlackNotifier",
    "WebhookNotifier",
    # Utilities
    "setup_logging",
    "get_logger",
    "TokenBucketRateLimiter",
    "ParallelBatchProcessor",
    "async_retry",
    # === PHASE 2 - ADVANCED FEATURES (v79.0-v80.0) ===
    # Advanced Data Processing
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "QualityScorer",
    "DeduplicationStrategy",
    "SyntheticDataGenerator",
    "SyntheticDataConfig",
    "AugmentationStrategy",
    "ActiveLearningLoop",
    "ActiveLearningConfig",
    "SamplingStrategy",
    # Advanced Training - Curriculum Learning
    "CurriculumLearner",
    "CurriculumConfig",
    "CurriculumStrategy",
    # Advanced Training - Meta-Learning
    "MAMLTrainer",
    "MAMLConfig",
    "ReptileTrainer",
    "MetaSGDTrainer",
    # Advanced Training - World Models
    "WorldModel",
    "WorldModelConfig",
    "WorldModelTrainer",
    "CounterfactualReasoner",
    # Advanced Training - Causal Reasoning
    "CausalGraph",
    "StructuralCausalModel",
    "NeuralCausalModel",
    "CausalDiscovery",
    # Advanced Training - DPO/RLHF
    "DPOTrainer",
    "DPOConfig",
    "PPOTrainer",
    "RLHFConfig",
    # Trinity Integration
    "TrinityConnector",
    "get_trinity_connector",
    "EventBridge",
    "create_event_bridge",
]
