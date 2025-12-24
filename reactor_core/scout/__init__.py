"""
Safe Scout module for Night Shift Training Engine.

Provides:
- Defensive web ingestion with compliance checking
- Domain allowlist/blocklist management
- Sandboxed browser execution
- Content extraction and cleaning
- Synthetic Q&A generation from documentation

IMPORTANT: The Scout treats the internet as UNTRUSTED by default.
All URLs must pass validation before scouting:
- Domain must be on allowlist OR explicitly added
- URL must comply with robots.txt
- Content must pass compliance filter (no paywalls, malware, etc.)
"""

# Topic Queue
from reactor_core.scout.topic_queue import (
    TopicQueue,
    TopicQueueConfig,
    LearningTopic,
    TopicStatus,
    TopicPriority,
    TopicCategory,
    SQLiteQueueBackend,
    create_documentation_topic,
    create_release_notes_topic,
    create_tutorial_topic,
)

# URL Validation
from reactor_core.scout.url_validator import (
    URLValidator,
    URLValidatorConfig,
    URLValidationResult,
    URLSafetyLevel,
    BlockReason,
    DomainTrustLevel,
    DEFAULT_TRUSTED_DOMAINS,
    DEFAULT_BLOCKED_DOMAINS,
)

# Compliance Filter
from reactor_core.scout.compliance_filter import (
    ComplianceFilter,
    ComplianceFilterConfig,
    ComplianceResult,
    ComplianceStatus,
    ComplianceViolation,
)

# Sandbox Executor
from reactor_core.scout.sandbox_executor import (
    SandboxExecutor,
    SandboxConfig,
    SandboxResult,
    ExecutionMode,
    PageLoadStatus,
)

# Content Extractor
from reactor_core.scout.content_extractor import (
    ContentExtractor,
    ExtractorConfig,
    ExtractedContent,
    ContentType,
    Section,
    CodeBlock,
)

# Knowledge Synthesizer
from reactor_core.scout.knowledge_synthesizer import (
    KnowledgeSynthesizer,
    SynthesizerConfig,
    SynthesizedPair,
    SynthesisResult,
    SynthesisStrategy,
    SynthesisQuality,
)

# Safe Scout Orchestrator
from reactor_core.scout.safe_scout_orchestrator import (
    SafeScoutOrchestrator,
    ScoutConfig,
    ScoutProgress,
    ScoutStage,
)

# Topic Discovery
from reactor_core.scout.topic_discovery import (
    TopicDiscovery,
    DiscoveredTopic,
    DiscoveryConfig,
    DiscoverySource,
    auto_discover_topics,
)

__all__ = [
    # Topic Queue
    "TopicQueue",
    "TopicQueueConfig",
    "LearningTopic",
    "TopicStatus",
    "TopicPriority",
    "TopicCategory",
    "SQLiteQueueBackend",
    "create_documentation_topic",
    "create_release_notes_topic",
    "create_tutorial_topic",
    # URL Validation
    "URLValidator",
    "URLValidatorConfig",
    "URLValidationResult",
    "URLSafetyLevel",
    "BlockReason",
    "DomainTrustLevel",
    "DEFAULT_TRUSTED_DOMAINS",
    "DEFAULT_BLOCKED_DOMAINS",
    # Compliance
    "ComplianceFilter",
    "ComplianceFilterConfig",
    "ComplianceResult",
    "ComplianceStatus",
    "ComplianceViolation",
    # Sandbox
    "SandboxExecutor",
    "SandboxConfig",
    "SandboxResult",
    "ExecutionMode",
    "PageLoadStatus",
    # Content Extraction
    "ContentExtractor",
    "ExtractorConfig",
    "ExtractedContent",
    "ContentType",
    "Section",
    "CodeBlock",
    # Knowledge Synthesis
    "KnowledgeSynthesizer",
    "SynthesizerConfig",
    "SynthesizedPair",
    "SynthesisResult",
    "SynthesisStrategy",
    "SynthesisQuality",
    # Orchestrator
    "SafeScoutOrchestrator",
    "ScoutConfig",
    "ScoutProgress",
    "ScoutStage",
    # Topic Discovery
    "TopicDiscovery",
    "DiscoveredTopic",
    "DiscoveryConfig",
    "DiscoverySource",
    "auto_discover_topics",
]
