"""
Night Shift Pipeline orchestration.

Provides:
- End-to-end training pipeline with Safe Scout integration
- Dual data sources: Web documentation + JARVIS experience logs
- Stage management and recovery
- Checkpoint-based resumption
- Progress tracking with async event streaming
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    IDLE = "idle"
    SCOUTING = "scouting"          # NEW: Web documentation ingestion
    INGESTING = "ingesting"         # JARVIS experience ingestion
    FORMATTING = "formatting"
    DISTILLING = "distilling"
    TRAINING = "training"
    EVALUATING = "evaluating"
    QUANTIZING = "quantizing"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


class DataSource(Enum):
    """Sources of training data."""
    SCOUT = "scout"              # Web documentation
    JARVIS_EXPERIENCE = "jarvis_experience"  # JARVIS logs
    JARVIS_CORRECTIONS = "jarvis_corrections"  # User corrections
    SYNTHETIC = "synthetic"      # Teacher-generated


@dataclass
class PipelineState:
    """Current state of the pipeline."""
    run_id: str
    stage: PipelineStage
    started_at: datetime
    last_updated: datetime = field(default_factory=datetime.now)
    last_completed_stage: Optional[PipelineStage] = None

    # Data source tracking
    enabled_sources: Set[DataSource] = field(default_factory=lambda: {
        DataSource.SCOUT, DataSource.JARVIS_EXPERIENCE
    })

    # Scout stage counters
    scout_topics_processed: int = 0
    scout_pages_fetched: int = 0
    scout_pages_blocked: int = 0
    scout_examples_synthesized: int = 0

    # Progress counters
    ingestion_count: int = 0
    formatted_count: int = 0
    distilled_count: int = 0

    # Training state
    training_checkpoint: Optional[str] = None
    training_step: int = 0

    # Evaluation state
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    gatekeeper_passed: bool = False

    # Output artifacts
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    quantized_path: Optional[str] = None

    # Error tracking
    error: Optional[str] = None
    error_stage: Optional[PipelineStage] = None
    stage_errors: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage.value,
            "started_at": self.started_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "last_completed_stage": (
                self.last_completed_stage.value
                if self.last_completed_stage else None
            ),
            "enabled_sources": [s.value for s in self.enabled_sources],
            "scout_topics_processed": self.scout_topics_processed,
            "scout_pages_fetched": self.scout_pages_fetched,
            "scout_pages_blocked": self.scout_pages_blocked,
            "scout_examples_synthesized": self.scout_examples_synthesized,
            "ingestion_count": self.ingestion_count,
            "formatted_count": self.formatted_count,
            "distilled_count": self.distilled_count,
            "training_checkpoint": self.training_checkpoint,
            "training_step": self.training_step,
            "eval_metrics": self.eval_metrics,
            "gatekeeper_passed": self.gatekeeper_passed,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "quantized_path": self.quantized_path,
            "error": self.error,
            "error_stage": self.error_stage.value if self.error_stage else None,
            "stage_errors": self.stage_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        enabled_sources = {
            DataSource(s) for s in data.get("enabled_sources", ["scout", "jarvis_experience"])
        }
        return cls(
            run_id=data["run_id"],
            stage=PipelineStage(data["stage"]),
            started_at=datetime.fromisoformat(data["started_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            last_completed_stage=(
                PipelineStage(data["last_completed_stage"])
                if data.get("last_completed_stage") else None
            ),
            enabled_sources=enabled_sources,
            scout_topics_processed=data.get("scout_topics_processed", 0),
            scout_pages_fetched=data.get("scout_pages_fetched", 0),
            scout_pages_blocked=data.get("scout_pages_blocked", 0),
            scout_examples_synthesized=data.get("scout_examples_synthesized", 0),
            ingestion_count=data.get("ingestion_count", 0),
            formatted_count=data.get("formatted_count", 0),
            distilled_count=data.get("distilled_count", 0),
            training_checkpoint=data.get("training_checkpoint"),
            training_step=data.get("training_step", 0),
            eval_metrics=data.get("eval_metrics", {}),
            gatekeeper_passed=data.get("gatekeeper_passed", False),
            model_path=data.get("model_path"),
            adapter_path=data.get("adapter_path"),
            quantized_path=data.get("quantized_path"),
            error=data.get("error"),
            error_stage=(
                PipelineStage(data["error_stage"])
                if data.get("error_stage") else None
            ),
            stage_errors=data.get("stage_errors", {}),
        )

    def update_stage(self, stage: PipelineStage) -> None:
        """Update current stage."""
        if self.stage != PipelineStage.FAILED:
            self.last_completed_stage = self.stage
        self.stage = stage
        self.last_updated = datetime.now()

    def set_error(self, error: str) -> None:
        """Set error state."""
        self.error = error
        self.error_stage = self.stage
        self.stage = PipelineStage.FAILED
        self.last_updated = datetime.now()


@dataclass
class PipelineResult:
    """Final result of pipeline execution."""
    success: bool
    run_id: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    final_state: PipelineState
    artifacts: Dict[str, str]
    metrics: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "final_state": self.final_state.to_dict(),
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "error": self.error,
        }

    def summary(self) -> str:
        lines = [
            f"Pipeline Run: {self.run_id}",
            f"Status: {'SUCCESS' if self.success else 'FAILED'}",
            f"Duration: {self.duration_seconds / 60:.1f} minutes",
            "",
            "Stages Completed:",
        ]

        if self.final_state.last_completed_stage:
            stages = list(PipelineStage)
            idx = stages.index(self.final_state.last_completed_stage)
            for s in stages[1:idx+1]:  # Skip IDLE
                lines.append(f"  ✓ {s.value}")

        if self.error:
            lines.append(f"\nError: {self.error}")

        if self.artifacts:
            lines.append("\nArtifacts:")
            for name, path in self.artifacts.items():
                lines.append(f"  {name}: {path}")

        return "\n".join(lines)


@dataclass
class PipelineConfig:
    """Configuration for the Night Shift pipeline."""
    # Directories
    work_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "NIGHTSHIFT_WORK_DIR",
            Path.home() / ".jarvis" / "nightshift"
        ))
    )
    log_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # Stage configuration
    skip_stages: List[PipelineStage] = field(default_factory=list)
    stop_after: Optional[PipelineStage] = None

    # Data sources
    enabled_sources: Set[DataSource] = field(
        default_factory=lambda: {DataSource.SCOUT, DataSource.JARVIS_EXPERIENCE}
    )

    # Scout configuration
    scout_max_topics: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_MAX_TOPICS", "50"))
    )
    scout_max_pages_per_topic: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_MAX_PAGES", "10"))
    )
    scout_concurrency: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_CONCURRENCY", "5"))
    )
    scout_use_docker: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SCOUT_DOCKER", "true").lower() == "true"
    )
    scout_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_TIMEOUT", "30"))
    )
    scout_synthesis_model: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_SCOUT_MODEL", "gemini-1.5-flash"
        )
    )
    scout_trusted_domains: Optional[List[str]] = None
    scout_blocked_domains: Optional[List[str]] = None

    # JARVIS ingestion
    jarvis_repo_path: Path = field(
        default_factory=lambda: Path(os.getenv(
            "JARVIS_REPO_PATH",
            Path.home() / "Documents" / "repos" / "JARVIS-AI-Agent"
        ))
    )
    jarvis_lookback_hours: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LOOKBACK_HOURS", "168"))
    )
    log_sources: List[Path] = field(default_factory=list)
    min_examples: int = 100

    # JARVIS Prime integration
    prime_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_PRIME_ENABLED", "false").lower() == "true"
    )
    prime_host: str = field(
        default_factory=lambda: os.getenv("JARVIS_PRIME_HOST", "localhost")
    )
    prime_port: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_PRIME_PORT", "8002"))
    )

    # Distillation
    enable_distillation: bool = True
    distillation_budget: float = 10.0
    distillation_model: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_DISTILL_MODEL", "gpt-4o"
        )
    )

    # Training
    base_model: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_BASE_MODEL",
            "meta-llama/Llama-3.2-3B"
        )
    )
    num_epochs: int = 3
    lora_rank: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_RANK", "64"))
    )
    lora_alpha: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_ALPHA", "128"))
    )

    # Evaluation
    eval_threshold: float = 0.7
    require_gatekeeper: bool = True

    # Quantization
    quantization_method: str = "q4_k_m"
    skip_quantization: bool = False

    # Recovery
    state_file: Optional[Path] = None
    resume_on_error: bool = True
    max_retries_per_stage: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_dir": str(self.work_dir),
            "log_dir": str(self.log_dir) if self.log_dir else None,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "skip_stages": [s.value for s in self.skip_stages],
            "stop_after": self.stop_after.value if self.stop_after else None,
            "enabled_sources": [s.value for s in self.enabled_sources],
            # Scout config
            "scout_max_topics": self.scout_max_topics,
            "scout_max_pages_per_topic": self.scout_max_pages_per_topic,
            "scout_concurrency": self.scout_concurrency,
            "scout_use_docker": self.scout_use_docker,
            "scout_timeout_seconds": self.scout_timeout_seconds,
            "scout_synthesis_model": self.scout_synthesis_model,
            "scout_trusted_domains": self.scout_trusted_domains,
            "scout_blocked_domains": self.scout_blocked_domains,
            # JARVIS config
            "jarvis_repo_path": str(self.jarvis_repo_path),
            "jarvis_lookback_hours": self.jarvis_lookback_hours,
            "log_sources": [str(p) for p in self.log_sources],
            "min_examples": self.min_examples,
            # Prime config
            "prime_enabled": self.prime_enabled,
            "prime_host": self.prime_host,
            "prime_port": self.prime_port,
            # Distillation config
            "enable_distillation": self.enable_distillation,
            "distillation_budget": self.distillation_budget,
            "distillation_model": self.distillation_model,
            # Training config
            "base_model": self.base_model,
            "num_epochs": self.num_epochs,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            # Eval config
            "eval_threshold": self.eval_threshold,
            "require_gatekeeper": self.require_gatekeeper,
            # Quantization config
            "quantization_method": self.quantization_method,
            "skip_quantization": self.skip_quantization,
            # Recovery config
            "resume_on_error": self.resume_on_error,
            "max_retries_per_stage": self.max_retries_per_stage,
        }


class NightShiftPipeline:
    """
    Night Shift autonomous training pipeline.

    Orchestrates the full training cycle:
    1. Ingestion - Parse JARVIS logs
    2. Formatting - Convert to training format
    3. Distillation - Improve examples with teacher model
    4. Training - Fine-tune with LoRA
    5. Evaluation - Run benchmarks
    6. Quantization - Convert to GGUF
    7. Deployment - Update model registry
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize directories
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_dir:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # State management
        self._state: Optional[PipelineState] = None
        self._state_file = self.config.state_file or (
            self.config.work_dir / "pipeline_state.json"
        )

        # Stage handlers
        self._stage_handlers: Dict[PipelineStage, Callable] = {}

        # Callbacks
        self._progress_callback: Optional[Callable[[PipelineState], None]] = None
        self._error_callback: Optional[Callable[[Exception, PipelineStage], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[PipelineState], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def set_error_callback(
        self,
        callback: Callable[[Exception, PipelineStage], None],
    ) -> None:
        """Set callback for error handling."""
        self._error_callback = callback

    def register_stage_handler(
        self,
        stage: PipelineStage,
        handler: Callable,
    ) -> None:
        """Register a custom handler for a stage."""
        self._stage_handlers[stage] = handler

    def _save_state(self) -> None:
        """Save pipeline state to file."""
        if self._state:
            with open(self._state_file, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)

    def _load_state(self) -> Optional[PipelineState]:
        """Load pipeline state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                return PipelineState.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return None

    def _update_stage(self, stage: PipelineStage) -> None:
        """Update current stage and notify."""
        if self._state:
            self._state.update_stage(stage)
            self._save_state()

            if self._progress_callback:
                self._progress_callback(self._state)

            logger.info(f"Pipeline stage: {stage.value}")

    async def _run_stage(
        self,
        stage: PipelineStage,
        handler: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run a pipeline stage with error handling."""
        if stage in self.config.skip_stages:
            logger.info(f"Skipping stage: {stage.value}")
            return None

        self._update_stage(stage)

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler, *args)

            return result

        except Exception as e:
            logger.error(f"Stage {stage.value} failed: {e}")

            if self._error_callback:
                self._error_callback(e, stage)

            if self._state:
                self._state.set_error(str(e))
                self._save_state()

            raise

    async def _run_scouting(self) -> Dict[str, int]:
        """
        Run Safe Scout web documentation ingestion.

        Returns:
            Dict with scout statistics
        """
        if DataSource.SCOUT not in self.config.enabled_sources:
            logger.info("Scout source disabled, skipping...")
            return {"topics": 0, "pages": 0, "examples": 0}

        logger.info("Starting Scout stage - web documentation ingestion...")

        # Import scout modules
        from reactor_core.scout import (
            TopicQueue,
            TopicQueueConfig,
            URLValidator,
            URLValidatorConfig,
            ComplianceFilter,
            SandboxExecutor,
            SandboxConfig,
            ExecutionMode,
            ContentExtractor,
            KnowledgeSynthesizer,
        )
        from reactor_core.distillation import GeminiClient, create_teacher_client

        # Initialize scout components
        queue_config = TopicQueueConfig(
            db_path=self.config.work_dir / "scout_queue.db",
            max_concurrent_topics=self.config.scout_concurrency,
        )
        topic_queue = TopicQueue(queue_config)

        validator_config = URLValidatorConfig(
            check_robots_txt=True,
            check_safe_browsing=False,  # Requires API key
            request_timeout=10.0,
        )
        # Apply custom trusted/blocked domains if provided
        if self.config.scout_trusted_domains:
            validator_config.additional_trusted = self.config.scout_trusted_domains
        if self.config.scout_blocked_domains:
            validator_config.additional_blocked = self.config.scout_blocked_domains

        validator = URLValidator(validator_config)
        compliance = ComplianceFilter()

        # Determine execution mode
        exec_mode = (
            ExecutionMode.DOCKER if self.config.scout_use_docker
            else ExecutionMode.SUBPROCESS
        )
        sandbox_config = SandboxConfig(
            mode=exec_mode,
            timeout_seconds=self.config.scout_timeout_seconds,
            max_concurrent=self.config.scout_concurrency,
        )
        sandbox = SandboxExecutor(sandbox_config)

        extractor = ContentExtractor()

        # Initialize teacher for synthesis
        teacher = create_teacher_client(self.config.scout_synthesis_model)
        synthesizer = KnowledgeSynthesizer(teacher)

        # Statistics
        stats = {
            "topics_processed": 0,
            "pages_fetched": 0,
            "pages_blocked": 0,
            "pages_failed": 0,
            "examples_synthesized": 0,
            "errors": [],
        }

        # Process topics from queue
        topics = await topic_queue.get_pending_topics(
            limit=self.config.scout_max_topics
        )
        logger.info(f"Found {len(topics)} pending topics to process")

        for topic in topics:
            try:
                await topic_queue.mark_processing(topic.topic_id)
                stats["topics_processed"] += 1

                # Get URLs from topic
                urls = topic.urls[:self.config.scout_max_pages_per_topic]

                # Validate URLs in parallel
                validation_results = await validator.validate_batch(urls)

                for url, validation in zip(urls, validation_results):
                    if not validation.is_safe:
                        stats["pages_blocked"] += 1
                        logger.debug(f"Blocked URL: {url} - {validation.block_reason}")
                        continue

                    try:
                        # Fetch content via sandbox
                        sandbox_result = await sandbox.execute(url)

                        if not sandbox_result.success:
                            stats["pages_failed"] += 1
                            continue

                        stats["pages_fetched"] += 1

                        # Check compliance
                        compliance_result = compliance.check_compliance(
                            sandbox_result.html_content or "",
                            url
                        )

                        if not compliance_result.is_compliant:
                            stats["pages_blocked"] += 1
                            logger.debug(
                                f"Compliance block: {url} - {compliance_result.violations}"
                            )
                            continue

                        # Extract content
                        extracted = extractor.extract(
                            sandbox_result.html_content or "",
                            url
                        )

                        if not extracted.text_content:
                            continue

                        # Synthesize Q&A pairs
                        synthesis_result = await synthesizer.synthesize(
                            content=extracted.text_content,
                            title=extracted.title or topic.name,
                            code_blocks=extracted.code_blocks,
                            max_pairs=5,
                        )

                        stats["examples_synthesized"] += len(synthesis_result.pairs)

                        # Store synthesized pairs
                        output_dir = self.config.work_dir / "scout_data"
                        output_dir.mkdir(exist_ok=True)

                        for pair in synthesis_result.pairs:
                            pair_file = output_dir / f"{pair.pair_id}.json"
                            with open(pair_file, "w") as f:
                                json.dump(pair.to_dict(), f, indent=2)

                    except Exception as e:
                        stats["pages_failed"] += 1
                        stats["errors"].append(f"{url}: {str(e)}")
                        logger.warning(f"Error processing URL {url}: {e}")

                await topic_queue.mark_completed(topic.topic_id)

            except Exception as e:
                await topic_queue.mark_failed(topic.topic_id, str(e))
                stats["errors"].append(f"Topic {topic.name}: {str(e)}")
                logger.error(f"Error processing topic {topic.name}: {e}")

        # Update pipeline state
        if self._state:
            self._state.scout_topics_processed = stats["topics_processed"]
            self._state.scout_pages_fetched = stats["pages_fetched"]
            self._state.scout_pages_blocked = stats["pages_blocked"]
            self._state.scout_examples_synthesized = stats["examples_synthesized"]

        # Cleanup
        await sandbox.cleanup()
        await topic_queue.close()

        logger.info(
            f"Scout complete: {stats['topics_processed']} topics, "
            f"{stats['pages_fetched']} pages, "
            f"{stats['examples_synthesized']} examples"
        )

        return stats

    async def _run_ingestion(self) -> int:
        """
        Run JARVIS experience ingestion stage.

        Ingests logs from JARVIS-AI-Agent and optionally JARVIS Prime.
        """
        if DataSource.JARVIS_EXPERIENCE not in self.config.enabled_sources:
            logger.info("JARVIS experience source disabled, skipping...")
            return 0

        logger.info("Starting JARVIS ingestion stage...")

        # Import modules
        from reactor_core.integration import (
            JARVISConnector,
            JARVISConnectorConfig,
        )
        from reactor_core.ingestion import BatchIngestionProcessor

        total = 0

        # JARVIS-AI-Agent logs
        if self.config.jarvis_repo_path.exists():
            connector_config = JARVISConnectorConfig(
                jarvis_repo_path=self.config.jarvis_repo_path,
                lookback_hours=self.config.jarvis_lookback_hours,
            )
            connector = JARVISConnector(connector_config)

            # Get events
            events = await connector.get_events(limit=5000)
            logger.info(f"Found {len(events)} JARVIS events")

            # Get corrections specifically
            if DataSource.JARVIS_CORRECTIONS in self.config.enabled_sources:
                corrections = await connector.get_corrections()
                logger.info(f"Found {len(corrections)} correction events")

            total += len(events)

            # Store events for formatting
            events_dir = self.config.work_dir / "jarvis_events"
            events_dir.mkdir(exist_ok=True)

            for event in events:
                event_file = events_dir / f"{event.event_id}.json"
                with open(event_file, "w") as f:
                    json.dump(event.to_dict(), f, indent=2)

        else:
            logger.warning(
                f"JARVIS repo not found at {self.config.jarvis_repo_path}"
            )

        # JARVIS Prime integration (if enabled)
        if self.config.prime_enabled:
            try:
                from reactor_core.integration import PrimeConnector

                prime = PrimeConnector(
                    host=self.config.prime_host,
                    port=self.config.prime_port,
                )

                prime_events = await prime.get_recent_interactions(
                    hours=self.config.jarvis_lookback_hours
                )
                logger.info(f"Found {len(prime_events)} JARVIS Prime events")
                total += len(prime_events)

            except ImportError:
                logger.warning("PrimeConnector not available")
            except Exception as e:
                logger.error(f"Error connecting to JARVIS Prime: {e}")

        # Process additional log sources
        processor = BatchIngestionProcessor()
        for source in self.config.log_sources:
            if source.exists():
                count = await processor.process_directory(source)
                total += count

        if self._state:
            self._state.ingestion_count = total

        logger.info(f"Ingested {total} raw interactions")
        return total

    async def _run_formatting(self) -> int:
        """Run formatting stage."""
        logger.info("Starting formatting stage...")

        # Placeholder - actual implementation would use formatting module
        formatted_count = self._state.ingestion_count if self._state else 0

        if self._state:
            self._state.formatted_count = formatted_count

        logger.info(f"Formatted {formatted_count} examples")
        return formatted_count

    async def _run_distillation(self) -> int:
        """Run distillation stage."""
        if not self.config.enable_distillation:
            logger.info("Distillation disabled, skipping...")
            return 0

        logger.info("Starting distillation stage...")

        # Placeholder - actual implementation would use distillation module
        distilled_count = 0

        if self._state:
            self._state.distilled_count = distilled_count

        logger.info(f"Distilled {distilled_count} examples")
        return distilled_count

    async def _run_training(self) -> Dict[str, Any]:
        """Run training stage."""
        logger.info("Starting training stage...")

        # Placeholder - actual implementation would use training module
        training_result = {
            "model_path": str(self.config.work_dir / "model"),
            "adapter_path": str(self.config.work_dir / "adapter"),
            "final_loss": 0.5,
        }

        if self._state:
            self._state.model_path = training_result["model_path"]
            self._state.adapter_path = training_result["adapter_path"]

        logger.info(f"Training complete: {training_result}")
        return training_result

    async def _run_evaluation(self) -> Dict[str, float]:
        """Run evaluation stage."""
        logger.info("Starting evaluation stage...")

        # Placeholder - actual implementation would use eval module
        metrics = {
            "overall_score": 0.85,
            "safety": 0.98,
            "instruction_following": 0.82,
        }

        gatekeeper_passed = metrics["overall_score"] >= self.config.eval_threshold

        if self._state:
            self._state.eval_metrics = metrics
            self._state.gatekeeper_passed = gatekeeper_passed

        logger.info(f"Evaluation complete: {metrics}, gatekeeper: {gatekeeper_passed}")
        return metrics

    async def _run_quantization(self) -> str:
        """Run quantization stage."""
        if self.config.skip_quantization:
            logger.info("Quantization disabled, skipping...")
            return ""

        logger.info("Starting quantization stage...")

        # Placeholder - actual implementation would use quantization module
        output_path = str(
            self.config.work_dir / f"model-{self.config.quantization_method}.gguf"
        )

        if self._state:
            self._state.quantized_path = output_path

        logger.info(f"Quantization complete: {output_path}")
        return output_path

    async def _run_deployment(self) -> None:
        """Run deployment stage."""
        logger.info("Starting deployment stage...")

        # Check gatekeeper
        if self.config.require_gatekeeper:
            if not self._state or not self._state.gatekeeper_passed:
                raise RuntimeError("Gatekeeper did not approve deployment")

        # Placeholder - actual deployment logic
        logger.info("Deployment complete")

    async def run(
        self,
        resume: bool = False,
    ) -> PipelineResult:
        """
        Run the complete pipeline.

        Args:
            resume: Resume from previous state if available

        Returns:
            PipelineResult with final status
        """
        import time
        start_time = time.time()

        # Initialize or resume state
        if resume:
            self._state = self._load_state()

        if self._state is None:
            self._state = PipelineState(
                run_id=str(uuid.uuid4())[:8],
                stage=PipelineStage.IDLE,
                started_at=datetime.now(),
            )

        run_id = self._state.run_id
        logger.info(f"Starting pipeline run: {run_id}")

        try:
            # Determine starting stage
            start_stage = self._state.stage
            if start_stage == PipelineStage.FAILED and self.config.resume_on_error:
                start_stage = self._state.error_stage or PipelineStage.IDLE

            stages = [
                (PipelineStage.SCOUTING, self._run_scouting),
                (PipelineStage.INGESTING, self._run_ingestion),
                (PipelineStage.FORMATTING, self._run_formatting),
                (PipelineStage.DISTILLING, self._run_distillation),
                (PipelineStage.TRAINING, self._run_training),
                (PipelineStage.EVALUATING, self._run_evaluation),
                (PipelineStage.QUANTIZING, self._run_quantization),
                (PipelineStage.DEPLOYING, self._run_deployment),
            ]

            # Skip completed stages on resume
            if resume and start_stage != PipelineStage.IDLE:
                stage_order = [s for s, _ in stages]
                if start_stage in stage_order:
                    idx = stage_order.index(start_stage)
                    stages = stages[idx:]

            # Run stages
            for stage, handler in stages:
                # Check for custom handler
                if stage in self._stage_handlers:
                    handler = self._stage_handlers[stage]

                await self._run_stage(stage, handler)

                # Check stop condition
                if self.config.stop_after == stage:
                    logger.info(f"Stopping after {stage.value} as configured")
                    break

            # Mark completed
            self._update_stage(PipelineStage.COMPLETED)

            duration = time.time() - start_time

            return PipelineResult(
                success=True,
                run_id=run_id,
                started_at=self._state.started_at,
                completed_at=datetime.now(),
                duration_seconds=duration,
                final_state=self._state,
                artifacts={
                    "model": self._state.model_path or "",
                    "adapter": self._state.adapter_path or "",
                    "quantized": self._state.quantized_path or "",
                },
                metrics=self._state.eval_metrics,
            )

        except Exception as e:
            duration = time.time() - start_time

            return PipelineResult(
                success=False,
                run_id=run_id,
                started_at=self._state.started_at,
                completed_at=datetime.now(),
                duration_seconds=duration,
                final_state=self._state,
                artifacts={},
                metrics=self._state.eval_metrics if self._state else {},
                error=str(e),
            )

    def get_state(self) -> Optional[PipelineState]:
        """Get current pipeline state."""
        return self._state

    def reset(self) -> None:
        """Reset pipeline state."""
        self._state = None
        if self._state_file.exists():
            self._state_file.unlink()


# Convenience exports
__all__ = [
    "NightShiftPipeline",
    "PipelineConfig",
    "PipelineStage",
    "PipelineState",
    "PipelineResult",
    "DataSource",
]
