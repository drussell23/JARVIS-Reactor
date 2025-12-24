"""
Safe Scout Orchestrator - Unified Web Documentation Ingestion.

This orchestrator combines all Scout components into a single,
easy-to-use interface for defensively ingesting web documentation.

Provides:
- End-to-end topic processing pipeline
- Parallel URL processing with concurrency control
- Automatic retry with exponential backoff
- Progress tracking and statistics
- Output in multiple formats
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ScoutStage(Enum):
    """Stages of the Scout pipeline."""
    IDLE = "idle"
    FETCHING_TOPICS = "fetching_topics"
    VALIDATING_URLS = "validating_urls"
    FETCHING_PAGES = "fetching_pages"
    CHECKING_COMPLIANCE = "checking_compliance"
    EXTRACTING_CONTENT = "extracting_content"
    SYNTHESIZING = "synthesizing"
    SAVING_OUTPUT = "saving_output"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScoutProgress:
    """Progress tracking for Scout orchestrator."""
    stage: ScoutStage = ScoutStage.IDLE
    topics_total: int = 0
    topics_processed: int = 0
    urls_total: int = 0
    urls_validated: int = 0
    urls_blocked: int = 0
    pages_fetched: int = 0
    pages_failed: int = 0
    examples_synthesized: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        total = self.pages_fetched + self.pages_failed
        return self.pages_fetched / total if total > 0 else 0.0

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "topics_total": self.topics_total,
            "topics_processed": self.topics_processed,
            "urls_total": self.urls_total,
            "urls_validated": self.urls_validated,
            "urls_blocked": self.urls_blocked,
            "pages_fetched": self.pages_fetched,
            "pages_failed": self.pages_failed,
            "examples_synthesized": self.examples_synthesized,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors[-10:],  # Last 10 errors
        }


@dataclass
class ScoutConfig:
    """Configuration for Safe Scout Orchestrator."""
    # Work directory
    work_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "NIGHTSHIFT_WORK_DIR",
            Path.home() / ".jarvis" / "nightshift"
        ))
    )

    # Topic processing
    max_topics: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_MAX_TOPICS", "50"))
    )
    max_pages_per_topic: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_MAX_PAGES", "10"))
    )

    # Concurrency
    url_concurrency: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_CONCURRENCY", "5"))
    )
    synthesis_concurrency: int = 3

    # Timeouts
    page_timeout_seconds: int = 30
    synthesis_timeout_seconds: int = 60

    # Sandbox settings
    use_docker: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SCOUT_DOCKER", "true").lower() == "true"
    )

    # Synthesis
    synthesis_model: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SCOUT_MODEL", "gemini-1.5-flash")
    )
    max_pairs_per_page: int = 5

    # Validation
    check_robots_txt: bool = True
    check_safe_browsing: bool = False  # Requires API key

    # Custom domain rules
    additional_trusted_domains: List[str] = field(default_factory=list)
    additional_blocked_domains: List[str] = field(default_factory=list)

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff: float = 2.0

    # Output
    output_format: str = "json"  # json, jsonl, chatml

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_dir": str(self.work_dir),
            "max_topics": self.max_topics,
            "max_pages_per_topic": self.max_pages_per_topic,
            "url_concurrency": self.url_concurrency,
            "use_docker": self.use_docker,
            "synthesis_model": self.synthesis_model,
            "check_robots_txt": self.check_robots_txt,
        }


class SafeScoutOrchestrator:
    """
    Main orchestrator for Safe Scout web documentation ingestion.

    Combines topic queue, URL validation, sandbox execution,
    compliance filtering, content extraction, and knowledge synthesis
    into a unified pipeline.
    """

    def __init__(
        self,
        config: Optional[ScoutConfig] = None,
    ):
        """
        Initialize the Scout orchestrator.

        Args:
            config: Configuration options
        """
        self.config = config or ScoutConfig()
        self._progress = ScoutProgress()
        self._progress_callbacks: List[Callable[[ScoutProgress], None]] = []
        self._running = False
        self._cancelled = False

        # Components (lazy initialized)
        self._queue = None
        self._validator = None
        self._compliance = None
        self._sandbox = None
        self._extractor = None
        self._synthesizer = None
        self._teacher = None

    @property
    def progress(self) -> ScoutProgress:
        """Get current progress."""
        return self._progress

    def add_progress_callback(
        self,
        callback: Callable[[ScoutProgress], None],
    ) -> None:
        """Add callback for progress updates."""
        self._progress_callbacks.append(callback)

    def _notify_progress(self) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self._progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _update_stage(self, stage: ScoutStage) -> None:
        """Update current stage."""
        self._progress.stage = stage
        self._notify_progress()
        logger.info(f"Scout stage: {stage.value}")

    async def _initialize_components(self) -> None:
        """Initialize all Scout components."""
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
        from reactor_core.distillation import create_teacher_client

        # Ensure work directory exists
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # Topic queue
        queue_config = TopicQueueConfig(
            db_path=self.config.work_dir / "scout_queue.db",
            max_concurrent_topics=self.config.url_concurrency,
        )
        self._queue = TopicQueue(queue_config)

        # URL validator
        validator_config = URLValidatorConfig(
            check_robots_txt=self.config.check_robots_txt,
            check_safe_browsing=self.config.check_safe_browsing,
            request_timeout=10.0,
        )
        if self.config.additional_trusted_domains:
            validator_config.additional_trusted = self.config.additional_trusted_domains
        if self.config.additional_blocked_domains:
            validator_config.additional_blocked = self.config.additional_blocked_domains

        self._validator = URLValidator(validator_config)

        # Compliance filter
        self._compliance = ComplianceFilter()

        # Sandbox executor
        exec_mode = ExecutionMode.DOCKER if self.config.use_docker else ExecutionMode.SUBPROCESS
        sandbox_config = SandboxConfig(
            mode=exec_mode,
            timeout_seconds=self.config.page_timeout_seconds,
            max_concurrent=self.config.url_concurrency,
        )
        self._sandbox = SandboxExecutor(sandbox_config)

        # Content extractor
        self._extractor = ContentExtractor()

        # Knowledge synthesizer
        try:
            self._teacher = create_teacher_client(self.config.synthesis_model)
            self._synthesizer = KnowledgeSynthesizer(self._teacher)
        except Exception as e:
            logger.warning(f"Could not initialize synthesizer: {e}")
            self._synthesizer = None

    async def _cleanup_components(self) -> None:
        """Cleanup all components."""
        if self._sandbox:
            await self._sandbox.cleanup()
        if self._queue:
            await self._queue.close()

    async def _process_url(
        self,
        url: str,
        topic_name: str,
        semaphore: asyncio.Semaphore,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single URL through the pipeline.

        Returns:
            Dict with processing result or None if failed
        """
        async with semaphore:
            if self._cancelled:
                return None

            result = {
                "url": url,
                "topic": topic_name,
                "success": False,
                "content": None,
                "pairs": [],
                "error": None,
            }

            # Validate URL
            validation = await self._validator.validate(url)
            self._progress.urls_validated += 1

            if not validation.is_safe:
                self._progress.urls_blocked += 1
                result["error"] = f"Blocked: {validation.block_reason}"
                return result

            # Fetch via sandbox
            try:
                sandbox_result = await self._sandbox.execute(url)

                if not sandbox_result.success:
                    self._progress.pages_failed += 1
                    result["error"] = sandbox_result.error or "Fetch failed"
                    return result

                self._progress.pages_fetched += 1

                # Check compliance
                compliance_result = self._compliance.check_compliance(
                    sandbox_result.html_content or "",
                    url
                )

                if not compliance_result.is_compliant:
                    self._progress.urls_blocked += 1
                    result["error"] = f"Compliance: {compliance_result.violations}"
                    return result

                # Extract content
                extracted = self._extractor.extract(
                    sandbox_result.html_content or "",
                    url
                )

                if not extracted.text_content:
                    result["error"] = "No content extracted"
                    return result

                result["content"] = {
                    "title": extracted.title,
                    "text": extracted.text_content[:5000],  # Limit size
                    "code_blocks": len(extracted.code_blocks),
                }

                # Synthesize Q&A pairs
                if self._synthesizer and extracted.text_content:
                    try:
                        syn_result = await asyncio.wait_for(
                            self._synthesizer.synthesize(
                                content=extracted.text_content,
                                title=extracted.title or topic_name,
                                code_blocks=extracted.code_blocks,
                                max_pairs=self.config.max_pairs_per_page,
                            ),
                            timeout=self.config.synthesis_timeout_seconds,
                        )

                        result["pairs"] = [p.to_dict() for p in syn_result.pairs]
                        self._progress.examples_synthesized += len(syn_result.pairs)

                    except asyncio.TimeoutError:
                        result["error"] = "Synthesis timeout"
                    except Exception as e:
                        result["error"] = f"Synthesis error: {e}"

                result["success"] = True
                return result

            except Exception as e:
                self._progress.pages_failed += 1
                self._progress.errors.append(f"{url}: {str(e)}")
                result["error"] = str(e)
                return result

    async def run(
        self,
        resume: bool = False,
    ) -> ScoutProgress:
        """
        Run the Safe Scout pipeline.

        Args:
            resume: Resume from previous state

        Returns:
            Final progress with statistics
        """
        if self._running:
            raise RuntimeError("Scout is already running")

        self._running = True
        self._cancelled = False
        self._progress = ScoutProgress()
        self._progress.started_at = datetime.now()

        try:
            await self._initialize_components()

            # Get pending topics
            self._update_stage(ScoutStage.FETCHING_TOPICS)
            topics = await self._queue.get_pending_topics(limit=self.config.max_topics)
            self._progress.topics_total = len(topics)

            if not topics:
                logger.info("No pending topics in queue")
                self._update_stage(ScoutStage.COMPLETED)
                return self._progress

            # Create output directory
            output_dir = self.config.work_dir / "scout_data"
            output_dir.mkdir(exist_ok=True)

            # Process topics
            semaphore = asyncio.Semaphore(self.config.url_concurrency)

            for topic in topics:
                if self._cancelled:
                    break

                logger.info(f"Processing topic: {topic.name}")
                await self._queue.mark_processing(topic.topic_id)

                # Get URLs for this topic
                urls = topic.urls[:self.config.max_pages_per_topic]
                self._progress.urls_total += len(urls)

                # Process URLs in parallel
                self._update_stage(ScoutStage.FETCHING_PAGES)
                tasks = [
                    self._process_url(url, topic.name, semaphore)
                    for url in urls
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Save results
                self._update_stage(ScoutStage.SAVING_OUTPUT)
                for result in results:
                    if isinstance(result, Exception):
                        self._progress.errors.append(str(result))
                        continue

                    if result and result.get("pairs"):
                        for pair in result["pairs"]:
                            pair_id = pair.get("pair_id", datetime.now().isoformat())
                            pair_file = output_dir / f"{pair_id}.json"
                            with open(pair_file, "w") as f:
                                json.dump(pair, f, indent=2)

                await self._queue.mark_completed(topic.topic_id)
                self._progress.topics_processed += 1
                self._notify_progress()

            self._update_stage(ScoutStage.COMPLETED)

        except Exception as e:
            self._update_stage(ScoutStage.FAILED)
            self._progress.errors.append(str(e))
            logger.exception("Scout pipeline error")
            raise

        finally:
            self._progress.completed_at = datetime.now()
            await self._cleanup_components()
            self._running = False

        return self._progress

    async def cancel(self) -> None:
        """Cancel running pipeline."""
        self._cancelled = True
        logger.info("Scout cancellation requested")

    async def add_topic(
        self,
        name: str,
        urls: List[str],
        priority: str = "normal",
        category: str = "documentation",
    ) -> str:
        """
        Add a topic to the queue.

        Args:
            name: Topic name
            urls: List of URLs to process
            priority: Priority level
            category: Topic category

        Returns:
            Topic ID
        """
        from reactor_core.scout import (
            TopicQueue,
            TopicQueueConfig,
            LearningTopic,
            TopicPriority,
            TopicCategory,
        )

        priority_map = {
            "critical": TopicPriority.CRITICAL,
            "high": TopicPriority.HIGH,
            "normal": TopicPriority.NORMAL,
            "low": TopicPriority.LOW,
            "background": TopicPriority.BACKGROUND,
        }

        category_map = {
            "documentation": TopicCategory.DOCUMENTATION,
            "tutorial": TopicCategory.TUTORIAL,
            "api_reference": TopicCategory.API_REFERENCE,
            "release_notes": TopicCategory.RELEASE_NOTES,
            "blog": TopicCategory.BLOG,
            "paper": TopicCategory.PAPER,
            "other": TopicCategory.OTHER,
        }

        topic = LearningTopic(
            name=name,
            description=f"Topic: {name}",
            urls=urls,
            priority=priority_map.get(priority, TopicPriority.NORMAL),
            category=category_map.get(category, TopicCategory.OTHER),
        )

        # Initialize queue if needed
        if not self._queue:
            queue_config = TopicQueueConfig(
                db_path=self.config.work_dir / "scout_queue.db"
            )
            self._queue = TopicQueue(queue_config)

        await self._queue.add_topic(topic)
        return topic.topic_id

    async def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            "progress": self._progress.to_dict(),
            "config": self.config.to_dict(),
            "output_dir": str(self.config.work_dir / "scout_data"),
        }


# Convenience exports
__all__ = [
    "SafeScoutOrchestrator",
    "ScoutConfig",
    "ScoutProgress",
    "ScoutStage",
]
