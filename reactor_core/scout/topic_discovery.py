"""
Intelligent Topic Discovery from JARVIS Logs.

This module analyzes JARVIS interaction logs to automatically discover
learning topics that the Scout should explore.

Features:
- Extracts topics from failed/corrected interactions
- Identifies knowledge gaps from user questions
- Discovers trending technical topics
- Generates documentation URLs from topic names
- Prioritizes based on frequency and recency
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DiscoverySource(Enum):
    """Source of discovered topic."""
    FAILED_INTERACTION = "failed_interaction"
    CORRECTION = "correction"
    USER_QUESTION = "user_question"
    UNKNOWN_TERM = "unknown_term"
    TRENDING = "trending"
    MANUAL = "manual"


@dataclass
class DiscoveredTopic:
    """A topic discovered from JARVIS logs."""
    name: str
    description: str
    source: DiscoverySource
    confidence: float  # 0.0 to 1.0
    frequency: int = 1
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    # Generated URLs
    documentation_urls: List[str] = field(default_factory=list)
    tutorial_urls: List[str] = field(default_factory=list)

    # Context
    sample_queries: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)

    # Metadata
    topic_id: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.topic_id:
            self.topic_id = hashlib.md5(
                self.name.lower().encode()
            ).hexdigest()[:12]

    @property
    def priority_score(self) -> float:
        """Calculate priority score based on frequency and recency."""
        # Recency factor (higher for more recent)
        days_old = (datetime.now() - self.last_seen).days
        recency_factor = max(0, 1 - (days_old / 30))  # Decay over 30 days

        # Frequency factor (log scale)
        import math
        freq_factor = min(1.0, math.log(self.frequency + 1) / 5)

        # Source weight
        source_weights = {
            DiscoverySource.CORRECTION: 1.0,
            DiscoverySource.FAILED_INTERACTION: 0.9,
            DiscoverySource.USER_QUESTION: 0.7,
            DiscoverySource.UNKNOWN_TERM: 0.6,
            DiscoverySource.TRENDING: 0.5,
            DiscoverySource.MANUAL: 0.8,
        }
        source_factor = source_weights.get(self.source, 0.5)

        return (
            0.4 * self.confidence +
            0.3 * freq_factor +
            0.2 * recency_factor +
            0.1 * source_factor
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "description": self.description,
            "source": self.source.value,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "documentation_urls": self.documentation_urls,
            "tutorial_urls": self.tutorial_urls,
            "sample_queries": self.sample_queries[:5],
            "related_topics": self.related_topics,
            "tags": self.tags,
            "priority_score": self.priority_score,
        }


@dataclass
class DiscoveryConfig:
    """Configuration for topic discovery."""
    # Analysis settings
    min_frequency: int = 2  # Minimum occurrences to consider
    min_confidence: float = 0.5
    lookback_days: int = 30

    # Topic extraction
    max_topics_per_run: int = 50
    dedupe_similarity_threshold: float = 0.8

    # URL generation
    generate_urls: bool = True
    max_urls_per_topic: int = 5

    # Tech domains for URL generation
    documentation_domains: Dict[str, str] = field(default_factory=lambda: {
        "python": "https://docs.python.org/3/library/{topic}.html",
        "react": "https://react.dev/reference/react/{topic}",
        "typescript": "https://www.typescriptlang.org/docs/handbook/{topic}.html",
        "rust": "https://doc.rust-lang.org/std/{topic}/index.html",
        "go": "https://pkg.go.dev/{topic}",
        "node": "https://nodejs.org/api/{topic}.html",
        "fastapi": "https://fastapi.tiangolo.com/{topic}/",
        "pytorch": "https://pytorch.org/docs/stable/{topic}.html",
        "huggingface": "https://huggingface.co/docs/transformers/main/en/{topic}",
    })


class TopicDiscovery:
    """
    Intelligent topic discovery from JARVIS interaction logs.

    Analyzes logs to identify:
    - Failed interactions (knowledge gaps)
    - User corrections (incorrect responses)
    - Technical terms JARVIS doesn't know
    - Trending topics from conversations
    """

    def __init__(
        self,
        config: Optional[DiscoveryConfig] = None,
    ):
        self.config = config or DiscoveryConfig()
        self._discovered: Dict[str, DiscoveredTopic] = {}
        self._term_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for topic extraction."""
        return {
            # Programming concepts
            "python_module": re.compile(
                r"\b(asyncio|typing|pathlib|dataclass(?:es)?|logging|"
                r"collections|itertools|functools|contextlib|"
                r"multiprocessing|threading|concurrent|subprocess|"
                r"json|pickle|sqlite3|urllib|requests|aiohttp)\b",
                re.IGNORECASE
            ),
            # ML/AI terms
            "ml_term": re.compile(
                r"\b(transformer|attention|embedding|tokeniz\w+|"
                r"fine[- ]?tun\w+|lora|qlora|gguf|llama|gpt|bert|"
                r"diffusion|stable[- ]?diffusion|vae|gan|"
                r"backprop\w*|gradient|optimizer|loss|epoch|batch)\b",
                re.IGNORECASE
            ),
            # Web technologies
            "web_tech": re.compile(
                r"\b(react|vue|angular|svelte|next\.?js|nuxt|"
                r"fastapi|flask|django|express|graphql|rest|"
                r"websocket|http/?\d?|oauth|jwt|cors)\b",
                re.IGNORECASE
            ),
            # DevOps/Cloud
            "devops": re.compile(
                r"\b(docker|kubernetes|k8s|terraform|ansible|"
                r"aws|gcp|azure|cloudrun|lambda|ec2|s3|"
                r"ci/?cd|github[- ]?actions|jenkins)\b",
                re.IGNORECASE
            ),
            # Question patterns
            "how_to": re.compile(
                r"how (?:do|can|to|should) (?:I|we|you) (\w+(?:\s+\w+){0,3})",
                re.IGNORECASE
            ),
            "what_is": re.compile(
                r"what (?:is|are|does) (?:a |an |the )?(\w+(?:\s+\w+){0,2})",
                re.IGNORECASE
            ),
        }

    async def analyze_events(
        self,
        events: List[Dict[str, Any]],
    ) -> List[DiscoveredTopic]:
        """
        Analyze JARVIS events to discover learning topics.

        Args:
            events: List of JARVIS events (from JARVISConnector)

        Returns:
            List of discovered topics
        """
        topics_found: Dict[str, DiscoveredTopic] = {}

        for event in events:
            # Extract relevant fields
            user_input = event.get("user_input", "")
            response = event.get("jarvis_response", "") or event.get("response", "")
            success = event.get("success", True)
            is_correction = event.get("is_correction", False)
            confidence = event.get("confidence", 1.0)
            timestamp = event.get("timestamp")

            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = datetime.now()
            elif timestamp is None:
                timestamp = datetime.now()

            # Analyze based on event type
            if is_correction:
                # Corrections indicate knowledge gaps
                extracted = await self._extract_topics_from_text(
                    user_input,
                    DiscoverySource.CORRECTION,
                    confidence=0.9,
                )
                for topic in extracted:
                    topic.sample_queries.append(user_input[:200])
                    self._merge_topic(topics_found, topic)

            elif not success or confidence < 0.7:
                # Failed interactions
                extracted = await self._extract_topics_from_text(
                    user_input,
                    DiscoverySource.FAILED_INTERACTION,
                    confidence=0.8,
                )
                for topic in extracted:
                    topic.sample_queries.append(user_input[:200])
                    self._merge_topic(topics_found, topic)

            else:
                # Successful interactions - extract for trending
                extracted = await self._extract_topics_from_text(
                    user_input,
                    DiscoverySource.USER_QUESTION,
                    confidence=0.5,
                )
                for topic in extracted:
                    self._merge_topic(topics_found, topic)

        # Filter by minimum frequency and confidence
        filtered = [
            topic for topic in topics_found.values()
            if topic.frequency >= self.config.min_frequency
            and topic.confidence >= self.config.min_confidence
        ]

        # Sort by priority score
        filtered.sort(key=lambda t: t.priority_score, reverse=True)

        # Limit results
        filtered = filtered[:self.config.max_topics_per_run]

        # Generate URLs
        if self.config.generate_urls:
            for topic in filtered:
                await self._generate_urls(topic)

        return filtered

    async def _extract_topics_from_text(
        self,
        text: str,
        source: DiscoverySource,
        confidence: float = 0.7,
    ) -> List[DiscoveredTopic]:
        """Extract topics from text using patterns."""
        topics = []
        seen_names: Set[str] = set()

        for pattern_name, pattern in self._term_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # Normalize topic name
                if isinstance(match, tuple):
                    match = match[0]
                name = match.strip().lower()

                if len(name) < 3 or name in seen_names:
                    continue
                seen_names.add(name)

                # Create topic
                topic = DiscoveredTopic(
                    name=name,
                    description=f"Auto-discovered from {source.value}",
                    source=source,
                    confidence=confidence,
                    tags=[pattern_name],
                )
                topics.append(topic)

        return topics

    def _merge_topic(
        self,
        topics: Dict[str, DiscoveredTopic],
        new_topic: DiscoveredTopic,
    ) -> None:
        """Merge a new topic with existing topics."""
        key = new_topic.name.lower()

        if key in topics:
            existing = topics[key]
            existing.frequency += 1
            existing.last_seen = datetime.now()

            # Increase confidence if seen multiple times
            existing.confidence = min(1.0, existing.confidence + 0.05)

            # Merge sample queries
            for query in new_topic.sample_queries:
                if query not in existing.sample_queries:
                    existing.sample_queries.append(query)

            # Merge tags
            for tag in new_topic.tags:
                if tag not in existing.tags:
                    existing.tags.append(tag)
        else:
            topics[key] = new_topic

    async def _generate_urls(self, topic: DiscoveredTopic) -> None:
        """Generate documentation URLs for a topic."""
        name = topic.name.lower().replace(" ", "-")

        # Determine likely domain based on tags
        if "python_module" in topic.tags:
            template = self.config.documentation_domains.get("python")
            if template:
                topic.documentation_urls.append(
                    template.format(topic=topic.name.lower())
                )

        elif "ml_term" in topic.tags:
            # Try HuggingFace and PyTorch
            for domain in ["huggingface", "pytorch"]:
                template = self.config.documentation_domains.get(domain)
                if template:
                    topic.documentation_urls.append(
                        template.format(topic=name)
                    )

        elif "web_tech" in topic.tags:
            # Add generic MDN reference
            topic.documentation_urls.append(
                f"https://developer.mozilla.org/en-US/docs/Web/{name}"
            )

        # Add generic search fallback
        topic.documentation_urls.append(
            f"https://devdocs.io/#q={name}"
        )

        # Limit URLs
        topic.documentation_urls = topic.documentation_urls[:self.config.max_urls_per_topic]

    async def discover_from_jarvis(
        self,
        jarvis_connector,
        lookback_hours: Optional[int] = None,
    ) -> List[DiscoveredTopic]:
        """
        Discover topics directly from JARVIS connector.

        Args:
            jarvis_connector: JARVISConnector instance
            lookback_hours: Override lookback period

        Returns:
            List of discovered topics
        """
        hours = lookback_hours or (self.config.lookback_days * 24)

        # Get events from connector
        since = datetime.now() - timedelta(hours=hours)
        events = await jarvis_connector.get_events(since=since, limit=5000)

        # Convert to dicts
        event_dicts = [e.to_dict() if hasattr(e, "to_dict") else e for e in events]

        return await self.analyze_events(event_dicts)

    async def create_scout_topics(
        self,
        discovered: List[DiscoveredTopic],
        topic_queue,
    ) -> int:
        """
        Add discovered topics to the Scout queue.

        Args:
            discovered: List of discovered topics
            topic_queue: TopicQueue instance

        Returns:
            Number of topics added
        """
        from reactor_core.scout import LearningTopic, TopicPriority, TopicCategory

        added = 0

        for topic in discovered:
            # Determine priority based on score
            score = topic.priority_score
            if score >= 0.8:
                priority = TopicPriority.HIGH
            elif score >= 0.6:
                priority = TopicPriority.NORMAL
            else:
                priority = TopicPriority.LOW

            # Create learning topic
            learning_topic = LearningTopic(
                name=topic.name.title(),
                description=topic.description,
                urls=topic.documentation_urls,
                priority=priority,
                category=TopicCategory.DOCUMENTATION,
                metadata={
                    "source": topic.source.value,
                    "confidence": topic.confidence,
                    "frequency": topic.frequency,
                    "sample_queries": topic.sample_queries[:3],
                },
            )

            try:
                await topic_queue.add_topic(learning_topic)
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add topic {topic.name}: {e}")

        return added


async def auto_discover_topics(
    jarvis_repo_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    add_to_queue: bool = True,
) -> List[DiscoveredTopic]:
    """
    Convenience function to run auto-discovery.

    Args:
        jarvis_repo_path: Path to JARVIS-AI-Agent repo
        work_dir: Night Shift work directory
        add_to_queue: Whether to add to Scout queue

    Returns:
        List of discovered topics
    """
    from reactor_core.integration import JARVISConnector, JARVISConnectorConfig
    from reactor_core.scout import TopicQueue, TopicQueueConfig

    # Setup paths
    if jarvis_repo_path is None:
        jarvis_repo_path = Path(os.getenv(
            "JARVIS_REPO_PATH",
            Path.home() / "Documents" / "repos" / "JARVIS-AI-Agent"
        ))

    if work_dir is None:
        work_dir = Path(os.getenv(
            "NIGHTSHIFT_WORK_DIR",
            Path.home() / ".jarvis" / "nightshift"
        ))

    # Initialize components
    connector_config = JARVISConnectorConfig(jarvis_repo_path=jarvis_repo_path)
    connector = JARVISConnector(connector_config)

    discovery = TopicDiscovery()

    # Run discovery
    topics = await discovery.discover_from_jarvis(connector)

    logger.info(f"Discovered {len(topics)} topics from JARVIS logs")

    # Add to queue if requested
    if add_to_queue and topics:
        queue_config = TopicQueueConfig(db_path=work_dir / "scout_queue.db")
        queue = TopicQueue(queue_config)

        added = await discovery.create_scout_topics(topics, queue)
        await queue.close()

        logger.info(f"Added {added} topics to Scout queue")

    return topics


# Convenience exports
__all__ = [
    "TopicDiscovery",
    "DiscoveredTopic",
    "DiscoveryConfig",
    "DiscoverySource",
    "auto_discover_topics",
]
