"""
DPO Preference Pair Generator — v242.0
=======================================

Automatically generates Direct Preference Optimization (DPO) training pairs
from multi-model telemetry data. When the same query is answered by different
models (e.g., Mistral-7B gives x=11, Qwen-Math gives x=3), this module
creates {prompt, chosen, rejected} triples WITHOUT human labeling.

ARCHITECTURE:
    ~/.jarvis/telemetry/*.jsonl → Ingest → Group by Query → Compare → DPO Pairs
                                                                         ↓
                                                          UnifiedTrainingPipeline

SCORING STRATEGY:
    1. Outcome signal (strongest): success > failure
    2. Confidence score (medium): higher confidence = better
    3. Model specialization (tiebreaker): specialist > generalist for matching task_type
    4. Latency (weak): faster is slightly preferred (user experience)

USAGE:
    generator = DPOPairGenerator()
    pairs = await generator.generate_from_telemetry(
        telemetry_dir=Path("~/.jarvis/telemetry"),
        min_pairs=50,
    )
    # pairs = [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]

CROSS-REPO INTEGRATION:
    - Reads canonical ExperienceEvent format from ~/.jarvis/telemetry/
    - Writes DPO pairs to ~/.jarvis/training/dpo_pairs/
    - Feeds into UnifiedTrainingPipeline.add_experiences() with preference format
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class DPOConfig:
    """Configuration for DPO pair generation."""

    # Input
    telemetry_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "DPO_TELEMETRY_DIR",
            str(Path.home() / ".jarvis" / "telemetry"),
        ))
    )

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv(
            "DPO_OUTPUT_DIR",
            str(Path.home() / ".jarvis" / "training" / "dpo_pairs"),
        ))
    )

    # Matching
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("DPO_SIMILARITY_THRESHOLD", 0.95)
    )
    max_time_window_hours: int = field(
        default_factory=lambda: _env_int("DPO_TIME_WINDOW_HOURS", 168)  # 7 days
    )

    # Scoring weights (must sum to 1.0)
    weight_outcome: float = 0.50     # success vs failure
    weight_confidence: float = 0.25   # model confidence score
    weight_specialist: float = 0.15   # specialist model bonus
    weight_latency: float = 0.10      # faster = slightly better

    # Thresholds
    min_score_difference: float = field(
        default_factory=lambda: _env_float("DPO_MIN_SCORE_DIFF", 0.1)
    )
    min_response_length: int = field(
        default_factory=lambda: _env_int("DPO_MIN_RESPONSE_LENGTH", 5)
    )
    max_pairs_per_prompt: int = field(
        default_factory=lambda: _env_int("DPO_MAX_PAIRS_PER_PROMPT", 3)
    )

    # Model specialization mapping (task_type → preferred model pattern)
    # Used for the specialist tiebreaker signal
    specialist_models: Dict[str, str] = field(default_factory=lambda: {
        "math_simple": "qwen",
        "math_complex": "math",
        "code_simple": "coder",
        "code_complex": "coder",
        "code_review": "coder",
        "code_explain": "coder",
        "code_debug": "coder",
        "reason_complex": "deepseek",
        "analyze": "deepseek",
        "creative_write": "llama",
        "creative_brainstorm": "llama",
        "summarize": "llama",
        "greeting": "phi",
        "simple_chat": "phi",
    })


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ResponseCandidate:
    """A single model response to a prompt."""
    response: str
    model_id: Optional[str]
    confidence: float
    outcome: str       # "success", "failure", "unknown"
    latency_ms: float
    task_type: Optional[str]
    timestamp: str
    event_id: str


@dataclass
class DPOPair:
    """A single DPO preference pair."""
    prompt: str
    chosen: str
    rejected: str
    chosen_model: Optional[str]
    rejected_model: Optional[str]
    chosen_score: float
    rejected_score: float
    task_type: Optional[str]
    generation_method: str   # "cross_model", "outcome_diff", "confidence_diff"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_model": self.chosen_model,
            "rejected_model": self.rejected_model,
            "chosen_score": round(self.chosen_score, 4),
            "rejected_score": round(self.rejected_score, 4),
            "task_type": self.task_type,
            "generation_method": self.generation_method,
            "metadata": self.metadata,
        }


# =============================================================================
# DPO Pair Generator
# =============================================================================

class DPOPairGenerator:
    """
    Generates DPO preference pairs from multi-model telemetry data.

    The key insight: when JARVIS routes the same query to different specialist
    models (via GCPModelSwapCoordinator), the responses naturally create
    preference pairs without any human labeling.

    Example:
        Query: "solve 5x+3=18"
        Model A (Mistral-7B):     "x = 11"  (wrong, success=false)
        Model B (Qwen-Math-7B):   "x = 3"   (correct, success=true)
        → DPO pair: chosen="x = 3", rejected="x = 11"
    """

    def __init__(self, config: Optional[DPOConfig] = None):
        self._config = config or DPOConfig()
        self._generated_pairs: List[DPOPair] = []
        self._seen_prompts: set = set()  # Dedup

    async def generate_from_telemetry(
        self,
        telemetry_dir: Optional[Path] = None,
        since: Optional[datetime] = None,
        min_pairs: int = 0,
    ) -> List[DPOPair]:
        """
        Generate DPO pairs from telemetry JSONL files.

        Args:
            telemetry_dir: Override telemetry directory
            since: Only consider events after this timestamp
            min_pairs: Minimum pairs required (returns empty if threshold not met)

        Returns:
            List of DPOPair objects
        """
        telemetry_dir = telemetry_dir or self._config.telemetry_dir

        if not telemetry_dir.exists():
            logger.warning(f"[DPO] Telemetry dir not found: {telemetry_dir}")
            return []

        # 1. Ingest all interaction events
        interactions = await self._ingest_telemetry(telemetry_dir, since)
        logger.info(f"[DPO] Ingested {len(interactions)} interactions from {telemetry_dir}")

        if len(interactions) < 2:
            return []

        # 2. Group by normalized prompt (semantic near-duplicates)
        prompt_groups = self._group_by_prompt(interactions)
        logger.info(f"[DPO] Grouped into {len(prompt_groups)} unique prompts")

        # 3. Generate pairs from groups with 2+ responses
        pairs = []
        for prompt_key, candidates in prompt_groups.items():
            if len(candidates) < 2:
                continue
            group_pairs = self._generate_pairs_from_group(prompt_key, candidates)
            pairs.extend(group_pairs)

        # 4. Deduplicate
        pairs = self._deduplicate_pairs(pairs)

        logger.info(f"[DPO] Generated {len(pairs)} DPO pairs from {len(interactions)} interactions")

        if len(pairs) < min_pairs:
            logger.info(f"[DPO] Below minimum threshold ({len(pairs)} < {min_pairs})")
            return []

        self._generated_pairs.extend(pairs)
        return pairs

    async def _ingest_telemetry(
        self,
        telemetry_dir: Path,
        since: Optional[datetime] = None,
    ) -> List[ResponseCandidate]:
        """Read JSONL files and extract interaction events."""
        interactions = []
        cutoff = since or (datetime.now() - timedelta(hours=self._config.max_time_window_hours))

        for jsonl_file in sorted(telemetry_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Extract fields using canonical names with fallbacks
                        user_input = (
                            data.get("user_input")
                            or data.get("input")
                            or data.get("query")
                            or data.get("prompt")
                            or ""
                        )
                        assistant_output = (
                            data.get("assistant_output")
                            or data.get("response")
                            or data.get("output")
                            or data.get("completion")
                            or ""
                        )

                        if not user_input or not assistant_output:
                            continue

                        # Skip short responses
                        if len(assistant_output) < self._config.min_response_length:
                            continue

                        # Extract metadata
                        metadata = data.get("metadata", {})
                        model_id = (
                            data.get("model_id")
                            or metadata.get("model_id")
                            or data.get("model")
                        )
                        task_type = (
                            data.get("task_type")
                            or metadata.get("task_type")
                        )
                        confidence = float(data.get("confidence", 1.0))
                        if confidence > 1.0:
                            confidence = confidence / 100.0

                        outcome = data.get("outcome", "unknown")
                        if isinstance(data.get("success"), bool):
                            outcome = "success" if data["success"] else "failure"

                        latency_ms = float(data.get("latency_ms", data.get("latency", 0.0)))
                        timestamp = data.get("timestamp", "")

                        interactions.append(ResponseCandidate(
                            response=assistant_output,
                            model_id=model_id,
                            confidence=confidence,
                            outcome=outcome,
                            latency_ms=latency_ms,
                            task_type=task_type,
                            timestamp=timestamp,
                            event_id=data.get("event_id", ""),
                        ))
                        # Store prompt on the candidate for grouping
                        interactions[-1]._prompt = user_input  # type: ignore

            except Exception as e:
                logger.warning(f"[DPO] Error reading {jsonl_file}: {e}")

        return interactions

    def _group_by_prompt(
        self,
        interactions: List[ResponseCandidate],
    ) -> Dict[str, List[ResponseCandidate]]:
        """Group interactions by normalized prompt text."""
        groups: Dict[str, List[ResponseCandidate]] = defaultdict(list)

        for candidate in interactions:
            prompt = getattr(candidate, '_prompt', '')
            # Normalize: lowercase, strip whitespace, collapse spaces
            normalized = ' '.join(prompt.lower().split())
            # Hash for consistent grouping
            prompt_key = hashlib.md5(normalized.encode()).hexdigest()[:16]
            # Store original prompt for output
            if prompt_key not in groups:
                groups[prompt_key] = []
                groups[prompt_key]._original_prompt = prompt  # type: ignore
            groups[prompt_key].append(candidate)

        return groups

    def _score_candidate(self, candidate: ResponseCandidate) -> float:
        """
        Score a response candidate for preference ranking.

        Higher score = better response. Combines:
        1. Outcome (50%): success=1.0, unknown=0.5, failure=0.0
        2. Confidence (25%): raw confidence score
        3. Specialist bonus (15%): 1.0 if model matches task specialist
        4. Latency penalty (10%): normalized, lower is better
        """
        # 1. Outcome signal
        outcome_map = {"success": 1.0, "partial": 0.5, "unknown": 0.5, "failure": 0.0}
        outcome_score = outcome_map.get(candidate.outcome, 0.5)

        # 2. Confidence
        confidence_score = max(0.0, min(1.0, candidate.confidence))

        # 3. Specialist match
        specialist_score = 0.5  # neutral default
        if candidate.task_type and candidate.model_id:
            preferred_pattern = self._config.specialist_models.get(candidate.task_type)
            if preferred_pattern:
                if preferred_pattern.lower() in (candidate.model_id or "").lower():
                    specialist_score = 1.0
                else:
                    specialist_score = 0.3

        # 4. Latency (inverse: lower latency = higher score)
        # Normalize: 0ms = 1.0, 30000ms = 0.0
        max_latency = 30000.0
        latency_score = max(0.0, 1.0 - (candidate.latency_ms / max_latency))

        # Weighted sum
        total = (
            self._config.weight_outcome * outcome_score
            + self._config.weight_confidence * confidence_score
            + self._config.weight_specialist * specialist_score
            + self._config.weight_latency * latency_score
        )

        return total

    def _generate_pairs_from_group(
        self,
        prompt_key: str,
        candidates: List[ResponseCandidate],
    ) -> List[DPOPair]:
        """Generate DPO pairs from a group of responses to the same prompt."""
        # Score all candidates
        scored = [(self._score_candidate(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)  # Best first

        original_prompt = getattr(candidates, '_original_prompt', '')
        if not original_prompt:
            original_prompt = getattr(candidates[0], '_prompt', '')

        pairs = []
        seen_pairs = set()

        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                if len(pairs) >= self._config.max_pairs_per_prompt:
                    break

                better_score, better = scored[i]
                worse_score, worse = scored[j]

                # Require minimum score difference
                if (better_score - worse_score) < self._config.min_score_difference:
                    continue

                # Skip if same model (no preference signal)
                if better.model_id and better.model_id == worse.model_id:
                    continue

                # Skip duplicate response text
                if better.response.strip() == worse.response.strip():
                    continue

                pair_key = f"{hash(better.response)}_{hash(worse.response)}"
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Determine generation method
                if better.outcome != worse.outcome:
                    method = "outcome_diff"
                elif better.model_id != worse.model_id:
                    method = "cross_model"
                else:
                    method = "confidence_diff"

                pairs.append(DPOPair(
                    prompt=original_prompt,
                    chosen=better.response,
                    rejected=worse.response,
                    chosen_model=better.model_id,
                    rejected_model=worse.model_id,
                    chosen_score=better_score,
                    rejected_score=worse_score,
                    task_type=better.task_type or worse.task_type,
                    generation_method=method,
                    metadata={
                        "prompt_key": prompt_key,
                        "score_difference": round(better_score - worse_score, 4),
                        "chosen_confidence": better.confidence,
                        "rejected_confidence": worse.confidence,
                        "chosen_outcome": better.outcome,
                        "rejected_outcome": worse.outcome,
                    },
                ))

        return pairs

    def _deduplicate_pairs(self, pairs: List[DPOPair]) -> List[DPOPair]:
        """Remove duplicate pairs based on content hash."""
        seen = set()
        unique = []
        for pair in pairs:
            key = hashlib.md5(
                f"{pair.prompt}|{pair.chosen}|{pair.rejected}".encode()
            ).hexdigest()
            if key not in seen:
                seen.add(key)
                unique.append(pair)
        return unique

    async def export_pairs(
        self,
        pairs: Optional[List[DPOPair]] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Export DPO pairs to JSONL file for training.

        Returns:
            Path to the exported JSONL file.
        """
        pairs = pairs or self._generated_pairs
        output_dir = output_dir or self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"dpo_pairs_{timestamp}.jsonl"

        with open(output_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict()) + "\n")

        logger.info(f"[DPO] Exported {len(pairs)} pairs to {output_file}")
        return output_file

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        method_counts: Dict[str, int] = defaultdict(int)
        model_pairs: Dict[str, int] = defaultdict(int)

        for pair in self._generated_pairs:
            method_counts[pair.generation_method] += 1
            key = f"{pair.chosen_model}>{pair.rejected_model}"
            model_pairs[key] += 1

        return {
            "total_pairs": len(self._generated_pairs),
            "by_method": dict(method_counts),
            "by_model_pair": dict(model_pairs),
            "avg_score_diff": (
                sum(p.chosen_score - p.rejected_score for p in self._generated_pairs)
                / max(len(self._generated_pairs), 1)
            ),
        }
