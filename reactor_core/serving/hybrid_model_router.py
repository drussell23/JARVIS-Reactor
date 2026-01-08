"""
Hybrid Model Router - Intelligent Model Selection
==================================================

Routes inference requests to optimal models based on:
- **Task Complexity** - Simple → Local M1, Complex → Cloud GCP
- **Memory Availability** - RAM-aware model selection
- **Model Availability** - Automatic fallback chains
- **Performance Requirements** - Latency vs Quality tradeoffs
- **Cost Optimization** - Prefer local when possible
- **Load Balancing** - Distribute across available models

Version: v83.0 (Unified Model Intelligence)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# COMPLEXITY ANALYSIS
# ============================================================================

class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = 1  # "What is 2+2?"
    SIMPLE = 2  # "Summarize this paragraph"
    MODERATE = 3  # "Write a function to sort a list"
    COMPLEX = 4  # "Design a microservices architecture"
    EXPERT = 5  # "Explain quantum field theory"


@dataclass
class ComplexityScore:
    """Result of complexity analysis."""
    score: float  # 0.0 - 1.0
    level: TaskComplexity
    factors: Dict[str, float]
    recommended_model_size: str  # "small", "medium", "large"
    recommended_backend: Optional[str] = None


class ComplexityAnalyzer:
    """
    Analyzes task complexity using multiple signals.

    Signals:
    - Prompt length
    - Technical vocabulary density
    - Question complexity indicators
    - Domain expertise requirements
    - Code generation indicators
    """

    # Technical terms that indicate complexity
    TECHNICAL_TERMS = {
        "algorithm", "architecture", "implement", "optimize", "design",
        "quantum", "cryptography", "blockchain", "neural", "distributed",
        "microservices", "async", "concurrent", "parallel", "scalable",
        "kubernetes", "docker", "devops", "cicd", "infrastructure",
    }

    # Simple question patterns
    SIMPLE_PATTERNS = [
        r"^what is\s+",
        r"^define\s+",
        r"^who is\s+",
        r"^when did\s+",
        r"^how many\s+",
    ]

    # Complex task patterns
    COMPLEX_PATTERNS = [
        r"implement\s+.*\s+class",
        r"design\s+.*\s+system",
        r"explain.*in\s+detail",
        r"compare.*\s+and\s+contrast",
        r"analyze.*\s+trade-?offs",
    ]

    def analyze(self, prompt: str) -> ComplexityScore:
        """
        Analyze prompt complexity.

        Args:
            prompt: User prompt

        Returns:
            ComplexityScore with detailed analysis
        """
        factors = {}

        # Factor 1: Prompt length
        # Longer prompts often indicate more complex tasks
        length_factor = min(len(prompt) / 500, 1.0)
        factors["length"] = length_factor

        # Factor 2: Technical vocabulary density
        words = set(prompt.lower().split())
        technical_count = sum(1 for term in self.TECHNICAL_TERMS if term in words)
        tech_density = min(technical_count / 5, 1.0)
        factors["technical_density"] = tech_density

        # Factor 3: Question complexity
        prompt_lower = prompt.lower()

        is_simple_question = any(
            re.match(pattern, prompt_lower, re.IGNORECASE)
            for pattern in self.SIMPLE_PATTERNS
        )

        is_complex_task = any(
            re.search(pattern, prompt_lower, re.IGNORECASE)
            for pattern in self.COMPLEX_PATTERNS
        )

        if is_simple_question:
            question_factor = 0.2
        elif is_complex_task:
            question_factor = 0.9
        else:
            question_factor = 0.5

        factors["question_complexity"] = question_factor

        # Factor 4: Code generation indicators
        code_keywords = ["function", "class", "implement", "code", "script", "program"]
        code_count = sum(1 for kw in code_keywords if kw in prompt_lower)
        code_factor = min(code_count / 3, 1.0)
        factors["code_generation"] = code_factor

        # Factor 5: Multi-step reasoning indicators
        multi_step_keywords = ["first", "then", "next", "finally", "step", "process"]
        multi_step_count = sum(1 for kw in multi_step_keywords if kw in prompt_lower)
        multi_step_factor = min(multi_step_count / 4, 1.0)
        factors["multi_step"] = multi_step_factor

        # Compute overall score (weighted average)
        weights = {
            "length": 0.15,
            "technical_density": 0.25,
            "question_complexity": 0.30,
            "code_generation": 0.20,
            "multi_step": 0.10,
        }

        score = sum(factors[k] * weights[k] for k in weights)

        # Determine complexity level
        if score < 0.2:
            level = TaskComplexity.TRIVIAL
            recommended_size = "small"
        elif score < 0.4:
            level = TaskComplexity.SIMPLE
            recommended_size = "small"
        elif score < 0.6:
            level = TaskComplexity.MODERATE
            recommended_size = "medium"
        elif score < 0.8:
            level = TaskComplexity.COMPLEX
            recommended_size = "large"
        else:
            level = TaskComplexity.EXPERT
            recommended_size = "large"

        return ComplexityScore(
            score=score,
            level=level,
            factors=factors,
            recommended_model_size=recommended_size,
        )


# ============================================================================
# ROUTING STRATEGIES
# ============================================================================

class RoutingStrategy(Enum):
    """Model routing strategies."""
    COMPLEXITY_BASED = "complexity"  # Route by task complexity
    LATENCY_OPTIMIZED = "latency"  # Minimize latency
    QUALITY_OPTIMIZED = "quality"  # Maximize quality
    COST_OPTIMIZED = "cost"  # Minimize cost
    BALANCED = "balanced"  # Balance all factors


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    model_id: str
    confidence: float  # 0.0 - 1.0
    reasoning: str
    fallback_chain: List[str]
    estimated_latency_ms: float
    estimated_cost: float
    metadata: Dict[str, any] = field(default_factory=dict)


class HybridModelRouter:
    """
    Intelligent model router with multi-factor decision making.

    Decision factors:
    1. Task complexity → Model capability
    2. Memory availability → Model size
    3. Latency requirements → Model location (local vs cloud)
    4. Cost constraints → Model pricing
    5. Load balancing → Model availability
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        prefer_local: bool = True,
        max_latency_ms: Optional[float] = None,
        max_cost: Optional[float] = None,
    ):
        self.strategy = strategy
        self.prefer_local = prefer_local
        self.max_latency_ms = max_latency_ms
        self.max_cost = max_cost

        self.complexity_analyzer = ComplexityAnalyzer()

        # Model characteristics (would be loaded from config)
        self.model_characteristics = {}

    async def route(
        self,
        prompt: str,
        available_models: Dict[str, any],  # model_id -> ModelMetadata
        constraints: Optional[Dict[str, any]] = None,
    ) -> RoutingDecision:
        """
        Route request to optimal model.

        Args:
            prompt: User prompt
            available_models: Available models with metadata
            constraints: Optional constraints (latency, cost, etc.)

        Returns:
            RoutingDecision
        """
        # Analyze complexity
        complexity = self.complexity_analyzer.analyze(prompt)

        # Score each model
        model_scores = {}
        for model_id, metadata in available_models.items():
            score = await self._score_model(
                model_id=model_id,
                metadata=metadata,
                complexity=complexity,
                constraints=constraints or {},
            )
            model_scores[model_id] = score

        # Select best model
        best_model_id = max(model_scores, key=lambda k: model_scores[k]["total_score"])
        best_score = model_scores[best_model_id]

        # Create fallback chain (other models sorted by score)
        fallback_chain = sorted(
            model_scores.keys(),
            key=lambda k: model_scores[k]["total_score"],
            reverse=True,
        )[1:]  # Exclude best model

        # Create decision
        decision = RoutingDecision(
            model_id=best_model_id,
            confidence=best_score["total_score"],
            reasoning=best_score["reasoning"],
            fallback_chain=fallback_chain[:3],  # Top 3 fallbacks
            estimated_latency_ms=best_score.get("estimated_latency_ms", 0),
            estimated_cost=best_score.get("estimated_cost", 0),
            metadata={
                "complexity_score": complexity.score,
                "complexity_level": complexity.level.name,
                "all_scores": model_scores,
            },
        )

        logger.info(
            f"Routed to {best_model_id} (complexity: {complexity.level.name}, "
            f"confidence: {decision.confidence:.2f})"
        )

        return decision

    async def _score_model(
        self,
        model_id: str,
        metadata: any,
        complexity: ComplexityScore,
        constraints: Dict[str, any],
    ) -> Dict[str, any]:
        """
        Score a model for the given task.

        Returns dict with:
        - total_score: Overall score (0.0 - 1.0)
        - reasoning: Human-readable explanation
        - component_scores: Individual factor scores
        """
        scores = {}

        # Factor 1: Capability match (does model have sufficient capacity?)
        capability_score = self._score_capability(metadata, complexity)
        scores["capability"] = capability_score

        # Factor 2: Latency (local models faster than cloud)
        latency_score = self._score_latency(metadata, constraints)
        scores["latency"] = latency_score

        # Factor 3: Cost (local free, cloud paid)
        cost_score = self._score_cost(metadata, constraints)
        scores["cost"] = cost_score

        # Factor 4: Availability (is model loaded/healthy?)
        availability_score = await self._score_availability(model_id, metadata)
        scores["availability"] = availability_score

        # Factor 5: Memory (can we load this model?)
        memory_score = self._score_memory(metadata)
        scores["memory"] = memory_score

        # Weighted combination based on strategy
        weights = self._get_strategy_weights()
        total_score = sum(scores[k] * weights.get(k, 1.0) for k in scores) / sum(weights.values())

        # Generate reasoning
        reasoning = self._generate_reasoning(model_id, metadata, scores, complexity)

        return {
            "total_score": total_score,
            "component_scores": scores,
            "reasoning": reasoning,
            "estimated_latency_ms": self._estimate_latency(metadata),
            "estimated_cost": self._estimate_cost(metadata, complexity),
        }

    def _score_capability(self, metadata: any, complexity: ComplexityScore) -> float:
        """Score model capability vs task complexity."""
        # Get model size category
        if not hasattr(metadata, 'parameters'):
            return 0.5  # Unknown, assume medium

        params = metadata.parameters

        # Categorize model
        if params < 1e9:  # < 1B
            model_tier = 1
        elif params < 7e9:  # < 7B
            model_tier = 2
        elif params < 30e9:  # < 30B
            model_tier = 3
        else:  # >= 30B
            model_tier = 4

        # Match to complexity
        complexity_tier = complexity.level.value

        # Perfect match = 1.0, close = 0.7, far = 0.3
        diff = abs(model_tier - complexity_tier)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.7
        elif diff == 2:
            return 0.5
        else:
            return 0.3

    def _score_latency(self, metadata: any, constraints: Dict[str, any]) -> float:
        """Score latency characteristics."""
        # Local models (GGUF, MLX) = fast
        # Cloud models (Transformers on GCP) = slower

        if hasattr(metadata, 'backend'):
            backend = metadata.backend.value
            if backend in ["gguf", "mlx", "llamacpp"]:
                latency_score = 1.0  # Fast local inference
            elif backend in ["transformers"]:
                # Check if local or cloud
                if hasattr(metadata, 'tags') and "cloud" in metadata.tags:
                    latency_score = 0.5  # Slower cloud
                else:
                    latency_score = 0.8  # Local transformers
            else:
                latency_score = 0.7
        else:
            latency_score = 0.5

        # Apply latency constraint penalty
        if self.max_latency_ms:
            estimated = self._estimate_latency(metadata)
            if estimated > self.max_latency_ms:
                latency_score *= 0.5  # Penalty for exceeding limit

        return latency_score

    def _score_cost(self, metadata: any, constraints: Dict[str, any]) -> float:
        """Score cost characteristics."""
        # Local = free (1.0)
        # Cloud = paid (0.3-0.7 depending on size)

        if hasattr(metadata, 'tags') and "cloud" in metadata.tags:
            # Cloud model - estimate cost
            if hasattr(metadata, 'parameters'):
                # Larger models more expensive
                if metadata.parameters > 70e9:
                    return 0.3
                elif metadata.parameters > 30e9:
                    return 0.5
                else:
                    return 0.7
            return 0.5
        else:
            # Local model - free
            return 1.0

    async def _score_availability(self, model_id: str, metadata: any) -> float:
        """Score model availability."""
        # In real implementation, would check:
        # - Is model currently loaded?
        # - Is model healthy?
        # - What's current load?

        # For now, assume all models available
        return 1.0

    def _score_memory(self, metadata: any) -> float:
        """Score memory feasibility."""
        import psutil

        # Get available system memory
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

        # Get model memory requirement
        if hasattr(metadata, 'optimal_memory_gb'):
            required = metadata.optimal_memory_gb
        elif hasattr(metadata, 'min_memory_gb'):
            required = metadata.min_memory_gb
        else:
            return 0.5  # Unknown

        # Score based on headroom
        headroom_ratio = available_memory_gb / required

        if headroom_ratio >= 2.0:
            return 1.0  # Plenty of memory
        elif headroom_ratio >= 1.5:
            return 0.8
        elif headroom_ratio >= 1.2:
            return 0.6
        elif headroom_ratio >= 1.0:
            return 0.4
        else:
            return 0.1  # Not enough memory

    def _get_strategy_weights(self) -> Dict[str, float]:
        """Get scoring weights for current strategy."""
        if self.strategy == RoutingStrategy.COMPLEXITY_BASED:
            return {
                "capability": 5.0,
                "latency": 1.0,
                "cost": 1.0,
                "availability": 2.0,
                "memory": 2.0,
            }
        elif self.strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            return {
                "capability": 2.0,
                "latency": 5.0,
                "cost": 0.5,
                "availability": 2.0,
                "memory": 1.0,
            }
        elif self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return {
                "capability": 5.0,
                "latency": 0.5,
                "cost": 0.5,
                "availability": 2.0,
                "memory": 2.0,
            }
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return {
                "capability": 2.0,
                "latency": 1.0,
                "cost": 5.0,
                "availability": 2.0,
                "memory": 1.0,
            }
        else:  # BALANCED
            return {
                "capability": 3.0,
                "latency": 2.0,
                "cost": 2.0,
                "availability": 2.0,
                "memory": 2.0,
            }

    def _estimate_latency(self, metadata: any) -> float:
        """Estimate latency in milliseconds."""
        # Rough estimates based on backend and size
        if not hasattr(metadata, 'backend'):
            return 1000.0

        backend = metadata.backend.value

        if backend in ["gguf", "mlx"]:
            base_latency = 100  # Fast local
        elif backend == "transformers":
            base_latency = 500  # Slower
        else:
            base_latency = 300

        # Adjust for model size
        if hasattr(metadata, 'parameters'):
            if metadata.parameters > 70e9:
                base_latency *= 3
            elif metadata.parameters > 30e9:
                base_latency *= 2
            elif metadata.parameters > 7e9:
                base_latency *= 1.5

        return base_latency

    def _estimate_cost(self, metadata: any, complexity: ComplexityScore) -> float:
        """Estimate cost in dollars."""
        # Local = $0
        # Cloud = based on tokens

        if hasattr(metadata, 'tags') and "cloud" in metadata.tags:
            # Estimate tokens (rough)
            estimated_tokens = 500

            # Cost per million tokens (rough estimates)
            if hasattr(metadata, 'parameters') and metadata.parameters > 70e9:
                cost_per_million = 5.0
            else:
                cost_per_million = 1.0

            return (estimated_tokens / 1_000_000) * cost_per_million
        else:
            return 0.0

    def _generate_reasoning(
        self,
        model_id: str,
        metadata: any,
        scores: Dict[str, float],
        complexity: ComplexityScore,
    ) -> str:
        """Generate human-readable reasoning for decision."""
        reasons = []

        # Capability
        if scores["capability"] > 0.8:
            reasons.append(f"excellent capability match for {complexity.level.name} task")
        elif scores["capability"] < 0.4:
            reasons.append(f"limited capability for {complexity.level.name} task")

        # Latency
        if scores["latency"] > 0.8:
            reasons.append("low latency (local inference)")
        elif scores["latency"] < 0.5:
            reasons.append("higher latency (cloud inference)")

        # Cost
        if scores["cost"] > 0.9:
            reasons.append("zero cost (local)")
        elif scores["cost"] < 0.5:
            reasons.append("higher cost (cloud)")

        # Memory
        if scores["memory"] < 0.5:
            reasons.append("tight memory constraints")

        reasoning = f"Selected {model_id}: " + ", ".join(reasons)
        return reasoning


__all__ = [
    # Enums
    "TaskComplexity",
    "RoutingStrategy",
    # Data structures
    "ComplexityScore",
    "RoutingDecision",
    # Core classes
    "ComplexityAnalyzer",
    "HybridModelRouter",
]
