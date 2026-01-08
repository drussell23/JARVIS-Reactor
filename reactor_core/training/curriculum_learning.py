"""
Curriculum Learning for JARVIS Reactor.

Implements adaptive curriculum learning strategies:
- Difficulty-based progression
- Performance-based pacing
- Multi-stage curricula
- Dynamic difficulty adjustment
- Automated difficulty scoring

Based on research:
- "Curriculum Learning" (Bengio et al., 2009)
- "Automated Curriculum Learning" (Graves et al., 2017)
- "Teacher-Student Curriculum Learning" (Matiisen et al., 2017)
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger(__name__)


# =============================================================================
# DIFFICULTY SCORING
# =============================================================================

class DifficultyMetric(Enum):
    """Metrics for measuring sample difficulty."""

    LOSS = auto()              # Training loss
    CONFIDENCE = auto()        # Model confidence
    UNCERTAINTY = auto()       # Prediction uncertainty
    PERPLEXITY = auto()        # Language model perplexity
    LENGTH = auto()            # Sequence length
    RARITY = auto()            # Feature rarity
    CUSTOM = auto()            # User-defined metric


@dataclass
class DifficultyScore:
    """
    Difficulty score for a training sample.

    Attributes:
        value: Difficulty value (0.0 = easiest, 1.0 = hardest)
        metric: Metric used for scoring
        confidence: Confidence in the score
        metadata: Additional metadata
    """

    value: float
    metric: DifficultyMetric
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate score."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Difficulty score must be in [0, 1], got {self.value}")


class DifficultyScorer(Protocol):
    """Protocol for difficulty scoring functions."""

    def score(self, sample: Any, model: Optional[nn.Module] = None) -> DifficultyScore:
        """
        Score the difficulty of a sample.

        Args:
            sample: Training sample
            model: Model for model-based scoring (optional)

        Returns:
            DifficultyScore
        """
        ...


class LossDifficultyScorer:
    """Score difficulty based on model loss."""

    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn

    def score(self, sample: Any, model: Optional[nn.Module] = None) -> DifficultyScore:
        """Score based on model loss."""
        if model is None:
            raise ValueError("Model required for loss-based difficulty scoring")

        model.eval()
        with torch.no_grad():
            # Forward pass
            if isinstance(sample, (tuple, list)):
                inputs, targets = sample
            else:
                inputs = sample
                targets = sample  # For autoencoding tasks

            outputs = model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Normalize loss to [0, 1] (higher loss = harder)
            difficulty = min(1.0, loss.item() / 10.0)  # Assume loss < 10 is typical

        return DifficultyScore(
            value=difficulty,
            metric=DifficultyMetric.LOSS,
            confidence=1.0,
            metadata={"loss": loss.item()},
        )


class LengthDifficultyScorer:
    """Score difficulty based on sequence length."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def score(self, sample: Any, model: Optional[nn.Module] = None) -> DifficultyScore:
        """Score based on sequence length."""
        # Extract length from sample
        if hasattr(sample, "__len__"):
            length = len(sample)
        elif isinstance(sample, (tuple, list)):
            length = len(sample[0]) if hasattr(sample[0], "__len__") else 1
        else:
            length = 1

        # Normalize to [0, 1]
        difficulty = min(1.0, length / self.max_length)

        return DifficultyScore(
            value=difficulty,
            metric=DifficultyMetric.LENGTH,
            confidence=1.0,
            metadata={"length": length},
        )


# =============================================================================
# CURRICULUM STRATEGIES
# =============================================================================

class CurriculumStrategy(Enum):
    """Curriculum learning strategies."""

    FIXED = auto()              # Fixed progression schedule
    ADAPTIVE = auto()           # Adapt based on performance
    SELF_PACED = auto()         # Model selects its own pace
    TEACHER_STUDENT = auto()    # Teacher guides student
    COMPETENCE = auto()          # Progress based on competence
    MIXTURE = auto()            # Mixture of strategies


@dataclass
class CurriculumStage:
    """
    A stage in the curriculum.

    Attributes:
        name: Stage name
        difficulty_range: (min_difficulty, max_difficulty)
        num_epochs: Number of epochs for this stage
        sampling_strategy: How to sample within difficulty range
        success_threshold: Success rate to advance to next stage
    """

    name: str
    difficulty_range: Tuple[float, float]
    num_epochs: int = 1
    sampling_strategy: str = "uniform"  # "uniform", "weighted", "hard"
    success_threshold: float = 0.7

    def __post_init__(self):
        """Validate stage."""
        min_diff, max_diff = self.difficulty_range
        if not (0.0 <= min_diff <= max_diff <= 1.0):
            raise ValueError(
                f"Invalid difficulty range: {self.difficulty_range}"
            )


@dataclass
class CurriculumConfig:
    """
    Configuration for curriculum learning.

    Attributes:
        strategy: Curriculum strategy
        stages: List of curriculum stages
        difficulty_scorer: Function to score difficulty
        patience: Epochs to wait before advancing
        min_improvement: Minimum improvement to advance
        auto_adjust: Automatically adjust stages based on performance
    """

    strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE
    stages: List[CurriculumStage] = field(default_factory=list)
    difficulty_scorer: Optional[DifficultyScorer] = None
    patience: int = 3
    min_improvement: float = 0.01
    auto_adjust: bool = True


# =============================================================================
# CURRICULUM SAMPLER
# =============================================================================

class CurriculumSampler(Sampler):
    """
    Sampler that implements curriculum learning.

    Samples are ordered/filtered based on difficulty and current curriculum stage.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: Dict[int, DifficultyScore],
        current_stage: CurriculumStage,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.current_stage = current_stage
        self.shuffle = shuffle

        # Filter samples by difficulty range
        min_diff, max_diff = current_stage.difficulty_range
        self.valid_indices = [
            idx
            for idx, score in difficulty_scores.items()
            if min_diff <= score.value <= max_diff
        ]

        if not self.valid_indices:
            logger.warning(
                f"No samples in difficulty range {current_stage.difficulty_range}"
            )
            self.valid_indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[int]:
        """Iterate over valid sample indices."""
        indices = self.valid_indices.copy()

        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)

        # Apply sampling strategy
        if self.current_stage.sampling_strategy == "hard":
            # Prefer harder samples within range
            indices.sort(
                key=lambda idx: self.difficulty_scores.get(
                    idx, DifficultyScore(0.5, DifficultyMetric.CUSTOM)
                ).value,
                reverse=True,
            )

        return iter(indices)

    def __len__(self) -> int:
        """Number of valid samples."""
        return len(self.valid_indices)


# =============================================================================
# CURRICULUM LEARNER
# =============================================================================

class CurriculumLearner:
    """
    Main curriculum learning coordinator.

    Manages:
    - Difficulty scoring of all samples
    - Stage progression
    - Adaptive difficulty adjustment
    - Performance tracking
    - Curriculum scheduling

    Example:
        >>> config = CurriculumConfig(
        ...     strategy=CurriculumStrategy.ADAPTIVE,
        ...     stages=[
        ...         CurriculumStage("easy", (0.0, 0.3), num_epochs=2),
        ...         CurriculumStage("medium", (0.3, 0.7), num_epochs=3),
        ...         CurriculumStage("hard", (0.7, 1.0), num_epochs=5),
        ...     ],
        ...     difficulty_scorer=LossDifficultyScorer(nn.CrossEntropyLoss()),
        ... )
        >>>
        >>> learner = CurriculumLearner(config, model, dataset)
        >>> learner.score_all_samples()  # Score difficulty of all samples
        >>>
        >>> # Train with curriculum
        >>> for stage in learner.stages:
        ...     dataloader = learner.get_dataloader(stage)
        ...     for epoch in range(stage.num_epochs):
        ...         train_one_epoch(model, dataloader)
        ...         learner.update_performance(epoch_loss, epoch_acc)
        ...
        ...     if learner.should_advance():
        ...         learner.advance_stage()
    """

    def __init__(
        self,
        config: CurriculumConfig,
        model: nn.Module,
        dataset: Dataset,
        batch_size: int = 32,
    ):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

        # Difficulty scores for each sample
        self.difficulty_scores: Dict[int, DifficultyScore] = {}

        # Current stage tracking
        self.current_stage_idx = 0
        self.current_epoch = 0

        # Performance tracking
        self.stage_performance: Dict[str, List[float]] = defaultdict(list)
        self.best_performance: float = 0.0
        self.epochs_without_improvement: int = 0

        # History
        self.history: Dict[str, List[Any]] = defaultdict(list)

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        if not self.config.stages:
            # Default single stage
            return CurriculumStage(
                name="all",
                difficulty_range=(0.0, 1.0),
                num_epochs=10,
            )

        return self.config.stages[self.current_stage_idx]

    def score_all_samples(self) -> None:
        """
        Score difficulty of all samples in dataset.

        This should be called once before training starts.
        """
        logger.info(f"Scoring difficulty of {len(self.dataset)} samples...")

        scorer = self.config.difficulty_scorer or LengthDifficultyScorer()

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            score = scorer.score(sample, self.model)
            self.difficulty_scores[idx] = score

            if (idx + 1) % 1000 == 0:
                logger.debug(f"Scored {idx + 1}/{len(self.dataset)} samples")

        # Log statistics
        scores = [s.value for s in self.difficulty_scores.values()]
        logger.info(
            f"Difficulty scores: "
            f"min={min(scores):.3f}, "
            f"max={max(scores):.3f}, "
            f"mean={sum(scores)/len(scores):.3f}"
        )

    def get_dataloader(
        self,
        stage: Optional[CurriculumStage] = None,
        **dataloader_kwargs,
    ) -> DataLoader:
        """
        Get dataloader for current curriculum stage.

        Args:
            stage: Curriculum stage (uses current stage if None)
            **dataloader_kwargs: Additional arguments for DataLoader

        Returns:
            DataLoader with curriculum sampler
        """
        stage = stage or self.current_stage

        # Create curriculum sampler
        sampler = CurriculumSampler(
            dataset=self.dataset,
            difficulty_scores=self.difficulty_scores,
            current_stage=stage,
            shuffle=True,
        )

        # Create dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            **dataloader_kwargs,
        )

        logger.info(
            f"Created dataloader for stage '{stage.name}' "
            f"with {len(sampler)} samples "
            f"(difficulty {stage.difficulty_range[0]:.2f}-{stage.difficulty_range[1]:.2f})"
        )

        return dataloader

    def update_performance(
        self,
        loss: float,
        accuracy: Optional[float] = None,
    ) -> None:
        """
        Update performance metrics after an epoch.

        Args:
            loss: Training/validation loss
            accuracy: Accuracy metric (optional)
        """
        stage_name = self.current_stage.name

        # Record performance
        self.stage_performance[f"{stage_name}_loss"].append(loss)
        if accuracy is not None:
            self.stage_performance[f"{stage_name}_accuracy"].append(accuracy)

        # Track best performance
        metric = accuracy if accuracy is not None else -loss
        if metric > self.best_performance + self.config.min_improvement:
            self.best_performance = metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Log
        logger.info(
            f"Stage '{stage_name}' epoch {self.current_epoch}: "
            f"loss={loss:.4f}, "
            f"acc={accuracy:.4f if accuracy else 'N/A'}, "
            f"best={self.best_performance:.4f}"
        )

        # Record history
        self.history["loss"].append(loss)
        if accuracy is not None:
            self.history["accuracy"].append(accuracy)
        self.history["stage"].append(stage_name)
        self.history["epoch"].append(self.current_epoch)

        self.current_epoch += 1

    def should_advance(self) -> bool:
        """
        Check if we should advance to next curriculum stage.

        Returns:
            True if should advance, False otherwise
        """
        # Don't advance if already at last stage
        if self.current_stage_idx >= len(self.config.stages) - 1:
            return False

        # Strategy-specific logic
        if self.config.strategy == CurriculumStrategy.FIXED:
            # Advance after fixed number of epochs
            return self.current_epoch >= self.current_stage.num_epochs

        elif self.config.strategy == CurriculumStrategy.ADAPTIVE:
            # Advance if performance plateaus
            if self.epochs_without_improvement >= self.config.patience:
                logger.info(
                    f"No improvement for {self.config.patience} epochs, advancing stage"
                )
                return True

            # Or if success threshold reached
            stage_name = self.current_stage.name
            accuracies = self.stage_performance.get(f"{stage_name}_accuracy", [])
            if accuracies:
                recent_acc = sum(accuracies[-5:]) / len(accuracies[-5:])
                if recent_acc >= self.current_stage.success_threshold:
                    logger.info(
                        f"Success threshold reached ({recent_acc:.2%}), advancing stage"
                    )
                    return True

            return False

        else:
            # Default: fixed progression
            return self.current_epoch >= self.current_stage.num_epochs

    def advance_stage(self) -> None:
        """Advance to next curriculum stage."""
        if self.current_stage_idx < len(self.config.stages) - 1:
            self.current_stage_idx += 1
            self.current_epoch = 0
            self.epochs_without_improvement = 0

            logger.info(
                f"Advanced to stage '{self.current_stage.name}' "
                f"(difficulty {self.current_stage.difficulty_range})"
            )

            # Optionally rescore samples
            if self.config.auto_adjust:
                self.score_all_samples()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get curriculum learning statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "current_stage": self.current_stage.name,
            "current_epoch": self.current_epoch,
            "total_samples": len(self.dataset),
            "scored_samples": len(self.difficulty_scores),
            "stage_performance": dict(self.stage_performance),
            "best_performance": self.best_performance,
            "epochs_without_improvement": self.epochs_without_improvement,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_curriculum(
    num_stages: int = 3,
    total_epochs: int = 10,
) -> List[CurriculumStage]:
    """
    Create a default curriculum with evenly spaced stages.

    Args:
        num_stages: Number of curriculum stages
        total_epochs: Total number of epochs

    Returns:
        List of curriculum stages
    """
    stages = []
    difficulty_step = 1.0 / num_stages
    epochs_per_stage = total_epochs // num_stages

    for i in range(num_stages):
        min_diff = i * difficulty_step
        max_diff = (i + 1) * difficulty_step

        stage = CurriculumStage(
            name=f"stage_{i+1}",
            difficulty_range=(min_diff, max_diff),
            num_epochs=epochs_per_stage,
        )
        stages.append(stage)

    return stages


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DifficultyMetric",
    "DifficultyScore",
    "DifficultyScorer",
    "LossDifficultyScorer",
    "LengthDifficultyScorer",
    "CurriculumStrategy",
    "CurriculumStage",
    "CurriculumConfig",
    "CurriculumSampler",
    "CurriculumLearner",
    "create_default_curriculum",
]
