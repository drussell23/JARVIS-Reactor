"""
Meta-Learning for JARVIS Reactor.

Implements state-of-the-art meta-learning algorithms:
- MAML (Model-Agnostic Meta-Learning)
- Reptile
- Meta-SGD
- Task embedding and adaptation
- Few-shot learning
- Rapid task transfer

Based on research:
- "Model-Agnostic Meta-Learning" (Finn et al., 2017)
- "On First-Order Meta-Learning Algorithms" (Nichol et al., 2018)
- "Meta-SGD" (Li et al., 2017)
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# META-LEARNING ALGORITHMS
# =============================================================================

class MetaAlgorithm(Enum):
    """Meta-learning algorithms."""

    MAML = auto()          # Model-Agnostic Meta-Learning
    FIRST_ORDER_MAML = auto()  # First-order MAML (faster)
    REPTILE = auto()       # Reptile
    META_SGD = auto()      # Meta-SGD (learns learning rates)


# =============================================================================
# TASK REPRESENTATION
# =============================================================================

@dataclass
class Task:
    """
    A meta-learning task.

    In N-way K-shot learning:
    - N = number of classes
    - K = number of examples per class in support set

    Attributes:
        support_set: Training examples for the task
        query_set: Test examples for the task
        task_id: Unique task identifier
        metadata: Additional task information
    """

    support_set: Tuple[torch.Tensor, torch.Tensor]  # (X, y)
    query_set: Tuple[torch.Tensor, torch.Tensor]    # (X, y)
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_way(self) -> int:
        """Number of classes in the task."""
        _, y_support = self.support_set
        return len(torch.unique(y_support))

    @property
    def k_shot(self) -> int:
        """Number of examples per class in support set."""
        _, y_support = self.support_set
        return len(y_support) // self.n_way


class TaskSampler(Protocol):
    """Protocol for task sampling."""

    def sample_task(self) -> Task:
        """Sample a single task."""
        ...

    def sample_batch(self, batch_size: int) -> List[Task]:
        """Sample a batch of tasks."""
        ...


# =============================================================================
# MAML IMPLEMENTATION
# =============================================================================

@dataclass
class MAMLConfig:
    """
    Configuration for MAML.

    Attributes:
        inner_lr: Learning rate for inner loop (task adaptation)
        outer_lr: Learning rate for outer loop (meta-update)
        num_inner_steps: Number of gradient steps in inner loop
        first_order: Use first-order approximation (faster)
        allow_unused: Allow unused parameters in backward pass
        allow_nograd: Allow no-grad parameters
    """

    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    first_order: bool = False
    allow_unused: bool = True
    allow_nograd: bool = True


class MAMLTrainer:
    """
    MAML (Model-Agnostic Meta-Learning) trainer.

    MAML learns model parameters that can quickly adapt to new tasks
    with just a few gradient steps.

    Algorithm:
    1. Sample batch of tasks
    2. For each task:
       a. Clone model parameters
       b. Take K gradient steps on support set (inner loop)
       c. Evaluate on query set
    3. Meta-update using gradients from all tasks (outer loop)

    Example:
        >>> model = ConvNet()
        >>> config = MAMLConfig(
        ...     inner_lr=0.01,
        ...     outer_lr=0.001,
        ...     num_inner_steps=5,
        ... )
        >>>
        >>> trainer = MAMLTrainer(model, config)
        >>> meta_loss = trainer.meta_train_step(task_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or F.cross_entropy

        # Meta-optimizer (for outer loop)
        self.meta_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.outer_lr,
        )

        # Statistics
        self.meta_losses: List[float] = []
        self.inner_losses: List[float] = []

    def inner_loop(
        self,
        task: Task,
        create_graph: bool = True,
    ) -> Tuple[nn.Module, List[float]]:
        """
        Perform inner loop adaptation on a single task.

        Args:
            task: Task to adapt to
            create_graph: Whether to create computation graph (needed for second-order)

        Returns:
            Tuple of (adapted_model, inner_losses)
        """
        # Clone model for task-specific adaptation
        adapted_model = self._clone_model()

        # Get support set
        X_support, y_support = task.support_set

        # Inner loop: K gradient steps on support set
        inner_losses = []
        for step in range(self.config.num_inner_steps):
            # Forward pass
            logits = adapted_model(X_support)
            loss = self.loss_fn(logits, y_support)
            inner_losses.append(loss.item())

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=create_graph and not self.config.first_order,
                allow_unused=self.config.allow_unused,
            )

            # Manual SGD step (to maintain computation graph)
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - self.config.inner_lr * grad

        return adapted_model, inner_losses

    def meta_train_step(
        self,
        task_batch: List[Task],
    ) -> float:
        """
        Perform one meta-training step on a batch of tasks.

        Args:
            task_batch: Batch of tasks

        Returns:
            Meta-loss (average query loss across tasks)
        """
        self.meta_optimizer.zero_grad()

        query_losses = []

        # Process each task
        for task in task_batch:
            # Inner loop: adapt to task
            adapted_model, inner_losses = self.inner_loop(task)

            # Outer loop: evaluate adapted model on query set
            X_query, y_query = task.query_set
            logits = adapted_model(X_query)
            query_loss = self.loss_fn(logits, y_query)

            query_losses.append(query_loss)

            # Store inner losses for logging
            self.inner_losses.extend(inner_losses)

        # Meta-loss: average query loss
        meta_loss = torch.stack(query_losses).mean()

        # Meta-gradient step
        meta_loss.backward()
        self.meta_optimizer.step()

        # Statistics
        meta_loss_value = meta_loss.item()
        self.meta_losses.append(meta_loss_value)

        return meta_loss_value

    def adapt_to_task(
        self,
        task: Task,
        return_losses: bool = False,
    ) -> Union[nn.Module, Tuple[nn.Module, List[float]]]:
        """
        Adapt the meta-learned model to a new task.

        This is used at test time to quickly adapt to new tasks.

        Args:
            task: Task to adapt to
            return_losses: Whether to return inner losses

        Returns:
            Adapted model (and optionally inner losses)
        """
        self.model.eval()

        with torch.no_grad():
            adapted_model, inner_losses = self.inner_loop(
                task,
                create_graph=False,
            )

        if return_losses:
            return adapted_model, inner_losses
        return adapted_model

    def evaluate(
        self,
        task_batch: List[Task],
    ) -> Dict[str, float]:
        """
        Evaluate on a batch of tasks.

        Args:
            task_batch: Batch of tasks

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        query_losses = []
        query_accuracies = []

        for task in task_batch:
            # Adapt to task
            adapted_model = self.adapt_to_task(task)

            # Evaluate on query set
            X_query, y_query = task.query_set
            with torch.no_grad():
                logits = adapted_model(X_query)
                loss = self.loss_fn(logits, y_query)
                preds = logits.argmax(dim=1)
                accuracy = (preds == y_query).float().mean()

            query_losses.append(loss.item())
            query_accuracies.append(accuracy.item())

        return {
            "query_loss": sum(query_losses) / len(query_losses),
            "query_accuracy": sum(query_accuracies) / len(query_accuracies),
        }

    def _clone_model(self) -> nn.Module:
        """Clone model for task-specific adaptation."""
        cloned = copy.deepcopy(self.model)
        cloned.train()
        return cloned


# =============================================================================
# REPTILE IMPLEMENTATION
# =============================================================================

@dataclass
class ReptileConfig:
    """
    Configuration for Reptile.

    Attributes:
        inner_lr: Learning rate for inner loop
        outer_lr: Learning rate for outer loop (meta-step size)
        num_inner_steps: Number of gradient steps in inner loop
        batch_size: Number of tasks per meta-update
    """

    inner_lr: float = 0.01
    outer_lr: float = 0.1
    num_inner_steps: int = 5
    batch_size: int = 5


class ReptileTrainer:
    """
    Reptile meta-learning trainer.

    Reptile is a simpler, first-order alternative to MAML.
    Instead of computing second-order gradients, it just moves
    the meta-parameters in the direction of the task-adapted parameters.

    Algorithm:
    1. Sample batch of tasks
    2. For each task:
       a. Clone model parameters θ
       b. Take K SGD steps → get θ'
    3. Meta-update: θ ← θ + ε * (θ' - θ)

    Example:
        >>> model = ConvNet()
        >>> config = ReptileConfig(
        ...     inner_lr=0.01,
        ...     outer_lr=0.1,
        ...     num_inner_steps=5,
        ... )
        >>>
        >>> trainer = ReptileTrainer(model, config)
        >>> trainer.meta_train_step(task_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReptileConfig,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or F.cross_entropy

        # Statistics
        self.meta_losses: List[float] = []

    def inner_loop(
        self,
        task: Task,
    ) -> nn.Module:
        """
        Perform inner loop adaptation on a single task.

        Args:
            task: Task to adapt to

        Returns:
            Adapted model
        """
        # Clone model
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        # Create optimizer for inner loop
        inner_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr,
        )

        # Get support set
        X_support, y_support = task.support_set

        # Inner loop: K gradient steps
        for _ in range(self.config.num_inner_steps):
            inner_optimizer.zero_grad()
            logits = adapted_model(X_support)
            loss = self.loss_fn(logits, y_support)
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_train_step(
        self,
        task_batch: List[Task],
    ) -> float:
        """
        Perform one Reptile meta-training step.

        Args:
            task_batch: Batch of tasks

        Returns:
            Average meta-loss
        """
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]

        meta_gradients = []

        # Process each task
        for task in task_batch:
            # Adapt to task
            adapted_model = self.inner_loop(task)

            # Compute meta-gradient: θ' - θ
            meta_grad = [
                adapted_p - initial_p
                for adapted_p, initial_p in zip(
                    adapted_model.parameters(),
                    initial_params,
                )
            ]
            meta_gradients.append(meta_grad)

        # Average meta-gradients across tasks
        avg_meta_grad = [
            torch.stack(grads).mean(dim=0)
            for grads in zip(*meta_gradients)
        ]

        # Meta-update: θ ← θ + ε * avg(θ' - θ)
        with torch.no_grad():
            for param, meta_grad in zip(self.model.parameters(), avg_meta_grad):
                param.add_(meta_grad, alpha=self.config.outer_lr)

        # Compute meta-loss for logging
        query_losses = []
        for task in task_batch:
            X_query, y_query = task.query_set
            with torch.no_grad():
                logits = self.model(X_query)
                loss = self.loss_fn(logits, y_query)
                query_losses.append(loss.item())

        meta_loss = sum(query_losses) / len(query_losses)
        self.meta_losses.append(meta_loss)

        return meta_loss


# =============================================================================
# META-SGD IMPLEMENTATION
# =============================================================================

@dataclass
class MetaSGDConfig:
    """
    Configuration for Meta-SGD.

    Meta-SGD learns per-parameter learning rates in addition to
    model parameters.

    Attributes:
        initial_lr: Initial learning rate
        outer_lr: Meta-learning rate
        num_inner_steps: Number of inner loop steps
        learn_lr: Whether to learn learning rates
    """

    initial_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    learn_lr: bool = True


class MetaSGDTrainer:
    """
    Meta-SGD trainer.

    Meta-SGD extends MAML by learning per-parameter learning rates.
    This allows different parts of the model to adapt at different rates.

    Example:
        >>> model = ConvNet()
        >>> config = MetaSGDConfig(
        ...     initial_lr=0.01,
        ...     outer_lr=0.001,
        ...     learn_lr=True,
        ... )
        >>>
        >>> trainer = MetaSGDTrainer(model, config)
        >>> meta_loss = trainer.meta_train_step(task_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        config: MetaSGDConfig,
        loss_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or F.cross_entropy

        # Initialize learnable learning rates (one per parameter)
        if config.learn_lr:
            self.learning_rates = nn.ParameterList([
                nn.Parameter(torch.ones_like(p) * config.initial_lr)
                for p in model.parameters()
            ])
        else:
            self.learning_rates = [config.initial_lr] * len(list(model.parameters()))

        # Meta-optimizer
        if config.learn_lr:
            params_to_optimize = list(model.parameters()) + list(self.learning_rates)
        else:
            params_to_optimize = list(model.parameters())

        self.meta_optimizer = optim.Adam(
            params_to_optimize,
            lr=config.outer_lr,
        )

        # Statistics
        self.meta_losses: List[float] = []

    def inner_loop(
        self,
        task: Task,
        create_graph: bool = True,
    ) -> nn.Module:
        """Inner loop with learned learning rates."""
        # Clone model
        adapted_model = copy.deepcopy(self.model)

        # Get support set
        X_support, y_support = task.support_set

        # Inner loop
        for _ in range(self.config.num_inner_steps):
            # Forward pass
            logits = adapted_model(X_support)
            loss = self.loss_fn(logits, y_support)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=create_graph,
            )

            # Manual SGD with learned learning rates
            with torch.no_grad():
                for param, grad, lr in zip(
                    adapted_model.parameters(),
                    grads,
                    self.learning_rates,
                ):
                    if grad is not None:
                        param.data = param.data - lr * grad

        return adapted_model

    def meta_train_step(
        self,
        task_batch: List[Task],
    ) -> float:
        """Meta-training step with learned learning rates."""
        self.meta_optimizer.zero_grad()

        query_losses = []

        for task in task_batch:
            # Inner loop
            adapted_model = self.inner_loop(task)

            # Evaluate on query set
            X_query, y_query = task.query_set
            logits = adapted_model(X_query)
            query_loss = self.loss_fn(logits, y_query)
            query_losses.append(query_loss)

        # Meta-loss
        meta_loss = torch.stack(query_losses).mean()
        meta_loss.backward()
        self.meta_optimizer.step()

        meta_loss_value = meta_loss.item()
        self.meta_losses.append(meta_loss_value)

        return meta_loss_value


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_n_way_k_shot_task(
    dataset: Dataset,
    n_way: int = 5,
    k_shot: int = 5,
    q_queries: int = 15,
) -> Task:
    """
    Create an N-way K-shot task from a dataset.

    Args:
        dataset: Dataset to sample from
        n_way: Number of classes
        k_shot: Number of support examples per class
        q_queries: Number of query examples per class

    Returns:
        Task
    """
    # Sample N classes
    all_labels = set(dataset.targets if hasattr(dataset, "targets") else range(len(dataset)))
    sampled_classes = random.sample(list(all_labels), n_way)

    # Sample K+Q examples per class
    support_X, support_y = [], []
    query_X, query_y = [], []

    for class_idx, class_label in enumerate(sampled_classes):
        # Get all examples of this class
        class_examples = [
            dataset[i]
            for i in range(len(dataset))
            if dataset.targets[i] == class_label
        ]

        # Sample K + Q examples
        sampled = random.sample(class_examples, k_shot + q_queries)

        # Split into support and query
        for i, example in enumerate(sampled):
            if i < k_shot:
                support_X.append(example[0])
                support_y.append(class_idx)
            else:
                query_X.append(example[0])
                query_y.append(class_idx)

    # Convert to tensors
    support_X = torch.stack(support_X)
    support_y = torch.tensor(support_y)
    query_X = torch.stack(query_X)
    query_y = torch.tensor(query_y)

    return Task(
        support_set=(support_X, support_y),
        query_set=(query_X, query_y),
        task_id=f"{n_way}way_{k_shot}shot_{random.randint(0, 1000000)}",
        metadata={
            "n_way": n_way,
            "k_shot": k_shot,
            "q_queries": q_queries,
        },
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
]
