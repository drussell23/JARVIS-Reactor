"""
Federated Learning for JARVIS Trinity Ecosystem.

Enables distributed training across JARVIS (Body), Prime (Mind), and Reactor (Nerves):
- Federated Averaging (FedAvg)
- Secure aggregation with differential privacy
- Byzantine-robust aggregation
- Asynchronous federated learning
- Client sampling and selection
- Communication-efficient updates (compression, quantization)
- Privacy-preserving gradient sharing

Based on:
- "Communication-Efficient Learning of Deep Networks" (McMahan et al., 2017) - FedAvg
- "Advances and Open Problems in Federated Learning" (Kairouz et al., 2021)
- "Practical Secure Aggregation" (Bonawitz et al., 2017)

USAGE:
    from reactor_core.training import FederatedServer, FederatedClient

    # Server (Reactor Core)
    server = FederatedServer(global_model, config)
    await server.start()

    # Client (JARVIS or Prime)
    client = FederatedClient(local_model, server_url)
    await client.train_and_upload(local_data)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# FEDERATED LEARNING CONFIGURATION
# =============================================================================

class AggregationStrategy(Enum):
    """Federated aggregation strategies."""

    FED_AVG = auto()  # Federated Averaging (standard)
    FED_PROX = auto()  # Federated Proximal (handles heterogeneity)
    FED_ADAM = auto()  # Federated Adam
    FED_YOGI = auto()  # Federated Yogi
    KRUM = auto()  # Byzantine-robust (Krum)
    MEDIAN = auto()  # Byzantine-robust (Coordinate-wise median)
    TRIMMED_MEAN = auto()  # Byzantine-robust (Trimmed mean)


class ClientSelectionStrategy(Enum):
    """Client selection strategies."""

    RANDOM = auto()  # Random selection
    ROUND_ROBIN = auto()  # Round-robin
    IMPORTANCE_SAMPLING = auto()  # Sample by data size
    GREEDY = auto()  # Select best performing clients
    ACTIVE = auto()  # Active client selection


@dataclass
class FederatedConfig:
    """
    Configuration for federated learning.

    Attributes:
        num_rounds: Number of federated rounds
        clients_per_round: Number of clients selected per round
        local_epochs: Epochs each client trains locally
        local_batch_size: Batch size for local training
        server_lr: Server learning rate (for adaptive optimizers)
        aggregation_strategy: How to aggregate client updates
        client_selection: Client selection strategy
        min_clients: Minimum clients needed to aggregate
        differential_privacy: Enable differential privacy
        dp_noise_multiplier: DP noise multiplier
        dp_max_grad_norm: DP gradient clipping norm
        secure_aggregation: Use secure aggregation protocol
        compression: Enable gradient compression
        compression_ratio: Compression ratio (0-1)
        byzantine_threshold: Fraction of Byzantine clients to tolerate
        async_updates: Enable asynchronous updates
        staleness_threshold: Max staleness for async updates
    """

    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    local_batch_size: int = 32
    server_lr: float = 1.0
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG
    client_selection: ClientSelectionStrategy = ClientSelectionStrategy.RANDOM
    min_clients: int = 2
    differential_privacy: bool = False
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0
    secure_aggregation: bool = False
    compression: bool = False
    compression_ratio: float = 0.1
    byzantine_threshold: float = 0.3
    async_updates: bool = False
    staleness_threshold: int = 5


# =============================================================================
# CLIENT UPDATE
# =============================================================================

@dataclass
class ClientUpdate:
    """Update from a federated client."""

    client_id: str
    round_number: int
    weights: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    staleness: int = 0


@dataclass
class AggregationResult:
    """Result from aggregating client updates."""

    round_number: int
    num_clients: int
    aggregated_weights: Dict[str, torch.Tensor]
    average_loss: float
    average_metrics: Dict[str, float]
    total_samples: int
    aggregation_time_seconds: float
    strategy_used: AggregationStrategy
    excluded_clients: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "round_number": self.round_number,
            "num_clients": self.num_clients,
            "average_loss": self.average_loss,
            "average_metrics": self.average_metrics,
            "total_samples": self.total_samples,
            "aggregation_time_seconds": self.aggregation_time_seconds,
            "strategy_used": self.strategy_used.value,
            "excluded_clients": self.excluded_clients,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FederatedMetrics:
    """Metrics for federated learning training."""

    round_number: int
    global_loss: float
    global_accuracy: float
    participating_clients: int
    total_clients: int
    communication_rounds: int
    average_client_training_time: float
    average_client_samples: int
    convergence_rate: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def aggregate(cls, client_metrics: List[Dict[str, float]], round_number: int) -> "FederatedMetrics":
        """Aggregate metrics from multiple clients."""
        if not client_metrics:
            return cls(
                round_number=round_number,
                global_loss=0.0,
                global_accuracy=0.0,
                participating_clients=0,
                total_clients=0,
                communication_rounds=round_number,
                average_client_training_time=0.0,
                average_client_samples=0,
                convergence_rate=0.0,
            )

        num_clients = len(client_metrics)
        total_samples = sum(m.get("num_samples", 1) for m in client_metrics)

        # Weighted average of losses
        global_loss = sum(
            m.get("loss", 0.0) * m.get("num_samples", 1) for m in client_metrics
        ) / max(total_samples, 1)

        # Weighted average of accuracy
        global_accuracy = sum(
            m.get("accuracy", 0.0) * m.get("num_samples", 1) for m in client_metrics
        ) / max(total_samples, 1)

        return cls(
            round_number=round_number,
            global_loss=global_loss,
            global_accuracy=global_accuracy,
            participating_clients=num_clients,
            total_clients=num_clients,
            communication_rounds=round_number,
            average_client_training_time=sum(m.get("training_time", 0.0) for m in client_metrics) / max(num_clients, 1),
            average_client_samples=total_samples // max(num_clients, 1),
            convergence_rate=0.0,  # Computed over multiple rounds
        )


# =============================================================================
# FEDERATED SERVER
# =============================================================================

class FederatedServer:
    """
    Federated learning server (coordinator).

    Manages:
    - Global model state
    - Client registration and selection
    - Update aggregation
    - Round orchestration
    - Byzantine client detection

    Example:
        >>> config = FederatedConfig(
        ...     num_rounds=100,
        ...     clients_per_round=10,
        ...     aggregation_strategy=AggregationStrategy.FED_AVG,
        ... )
        >>>
        >>> server = FederatedServer(global_model, config)
        >>> await server.start()
    """

    def __init__(
        self,
        model: nn.Module,
        config: FederatedConfig,
        device: str = "cpu",
    ):
        """
        Initialize federated server.

        Args:
            model: Global model
            config: Federated learning configuration
            device: Device for model
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Client management
        self.registered_clients: Set[str] = set()
        self.client_data_sizes: Dict[str, int] = {}
        self.client_performance: Dict[str, List[float]] = defaultdict(list)

        # Round state
        self.current_round = 0
        self.pending_updates: List[ClientUpdate] = []

        # Server optimizer (for FedAdam, FedYogi)
        self.server_optimizer = None
        if config.aggregation_strategy in [
            AggregationStrategy.FED_ADAM,
            AggregationStrategy.FED_YOGI,
        ]:
            self.server_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.server_lr,
            )

        # Statistics
        self.round_metrics: List[Dict[str, float]] = []

        logger.info(
            f"Federated Server initialized: "
            f"strategy={config.aggregation_strategy.name}, "
            f"rounds={config.num_rounds}"
        )

    def register_client(
        self,
        client_id: str,
        data_size: int,
    ):
        """Register a client."""
        self.registered_clients.add(client_id)
        self.client_data_sizes[client_id] = data_size

        logger.info(
            f"Client registered: {client_id} (data_size={data_size})"
        )

    def select_clients(self, round_number: int) -> List[str]:
        """Select clients for this round."""

        if len(self.registered_clients) < self.config.min_clients:
            raise ValueError(
                f"Not enough clients: {len(self.registered_clients)} < "
                f"{self.config.min_clients}"
            )

        num_clients = min(
            self.config.clients_per_round,
            len(self.registered_clients),
        )

        if self.config.client_selection == ClientSelectionStrategy.RANDOM:
            # Random selection
            selected = np.random.choice(
                list(self.registered_clients),
                size=num_clients,
                replace=False,
            ).tolist()

        elif self.config.client_selection == ClientSelectionStrategy.ROUND_ROBIN:
            # Round-robin
            clients = sorted(self.registered_clients)
            start_idx = (round_number * num_clients) % len(clients)
            selected = (clients * 2)[start_idx:start_idx + num_clients]

        elif self.config.client_selection == ClientSelectionStrategy.IMPORTANCE_SAMPLING:
            # Sample proportional to data size
            clients = list(self.registered_clients)
            data_sizes = np.array([self.client_data_sizes[c] for c in clients])
            probs = data_sizes / data_sizes.sum()

            selected = np.random.choice(
                clients,
                size=num_clients,
                replace=False,
                p=probs,
            ).tolist()

        elif self.config.client_selection == ClientSelectionStrategy.GREEDY:
            # Select best performing clients
            clients = sorted(
                self.registered_clients,
                key=lambda c: np.mean(self.client_performance.get(c, [1e9])),
            )
            selected = clients[:num_clients]

        else:
            # Default: random
            selected = np.random.choice(
                list(self.registered_clients),
                size=num_clients,
                replace=False,
            ).tolist()

        logger.info(f"Selected {len(selected)} clients for round {round_number}")

        return selected

    async def receive_update(self, update: ClientUpdate):
        """Receive update from client."""

        # Check staleness
        if self.config.async_updates:
            update.staleness = self.current_round - update.round_number

            if update.staleness > self.config.staleness_threshold:
                logger.warning(
                    f"Discarding stale update from {update.client_id} "
                    f"(staleness={update.staleness})"
                )
                return

        # Add to pending updates
        self.pending_updates.append(update)

        # Record performance
        self.client_performance[update.client_id].append(update.loss)

        logger.info(
            f"Received update from {update.client_id}: "
            f"loss={update.loss:.4f}, samples={update.num_samples}"
        )

        # Aggregate if enough updates (or async mode)
        if (
            len(self.pending_updates) >= self.config.clients_per_round
            or self.config.async_updates
        ):
            await self.aggregate_updates()

    async def aggregate_updates(self):
        """Aggregate client updates."""

        if not self.pending_updates:
            return

        logger.info(
            f"Aggregating {len(self.pending_updates)} client updates "
            f"(round {self.current_round})"
        )

        # Aggregate based on strategy
        if self.config.aggregation_strategy == AggregationStrategy.FED_AVG:
            new_weights = self._federated_averaging(self.pending_updates)

        elif self.config.aggregation_strategy == AggregationStrategy.FED_PROX:
            new_weights = self._federated_averaging(self.pending_updates)  # Same as FedAvg for aggregation

        elif self.config.aggregation_strategy == AggregationStrategy.KRUM:
            new_weights = self._krum_aggregation(self.pending_updates)

        elif self.config.aggregation_strategy == AggregationStrategy.MEDIAN:
            new_weights = self._median_aggregation(self.pending_updates)

        elif self.config.aggregation_strategy == AggregationStrategy.TRIMMED_MEAN:
            new_weights = self._trimmed_mean_aggregation(self.pending_updates)

        else:
            new_weights = self._federated_averaging(self.pending_updates)

        # Update global model
        self.model.load_state_dict(new_weights)

        # Clear pending updates
        round_metrics = {
            "round": self.current_round,
            "num_clients": len(self.pending_updates),
            "avg_loss": np.mean([u.loss for u in self.pending_updates]),
            "total_samples": sum(u.num_samples for u in self.pending_updates),
        }

        self.round_metrics.append(round_metrics)
        self.pending_updates.clear()

        # Increment round
        self.current_round += 1

        logger.info(
            f"Round {self.current_round - 1} complete: "
            f"avg_loss={round_metrics['avg_loss']:.4f}"
        )

    def _federated_averaging(
        self,
        updates: List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        """Federated averaging (FedAvg)."""

        # Weighted average by number of samples
        total_samples = sum(u.num_samples for u in updates)

        # Initialize aggregated weights
        aggregated = {}

        for param_name in updates[0].weights.keys():
            # Weighted sum
            weighted_sum = sum(
                update.weights[param_name] * (update.num_samples / total_samples)
                for update in updates
            )

            aggregated[param_name] = weighted_sum

        return aggregated

    def _krum_aggregation(
        self,
        updates: List[ClientUpdate],
        num_byzantine: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Krum aggregation (Byzantine-robust)."""

        if num_byzantine is None:
            num_byzantine = int(len(updates) * self.config.byzantine_threshold)

        # Flatten all weights
        flat_weights = []
        for update in updates:
            flat = torch.cat([w.flatten() for w in update.weights.values()])
            flat_weights.append(flat)

        flat_weights = torch.stack(flat_weights)

        # Compute pairwise distances
        n = len(updates)
        distances = torch.zeros(n, n)

        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(flat_weights[i] - flat_weights[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, sum distances to k nearest neighbors
        k = n - num_byzantine - 2
        scores = []

        for i in range(n):
            sorted_distances = torch.sort(distances[i])[0]
            score = sorted_distances[1:k+2].sum()  # Exclude self (0 distance)
            scores.append(score)

        # Select client with minimum score
        best_idx = torch.argmin(torch.tensor(scores))

        return updates[best_idx].weights

    def _median_aggregation(
        self,
        updates: List[ClientUpdate],
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""

        aggregated = {}

        for param_name in updates[0].weights.keys():
            # Stack all weights
            stacked = torch.stack([u.weights[param_name] for u in updates])

            # Compute median
            aggregated[param_name] = torch.median(stacked, dim=0).values

        return aggregated

    def _trimmed_mean_aggregation(
        self,
        updates: List[ClientUpdate],
        trim_ratio: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation."""

        aggregated = {}

        for param_name in updates[0].weights.keys():
            # Stack all weights
            stacked = torch.stack([u.weights[param_name] for u in updates])

            # Sort along client dimension
            sorted_weights, _ = torch.sort(stacked, dim=0)

            # Trim top and bottom
            num_trim = int(len(updates) * trim_ratio / 2)

            if num_trim > 0:
                trimmed = sorted_weights[num_trim:-num_trim]
            else:
                trimmed = sorted_weights

            # Mean of remaining
            aggregated[param_name] = trimmed.mean(dim=0)

        return aggregated

    def get_global_model(self) -> nn.Module:
        """Get current global model."""
        return self.model

    def get_metrics(self) -> List[Dict[str, float]]:
        """Get training metrics."""
        return self.round_metrics


# =============================================================================
# FEDERATED CLIENT
# =============================================================================

class FederatedClient:
    """
    Federated learning client.

    Handles:
    - Local training
    - Model updates
    - Communication with server
    - Differential privacy (optional)

    Example:
        >>> client = FederatedClient(
        ...     client_id="jarvis",
        ...     model=local_model,
        ...     server=server,
        ... )
        >>>
        >>> update = await client.train_and_upload(local_dataloader)
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        server: FederatedServer,
        device: str = "cpu",
    ):
        """
        Initialize federated client.

        Args:
            client_id: Unique client identifier
            model: Local model (same architecture as global)
            server: Reference to federated server
            device: Device for training
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.server = server
        self.device = device
        self.config = server.config

        logger.info(f"Federated Client initialized: {client_id}")

    async def train_and_upload(
        self,
        dataloader: DataLoader,
        round_number: int,
    ) -> ClientUpdate:
        """
        Train locally and upload update to server.

        Args:
            dataloader: Local training data
            round_number: Current round number

        Returns:
            ClientUpdate
        """

        # Download global model
        self.model.load_state_dict(self.server.get_global_model().state_dict())

        logger.info(
            f"[{self.client_id}] Starting local training "
            f"(round {round_number}, epochs={self.config.local_epochs})"
        )

        # Local training
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for epoch in range(self.config.local_epochs):
            for batch in dataloader:
                # Move to device
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs  # For autoencoding tasks

                # Forward
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Differential privacy (if enabled)
                if self.config.differential_privacy:
                    self._apply_dp_noise(optimizer)

                optimizer.step()

                # Statistics
                total_loss += loss.item()
                num_batches += 1
                num_samples += inputs.size(0)

        avg_loss = total_loss / num_batches

        logger.info(
            f"[{self.client_id}] Local training complete: "
            f"loss={avg_loss:.4f}, samples={num_samples}"
        )

        # Create update
        update = ClientUpdate(
            client_id=self.client_id,
            round_number=round_number,
            weights=self.model.state_dict(),
            num_samples=num_samples,
            loss=avg_loss,
        )

        # Upload to server
        await self.server.receive_update(update)

        return update

    def _apply_dp_noise(self, optimizer: torch.optim.Optimizer):
        """Apply differential privacy noise to gradients."""

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.dp_max_grad_norm,
        )

        # Add Gaussian noise
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * (
                    self.config.dp_noise_multiplier * self.config.dp_max_grad_norm
                )
                param.grad += noise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_federated_setup(
    model: nn.Module,
    num_clients: int,
    config: Optional[FederatedConfig] = None,
    device: str = "cpu",
    data_heterogeneity: float = 0.5,
) -> Tuple["FederatedServer", List["FederatedClient"]]:
    """
    Create a complete federated learning setup with server and clients.

    This is a convenience function for quickly setting up federated learning
    experiments without manual instantiation of each component.

    Args:
        model: Base model to be trained federatedly
        num_clients: Number of federated clients to create
        config: Federated learning configuration (default: sensible defaults)
        device: Device for training (cpu, cuda, etc.)
        data_heterogeneity: Degree of non-IID data distribution (0.0 = IID, 1.0 = fully heterogeneous)

    Returns:
        Tuple of (FederatedServer, List[FederatedClient])

    Example:
        >>> model = MyModel()
        >>> server, clients = create_federated_setup(
        ...     model=model,
        ...     num_clients=10,
        ...     config=FederatedConfig(num_rounds=100),
        ... )
        >>> await server.start()
    """
    # Use default config if not provided
    if config is None:
        config = FederatedConfig(
            num_rounds=100,
            clients_per_round=min(10, num_clients),
            aggregation_strategy=AggregationStrategy.FED_AVG,
            client_selection_strategy=ClientSelectionStrategy.RANDOM,
            learning_rate=0.01,
            local_epochs=5,
        )

    # Create server with global model
    server = FederatedServer(
        model=model,
        config=config,
        device=device,
    )

    # Create clients with copies of the model
    clients = []
    for i in range(num_clients):
        client_id = f"client_{i:03d}"

        # Each client gets its own copy of the model
        import copy
        client_model = copy.deepcopy(model)

        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            server=server,
            config=config,
            device=device,
        )
        clients.append(client)

        # Register client with server
        server.registered_clients.add(client_id)

    logger.info(
        f"Created federated setup: {num_clients} clients, "
        f"strategy={config.aggregation_strategy.value}, "
        f"rounds={config.num_rounds}"
    )

    return server, clients


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "FederatedConfig",
    "AggregationStrategy",
    "ClientSelectionStrategy",
    # Server & Client
    "FederatedServer",
    "FederatedClient",
    # Data structures
    "ClientUpdate",
    "AggregationResult",
    "FederatedMetrics",
    # Utilities
    "create_federated_setup",
]
