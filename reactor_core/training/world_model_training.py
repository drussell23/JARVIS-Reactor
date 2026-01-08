"""
World Model Training for JARVIS Reactor Core.

Implements:
- Latent world models with encoder/decoder
- Transition dynamics learning
- Reward and value prediction
- Model-based planning
- Counterfactual reasoning
- Imagined rollouts for training
- Integration with model-based RL

Based on research:
- "World Models" (Ha & Schmidhuber, 2018)
- "Dreamer: Scalable RL using World Models" (Hafner et al., 2020)
- "MuZero: Mastering Atari, Go, Chess and Shogi" (Schrittwieser et al., 2020)
- "Model-Based RL" (Sutton & Barto, 2018)
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# WORLD MODEL COMPONENTS
# =============================================================================

class LatentEncoder(nn.Module):
    """
    Encode observations into latent state space.

    Maps high-dimensional observations (e.g., text embeddings, images)
    into compact latent representations.
    """

    def __init__(
        self,
        observation_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        """
        Initialize latent encoder.

        Args:
            observation_dim: Dimension of observations
            latent_dim: Dimension of latent space
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        hidden_dims = hidden_dims or [512, 256]

        # Build encoder network
        layers = []
        input_dim = observation_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim

        # Output: mean and log_std for stochastic latent
        self.encoder = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logstd = nn.Linear(hidden_dims[-1], latent_dim)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.ReLU())

    def forward(
        self,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode observation to latent.

        Args:
            observation: Observation tensor [batch, obs_dim]

        Returns:
            (latent, mean, logstd) tuple
        """
        hidden = self.encoder(observation)
        mean = self.fc_mean(hidden)
        logstd = self.fc_logstd(hidden)
        logstd = torch.clamp(logstd, min=-10, max=2)  # Stability

        # Reparameterization trick
        std = torch.exp(logstd)
        latent = mean + std * torch.randn_like(std)

        return latent, mean, logstd


class LatentDecoder(nn.Module):
    """
    Decode latent states back to observations.

    Reconstructs observations from latent representations for
    learning meaningful latent space.
    """

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        """
        Initialize latent decoder.

        Args:
            latent_dim: Dimension of latent space
            observation_dim: Dimension of observations
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        hidden_dims = hidden_dims or [256, 512]

        # Build decoder network
        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], observation_dim))

        self.decoder = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to observation.

        Args:
            latent: Latent tensor [batch, latent_dim]

        Returns:
            Reconstructed observation [batch, obs_dim]
        """
        return self.decoder(latent)


class TransitionModel(nn.Module):
    """
    Model state transitions: s_{t+1} = f(s_t, a_t).

    Learns how latent state evolves given actions/context.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        deterministic: bool = False,
    ):
        """
        Initialize transition model.

        Args:
            latent_dim: Dimension of latent state
            action_dim: Dimension of action/context
            hidden_dims: Hidden layer dimensions
            deterministic: Use deterministic transitions
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.deterministic = deterministic
        hidden_dims = hidden_dims or [512, 512]

        # Build transition network
        layers = []
        input_dim = latent_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim

        self.transition_net = nn.Sequential(*layers)

        if deterministic:
            # Deterministic transition
            self.fc_next = nn.Linear(hidden_dims[-1], latent_dim)
        else:
            # Stochastic transition
            self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
            self.fc_logstd = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Predict next latent state.

        Args:
            latent: Current latent state [batch, latent_dim]
            action: Action/context [batch, action_dim]

        Returns:
            next_latent if deterministic, else (next_latent, mean, logstd)
        """
        # Concatenate state and action
        state_action = torch.cat([latent, action], dim=-1)
        hidden = self.transition_net(state_action)

        if self.deterministic:
            next_latent = self.fc_next(hidden)
            return next_latent
        else:
            mean = self.fc_mean(hidden)
            logstd = self.fc_logstd(hidden)
            logstd = torch.clamp(logstd, min=-10, max=2)

            std = torch.exp(logstd)
            next_latent = mean + std * torch.randn_like(std)

            return next_latent, mean, logstd


class RewardModel(nn.Module):
    """
    Predict rewards from latent states.

    Learns to predict immediate rewards for model-based RL.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int] = None,
    ):
        """
        Initialize reward model.

        Args:
            latent_dim: Dimension of latent state
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        hidden_dims = hidden_dims or [256, 128]

        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.reward_net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state."""
        return self.reward_net(latent).squeeze(-1)


class ValueModel(nn.Module):
    """
    Predict long-term value from latent states.

    Estimates future cumulative rewards for planning.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int] = None,
    ):
        """
        Initialize value model.

        Args:
            latent_dim: Dimension of latent state
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        hidden_dims = hidden_dims or [256, 128]

        layers = []
        input_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.value_net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict value from latent state."""
        return self.value_net(latent).squeeze(-1)


# =============================================================================
# WORLD MODEL
# =============================================================================

@dataclass
class WorldModelConfig:
    """
    Configuration for world model.

    Attributes:
        observation_dim: Dimension of observations
        latent_dim: Dimension of latent space
        action_dim: Dimension of actions/context
        hidden_dims: Hidden layer dimensions
        deterministic_transition: Use deterministic transitions
        learning_rate: Learning rate
        kl_weight: Weight for KL divergence loss
        reconstruction_weight: Weight for reconstruction loss
    """

    observation_dim: int
    latent_dim: int = 256
    action_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    deterministic_transition: bool = False
    learning_rate: float = 1e-4
    kl_weight: float = 1.0
    reconstruction_weight: float = 1.0
    reward_weight: float = 1.0
    value_weight: float = 1.0


class WorldModel(nn.Module):
    """
    Complete world model with encoder, decoder, transition, reward, value.

    Learns latent dynamics model of the world for planning and reasoning.

    Example:
        >>> config = WorldModelConfig(
        ...     observation_dim=768,
        ...     latent_dim=256,
        ...     action_dim=64,
        ... )
        >>> world_model = WorldModel(config)
        >>> trainer = WorldModelTrainer(world_model, config)
        >>> await trainer.train(dataset)
    """

    def __init__(self, config: WorldModelConfig):
        """Initialize world model."""
        super().__init__()

        self.config = config

        # Components
        self.encoder = LatentEncoder(
            observation_dim=config.observation_dim,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
        )

        self.decoder = LatentDecoder(
            latent_dim=config.latent_dim,
            observation_dim=config.observation_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
        )

        self.transition = TransitionModel(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            deterministic=config.deterministic_transition,
        )

        self.reward_model = RewardModel(
            latent_dim=config.latent_dim,
            hidden_dims=[256, 128],
        )

        self.value_model = ValueModel(
            latent_dim=config.latent_dim,
            hidden_dims=[256, 128],
        )

    def encode(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation to latent."""
        return self.encoder(observation)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to observation."""
        return self.decoder(latent)

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor):
        """Predict next latent state."""
        return self.transition(latent, action)

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent."""
        return self.reward_model(latent)

    def predict_value(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict value from latent."""
        return self.value_model(latent)

    def imagine_rollout(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine future trajectory in latent space.

        Args:
            initial_latent: Starting latent state [batch, latent_dim]
            actions: Action sequence [batch, horizon, action_dim]
            horizon: Planning horizon

        Returns:
            Dictionary with imagined trajectory
        """
        batch_size = initial_latent.size(0)
        device = initial_latent.device

        # Initialize trajectory
        latents = [initial_latent]
        rewards = []
        values = []

        current_latent = initial_latent

        # Roll out in imagination
        for t in range(horizon):
            action_t = actions[:, t, :]

            # Predict next state
            if self.config.deterministic_transition:
                next_latent = self.transition(current_latent, action_t)
            else:
                next_latent, _, _ = self.transition(current_latent, action_t)

            # Predict reward and value
            reward_t = self.reward_model(next_latent)
            value_t = self.value_model(next_latent)

            latents.append(next_latent)
            rewards.append(reward_t)
            values.append(value_t)

            current_latent = next_latent

        return {
            "latents": torch.stack(latents, dim=1),  # [batch, horizon+1, latent_dim]
            "rewards": torch.stack(rewards, dim=1),  # [batch, horizon]
            "values": torch.stack(values, dim=1),    # [batch, horizon]
        }


# =============================================================================
# WORLD MODEL TRAINER
# =============================================================================

@dataclass
class WorldModelTrainingConfig:
    """Training configuration for world model."""

    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    kl_weight: float = 1.0
    reconstruction_weight: float = 1.0
    reward_weight: float = 1.0
    value_weight: float = 0.5
    imagination_horizon: int = 15
    device: str = "cpu"
    checkpoint_dir: Optional[Path] = None
    log_frequency: int = 100


class WorldModelTrainer:
    """
    Trainer for world models.

    Trains encoder, decoder, transition, reward, and value models jointly.
    """

    def __init__(
        self,
        world_model: WorldModel,
        config: WorldModelTrainingConfig,
    ):
        """
        Initialize world model trainer.

        Args:
            world_model: WorldModel instance
            config: Training configuration
        """
        self.world_model = world_model
        self.config = config

        self.world_model.to(config.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.learning_rate,
        )

        # Statistics
        self.step = 0
        self.losses = {
            "total": [],
            "reconstruction": [],
            "kl": [],
            "transition": [],
            "reward": [],
            "value": [],
        }

    async def train(
        self,
        dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Train world model.

        Args:
            dataset: Training dataset
            validation_dataset: Validation dataset (optional)

        Returns:
            Dictionary of training losses
        """
        logger.info(
            f"Training world model for {self.config.num_epochs} epochs "
            f"on {len(dataset)} samples"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        for epoch in range(self.config.num_epochs):
            epoch_losses = await self._train_epoch(dataloader)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Loss: {epoch_losses['total']:.4f} "
                f"(recon: {epoch_losses['reconstruction']:.4f}, "
                f"kl: {epoch_losses['kl']:.4f}, "
                f"transition: {epoch_losses['transition']:.4f})"
            )

            # Validation
            if validation_dataset and (epoch + 1) % 10 == 0:
                val_losses = await self._validate(validation_dataset)
                logger.info(f"Validation loss: {val_losses['total']:.4f}")

            # Checkpoint
            if self.config.checkpoint_dir and (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

        return self.losses

    async def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch."""
        self.world_model.train()
        epoch_losses = {key: 0.0 for key in self.losses.keys()}
        num_batches = 0

        for batch in dataloader:
            # Unpack batch
            # Expected: (observations, actions, next_observations, rewards)
            if isinstance(batch, (tuple, list)) and len(batch) >= 4:
                observations, actions, next_observations, rewards = batch[:4]
            else:
                # Skip malformed batches
                continue

            observations = observations.to(self.config.device)
            actions = actions.to(self.config.device)
            next_observations = next_observations.to(self.config.device)
            rewards = rewards.to(self.config.device)

            # Compute losses
            losses = self._compute_losses(
                observations,
                actions,
                next_observations,
                rewards,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key].item()

            num_batches += 1
            self.step += 1

        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= max(1, num_batches)
            self.losses[key].append(epoch_losses[key])

        return epoch_losses

    def _compute_losses(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        # Encode current and next observations
        latent, mean, logstd = self.world_model.encode(observations)
        next_latent_true, next_mean, next_logstd = self.world_model.encode(next_observations)

        # Reconstruction loss
        reconstructed_obs = self.world_model.decode(latent)
        reconstruction_loss = F.mse_loss(reconstructed_obs, observations)

        # KL divergence loss (regularization)
        kl_loss = -0.5 * torch.sum(
            1 + 2 * logstd - mean.pow(2) - (2 * logstd).exp()
        ) / latent.size(0)

        # Transition loss
        if self.world_model.config.deterministic_transition:
            next_latent_pred = self.world_model.predict_next(latent, actions)
            transition_loss = F.mse_loss(next_latent_pred, next_latent_true)
        else:
            next_latent_pred, pred_mean, pred_logstd = self.world_model.predict_next(latent, actions)
            transition_loss = F.mse_loss(next_latent_pred, next_latent_true)

        # Reward prediction loss
        predicted_rewards = self.world_model.predict_reward(next_latent_true)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        # Value prediction loss (simplified - would use TD targets in practice)
        predicted_values = self.world_model.predict_value(latent)
        value_targets = rewards  # Simplified
        value_loss = F.mse_loss(predicted_values, value_targets)

        # Combine losses
        total_loss = (
            self.config.reconstruction_weight * reconstruction_loss +
            self.config.kl_weight * kl_loss +
            transition_loss +
            self.config.reward_weight * reward_loss +
            self.config.value_weight * value_loss
        )

        return {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "kl": kl_loss,
            "transition": transition_loss,
            "reward": reward_loss,
            "value": value_loss,
        }

    async def _validate(self, dataset: Dataset) -> Dict[str, float]:
        """Validate on validation set."""
        self.world_model.eval()
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        val_losses = {key: 0.0 for key in self.losses.keys()}
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 4:
                    observations, actions, next_observations, rewards = batch[:4]
                else:
                    continue

                observations = observations.to(self.config.device)
                actions = actions.to(self.config.device)
                next_observations = next_observations.to(self.config.device)
                rewards = rewards.to(self.config.device)

                losses = self._compute_losses(observations, actions, next_observations, rewards)

                for key in val_losses.keys():
                    val_losses[key] += losses[key].item()

                num_batches += 1

        # Average
        for key in val_losses.keys():
            val_losses[key] /= max(1, num_batches)

        return val_losses

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        if self.config.checkpoint_dir:
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = self.config.checkpoint_dir / f"world_model_epoch_{epoch+1}.pt"

            torch.save({
                "epoch": epoch,
                "model_state_dict": self.world_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "losses": self.losses,
            }, checkpoint_path)

            logger.info(f"Saved checkpoint: {checkpoint_path}")


# =============================================================================
# COUNTERFACTUAL REASONING
# =============================================================================

class CounterfactualReasoner:
    """
    Perform counterfactual reasoning with world model.

    Answers "what if" questions by simulating alternative scenarios.
    """

    def __init__(self, world_model: WorldModel):
        """Initialize counterfactual reasoner."""
        self.world_model = world_model
        self.world_model.eval()

    def what_if(
        self,
        observation: torch.Tensor,
        factual_actions: torch.Tensor,
        counterfactual_actions: torch.Tensor,
        horizon: int,
    ) -> Dict[str, Any]:
        """
        Compare factual vs counterfactual outcomes.

        Args:
            observation: Initial observation
            factual_actions: Actual action sequence
            counterfactual_actions: Alternative action sequence
            horizon: Planning horizon

        Returns:
            Dictionary comparing factual and counterfactual trajectories
        """
        with torch.no_grad():
            # Encode initial state
            initial_latent, _, _ = self.world_model.encode(observation)

            # Factual rollout
            factual_trajectory = self.world_model.imagine_rollout(
                initial_latent,
                factual_actions,
                horizon,
            )

            # Counterfactual rollout
            counterfactual_trajectory = self.world_model.imagine_rollout(
                initial_latent,
                counterfactual_actions,
                horizon,
            )

            # Compare
            factual_return = factual_trajectory["rewards"].sum(dim=1)
            counterfactual_return = counterfactual_trajectory["rewards"].sum(dim=1)

            improvement = counterfactual_return - factual_return

            return {
                "factual_trajectory": factual_trajectory,
                "counterfactual_trajectory": counterfactual_trajectory,
                "factual_return": factual_return,
                "counterfactual_return": counterfactual_return,
                "improvement": improvement,
                "better_action": "counterfactual" if improvement > 0 else "factual",
            }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Components
    "LatentEncoder",
    "LatentDecoder",
    "TransitionModel",
    "RewardModel",
    "ValueModel",
    # World Model
    "WorldModelConfig",
    "WorldModel",
    # Training
    "WorldModelTrainingConfig",
    "WorldModelTrainer",
    # Reasoning
    "CounterfactualReasoner",
]
