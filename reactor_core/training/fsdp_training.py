"""
FSDP (Fully Sharded Data Parallel) Training for JARVIS Reactor Core.

Enables training massive models across multiple GPUs/nodes with:
- Full parameter sharding across devices
- Mixed precision (BF16/FP16) training
- Gradient accumulation across shards
- CPU offloading for memory efficiency
- Activation checkpointing
- Communication optimization
- State dict consolidation for checkpointing

Based on:
- "ZeRO: Memory Optimizations for Deep Learning" (Rajbhandari et al., 2020)
- "PyTorch FSDP" (Meta AI, 2021+)
- "Megatron-LM" (NVIDIA, 2021)

USAGE:
    from reactor_core.training import FSDPTrainer, FSDPConfig

    config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        mixed_precision="bf16",
        cpu_offload=True,
    )

    trainer = FSDPTrainer(model, config)
    await trainer.train(dataset)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

logger = logging.getLogger(__name__)


# =============================================================================
# FSDP CONFIGURATION
# =============================================================================

class FSDPShardingStrategy(Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "FULL_SHARD"  # Shard parameters, gradients, and optimizer states
    SHARD_GRAD_OP = "SHARD_GRAD_OP"  # Shard gradients and optimizer states only
    NO_SHARD = "NO_SHARD"  # No sharding (DDP equivalent)
    HYBRID_SHARD = "HYBRID_SHARD"  # Shard within node, replicate across nodes
    HYBRID_SHARD_ZERO2 = "_HYBRID_SHARD_ZERO2"  # Hybrid with ZeRO-2


class FSDPMixedPrecisionPolicy(Enum):
    """Mixed precision policies."""

    BF16 = "bf16"  # BFloat16 (recommended for A100/H100)
    FP16 = "fp16"  # Float16
    FP32 = "fp32"  # Full precision
    BF16_WORKING = "bf16_working"  # BF16 for forward/backward, FP32 for params


@dataclass
class FSDPConfig:
    """
    Configuration for FSDP training.

    Attributes:
        sharding_strategy: How to shard the model
        mixed_precision: Mixed precision policy
        cpu_offload: Offload parameters to CPU when not in use
        activation_checkpointing: Use gradient checkpointing
        backward_prefetch: Prefetch strategy for gradients
        forward_prefetch: Prefetch next layer during forward
        limit_all_gathers: Limit concurrent all-gathers
        sync_module_states: Synchronize module states at init
        min_num_params: Minimum parameters to wrap a layer
        auto_wrap_policy: Auto-wrapping policy ('transformer', 'size', custom)
        transformer_layer_cls: Transformer layer class for auto-wrap
        use_orig_params: Use original parameter format
        ignored_modules: Modules to not wrap with FSDP
        device_id: CUDA device ID
        gradient_accumulation_steps: Number of gradient accumulation steps
        compile: Use torch.compile for optimization
    """

    sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD
    mixed_precision: FSDPMixedPrecisionPolicy = FSDPMixedPrecisionPolicy.BF16
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    backward_prefetch: str = "BACKWARD_PRE"  # or "BACKWARD_POST"
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    sync_module_states: bool = True
    min_num_params: int = 1e8  # 100M parameters
    auto_wrap_policy: str = "transformer"  # or "size"
    transformer_layer_cls: Optional[List[type]] = None
    use_orig_params: bool = True
    ignored_modules: List[nn.Module] = field(default_factory=list)
    device_id: Optional[int] = None
    gradient_accumulation_steps: int = 1
    compile: bool = False


# =============================================================================
# FSDP UTILITIES
# =============================================================================

def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """
    Initialize distributed training.

    Args:
        backend: Communication backend (nccl, gloo, mpi)
        init_method: Initialization method
        timeout_seconds: Timeout for operations

    Returns:
        Dictionary with rank, world_size, local_rank
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            timeout=timedelta(seconds=timeout_seconds),
        )

    # Get distributed info
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Initialized distributed training: "
        f"rank={rank}/{world_size}, local_rank={local_rank}, "
        f"backend={backend}"
    )

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "backend": backend,
    }


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def get_mixed_precision_policy(
    policy: FSDPMixedPrecisionPolicy,
) -> Optional[MixedPrecision]:
    """Get FSDP mixed precision policy."""

    if policy == FSDPMixedPrecisionPolicy.FP32:
        return None

    if policy == FSDPMixedPrecisionPolicy.BF16:
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    if policy == FSDPMixedPrecisionPolicy.FP16:
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    if policy == FSDPMixedPrecisionPolicy.BF16_WORKING:
        # BF16 for forward/backward, FP32 for parameters
        return MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    return None


def get_auto_wrap_policy(
    config: FSDPConfig,
    model: nn.Module,
) -> Optional[Callable]:
    """Get auto-wrapping policy for FSDP."""

    if config.auto_wrap_policy == "transformer":
        # Transformer-based wrapping
        if config.transformer_layer_cls is None:
            # Try to detect common transformer layers
            try:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                from transformers.models.gpt2.modeling_gpt2 import GPT2Block
                from transformers.models.opt.modeling_opt import OPTDecoderLayer

                transformer_cls = {
                    LlamaDecoderLayer,
                    GPT2Block,
                    OPTDecoderLayer,
                }
            except ImportError:
                logger.warning("Could not import transformer layers, using size-based policy")
                return size_based_auto_wrap_policy
        else:
            transformer_cls = set(config.transformer_layer_cls)

        return lambda module: transformer_auto_wrap_policy(
            module,
            transformer_layer_cls=transformer_cls,
        )

    elif config.auto_wrap_policy == "size":
        # Size-based wrapping
        return lambda module: size_based_auto_wrap_policy(
            module,
            min_num_params=config.min_num_params,
        )

    elif callable(config.auto_wrap_policy):
        # Custom policy
        return config.auto_wrap_policy

    return None


# =============================================================================
# FSDP TRAINER
# =============================================================================

class FSDPTrainer:
    """
    FSDP-based trainer for massive model training.

    Handles:
    - Model sharding across GPUs
    - Mixed precision training
    - Gradient accumulation
    - Checkpoint saving/loading
    - Communication optimization
    - Memory profiling

    Example:
        >>> config = FSDPConfig(
        ...     sharding_strategy=FSDPShardingStrategy.FULL_SHARD,
        ...     mixed_precision=FSDPMixedPrecisionPolicy.BF16,
        ...     cpu_offload=True,
        ...     activation_checkpointing=True,
        ... )
        >>>
        >>> trainer = FSDPTrainer(model, config)
        >>> await trainer.train(dataloader, num_epochs=3)
    """

    def __init__(
        self,
        model: nn.Module,
        config: FSDPConfig,
        optimizer_class: type = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize FSDP trainer.

        Args:
            model: Model to train
            config: FSDP configuration
            optimizer_class: Optimizer class
            optimizer_kwargs: Optimizer arguments
        """
        self.config = config
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Setup distributed
        self.dist_info = setup_distributed()
        self.rank = self.dist_info["rank"]
        self.world_size = self.dist_info["world_size"]
        self.local_rank = self.dist_info["local_rank"]

        # Device
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Wrap model with FSDP
        self.model = self._wrap_model_with_fsdp(model)

        # Optimizer
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )

        # Scheduler (optional, can be set later)
        self.scheduler = None

        # Statistics
        self.step = 0
        self.epoch = 0

        logger.info(
            f"[Rank {self.rank}] FSDP Trainer initialized: "
            f"world_size={self.world_size}, "
            f"sharding={config.sharding_strategy.value}, "
            f"mixed_precision={config.mixed_precision.value}"
        )

    def _wrap_model_with_fsdp(self, model: nn.Module) -> FSDP:
        """Wrap model with FSDP."""

        # Move model to device
        model = model.to(self.device)

        # Get sharding strategy
        sharding_strategy_map = {
            FSDPShardingStrategy.FULL_SHARD: ShardingStrategy.FULL_SHARD,
            FSDPShardingStrategy.SHARD_GRAD_OP: ShardingStrategy.SHARD_GRAD_OP,
            FSDPShardingStrategy.NO_SHARD: ShardingStrategy.NO_SHARD,
            FSDPShardingStrategy.HYBRID_SHARD: ShardingStrategy.HYBRID_SHARD,
            FSDPShardingStrategy.HYBRID_SHARD_ZERO2: ShardingStrategy._HYBRID_SHARD_ZERO2,
        }
        sharding_strategy = sharding_strategy_map[self.config.sharding_strategy]

        # Get mixed precision policy
        mixed_precision = get_mixed_precision_policy(self.config.mixed_precision)

        # CPU offload config
        cpu_offload = CPUOffload(offload_params=True) if self.config.cpu_offload else None

        # Backward prefetch
        backward_prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = backward_prefetch_map.get(self.config.backward_prefetch)

        # Auto-wrap policy
        auto_wrap_policy = get_auto_wrap_policy(self.config, model)

        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            forward_prefetch=self.config.forward_prefetch,
            limit_all_gathers=self.config.limit_all_gathers,
            sync_module_states=self.config.sync_module_states,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=self.config.use_orig_params,
            ignored_modules=self.config.ignored_modules,
            device_id=self.local_rank,
        )

        # Apply activation checkpointing if enabled
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing(fsdp_model)

        # Compile if enabled (PyTorch 2.0+)
        if self.config.compile:
            try:
                fsdp_model = torch.compile(fsdp_model)
                logger.info(f"[Rank {self.rank}] Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"[Rank {self.rank}] Failed to compile model: {e}")

        return fsdp_model

    def _apply_activation_checkpointing(self, model: FSDP):
        """Apply activation checkpointing to reduce memory."""

        # Get auto-wrap policy for checkpointing
        auto_wrap_policy = get_auto_wrap_policy(self.config, model)

        if auto_wrap_policy is None:
            logger.warning(
                f"[Rank {self.rank}] No auto-wrap policy, "
                "skipping activation checkpointing"
            )
            return

        # Apply checkpointing
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=lambda submodule: checkpoint_wrapper(
                submodule,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda submodule: auto_wrap_policy(submodule),
        )

        logger.info(f"[Rank {self.rank}] Activation checkpointing applied")

    async def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 1,
        log_interval: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train model with FSDP.

        Args:
            dataloader: Training dataloader
            num_epochs: Number of epochs
            eval_dataloader: Evaluation dataloader (optional)
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Save checkpoint every N epochs
            log_interval: Log every N steps

        Returns:
            Training metrics
        """

        logger.info(
            f"[Rank {self.rank}] Starting FSDP training: "
            f"epochs={num_epochs}, steps_per_epoch={len(dataloader)}"
        )

        metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }

        self.model.train()

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            epoch_loss = await self._train_epoch(
                dataloader,
                log_interval=log_interval,
            )

            metrics["train_loss"].append(epoch_loss)

            logger.info(
                f"[Rank {self.rank}] Epoch {epoch + 1}/{num_epochs}: "
                f"train_loss={epoch_loss:.4f}"
            )

            # Evaluation
            if eval_dataloader is not None:
                eval_loss = await self._evaluate(eval_dataloader)
                metrics["eval_loss"].append(eval_loss)

                logger.info(f"[Rank {self.rank}] Eval loss: {eval_loss:.4f}")

            # Checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                await self._save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1}",
                    epoch=epoch + 1,
                )

        logger.info(f"[Rank {self.rank}] Training complete")

        return metrics

    async def _train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 100,
    ) -> float:
        """Train one epoch."""

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            if isinstance(batch, (tuple, list)):
                batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch)
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = batch.to(self.device)

            # Forward pass
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(*batch)
                    targets = batch[-1]
            elif isinstance(batch, dict):
                outputs = self.model(**batch)
                targets = batch.get("labels", batch.get("target"))
            else:
                outputs = self.model(batch)
                targets = batch

            # Compute loss
            loss = self._compute_loss(outputs, targets)

            # Backward pass with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

                self.step += 1

            # Statistics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Logging
            if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]

                logger.info(
                    f"[Rank {self.rank}] Step {self.step}: "
                    f"loss={avg_loss:.4f}, lr={lr:.2e}"
                )

        return total_loss / max(1, num_batches)

    async def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model."""

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                if isinstance(batch, (tuple, list)):
                    batch = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch)
                else:
                    batch = batch.to(self.device)

                # Forward
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[:2]
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(batch)
                    targets = batch

                # Loss
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        self.model.train()

        return total_loss / max(1, num_batches)

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss (override for custom loss)."""

        # Default: cross-entropy
        if hasattr(outputs, "logits"):
            outputs = outputs.logits

        return nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
        )

    async def _save_checkpoint(
        self,
        checkpoint_path: Path,
        epoch: int,
    ):
        """Save FSDP checkpoint."""

        if self.rank != 0:
            return

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save full state dict (consolidated on rank 0)
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
        ):
            state_dict = self.model.state_dict()

            if self.rank == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "step": self.step,
                        "model_state_dict": state_dict,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                    },
                    checkpoint_path / "checkpoint.pt",
                )

                logger.info(
                    f"[Rank {self.rank}] Checkpoint saved: {checkpoint_path}"
                )

    def cleanup(self):
        """Cleanup distributed training."""
        cleanup_distributed()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "FSDPConfig",
    "FSDPShardingStrategy",
    "FSDPMixedPrecisionPolicy",
    # Trainer
    "FSDPTrainer",
    # Utilities
    "setup_distributed",
    "cleanup_distributed",
]
