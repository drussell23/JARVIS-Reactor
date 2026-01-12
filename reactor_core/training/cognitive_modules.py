"""
Cognitive Module Training Framework for JARVIS AGI System
==========================================================

Specialized training for modular cognitive capabilities:
- Planning Module: Goal decomposition and action sequencing
- Reasoning Module: Logical inference and problem solving
- Memory Module: Long-term and working memory management
- Perception Module: Multi-modal input processing
- Meta-Cognitive Module: Self-monitoring and adaptation

Each module is trained independently but can be composed for complex AGI behaviors.

Version: v81.0 (Phase 3 - Ultimate Scale)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Protocol, Union
from enum import Enum
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class CognitiveModuleType(Enum):
    """Types of cognitive modules."""
    PLANNING = "planning"
    REASONING = "reasoning"
    MEMORY = "memory"
    PERCEPTION = "perception"
    META_COGNITIVE = "meta_cognitive"
    ATTENTION = "attention"
    LANGUAGE = "language"
    MOTOR = "motor"


class PlanningStrategy(Enum):
    """Planning strategies for the Planning Module."""
    FORWARD_SEARCH = "forward_search"
    BACKWARD_CHAINING = "backward_chaining"
    HIERARCHICAL_TASK_NETWORK = "htn"
    MONTE_CARLO_TREE_SEARCH = "mcts"
    REINFORCEMENT_LEARNING = "rl"


class ReasoningType(Enum):
    """Types of reasoning."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


class MemoryType(Enum):
    """Types of memory systems."""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PROSPECTIVE = "prospective"


@dataclass
class CognitiveState:
    """Current state of cognitive processing."""
    module_activations: Dict[str, torch.Tensor]
    working_memory: List[torch.Tensor]
    attention_weights: torch.Tensor
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveModuleConfig:
    """Configuration for a cognitive module."""
    module_type: CognitiveModuleType
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 3
    dropout: float = 0.1
    use_attention: bool = True
    use_layer_norm: bool = True
    activation: str = "gelu"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    checkpoint_dir: Optional[Path] = None


@dataclass
class TrainingBatch:
    """Batch of cognitive training data."""
    inputs: torch.Tensor
    targets: torch.Tensor
    contexts: Optional[torch.Tensor] = None
    masks: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# BASE COGNITIVE MODULE
# ============================================================================

class BaseCognitiveModule(nn.Module):
    """
    Base class for all cognitive modules.

    Provides:
    - Standard architecture (attention, layer norm, residual connections)
    - Training interface
    - State management
    - Checkpointing
    """

    def __init__(self, config: CognitiveModuleConfig):
        super().__init__()
        self.config = config
        self.module_type = config.module_type

        # Core layers
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)

        # Multi-layer transformer-style processing
        self.layers = nn.ModuleList([
            CognitiveLayer(
                dim=config.hidden_dim,
                use_attention=config.use_attention,
                dropout=config.dropout,
                activation=config.activation,
            )
            for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)

        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.layer_norm = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # State tracking
        self.current_state: Optional[CognitiveState] = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, CognitiveState]]:
        """
        Forward pass through the cognitive module.

        Args:
            x: Input tensor [batch, seq_len, input_dim]
            context: Optional context tensor [batch, ctx_len, hidden_dim]
            mask: Optional attention mask [batch, seq_len]
            return_state: Whether to return cognitive state

        Returns:
            Output tensor [batch, seq_len, output_dim] and optionally state
        """
        # Project input
        h = self.input_projection(x)
        h = self.layer_norm(h)

        # Process through layers
        attention_weights = []
        for layer in self.layers:
            h, attn = layer(h, context=context, mask=mask)
            if attn is not None:
                attention_weights.append(attn)

        # Output projection
        output = self.output_projection(h)
        output = self.dropout(output)

        # Create cognitive state if requested
        if return_state:
            state = CognitiveState(
                module_activations={self.module_type.value: h},
                working_memory=[h],
                attention_weights=torch.stack(attention_weights) if attention_weights else None,
                confidence=self._compute_confidence(output),
                timestamp=torch.cuda.Event().elapsed_time(torch.cuda.Event()) if torch.cuda.is_available() else 0.0,
            )
            self.current_state = state
            return output, state

        return output

    def _compute_confidence(self, output: torch.Tensor) -> float:
        """Compute confidence score for the output."""
        # Use entropy of output distribution as confidence
        probs = F.softmax(output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = np.log(output.size(-1))
        confidence = 1.0 - (entropy.mean().item() / max_entropy)
        return float(np.clip(confidence, 0.0, 1.0))

    def save_checkpoint(self, path: Path):
        """Save module checkpoint."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'current_state': self.current_state,
        }, path)
        logger.info(f"Saved {self.module_type.value} module checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load module checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.current_state = checkpoint.get('current_state')
        logger.info(f"Loaded {self.module_type.value} module checkpoint from {path}")


class CognitiveLayer(nn.Module):
    """Single layer of cognitive processing."""

    def __init__(
        self,
        dim: int,
        use_attention: bool = True,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(dim)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through cognitive layer."""
        attn_weights = None

        # Self-attention or cross-attention
        if self.use_attention:
            if context is not None:
                # Cross-attention with context
                attn_out, attn_weights = self.attention(
                    query=x,
                    key=context,
                    value=context,
                    key_padding_mask=mask,
                )
            else:
                # Self-attention
                attn_out, attn_weights = self.attention(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=mask,
                )
            # Residual connection
            x = self.attn_norm(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)

        return x, attn_weights


# ============================================================================
# SPECIALIZED COGNITIVE MODULES
# ============================================================================

class PlanningModule(BaseCognitiveModule):
    """
    Planning Module for goal decomposition and action sequencing.

    Capabilities:
    - Hierarchical goal decomposition
    - Action sequence generation
    - Plan verification and adaptation
    - Multi-step lookahead
    """

    def __init__(self, config: CognitiveModuleConfig):
        config.module_type = CognitiveModuleType.PLANNING
        super().__init__(config)

        # Goal encoder
        self.goal_encoder = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Action sequence decoder
        self.action_decoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
        )

        # Plan quality estimator
        self.quality_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def plan(
        self,
        goal: torch.Tensor,
        current_state: torch.Tensor,
        max_steps: int = 10,
        strategy: PlanningStrategy = PlanningStrategy.FORWARD_SEARCH,
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Generate action plan to achieve goal.

        Args:
            goal: Goal representation [batch, goal_dim]
            current_state: Current state [batch, state_dim]
            max_steps: Maximum planning steps
            strategy: Planning strategy to use

        Returns:
            List of actions and plan quality score
        """
        # Encode goal
        goal_encoding = self.goal_encoder(goal)

        # Initialize decoder
        hidden = goal_encoding.unsqueeze(0).repeat(2, 1, 1)
        input_state = current_state.unsqueeze(1)

        # Generate action sequence
        actions = []
        for _ in range(max_steps):
            output, hidden = self.action_decoder(input_state, hidden)
            action = self.output_projection(output.squeeze(1))
            actions.append(action)

            # Use predicted action as next input
            input_state = output

        # Estimate plan quality
        final_hidden = hidden[-1]
        quality = self.quality_head(final_hidden).item()

        return actions, quality


class ReasoningModule(BaseCognitiveModule):
    """
    Reasoning Module for logical inference and problem solving.

    Capabilities:
    - Deductive reasoning (rules → conclusions)
    - Inductive reasoning (examples → patterns)
    - Abductive reasoning (observations → explanations)
    - Analogical reasoning (transfer from similar problems)
    """

    def __init__(self, config: CognitiveModuleConfig):
        config.module_type = CognitiveModuleType.REASONING
        super().__init__(config)

        # Reasoning type selector
        self.reasoning_type_head = nn.Linear(config.hidden_dim, len(ReasoningType))

        # Logic encoder
        self.logic_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=3,
        )

        # Inference head
        self.inference_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def reason(
        self,
        premises: torch.Tensor,
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
    ) -> Tuple[torch.Tensor, float]:
        """
        Perform reasoning over premises.

        Args:
            premises: Input premises [batch, num_premises, premise_dim]
            reasoning_type: Type of reasoning to perform

        Returns:
            Conclusion tensor and confidence score
        """
        # Encode premises
        encoded = self.logic_encoder(premises)

        # Pooling across premises
        pooled = encoded.mean(dim=1)

        # Generate conclusion
        conclusion = self.inference_head(pooled)

        # Compute confidence
        confidence = self._compute_confidence(conclusion)

        return conclusion, confidence


class MemoryModule(BaseCognitiveModule):
    """
    Memory Module for long-term and working memory management.

    Capabilities:
    - Working memory buffer (limited capacity)
    - Episodic memory (experiences)
    - Semantic memory (facts and concepts)
    - Memory consolidation and retrieval
    """

    def __init__(self, config: CognitiveModuleConfig, memory_capacity: int = 1000):
        config.module_type = CognitiveModuleType.MEMORY
        super().__init__(config)

        self.memory_capacity = memory_capacity

        # Memory buffers
        self.episodic_memory: List[torch.Tensor] = []
        self.semantic_memory: Dict[str, torch.Tensor] = {}
        self.working_memory: List[torch.Tensor] = []

        # Memory encoder/decoder
        self.memory_encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.memory_decoder = nn.Linear(config.hidden_dim, config.output_dim)

        # Retrieval attention
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True,
        )

        # Importance scorer
        self.importance_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def store(self, memory: torch.Tensor, memory_type: MemoryType = MemoryType.EPISODIC):
        """Store memory with importance-based consolidation."""
        encoded = self.memory_encoder(memory)
        importance = self.importance_head(encoded).item()

        if memory_type == MemoryType.EPISODIC:
            self.episodic_memory.append(encoded)

            # Consolidate if capacity exceeded
            if len(self.episodic_memory) > self.memory_capacity:
                self._consolidate_episodic_memory()

        elif memory_type == MemoryType.WORKING:
            self.working_memory.append(encoded)

            # Working memory has limited capacity (e.g., 7±2 items)
            if len(self.working_memory) > 9:
                self.working_memory.pop(0)

    def retrieve(
        self,
        query: torch.Tensor,
        memory_type: MemoryType = MemoryType.EPISODIC,
        k: int = 5,
    ) -> List[torch.Tensor]:
        """Retrieve relevant memories using attention."""
        query_encoded = self.memory_encoder(query).unsqueeze(0)

        # Select memory buffer
        if memory_type == MemoryType.EPISODIC:
            memory_buffer = self.episodic_memory
        elif memory_type == MemoryType.WORKING:
            memory_buffer = self.working_memory
        else:
            return []

        if not memory_buffer:
            return []

        # Stack memories for attention
        memories = torch.stack(memory_buffer).unsqueeze(0)

        # Retrieve using attention
        retrieved, _ = self.retrieval_attention(
            query=query_encoded,
            key=memories,
            value=memories,
        )

        # Return top-k memories
        return [self.memory_decoder(retrieved[0, i]) for i in range(min(k, retrieved.size(1)))]

    def _consolidate_episodic_memory(self):
        """Consolidate episodic memory by removing least important items."""
        if not self.episodic_memory:
            return

        # Score importance
        memories = torch.stack(self.episodic_memory)
        importance_scores = self.importance_head(memories).squeeze()

        # Keep top important memories
        keep_indices = torch.topk(importance_scores, k=self.memory_capacity).indices
        self.episodic_memory = [self.episodic_memory[i] for i in keep_indices]


class PerceptionModule(BaseCognitiveModule):
    """
    Perception Module for multi-modal input processing.

    Capabilities:
    - Vision processing
    - Audio processing
    - Text processing
    - Multi-modal fusion
    """

    def __init__(self, config: CognitiveModuleConfig):
        config.module_type = CognitiveModuleType.PERCEPTION
        super().__init__(config)

        # Modality-specific encoders
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, config.hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, config.hidden_dim, kernel_size=7, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.text_encoder = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Multi-modal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True,
        )

    def perceive(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process multi-modal inputs."""
        modalities = []

        if vision is not None:
            vision_feat = self.vision_encoder(vision)
            modalities.append(vision_feat.unsqueeze(1))

        if audio is not None:
            audio_feat = self.audio_encoder(audio)
            modalities.append(audio_feat.unsqueeze(1))

        if text is not None:
            text_feat, _ = self.text_encoder(text)
            # Use final hidden state
            text_feat = text_feat[:, -1, :self.config.hidden_dim]
            modalities.append(text_feat.unsqueeze(1))

        if not modalities:
            raise ValueError("At least one modality must be provided")

        # Concatenate modalities
        multi_modal = torch.cat(modalities, dim=1)

        # Fuse with attention
        fused, _ = self.fusion(multi_modal, multi_modal, multi_modal)

        # Pool across modalities
        output = fused.mean(dim=1)

        return output


# ============================================================================
# COGNITIVE ORCHESTRATOR
# ============================================================================

class CognitiveOrchestrator(nn.Module):
    """
    Orchestrates multiple cognitive modules for complex AGI behaviors.

    Manages:
    - Module activation scheduling
    - Inter-module communication
    - Resource allocation
    - Conflict resolution
    """

    def __init__(
        self,
        modules: Dict[CognitiveModuleType, BaseCognitiveModule],
        coordination_dim: int = 512,
    ):
        super().__init__()
        self.modules = nn.ModuleDict({
            mod_type.value: module for mod_type, module in modules.items()
        })

        # Module coordinator
        self.coordinator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=coordination_dim,
                nhead=8,
                dim_feedforward=coordination_dim * 4,
                batch_first=True,
            ),
            num_layers=3,
        )

        # Module activation gating
        self.activation_gates = nn.ModuleDict({
            name: nn.Linear(coordination_dim, 1)
            for name in self.modules.keys()
        })

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        active_modules: Optional[List[CognitiveModuleType]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute cognitive processing through orchestrated modules.

        Args:
            inputs: Dictionary of inputs for each module
            active_modules: Modules to activate (None = all)

        Returns:
            Dictionary of outputs from each module
        """
        if active_modules is None:
            active_modules = list(CognitiveModuleType)

        # Collect module outputs
        outputs = {}
        module_states = []

        for mod_type in active_modules:
            mod_name = mod_type.value
            if mod_name in self.modules and mod_name in inputs:
                module = self.modules[mod_name]
                output, state = module(inputs[mod_name], return_state=True)
                outputs[mod_name] = output
                module_states.append(state.module_activations[mod_name])

        # Coordinate modules if multiple active
        if len(module_states) > 1:
            stacked_states = torch.stack(module_states, dim=1)
            coordinated = self.coordinator(stacked_states)

            # Apply gating
            for i, (mod_type, mod_name) in enumerate(zip(active_modules, outputs.keys())):
                gate = torch.sigmoid(self.activation_gates[mod_name](coordinated[:, i]))
                outputs[mod_name] = outputs[mod_name] * gate

        return outputs


# ============================================================================
# TRAINING
# ============================================================================

@dataclass
class CognitiveTrainingConfig:
    """Configuration for cognitive module training."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    eval_every: int = 10
    checkpoint_every: int = 20
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True  # Automatic Mixed Precision


class CognitiveModuleTrainer:
    """Trainer for cognitive modules."""

    def __init__(
        self,
        module: BaseCognitiveModule,
        config: CognitiveTrainingConfig,
    ):
        self.module = module.to(config.device)
        self.config = config
        self.device = config.device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    async def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the cognitive module.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Custom loss function (default: MSE)

        Returns:
            Dictionary with training history
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        logger.info(f"Training {self.module.module_type.value} module for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_loss = await self._train_epoch(train_loader, loss_fn)
            self.train_losses.append(train_loss)

            # Validation phase
            if val_loader and (epoch + 1) % self.config.eval_every == 0:
                val_loss = await self._validate(val_loader, loss_fn)
                self.val_losses.append(val_loss)

                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    # Save best checkpoint
                    if self.module.config.checkpoint_dir:
                        checkpoint_path = self.module.config.checkpoint_dir / "best_model.pt"
                        self.module.save_checkpoint(checkpoint_path)
                else:
                    self.patience_counter += 1

                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.4f}")

            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                if self.module.config.checkpoint_dir:
                    checkpoint_path = self.module.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                    self.module.save_checkpoint(checkpoint_path)

            # Step scheduler
            self.scheduler.step()

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

    async def _train_epoch(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        """Train for one epoch."""
        self.module.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move to device
            inputs = batch.inputs.to(self.device)
            targets = batch.targets.to(self.device)
            context = batch.contexts.to(self.device) if batch.contexts is not None else None
            mask = batch.masks.to(self.device) if batch.masks is not None else None

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.module(inputs, context=context, mask=mask)
                    loss = loss_fn(outputs, targets)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.config.gradient_clip)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.module(inputs, context=context, mask=mask)
                loss = loss_fn(outputs, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.config.gradient_clip)

                # Optimizer step
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    async def _validate(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        """Validate the module."""
        self.module.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch.inputs.to(self.device)
                targets = batch.targets.to(self.device)
                context = batch.contexts.to(self.device) if batch.contexts is not None else None
                mask = batch.masks.to(self.device) if batch.masks is not None else None

                outputs = self.module(inputs, context=context, mask=mask)
                loss = loss_fn(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches


# ============================================================================
# UTILITIES
# ============================================================================

def create_cognitive_system(
    input_dim: int = 512,
    hidden_dim: int = 1024,
    output_dim: int = 512,
) -> CognitiveOrchestrator:
    """
    Create a complete cognitive system with all modules.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension

    Returns:
        CognitiveOrchestrator with all modules initialized
    """
    # Create module configs
    base_config = CognitiveModuleConfig(
        module_type=CognitiveModuleType.PLANNING,  # Will be overridden
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    # Initialize modules
    modules = {
        CognitiveModuleType.PLANNING: PlanningModule(base_config),
        CognitiveModuleType.REASONING: ReasoningModule(base_config),
        CognitiveModuleType.MEMORY: MemoryModule(base_config),
        CognitiveModuleType.PERCEPTION: PerceptionModule(base_config),
    }

    # Create orchestrator
    orchestrator = CognitiveOrchestrator(modules, coordination_dim=hidden_dim)

    logger.info(f"Created cognitive system with {len(modules)} modules")
    return orchestrator


__all__ = [
    # Enums
    "CognitiveModuleType",
    "PlanningStrategy",
    "ReasoningType",
    "MemoryType",
    # Data structures
    "CognitiveState",
    "CognitiveModuleConfig",
    "TrainingBatch",
    # Base module
    "BaseCognitiveModule",
    "CognitiveLayer",
    # Specialized modules
    "PlanningModule",
    "ReasoningModule",
    "MemoryModule",
    "PerceptionModule",
    # Orchestrator
    "CognitiveOrchestrator",
    # Training
    "CognitiveTrainingConfig",
    "CognitiveModuleTrainer",
    # Utilities
    "create_cognitive_system",
]
