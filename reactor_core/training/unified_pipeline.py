"""
Unified Training Pipeline - v73.0 PROJECT TRINITY Integration
==============================================================

End-to-end training pipeline that connects:
- Telemetry ingestion from JARVIS
- Dataset building from experiences
- LoRA/QLoRA fine-tuning
- GGUF export for llama.cpp
- Auto-deployment to J-Prime
- Trinity heartbeat integration

ARCHITECTURE:
    JARVIS Telemetry → TelemetryIngestor → JARVISDataset → AsyncTrainer
                                                              ↓
    J-Prime ← Auto-Deploy ← GGUF Export ← Trained Model ← Checkpoints

USAGE:
    pipeline = UnifiedTrainingPipeline()
    await pipeline.initialize()

    # Run full training cycle
    result = await pipeline.run_training_cycle(
        telemetry_dir=Path("~/.jarvis/telemetry"),
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    # Deploy to J-Prime
    await pipeline.deploy_to_jprime(result.model_path)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the unified training pipeline."""

    # Data collection
    telemetry_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "telemetry"
    )
    min_samples: int = 100
    max_samples: int = 50000
    min_confidence: float = 0.7
    include_failures: bool = False

    # Model
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_qlora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128

    # Training
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4

    # Output
    output_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "training" / "output"
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "training" / "checkpoints"
    )

    # Export
    export_gguf: bool = True
    gguf_quantization: str = "Q4_K_M"

    # Deployment
    jprime_models_dir: Optional[Path] = None
    auto_deploy: bool = True

    # Trinity
    trinity_enabled: bool = True
    heartbeat_interval: float = 5.0

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.telemetry_dir, str):
            self.telemetry_dir = Path(self.telemetry_dir).expanduser()
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir).expanduser()
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir).expanduser()
        if isinstance(self.jprime_models_dir, str):
            self.jprime_models_dir = Path(self.jprime_models_dir).expanduser()


class PipelineState(Enum):
    """Pipeline execution states."""
    IDLE = "idle"
    COLLECTING = "collecting"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EXPORTING = "exporting"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineProgress:
    """Track pipeline progress."""
    state: PipelineState = PipelineState.IDLE
    stage: str = ""
    message: str = ""
    progress_percent: float = 0.0
    samples_collected: int = 0
    samples_total: int = 0
    training_step: int = 0
    training_total_steps: int = 0
    training_loss: float = 0.0
    elapsed_seconds: float = 0.0
    eta_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "stage": self.stage,
            "message": self.message,
            "progress_percent": round(self.progress_percent, 2),
            "samples_collected": self.samples_collected,
            "samples_total": self.samples_total,
            "training_step": self.training_step,
            "training_total_steps": self.training_total_steps,
            "training_loss": round(self.training_loss, 4) if self.training_loss else 0,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "eta_seconds": round(self.eta_seconds, 2) if self.eta_seconds else None,
        }


@dataclass
class PipelineResult:
    """Result from a training pipeline run."""
    success: bool
    state: PipelineState
    model_path: Optional[Path] = None
    gguf_path: Optional[Path] = None
    adapter_path: Optional[Path] = None
    deployed_to: Optional[Path] = None
    samples_used: int = 0
    training_steps: int = 0
    final_loss: float = 0.0
    training_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "state": self.state.value,
            "model_path": str(self.model_path) if self.model_path else None,
            "gguf_path": str(self.gguf_path) if self.gguf_path else None,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "deployed_to": str(self.deployed_to) if self.deployed_to else None,
            "samples_used": self.samples_used,
            "training_steps": self.training_steps,
            "final_loss": round(self.final_loss, 4),
            "training_time_seconds": round(self.training_time_seconds, 2),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


# =============================================================================
# Trinity Integration
# =============================================================================

class TrinityHeartbeat:
    """
    Trinity heartbeat broadcaster for training pipeline.

    Writes pipeline state to ~/.jarvis/trinity/components/reactor_core_training.json
    """

    COMPONENTS_DIR = Path.home() / ".jarvis" / "trinity" / "components"

    def __init__(self, enabled: bool = True, interval: float = 5.0):
        self.enabled = enabled
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._progress: Optional[PipelineProgress] = None
        self._start_time = time.time()

    async def start(self, progress: PipelineProgress) -> None:
        """Start heartbeat broadcasting."""
        if not self.enabled:
            return

        self._progress = progress
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("[Trinity] Training pipeline heartbeat started")

    async def stop(self) -> None:
        """Stop heartbeat broadcasting."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[Trinity] Training pipeline heartbeat stopped")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat broadcast loop."""
        while self._running:
            try:
                await self._broadcast()
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Trinity] Heartbeat error: {e}")
                await asyncio.sleep(self.interval)

    async def _broadcast(self) -> None:
        """Broadcast current state."""
        if not self._progress:
            return

        self.COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
        state_file = self.COMPONENTS_DIR / "reactor_core_training.json"

        state = {
            "component_type": "reactor_core_training",
            "instance_id": f"training-{os.getpid()}",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "progress": self._progress.to_dict(),
        }

        # Use atomic write
        try:
            from reactor_core.orchestration.trinity_orchestrator import write_json_atomic
            write_json_atomic(state_file, state)
        except ImportError:
            # Fallback to regular write
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)


# =============================================================================
# Unified Pipeline
# =============================================================================

class UnifiedTrainingPipeline:
    """
    Unified training pipeline connecting all Reactor-Core components.

    This is the main entry point for running training from JARVIS experiences.

    Features:
        - Automatic telemetry collection and parsing
        - Dataset building with quality filtering
        - LoRA/QLoRA fine-tuning with progress tracking
        - GGUF export for llama.cpp
        - Auto-deployment to J-Prime
        - Trinity status integration
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ):
        """
        Initialize the unified training pipeline.

        Args:
            config: Pipeline configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or PipelineConfig()
        self.progress_callback = progress_callback

        # State
        self._progress = PipelineProgress()
        self._start_time: Optional[float] = None
        self._trinity = TrinityHeartbeat(
            enabled=self.config.trinity_enabled,
            interval=self.config.heartbeat_interval,
        )

        # Components (lazy loaded)
        self._ingestor = None
        self._dataset_builder = None
        self._trainer = None

    async def initialize(self) -> None:
        """Initialize pipeline components."""
        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[Pipeline] Initialized")

    async def run_training_cycle(
        self,
        telemetry_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> PipelineResult:
        """
        Run a complete training cycle.

        Args:
            telemetry_dir: Override telemetry directory
            model_name: Override base model
            since: Only use telemetry since this time
            until: Only use telemetry until this time

        Returns:
            PipelineResult with training outcomes
        """
        self._start_time = time.time()
        result = PipelineResult(success=False, state=PipelineState.FAILED)

        # Apply overrides
        if telemetry_dir:
            self.config.telemetry_dir = telemetry_dir
        if model_name:
            self.config.base_model = model_name

        try:
            # Start Trinity heartbeat
            await self._trinity.start(self._progress)

            # Step 1: Collect experiences
            await self._update_state(PipelineState.COLLECTING, "Collecting experiences")
            raw_interactions = await self._collect_experiences(since, until)

            if len(raw_interactions) < self.config.min_samples:
                raise ValueError(
                    f"Insufficient samples: {len(raw_interactions)} < {self.config.min_samples}"
                )

            result.samples_used = len(raw_interactions)
            self._progress.samples_collected = len(raw_interactions)

            # Step 2: Build dataset
            await self._update_state(PipelineState.PREPROCESSING, "Building dataset")
            train_dataset, eval_dataset = await self._build_dataset(raw_interactions)

            # Step 3: Train model
            await self._update_state(PipelineState.TRAINING, "Training model")

            # Voice announcement: Training started
            try:
                from reactor_core.voice_integration import announce_training_started
                asyncio.create_task(announce_training_started(
                    model_name=self.config.base_model.split("/")[-1],
                    samples=len(raw_interactions),
                    epochs=self.config.num_epochs,
                ))
            except Exception:
                pass

            training_result = await self._train_model(train_dataset, eval_dataset)

            if not training_result.success:
                # Voice announcement: Training failed
                try:
                    from reactor_core.voice_integration import announce_training_failed
                    asyncio.create_task(announce_training_failed(
                        model_name=self.config.base_model.split("/")[-1],
                        error_message=training_result.error_message or "Unknown error",
                        steps_completed=training_result.total_steps or 0,
                    ))
                except Exception:
                    pass

                raise RuntimeError(f"Training failed: {training_result.error_message}")

            result.adapter_path = training_result.adapter_path
            result.model_path = training_result.merged_model_path or training_result.adapter_path
            result.training_steps = training_result.total_steps
            result.final_loss = training_result.final_loss
            result.training_time_seconds = training_result.training_time_seconds

            # Voice announcement: Training complete
            try:
                from reactor_core.voice_integration import announce_training_complete
                asyncio.create_task(announce_training_complete(
                    model_name=self.config.base_model.split("/")[-1],
                    steps=training_result.total_steps or 0,
                    loss=training_result.final_loss or 0.0,
                    duration_seconds=training_result.training_time_seconds or 0.0,
                    success=True,
                ))
            except Exception:
                pass

            # Step 4: Export to GGUF (if configured)
            if self.config.export_gguf and result.model_path:
                await self._update_state(PipelineState.EXPORTING, "Exporting to GGUF")

                # Voice announcement: Export started
                try:
                    from reactor_core.voice_integration import announce_export_started
                    asyncio.create_task(announce_export_started(
                        format="GGUF",
                        quantization=self.config.gguf_quantization,
                    ))
                except Exception:
                    pass

                result.gguf_path = await self._export_gguf(result.model_path)

                # Voice announcement: Export complete
                if result.gguf_path:
                    try:
                        from reactor_core.voice_integration import announce_export_complete
                        file_size_mb = result.gguf_path.stat().st_size / (1024 * 1024) if result.gguf_path.exists() else None
                        asyncio.create_task(announce_export_complete(
                            format="GGUF",
                            file_size_mb=file_size_mb,
                        ))
                    except Exception:
                        pass

            # Step 5: Deploy to J-Prime (if configured)
            if self.config.auto_deploy and (result.gguf_path or result.model_path):
                await self._update_state(PipelineState.DEPLOYING, "Deploying to J-Prime")

                # Voice announcement: Deployment started
                try:
                    from reactor_core.voice_integration import announce_deployment_started
                    asyncio.create_task(announce_deployment_started(
                        target="JARVIS-Prime",
                    ))
                except Exception:
                    pass

                result.deployed_to = await self._deploy_to_jprime(
                    result.gguf_path or result.model_path
                )

                # Voice announcement: Deployment complete
                if result.deployed_to:
                    try:
                        from reactor_core.voice_integration import announce_deployment_complete
                        model_version = result.gguf_path.name if result.gguf_path else None
                        asyncio.create_task(announce_deployment_complete(
                            target="JARVIS-Prime",
                            model_version=model_version,
                        ))
                    except Exception:
                        pass

            # Success
            await self._update_state(PipelineState.COMPLETED, "Training cycle complete")
            result.success = True
            result.state = PipelineState.COMPLETED

            logger.info(
                f"[Pipeline] Training cycle complete: "
                f"{result.training_steps} steps, loss={result.final_loss:.4f}"
            )

        except Exception as e:
            import traceback
            await self._update_state(PipelineState.FAILED, f"Failed: {e}")
            result.error_message = str(e)
            result.metrics["traceback"] = traceback.format_exc()
            logger.error(f"[Pipeline] Training cycle failed: {e}")

        finally:
            await self._trinity.stop()
            result.total_time_seconds = time.time() - self._start_time

        return result

    async def _update_state(
        self,
        state: PipelineState,
        message: str,
        progress: float = None,
    ) -> None:
        """Update pipeline state and notify."""
        self._progress.state = state
        self._progress.message = message
        self._progress.stage = state.value

        if progress is not None:
            self._progress.progress_percent = progress

        if self._start_time:
            self._progress.elapsed_seconds = time.time() - self._start_time

        # Notify callback
        if self.progress_callback:
            try:
                self.progress_callback(self._progress)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

        logger.info(f"[Pipeline] {state.value}: {message}")

    async def _collect_experiences(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Any]:
        """Collect experiences from telemetry."""
        from reactor_core.ingestion.telemetry_ingestor import TelemetryIngestor

        ingestor = TelemetryIngestor(
            min_confidence=self.config.min_confidence,
            include_failures=self.config.include_failures,
        )

        interactions = []
        telemetry_files = list(self.config.telemetry_dir.glob("*.jsonl"))

        logger.info(f"[Pipeline] Found {len(telemetry_files)} telemetry files")

        for file_path in telemetry_files:
            async for interaction in ingestor.ingest(file_path, since=since, until=until):
                interactions.append(interaction)

                # Update progress
                self._progress.samples_collected = len(interactions)

                # Check max samples
                if len(interactions) >= self.config.max_samples:
                    break

            if len(interactions) >= self.config.max_samples:
                break

        logger.info(f"[Pipeline] Collected {len(interactions)} interactions")
        return interactions

    async def _build_dataset(
        self,
        interactions: List[Any],
    ) -> Tuple[Any, Any]:
        """Build HuggingFace datasets from interactions."""
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError("datasets required: pip install datasets")

        # Convert interactions to training format
        examples = []
        for interaction in interactions:
            if interaction.user_input and interaction.assistant_output:
                examples.append({
                    "messages": [
                        {"role": "user", "content": interaction.user_input},
                        {"role": "assistant", "content": interaction.assistant_output},
                    ],
                    "text": self._format_example(interaction),
                })

        if not examples:
            raise ValueError("No valid training examples found")

        logger.info(f"[Pipeline] Built {len(examples)} training examples")

        # Create dataset
        full_dataset = Dataset.from_list(examples)

        # Split into train/eval (90/10)
        split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        logger.info(
            f"[Pipeline] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}"
        )

        return train_dataset, eval_dataset

    def _format_example(self, interaction: Any) -> str:
        """Format interaction as training text."""
        # ChatML format
        parts = []

        if hasattr(interaction, "system_context") and interaction.system_context:
            parts.append(f"<|system|>\n{interaction.system_context}</s>")

        if interaction.user_input:
            parts.append(f"<|user|>\n{interaction.user_input}</s>")

        if interaction.assistant_output:
            parts.append(f"<|assistant|>\n{interaction.assistant_output}</s>")

        return "\n".join(parts)

    async def _train_model(
        self,
        train_dataset: Any,
        eval_dataset: Any,
    ) -> Any:
        """Run model training."""
        from reactor_core.training.trainer import AsyncTrainer, TrainingConfig

        # Create training config
        config = TrainingConfig(
            model_name=self.config.base_model,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            max_seq_length=self.config.max_seq_length,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            use_lora=True,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            use_qlora=self.config.use_qlora,
            output_dir=self.config.output_dir,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        # Create trainer with progress callback
        async def training_progress(progress):
            self._progress.training_step = progress.current_step
            self._progress.training_total_steps = progress.total_steps
            self._progress.training_loss = progress.loss
            self._progress.eta_seconds = progress.eta_seconds

            if progress.total_steps > 0:
                # Training is 40-90% of total progress
                train_pct = progress.current_step / progress.total_steps
                self._progress.progress_percent = 40 + (train_pct * 50)

            if self.progress_callback:
                self.progress_callback(self._progress)

        trainer = AsyncTrainer(
            config=config,
            progress_callback=training_progress,
        )

        # Run training
        result = await trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        return result

    async def _export_gguf(self, model_path: Path) -> Optional[Path]:
        """Export model to GGUF format using the reactor_core GGUF converter."""
        try:
            from reactor_core.quantization.gguf_converter import (
                GGUFConverter,
                GGUFConfig,
                QuantizationMethod,
            )

            # Map config string to QuantizationMethod enum
            quant_map = {
                "Q4_K_M": QuantizationMethod.Q4_K_M,
                "Q5_K_M": QuantizationMethod.Q5_K_M,
                "Q8_0": QuantizationMethod.Q8_0,
                "Q3_K_M": QuantizationMethod.Q3_K_M,
                "F16": QuantizationMethod.F16,
            }
            method = quant_map.get(
                self.config.gguf_quantization.upper(),
                QuantizationMethod.Q4_K_M,
            )

            # Configure GGUF converter
            gguf_config = GGUFConfig(
                method=method,
                output_dir=model_path.parent,
                output_name=f"{model_path.name}-{method.value}.gguf",
            )

            converter = GGUFConverter(gguf_config)

            # Run conversion
            logger.info(f"[Pipeline] Converting to GGUF ({method.value})...")

            result = await converter.convert(model_path)

            if result.success:
                logger.info(
                    f"[Pipeline] GGUF export complete: {result.output_path} "
                    f"({result.quantized_size_mb:.1f}MB, {result.compression_ratio:.1f}x compression)"
                )
                return result.output_path
            else:
                logger.error(f"[Pipeline] GGUF conversion failed: {result.error}")
                return None

        except ImportError as e:
            logger.warning(f"[Pipeline] GGUF converter not available: {e}")
            # Fall back to direct llama.cpp script (legacy behavior)
            return await self._export_gguf_legacy(model_path)
        except Exception as e:
            logger.error(f"[Pipeline] GGUF export error: {e}")
            return None

    async def _export_gguf_legacy(self, model_path: Path) -> Optional[Path]:
        """Legacy GGUF export using llama.cpp scripts directly."""
        try:
            gguf_path = model_path.parent / f"{model_path.name}.{self.config.gguf_quantization}.gguf"

            # Check if llama.cpp convert script exists
            convert_script = shutil.which("convert-hf-to-gguf.py")
            if not convert_script:
                common_paths = [
                    Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
                    Path.home() / ".local" / "llama.cpp" / "convert_hf_to_gguf.py",
                    Path("/usr/local/bin/convert-hf-to-gguf.py"),
                ]
                for path in common_paths:
                    if path.exists():
                        convert_script = str(path)
                        break

            if not convert_script:
                logger.warning("[Pipeline] GGUF conversion script not found")
                return None

            logger.info(f"[Pipeline] Converting to GGUF (legacy): {gguf_path}")

            proc = await asyncio.create_subprocess_exec(
                "python3", convert_script,
                str(model_path),
                "--outfile", str(gguf_path),
                "--outtype", self.config.gguf_quantization.lower().replace("_", "-"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"[Pipeline] GGUF conversion failed: {stderr.decode()}")
                return None

            logger.info(f"[Pipeline] GGUF export complete (legacy): {gguf_path}")
            return gguf_path

        except Exception as e:
            logger.error(f"[Pipeline] Legacy GGUF export error: {e}")
            return None

    async def _deploy_to_jprime(self, model_path: Path) -> Optional[Path]:
        """Deploy model to J-Prime."""
        try:
            # Determine J-Prime models directory
            jprime_dir = self.config.jprime_models_dir
            if not jprime_dir:
                # Try to find it
                candidates = [
                    Path.home() / "Documents" / "repos" / "jarvis-prime" / "models",
                    Path("../jarvis-prime/models"),
                    Path.home() / ".jarvis" / "models",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        jprime_dir = candidate
                        break

            if not jprime_dir:
                logger.warning("[Pipeline] J-Prime models directory not found")
                return None

            # Copy model
            dest_path = jprime_dir / model_path.name
            if model_path.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(model_path, dest_path)
            else:
                shutil.copy2(model_path, dest_path)

            # Update current.gguf symlink if it's a GGUF
            if model_path.suffix == ".gguf":
                current_link = jprime_dir / "current.gguf"
                if current_link.exists() or current_link.is_symlink():
                    current_link.unlink()
                current_link.symlink_to(dest_path.name)
                logger.info(f"[Pipeline] Updated current.gguf symlink")

            logger.info(f"[Pipeline] Deployed to J-Prime: {dest_path}")
            return dest_path

        except Exception as e:
            logger.error(f"[Pipeline] Deployment error: {e}")
            return None

    def get_progress(self) -> PipelineProgress:
        """Get current pipeline progress."""
        return self._progress


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_night_shift(
    telemetry_dir: Optional[Path] = None,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    auto_deploy: bool = True,
) -> PipelineResult:
    """
    Convenience function to run a night shift training cycle.

    This is the main entry point for scheduled training runs.

    Args:
        telemetry_dir: Directory containing telemetry JSONL files
        model_name: Base model to fine-tune
        auto_deploy: Whether to auto-deploy to J-Prime

    Returns:
        PipelineResult with training outcomes
    """
    config = PipelineConfig(
        base_model=model_name,
        auto_deploy=auto_deploy,
    )
    if telemetry_dir:
        config.telemetry_dir = Path(telemetry_dir).expanduser()

    pipeline = UnifiedTrainingPipeline(config)
    await pipeline.initialize()

    return await pipeline.run_training_cycle()


async def quick_finetune(
    data_file: Path,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    epochs: int = 1,
) -> PipelineResult:
    """
    Quick fine-tuning from a single data file.

    Args:
        data_file: JSONL file with training examples
        model_name: Base model
        epochs: Number of training epochs

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        base_model=model_name,
        num_epochs=epochs,
        auto_deploy=False,
        export_gguf=False,
    )

    # Create temp dir with the data file
    data_dir = data_file.parent
    config.telemetry_dir = data_dir
    config.min_samples = 1  # Allow small datasets

    pipeline = UnifiedTrainingPipeline(config)
    await pipeline.initialize()

    return await pipeline.run_training_cycle()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core classes
    "UnifiedTrainingPipeline",
    "PipelineConfig",
    "PipelineProgress",
    "PipelineResult",
    "PipelineState",
    # Trinity integration
    "TrinityHeartbeat",
    # Convenience functions
    "run_night_shift",
    "quick_finetune",
]
