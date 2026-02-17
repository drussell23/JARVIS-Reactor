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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Singleton Pattern for Trinity Integration
# =============================================================================

_unified_trainer_instance: Optional["UnifiedTrainingPipeline"] = None
_unified_trainer_lock: threading.Lock = threading.Lock()


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
    VALIDATING = "validating"  # v2.1: Deployment gate validation
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    GATE_REJECTED = "gate_rejected"  # v2.1: Model failed deployment gate


class _GateRejectionError(Exception):
    """Internal sentinel to short-circuit the pipeline when the deployment gate rejects a model.

    This is NOT a real failure — it's a controlled flow to skip deployment steps
    after the gate has already set the appropriate result fields. Caught in the
    outer try/except of run_training_cycle and _run_training_from_interactions.
    """
    pass


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
    gate_summary: Optional[str] = None  # v2.1: Deployment gate result summary
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
            "gate_summary": self.gate_summary,
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

        # Experience buffer for real-time data ingestion
        self._experience_buffer: List[Dict[str, Any]] = []
        self._experience_lock = asyncio.Lock()
        self._flush_in_progress: bool = False
        self._buffer_flush_threshold = int(
            os.getenv("REACTOR_EXPERIENCE_BUFFER_THRESHOLD", "100")
        )

        # v242.3: Strong references to background tasks to prevent GC collection.
        # Without this, fire-and-forget tasks created with asyncio.create_task()
        # can be garbage-collected before completion, silently dropping work.
        self._background_tasks: set = set()

    async def add_experiences(
        self,
        experiences: Union[Dict[str, Any], List[Dict[str, Any]]],
        flush: bool = False,
    ) -> int:
        """
        Add experiences to the training buffer for real-time data ingestion.

        This method enables the Trinity Loop by allowing JARVIS to send
        experiences directly to the training pipeline without file I/O.

        Args:
            experiences: Single experience dict or list of experience dicts.
                Each experience should have at minimum:
                - user_input: str
                - assistant_output: str
                Optional fields:
                - system_context: str
                - confidence: float (0-1)
                - timestamp: str (ISO format)
                - session_id: str
                - metadata: dict
            flush: If True, trigger an immediate training cycle if buffer
                   exceeds threshold.

        Returns:
            Current buffer size after adding experiences.

        Example:
            >>> trainer = get_unified_trainer()
            >>> await trainer.add_experiences({
            ...     "user_input": "What's the weather?",
            ...     "assistant_output": "It's sunny today.",
            ...     "confidence": 0.95
            ... })
        """
        # Normalize to list
        if isinstance(experiences, dict):
            experiences = [experiences]

        # v242.1: Validate via canonical schema (Pydantic) at the pipeline boundary.
        # This catches malformed data (wrong types, missing fields, null values)
        # BEFORE it enters the training buffer where it could corrupt a training run.
        valid_experiences = []
        _schema_available = False
        try:
            from reactor_core.schemas.experience_schema import ExperienceEvent, from_raw_dict
            _schema_available = True
        except ImportError:
            try:
                import sys
                schema_dir = str(Path.home() / ".jarvis" / "schemas")
                if schema_dir not in sys.path:
                    sys.path.insert(0, schema_dir)
                from experience_schema import ExperienceEvent, from_raw_dict
                _schema_available = True
            except ImportError:
                pass

        for exp in experiences:
            if not isinstance(exp, dict):
                logger.warning(f"Skipping non-dict experience: {type(exp)}")
                continue

            try:
                if _schema_available:
                    # Full Pydantic validation — rejects bad types, nulls, missing fields
                    canonical = from_raw_dict(exp)
                    validated = canonical.to_reactor_core_format()
                else:
                    # Fallback: manual field check
                    validated = exp
                    if not exp.get("user_input") and not exp.get("input") and not exp.get("query"):
                        logger.debug("Skipping experience missing user_input")
                        continue
                    if not exp.get("assistant_output") and not exp.get("response"):
                        logger.debug("Skipping experience missing assistant_output")
                        continue
            except Exception as val_err:
                logger.warning(f"[Pipeline] Rejecting malformed experience at boundary: {val_err}")
                continue

            # Require minimum fields (post-normalization)
            if not validated.get("user_input") or not validated.get("assistant_output"):
                logger.warning("[Pipeline] Rejecting experience: missing user_input or assistant_output")
                continue

            # Check confidence threshold
            confidence = validated.get("confidence", 1.0)
            if confidence < self.config.min_confidence:
                logger.debug(f"Skipping low-confidence experience: {confidence}")
                continue

            # Add timestamp if missing
            if "timestamp" not in validated:
                validated["timestamp"] = datetime.now().isoformat()

            valid_experiences.append(validated)

        if not valid_experiences:
            logger.debug("No valid experiences to add")
            return len(self._experience_buffer)

        # Thread-safe buffer append with proper race condition prevention
        should_flush = False
        async with self._experience_lock:
            self._experience_buffer.extend(valid_experiences)
            buffer_size = len(self._experience_buffer)

            # v242.0: Check threshold AND flush-in-progress flag while holding lock
            # to prevent duplicate training jobs (the old code released the lock
            # before creating the task, allowing a second coroutine to also trigger)
            if flush and buffer_size >= self._buffer_flush_threshold:
                if not self._flush_in_progress:
                    self._flush_in_progress = True
                    should_flush = True

        logger.info(f"[Pipeline] Added {len(valid_experiences)} experiences (buffer: {buffer_size})")

        # Trigger training outside the lock but only if we won the race
        if should_flush:
            logger.info(f"[Pipeline] Buffer threshold reached ({buffer_size}), scheduling training")
            task = self._spawn_background_task(
                self._flush_experiences_to_training(),
                name="experience_flush_training",
            )
            # v242.0: Clear flush-in-progress when task completes
            task.add_done_callback(lambda _: setattr(self, '_flush_in_progress', False))

        return buffer_size

    def _handle_flush_task_result(self, task: asyncio.Task) -> None:
        """Handle completion of flush task, logging any errors."""
        try:
            exc = task.exception()
            if exc:
                logger.error(f"[Pipeline] Flush task failed: {exc}")
        except asyncio.CancelledError:
            logger.debug("[Pipeline] Flush task was cancelled")
        except asyncio.InvalidStateError:
            pass  # Task not done yet

    def _spawn_background_task(self, coro, *, name: str = "") -> asyncio.Task:
        """Create a background task with error logging and GC protection.

        All fire-and-forget coroutines should go through this method instead
        of bare ``asyncio.create_task()`` to ensure:
        1. Strong reference prevents garbage collection before completion.
        2. Done callback logs exceptions instead of silently swallowing them.
        3. Reference is auto-cleaned on completion to avoid memory leak.
        """
        task = asyncio.create_task(coro, name=name or None)
        self._background_tasks.add(task)

        def _on_done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            try:
                exc = t.exception()
                if exc:
                    logger.warning(f"[Pipeline] Background task '{t.get_name()}' failed: {exc}")
            except asyncio.CancelledError:
                pass
            except asyncio.InvalidStateError:
                pass

        task.add_done_callback(_on_done)
        return task

    async def get_buffered_experiences(self) -> List[Dict[str, Any]]:
        """
        Get a copy of the current experience buffer without clearing it.

        Returns:
            List of buffered experiences.
        """
        async with self._experience_lock:
            return list(self._experience_buffer)

    async def clear_experience_buffer(self) -> int:
        """
        Clear the experience buffer and return the number of cleared items.

        Returns:
            Number of experiences that were cleared.
        """
        async with self._experience_lock:
            count = len(self._experience_buffer)
            self._experience_buffer.clear()
            return count

    async def _flush_experiences_to_training(self) -> Optional["PipelineResult"]:
        """
        Flush buffered experiences to training.

        This converts buffered experiences to the format expected by
        _build_dataset and triggers a training cycle.
        """
        async with self._experience_lock:
            if len(self._experience_buffer) < self.config.min_samples:
                logger.info(
                    f"[Pipeline] Not enough samples for training: "
                    f"{len(self._experience_buffer)} < {self.config.min_samples}"
                )
                return None

            # Move buffer to local variable and clear
            experiences = self._experience_buffer.copy()
            self._experience_buffer.clear()

        logger.info(f"[Pipeline] Flushing {len(experiences)} experiences to training")

        # Convert to interaction format expected by _build_dataset
        class ExperienceInteraction:
            """Adapter class to match TelemetryIngestor output format."""
            def __init__(self, exp: Dict[str, Any]):
                self.user_input = exp.get("user_input", "")
                self.assistant_output = exp.get("assistant_output", "")
                self.system_context = exp.get("system_context", "")
                self.confidence = exp.get("confidence", 1.0)
                self.timestamp = exp.get("timestamp", datetime.now().isoformat())
                self.session_id = exp.get("session_id", "")
                self.metadata = exp.get("metadata", {})

        interactions = [ExperienceInteraction(exp) for exp in experiences]

        # Run training cycle with these interactions
        return await self._run_training_from_interactions(interactions)

    async def _run_training_from_interactions(
        self,
        interactions: List[Any],
    ) -> "PipelineResult":
        """
        Run training cycle from pre-processed interactions.

        This is an internal method that bypasses telemetry collection
        since we already have the interactions.
        """
        self._start_time = time.time()
        result = PipelineResult(success=False, state=PipelineState.FAILED)

        try:
            await self._trinity.start(self._progress)

            result.samples_used = len(interactions)
            self._progress.samples_collected = len(interactions)

            # Skip collection, go straight to preprocessing
            await self._update_state(PipelineState.PREPROCESSING, "Building dataset from experiences")
            train_dataset, eval_dataset = await self._build_dataset(interactions)

            # Continue with normal training flow
            await self._update_state(PipelineState.TRAINING, "Training model")

            training_result = await self._train_model(train_dataset, eval_dataset)

            if not training_result.success:
                raise RuntimeError(f"Training failed: {training_result.error_message}")

            result.adapter_path = training_result.adapter_path
            result.model_path = training_result.merged_model_path or training_result.adapter_path
            result.training_steps = training_result.total_steps
            result.final_loss = training_result.final_loss
            result.training_time_seconds = training_result.training_time_seconds

            # Export and deploy as normal
            if self.config.export_gguf and result.model_path:
                await self._update_state(PipelineState.EXPORTING, "Exporting to GGUF")
                result.gguf_path = await self._export_gguf(result.model_path)

            # v2.1: Deployment gate — validate GGUF before deployment
            if result.gguf_path:
                await self._update_state(PipelineState.VALIDATING, "Validating GGUF via deployment gate")
                try:
                    from reactor_core.deployment.gate import DeploymentGate

                    gate = DeploymentGate(
                        skip_inference_check=bool(os.getenv("REACTOR_SKIP_INFERENCE_CHECK", "")),
                    )
                    gate_result = await gate.validate(result.gguf_path)
                    result.gate_summary = gate_result.summary()

                    if not gate_result.passed:
                        logger.error(f"[Pipeline] Deployment gate REJECTED: {gate_result.summary()}")
                        failed_dir = self.config.output_dir / "failed"
                        failed_dir.mkdir(parents=True, exist_ok=True)
                        failed_dest = failed_dir / result.gguf_path.name
                        result.gguf_path.rename(failed_dest)
                        logger.info(f"[Pipeline] Rejected GGUF moved to {failed_dest}")

                        result.gguf_path = None
                        result.metrics["gate_rejected"] = True
                        result.metrics["gate_summary"] = gate_result.summary()
                        result.metrics["gate_failed_path"] = str(failed_dest)

                        await self._update_state(
                            PipelineState.GATE_REJECTED,
                            f"GGUF failed deployment gate: {gate_result.summary()}",
                        )
                        result.success = False
                        result.state = PipelineState.GATE_REJECTED
                        result.error_message = f"Deployment gate rejected: {gate_result.summary()}"
                        raise _GateRejectionError(gate_result.summary())
                    else:
                        logger.info(f"[Pipeline] Deployment gate APPROVED: {gate_result.summary()}")
                        result.metrics["gate_approved"] = True
                        result.metrics["gate_summary"] = gate_result.summary()
                except _GateRejectionError:
                    raise
                except ImportError:
                    logger.warning("[Pipeline] DeploymentGate not available — skipping validation")
                except Exception as gate_err:
                    logger.warning(f"[Pipeline] Deployment gate error (non-blocking): {gate_err}")

            if self.config.auto_deploy and (result.gguf_path or result.model_path):
                await self._update_state(PipelineState.DEPLOYING, "Deploying to J-Prime")
                result.deployed_to = await self._deploy_to_jprime(
                    result.gguf_path or result.model_path
                )

            await self._update_state(PipelineState.COMPLETED, "Training from experiences complete")
            result.success = True
            result.state = PipelineState.COMPLETED

            logger.info(
                f"[Pipeline] Experience training complete: "
                f"{result.training_steps} steps, loss={result.final_loss:.4f}"
            )

        except _GateRejectionError:
            # v2.1: Gate rejection — result fields already set, don't overwrite with FAILED
            logger.warning("[Pipeline] Experience training ended: deployment gate rejected model")

        except Exception as e:
            import traceback
            await self._update_state(PipelineState.FAILED, f"Failed: {e}")
            result.error_message = str(e)
            result.metrics["traceback"] = traceback.format_exc()
            logger.error(f"[Pipeline] Experience training failed: {e}")

        finally:
            await self._trinity.stop()
            result.total_time_seconds = time.time() - self._start_time

        return result

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

        # v242.2: Training-level mutual exclusion — prevents concurrent training from
        # ANY entry point (server.py API, NightShiftPipeline, PipelineScheduler, manual).
        # Lock is at the training layer, not the HTTP layer, so all paths are protected.
        _lock_fd = None
        try:
            import fcntl
            lock_dir = Path.home() / ".reactor_core" / "scheduler" / "locks"
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_file = lock_dir / "training_pipeline.lock"
            _lock_fd = open(lock_file, "w")
            try:
                fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                _lock_fd.write(f"{os.getpid()}:{time.time()}\n")
                _lock_fd.flush()
            except (IOError, OSError):
                _lock_fd.close()
                _lock_fd = None
                logger.warning("[Pipeline] Training already in progress (lock held) — skipping")
                result.error_message = "Training already in progress — concurrent run blocked"
                return result
        except ImportError:
            pass  # fcntl not available (Windows) — skip locking

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

            # Step 1b: Generate DPO preference pairs from multi-model telemetry
            dpo_pairs = []
            try:
                from reactor_core.training.dpo_pair_generator import DPOPairGenerator
                dpo_gen = DPOPairGenerator()
                dpo_pairs = await dpo_gen.generate_from_telemetry(
                    telemetry_dir=self.config.telemetry_dir,
                    since=since,
                )
                if dpo_pairs:
                    dpo_export_path = await dpo_gen.export_pairs(dpo_pairs)
                    result.metrics["dpo_pairs_generated"] = len(dpo_pairs)
                    result.metrics["dpo_export_path"] = str(dpo_export_path)
                    logger.info(f"[Pipeline] Generated {len(dpo_pairs)} DPO pairs")
                else:
                    logger.info("[Pipeline] No DPO pairs generated (insufficient multi-model data)")
            except ImportError:
                logger.debug("[Pipeline] DPO pair generator not available")
            except Exception as dpo_err:
                logger.warning(f"[Pipeline] DPO generation failed (non-blocking): {dpo_err}")

            # Step 2: Build dataset
            await self._update_state(PipelineState.PREPROCESSING, "Building dataset")
            train_dataset, eval_dataset = await self._build_dataset(raw_interactions)

            # Step 3: Train model
            await self._update_state(PipelineState.TRAINING, "Training model")

            # Voice announcement: Training started
            try:
                from reactor_core.voice_integration import announce_training_started
                self._spawn_background_task(announce_training_started(
                    model_name=self.config.base_model.split("/")[-1],
                    samples=len(raw_interactions),
                    epochs=self.config.num_epochs,
                ), name="voice_announce_training_started")
            except Exception:
                pass

            # Trinity Event: Training started
            try:
                from reactor_core.integration.trinity_publisher import publish_training_started
                self._spawn_background_task(publish_training_started(
                    model_name=self.config.base_model.split("/")[-1],
                    config={
                        "epochs": self.config.num_epochs,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "samples": len(raw_interactions),
                        "lora_rank": self.config.lora_rank,
                    },
                ), name="trinity_training_started")
            except Exception as e:
                logger.debug(f"Trinity event publish failed: {e}")

            training_result = await self._train_model(train_dataset, eval_dataset)

            # Step 3b: DPO preference training (if pairs were generated in Step 1b)
            # This runs AFTER SFT as a refinement step on the same model.
            # DPO requires a full (merged) model — LoRA adapters alone aren't loadable.
            # If SFT produced only adapters (QLoRA path), we must merge first.
            dpo_model_path = None
            if dpo_pairs and training_result.success:
                try:
                    # Resolve the model path for DPO — needs a full model, not just adapters
                    dpo_input_path = training_result.merged_model_path
                    if not dpo_input_path and training_result.adapter_path:
                        # SFT produced LoRA adapters only (QLoRA path) — merge before DPO
                        logger.info("[Pipeline] Merging LoRA adapters into base model for DPO input")
                        dpo_input_path = await self._merge_lora_for_dpo(
                            adapter_path=training_result.adapter_path,
                        )

                    if dpo_input_path:
                        preference_pairs = [p.to_preference_pair() for p in dpo_pairs]
                        dpo_result = await self._train_dpo(
                            preference_pairs=preference_pairs,
                            model_path=dpo_input_path,
                        )
                        if dpo_result and dpo_result.get("model_path"):
                            # Use DPO-refined model for all downstream steps (GGUF, deploy)
                            dpo_model_path = Path(dpo_result["model_path"])
                            result.metrics["dpo_training_loss"] = dpo_result.get("final_loss", 0.0)
                            result.metrics["dpo_training_steps"] = dpo_result.get("total_steps", 0)
                            logger.info(
                                f"[Pipeline] DPO training complete: "
                                f"{dpo_result.get('total_steps', 0)} steps, "
                                f"loss={dpo_result.get('final_loss', 0):.4f}"
                            )
                    else:
                        logger.warning("[Pipeline] No mergeable model path for DPO — skipping DPO step")
                except ImportError:
                    logger.info("[Pipeline] DPO trainer (advanced_training) not available — skipping DPO step")
                except Exception as dpo_train_err:
                    logger.warning(f"[Pipeline] DPO training failed (non-blocking, SFT model preserved): {dpo_train_err}")

            if not training_result.success:
                # Voice announcement: Training failed
                try:
                    from reactor_core.voice_integration import announce_training_failed
                    self._spawn_background_task(announce_training_failed(
                        model_name=self.config.base_model.split("/")[-1],
                        error_message=training_result.error_message or "Unknown error",
                        steps_completed=training_result.total_steps or 0,
                    ), name="voice_announce_training_failed")
                except Exception:
                    pass

                raise RuntimeError(f"Training failed: {training_result.error_message}")

            result.adapter_path = training_result.adapter_path
            # Use DPO-refined model if available, otherwise fall back to SFT output
            result.model_path = dpo_model_path or training_result.merged_model_path or training_result.adapter_path
            result.training_steps = training_result.total_steps
            result.final_loss = training_result.final_loss
            result.training_time_seconds = training_result.training_time_seconds

            # Voice announcement: Training complete
            try:
                from reactor_core.voice_integration import announce_training_complete
                self._spawn_background_task(announce_training_complete(
                    model_name=self.config.base_model.split("/")[-1],
                    steps=training_result.total_steps or 0,
                    loss=training_result.final_loss or 0.0,
                    duration_seconds=training_result.training_time_seconds or 0.0,
                    success=True,
                ), name="voice_announce_training_complete")
            except Exception:
                pass

            # Trinity Event: Training complete
            try:
                from reactor_core.integration.trinity_publisher import publish_training_complete
                self._spawn_background_task(publish_training_complete(
                    model_name=self.config.base_model.split("/")[-1],
                    model_path=str(training_result.merged_model_path or training_result.adapter_path or ""),
                    metrics={
                        "final_loss": training_result.final_loss,
                        "total_steps": training_result.total_steps,
                    },
                    total_steps=training_result.total_steps,
                    training_time_seconds=training_result.training_time_seconds,
                ), name="trinity_training_complete")
            except Exception as e:
                logger.debug(f"Trinity event publish failed: {e}")

            # Step 4: Export to GGUF (if configured)
            if self.config.export_gguf and result.model_path:
                await self._update_state(PipelineState.EXPORTING, "Exporting to GGUF")

                # Voice announcement: Export started
                try:
                    from reactor_core.voice_integration import announce_export_started
                    self._spawn_background_task(announce_export_started(
                        format="GGUF",
                        quantization=self.config.gguf_quantization,
                    ), name="voice_announce_export_started")
                except Exception:
                    pass

                result.gguf_path = await self._export_gguf(result.model_path)

                # Voice announcement: Export complete
                if result.gguf_path:
                    try:
                        from reactor_core.voice_integration import announce_export_complete
                        file_size_mb = result.gguf_path.stat().st_size / (1024 * 1024) if result.gguf_path.exists() else None
                        self._spawn_background_task(announce_export_complete(
                            format="GGUF",
                            file_size_mb=file_size_mb,
                        ), name="voice_announce_export_complete")
                    except Exception:
                        pass

            # Step 4b: Deployment gate — validate GGUF before deployment (v2.1)
            if result.gguf_path:
                await self._update_state(PipelineState.VALIDATING, "Validating GGUF via deployment gate")
                try:
                    from reactor_core.deployment.gate import DeploymentGate

                    gate = DeploymentGate(
                        skip_inference_check=bool(os.getenv("REACTOR_SKIP_INFERENCE_CHECK", "")),
                    )
                    gate_result = await gate.validate(result.gguf_path)
                    result.gate_summary = gate_result.summary()

                    if not gate_result.passed:
                        logger.error(f"[Pipeline] Deployment gate REJECTED: {gate_result.summary()}")

                        # Move to failed/ directory instead of deploying
                        failed_dir = self.config.output_dir / "failed"
                        failed_dir.mkdir(parents=True, exist_ok=True)
                        failed_dest = failed_dir / result.gguf_path.name
                        result.gguf_path.rename(failed_dest)
                        logger.info(f"[Pipeline] Rejected GGUF moved to {failed_dest}")

                        result.gguf_path = None  # Clear so deploy step is skipped
                        result.metrics["gate_rejected"] = True
                        result.metrics["gate_summary"] = gate_result.summary()
                        result.metrics["gate_failed_path"] = str(failed_dest)

                        # Voice announcement: Gate rejected
                        try:
                            from reactor_core.voice_integration import announce_training_failed
                            self._spawn_background_task(announce_training_failed(
                                model_name=self.config.base_model.split("/")[-1],
                                error_message=f"Deployment gate rejected: {gate_result.summary()}",
                                steps_completed=result.training_steps,
                            ), name="voice_announce_gate_rejected")
                        except Exception:
                            pass

                        await self._update_state(
                            PipelineState.GATE_REJECTED,
                            f"GGUF failed deployment gate: {gate_result.summary()}",
                        )
                        result.success = False
                        result.state = PipelineState.GATE_REJECTED
                        result.error_message = f"Deployment gate rejected: {gate_result.summary()}"
                        # Skip deployment — jump to finally via early return path
                        raise _GateRejectionError(gate_result.summary())
                    else:
                        logger.info(f"[Pipeline] Deployment gate APPROVED: {gate_result.summary()}")
                        result.metrics["gate_approved"] = True
                        result.metrics["gate_summary"] = gate_result.summary()
                except _GateRejectionError:
                    raise  # Re-raise to skip deployment and reach the outer except/finally
                except ImportError:
                    logger.warning("[Pipeline] DeploymentGate not available — skipping validation")
                except Exception as gate_err:
                    logger.warning(f"[Pipeline] Deployment gate error (non-blocking): {gate_err}")

            # Step 5: Deploy to J-Prime (if configured)
            if self.config.auto_deploy and (result.gguf_path or result.model_path):
                await self._update_state(PipelineState.DEPLOYING, "Deploying to J-Prime")

                # Voice announcement: Deployment started
                try:
                    from reactor_core.voice_integration import announce_deployment_started
                    self._spawn_background_task(announce_deployment_started(
                        target="JARVIS-Prime",
                    ), name="voice_announce_deployment_started")
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
                        self._spawn_background_task(announce_deployment_complete(
                            target="JARVIS-Prime",
                            model_version=model_version,
                        ), name="voice_announce_deployment_complete")
                    except Exception:
                        pass

                    # Trinity Event: Model ready for hot-swap
                    # This is THE KEY EVENT that closes the loop!
                    try:
                        from reactor_core.integration.trinity_publisher import publish_model_ready
                        self._spawn_background_task(publish_model_ready(
                            model_name=self.config.base_model.split("/")[-1],
                            model_path=str(result.deployed_to or result.gguf_path or result.model_path),
                            capabilities=["text_generation", "instruction_following"],
                            model_type="llm",
                            metadata={
                                "final_loss": result.final_loss,
                                "training_steps": result.training_steps,
                                "samples_used": result.samples_used,
                                "quantization": self.config.gguf_quantization,
                            },
                        ), name="trinity_model_ready")
                        logger.info("[Trinity] Published MODEL_READY - hot-swap can now occur!")
                    except Exception as e:
                        logger.debug(f"Trinity MODEL_READY publish failed: {e}")

            # Success
            await self._update_state(PipelineState.COMPLETED, "Training cycle complete")
            result.success = True
            result.state = PipelineState.COMPLETED

            logger.info(
                f"[Pipeline] Training cycle complete: "
                f"{result.training_steps} steps, loss={result.final_loss:.4f}"
            )

        except _GateRejectionError:
            # v2.1: Gate rejection — result fields already set above, don't overwrite
            logger.warning(f"[Pipeline] Training cycle ended: deployment gate rejected model")

        except Exception as e:
            import traceback
            await self._update_state(PipelineState.FAILED, f"Failed: {e}")
            result.error_message = str(e)
            result.metrics["traceback"] = traceback.format_exc()
            logger.error(f"[Pipeline] Training cycle failed: {e}")

            # Trinity Event: Training failed
            try:
                from reactor_core.integration.trinity_publisher import publish_training_failed
                self._spawn_background_task(publish_training_failed(
                    model_name=self.config.base_model.split("/")[-1],
                    error_message=str(e),
                    step=self._progress.training_step,
                    traceback=traceback.format_exc(),
                ), name="trinity_training_failed")
            except Exception:
                pass  # Don't let event publishing failure mask the real error

        finally:
            await self._trinity.stop()
            result.total_time_seconds = time.time() - self._start_time

            # v242.2: Release training lock
            if _lock_fd:
                try:
                    import fcntl
                    fcntl.flock(_lock_fd.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                finally:
                    _lock_fd.close()

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

    async def _merge_lora_for_dpo(self, adapter_path: Path) -> Optional[Path]:
        """
        Merge LoRA adapters into the base model to produce a full model for DPO.

        When SFT used QLoRA, only adapter weights are saved — the DPO trainer needs
        a full model to load and create a reference copy. This merges the adapters
        into the base model and saves the result.

        Returns:
            Path to the merged model directory, or None on failure.
        """
        try:
            from peft import PeftModel, PeftConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Load the PEFT config to find the base model name
            peft_config = PeftConfig.from_pretrained(str(adapter_path))
            base_model_name = peft_config.base_model_name_or_path or self.config.base_model

            logger.info(f"[Pipeline] Loading base model {base_model_name} for LoRA merge")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)

            # Load and merge adapters
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            merged_model = model.merge_and_unload()

            # Save merged model
            merged_dir = self.config.output_dir / "sft_merged_for_dpo"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))

            logger.info(f"[Pipeline] LoRA adapters merged for DPO: {merged_dir}")
            return merged_dir

        except ImportError as e:
            logger.warning(f"[Pipeline] peft not available for LoRA merge: {e}")
            return None
        except Exception as e:
            logger.error(f"[Pipeline] LoRA merge for DPO failed: {e}")
            return None

    async def _train_dpo(
        self,
        preference_pairs: List[Any],
        model_path: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run DPO preference training as a refinement step after SFT.

        Uses the DPOTrainer from advanced_training.py with the SFT-trained
        model as the starting point. The reference model is a frozen copy.

        Args:
            preference_pairs: List of PreferencePair objects (or compatible dicts)
            model_path: Path to the SFT-trained model to refine

        Returns:
            Dict with training metrics, or None if DPO training is not possible
        """
        from reactor_core.training.advanced_training import (
            DPOTrainer,
            DPOConfig as AdvancedDPOConfig,
            PreferencePair,
        )

        if not preference_pairs:
            return None

        # Convert any dict-form pairs to PreferencePair objects
        converted_pairs = []
        for pair in preference_pairs:
            if isinstance(pair, PreferencePair):
                converted_pairs.append(pair)
            elif isinstance(pair, dict):
                converted_pairs.append(PreferencePair.from_dict(pair))
            else:
                # Assume it's a DPOPair with to_preference_pair() already called
                converted_pairs.append(pair)

        if not converted_pairs:
            return None

        logger.info(f"[Pipeline] Starting DPO training with {len(converted_pairs)} preference pairs")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load the SFT-trained model for DPO refinement
            model_name = str(model_path) if model_path else self.config.base_model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            dpo_config = AdvancedDPOConfig(
                beta=float(os.getenv("DPO_BETA", "0.1")),
                batch_size=int(os.getenv("DPO_BATCH_SIZE", "2")),
                learning_rate=float(os.getenv("DPO_LEARNING_RATE", "5e-7")),
                max_length=int(os.getenv("DPO_MAX_LENGTH", "512")),
            )

            dpo_trainer = DPOTrainer(
                model=model,
                tokenizer=tokenizer,
                config=dpo_config,
            )

            # Train
            num_epochs = int(os.getenv("DPO_NUM_EPOCHS", "1"))

            async def dpo_progress(metrics):
                self._progress.message = f"DPO step {metrics.step}: loss={metrics.loss:.4f}"
                if self.progress_callback:
                    self.progress_callback(self._progress)

            all_metrics = await dpo_trainer.train_on_preferences(
                preference_pairs=converted_pairs,
                num_epochs=num_epochs,
                progress_callback=dpo_progress,
            )

            final_loss = all_metrics[-1].loss if all_metrics else 0.0
            total_steps = len(all_metrics)

            # Save the DPO-refined model
            dpo_output_dir = self.config.output_dir / "dpo_refined"
            dpo_output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(dpo_output_dir)
            tokenizer.save_pretrained(dpo_output_dir)
            logger.info(f"[Pipeline] DPO-refined model saved to {dpo_output_dir}")

            return {
                "final_loss": final_loss,
                "total_steps": total_steps,
                "model_path": str(dpo_output_dir),
            }

        except ImportError as e:
            logger.warning(f"[Pipeline] DPO training dependencies not available: {e}")
            return None
        except Exception as e:
            logger.error(f"[Pipeline] DPO training error: {e}")
            return None

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
                # v242.0: Env-var-driven discovery instead of hardcoded paths
                jprime_env = os.getenv("JPRIME_MODELS_DIR")
                if jprime_env:
                    jprime_dir = Path(jprime_env)
                else:
                    # Fall back to dynamic discovery
                    candidates = [
                        Path(os.getenv("JPRIME_REPO_DIR", "")) / "models",
                        Path.home() / ".jarvis" / "models",
                        Path.home() / "Documents" / "repos" / "jarvis-prime" / "models",
                    ]
                    for candidate in candidates:
                        if str(candidate) and candidate.exists():
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

            # v242.0: Notify J-Prime that a new model is available via Trinity event
            try:
                from reactor_core.integration.trinity_publisher import get_trinity_publisher
                publisher = get_trinity_publisher()
                if publisher:
                    await publisher.publish_event(
                        event_type="model_ready",
                        payload={
                            "model_path": str(dest_path),
                            "model_name": dest_path.name,
                            "deployed_at": datetime.now().isoformat(),
                            "source": "reactor_core_pipeline",
                        },
                    )
                    logger.info(f"[Pipeline] Published model_ready event for {dest_path.name}")
            except Exception as notify_err:
                logger.debug(f"[Pipeline] model_ready notification failed (non-critical): {notify_err}")

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
# Singleton Accessor
# =============================================================================

def get_unified_trainer(
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    reinitialize: bool = False,
) -> UnifiedTrainingPipeline:
    """
    Get the singleton UnifiedTrainingPipeline instance.

    This function provides thread-safe access to a single training pipeline
    instance, enabling the Trinity Loop integration where JARVIS can send
    experiences to Reactor-Core for training.

    Args:
        config: Optional pipeline configuration (only used on first call
            or when reinitialize=True).
        progress_callback: Optional callback for progress updates (only used
            on first call or when reinitialize=True).
        reinitialize: If True, create a new instance even if one exists.
            Use with caution as this may interrupt ongoing training.

    Returns:
        The singleton UnifiedTrainingPipeline instance.

    Example:
        >>> from reactor_core.training.unified_pipeline import get_unified_trainer
        >>> trainer = get_unified_trainer()
        >>> await trainer.add_experiences([
        ...     {"user_input": "Hello", "assistant_output": "Hi there!"}
        ... ])

    Thread Safety:
        This function uses double-checked locking to ensure thread-safe
        singleton creation without excessive lock contention.
    """
    global _unified_trainer_instance

    # Fast path: instance already exists and no reinitialization requested
    if _unified_trainer_instance is not None and not reinitialize:
        return _unified_trainer_instance

    # Slow path: need to create instance (with lock)
    with _unified_trainer_lock:
        # Double-check after acquiring lock
        if _unified_trainer_instance is not None and not reinitialize:
            return _unified_trainer_instance

        # Create new instance
        _unified_trainer_instance = UnifiedTrainingPipeline(
            config=config,
            progress_callback=progress_callback,
        )
        logger.info("[Pipeline] Singleton UnifiedTrainingPipeline created")

    return _unified_trainer_instance


async def get_unified_trainer_async(
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    auto_initialize: bool = True,
) -> UnifiedTrainingPipeline:
    """
    Get the singleton UnifiedTrainingPipeline instance with async initialization.

    This is the preferred method when calling from async context as it ensures
    the pipeline is fully initialized before returning.

    Args:
        config: Optional pipeline configuration.
        progress_callback: Optional callback for progress updates.
        auto_initialize: If True, automatically call initialize() on new instances.

    Returns:
        The initialized singleton UnifiedTrainingPipeline instance.

    Example:
        >>> trainer = await get_unified_trainer_async()
        >>> await trainer.add_experiences(experience_data)
    """
    trainer = get_unified_trainer(
        config=config,
        progress_callback=progress_callback,
    )

    # Initialize if needed
    if auto_initialize:
        await trainer.initialize()

    return trainer


def reset_unified_trainer() -> None:
    """
    Reset the singleton instance to None.

    This should only be used for testing or when explicitly cleaning up
    resources. Any ongoing training will be interrupted.
    """
    global _unified_trainer_instance
    with _unified_trainer_lock:
        if _unified_trainer_instance is not None:
            logger.info("[Pipeline] Resetting singleton UnifiedTrainingPipeline")
            _unified_trainer_instance = None


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
    # Singleton accessors
    "get_unified_trainer",
    "get_unified_trainer_async",
    "reset_unified_trainer",
    # Convenience functions
    "run_night_shift",
    "quick_finetune",
]
