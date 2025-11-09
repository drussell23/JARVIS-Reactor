"""
GCP Spot VM Checkpoint Manager
Auto-saves and resumes training on preemptible VM shutdowns
"""
import os
import json
import torch
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Training checkpoint state"""
    global_step: int
    epoch: int
    best_loss: float
    model_path: str
    optimizer_state_path: str
    scheduler_state_path: Optional[str] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = None


class CheckpointManager:
    """
    Manages training checkpoints with auto-save and resume
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_interval: int = 500,
        max_checkpoints: int = 3,
        gcs_bucket: Optional[str] = None,
    ):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_dir: Local directory for checkpoints
            checkpoint_interval: Save checkpoint every N steps
            max_checkpoints: Maximum number of checkpoints to keep
            gcs_bucket: Optional GCS bucket for cloud backup (e.g., gs://my-bucket/checkpoints)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.gcs_bucket = gcs_bucket

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # GCS client (lazy init)
        self._gcs_client = None

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")

    def should_checkpoint(self, step: int) -> bool:
        """Check if we should save a checkpoint at this step"""
        return step > 0 and step % self.checkpoint_interval == 0

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        global_step: int,
        epoch: int,
        best_loss: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save training checkpoint

        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional scheduler to save
            global_step: Current training step
            epoch: Current epoch
            best_loss: Best validation loss so far
            metadata: Optional additional metadata

        Returns:
            Path to checkpoint directory
        """
        from datetime import datetime

        # Create checkpoint directory
        checkpoint_name = f"checkpoint-step-{global_step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint at step {global_step}...")

        # Save model
        model_path = checkpoint_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Save optimizer
        optimizer_path = checkpoint_path / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)

        # Save scheduler if provided
        scheduler_path = None
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)

        # Save checkpoint state
        state = CheckpointState(
            global_step=global_step,
            epoch=epoch,
            best_loss=best_loss,
            model_path=str(model_path),
            optimizer_state_path=str(optimizer_path),
            scheduler_state_path=str(scheduler_path) if scheduler_path else None,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        state_path = checkpoint_path / "checkpoint_state.json"
        with open(state_path, "w") as f:
            json.dump(asdict(state), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Backup to GCS if configured
        if self.gcs_bucket:
            asyncio.create_task(self._upload_to_gcs(checkpoint_path))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_latest_checkpoint(self) -> Optional[CheckpointState]:
        """
        Load the latest checkpoint

        Returns:
            CheckpointState if found, None otherwise
        """
        # Find latest checkpoint
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-step-*"))

        if not checkpoints:
            logger.info("No checkpoints found")
            return None

        # Sort by step number (descending)
        checkpoints.sort(
            key=lambda p: int(p.name.split("-")[-1]),
            reverse=True
        )

        latest_checkpoint = checkpoints[0]
        state_path = latest_checkpoint / "checkpoint_state.json"

        if not state_path.exists():
            logger.warning(f"Checkpoint state not found: {state_path}")
            return None

        # Load checkpoint state
        with open(state_path, "r") as f:
            state_dict = json.load(f)

        state = CheckpointState(**state_dict)
        logger.info(f"Loaded checkpoint from step {state.global_step}")

        return state

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_state: Optional[CheckpointState] = None,
    ) -> Optional[CheckpointState]:
        """
        Load checkpoint into model, optimizer, and scheduler

        Args:
            model: Model to load into
            optimizer: Optional optimizer to load into
            scheduler: Optional scheduler to load into
            checkpoint_state: Optional specific checkpoint state to load

        Returns:
            Loaded CheckpointState if successful, None otherwise
        """
        if checkpoint_state is None:
            checkpoint_state = self.load_latest_checkpoint()

        if checkpoint_state is None:
            return None

        # Load model
        logger.info(f"Loading model from {checkpoint_state.model_path}")
        model.load_state_dict(torch.load(checkpoint_state.model_path))

        # Load optimizer
        if optimizer is not None and checkpoint_state.optimizer_state_path:
            logger.info(f"Loading optimizer from {checkpoint_state.optimizer_state_path}")
            optimizer.load_state_dict(torch.load(checkpoint_state.optimizer_state_path))

        # Load scheduler
        if scheduler is not None and checkpoint_state.scheduler_state_path:
            logger.info(f"Loading scheduler from {checkpoint_state.scheduler_state_path}")
            scheduler.load_state_dict(torch.load(checkpoint_state.scheduler_state_path))

        logger.info(f"Checkpoint loaded successfully (step {checkpoint_state.global_step})")
        return checkpoint_state

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints most recent"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-step-*"))

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Sort by step number (ascending)
        checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))

        # Remove oldest checkpoints
        num_to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:num_to_remove]:
            logger.info(f"Removing old checkpoint: {checkpoint}")
            import shutil
            shutil.rmtree(checkpoint)

    async def _upload_to_gcs(self, checkpoint_path: Path):
        """Upload checkpoint to GCS (async)"""
        try:
            from google.cloud import storage

            if self._gcs_client is None:
                self._gcs_client = storage.Client()

            # Parse bucket and path
            bucket_name = self.gcs_bucket.replace("gs://", "").split("/")[0]
            prefix = "/".join(self.gcs_bucket.replace("gs://", "").split("/")[1:])

            bucket = self._gcs_client.bucket(bucket_name)

            # Upload all files in checkpoint
            for file_path in checkpoint_path.rglob("*"):
                if file_path.is_file():
                    blob_name = f"{prefix}/{checkpoint_path.name}/{file_path.relative_to(checkpoint_path)}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))

            logger.info(f"Checkpoint uploaded to GCS: {self.gcs_bucket}/{checkpoint_path.name}")

        except Exception as e:
            logger.error(f"Failed to upload checkpoint to GCS: {e}")


class SpotVMCheckpointer(CheckpointManager):
    """
    Specialized checkpoint manager for GCP Spot VMs
    Automatically detects preemption and handles graceful shutdown
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_interval: int = 500,
        gcs_bucket: Optional[str] = None,
    ):
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            max_checkpoints=2,  # Keep only 2 for Spot VMs (save space)
            gcs_bucket=gcs_bucket,
        )

        # Start preemption monitor
        self._monitor_task = None
        self._preemption_detected = False

    def start_monitoring(self):
        """Start monitoring for Spot VM preemption"""
        self._monitor_task = asyncio.create_task(self._monitor_preemption())

    async def _monitor_preemption(self):
        """Monitor GCP metadata for preemption signals"""
        while True:
            try:
                import requests
                response = requests.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
                    headers={"Metadata-Flavor": "Google"},
                    timeout=1,
                )

                if response.text == "TRUE":
                    logger.warning("🚨 SPOT VM PREEMPTION DETECTED! Saving emergency checkpoint...")
                    self._preemption_detected = True
                    # Emergency checkpoint is handled by training loop

            except Exception as e:
                logger.debug(f"Preemption check failed (normal if not on GCP): {e}")

            await asyncio.sleep(5)  # Check every 5 seconds

    def is_preemption_detected(self) -> bool:
        """Check if preemption has been detected"""
        return self._preemption_detected


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create checkpointer
    checkpointer = SpotVMCheckpointer(
        checkpoint_dir="./checkpoints",
        checkpoint_interval=100,
        gcs_bucket="gs://my-training-bucket/checkpoints",
    )

    print("✅ SpotVMCheckpointer initialized")
    print(f"Checkpoint directory: {checkpointer.checkpoint_dir}")
    print(f"Checkpoint interval: {checkpointer.checkpoint_interval}")
    print(f"GCS bucket: {checkpointer.gcs_bucket}")
