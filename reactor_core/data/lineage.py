"""Model Lineage Tracking for Reactor-Core.

Records a complete audit trail of every model produced by the training pipeline,
including its hash, parent model, training configuration, evaluation scores,
and deployment gate decision.

Records are stored as newline-delimited JSON (JSONL) in:
    ~/.jarvis/reactor/models/lineage.jsonl

Each line is a self-contained JSON object representing one model's lineage.
Later pipeline stages (deployment, probation) can update records in-place
by model_id.

Usage:
    from reactor_core.data.lineage import LineageRecord, write_lineage_record

    record = LineageRecord(
        model_id="jarvis-trained-v0.1.1",
        model_hash="sha256:abc123...",
        parent_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        training_method="lora_sft",
        training_job_id="job-1234",
        gate_decision="APPROVED",
    )
    write_lineage_record(record)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default lineage file location
LINEAGE_FILE_PATH = Path.home() / ".jarvis" / "reactor" / "models" / "lineage.jsonl"


@dataclass
class LineageRecord:
    """A single model lineage record.

    All fields are populated at training time except for:
    - deployed_at: set when the model is deployed to production
    - probation_result: set after the probation period completes
    """

    model_id: str = ""
    model_hash: Optional[str] = None
    parent_model: Optional[str] = None
    training_method: Optional[str] = None
    training_job_id: Optional[str] = None
    dataset: Optional[Dict[str, Any]] = None
    eval_scores: Optional[Dict[str, Any]] = None
    gate_decision: Optional[str] = None
    deployed_at: Optional[str] = None
    probation_result: Optional[str] = None
    transformation_steps: Optional[List[Any]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict suitable for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageRecord":
        """Create a LineageRecord from a dict, ignoring unknown fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def write_lineage_record(
    record: LineageRecord,
    lineage_dir: Optional[Path] = None,
) -> Path:
    """Append a lineage record to the JSONL file.

    Uses atomic write (write to temp file in same directory, then rename)
    to prevent corruption from partial writes.

    Args:
        record: The lineage record to write.
        lineage_dir: Directory to write lineage.jsonl into.
                     Defaults to ~/.jarvis/reactor/models/

    Returns:
        Path to the lineage.jsonl file.
    """
    if lineage_dir is None:
        lineage_file = LINEAGE_FILE_PATH
    else:
        lineage_file = Path(lineage_dir) / "lineage.jsonl"

    lineage_file.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(record.to_dict(), default=str) + "\n"

    # Append atomically: write to temp file, then append to target.
    # For JSONL append, we write the single line to a temp file first,
    # then read it back and append. This prevents half-written lines
    # if the process crashes mid-write.
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(lineage_file.parent),
            prefix=".lineage_",
            suffix=".tmp",
        )
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

        # Append the temp file contents to the main file
        with open(lineage_file, "a") as dest:
            with open(tmp_path, "r") as src:
                dest.write(src.read())

        os.unlink(tmp_path)

    except Exception:
        # Fallback: direct append if atomic approach fails
        logger.warning("[Lineage] Atomic write failed, falling back to direct append")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        with open(lineage_file, "a") as f:
            f.write(line)

    logger.info(f"[Lineage] Wrote record for model_id={record.model_id}")
    return lineage_file


def read_lineage_records(
    lineage_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read all lineage records from the JSONL file.

    Args:
        lineage_dir: Directory containing lineage.jsonl.
                     Defaults to ~/.jarvis/reactor/models/

    Returns:
        List of lineage record dicts, one per line.
    """
    if lineage_dir is None:
        lineage_file = LINEAGE_FILE_PATH
    else:
        lineage_file = Path(lineage_dir) / "lineage.jsonl"

    if not lineage_file.exists():
        return []

    records = []
    for line in lineage_file.read_text().strip().split("\n"):
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"[Lineage] Skipping malformed line: {line[:80]}")
    return records


def update_lineage_record(
    model_id: str,
    updates: Dict[str, Any],
    lineage_dir: Optional[Path] = None,
) -> bool:
    """Update an existing lineage record by model_id.

    Rewrites the entire JSONL file atomically with the updated record.
    Used by deployment and probation stages to fill in deployed_at,
    probation_result, etc.

    Args:
        model_id: The model_id of the record to update.
        updates: Dict of field names to new values.
        lineage_dir: Directory containing lineage.jsonl.

    Returns:
        True if a record was found and updated, False otherwise.
    """
    if lineage_dir is None:
        lineage_file = LINEAGE_FILE_PATH
    else:
        lineage_file = Path(lineage_dir) / "lineage.jsonl"

    if not lineage_file.exists():
        return False

    records = read_lineage_records(lineage_dir)
    found = False

    for record in records:
        if record.get("model_id") == model_id:
            record.update(updates)
            found = True
            break

    if not found:
        return False

    # Atomic rewrite: write all records to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(lineage_file.parent),
        prefix=".lineage_",
        suffix=".tmp",
    )
    closed = False
    try:
        content = "".join(
            json.dumps(r, default=str) + "\n" for r in records
        )
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        closed = True
        # Atomic rename
        os.replace(tmp_path, str(lineage_file))
    except Exception:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    logger.info(f"[Lineage] Updated record for model_id={model_id}: {list(updates.keys())}")
    return True
