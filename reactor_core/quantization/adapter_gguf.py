"""What did training actually produce, and how is it converted?

The night-shift quantization stage assumed one answer: a full HuggingFace
model directory, which ``GGUFConverter`` turns into a deployable GGUF. That
was true while the only trainer was supervised fine-tuning with the adapter
merged back in. It is false for GRPO, whose artifact is a PEFT adapter --
13.4M parameters over 192 attention projections, ~27 MB -- and merging one
into the base to restore the old assumption needs the base in bf16: 16
shards, ~57 GiB, against a WSL guest capped at 47 GiB. On this host that is
not a slow path, it is the desktop.

So the stage must ASK rather than assume, and the question has a structural
answer. PEFT writes ``adapter_config.json`` plus ``adapter_model.*`` and
nothing else; a full model has a ``config.json`` and weight shards. Reading
the directory is the whole classification -- no flag to set, nothing to keep
in sync with what the trainer chose.

Conversion of an adapter goes through llama.cpp's own
``convert_lora_to_gguf.py``, the same tool the host bridge uses. It reads
only the adapter, so peak memory is the adapter's size and the base is never
opened.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Where llama.cpp is checked out. The converter scripts live at its root.
ENV_LLAMA_CPP_DIR = "LLAMA_CPP_DIR"
#: Interpreter for the converter. It needs `gguf`, which is not necessarily
#: installed in the interpreter running the pipeline.
ENV_CONVERT_PYTHON = "REACTOR_GGUF_CONVERT_PYTHON"
#: Seconds. An adapter converts in seconds; the ceiling only catches a hang.
ENV_CONVERT_TIMEOUT_S = "REACTOR_ADAPTER_CONVERT_TIMEOUT_S"
DEFAULT_CONVERT_TIMEOUT_S = 900.0

#: Adapters are deltas. Quantizing a 27 MB delta saves nothing measurable and
#: costs precision on the weights that were actually trained.
ENV_ADAPTER_OUTTYPE = "REACTOR_ADAPTER_OUTTYPE"
DEFAULT_ADAPTER_OUTTYPE = "f16"

_CONVERTER_NAME = "convert_lora_to_gguf.py"
#: Searched in order when the env var is unset.
_LLAMA_CPP_CANDIDATES: Tuple[str, ...] = (
    "~/llama.cpp", "/opt/llama.cpp", "/usr/local/llama.cpp", "~/src/llama.cpp",
)


class ArtifactKind(str, Enum):
    """What a training run left on disk."""

    ADAPTER = "adapter"        # PEFT: adapter_config.json + adapter_model.*
    MODEL = "model"            # a full HF model directory
    GGUF = "gguf"              # already converted
    UNKNOWN = "unknown"        # missing, empty, or unrecognisable


@dataclass
class ConversionOutcome:
    """Deliberately the shape ``GGUFConverter.ConversionResult`` exposes, so
    the pipeline stage can treat both conversions identically."""

    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    quantized_size_mb: float = 0.0
    kind: ArtifactKind = ArtifactKind.UNKNOWN
    command: Tuple[str, ...] = ()
    log_tail: List[str] = field(default_factory=list)


def classify_artifact(path: Optional[Path]) -> ArtifactKind:
    """Read the directory and say what it is. Never raises.

    Order matters: a PEFT adapter dir can also carry a ``config.json``
    copied from the base, so the adapter markers are checked FIRST. The
    reverse order would classify every adapter as a model and send it into
    a conversion that needs 57 GiB.
    """
    if path is None:
        return ArtifactKind.UNKNOWN
    p = Path(path)
    try:
        if p.is_file():
            return ArtifactKind.GGUF if p.suffix.lower() == ".gguf" else ArtifactKind.UNKNOWN
        if not p.is_dir():
            return ArtifactKind.UNKNOWN
        has_cfg = (p / "adapter_config.json").is_file()
        has_weights = any(p.glob("adapter_model.*"))
        if has_cfg and has_weights:
            return ArtifactKind.ADAPTER
        if has_cfg or has_weights:
            # Half an adapter is a run that died mid-save. Saying UNKNOWN
            # stops it here rather than at a converter that would report
            # something less legible.
            logger.warning(
                "[adapter-gguf] %s has adapter_config=%s adapter_model=%s — "
                "an incomplete save, not a usable artifact",
                p, has_cfg, has_weights,
            )
            return ArtifactKind.UNKNOWN
        if (p / "config.json").is_file() and (
            any(p.glob("*.safetensors")) or any(p.glob("*.bin"))
        ):
            return ArtifactKind.MODEL
        if any(p.glob("*.gguf")):
            return ArtifactKind.GGUF
    except OSError:  # unreadable path is a classification answer, not a crash
        logger.debug("[adapter-gguf] cannot read %s", p, exc_info=True)
    return ArtifactKind.UNKNOWN


def find_converter() -> Optional[Path]:
    """llama.cpp's ``convert_lora_to_gguf.py``, or None."""
    override = (os.environ.get(ENV_LLAMA_CPP_DIR, "") or "").strip()
    roots: List[str] = [override] if override else list(_LLAMA_CPP_CANDIDATES)
    for root in roots:
        if not root:
            continue
        candidate = Path(os.path.expanduser(root)) / _CONVERTER_NAME
        if candidate.is_file():
            return candidate
    return None


def convert_python() -> str:
    """The interpreter that runs the converter.

    Defaults to the one running this process, which is right when the
    pipeline's venv has ``gguf``. A box where it does not sets the env var
    rather than having this module guess at venv layouts.
    """
    return (os.environ.get(ENV_CONVERT_PYTHON, "") or "").strip() or sys.executable


def _timeout_s() -> float:
    raw = (os.environ.get(ENV_CONVERT_TIMEOUT_S, "") or "").strip()
    try:
        v = float(raw)
        return v if v > 0 else DEFAULT_CONVERT_TIMEOUT_S
    except ValueError:
        return DEFAULT_CONVERT_TIMEOUT_S


def build_convert_argv(
    adapter_dir: Path, out_path: Path, *, converter: Path,
) -> Tuple[str, ...]:
    """The converter's command line. Separated so a test can read it."""
    outtype = (os.environ.get(ENV_ADAPTER_OUTTYPE, "") or "").strip() \
        or DEFAULT_ADAPTER_OUTTYPE
    return (
        convert_python(), str(converter), str(adapter_dir),
        "--outfile", str(out_path), "--outtype", outtype,
    )


async def convert_adapter_to_gguf(
    adapter_dir: Path, out_path: Path,
) -> ConversionOutcome:
    """Convert a PEFT adapter to a GGUF adapter. The base is never opened.

    Every failure is reported, never raised: the caller is a pipeline stage
    whose job is to record what happened and move on. ``CancelledError``
    propagates.
    """
    adapter_dir = Path(adapter_dir)
    out_path = Path(out_path)

    kind = classify_artifact(adapter_dir)
    if kind is not ArtifactKind.ADAPTER:
        return ConversionOutcome(
            success=False, kind=kind,
            error=f"{adapter_dir} is not a PEFT adapter (classified {kind.value})",
        )
    converter = find_converter()
    if converter is None:
        return ConversionOutcome(
            success=False, kind=kind,
            error=(
                f"{_CONVERTER_NAME} not found; clone llama.cpp or set "
                f"{ENV_LLAMA_CPP_DIR}"
            ),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = build_convert_argv(adapter_dir, out_path, converter=converter)
    logger.info("[adapter-gguf] converting %s -> %s", adapter_dir, out_path)
    tail: List[str] = []
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    tail.append(line)
                    del tail[:-30]
            code = await asyncio.wait_for(proc.wait(), timeout=_timeout_s())
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ConversionOutcome(
                success=False, kind=kind, command=argv, log_tail=tail,
                error=f"conversion timed out after {_timeout_s():.0f}s",
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a stage reports, never explodes
        return ConversionOutcome(
            success=False, kind=kind, command=argv,
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    if code != 0:
        return ConversionOutcome(
            success=False, kind=kind, command=argv, log_tail=tail,
            error=f"converter exited {code}: {' | '.join(tail[-3:])[:300]}",
        )
    if not out_path.is_file():
        # Exit 0 with no file is the shape that would ship nothing while
        # reporting success, so it is a failure here.
        return ConversionOutcome(
            success=False, kind=kind, command=argv, log_tail=tail,
            error=f"converter exited 0 but wrote no file at {out_path}",
        )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("[adapter-gguf] wrote %s (%.1f MB)", out_path, size_mb)
    return ConversionOutcome(
        success=True, output_path=out_path, quantized_size_mb=size_mb,
        kind=kind, command=argv, log_tail=tail,
    )
