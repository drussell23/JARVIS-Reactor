"""One-shot VRAM capacity probe.

The single place in this repo that answers "how much video memory does
this machine have, and how much is free right now". Both the deployment
gate (can this artifact be served here at all?) and the scheduler's
admission path (is the card busy right now?) are asking questions about
the same physical number, and answering it twice is how the two drift.

Deliberately NOT torch: this runs on a serving-only box where the
training stack is absent, and it must not drag it in. nvidia-smi is the
lowest common denominator that works in WSL2, on bare Linux, and on a
Windows host.

Every probe is fail-soft -> None. A machine with no GPU, no driver, or a
timing-out nvidia-smi is a machine we cannot size for, and None says so.
Callers must decide what an unknown capacity means for them -- for the
gate it means "fall back to the static ceiling", never "assume infinite".
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

VRAM_PROBE_SCHEMA_VERSION = "vram_probe.1"

_ENV_TIMEOUT_S = "REACTOR_VRAM_PROBE_TIMEOUT_S"
_ENV_OVERRIDE_MIB = "REACTOR_VRAM_TOTAL_MIB"

_DEFAULT_TIMEOUT_S = 10.0

_MIB = 1024 * 1024


def _timeout_s() -> float:
    try:
        return max(1.0, min(60.0, float(
            os.getenv(_ENV_TIMEOUT_S, str(_DEFAULT_TIMEOUT_S))
        )))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def _query() -> Optional[Tuple[int, int]]:
    """Return (used_mib, total_mib) for GPU 0, or None.

    Reads memory.used/memory.total -- true OCCUPANCY. Never
    ``utilization.memory``, which is the percent of time the memory bus
    was active and reads 0 on an idle card holding 29 GiB of weights.
    """
    override = os.getenv(_ENV_OVERRIDE_MIB, "").strip()
    if override:
        try:
            total = int(override)
            if total > 0:
                return (0, total)
        except ValueError:
            pass

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=_timeout_s(),
        )
        if out.returncode != 0:
            return None
        line = (out.stdout or "").strip().splitlines()
        if not line:
            return None
        used, total = (int(p.strip()) for p in line[0].split(",")[:2])
        return (used, total) if total > 0 else None
    except Exception:  # noqa: BLE001 -- an unprobeable card is None, not a crash
        logger.debug("[VRAM] probe failed", exc_info=True)
        return None


def total_vram_bytes() -> Optional[int]:
    """Total video memory on GPU 0, in bytes. None when unknown."""
    q = _query()
    return q[1] * _MIB if q else None


def free_vram_bytes() -> Optional[int]:
    """Currently unused video memory on GPU 0, in bytes. None when unknown."""
    q = _query()
    return (q[1] - q[0]) * _MIB if q else None


def used_fraction() -> Optional[float]:
    """Occupied share of GPU 0 in [0, 1]. None when unknown."""
    q = _query()
    if not q:
        return None
    used, total = q
    return used / total if total else None


__all__ = [
    "VRAM_PROBE_SCHEMA_VERSION",
    "free_vram_bytes",
    "total_vram_bytes",
    "used_fraction",
]
