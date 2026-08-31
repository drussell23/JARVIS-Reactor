"""Cross-process GPU exclusivity for the Trinity flywheel.

The 5090 holds ONE large model at a time. A soak with the 32B resident
occupies ~29/32.6 GiB, so a training or conversion job that starts
alongside it does not run slowly -- it OOMs, and it takes the soak with
it. Something must make "the GPU is mine" a fact both processes can see.

## Why this module exists rather than reusing reactor's lock directly

``reactor_core.utils.trinity_lock_bridge`` already aims at exactly this,
but its two degraded paths are both wrong for a GPU:

  * With ``reactor_core`` importable and **Redis down**,
    ``DistributedLock.acquire`` falls back to an ``asyncio.Lock`` and
    returns True ("Acquired local lock (Redis unavailable)"). An
    in-process lock cannot exclude a soak in another process, and the
    bridge stamps ``backend="redis"`` on the metadata regardless -- so
    the caller is told it holds a distributed lock that guards nothing.
  * With ``reactor_core`` not importable it yields False while logging
    "Using file-based fallback", which no code implements.

JARVIS already ships the working answer: ``cross_repo_lock_bridge`` with
a **file** backend in the canonical shared directory
``~/.jarvis/cross_repo/locks/``. Verified by two-process probe: holder
acquired ``backend=file``, contender was REFUSED during the hold and
granted after release.

So this module does not implement a lock. It resolves JARVIS's manager
and refuses when it cannot.

## Polarity: defer unless proven exclusive

The opposite of a debris sweeper. An unprovable lock state must BLOCK
the GPU job, never allow it: a false "free" costs an OOM that kills a
live soak, while a false "busy" only delays training until the next
window.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Tuple

logger = logging.getLogger(__name__)

GPU_LEASE_SCHEMA_VERSION = "gpu_lease.1"

#: Canonical lock name. Every GPU-exclusive job in every repo must use
#: this exact string or the exclusion does not exist.
GPU_LOCK_NAME = "trinity_gpu_vram"

_ENV_JARVIS_REPO = "JARVIS_REPO_PATH"
_ENV_LEASE_TTL_S = "TRINITY_GPU_LEASE_TTL_S"
_ENV_LEASE_TIMEOUT_S = "TRINITY_GPU_LEASE_TIMEOUT_S"
_ENV_ALLOW_UNSAFE = "TRINITY_GPU_LEASE_ALLOW_UNSAFE"

_DEFAULT_TTL_S = 3600.0
_DEFAULT_TIMEOUT_S = 5.0


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LeaseVerdict:
    """Outcome of a lease attempt.

    ``held`` is the only field a caller may branch on to touch the GPU.
    ``backend`` names what actually enforced it -- never a hardcoded
    guess -- so an operator can tell a real exclusion from a degraded one.
    """

    held: bool
    backend: str
    reason: str
    fencing_token: int = 0

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.held


def jarvis_repo_path() -> Optional[Path]:
    """Resolve the JARVIS checkout that owns the lock protocol."""
    raw = os.getenv(_ENV_JARVIS_REPO, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def _load_jarvis_bridge() -> Optional[Any]:
    """Import JARVIS's cross-repo lock bridge, or None.

    Kept deliberately narrow: the bridge imports standalone in ~0.1s and
    pulls no heavy dependencies, so this does not drag the JARVIS world
    into a training process.
    """
    repo = jarvis_repo_path()
    if repo is None:
        return None
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from backend.core.cross_repo_lock_bridge import (  # noqa: PLC0415
            acquire_trinity_lock,
        )
        return acquire_trinity_lock
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[GPULease] JARVIS lock bridge present but unimportable: %s", exc,
        )
        return None


@asynccontextmanager
async def gpu_lease(
    *,
    reason: str,
    ttl_s: Optional[float] = None,
    timeout_s: Optional[float] = None,
    lock_name: str = GPU_LOCK_NAME,
) -> AsyncIterator[LeaseVerdict]:
    """Hold cross-process GPU exclusivity for the duration of the block.

    Yields a :class:`LeaseVerdict`. Callers MUST check ``.held`` and do
    nothing to the GPU when it is False::

        async with gpu_lease(reason="qlora-32b") as lease:
            if not lease.held:
                logger.info("deferring: %s", lease.reason)
                return
            await train()

    NEVER raises on lock-infrastructure problems -- it reports them as a
    refusal, because a training job that crashes on a missing lock and a
    training job that waits are equally safe, but only one is legible.
    """
    ttl = ttl_s if ttl_s is not None else _env_float(
        _ENV_LEASE_TTL_S, _DEFAULT_TTL_S, 30.0, 86_400.0
    )
    wait = timeout_s if timeout_s is not None else _env_float(
        _ENV_LEASE_TIMEOUT_S, _DEFAULT_TIMEOUT_S, 0.0, 3_600.0
    )

    acquire = _load_jarvis_bridge()
    if acquire is None:
        if _truthy(_ENV_ALLOW_UNSAFE):
            # Explicit operator override. Named UNSAFE because it is: the
            # job proceeds with no cross-process exclusion whatsoever.
            logger.warning(
                "[GPULease] %s=1 -- proceeding with NO GPU exclusion "
                "(reason=%s)", _ENV_ALLOW_UNSAFE, reason,
            )
            yield LeaseVerdict(
                held=True, backend="none-unsafe-override",
                reason="operator override; no cross-process exclusion",
            )
            return
        yield LeaseVerdict(
            held=False,
            backend="unavailable",
            reason=(
                f"no enforceable cross-process lock: set {_ENV_JARVIS_REPO} "
                "to the JARVIS checkout (its file backend is the protocol), "
                f"or set {_ENV_ALLOW_UNSAFE}=1 to proceed without exclusion"
            ),
        )
        return

    try:
        async with acquire(
            lock_name, repo="reactor-core", timeout=wait, ttl=ttl,
        ) as (acquired, meta):
            backend = str(getattr(meta, "backend", "") or "unknown")
            token = int(getattr(meta, "fencing_token", 0) or 0)
            if not acquired:
                yield LeaseVerdict(
                    held=False, backend=backend,
                    reason=(
                        "GPU is held by another Trinity process "
                        "(a soak or another training job)"
                    ),
                )
                return
            # An in-process backend cannot exclude the soak, which is the
            # entire point. Treat it as NOT held rather than trusting a
            # label. This is the check reactor's own bridge omits.
            if backend in ("local", "asyncio", "memory", "inprocess"):
                yield LeaseVerdict(
                    held=False, backend=backend,
                    reason=(
                        f"lock served by in-process backend {backend!r}, "
                        "which cannot exclude another process"
                    ),
                )
                return
            logger.info(
                "[GPULease] held backend=%s token=%d reason=%s",
                backend, token, reason,
            )
            yield LeaseVerdict(
                held=True, backend=backend, reason=reason,
                fencing_token=token,
            )
    except Exception as exc:  # noqa: BLE001 -- refuse, never crash
        logger.error("[GPULease] lock attempt failed: %s", exc)
        yield LeaseVerdict(
            held=False, backend="error", reason=f"lock error: {exc}",
        )


__all__ = [
    "GPU_LEASE_SCHEMA_VERSION",
    "GPU_LOCK_NAME",
    "LeaseVerdict",
    "gpu_lease",
    "jarvis_repo_path",
]
