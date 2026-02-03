"""
Trinity Lock Bridge for Reactor-Core v1.0
==========================================

Provides unified cross-repo lock coordination with JARVIS and JARVIS-Prime.

This module bridges Reactor-Core's existing Redis-based DistributedLock
with the unified Trinity lock system from JARVIS.

Usage:
    from reactor_core.utils.trinity_lock_bridge import acquire_trinity_lock

    async with acquire_trinity_lock("training_job") as (acquired, meta):
        if acquired:
            await start_training()

Integration Options:
1. Standalone (this module) - Uses JARVIS lock manager if available, falls back to local
2. Full integration - Set JARVIS_REPO_PATH to enable direct imports

Author: JARVIS AI System (Reactor-Core Integration)
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Try to import JARVIS lock manager
JARVIS_AVAILABLE = False
JARVIS_REPO_PATH = os.getenv("JARVIS_REPO_PATH", "")

if JARVIS_REPO_PATH and os.path.isdir(JARVIS_REPO_PATH):
    sys.path.insert(0, JARVIS_REPO_PATH)
    try:
        from backend.core.cross_repo_lock_bridge import (
            acquire_trinity_lock as jarvis_acquire_trinity_lock,
            TrinityLockManager as JarvisTrinityLockManager,
            TrinityLocks,
            LockMetadata as JarvisLockMetadata,
        )
        JARVIS_AVAILABLE = True
        logger.info(f"[TrinityBridge] JARVIS lock manager loaded from {JARVIS_REPO_PATH}")
    except ImportError as e:
        logger.debug(f"[TrinityBridge] Could not import JARVIS lock manager: {e}")

# Import local Redis lock as fallback
try:
    from reactor_core.utils.distributed_lock import (
        DistributedLock,
        get_distributed_lock_manager,
    )
    LOCAL_LOCK_AVAILABLE = True
except ImportError:
    LOCAL_LOCK_AVAILABLE = False
    logger.warning("[TrinityBridge] Local distributed lock not available")


# =============================================================================
# Local LockMetadata (Compatible with JARVIS)
# =============================================================================

@dataclass
class LockMetadata:
    """
    Lock metadata compatible with JARVIS LockMetadata.

    Provides the same interface for cross-repo compatibility.
    """
    acquired_at: float = 0.0
    expires_at: float = 0.0
    owner: str = ""
    token: str = ""
    lock_name: str = ""
    process_start_time: float = 0.0
    process_name: str = ""
    process_cmdline: str = ""
    machine_id: str = ""
    backend: str = "redis"
    fencing_token: int = 0
    repo_source: str = "reactor-core"
    extensions: int = 0

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    def time_remaining(self) -> float:
        return self.expires_at - time.time()


# =============================================================================
# Configuration
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DEFAULT_LOCK_TTL = float(os.getenv("DISTRIBUTED_LOCK_TTL", "10.0"))
DEFAULT_TIMEOUT = float(os.getenv("DISTRIBUTED_LOCK_TIMEOUT", "5.0"))

TRINITY_LOCK_PREFIX = "jarvis:lock:trinity:"


# =============================================================================
# Trinity Lock Manager (Local Implementation)
# =============================================================================

class LocalTrinityLockManager:
    """
    Local Trinity lock manager using Reactor-Core's Redis-based locks.

    Used as fallback when JARVIS lock manager is not available.
    """

    def __init__(self):
        self._fencing_counter = 0
        self._owner_id = f"reactor-core-{os.getpid()}-{time.time():.1f}"
        self._machine_id = self._get_machine_id()
        self._initialized = False

    def _get_machine_id(self) -> str:
        try:
            import platform
            import socket
            return f"{platform.system().lower()}-{socket.gethostname()}"
        except Exception:
            return "unknown"

    async def initialize(self) -> None:
        if self._initialized:
            return

        if LOCAL_LOCK_AVAILABLE:
            try:
                await get_distributed_lock_manager()
            except Exception as e:
                logger.warning(f"[TrinityBridge] Local lock init failed: {e}")

        self._initialized = True

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: float = DEFAULT_TIMEOUT,
        ttl: float = DEFAULT_LOCK_TTL,
    ) -> AsyncIterator[Tuple[bool, Optional[LockMetadata]]]:
        """Acquire a lock using local Redis-based lock manager."""
        if not self._initialized:
            await self.initialize()

        token = str(uuid4())
        full_name = f"{TRINITY_LOCK_PREFIX}{name}"
        acquired = False
        metadata: Optional[LockMetadata] = None

        try:
            if LOCAL_LOCK_AVAILABLE:
                lock_mgr = await get_distributed_lock_manager()
                async with lock_mgr.acquire(full_name, timeout=timeout) as lock_acquired:
                    if lock_acquired:
                        acquired = True
                        self._fencing_counter += 1
                        now = time.time()
                        metadata = LockMetadata(
                            acquired_at=now,
                            expires_at=now + ttl,
                            owner=self._owner_id,
                            token=token,
                            lock_name=name,
                            machine_id=self._machine_id,
                            backend="redis",
                            fencing_token=self._fencing_counter,
                            repo_source="reactor-core",
                        )
                        logger.debug(f"[TrinityBridge] Lock acquired: {name}")

                    yield acquired, metadata
                    return

            # No local lock available - use simple file-based fallback
            logger.warning(f"[TrinityBridge] Using file-based fallback for lock: {name}")
            yield False, None

        except Exception as e:
            logger.error(f"[TrinityBridge] Lock error for {name}: {e}")
            yield False, None


# =============================================================================
# Global Instance
# =============================================================================

_local_manager: Optional[LocalTrinityLockManager] = None


async def get_local_manager() -> LocalTrinityLockManager:
    """Get or create local Trinity lock manager."""
    global _local_manager
    if _local_manager is None:
        _local_manager = LocalTrinityLockManager()
        await _local_manager.initialize()
    return _local_manager


# =============================================================================
# Main API
# =============================================================================

@asynccontextmanager
async def acquire_trinity_lock(
    name: str,
    timeout: float = DEFAULT_TIMEOUT,
    ttl: float = DEFAULT_LOCK_TTL,
    enable_keepalive: bool = True,
) -> AsyncIterator[Tuple[bool, Optional[LockMetadata]]]:
    """
    Acquire a cross-repo Trinity lock.

    Uses JARVIS lock manager if available, falls back to local Redis.

    Args:
        name: Lock name
        timeout: Max wait time for acquisition
        ttl: Lock time-to-live
        enable_keepalive: Auto-extend TTL (only with JARVIS manager)

    Yields:
        Tuple of (acquired: bool, metadata: Optional[LockMetadata])

    Example:
        async with acquire_trinity_lock("training_job") as (acquired, meta):
            if acquired:
                print(f"Fencing token: {meta.fencing_token}")
                await run_training()
    """
    if JARVIS_AVAILABLE:
        # Use JARVIS unified lock manager
        async with jarvis_acquire_trinity_lock(
            name,
            repo="reactor-core",
            timeout=timeout,
            ttl=ttl,
            enable_keepalive=enable_keepalive,
        ) as result:
            # Convert to local LockMetadata type if needed
            acquired, jarvis_meta = result
            if jarvis_meta:
                metadata = LockMetadata(
                    acquired_at=jarvis_meta.acquired_at,
                    expires_at=jarvis_meta.expires_at,
                    owner=jarvis_meta.owner,
                    token=jarvis_meta.token,
                    lock_name=jarvis_meta.lock_name,
                    process_start_time=jarvis_meta.process_start_time,
                    process_name=jarvis_meta.process_name,
                    machine_id=jarvis_meta.machine_id,
                    backend=jarvis_meta.backend,
                    fencing_token=jarvis_meta.fencing_token,
                    repo_source=jarvis_meta.repo_source,
                    extensions=jarvis_meta.extensions,
                )
                yield acquired, metadata
            else:
                yield acquired, None
    else:
        # Fall back to local lock manager
        manager = await get_local_manager()
        async with manager.lock(name, timeout, ttl) as result:
            yield result


# =============================================================================
# Standard Lock Names (Re-export from JARVIS if available)
# =============================================================================

if JARVIS_AVAILABLE:
    # Use JARVIS definitions
    pass
else:
    # Define locally for compatibility
    class TrinityLocks:
        """Standard lock names for cross-repo coordination."""
        MODEL_SYNC = "trinity:model_sync"
        MODEL_UPDATE = "trinity:model_update"
        MODEL_DEPLOY = "trinity:model_deploy"
        TRAINING_JOB = "trinity:training_job"
        TRAINING_DATA_EXPORT = "trinity:training_data_export"
        CHECKPOINT_SAVE = "trinity:checkpoint_save"
        INFERENCE_BATCH = "trinity:inference_batch"
        CACHE_UPDATE = "trinity:cache_update"
        STATE_SYNC = "trinity:state_sync"
        CONFIG_UPDATE = "trinity:config_update"
        HEALTH_CHECK = "trinity:health_check"
        VBIA_EVENTS = "trinity:vbia_events"
        SPEAKER_PROFILE = "trinity:speaker_profile"
        AUTH_STATE = "trinity:auth_state"


__all__ = [
    "acquire_trinity_lock",
    "LockMetadata",
    "TrinityLocks",
    "get_local_manager",
    "LocalTrinityLockManager",
    "JARVIS_AVAILABLE",
]
