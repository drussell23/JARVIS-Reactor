"""
Distributed Redis Locks for Reactor-Core Training v1.0
=======================================================

Provides distributed locking primitives for coordinating:
- Training job exclusivity (only one active training at a time)
- Model artifact access (prevent concurrent writes)
- VM provisioning coordination (prevent duplicate VMs)
- Resource allocation (prevent over-subscription)

Features:
- Redis-backed locks with fencing tokens
- Automatic lock extension (keepalive)
- Deadlock prevention with TTL
- Graceful degradation (fallback to local locks)
- Lock acquisition with timeout and retry
- Distributed lock context managers

Architecture:
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │  Training Job 1 │────>│                  │<────│  Training Job 2 │
    │  (VM 1)         │     │   Redis Server   │     │  (VM 2)         │
    │  acquire_lock() │     │   SETNX + TTL    │     │  wait_for_lock()│
    └─────────────────┘     └──────────────────┘     └─────────────────┘

Usage:
    from reactor_core.utils.distributed_lock import (
        DistributedLock,
        get_training_lock,
        get_model_lock,
    )

    # Exclusive training lock
    async with get_training_lock() as lock:
        if lock.acquired:
            await run_training()
        else:
            logger.warning("Another training job is running")

    # Model artifact lock
    async with get_model_lock("prime-v2") as lock:
        await save_model_artifacts()

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Redis configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_LOCK_DB", "1"))  # Separate DB for locks

# Lock configuration
DEFAULT_LOCK_TTL = float(os.getenv("DISTRIBUTED_LOCK_TTL", "60.0"))  # 60 seconds
DEFAULT_ACQUIRE_TIMEOUT = float(os.getenv("DISTRIBUTED_LOCK_TIMEOUT", "30.0"))
DEFAULT_RETRY_INTERVAL = float(os.getenv("DISTRIBUTED_LOCK_RETRY_INTERVAL", "0.5"))
KEEPALIVE_INTERVAL = float(os.getenv("DISTRIBUTED_LOCK_KEEPALIVE", "10.0"))

# Lock key prefixes
LOCK_PREFIX = "reactor:lock:"
TRAINING_LOCK_KEY = f"{LOCK_PREFIX}training:active"
MODEL_LOCK_PREFIX = f"{LOCK_PREFIX}model:"
VM_LOCK_PREFIX = f"{LOCK_PREFIX}vm:"


# =============================================================================
# LOCK STATE
# =============================================================================

class LockState(Enum):
    """States of a distributed lock."""
    RELEASED = "released"
    ACQUIRING = "acquiring"
    ACQUIRED = "acquired"
    EXTENDING = "extending"
    FAILED = "failed"


@dataclass
class LockInfo:
    """Information about a held lock."""
    key: str
    owner: str
    fencing_token: int
    acquired_at: float
    ttl_seconds: float
    extensions: int = 0
    state: LockState = LockState.RELEASED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "owner": self.owner,
            "fencing_token": self.fencing_token,
            "acquired_at": self.acquired_at,
            "ttl_seconds": self.ttl_seconds,
            "extensions": self.extensions,
            "state": self.state.value,
            "held_seconds": time.time() - self.acquired_at if self.acquired_at else 0,
        }


# =============================================================================
# REDIS CLIENT WRAPPER
# =============================================================================

class AsyncRedisClient:
    """
    Async Redis client wrapper with connection management.

    Falls back gracefully when Redis is unavailable.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        password: Optional[str] = REDIS_PASSWORD,
        db: int = REDIS_DB,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self._client = None
        self._available = False
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to Redis."""
        async with self._lock:
            if self._client is not None:
                return self._available

            try:
                import aioredis

                self._client = await aioredis.from_url(
                    f"redis://{self.host}:{self.port}/{self.db}",
                    password=self.password,
                    encoding="utf-8",
                    decode_responses=True,
                )

                # Test connection
                await self._client.ping()
                self._available = True
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
                return True

            except ImportError:
                logger.warning("aioredis not available, using local locks only")
                self._available = False
                return False

            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using local locks")
                self._available = False
                return False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        async with self._lock:
            if self._client:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._client = None
                self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    async def set_nx(
        self,
        key: str,
        value: str,
        ttl_seconds: float,
    ) -> bool:
        """Set if not exists with TTL."""
        if not self._available:
            return False

        try:
            result = await self._client.set(
                key, value, nx=True, ex=int(ttl_seconds)
            )
            return result is not None
        except Exception as e:
            logger.error(f"Redis SETNX failed: {e}")
            return False

    async def get(self, key: str) -> Optional[str]:
        """Get value."""
        if not self._available:
            return None

        try:
            return await self._client.get(key)
        except Exception as e:
            logger.error(f"Redis GET failed: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete key."""
        if not self._available:
            return False

        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE failed: {e}")
            return False

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set TTL on key."""
        if not self._available:
            return False

        try:
            return await self._client.expire(key, ttl_seconds)
        except Exception as e:
            logger.error(f"Redis EXPIRE failed: {e}")
            return False

    async def incr(self, key: str) -> int:
        """Increment and return new value."""
        if not self._available:
            return 0

        try:
            return await self._client.incr(key)
        except Exception as e:
            logger.error(f"Redis INCR failed: {e}")
            return 0


# =============================================================================
# DISTRIBUTED LOCK
# =============================================================================

class DistributedLock:
    """
    Distributed lock with Redis backend and local fallback.

    Features:
    - Fencing tokens for monotonic ordering
    - Automatic TTL extension (keepalive)
    - Graceful fallback to local locks
    - Context manager support
    """

    # Shared Redis client
    _redis_client: Optional[AsyncRedisClient] = None
    _redis_lock = threading.Lock()

    # Local fallback locks
    _local_locks: Dict[str, asyncio.Lock] = {}
    _local_locks_lock = threading.Lock()

    # Fencing token counter
    _fencing_counter: int = 0

    def __init__(
        self,
        key: str,
        ttl_seconds: float = DEFAULT_LOCK_TTL,
        owner: Optional[str] = None,
    ):
        """
        Initialize a distributed lock.

        Args:
            key: Unique lock identifier
            ttl_seconds: Lock TTL (auto-extends while held)
            owner: Owner identifier (defaults to hostname:pid:uuid)
        """
        self.key = key
        self.ttl_seconds = ttl_seconds
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

        self._lock_info: Optional[LockInfo] = None
        self._keepalive_task: Optional[asyncio.Task] = None
        self._acquired_locally = False

    @classmethod
    async def _get_redis_client(cls) -> AsyncRedisClient:
        """Get or create shared Redis client."""
        with cls._redis_lock:
            if cls._redis_client is None:
                cls._redis_client = AsyncRedisClient()

        if not cls._redis_client.is_available:
            await cls._redis_client.connect()

        return cls._redis_client

    @classmethod
    def _get_local_lock(cls, key: str) -> asyncio.Lock:
        """Get or create local fallback lock."""
        with cls._local_locks_lock:
            if key not in cls._local_locks:
                cls._local_locks[key] = asyncio.Lock()
            return cls._local_locks[key]

    @classmethod
    def _next_fencing_token(cls) -> int:
        """Get next fencing token."""
        cls._fencing_counter += 1
        return cls._fencing_counter

    async def acquire(
        self,
        timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
        retry_interval: float = DEFAULT_RETRY_INTERVAL,
    ) -> bool:
        """
        Acquire the lock.

        Args:
            timeout: Maximum time to wait for lock
            retry_interval: Time between retry attempts

        Returns:
            True if lock acquired, False otherwise
        """
        start_time = time.time()
        redis = await self._get_redis_client()

        # Try Redis lock first
        if redis.is_available:
            while time.time() - start_time < timeout:
                # Try to acquire
                lock_value = f"{self.owner}:{time.time()}"
                acquired = await redis.set_nx(self.key, lock_value, self.ttl_seconds)

                if acquired:
                    # Successfully acquired
                    fencing_token = self._next_fencing_token()
                    self._lock_info = LockInfo(
                        key=self.key,
                        owner=self.owner,
                        fencing_token=fencing_token,
                        acquired_at=time.time(),
                        ttl_seconds=self.ttl_seconds,
                        state=LockState.ACQUIRED,
                    )

                    # Start keepalive
                    self._start_keepalive()

                    logger.info(f"Acquired distributed lock: {self.key} (token={fencing_token})")
                    return True

                # Check if we own the lock already
                current_owner = await redis.get(self.key)
                if current_owner and current_owner.startswith(self.owner):
                    # We already own it (reentrant)
                    logger.debug(f"Lock {self.key} already owned by us")
                    return True

                # Wait and retry
                await asyncio.sleep(retry_interval)

            logger.warning(f"Timeout acquiring distributed lock: {self.key}")
            return False

        # Fallback to local lock
        else:
            local_lock = self._get_local_lock(self.key)
            try:
                acquired = await asyncio.wait_for(
                    local_lock.acquire(),
                    timeout=timeout,
                )
                if acquired:
                    self._acquired_locally = True
                    self._lock_info = LockInfo(
                        key=self.key,
                        owner=self.owner,
                        fencing_token=self._next_fencing_token(),
                        acquired_at=time.time(),
                        ttl_seconds=self.ttl_seconds,
                        state=LockState.ACQUIRED,
                    )
                    logger.info(f"Acquired local lock (Redis unavailable): {self.key}")
                    return True
            except asyncio.TimeoutError:
                logger.warning(f"Timeout acquiring local lock: {self.key}")
                return False

        return False

    async def release(self) -> bool:
        """Release the lock."""
        if not self._lock_info:
            return True

        # Stop keepalive
        self._stop_keepalive()

        redis = await self._get_redis_client()

        if redis.is_available and not self._acquired_locally:
            # Only delete if we still own the lock
            current_owner = await redis.get(self.key)
            if current_owner and current_owner.startswith(self.owner):
                deleted = await redis.delete(self.key)
                if deleted:
                    logger.info(f"Released distributed lock: {self.key}")
                    self._lock_info.state = LockState.RELEASED
                    self._lock_info = None
                    return True
            else:
                logger.warning(f"Lock {self.key} owned by someone else, cannot release")

        elif self._acquired_locally:
            # Release local lock
            local_lock = self._get_local_lock(self.key)
            local_lock.release()
            self._acquired_locally = False
            logger.info(f"Released local lock: {self.key}")
            self._lock_info.state = LockState.RELEASED
            self._lock_info = None
            return True

        return False

    def _start_keepalive(self) -> None:
        """Start keepalive task to extend lock TTL."""
        if self._keepalive_task is not None:
            return

        async def keepalive_loop():
            redis = await self._get_redis_client()
            while self._lock_info and self._lock_info.state == LockState.ACQUIRED:
                try:
                    await asyncio.sleep(KEEPALIVE_INTERVAL)

                    if not self._lock_info:
                        break

                    # Extend TTL
                    if redis.is_available:
                        success = await redis.expire(self.key, int(self.ttl_seconds))
                        if success:
                            self._lock_info.extensions += 1
                            logger.debug(
                                f"Extended lock TTL: {self.key} "
                                f"(extensions={self._lock_info.extensions})"
                            )
                        else:
                            logger.warning(f"Failed to extend lock TTL: {self.key}")

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Keepalive error: {e}")

        self._keepalive_task = asyncio.create_task(
            keepalive_loop(),
            name=f"lock_keepalive_{self.key}"
        )

    def _stop_keepalive(self) -> None:
        """Stop keepalive task."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

    @property
    def acquired(self) -> bool:
        """Check if lock is currently held."""
        return (
            self._lock_info is not None and
            self._lock_info.state == LockState.ACQUIRED
        )

    @property
    def fencing_token(self) -> Optional[int]:
        """Get fencing token for this lock."""
        return self._lock_info.fencing_token if self._lock_info else None

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get lock information."""
        return self._lock_info.to_dict() if self._lock_info else None

    async def __aenter__(self) -> "DistributedLock":
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.release()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@asynccontextmanager
async def get_training_lock(
    timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ttl_seconds: float = DEFAULT_LOCK_TTL * 10,  # Longer TTL for training
) -> DistributedLock:
    """
    Get exclusive training lock.

    Only one training job should run at a time across all instances.

    Usage:
        async with get_training_lock() as lock:
            if lock.acquired:
                await run_training()
    """
    lock = DistributedLock(
        key=TRAINING_LOCK_KEY,
        ttl_seconds=ttl_seconds,
    )

    try:
        await lock.acquire(timeout=timeout)
        yield lock
    finally:
        await lock.release()


@asynccontextmanager
async def get_model_lock(
    model_name: str,
    timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
) -> DistributedLock:
    """
    Get lock for a specific model's artifacts.

    Prevents concurrent writes to model files.

    Usage:
        async with get_model_lock("prime-v2") as lock:
            await save_model_artifacts()
    """
    lock = DistributedLock(
        key=f"{MODEL_LOCK_PREFIX}{model_name}",
        ttl_seconds=DEFAULT_LOCK_TTL,
    )

    try:
        await lock.acquire(timeout=timeout)
        yield lock
    finally:
        await lock.release()


@asynccontextmanager
async def get_vm_provisioning_lock(
    vm_name: str,
    timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
) -> DistributedLock:
    """
    Get lock for VM provisioning.

    Prevents duplicate VM creation.

    Usage:
        async with get_vm_provisioning_lock("training-vm-1") as lock:
            if lock.acquired:
                await create_vm()
    """
    lock = DistributedLock(
        key=f"{VM_LOCK_PREFIX}{vm_name}",
        ttl_seconds=DEFAULT_LOCK_TTL * 2,  # VM creation can be slow
    )

    try:
        await lock.acquire(timeout=timeout)
        yield lock
    finally:
        await lock.release()


async def check_training_lock_status() -> Dict[str, Any]:
    """
    Check if training lock is currently held.

    Returns lock status without attempting to acquire.
    """
    redis = await DistributedLock._get_redis_client()

    if not redis.is_available:
        return {
            "redis_available": False,
            "lock_held": False,
            "owner": None,
            "message": "Redis unavailable, cannot determine lock status",
        }

    current_value = await redis.get(TRAINING_LOCK_KEY)

    if current_value:
        parts = current_value.split(":")
        return {
            "redis_available": True,
            "lock_held": True,
            "owner": current_value,
            "hostname": parts[0] if len(parts) > 0 else "unknown",
            "pid": parts[1] if len(parts) > 1 else "unknown",
        }
    else:
        return {
            "redis_available": True,
            "lock_held": False,
            "owner": None,
        }


# =============================================================================
# LIFECYCLE
# =============================================================================

async def shutdown_distributed_locks() -> None:
    """Shutdown distributed lock resources."""
    if DistributedLock._redis_client:
        await DistributedLock._redis_client.disconnect()
        DistributedLock._redis_client = None
    logger.info("Distributed locks shutdown complete")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "DistributedLock",
    "LockInfo",
    "LockState",
    "AsyncRedisClient",
    # Convenience functions
    "get_training_lock",
    "get_model_lock",
    "get_vm_provisioning_lock",
    "check_training_lock_status",
    # Lifecycle
    "shutdown_distributed_locks",
]
