"""
Night Shift Scheduler for Automated Training.

Provides intelligent scheduling of training jobs based on:
- Cron expressions
- Resource availability
- Experience accumulation thresholds
- System load monitoring
- Cross-repo coordination

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Night Shift Scheduler                      │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │   Cron      │  │  Resource   │  │   Experience        │  │
    │  │  Scheduler  │  │  Monitor    │  │   Threshold Trigger │  │
    │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
    │         │                │                    │              │
    │         └────────────────┼────────────────────┘              │
    │                          ▼                                   │
    │              ┌──────────────────────┐                        │
    │              │   Schedule Arbiter   │                        │
    │              │  (conflict resolution│                        │
    │              │   & prioritization)  │                        │
    │              └──────────┬───────────┘                        │
    │                         ▼                                    │
    │              ┌──────────────────────┐                        │
    │              │   Training Executor  │ ──────► Training API   │
    │              └──────────────────────┘                        │
    └─────────────────────────────────────────────────────────────┘

Features:
- Cron-based scheduling with next run prediction
- Resource-aware job admission control
- Experience threshold triggers with debouncing
- Distributed lock for multi-instance coordination
- Job prioritization and conflict resolution
- Automatic retry with exponential backoff
- Dead letter queue for failed schedules
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class SchedulerConfig:
    """Night Shift scheduler configuration."""

    # Scheduler timing
    CHECK_INTERVAL_SECONDS = float(os.getenv("SCHEDULER_CHECK_INTERVAL", "60.0"))
    MIN_SCHEDULE_INTERVAL = int(os.getenv("SCHEDULER_MIN_INTERVAL", "300"))  # 5 minutes

    # Resource thresholds
    CPU_THRESHOLD = float(os.getenv("SCHEDULER_CPU_THRESHOLD", "80.0"))
    MEMORY_THRESHOLD = float(os.getenv("SCHEDULER_MEMORY_THRESHOLD", "85.0"))
    GPU_THRESHOLD = float(os.getenv("SCHEDULER_GPU_THRESHOLD", "90.0"))

    # Experience triggers
    EXPERIENCE_THRESHOLD = int(os.getenv("SCHEDULER_EXP_THRESHOLD", "100"))
    EXPERIENCE_DEBOUNCE_SECONDS = int(os.getenv("SCHEDULER_EXP_DEBOUNCE", "600"))  # 10 minutes

    # Retry settings
    MAX_RETRIES = int(os.getenv("SCHEDULER_MAX_RETRIES", "3"))
    RETRY_BASE_DELAY = float(os.getenv("SCHEDULER_RETRY_DELAY", "60.0"))
    RETRY_EXPONENTIAL_BASE = float(os.getenv("SCHEDULER_RETRY_EXP_BASE", "2.0"))

    # Night shift hours (prefer training during low-traffic hours)
    NIGHT_SHIFT_START_HOUR = int(os.getenv("SCHEDULER_NIGHT_START", "22"))  # 10 PM
    NIGHT_SHIFT_END_HOUR = int(os.getenv("SCHEDULER_NIGHT_END", "6"))  # 6 AM

    # Persistence
    STATE_PATH = Path(os.getenv("SCHEDULER_STATE_PATH", str(Path.home() / ".reactor_core" / "scheduler")))


# ============================================================================
# Data Models
# ============================================================================

class ScheduleType(Enum):
    """Types of schedule triggers."""
    CRON = auto()
    INTERVAL = auto()
    THRESHOLD = auto()
    MANUAL = auto()
    WEBHOOK = auto()


class JobPriority(Enum):
    """Training job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class JobStatus(Enum):
    """Scheduled job status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # Skipped due to resource constraints


@dataclass
class ScheduleRule:
    """Schedule rule definition."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "default"
    schedule_type: ScheduleType = ScheduleType.CRON
    cron_expression: Optional[str] = None  # For cron type
    interval_seconds: Optional[int] = None  # For interval type
    threshold_value: Optional[int] = None  # For threshold type
    priority: JobPriority = JobPriority.NORMAL
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)  # Additional conditions
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_triggered: Optional[float] = None
    next_scheduled: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "schedule_type": self.schedule_type.name,
            "priority": self.priority.name,
        }


@dataclass
class ScheduledJob:
    """A scheduled training job."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    rule_id: str = ""
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    scheduled_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    error: Optional[str] = None
    training_job_id: Optional[str] = None  # ID from training API
    result: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "priority": self.priority.name,
        }


@dataclass
class ResourceSnapshot:
    """System resource snapshot."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    disk_percent: float = 0.0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def is_training_allowed(self) -> Tuple[bool, str]:
        """Check if resource levels allow training."""
        if self.cpu_percent > SchedulerConfig.CPU_THRESHOLD:
            return False, f"CPU usage too high: {self.cpu_percent:.1f}%"
        if self.memory_percent > SchedulerConfig.MEMORY_THRESHOLD:
            return False, f"Memory usage too high: {self.memory_percent:.1f}%"
        if self.gpu_percent is not None and self.gpu_percent > SchedulerConfig.GPU_THRESHOLD:
            return False, f"GPU usage too high: {self.gpu_percent:.1f}%"
        return True, "Resources available"


# ============================================================================
# Cron Parser
# ============================================================================

class CronParser:
    """
    Parse cron expressions and calculate next run times.

    Supports standard cron format: minute hour day month weekday

    Examples:
        "0 2 * * *" - Every day at 2 AM
        "*/15 * * * *" - Every 15 minutes
        "0 22-6 * * *" - Every hour from 10 PM to 6 AM
        "0 3 * * 0" - Every Sunday at 3 AM
    """

    FIELD_NAMES = ["minute", "hour", "day", "month", "weekday"]
    FIELD_RANGES = {
        "minute": (0, 59),
        "hour": (0, 23),
        "day": (1, 31),
        "month": (1, 12),
        "weekday": (0, 6),  # 0 = Monday
    }

    @classmethod
    def parse(cls, expression: str) -> Dict[str, List[int]]:
        """Parse cron expression into field values."""
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression} (expected 5 fields)")

        result = {}
        for i, (field_name, part) in enumerate(zip(cls.FIELD_NAMES, parts)):
            result[field_name] = cls._parse_field(part, field_name)
        return result

    @classmethod
    def _parse_field(cls, field: str, field_name: str) -> List[int]:
        """Parse a single cron field."""
        min_val, max_val = cls.FIELD_RANGES[field_name]
        values = set()

        for part in field.split(","):
            if part == "*":
                values.update(range(min_val, max_val + 1))
            elif "/" in part:
                # Step values: */15 or 1-10/2
                base, step = part.split("/")
                step = int(step)
                if base == "*":
                    start, end = min_val, max_val
                elif "-" in base:
                    start, end = map(int, base.split("-"))
                else:
                    start = int(base)
                    end = max_val
                values.update(range(start, end + 1, step))
            elif "-" in part:
                # Range: 1-10
                start, end = map(int, part.split("-"))
                values.update(range(start, end + 1))
            else:
                # Single value
                values.add(int(part))

        # Validate values
        for v in values:
            if v < min_val or v > max_val:
                raise ValueError(f"Value {v} out of range for {field_name} ({min_val}-{max_val})")

        return sorted(values)

    @classmethod
    def next_run(cls, expression: str, after: Optional[datetime] = None) -> datetime:
        """Calculate next run time after given datetime."""
        if after is None:
            after = datetime.now()

        schedule = cls.parse(expression)
        current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Iterate to find next matching time (max 1 year ahead)
        max_iterations = 525600  # Minutes in a year
        for _ in range(max_iterations):
            if (
                current.minute in schedule["minute"]
                and current.hour in schedule["hour"]
                and current.day in schedule["day"]
                and current.month in schedule["month"]
                and current.weekday() in schedule["weekday"]
            ):
                return current
            current += timedelta(minutes=1)

        raise ValueError(f"No next run found for expression: {expression}")

    @classmethod
    def is_night_shift(cls, dt: Optional[datetime] = None) -> bool:
        """Check if datetime falls within night shift hours."""
        if dt is None:
            dt = datetime.now()

        hour = dt.hour
        start = SchedulerConfig.NIGHT_SHIFT_START_HOUR
        end = SchedulerConfig.NIGHT_SHIFT_END_HOUR

        if start > end:  # Crosses midnight
            return hour >= start or hour < end
        else:
            return start <= hour < end


# ============================================================================
# Resource Monitor
# ============================================================================

class ResourceMonitor:
    """
    Monitor system resources for training admission control.

    Checks CPU, memory, GPU, and disk before allowing training jobs.
    """

    def __init__(self, check_interval: float = 30.0):
        self._check_interval = check_interval
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._last_check: float = 0
        self._lock = asyncio.Lock()

    async def get_snapshot(self, force_refresh: bool = False) -> ResourceSnapshot:
        """Get current resource snapshot."""
        async with self._lock:
            now = time.time()

            if not force_refresh and self._last_snapshot is not None:
                if now - self._last_check < self._check_interval:
                    return self._last_snapshot

            snapshot = await self._collect_metrics()
            self._last_snapshot = snapshot
            self._last_check = now
            return snapshot

    async def _collect_metrics(self) -> ResourceSnapshot:
        """Collect system resource metrics."""
        snapshot = ResourceSnapshot()

        # Try psutil if available
        try:
            import psutil

            snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            snapshot.memory_percent = mem.percent
            disk = psutil.disk_usage("/")
            snapshot.disk_percent = disk.percent

            try:
                snapshot.load_average = psutil.getloadavg()
            except (AttributeError, OSError):
                pass

        except ImportError:
            # Fallback to basic OS metrics
            try:
                with open("/proc/stat") as f:
                    lines = f.readlines()
                    cpu_line = lines[0].split()
                    total = sum(int(x) for x in cpu_line[1:])
                    idle = int(cpu_line[4])
                    snapshot.cpu_percent = 100 * (1 - idle / total) if total > 0 else 0
            except Exception:
                pass

            try:
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            meminfo[parts[0].rstrip(":")] = int(parts[1])
                    total = meminfo.get("MemTotal", 1)
                    available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
                    snapshot.memory_percent = 100 * (1 - available / total)
            except Exception:
                pass

        # Try GPU metrics
        snapshot.gpu_percent, snapshot.gpu_memory_percent = await self._get_gpu_metrics()

        return snapshot

    async def _get_gpu_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU utilization metrics."""
        # Try nvidia-smi
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                line = stdout.decode().strip().split("\n")[0]
                gpu_util, mem_util = map(float, line.split(","))
                return gpu_util, mem_util
        except Exception:
            pass

        # Try MPS (Apple Silicon) - no direct utilization API
        # Could potentially use powermetrics but requires sudo

        return None, None

    async def is_training_allowed(self) -> Tuple[bool, str]:
        """Check if training is allowed based on current resources."""
        snapshot = await self.get_snapshot()
        return snapshot.is_training_allowed()


# ============================================================================
# Experience Threshold Trigger
# ============================================================================

class ExperienceThresholdTrigger:
    """
    Trigger training based on accumulated experience count.

    Features:
    - Debouncing to prevent excessive triggers
    - Configurable threshold
    - Automatic reset after training
    """

    def __init__(
        self,
        threshold: int = SchedulerConfig.EXPERIENCE_THRESHOLD,
        debounce_seconds: float = SchedulerConfig.EXPERIENCE_DEBOUNCE_SECONDS,
    ):
        self._threshold = threshold
        self._debounce = debounce_seconds
        self._current_count = 0
        self._last_trigger: float = 0
        self._lock = asyncio.Lock()

    async def add_experiences(self, count: int) -> Tuple[bool, int]:
        """
        Add experiences and check if threshold is reached.

        Returns:
            (should_trigger, current_count)
        """
        async with self._lock:
            self._current_count += count

            if self._current_count < self._threshold:
                return False, self._current_count

            # Check debounce
            now = time.time()
            if now - self._last_trigger < self._debounce:
                return False, self._current_count

            self._last_trigger = now
            return True, self._current_count

    async def reset(self):
        """Reset experience count after training."""
        async with self._lock:
            self._current_count = 0

    async def get_status(self) -> Dict[str, Any]:
        """Get trigger status."""
        async with self._lock:
            return {
                "current_count": self._current_count,
                "threshold": self._threshold,
                "progress": self._current_count / self._threshold if self._threshold > 0 else 0,
                "time_until_debounce": max(0, self._debounce - (time.time() - self._last_trigger)),
            }


# ============================================================================
# Schedule Arbiter (Conflict Resolution)
# ============================================================================

class ScheduleArbiter:
    """
    Resolve conflicts between scheduled jobs.

    Handles:
    - Priority-based scheduling
    - Resource constraint satisfaction
    - Maximum concurrent job limits
    """

    def __init__(self, max_concurrent: int = 1):
        self._max_concurrent = max_concurrent
        self._running_jobs: Dict[str, ScheduledJob] = {}
        self._lock = asyncio.Lock()

    async def can_admit(self, job: ScheduledJob, resources: ResourceSnapshot) -> Tuple[bool, str]:
        """Check if job can be admitted."""
        async with self._lock:
            # Check concurrent limit
            if len(self._running_jobs) >= self._max_concurrent:
                # Check if new job has higher priority
                lowest_priority_job = min(
                    self._running_jobs.values(),
                    key=lambda j: j.priority.value,
                    default=None,
                )
                if lowest_priority_job and job.priority.value <= lowest_priority_job.priority.value:
                    return False, f"Max concurrent jobs ({self._max_concurrent}) reached"
                # Could preempt lower priority job - for now just reject

            # Check resources
            allowed, reason = resources.is_training_allowed()
            if not allowed:
                return False, reason

            return True, "Admitted"

    async def admit(self, job: ScheduledJob):
        """Admit job to running set."""
        async with self._lock:
            self._running_jobs[job.job_id] = job

    async def release(self, job_id: str):
        """Release job from running set."""
        async with self._lock:
            self._running_jobs.pop(job_id, None)

    async def get_running(self) -> List[ScheduledJob]:
        """Get list of running jobs."""
        async with self._lock:
            return list(self._running_jobs.values())


# ============================================================================
# Distributed Lock (for multi-instance coordination)
# ============================================================================

class DistributedLock:
    """
    Simple file-based distributed lock for scheduler coordination.

    Prevents multiple scheduler instances from triggering the same job.
    """

    def __init__(self, lock_dir: Optional[Path] = None):
        self._lock_dir = lock_dir or SchedulerConfig.STATE_PATH / "locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._held_locks: Set[str] = set()

    async def acquire(self, lock_name: str, timeout: float = 30.0) -> bool:
        """Acquire a named lock."""
        lock_file = self._lock_dir / f"{lock_name}.lock"
        lock_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        start = time.time()

        while time.time() - start < timeout:
            try:
                # Try to create lock file exclusively
                if not lock_file.exists():
                    lock_file.write_text(json.dumps({
                        "holder": lock_id,
                        "acquired_at": time.time(),
                        "hostname": os.uname().nodename,
                    }))
                    self._held_locks.add(lock_name)
                    return True

                # Check if existing lock is stale (older than 5 minutes)
                try:
                    lock_data = json.loads(lock_file.read_text())
                    if time.time() - lock_data.get("acquired_at", 0) > 300:
                        lock_file.unlink()
                        continue
                except Exception:
                    lock_file.unlink()
                    continue

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"[Lock] Error acquiring {lock_name}: {e}")
                await asyncio.sleep(0.5)

        return False

    async def release(self, lock_name: str):
        """Release a named lock."""
        lock_file = self._lock_dir / f"{lock_name}.lock"
        try:
            if lock_file.exists():
                lock_file.unlink()
            self._held_locks.discard(lock_name)
        except Exception as e:
            logger.error(f"[Lock] Error releasing {lock_name}: {e}")

    async def cleanup(self):
        """Release all held locks."""
        for lock_name in list(self._held_locks):
            await self.release(lock_name)


# ============================================================================
# Night Shift Scheduler
# ============================================================================

class NightShiftScheduler:
    """
    Main scheduler for automated training.

    Coordinates all scheduling components:
    - Cron-based schedules
    - Resource monitoring
    - Experience thresholds
    - Job arbitration
    """

    def __init__(self, training_trigger_callback: Optional[Callable[..., Awaitable[Any]]] = None):
        self._rules: Dict[str, ScheduleRule] = {}
        self._jobs: Dict[str, ScheduledJob] = {}
        self._resource_monitor = ResourceMonitor()
        self._experience_trigger = ExperienceThresholdTrigger()
        self._arbiter = ScheduleArbiter()
        self._lock = DistributedLock()

        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._training_callback = training_trigger_callback

        # Statistics
        self._triggered_count = 0
        self._skipped_count = 0
        self._failed_count = 0
        self._start_time = time.time()

        # Event callbacks
        self._on_job_started: List[Callable[[ScheduledJob], Awaitable[None]]] = []
        self._on_job_completed: List[Callable[[ScheduledJob], Awaitable[None]]] = []

        # v258.0: Strong references for fire-and-forget background tasks
        self._background_tasks: set = set()

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info("[NightShift] Scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        await self._lock.cleanup()
        logger.info("[NightShift] Scheduler stopped")

    def add_rule(self, rule: ScheduleRule) -> str:
        """Add a schedule rule."""
        self._rules[rule.rule_id] = rule

        # Calculate next run for cron rules
        if rule.schedule_type == ScheduleType.CRON and rule.cron_expression:
            try:
                rule.next_scheduled = CronParser.next_run(rule.cron_expression).timestamp()
            except Exception as e:
                logger.error(f"[NightShift] Invalid cron expression: {e}")

        logger.info(f"[NightShift] Added rule: {rule.name} ({rule.rule_id})")
        return rule.rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a schedule rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rules(self) -> List[ScheduleRule]:
        """Get all schedule rules."""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[ScheduleRule]:
        """Get a specific rule."""
        return self._rules.get(rule_id)

    async def trigger_now(
        self,
        priority: JobPriority = JobPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledJob:
        """Manually trigger training immediately."""
        job = ScheduledJob(
            rule_id="manual",
            priority=priority,
            metadata=metadata or {},
        )
        await self._execute_job(job)
        return job

    async def add_experiences(self, count: int) -> Optional[ScheduledJob]:
        """
        Add experiences and potentially trigger training.

        Returns ScheduledJob if threshold was reached and training triggered.
        """
        should_trigger, current = await self._experience_trigger.add_experiences(count)

        if should_trigger:
            logger.info(f"[NightShift] Experience threshold reached: {current}")
            job = ScheduledJob(
                rule_id="threshold",
                priority=JobPriority.NORMAL,
                metadata={"experience_count": current},
            )
            await self._execute_job(job)
            await self._experience_trigger.reset()
            return job

        return None

    def on_job_started(self, callback: Callable[[ScheduledJob], Awaitable[None]]):
        """Register callback for job started events."""
        self._on_job_started.append(callback)

    def on_job_completed(self, callback: Callable[[ScheduledJob], Awaitable[None]]):
        """Register callback for job completed events."""
        self._on_job_completed.append(callback)

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_schedules()
                await asyncio.sleep(SchedulerConfig.CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[NightShift] Scheduler loop error: {e}")
                await asyncio.sleep(10)

    async def _check_schedules(self):
        """Check all schedule rules and trigger if needed."""
        now = time.time()

        for rule_id, rule in list(self._rules.items()):
            if not rule.enabled:
                continue

            should_trigger = False

            if rule.schedule_type == ScheduleType.CRON:
                if rule.next_scheduled and now >= rule.next_scheduled:
                    should_trigger = True
                    # Calculate next run
                    try:
                        rule.next_scheduled = CronParser.next_run(
                            rule.cron_expression,
                            datetime.fromtimestamp(now),
                        ).timestamp()
                    except Exception as e:
                        logger.error(f"[NightShift] Error calculating next run: {e}")

            elif rule.schedule_type == ScheduleType.INTERVAL:
                if rule.interval_seconds:
                    last = rule.last_triggered or 0
                    if now - last >= rule.interval_seconds:
                        should_trigger = True

            if should_trigger:
                # Try to acquire distributed lock
                lock_name = f"schedule-{rule_id}"
                if await self._lock.acquire(lock_name, timeout=5):
                    try:
                        job = ScheduledJob(
                            rule_id=rule_id,
                            priority=rule.priority,
                            metadata={"rule_name": rule.name},
                        )
                        await self._execute_job(job)
                        rule.last_triggered = now
                    finally:
                        await self._lock.release(lock_name)

    async def _execute_job(self, job: ScheduledJob):
        """Execute a scheduled job."""
        # Check resources
        resources = await self._resource_monitor.get_snapshot()
        can_admit, reason = await self._arbiter.can_admit(job, resources)

        if not can_admit:
            logger.warning(f"[NightShift] Job {job.job_id} skipped: {reason}")
            job.status = JobStatus.SKIPPED
            job.error = reason
            self._skipped_count += 1
            self._jobs[job.job_id] = job
            return

        # Admit job
        await self._arbiter.admit(job)
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._jobs[job.job_id] = job

        # Fire started callbacks
        for callback in self._on_job_started:
            try:
                await callback(job)
            except Exception as e:
                logger.error(f"[NightShift] Callback error: {e}")

        try:
            # Execute training
            if self._training_callback:
                result = await self._training_callback(
                    priority=job.priority.name.lower(),
                    triggered_by="scheduler",
                    metadata=job.metadata,
                )
                job.training_job_id = result.get("job_id") if isinstance(result, dict) else None
                job.result = result if isinstance(result, dict) else {"response": str(result)}

            job.status = JobStatus.COMPLETED
            self._triggered_count += 1

        except Exception as e:
            logger.error(f"[NightShift] Job {job.job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            self._failed_count += 1

            # Retry logic
            if job.retry_count < SchedulerConfig.MAX_RETRIES:
                job.retry_count += 1
                delay = SchedulerConfig.RETRY_BASE_DELAY * (
                    SchedulerConfig.RETRY_EXPONENTIAL_BASE ** (job.retry_count - 1)
                )
                _task = asyncio.create_task(
                    self._retry_job(job, delay),
                    name=f"retry_job_{job.job_id}",
                )
                self._background_tasks.add(_task)
                _task.add_done_callback(self._background_tasks.discard)

        finally:
            job.completed_at = time.time()
            await self._arbiter.release(job.job_id)

            # Fire completed callbacks
            for callback in self._on_job_completed:
                try:
                    await callback(job)
                except Exception as e:
                    logger.error(f"[NightShift] Callback error: {e}")

    async def _retry_job(self, job: ScheduledJob, delay: float):
        """Retry a failed job after delay."""
        await asyncio.sleep(delay)
        logger.info(f"[NightShift] Retrying job {job.job_id} (attempt {job.retry_count})")
        await self._execute_job(job)

    def get_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[ScheduledJob]:
        """Get scheduled jobs."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.scheduled_at, reverse=True)[:limit]

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a specific job."""
        return self._jobs.get(job_id)

    async def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        resources = await self._resource_monitor.get_snapshot()
        exp_status = await self._experience_trigger.get_status()
        running = await self._arbiter.get_running()

        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._start_time,
            "rules_count": len(self._rules),
            "rules_enabled": sum(1 for r in self._rules.values() if r.enabled),
            "jobs_triggered": self._triggered_count,
            "jobs_skipped": self._skipped_count,
            "jobs_failed": self._failed_count,
            "jobs_running": len(running),
            "is_night_shift": CronParser.is_night_shift(),
            "resources": {
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "gpu_percent": resources.gpu_percent,
                "training_allowed": resources.is_training_allowed()[0],
            },
            "experience_trigger": exp_status,
            "next_scheduled": self._get_next_scheduled(),
        }

    def _get_next_scheduled(self) -> Optional[Dict[str, Any]]:
        """Get next scheduled job."""
        next_rule = None
        next_time = float("inf")

        for rule in self._rules.values():
            if rule.enabled and rule.next_scheduled:
                if rule.next_scheduled < next_time:
                    next_time = rule.next_scheduled
                    next_rule = rule

        if next_rule:
            return {
                "rule_id": next_rule.rule_id,
                "rule_name": next_rule.name,
                "scheduled_at": next_time,
                "scheduled_datetime": datetime.fromtimestamp(next_time).isoformat(),
            }
        return None


# ============================================================================
# Predefined Schedule Templates
# ============================================================================

class ScheduleTemplates:
    """Predefined schedule templates for common use cases."""

    @staticmethod
    def nightly() -> ScheduleRule:
        """Daily training at 2 AM."""
        return ScheduleRule(
            name="nightly",
            schedule_type=ScheduleType.CRON,
            cron_expression="0 2 * * *",
            priority=JobPriority.NORMAL,
        )

    @staticmethod
    def night_shift_hourly() -> ScheduleRule:
        """Hourly during night shift hours (10 PM - 6 AM)."""
        return ScheduleRule(
            name="night_shift_hourly",
            schedule_type=ScheduleType.CRON,
            cron_expression="0 22-23,0-6 * * *",
            priority=JobPriority.LOW,
        )

    @staticmethod
    def weekday_morning() -> ScheduleRule:
        """Weekday mornings at 5 AM."""
        return ScheduleRule(
            name="weekday_morning",
            schedule_type=ScheduleType.CRON,
            cron_expression="0 5 * * 1-5",
            priority=JobPriority.NORMAL,
        )

    @staticmethod
    def weekend_intensive() -> ScheduleRule:
        """Every 4 hours on weekends."""
        return ScheduleRule(
            name="weekend_intensive",
            schedule_type=ScheduleType.CRON,
            cron_expression="0 */4 * * 0,6",
            priority=JobPriority.HIGH,
        )

    @staticmethod
    def experience_threshold(threshold: int = 100) -> ScheduleRule:
        """Trigger when experience threshold is reached."""
        return ScheduleRule(
            name=f"exp_threshold_{threshold}",
            schedule_type=ScheduleType.THRESHOLD,
            threshold_value=threshold,
            priority=JobPriority.NORMAL,
        )

    @staticmethod
    def interval(seconds: int, name: Optional[str] = None) -> ScheduleRule:
        """Fixed interval schedule."""
        return ScheduleRule(
            name=name or f"interval_{seconds}s",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=seconds,
            priority=JobPriority.LOW,
        )


# ============================================================================
# Global Scheduler Instance
# ============================================================================

_scheduler: Optional[NightShiftScheduler] = None


def get_scheduler() -> NightShiftScheduler:
    """Get global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = NightShiftScheduler()
    return _scheduler


async def init_scheduler(training_callback: Optional[Callable[..., Awaitable[Any]]] = None) -> NightShiftScheduler:
    """Initialize and start the scheduler with optional training callback."""
    global _scheduler
    _scheduler = NightShiftScheduler(training_trigger_callback=training_callback)
    await _scheduler.start()
    return _scheduler
