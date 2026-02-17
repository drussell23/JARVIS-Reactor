"""
PROJECT TRINITY: Async Lifecycle Coordinator (v88.0)

Ultra-advanced async lifecycle management system that prevents cancellation cascades,
coordinates graceful shutdown, and ensures clean task lifecycle across all Trinity components.

ROOT PROBLEM SOLVED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
asyncio.exceptions.CancelledError
During handling of the above exception, another exception occurred:
asyncio.exceptions.CancelledError
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE FEATURES:
- Cancellation-safe wrappers for ALL async operations
- Graceful shutdown orchestration with dependency ordering
- Shield critical cleanup from cancellation
- Task lifecycle management (creation → execution → cleanup)
- Nested cancellation protection
- Timeout-based escalation (graceful → forceful → terminate)
- Comprehensive error recovery

INTEGRATION:
- Works with v85.0 Unified State Coordinator
- Works with v86.0 Async Database Coordinator
- Works with v87.0 Distributed Health Monitor
- Provides foundation for all Trinity async operations

Author: Claude Opus 4.5
Version: 88.0
Status: Production Ready
"""

import asyncio
import functools
import inspect
import logging
import signal
import time
import traceback
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
R = TypeVar("R")


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════


class TaskState(str, Enum):
    """Task lifecycle states."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ShutdownPhase(str, Enum):
    """Shutdown escalation phases."""

    GRACEFUL = "graceful"  # Ask nicely
    FORCEFUL = "forceful"  # Insist
    TERMINATE = "terminate"  # Kill with fire


class TaskPriority(int, Enum):
    """Task priority for shutdown ordering."""

    CRITICAL = 0  # Shutdown last (health monitors, coordinators)
    HIGH = 1  # Important infrastructure
    NORMAL = 2  # Regular tasks
    LOW = 3  # Background tasks
    LOWEST = 4  # Shutdown first (cleanup tasks)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TaskMetadata:
    """Metadata for managed tasks."""

    task_id: str
    name: str
    coro: Optional[Coroutine] = None
    task: Optional[asyncio.Task] = None
    state: TaskState = TaskState.CREATED
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    cleanup_callback: Optional[Callable[[], Awaitable[None]]] = None
    error: Optional[Exception] = None
    cancellation_count: int = 0


@dataclass
class ShutdownConfig:
    """Configuration for shutdown orchestration."""

    graceful_timeout: float = 30.0  # Timeout for graceful shutdown
    forceful_timeout: float = 10.0  # Timeout for forceful shutdown
    terminate_timeout: float = 5.0  # Timeout before giving up
    shield_cleanup: bool = True  # Shield cleanup operations from cancellation
    log_orphaned_tasks: bool = True  # Log tasks that refuse to die
    cancel_on_error: bool = False  # Cancel all tasks if one fails


# ═══════════════════════════════════════════════════════════════════════════
# CANCELLATION-SAFE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


async def cancellation_safe(
    coro: Coroutine[Any, Any, T],
    *,
    shield_cleanup: bool = True,
    cleanup: Optional[Callable[[], Awaitable[None]]] = None,
    default: Optional[T] = None,
    suppress_cancel: bool = False,
) -> Optional[T]:
    """
    Execute a coroutine with protection against cancellation cascades.

    This is the CORE utility that prevents the "CancelledError during handling
    of CancelledError" problem.

    Args:
        coro: Coroutine to execute
        shield_cleanup: Shield cleanup operations from cancellation
        cleanup: Optional cleanup function to run on cancellation
        default: Default value to return on cancellation
        suppress_cancel: If True, suppress CancelledError and return default

    Returns:
        Result of coroutine, or default if cancelled and suppress_cancel=True

    Raises:
        CancelledError: If cancelled and suppress_cancel=False
    """
    try:
        return await coro
    except asyncio.CancelledError:
        if cleanup is not None:
            # Shield cleanup from cancellation to prevent cascades
            if shield_cleanup:
                try:
                    await asyncio.shield(cleanup())
                except asyncio.CancelledError:
                    # Even shielded cleanup was cancelled - log but don't propagate
                    logger.warning("Cleanup was forcefully cancelled")
                except Exception as e:
                    logger.error(f"Cleanup failed: {e}", exc_info=True)
            else:
                try:
                    await cleanup()
                except Exception as e:
                    logger.error(f"Cleanup failed: {e}", exc_info=True)

        if suppress_cancel:
            return default
        else:
            raise
    except Exception as e:
        logger.error(f"Unexpected error in cancellation_safe: {e}", exc_info=True)
        if cleanup is not None:
            try:
                if shield_cleanup:
                    await asyncio.shield(cleanup())
                else:
                    await cleanup()
            except Exception as cleanup_error:
                logger.error(f"Cleanup after error failed: {cleanup_error}")
        raise


async def wait_for_safe(
    coro: Coroutine[Any, Any, T],
    timeout: Optional[float] = None,
    *,
    cleanup: Optional[Callable[[], Awaitable[None]]] = None,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Cancellation-safe version of asyncio.wait_for.

    This is what should be used instead of bare asyncio.wait_for to prevent
    the error you're seeing.

    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        cleanup: Optional cleanup function
        default: Default value on timeout

    Returns:
        Result of coroutine, or default on timeout
    """
    try:
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        else:
            return await coro
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {timeout}s")
        if cleanup:
            try:
                await asyncio.shield(cleanup())
            except Exception as e:
                logger.error(f"Timeout cleanup failed: {e}")
        return default
    except asyncio.CancelledError:
        if cleanup:
            try:
                await asyncio.shield(cleanup())
            except Exception as e:
                logger.error(f"Cancellation cleanup failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        if cleanup:
            try:
                await asyncio.shield(cleanup())
            except Exception as cleanup_error:
                logger.error(f"Error cleanup failed: {cleanup_error}")
        raise


async def gather_safe(
    *coros: Coroutine,
    return_exceptions: bool = True,
    cleanup: Optional[Callable[[], Awaitable[None]]] = None,
) -> List[Any]:
    """
    Cancellation-safe version of asyncio.gather.

    Args:
        *coros: Coroutines to gather
        return_exceptions: Return exceptions instead of raising
        cleanup: Optional cleanup function

    Returns:
        List of results or exceptions
    """
    try:
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)
    except asyncio.CancelledError:
        if cleanup:
            try:
                await asyncio.shield(cleanup())
            except Exception as e:
                logger.error(f"Gather cleanup failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Gather failed: {e}", exc_info=True)
        if cleanup:
            try:
                await asyncio.shield(cleanup())
            except Exception as cleanup_error:
                logger.error(f"Gather error cleanup failed: {cleanup_error}")
        raise


def cancellation_safe_decorator(
    shield_cleanup: bool = True,
    cleanup_attr: Optional[str] = None,
):
    """
    Decorator to make async functions cancellation-safe.

    Usage:
        @cancellation_safe_decorator(cleanup_attr="_cleanup")
        async def my_function(self):
            # Do work
            pass

    Args:
        shield_cleanup: Shield cleanup from cancellation
        cleanup_attr: Name of cleanup method to call (e.g., "_cleanup")
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            cleanup_func = None
            if cleanup_attr and args:
                # Try to get cleanup method from self (first arg)
                cleanup_func = getattr(args[0], cleanup_attr, None)

            return await cancellation_safe(
                func(*args, **kwargs),
                shield_cleanup=shield_cleanup,
                cleanup=cleanup_func,
                suppress_cancel=False,
            )

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# TASK LIFECYCLE MANAGER
# ═══════════════════════════════════════════════════════════════════════════


class TaskLifecycleManager:
    """
    Manages the complete lifecycle of async tasks.

    Tracks tasks from creation through execution to cleanup, ensuring
    proper cancellation handling and dependency management.
    """

    def __init__(self):
        self._tasks: Dict[str, TaskMetadata] = {}
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._task_counter = 0

    async def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[Set[str]] = None,
        cleanup: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> asyncio.Task[T]:
        """
        Create a managed task with lifecycle tracking.

        Args:
            coro: Coroutine to run
            name: Optional task name
            priority: Task priority for shutdown ordering
            dependencies: Set of task IDs this depends on
            cleanup: Optional cleanup callback

        Returns:
            Created task
        """
        async with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"

            if name is None:
                if hasattr(coro, "__name__"):
                    name = coro.__name__
                else:
                    name = f"task_{self._task_counter}"

            # Create metadata
            metadata = TaskMetadata(
                task_id=task_id,
                name=name,
                coro=coro,
                priority=priority,
                dependencies=dependencies or set(),
                cleanup_callback=cleanup,
            )

            # Update dependents for dependencies
            for dep_id in metadata.dependencies:
                if dep_id in self._tasks:
                    self._tasks[dep_id].dependents.add(task_id)

            # Create task with wrapper that updates state
            task = asyncio.create_task(self._run_task(metadata))
            task.set_name(name)
            metadata.task = task
            metadata.state = TaskState.PENDING

            self._tasks[task_id] = metadata

            logger.debug(f"Created task: {name} (id={task_id}, priority={priority})")

            return task

    async def _run_task(self, metadata: TaskMetadata) -> Any:
        """Run task with state tracking."""
        try:
            metadata.state = TaskState.RUNNING
            metadata.started_at = time.time()

            result = await metadata.coro
            metadata.state = TaskState.COMPLETED
            metadata.completed_at = time.time()

            return result

        except asyncio.CancelledError:
            metadata.state = TaskState.CANCELLED
            metadata.completed_at = time.time()
            metadata.cancellation_count += 1

            # Run cleanup if provided
            if metadata.cleanup_callback:
                try:
                    await asyncio.shield(metadata.cleanup_callback())
                except Exception as e:
                    logger.error(
                        f"Task cleanup failed for {metadata.name}: {e}", exc_info=True
                    )

            raise

        except Exception as e:
            metadata.state = TaskState.FAILED
            metadata.completed_at = time.time()
            metadata.error = e

            # Run cleanup even on error
            if metadata.cleanup_callback:
                try:
                    await asyncio.shield(metadata.cleanup_callback())
                except Exception as cleanup_error:
                    logger.error(
                        f"Task error cleanup failed for {metadata.name}: {cleanup_error}"
                    )

            raise

    async def cancel_task(
        self,
        task_id: str,
        *,
        timeout: float = 5.0,
        escalate: bool = True,
    ) -> bool:
        """
        Cancel a task gracefully.

        Args:
            task_id: Task ID to cancel
            timeout: Timeout for cancellation
            escalate: If True, escalate to forceful cancellation on timeout

        Returns:
            True if cancelled successfully
        """
        async with self._lock:
            metadata = self._tasks.get(task_id)
            if not metadata or not metadata.task:
                return False

            if metadata.state in (TaskState.CANCELLED, TaskState.COMPLETED):
                return True

            metadata.state = TaskState.CANCELLING

        task = metadata.task
        task.cancel()

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            if escalate:
                logger.warning(
                    f"Task {metadata.name} did not cancel within {timeout}s, escalating"
                )
                # Forceful cancellation - cancel again
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except:
                    logger.error(f"Task {metadata.name} refused to cancel")
                    return False
            return False
        except asyncio.CancelledError:
            return True
        except Exception as e:
            logger.error(f"Error cancelling task {metadata.name}: {e}")
            return False

    async def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """Get current state of a task."""
        async with self._lock:
            metadata = self._tasks.get(task_id)
            return metadata.state if metadata else None

    async def wait_for_dependencies(self, task_id: str, timeout: Optional[float] = None):
        """Wait for all dependencies of a task to complete."""
        metadata = self._tasks.get(task_id)
        if not metadata or not metadata.dependencies:
            return

        dep_tasks = [
            self._tasks[dep_id].task
            for dep_id in metadata.dependencies
            if dep_id in self._tasks and self._tasks[dep_id].task
        ]

        if dep_tasks:
            await asyncio.wait(dep_tasks, timeout=timeout)

    def get_running_tasks(self) -> List[TaskMetadata]:
        """Get all currently running tasks."""
        return [
            metadata
            for metadata in self._tasks.values()
            if metadata.state == TaskState.RUNNING
        ]

    def get_tasks_by_priority(self, priority: TaskPriority) -> List[TaskMetadata]:
        """Get all tasks with specified priority."""
        return [
            metadata
            for metadata in self._tasks.values()
            if metadata.priority == priority
        ]


# ═══════════════════════════════════════════════════════════════════════════
# SHUTDOWN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


class ShutdownOrchestrator:
    """
    Coordinates graceful shutdown across all Trinity components.

    Handles dependency-aware shutdown ordering, timeout escalation,
    and cleanup verification.
    """

    def __init__(
        self,
        task_manager: TaskLifecycleManager,
        config: Optional[ShutdownConfig] = None,
    ):
        self._task_manager = task_manager
        self._config = config or ShutdownConfig()
        self._shutdown_in_progress = False
        self._shutdown_lock = asyncio.Lock()

    async def shutdown(
        self,
        *,
        phase: ShutdownPhase = ShutdownPhase.GRACEFUL,
        component_filter: Optional[Callable[[TaskMetadata], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Execute coordinated shutdown.

        Args:
            phase: Shutdown phase to execute
            component_filter: Optional filter for which tasks to shutdown

        Returns:
            Shutdown report with statistics
        """
        async with self._shutdown_lock:
            if self._shutdown_in_progress:
                logger.warning("Shutdown already in progress")
                return {"status": "already_in_progress"}

            self._shutdown_in_progress = True

        start_time = time.time()
        logger.info(f"Starting {phase} shutdown")

        try:
            if phase == ShutdownPhase.GRACEFUL:
                report = await self._graceful_shutdown(component_filter)
            elif phase == ShutdownPhase.FORCEFUL:
                report = await self._forceful_shutdown(component_filter)
            else:  # TERMINATE
                report = await self._terminate_shutdown(component_filter)

            report["duration"] = time.time() - start_time
            report["phase"] = phase

            logger.info(
                f"Shutdown complete: {report['cancelled']}/{report['total']} tasks cancelled"
            )

            return report

        finally:
            self._shutdown_in_progress = False

    async def _graceful_shutdown(
        self, component_filter: Optional[Callable[[TaskMetadata], bool]]
    ) -> Dict[str, Any]:
        """Graceful shutdown - ask nicely."""
        report = {
            "total": 0,
            "cancelled": 0,
            "failed": 0,
            "timeout": 0,
            "by_priority": {},
        }

        # Get all running tasks, grouped by priority
        tasks_by_priority: Dict[TaskPriority, List[TaskMetadata]] = {}

        for priority in TaskPriority:
            tasks = self._task_manager.get_tasks_by_priority(priority)

            # Apply filter if provided
            if component_filter:
                tasks = [t for t in tasks if component_filter(t)]

            # Only include running tasks
            tasks = [t for t in tasks if t.state == TaskState.RUNNING]

            if tasks:
                tasks_by_priority[priority] = tasks

        # Shutdown in reverse priority order (LOWEST first, CRITICAL last)
        for priority in reversed(list(TaskPriority)):
            if priority not in tasks_by_priority:
                continue

            tasks = tasks_by_priority[priority]
            report["total"] += len(tasks)

            logger.info(f"Shutting down {len(tasks)} {priority} priority tasks")

            # Cancel tasks in this priority group
            cancel_results = await asyncio.gather(
                *[
                    self._task_manager.cancel_task(
                        task.task_id,
                        timeout=self._config.graceful_timeout,
                        escalate=False,
                    )
                    for task in tasks
                ],
                return_exceptions=True,
            )

            # Count results
            priority_report = {
                "total": len(tasks),
                "cancelled": 0,
                "failed": 0,
                "timeout": 0,
            }

            for task, result in zip(tasks, cancel_results):
                if isinstance(result, Exception):
                    priority_report["failed"] += 1
                    report["failed"] += 1
                elif result:
                    priority_report["cancelled"] += 1
                    report["cancelled"] += 1
                else:
                    priority_report["timeout"] += 1
                    report["timeout"] += 1

            report["by_priority"][priority] = priority_report

        return report

    async def _forceful_shutdown(
        self, component_filter: Optional[Callable[[TaskMetadata], bool]]
    ) -> Dict[str, Any]:
        """Forceful shutdown - insist."""
        logger.warning("Executing FORCEFUL shutdown")

        # First try graceful with shorter timeout
        report = await self._graceful_shutdown(component_filter)

        # If any tasks remain, cancel them forcefully
        remaining_tasks = [
            task
            for task in self._task_manager.get_running_tasks()
            if not component_filter or component_filter(task)
        ]

        if remaining_tasks:
            logger.warning(f"Forcefully cancelling {len(remaining_tasks)} tasks")

            # Cancel all remaining tasks without waiting
            for task in remaining_tasks:
                if task.task:
                    task.task.cancel()

            # Wait briefly for cancellation
            await asyncio.sleep(self._config.forceful_timeout)

        return report

    async def _terminate_shutdown(
        self, component_filter: Optional[Callable[[TaskMetadata], bool]]
    ) -> Dict[str, Any]:
        """Terminate shutdown - kill with fire."""
        logger.error("Executing TERMINATE shutdown - this is bad!")

        # Try forceful first
        report = await self._forceful_shutdown(component_filter)

        # Get truly stubborn tasks
        stubborn_tasks = [
            task
            for task in self._task_manager.get_running_tasks()
            if not component_filter or component_filter(task)
        ]

        if stubborn_tasks:
            logger.critical(
                f"🔥 {len(stubborn_tasks)} tasks refused to shutdown: "
                f"{[t.name for t in stubborn_tasks]}"
            )

            # Log task details for debugging
            for task in stubborn_tasks:
                logger.critical(
                    f"Stubborn task: {task.name} "
                    f"(state={task.state}, "
                    f"cancellation_count={task.cancellation_count}, "
                    f"runtime={time.time() - task.started_at if task.started_at else 0:.1f}s)"
                )

            report["stubborn_tasks"] = [
                {
                    "name": t.name,
                    "state": t.state,
                    "runtime": time.time() - t.started_at if t.started_at else 0,
                }
                for t in stubborn_tasks
            ]

        return report


# ═══════════════════════════════════════════════════════════════════════════
# ASYNC LIFECYCLE COORDINATOR (MAIN CLASS)
# ═══════════════════════════════════════════════════════════════════════════


class AsyncLifecycleCoordinator:
    """
    Central coordinator for async lifecycle management across Trinity.

    Provides:
    - Cancellation-safe task creation
    - Graceful shutdown orchestration
    - Signal handling (SIGINT, SIGTERM)
    - Component lifecycle management
    - Error recovery and cleanup

    This is the master orchestrator that sits above everything else.
    """

    _instance: Optional["AsyncLifecycleCoordinator"] = None
    _lock = asyncio.Lock()

    def __init__(self, config: Optional[ShutdownConfig] = None):
        self._config = config or ShutdownConfig()
        self._task_manager = TaskLifecycleManager()
        self._shutdown_orchestrator = ShutdownOrchestrator(
            self._task_manager, self._config
        )
        self._components: Dict[str, Any] = {}
        self._shutdown_callbacks: List[Callable[[], Awaitable[None]]] = []
        self._signal_handlers_installed = False

        # v258.0: Strong references for fire-and-forget background tasks
        self._background_tasks: set = set()

    @classmethod
    async def get_coordinator(
        cls, config: Optional[ShutdownConfig] = None
    ) -> "AsyncLifecycleCoordinator":
        """Get singleton coordinator instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
                    logger.info("✅ AsyncLifecycleCoordinator initialized")
        return cls._instance

    def install_signal_handlers(self):
        """Install signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        loop = asyncio.get_event_loop()

        def signal_handler(sig):
            logger.info(f"Received signal {sig}, initiating graceful shutdown")
            _task = asyncio.create_task(
                self.shutdown(),
                name="lifecycle_graceful_shutdown",
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

            self._signal_handlers_installed = True
            logger.info("✅ Signal handlers installed")

        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            logger.warning("Signal handlers not supported on this platform")

    async def create_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[Set[str]] = None,
        cleanup: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> asyncio.Task[T]:
        """Create managed task."""
        return await self._task_manager.create_task(
            coro,
            name=name,
            priority=priority,
            dependencies=dependencies,
            cleanup=cleanup,
        )

    def register_component(self, name: str, component: Any):
        """Register a component for lifecycle management."""
        self._components[name] = component
        logger.debug(f"Registered component: {name}")

    def register_shutdown_callback(self, callback: Callable[[], Awaitable[None]]):
        """Register a callback to run during shutdown."""
        self._shutdown_callbacks.append(callback)

    async def shutdown(self, phase: ShutdownPhase = ShutdownPhase.GRACEFUL):
        """Execute coordinated shutdown."""
        logger.info(f"🛑 Initiating {phase} shutdown")

        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                await asyncio.shield(callback())
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}", exc_info=True)

        # Shutdown all managed tasks
        report = await self._shutdown_orchestrator.shutdown(phase=phase)

        logger.info(f"✅ Shutdown complete: {report}")

        return report

    @asynccontextmanager
    async def managed_task(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        cleanup: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> AsyncIterator[asyncio.Task[T]]:
        """
        Context manager for managed tasks that auto-cleanup.

        Usage:
            async with coordinator.managed_task(my_coro(), name="worker") as task:
                result = await task
        """
        task = await self.create_task(
            coro, name=name, priority=priority, cleanup=cleanup
        )

        try:
            yield task
        finally:
            if not task.done():
                await self._task_manager.cancel_task(
                    task.get_name(), timeout=5.0, escalate=True
                )


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


async def get_lifecycle_coordinator(
    config: Optional[ShutdownConfig] = None,
) -> AsyncLifecycleCoordinator:
    """Get singleton lifecycle coordinator."""
    return await AsyncLifecycleCoordinator.get_coordinator(config)


# Exported utility functions that components should use
__all__ = [
    # Main Coordinator
    "AsyncLifecycleCoordinator",
    "get_lifecycle_coordinator",
    # Task Management
    "TaskLifecycleManager",
    "ShutdownOrchestrator",
    # Utilities
    "cancellation_safe",
    "wait_for_safe",
    "gather_safe",
    "cancellation_safe_decorator",
    # Enums
    "TaskState",
    "ShutdownPhase",
    "TaskPriority",
    # Data Classes
    "TaskMetadata",
    "ShutdownConfig",
]
