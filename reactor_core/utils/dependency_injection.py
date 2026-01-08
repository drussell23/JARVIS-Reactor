"""
Advanced Dependency Injection System for JARVIS Reactor.

Provides:
- Service locator pattern with lazy initialization
- Lifecycle management (singleton, transient, scoped)
- Circular dependency detection
- Factory functions and builders
- Async-safe service resolution
- Health checks and graceful degradation
- Service versioning and compatibility

This is production-grade DI inspired by Spring, .NET Core DI, and Python Injector.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
TService = TypeVar("TService")


# =============================================================================
# SERVICE LIFECYCLE
# =============================================================================

class ServiceLifetime(Enum):
    """Service lifetime scopes."""

    SINGLETON = auto()  # One instance for entire application
    SCOPED = auto()      # One instance per scope (request, transaction, etc.)
    TRANSIENT = auto()   # New instance every time


class ServiceStatus(Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceDescriptor:
    """
    Describes how a service should be created and managed.

    Attributes:
        service_type: The interface/abstract class type
        implementation_type: The concrete implementation class
        factory: Factory function to create the service
        lifetime: Service lifetime scope
        dependencies: List of dependency types
        metadata: Additional metadata (version, tags, etc.)
    """

    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate descriptor."""
        if self.implementation_type is None and self.factory is None:
            raise ValueError(
                f"ServiceDescriptor for {self.service_type} must have either "
                "implementation_type or factory"
            )


# =============================================================================
# SERVICE HEALTH
# =============================================================================

class HealthCheck(Protocol):
    """Protocol for service health checks."""

    async def check_health(self) -> Tuple[ServiceStatus, str]:
        """
        Check service health.

        Returns:
            Tuple of (status, message)
        """
        ...


# =============================================================================
# DEPENDENCY INJECTION CONTAINER
# =============================================================================

class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class ServiceNotFoundError(Exception):
    """Raised when requested service is not registered."""
    pass


class DependencyInjectionContainer:
    """
    Advanced dependency injection container.

    Features:
    - Service registration with lifetime management
    - Automatic dependency resolution
    - Circular dependency detection
    - Lazy initialization
    - Async-safe service creation
    - Health checking
    - Graceful degradation
    - Service versioning

    Example:
        >>> container = DependencyInjectionContainer()
        >>>
        >>> # Register services
        >>> container.register(
        ...     IDatabase,
        ...     PostgresDatabase,
        ...     lifetime=ServiceLifetime.SINGLETON
        ... )
        >>> container.register(
        ...     ICache,
        ...     RedisCache,
        ...     lifetime=ServiceLifetime.SINGLETON
        ... )
        >>> container.register(
        ...     UserService,
        ...     lifetime=ServiceLifetime.SCOPED
        ... )
        >>>
        >>> # Resolve service (dependencies auto-injected)
        >>> user_service = await container.resolve(UserService)
    """

    def __init__(self):
        self._descriptors: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._lock = threading.RLock()
        self._resolution_stack: Set[Type] = set()
        self._health_checks: Dict[Type, Callable] = {}

    def register(
        self,
        service_type: Type[TService],
        implementation_type: Optional[Type[TService]] = None,
        factory: Optional[Callable[..., TService]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
        **metadata,
    ) -> None:
        """
        Register a service.

        Args:
            service_type: Interface or abstract class
            implementation_type: Concrete implementation
            factory: Factory function to create instance
            lifetime: Service lifetime scope
            **metadata: Additional metadata (version, tags, etc.)
        """
        with self._lock:
            # Auto-detect dependencies from type hints
            dependencies = self._extract_dependencies(
                implementation_type or factory  # type: ignore
            )

            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                lifetime=lifetime,
                dependencies=dependencies,
                metadata=metadata,
            )

            self._descriptors[service_type] = descriptor
            logger.debug(
                f"Registered {service_type.__name__} "
                f"(lifetime={lifetime.name}, deps={len(dependencies)})"
            )

    def register_singleton(
        self,
        service_type: Type[TService],
        instance: TService,
    ) -> None:
        """
        Register an existing instance as a singleton.

        Args:
            service_type: Service type
            instance: Service instance
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=lambda: instance,
                lifetime=ServiceLifetime.SINGLETON,
            )
            self._descriptors[service_type] = descriptor
            self._singletons[service_type] = instance
            logger.debug(f"Registered singleton instance of {service_type.__name__}")

    async def resolve(
        self,
        service_type: Type[TService],
        scope_id: Optional[str] = None,
    ) -> TService:
        """
        Resolve a service instance.

        Args:
            service_type: Service type to resolve
            scope_id: Scope identifier for SCOPED services

        Returns:
            Service instance with all dependencies injected

        Raises:
            ServiceNotFoundError: Service not registered
            CircularDependencyError: Circular dependencies detected
        """
        # Check if service is registered
        if service_type not in self._descriptors:
            raise ServiceNotFoundError(
                f"Service {service_type.__name__} is not registered"
            )

        descriptor = self._descriptors[service_type]

        # SINGLETON: Return cached instance or create new one
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            with self._lock:
                if service_type not in self._singletons:
                    instance = await self._create_instance(descriptor)
                    self._singletons[service_type] = instance
                return self._singletons[service_type]

        # SCOPED: Return cached instance for this scope or create new one
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if scope_id is None:
                raise ValueError(
                    f"Scope ID required for SCOPED service {service_type.__name__}"
                )

            with self._lock:
                if scope_id not in self._scoped_instances:
                    self._scoped_instances[scope_id] = {}

                scoped_cache = self._scoped_instances[scope_id]
                if service_type not in scoped_cache:
                    instance = await self._create_instance(descriptor, scope_id)
                    scoped_cache[service_type] = instance

                return scoped_cache[service_type]

        # TRANSIENT: Always create new instance
        else:
            return await self._create_instance(descriptor, scope_id)

    async def _create_instance(
        self,
        descriptor: ServiceDescriptor,
        scope_id: Optional[str] = None,
    ) -> Any:
        """
        Create a service instance with dependency injection.

        Args:
            descriptor: Service descriptor
            scope_id: Scope identifier

        Returns:
            Service instance
        """
        # Detect circular dependencies
        if descriptor.service_type in self._resolution_stack:
            cycle = " -> ".join(t.__name__ for t in self._resolution_stack)
            cycle += f" -> {descriptor.service_type.__name__}"
            raise CircularDependencyError(
                f"Circular dependency detected: {cycle}"
            )

        self._resolution_stack.add(descriptor.service_type)

        try:
            # Resolve dependencies
            dependencies = {}
            for dep_type in descriptor.dependencies:
                dep_instance = await self.resolve(dep_type, scope_id)
                dependencies[dep_type] = dep_instance

            # Create instance
            if descriptor.factory:
                # Call factory function
                instance = descriptor.factory(**dependencies)

                # Await if factory is async
                if inspect.iscoroutine(instance):
                    instance = await instance

            elif descriptor.implementation_type:
                # Instantiate class with dependencies
                instance = descriptor.implementation_type(**dependencies)

            else:
                raise ValueError(
                    f"No factory or implementation type for {descriptor.service_type}"
                )

            logger.debug(
                f"Created instance of {descriptor.service_type.__name__} "
                f"with {len(dependencies)} dependencies"
            )

            return instance

        finally:
            self._resolution_stack.remove(descriptor.service_type)

    def _extract_dependencies(self, cls_or_func: Any) -> List[Type]:
        """
        Extract dependency types from class __init__ or function signature.

        Args:
            cls_or_func: Class or function

        Returns:
            List of dependency types
        """
        # Get signature
        if inspect.isclass(cls_or_func):
            sig = inspect.signature(cls_or_func.__init__)
        else:
            sig = inspect.signature(cls_or_func)

        # Extract type hints (skip 'self' and parameters without annotations)
        dependencies = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                dependencies.append(param.annotation)

        return dependencies

    @contextmanager
    def create_scope(self) -> Iterator[str]:
        """
        Create a new dependency injection scope.

        Yields:
            Scope ID

        Example:
            >>> with container.create_scope() as scope_id:
            ...     service = await container.resolve(MyService, scope_id)
            ...     await service.do_work()
            ... # Scoped instances are cleaned up after this block
        """
        import uuid
        scope_id = str(uuid.uuid4())

        try:
            yield scope_id
        finally:
            # Clean up scoped instances
            with self._lock:
                if scope_id in self._scoped_instances:
                    # Call cleanup methods if they exist
                    for instance in self._scoped_instances[scope_id].values():
                        if hasattr(instance, "cleanup"):
                            try:
                                instance.cleanup()
                            except Exception as e:
                                logger.warning(
                                    f"Error cleaning up {type(instance).__name__}: {e}"
                                )

                    del self._scoped_instances[scope_id]

    async def check_health(
        self,
        service_type: Type,
    ) -> Tuple[ServiceStatus, str]:
        """
        Check health of a service.

        Args:
            service_type: Service type to check

        Returns:
            Tuple of (status, message)
        """
        try:
            instance = await self.resolve(service_type)

            # If service implements HealthCheck protocol
            if hasattr(instance, "check_health"):
                return await instance.check_health()

            # Default: service is healthy if it can be resolved
            return ServiceStatus.HEALTHY, "Service resolved successfully"

        except Exception as e:
            return ServiceStatus.UNHEALTHY, f"Failed to resolve: {e}"

    def get_service_graph(self) -> Dict[str, List[str]]:
        """
        Get dependency graph of all registered services.

        Returns:
            Dict mapping service names to their dependency names
        """
        graph = {}
        for service_type, descriptor in self._descriptors.items():
            graph[service_type.__name__] = [
                dep.__name__ for dep in descriptor.dependencies
            ]
        return graph

    def validate_registrations(self) -> List[str]:
        """
        Validate all service registrations.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        for service_type, descriptor in self._descriptors.items():
            # Check that all dependencies are registered
            for dep_type in descriptor.dependencies:
                if dep_type not in self._descriptors:
                    errors.append(
                        f"{service_type.__name__} depends on unregistered "
                        f"service {dep_type.__name__}"
                    )

            # Check for circular dependencies
            try:
                visited = set()
                self._check_circular_deps(service_type, visited, [])
            except CircularDependencyError as e:
                errors.append(str(e))

        return errors

    def _check_circular_deps(
        self,
        service_type: Type,
        visited: Set[Type],
        path: List[Type],
    ) -> None:
        """Check for circular dependencies recursively."""
        if service_type in path:
            cycle = " -> ".join(t.__name__ for t in path)
            cycle += f" -> {service_type.__name__}"
            raise CircularDependencyError(f"Circular dependency: {cycle}")

        if service_type in visited:
            return

        visited.add(service_type)

        if service_type in self._descriptors:
            descriptor = self._descriptors[service_type]
            for dep_type in descriptor.dependencies:
                self._check_circular_deps(dep_type, visited, path + [service_type])


# =============================================================================
# GLOBAL CONTAINER
# =============================================================================

_global_container: Optional[DependencyInjectionContainer] = None
_global_container_lock = threading.Lock()


def get_container() -> DependencyInjectionContainer:
    """Get the global dependency injection container."""
    global _global_container

    if _global_container is None:
        with _global_container_lock:
            if _global_container is None:
                _global_container = DependencyInjectionContainer()

    return _global_container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    with _global_container_lock:
        _global_container = None


# =============================================================================
# DECORATORS
# =============================================================================

def injectable(
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    **metadata,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to mark a class as injectable.

    Args:
        lifetime: Service lifetime
        **metadata: Additional metadata

    Example:
        >>> @injectable(lifetime=ServiceLifetime.SINGLETON)
        ... class UserService:
        ...     def __init__(self, db: IDatabase, cache: ICache):
        ...         self.db = db
        ...         self.cache = cache
        >>>
        >>> # Auto-registered with container
        >>> service = await get_container().resolve(UserService)
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Auto-register with global container
        get_container().register(
            cls,
            implementation_type=cls,
            lifetime=lifetime,
            **metadata,
        )
        return cls

    return decorator


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    "DependencyInjectionContainer",
    "ServiceLifetime",
    "ServiceStatus",
    "ServiceDescriptor",
    "HealthCheck",
    "CircularDependencyError",
    "ServiceNotFoundError",
    "get_container",
    "reset_container",
    "injectable",
]
