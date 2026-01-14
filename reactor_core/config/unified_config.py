"""
Unified Cross-Repository Configuration System.

This module provides a centralized configuration system that works across:
- reactor-core (Night Shift Training Engine)
- JARVIS-AI-Agent (Main AGI system)
- jarvis-prime (Cloud deployment)

Features:
- Environment variable cascading with intelligent defaults
- Cross-repo path discovery
- Service endpoint management
- Shared secrets management
- Runtime configuration updates via Redis/file
- Connection pooling for HTTP and Redis clients
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceType(Enum):
    """Types of services in the JARVIS ecosystem."""
    JARVIS_AGENT = "jarvis_agent"       # Main JARVIS-AI-Agent
    JARVIS_PRIME = "jarvis_prime"       # Cloud deployment
    REACTOR_CORE = "reactor_core"       # Night Shift Training
    SCOUT = "scout"                      # Safe Scout web ingestion
    REDIS = "redis"                      # Cache/queue
    POSTGRES = "postgres"                # Database


class Environment(Enum):
    """Deployment environment."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ServiceEndpoint:
    """Configuration for a service endpoint."""
    service_type: ServiceType
    host: str
    port: int
    protocol: str = "http"
    api_version: str = "v1"
    health_path: str = "/health"
    is_available: bool = False
    last_check: Optional[datetime] = None

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/api/{self.api_version}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"

    @property
    def ws_url(self) -> str:
        ws_protocol = "wss" if self.protocol == "https" else "ws"
        return f"{ws_protocol}://{self.host}:{self.port}/ws"


# =============================================================================
# CONNECTION POOLING
# =============================================================================

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""
    # HTTP connection pool settings
    http_pool_size: int = field(
        default_factory=lambda: int(os.getenv("HTTP_POOL_SIZE", "100"))
    )
    http_pool_per_host: int = field(
        default_factory=lambda: int(os.getenv("HTTP_POOL_PER_HOST", "10"))
    )
    http_keepalive_timeout: float = field(
        default_factory=lambda: float(os.getenv("HTTP_KEEPALIVE_TIMEOUT", "30.0"))
    )
    http_connect_timeout: float = field(
        default_factory=lambda: float(os.getenv("HTTP_CONNECT_TIMEOUT", "10.0"))
    )
    http_read_timeout: float = field(
        default_factory=lambda: float(os.getenv("HTTP_READ_TIMEOUT", "30.0"))
    )

    # Redis connection pool settings
    redis_pool_size: int = field(
        default_factory=lambda: int(os.getenv("REDIS_POOL_SIZE", "10"))
    )
    redis_pool_min_size: int = field(
        default_factory=lambda: int(os.getenv("REDIS_POOL_MIN_SIZE", "2"))
    )


class HTTPConnectionPool:
    """
    Managed HTTP connection pool using aiohttp.

    Features:
    - Singleton pattern per configuration
    - Automatic session lifecycle management
    - Connection reuse and keepalive
    - Thread-safe initialization
    """

    _instances: Dict[str, "HTTPConnectionPool"] = {}
    _lock: asyncio.Lock = None

    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        self.config = config or ConnectionPoolConfig()
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @classmethod
    async def get_instance(
        cls,
        name: str = "default",
        config: Optional[ConnectionPoolConfig] = None,
    ) -> "HTTPConnectionPool":
        """Get or create a named connection pool instance."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(config)
            return cls._instances[name]

    async def _ensure_session(self) -> Any:
        """Ensure the session is initialized."""
        if self._session is None or self._session.closed:
            async with self._init_lock:
                if self._session is None or self._session.closed:
                    await self._create_session()
        return self._session

    async def _create_session(self) -> None:
        """Create a new aiohttp session with connection pooling."""
        try:
            import aiohttp

            # Configure connection pool
            connector = aiohttp.TCPConnector(
                limit=self.config.http_pool_size,
                limit_per_host=self.config.http_pool_per_host,
                keepalive_timeout=self.config.http_keepalive_timeout,
                enable_cleanup_closed=True,
                force_close=False,
            )

            # Configure timeout
            timeout = aiohttp.ClientTimeout(
                connect=self.config.http_connect_timeout,
                total=self.config.http_read_timeout,
            )

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            self._initialized = True
            logger.debug("HTTP connection pool created")

        except ImportError:
            logger.warning("aiohttp not available, HTTP pool disabled")
            self._session = None

    async def get(
        self,
        url: str,
        **kwargs,
    ) -> Any:
        """Make a GET request using the pooled session."""
        session = await self._ensure_session()
        if session is None:
            raise RuntimeError("HTTP session not available")
        return await session.get(url, **kwargs)

    async def post(
        self,
        url: str,
        **kwargs,
    ) -> Any:
        """Make a POST request using the pooled session."""
        session = await self._ensure_session()
        if session is None:
            raise RuntimeError("HTTP session not available")
        return await session.post(url, **kwargs)

    @asynccontextmanager
    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> AsyncIterator[Any]:
        """Context manager for making requests with automatic cleanup."""
        session = await self._ensure_session()
        if session is None:
            raise RuntimeError("HTTP session not available")

        async with session.request(method, url, **kwargs) as response:
            yield response

    async def close(self) -> None:
        """Close the session and release connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            self._initialized = False
            logger.debug("HTTP connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if self._session is None:
            return {"status": "not_initialized"}

        connector = self._session.connector
        if connector is None:
            return {"status": "no_connector"}

        return {
            "status": "active",
            "limit": connector.limit,
            "limit_per_host": connector.limit_per_host,
        }

    @classmethod
    async def close_all(cls) -> None:
        """Close all pool instances."""
        if cls._lock is None:
            return

        async with cls._lock:
            for name, pool in list(cls._instances.items()):
                try:
                    await pool.close()
                except Exception as e:
                    logger.warning(f"Error closing pool {name}: {e}")
            cls._instances.clear()


class RedisConnectionPool:
    """
    Managed Redis connection pool.

    Features:
    - Lazy initialization
    - Connection reuse
    - Automatic reconnection
    """

    _instance: Optional["RedisConnectionPool"] = None
    _lock: asyncio.Lock = None

    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        self.config = config or ConnectionPoolConfig()
        self._pool = None
        self._redis = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @classmethod
    async def get_instance(
        cls,
        config: Optional[ConnectionPoolConfig] = None,
    ) -> "RedisConnectionPool":
        """Get the singleton Redis pool instance."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    async def _ensure_pool(self, host: str = "localhost", port: int = 6379) -> Any:
        """Ensure the Redis pool is initialized."""
        if self._pool is None:
            async with self._init_lock:
                if self._pool is None:
                    await self._create_pool(host, port)
        return self._pool

    async def _create_pool(self, host: str, port: int) -> None:
        """Create a Redis connection pool."""
        try:
            import redis.asyncio as redis

            self._pool = redis.ConnectionPool(
                host=host,
                port=port,
                max_connections=self.config.redis_pool_size,
                decode_responses=True,
            )
            self._redis = redis.Redis(connection_pool=self._pool)
            self._initialized = True
            logger.debug(f"Redis connection pool created ({host}:{port})")

        except ImportError:
            logger.warning("redis library not available, Redis pool disabled")
            self._pool = None
            self._redis = None

    async def get_client(
        self,
        host: str = "localhost",
        port: int = 6379,
    ) -> Any:
        """Get a Redis client from the pool."""
        await self._ensure_pool(host, port)
        if self._redis is None:
            raise RuntimeError("Redis not available")
        return self._redis

    async def close(self) -> None:
        """Close the Redis pool."""
        if self._redis:
            await self._redis.close()
            self._redis = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        self._initialized = False
        logger.debug("Redis connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis pool statistics."""
        if self._pool is None:
            return {"status": "not_initialized"}

        return {
            "status": "active",
            "max_connections": self.config.redis_pool_size,
        }

    @classmethod
    async def close_instance(cls) -> None:
        """Close the singleton instance."""
        if cls._lock is None:
            return

        async with cls._lock:
            if cls._instance:
                await cls._instance.close()
                cls._instance = None


@dataclass
class RepoConfig:
    """Configuration for a repository."""
    name: str
    path: Path
    exists: bool = False
    log_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    config_file: Optional[Path] = None

    def validate(self) -> bool:
        """Validate the repository exists and is configured."""
        self.exists = self.path.exists() and self.path.is_dir()
        if self.exists:
            # Auto-discover common directories
            if not self.log_dir:
                candidates = [
                    self.path / "backend" / "logs",
                    self.path / "logs",
                    self.path / "data" / "logs",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        self.log_dir = candidate
                        break

            if not self.data_dir:
                candidates = [
                    self.path / "backend" / "data",
                    self.path / "data",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        self.data_dir = candidate
                        break

        return self.exists


@dataclass
class UnifiedConfig:
    """
    Unified configuration for the JARVIS ecosystem.

    This class provides centralized configuration management across
    all JARVIS repositories with intelligent defaults and environment
    variable overrides.
    """
    # Environment
    environment: Environment = field(
        default_factory=lambda: Environment(
            os.getenv("JARVIS_ENV", "local").lower()
        )
    )

    # Base paths (auto-discovered)
    repos_base: Path = field(
        default_factory=lambda: Path(
            os.getenv("JARVIS_REPOS_BASE", Path.home() / "Documents" / "repos")
        )
    )

    # Repository configurations
    repos: Dict[str, RepoConfig] = field(default_factory=dict)

    # Service endpoints
    services: Dict[ServiceType, ServiceEndpoint] = field(default_factory=dict)

    # API Keys (from environment)
    api_keys: Dict[str, str] = field(default_factory=dict)

    # Night Shift specific
    nightshift_work_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NIGHTSHIFT_WORK_DIR", Path.home() / ".jarvis" / "nightshift")
        )
    )

    # Scout configuration
    scout_enabled: bool = field(
        default_factory=lambda: os.getenv("SCOUT_ENABLED", "true").lower() == "true"
    )
    scout_max_topics: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_MAX_TOPICS", "50"))
    )
    scout_concurrency: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_SCOUT_CONCURRENCY", "5"))
    )
    scout_synthesis_model: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SCOUT_MODEL", "gemini-1.5-flash")
    )

    # Training configuration
    base_model: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_BASE_MODEL", "meta-llama/Llama-3.2-3B"
        )
    )
    lora_rank: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_RANK", "64"))
    )

    # Runtime state
    _initialized: bool = False
    _callbacks: List[Callable[[str, Any], None]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize configuration after creation."""
        if not self._initialized:
            self._discover_repos()
            self._setup_services()
            self._load_api_keys()
            self._initialized = True

    def _discover_repos(self) -> None:
        """Auto-discover JARVIS repositories."""
        repo_mappings = {
            "jarvis_agent": ["JARVIS-AI-Agent", "jarvis-ai-agent"],
            "jarvis_prime": ["jarvis-prime", "JARVIS-Prime"],
            "reactor_core": ["reactor-core", "REACTOR-CORE"],
        }

        for repo_name, possible_names in repo_mappings.items():
            for name in possible_names:
                path = self.repos_base / name
                if path.exists():
                    config = RepoConfig(name=repo_name, path=path)
                    config.validate()
                    self.repos[repo_name] = config
                    logger.debug(f"Discovered repo: {repo_name} at {path}")
                    break

    def _setup_services(self) -> None:
        """Setup service endpoints with defaults."""
        # JARVIS-AI-Agent
        self.services[ServiceType.JARVIS_AGENT] = ServiceEndpoint(
            service_type=ServiceType.JARVIS_AGENT,
            host=os.getenv("JARVIS_HOST", "localhost"),
            port=int(os.getenv("JARVIS_PORT", "8000")),
            health_path="/health/ping",
        )

        # JARVIS Prime
        self.services[ServiceType.JARVIS_PRIME] = ServiceEndpoint(
            service_type=ServiceType.JARVIS_PRIME,
            host=os.getenv("JARVIS_PRIME_HOST", "localhost"),
            port=int(os.getenv("JARVIS_PRIME_PORT", "8002")),
        )

        # Reactor Core / Night Shift
        self.services[ServiceType.REACTOR_CORE] = ServiceEndpoint(
            service_type=ServiceType.REACTOR_CORE,
            host=os.getenv("REACTOR_HOST", "localhost"),
            port=int(os.getenv("REACTOR_PORT", "8080")),
        )

        # Redis
        self.services[ServiceType.REDIS] = ServiceEndpoint(
            service_type=ServiceType.REDIS,
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            protocol="redis",
        )

    def _load_api_keys(self) -> None:
        """Load API keys from environment."""
        key_names = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "HUGGINGFACE_TOKEN",
            "JARVIS_PRIME_API_KEY",
        ]

        for key_name in key_names:
            value = os.getenv(key_name)
            if value:
                self.api_keys[key_name] = value

    def get_repo(self, name: str) -> Optional[RepoConfig]:
        """Get repository configuration by name."""
        return self.repos.get(name)

    def get_service(self, service_type: ServiceType) -> Optional[ServiceEndpoint]:
        """Get service endpoint configuration."""
        return self.services.get(service_type)

    def get_api_key(self, name: str) -> Optional[str]:
        """Get API key by name."""
        return self.api_keys.get(name)

    async def check_service_health(
        self,
        service_type: ServiceType,
    ) -> bool:
        """Check if a service is available using connection pool."""
        endpoint = self.services.get(service_type)
        if not endpoint:
            return False

        try:
            # Use connection pool for efficient connection reuse
            pool = await HTTPConnectionPool.get_instance("unified_config")

            async with pool.request("GET", endpoint.health_url) as response:
                endpoint.is_available = response.status == 200
                endpoint.last_check = datetime.now()
                return endpoint.is_available

        except Exception as e:
            logger.debug(f"Health check failed for {service_type.value}: {e}")
            endpoint.is_available = False
            endpoint.last_check = datetime.now()
            return False

    async def check_all_services(self) -> Dict[ServiceType, bool]:
        """Check health of all services concurrently."""
        # Use asyncio.gather for concurrent health checks
        service_types = list(self.services.keys())

        async def check_single(stype: ServiceType) -> tuple:
            result = await self.check_service_health(stype)
            return (stype, result)

        tasks = [check_single(stype) for stype in service_types]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in results_list:
            if isinstance(item, tuple):
                stype, result = item
                results[stype] = result
            else:
                # Exception occurred
                logger.warning(f"Health check exception: {item}")

        return results

    async def get_redis_client(self) -> Any:
        """Get a Redis client from the connection pool."""
        redis_endpoint = self.services.get(ServiceType.REDIS)
        if not redis_endpoint:
            raise RuntimeError("Redis service not configured")

        pool = await RedisConnectionPool.get_instance()
        return await pool.get_client(
            host=redis_endpoint.host,
            port=redis_endpoint.port,
        )

    async def cleanup_pools(self) -> None:
        """Cleanup all connection pools (call on shutdown)."""
        await HTTPConnectionPool.close_all()
        await RedisConnectionPool.close_instance()
        logger.info("Connection pools cleaned up")

    def add_config_callback(
        self,
        callback: Callable[[str, Any], None],
    ) -> None:
        """Add callback for configuration changes."""
        self._callbacks.append(callback)

    def update_config(self, key: str, value: Any) -> None:
        """Update a configuration value and notify callbacks."""
        if hasattr(self, key):
            setattr(self, key, value)
            for callback in self._callbacks:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.warning(f"Config callback error: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "repos_base": str(self.repos_base),
            "repos": {
                name: {
                    "path": str(repo.path),
                    "exists": repo.exists,
                    "log_dir": str(repo.log_dir) if repo.log_dir else None,
                    "data_dir": str(repo.data_dir) if repo.data_dir else None,
                }
                for name, repo in self.repos.items()
            },
            "services": {
                stype.value: {
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "base_url": endpoint.base_url,
                    "is_available": endpoint.is_available,
                }
                for stype, endpoint in self.services.items()
            },
            "nightshift": {
                "work_dir": str(self.nightshift_work_dir),
                "scout_enabled": self.scout_enabled,
                "scout_max_topics": self.scout_max_topics,
                "base_model": self.base_model,
            },
            "api_keys_available": list(self.api_keys.keys()),
        }

    def save_to_file(self, path: Optional[Path] = None) -> Path:
        """Save configuration to file."""
        if path is None:
            path = self.nightshift_work_dir / "config.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return path

    @classmethod
    def load_from_file(cls, path: Path) -> "UnifiedConfig":
        """Load configuration from file (uses as overrides)."""
        config = cls()

        if path.exists():
            with open(path) as f:
                data = json.load(f)
                # Apply overrides
                if "environment" in data:
                    config.environment = Environment(data["environment"])
                # Add more overrides as needed

        return config


# Global singleton instance
_config_instance: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = UnifiedConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None


# Convenience exports
__all__ = [
    "UnifiedConfig",
    "ServiceType",
    "ServiceEndpoint",
    "RepoConfig",
    "Environment",
    "ConnectionPoolConfig",
    "HTTPConnectionPool",
    "RedisConnectionPool",
    "get_config",
    "reset_config",
]
