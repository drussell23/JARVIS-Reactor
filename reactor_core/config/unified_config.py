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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


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
        """Check if a service is available."""
        import aiohttp

        endpoint = self.services.get(service_type)
        if not endpoint:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint.health_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    endpoint.is_available = response.status == 200
                    endpoint.last_check = datetime.now()
                    return endpoint.is_available
        except Exception:
            endpoint.is_available = False
            endpoint.last_check = datetime.now()
            return False

    async def check_all_services(self) -> Dict[ServiceType, bool]:
        """Check health of all services."""
        results = {}
        for service_type in self.services.keys():
            results[service_type] = await self.check_service_health(service_type)
        return results

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
    "get_config",
    "reset_config",
]
