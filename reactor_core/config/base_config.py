"""
Base configuration system for Night Shift Training Engine.

Features:
- Dataclass-based configuration with type hints
- YAML file loading with environment variable interpolation
- Dynamic config reloading with hot-reload support
- Validation and defaults
- XDG Base Directory Specification compliance
- Service discovery with health-aware endpoints
- Thread-safe singleton with proper async locking

v2.0 - Advanced Dynamic Configuration
"""

from __future__ import annotations

import os
import re
import asyncio
import signal
import threading
import weakref
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseConfig")

# Singleton config instance with proper locking
_config_instance: Optional["NightShiftConfig"] = None
_config_lock = asyncio.Lock()
_config_thread_lock = threading.Lock()

# Configuration change callbacks
_config_callbacks: List[Callable[[str, Any, Any], None]] = []


# ============================================================================
# DYNAMIC PATH RESOLUTION (XDG Compliant)
# ============================================================================

class PathResolver:
    """
    Intelligent path resolution with XDG compliance and fallbacks.

    Resolution order:
    1. Explicit environment variable
    2. XDG Base Directory Specification
    3. Platform-specific defaults
    4. Fallback to ~/.jarvis (backwards compatibility)
    """

    _cache: ClassVar[Dict[str, Path]] = {}

    @classmethod
    def resolve(
        cls,
        name: str,
        env_var: Optional[str] = None,
        xdg_type: Literal["data", "config", "cache", "state", "runtime"] = "data",
        subdir: str = "",
        create: bool = True,
    ) -> Path:
        """
        Resolve a path dynamically - NO HARDCODING!

        Args:
            name: Human-readable name for logging
            env_var: Environment variable to check first
            xdg_type: XDG base directory type
            subdir: Subdirectory under base path
            create: Create directory if it doesn't exist

        Returns:
            Resolved absolute path
        """
        cache_key = f"{name}:{env_var}:{xdg_type}:{subdir}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        path: Optional[Path] = None

        # 1. Check explicit environment variable
        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                path = Path(env_value).expanduser().resolve()
                logger.debug(f"[PathResolver] {name}: env {env_var} = {path}")

        # 2. Try XDG directories
        if path is None:
            xdg_base = cls._get_xdg_base(xdg_type)
            if xdg_base:
                path = xdg_base / "reactor-core"
                if subdir:
                    path = path / subdir
                logger.debug(f"[PathResolver] {name}: XDG {xdg_type} = {path}")

        # 3. Fallback to ~/.jarvis (backwards compatibility)
        if path is None:
            path = Path.home() / ".jarvis"
            if subdir:
                path = path / subdir
            logger.debug(f"[PathResolver] {name}: fallback = {path}")

        # Ensure directory exists
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"[PathResolver] Could not create {path}: {e}")

        cls._cache[cache_key] = path
        return path

    @classmethod
    def _get_xdg_base(cls, xdg_type: str) -> Optional[Path]:
        """Get XDG base directory."""
        home = Path.home()

        xdg_vars = {
            "data": ("XDG_DATA_HOME", home / ".local" / "share"),
            "config": ("XDG_CONFIG_HOME", home / ".config"),
            "cache": ("XDG_CACHE_HOME", home / ".cache"),
            "state": ("XDG_STATE_HOME", home / ".local" / "state"),
            "runtime": ("XDG_RUNTIME_DIR", Path(f"/run/user/{os.getuid()}") if hasattr(os, 'getuid') else None),
        }

        if xdg_type in xdg_vars:
            env_var, default = xdg_vars[xdg_type]
            env_value = os.environ.get(env_var)
            if env_value:
                return Path(env_value)
            return default
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear path resolution cache."""
        cls._cache.clear()


def resolve_path(
    name: str,
    env_var: Optional[str] = None,
    xdg_type: Literal["data", "config", "cache", "state", "runtime"] = "data",
    subdir: str = "",
) -> Path:
    """
    Convenience function for dynamic path resolution.

    USE THIS INSTEAD OF HARDCODED PATHS!

    Example:
        # Instead of: Path.home() / ".jarvis" / "logs"
        # Use: resolve_path("logs", env_var="REACTOR_LOG_DIR", xdg_type="state", subdir="logs")
    """
    return PathResolver.resolve(name, env_var, xdg_type, subdir)


# ============================================================================
# SERVICE ENDPOINT DISCOVERY
# ============================================================================

@dataclass
class ServiceEndpoint:
    """Dynamic service endpoint with health awareness."""
    name: str
    host: str
    port: int
    protocol: str = "http"
    health_path: str = "/health"
    timeout_seconds: float = 5.0

    # Health state (updated by health checks)
    is_healthy: bool = False
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    latency_ms: Optional[float] = None

    @property
    def base_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"

    @property
    def ws_url(self) -> str:
        ws_proto = "wss" if self.protocol == "https" else "ws"
        return f"{ws_proto}://{self.host}:{self.port}/ws"

    @classmethod
    def from_env(
        cls,
        name: str,
        host_var: str,
        port_var: str,
        default_host: str = "localhost",
        default_port: int = 8080,
        **kwargs,
    ) -> "ServiceEndpoint":
        """Create endpoint from environment variables."""
        host = os.environ.get(host_var, default_host)
        port_str = os.environ.get(port_var, str(default_port))
        try:
            port = int(port_str)
        except ValueError:
            port = default_port

        return cls(name=name, host=host, port=port, **kwargs)

    async def check_health(self) -> bool:
        """Check endpoint health with latency measurement."""
        import time
        try:
            import aiohttp
            start = time.monotonic()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.health_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                ) as response:
                    self.latency_ms = (time.monotonic() - start) * 1000
                    self.is_healthy = response.status == 200
                    self.consecutive_failures = 0 if self.is_healthy else self.consecutive_failures + 1
                    self.last_check = datetime.now()
                    return self.is_healthy
        except Exception as e:
            self.is_healthy = False
            self.consecutive_failures += 1
            self.last_check = datetime.now()
            logger.debug(f"[ServiceEndpoint] Health check failed for {self.name}: {e}")
            return False


# ============================================================================
# CONFIGURATION CALLBACKS FOR HOT-RELOAD
# ============================================================================

def register_config_callback(callback: Callable[[str, Any, Any], None]) -> None:
    """Register a callback for configuration changes."""
    _config_callbacks.append(callback)


def unregister_config_callback(callback: Callable[[str, Any, Any], None]) -> None:
    """Unregister a configuration change callback."""
    if callback in _config_callbacks:
        _config_callbacks.remove(callback)


def _notify_config_change(key: str, old_value: Any, new_value: Any) -> None:
    """Notify all callbacks of a configuration change."""
    for callback in _config_callbacks:
        try:
            callback(key, old_value, new_value)
        except Exception as e:
            logger.error(f"[Config] Callback error: {e}")


def _interpolate_env_vars(value: Any) -> Any:
    """
    Recursively interpolate environment variables in config values.

    Supports formats:
    - ${VAR_NAME} - Required, raises if not set
    - ${VAR_NAME:-default} - Optional with default
    - ${VAR_NAME:?error message} - Required with custom error
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME}, ${VAR_NAME:-default}, ${VAR_NAME:?error}
        pattern = r"\$\{([A-Z_][A-Z0-9_]*)(?:(:-)([^}]*))?(?:(:\?)([^}]*))?\}"

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) or ""
            has_error = match.group(4) is not None
            error_msg = match.group(5) or f"Required environment variable {var_name} is not set"

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif has_default:
                return default_value
            elif has_error:
                raise ValueError(error_msg)
            else:
                # Check if the entire string is just the variable
                if match.group(0) == value:
                    raise ValueError(f"Environment variable {var_name} is not set")
                return match.group(0)  # Keep original if part of larger string

        result = re.sub(pattern, replace_var, value)

        # Handle ~ for home directory
        if result.startswith("~"):
            result = str(Path(result).expanduser())

        return result

    elif isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]

    return value


def _coerce_type(value: Any, target_type: type) -> Any:
    """Coerce a value to the target type."""
    if value is None:
        return None

    origin = getattr(target_type, "__origin__", None)

    # Handle Optional types
    if origin is Union:
        args = target_type.__args__
        if type(None) in args:
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return _coerce_type(value, non_none_types[0])

    # Handle Path
    if target_type is Path or (hasattr(target_type, "__origin__") and target_type.__origin__ is Path):
        return Path(value).expanduser() if value else None

    # Handle List
    if origin is list:
        item_type = target_type.__args__[0] if target_type.__args__ else str
        if isinstance(value, list):
            return [_coerce_type(item, item_type) for item in value]
        return [_coerce_type(value, item_type)]

    # Handle bool (special case because bool("false") is True)
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    # Handle numeric types
    if target_type is int:
        return int(float(value)) if value else 0
    if target_type is float:
        return float(value) if value else 0.0

    # Default: try direct conversion
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return value


@dataclass
class BaseConfig:
    """
    Base configuration class with YAML loading and env var interpolation.
    """

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create config from dictionary with env var interpolation."""
        interpolated = _interpolate_env_vars(data)

        # Get field types for coercion
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

        # Only include fields that exist in the dataclass
        filtered = {}
        for key, value in interpolated.items():
            if key in field_types:
                filtered[key] = _coerce_type(value, field_types[key])

        return cls(**filtered)

    @classmethod
    def from_yaml(cls: Type[T], path: Union[str, Path]) -> T:
        """Load config from YAML file with env var interpolation."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_env(cls: Type[T], prefix: str = "") -> T:
        """Create config entirely from environment variables."""
        data = {}

        for field_info in cls.__dataclass_fields__.values():
            env_key = f"{prefix}{field_info.name}".upper()
            env_value = os.environ.get(env_key)

            if env_value is not None:
                data[field_info.name] = env_value

        return cls.from_dict(data) if data else cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def merge(self: T, other: Dict[str, Any]) -> T:
        """Create new config with overrides merged in."""
        current = self.to_dict()
        interpolated = _interpolate_env_vars(other)
        current.update(interpolated)
        return self.__class__.from_dict(current)


@dataclass
class IngestionConfig(BaseConfig):
    """Configuration for data ingestion from JARVIS logs."""

    # Source paths - DYNAMIC RESOLUTION
    jarvis_logs_dir: Path = field(
        default_factory=lambda: resolve_path(
            "jarvis_logs", env_var="NIGHTSHIFT_JARVIS_LOGS",
            xdg_type="state", subdir="logs"
        )
    )
    jarvis_data_dir: Path = field(
        default_factory=lambda: resolve_path(
            "jarvis_data", env_var="NIGHTSHIFT_JARVIS_DATA",
            xdg_type="data", subdir="data"
        )
    )

    # Processing settings
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_BATCH_SIZE", "1000"))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_MAX_WORKERS", "4"))
    )
    streaming_enabled: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_STREAMING", "false").lower() == "true"
    )

    # Quality thresholds
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MIN_CONFIDENCE", "0.7"))
    )
    deduplicate: bool = True
    dedup_similarity_threshold: float = 0.95

    # Time range
    lookback_days: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LOOKBACK_DAYS", "7"))
    )

    # Source types to ingest
    ingest_telemetry: bool = True
    ingest_feedback: bool = True
    ingest_auth_records: bool = True
    ingest_raw_logs: bool = True


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training."""

    # Model settings
    base_model: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_BASE_MODEL", "meta-llama/Llama-3.2-3B")
    )
    model_revision: Optional[str] = None
    trust_remote_code: bool = False

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_RANK", "64"))
    )
    lora_alpha: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_LORA_ALPHA", "128"))
    )
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # QLoRA settings
    use_qlora: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_USE_QLORA", "true").lower() == "true"
    )
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training hyperparameters
    learning_rate: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_LR", "2e-5"))
    )
    num_epochs: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_EPOCHS", "3"))
    )
    per_device_batch_size: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_TRAIN_BATCH_SIZE", "4"))
    )
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence settings
    max_seq_length: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_MAX_SEQ_LEN", "2048"))
    )

    # Checkpointing - DYNAMIC RESOLUTION
    checkpoint_dir: Path = field(
        default_factory=lambda: resolve_path(
            "checkpoints", env_var="NIGHTSHIFT_CHECKPOINT_DIR",
            xdg_type="data", subdir="training/checkpoints"
        )
    )
    save_steps: int = 500
    eval_steps: int = 500
    max_checkpoints: int = 3
    resume_from_checkpoint: bool = True

    # Output - DYNAMIC RESOLUTION
    output_dir: Path = field(
        default_factory=lambda: resolve_path(
            "training_output", env_var="NIGHTSHIFT_OUTPUT_DIR",
            xdg_type="data", subdir="training/output"
        )
    )

    # Distributed training
    use_fsdp: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_USE_FSDP", "false").lower() == "true"
    )

    # Device
    device: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_DEVICE", "auto")
    )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for knowledge distillation via teacher model."""

    # Teacher model settings
    teacher_provider: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TEACHER_PROVIDER", "openai")
    )
    teacher_model: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TEACHER_MODEL", "gpt-4o")
    )

    # API keys (from environment)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Rate limiting
    requests_per_minute: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_RPM", "60"))
    )
    max_concurrent_requests: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_CONCURRENT", "10"))
    )
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0

    # Cost controls
    max_daily_cost_usd: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MAX_DAILY_COST", "50.0"))
    )
    max_tokens_per_request: int = 4096

    # Quality thresholds
    min_quality_score: float = 0.6
    rewrite_threshold: float = 0.4

    # Distillation modes
    enable_scoring: bool = True
    enable_rewriting: bool = True
    enable_synthetic_generation: bool = False

    # Synthetic generation settings
    synthetic_examples_per_topic: int = 5
    synthetic_topics: List[str] = field(default_factory=list)


@dataclass
class QuantizationConfig(BaseConfig):
    """Configuration for model quantization."""

    # Output format
    output_format: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_QUANT_FORMAT", "gguf")
    )

    # GGUF settings
    gguf_quantization_type: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_GGUF_QUANT", "Q4_K_M")
    )

    # Output paths - DYNAMIC RESOLUTION
    output_dir: Path = field(
        default_factory=lambda: resolve_path(
            "quantized_models", env_var="NIGHTSHIFT_QUANT_OUTPUT",
            xdg_type="data", subdir="models/quantized"
        )
    )

    # llama.cpp settings
    llama_cpp_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("LLAMA_CPP_PATH", "")).expanduser()
        if os.getenv("LLAMA_CPP_PATH") else None
    )

    # Performance settings
    num_threads: int = field(
        default_factory=lambda: int(os.getenv("NIGHTSHIFT_QUANT_THREADS", "4"))
    )


@dataclass
class EvalConfig(BaseConfig):
    """Configuration for model evaluation and gatekeeper."""

    # Benchmarks to run
    run_humaneval: bool = True
    run_jarvis_eval: bool = True
    run_regression: bool = True
    run_perplexity: bool = True
    run_latency: bool = True

    # Thresholds for gatekeeper
    max_perplexity: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MAX_PERPLEXITY", "5.0"))
    )
    min_humaneval_pass_rate: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MIN_HUMANEVAL", "0.30"))
    )
    max_latency_ms: float = field(
        default_factory=lambda: float(os.getenv("NIGHTSHIFT_MAX_LATENCY", "100.0"))
    )
    max_regression_delta: float = 0.05  # 5% regression allowed

    # Test set paths
    jarvis_test_set_path: Optional[Path] = field(
        default_factory=lambda: Path(os.getenv("NIGHTSHIFT_TEST_SET", "")).expanduser()
        if os.getenv("NIGHTSHIFT_TEST_SET") else None
    )

    # Previous model for comparison
    previous_model_path: Optional[Path] = None

    # Output - DYNAMIC RESOLUTION
    eval_output_dir: Path = field(
        default_factory=lambda: resolve_path(
            "eval_output", env_var="NIGHTSHIFT_EVAL_OUTPUT",
            xdg_type="data", subdir="training/eval"
        )
    )


@dataclass
class OrchestrationConfig(BaseConfig):
    """Configuration for pipeline orchestration."""

    # Scheduling
    schedule_cron: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_CRON", "0 2 * * 0")
    )
    timezone: str = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_TIMEZONE", "America/Los_Angeles")
    )

    # State management - DYNAMIC RESOLUTION
    state_file: Path = field(
        default_factory=lambda: resolve_path(
            "state", env_var="NIGHTSHIFT_STATE_FILE",
            xdg_type="state", subdir="training"
        ) / "pipeline_state.json"
    )

    # Recovery settings
    max_retries: int = 3
    retry_delay_seconds: int = 300
    resume_on_failure: bool = True

    # Notifications
    slack_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SLACK_WEBHOOK")
    )
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_host: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_SMTP_HOST")
    )

    # Alerts
    alert_on_success: bool = True
    alert_on_failure: bool = True
    alert_on_gatekeeper_fail: bool = True

    # GCS artifact storage
    gcs_bucket: Optional[str] = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_GCS_BUCKET")
    )
    gcs_prefix: str = "nightshift/models"

    # Model registry - DYNAMIC RESOLUTION
    model_registry_path: Path = field(
        default_factory=lambda: resolve_path(
            "registry", env_var="NIGHTSHIFT_REGISTRY",
            xdg_type="data", subdir="models"
        ) / "registry.json"
    )
    max_model_versions: int = 5


# ============================================================================
# SERVICE ENDPOINTS CONFIGURATION
# ============================================================================

@dataclass
class ServicesConfig(BaseConfig):
    """Configuration for all service endpoints - NO HARDCODED PORTS!"""

    # JARVIS Body
    jarvis_host: str = field(
        default_factory=lambda: os.getenv("JARVIS_HOST", "localhost")
    )
    jarvis_port: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_PORT", "8000"))
    )

    # JARVIS Prime
    jprime_host: str = field(
        default_factory=lambda: os.getenv("JARVIS_PRIME_HOST", "localhost")
    )
    jprime_port: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_PRIME_PORT", "8002"))
    )

    # Reactor Core
    reactor_host: str = field(
        default_factory=lambda: os.getenv("REACTOR_HOST", "0.0.0.0")
    )
    reactor_port: int = field(
        default_factory=lambda: int(os.getenv("REACTOR_PORT", "8080"))
    )

    # Redis
    redis_host: str = field(
        default_factory=lambda: os.getenv("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: int(os.getenv("REDIS_PORT", "6379"))
    )

    def get_jarvis_endpoint(self) -> ServiceEndpoint:
        """Get JARVIS Body service endpoint."""
        return ServiceEndpoint(
            name="jarvis",
            host=self.jarvis_host,
            port=self.jarvis_port,
            health_path="/health/ping",
        )

    def get_jprime_endpoint(self) -> ServiceEndpoint:
        """Get JARVIS Prime service endpoint."""
        return ServiceEndpoint(
            name="jprime",
            host=self.jprime_host,
            port=self.jprime_port,
            health_path="/health",
        )

    def get_reactor_endpoint(self) -> ServiceEndpoint:
        """Get Reactor Core service endpoint."""
        return ServiceEndpoint(
            name="reactor",
            host=self.reactor_host,
            port=self.reactor_port,
            health_path="/health",
        )


@dataclass
class NightShiftConfig(BaseConfig):
    """
    Master configuration combining all Night Shift components.

    ALL VALUES ARE DYNAMICALLY CONFIGURABLE via environment variables.
    NO HARDCODING - use resolve_path() for paths and ServicesConfig for ports.
    """

    # Component configs
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)

    # Global settings
    run_id: Optional[str] = None
    dry_run: bool = False
    verbose: bool = field(
        default_factory=lambda: os.getenv("NIGHTSHIFT_VERBOSE", "false").lower() == "true"
    )

    # Feature flags (all configurable via env)
    enable_metrics: bool = field(
        default_factory=lambda: os.getenv("REACTOR_ENABLE_METRICS", "true").lower() == "true"
    )
    enable_tracing: bool = field(
        default_factory=lambda: os.getenv("REACTOR_ENABLE_TRACING", "true").lower() == "true"
    )
    enable_hot_reload: bool = field(
        default_factory=lambda: os.getenv("REACTOR_ENABLE_HOT_RELOAD", "true").lower() == "true"
    )

    # Debug
    debug: bool = field(
        default_factory=lambda: os.getenv("REACTOR_DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("REACTOR_LOG_LEVEL", "INFO")
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "NightShiftConfig":
        """Load full config from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Parse nested configs
        ingestion = IngestionConfig.from_dict(data.get("ingestion", {}))
        training = TrainingConfig.from_dict(data.get("training", {}))
        distillation = DistillationConfig.from_dict(data.get("distillation", {}))
        quantization = QuantizationConfig.from_dict(data.get("quantization", {}))
        eval_config = EvalConfig.from_dict(data.get("eval", {}))
        orchestration = OrchestrationConfig.from_dict(data.get("orchestration", {}))

        # Global settings
        global_settings = {
            k: v for k, v in data.items()
            if k not in ("ingestion", "training", "distillation", "quantization", "eval", "orchestration")
        }
        global_settings = _interpolate_env_vars(global_settings)

        return cls(
            ingestion=ingestion,
            training=training,
            distillation=distillation,
            quantization=quantization,
            eval=eval_config,
            orchestration=orchestration,
            **global_settings,
        )

    @classmethod
    def from_yaml_dir(cls, config_dir: Union[str, Path]) -> "NightShiftConfig":
        """Load config from directory of YAML files."""
        config_dir = Path(config_dir)

        ingestion = IngestionConfig()
        training = TrainingConfig()
        distillation = DistillationConfig()
        quantization = QuantizationConfig()
        eval_config = EvalConfig()
        orchestration = OrchestrationConfig()

        if (config_dir / "ingestion.yaml").exists():
            ingestion = IngestionConfig.from_yaml(config_dir / "ingestion.yaml")
        if (config_dir / "training.yaml").exists():
            training = TrainingConfig.from_yaml(config_dir / "training.yaml")
        if (config_dir / "distillation.yaml").exists():
            distillation = DistillationConfig.from_yaml(config_dir / "distillation.yaml")
        if (config_dir / "quantization.yaml").exists():
            quantization = QuantizationConfig.from_yaml(config_dir / "quantization.yaml")
        if (config_dir / "eval.yaml").exists():
            eval_config = EvalConfig.from_yaml(config_dir / "eval.yaml")
        if (config_dir / "orchestration.yaml").exists():
            orchestration = OrchestrationConfig.from_yaml(config_dir / "orchestration.yaml")

        return cls(
            ingestion=ingestion,
            training=training,
            distillation=distillation,
            quantization=quantization,
            eval=eval_config,
            orchestration=orchestration,
        )


async def load_config(
    path: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> NightShiftConfig:
    """
    Load or get cached configuration.

    Args:
        path: Path to config file or directory. If None, uses defaults.
        reload: Force reload even if cached.

    Returns:
        NightShiftConfig instance.
    """
    global _config_instance

    if _config_instance is not None and not reload:
        return _config_instance

    async with _config_lock:
        if _config_instance is not None and not reload:
            return _config_instance

        if path is None:
            # Use defaults with env var overrides
            _config_instance = NightShiftConfig()
        elif Path(path).is_dir():
            _config_instance = NightShiftConfig.from_yaml_dir(path)
        else:
            _config_instance = NightShiftConfig.from_yaml(path)

        logger.info(f"Configuration loaded: {_config_instance.training.base_model}")
        return _config_instance


def get_config() -> NightShiftConfig:
    """
    Get cached config synchronously (thread-safe).

    Returns default config if not yet loaded.
    """
    global _config_instance

    if _config_instance is not None:
        return _config_instance

    # Thread-safe lazy initialization
    with _config_thread_lock:
        if _config_instance is None:
            _config_instance = NightShiftConfig()
            logger.info("[Config] Default configuration loaded")

    return _config_instance


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific config value by dot-notation key."""
    config = get_config()
    parts = key.split(".")

    value = config
    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default

    return value


async def reload_config() -> NightShiftConfig:
    """
    Reload configuration from sources (hot-reload).

    Notifies all registered callbacks of changes.
    """
    global _config_instance

    async with _config_lock:
        old_config = _config_instance
        _config_instance = NightShiftConfig()

        if old_config:
            _notify_config_change("config", old_config, _config_instance)

        logger.info("[Config] Configuration reloaded")
        return _config_instance


def setup_signal_handlers() -> None:
    """
    Setup signal handlers for hot-reload.

    SIGHUP triggers configuration reload.
    """
    if hasattr(signal, 'SIGHUP'):
        def handle_sighup(signum, frame):
            logger.info("[Config] SIGHUP received, scheduling config reload")
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(reload_config())
            except RuntimeError:
                # No running loop, can't reload asynchronously
                pass

        signal.signal(signal.SIGHUP, handle_sighup)
        logger.debug("[Config] Registered SIGHUP handler for hot-reload")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Path resolution
    "PathResolver",
    "resolve_path",
    # Service endpoints
    "ServiceEndpoint",
    # Config classes
    "BaseConfig",
    "IngestionConfig",
    "TrainingConfig",
    "DistillationConfig",
    "QuantizationConfig",
    "EvalConfig",
    "OrchestrationConfig",
    "ServicesConfig",
    "NightShiftConfig",
    # Config functions
    "load_config",
    "get_config",
    "get_config_value",
    "reload_config",
    "setup_signal_handlers",
    # Callbacks
    "register_config_callback",
    "unregister_config_callback",
]
