"""
Dependency Management for AGI OS - Reactor Core (Nervous System)
================================================================

Provides robust dependency management with:
- Runtime dependency checking with version validation
- Optional dependencies with graceful degradation
- Lazy importing for faster startup
- Platform-specific handling (macOS/Linux, CUDA/MPS/CPU)
- Automatic fallback chains
- Import optimization and caching
- Dependency health monitoring

This module ensures the system can operate even when optional
dependencies are unavailable, with clear error messages and
automatic fallbacks where possible.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import subprocess
import sys
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DependencyStatus(Enum):
    """Status of a dependency."""
    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_ERROR = "import_error"
    PLATFORM_UNSUPPORTED = "platform_unsupported"
    OPTIONAL_MISSING = "optional_missing"


class Platform(Enum):
    """Supported platforms."""
    MACOS_ARM = "macos_arm"
    MACOS_INTEL = "macos_intel"
    LINUX = "linux"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class AcceleratorType(Enum):
    """Hardware accelerator types."""
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    ROCm = "rocm"  # AMD
    CPU = "cpu"


# Core dependencies required for basic functionality
CORE_DEPENDENCIES = [
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
]

# Optional dependencies with fallback behavior
OPTIONAL_DEPENDENCIES = {
    "vllm": {"platforms": [Platform.LINUX], "fallback": "transformers"},
    "flash_attn": {"platforms": [Platform.LINUX], "fallback": None},
    "bitsandbytes": {"platforms": [Platform.LINUX], "fallback": None},
    "triton": {"platforms": [Platform.LINUX], "fallback": None},
    "deepspeed": {"platforms": [Platform.LINUX], "fallback": None},
    "apex": {"platforms": [Platform.LINUX], "fallback": None},
    "llama_cpp": {"platforms": [Platform.LINUX, Platform.MACOS_ARM], "fallback": None},
    "mlx": {"platforms": [Platform.MACOS_ARM], "fallback": None},
}

# Version requirements
VERSION_REQUIREMENTS = {
    "torch": ">=2.0.0",
    "transformers": ">=4.30.0",
    "peft": ">=0.4.0",
    "trl": ">=0.7.0",
    "accelerate": ">=0.20.0",
}


# =============================================================================
# PLATFORM DETECTION
# =============================================================================

def detect_platform() -> Platform:
    """Detect current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if "arm" in machine or machine == "arm64":
            return Platform.MACOS_ARM
        return Platform.MACOS_INTEL
    elif system == "linux":
        return Platform.LINUX
    elif system == "windows":
        return Platform.WINDOWS

    return Platform.UNKNOWN


def detect_accelerator() -> AcceleratorType:
    """Detect available hardware accelerator."""
    try:
        import torch
        if torch.cuda.is_available():
            return AcceleratorType.CUDA
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return AcceleratorType.MPS
    except ImportError:
        pass

    # Check for ROCm
    if os.path.exists("/opt/rocm"):
        return AcceleratorType.ROCm

    return AcceleratorType.CPU


# =============================================================================
# DEPENDENCY INFO
# =============================================================================

@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    status: DependencyStatus
    installed_version: Optional[str] = None
    required_version: Optional[str] = None
    import_error: Optional[str] = None
    fallback: Optional[str] = None
    platforms: List[Platform] = field(default_factory=list)
    is_optional: bool = False
    module: Optional[Any] = None

    @property
    def is_available(self) -> bool:
        return self.status == DependencyStatus.AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "installed_version": self.installed_version,
            "required_version": self.required_version,
            "import_error": self.import_error,
            "fallback": self.fallback,
            "is_optional": self.is_optional,
        }


# =============================================================================
# VERSION CHECKING
# =============================================================================

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple for comparison."""
    # Remove common prefixes
    version_str = version_str.lstrip("v").split("+")[0].split("-")[0]

    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            # Handle versions like "2.0.0a1"
            numeric_part = ""
            for char in part:
                if char.isdigit():
                    numeric_part += char
                else:
                    break
            if numeric_part:
                parts.append(int(numeric_part))
            break

    return tuple(parts)


def check_version_requirement(
    installed: str,
    requirement: str,
) -> bool:
    """Check if installed version satisfies requirement."""
    if not requirement:
        return True

    installed_parts = parse_version(installed)

    # Parse requirement (e.g., ">=2.0.0", "==1.0.0", "~=1.0")
    if requirement.startswith(">="):
        required_parts = parse_version(requirement[2:])
        return installed_parts >= required_parts
    elif requirement.startswith("<="):
        required_parts = parse_version(requirement[2:])
        return installed_parts <= required_parts
    elif requirement.startswith("=="):
        required_parts = parse_version(requirement[2:])
        return installed_parts == required_parts
    elif requirement.startswith("~="):
        # Compatible release
        required_parts = parse_version(requirement[2:])
        return (
            installed_parts[:len(required_parts)-1] == required_parts[:-1]
            and installed_parts >= required_parts
        )
    elif requirement.startswith(">"):
        required_parts = parse_version(requirement[1:])
        return installed_parts > required_parts
    elif requirement.startswith("<"):
        required_parts = parse_version(requirement[1:])
        return installed_parts < required_parts

    # Plain version, treat as ==
    required_parts = parse_version(requirement)
    return installed_parts == required_parts


# =============================================================================
# DEPENDENCY CHECKER
# =============================================================================

class DependencyChecker:
    """
    Comprehensive dependency checker with caching and health monitoring.

    Example:
        checker = DependencyChecker()

        # Check all dependencies
        status = checker.check_all()

        # Check specific dependency
        info = checker.check("torch")

        # Get import with fallback
        vllm = checker.get_module("vllm", fallback="transformers")
    """

    _instance: Optional["DependencyChecker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "DependencyChecker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._cache: Dict[str, DependencyInfo] = {}
        self._platform = detect_platform()
        self._accelerator = detect_accelerator()
        self._initialized = True

        logger.info(f"DependencyChecker initialized - Platform: {self._platform.value}, Accelerator: {self._accelerator.value}")

    @property
    def platform(self) -> Platform:
        return self._platform

    @property
    def accelerator(self) -> AcceleratorType:
        return self._accelerator

    def check(self, name: str, force_refresh: bool = False) -> DependencyInfo:
        """Check a single dependency."""
        if name in self._cache and not force_refresh:
            return self._cache[name]

        info = self._check_dependency(name)
        self._cache[name] = info
        return info

    def _check_dependency(self, name: str) -> DependencyInfo:
        """Internal dependency check."""
        is_optional = name in OPTIONAL_DEPENDENCIES
        optional_config = OPTIONAL_DEPENDENCIES.get(name, {})
        platforms = optional_config.get("platforms", [])
        fallback = optional_config.get("fallback")
        required_version = VERSION_REQUIREMENTS.get(name)

        # Check platform compatibility
        if platforms and self._platform not in platforms:
            return DependencyInfo(
                name=name,
                status=DependencyStatus.PLATFORM_UNSUPPORTED,
                fallback=fallback,
                platforms=platforms,
                is_optional=is_optional,
            )

        # Check if module can be imported
        try:
            spec = importlib.util.find_spec(name)
            if spec is None:
                return DependencyInfo(
                    name=name,
                    status=DependencyStatus.OPTIONAL_MISSING if is_optional else DependencyStatus.MISSING,
                    fallback=fallback,
                    is_optional=is_optional,
                )

            # Get version
            try:
                installed_version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                installed_version = None

            # Check version requirement
            if required_version and installed_version:
                if not check_version_requirement(installed_version, required_version):
                    return DependencyInfo(
                        name=name,
                        status=DependencyStatus.VERSION_MISMATCH,
                        installed_version=installed_version,
                        required_version=required_version,
                        fallback=fallback,
                        is_optional=is_optional,
                    )

            # Try actual import
            module = importlib.import_module(name)

            return DependencyInfo(
                name=name,
                status=DependencyStatus.AVAILABLE,
                installed_version=installed_version,
                required_version=required_version,
                module=module,
                is_optional=is_optional,
            )

        except ImportError as e:
            return DependencyInfo(
                name=name,
                status=DependencyStatus.IMPORT_ERROR,
                import_error=str(e),
                fallback=fallback,
                is_optional=is_optional,
            )

    def check_all(self) -> Dict[str, DependencyInfo]:
        """Check all known dependencies."""
        all_deps = set(CORE_DEPENDENCIES) | set(OPTIONAL_DEPENDENCIES.keys())

        results = {}
        for name in all_deps:
            results[name] = self.check(name)

        return results

    def check_core(self) -> Tuple[bool, List[DependencyInfo]]:
        """Check core dependencies, return (all_ok, missing_list)."""
        missing = []

        for name in CORE_DEPENDENCIES:
            info = self.check(name)
            if not info.is_available:
                missing.append(info)

        return len(missing) == 0, missing

    def get_module(
        self,
        name: str,
        fallback: Optional[str] = None,
        required: bool = False,
    ) -> Optional[Any]:
        """
        Get module with optional fallback.

        Args:
            name: Module name to import
            fallback: Fallback module if primary unavailable
            required: If True, raise ImportError if not available

        Returns:
            Imported module or None
        """
        info = self.check(name)

        if info.is_available:
            return info.module

        # Try fallback
        fallback = fallback or info.fallback
        if fallback:
            fallback_info = self.check(fallback)
            if fallback_info.is_available:
                logger.info(f"Using fallback '{fallback}' for '{name}'")
                return fallback_info.module

        if required:
            raise ImportError(
                f"Required dependency '{name}' not available. "
                f"Status: {info.status.value}, Error: {info.import_error}"
            )

        return None

    def is_available(self, name: str) -> bool:
        """Quick check if dependency is available."""
        return self.check(name).is_available

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        all_deps = self.check_all()

        available = [d for d in all_deps.values() if d.is_available]
        missing = [d for d in all_deps.values() if d.status == DependencyStatus.MISSING]
        optional_missing = [d for d in all_deps.values() if d.status == DependencyStatus.OPTIONAL_MISSING]
        platform_unsupported = [d for d in all_deps.values() if d.status == DependencyStatus.PLATFORM_UNSUPPORTED]
        import_errors = [d for d in all_deps.values() if d.status == DependencyStatus.IMPORT_ERROR]
        version_mismatches = [d for d in all_deps.values() if d.status == DependencyStatus.VERSION_MISMATCH]

        return {
            "platform": self._platform.value,
            "accelerator": self._accelerator.value,
            "summary": {
                "available": len(available),
                "missing": len(missing),
                "optional_missing": len(optional_missing),
                "platform_unsupported": len(platform_unsupported),
                "import_errors": len(import_errors),
                "version_mismatches": len(version_mismatches),
            },
            "available": [d.name for d in available],
            "issues": {
                "missing": [d.to_dict() for d in missing],
                "optional_missing": [d.to_dict() for d in optional_missing],
                "platform_unsupported": [d.to_dict() for d in platform_unsupported],
                "import_errors": [d.to_dict() for d in import_errors],
                "version_mismatches": [d.to_dict() for d in version_mismatches],
            },
        }


# =============================================================================
# LAZY IMPORTER
# =============================================================================

class LazyModule:
    """
    Lazy module loader that defers import until first access.

    Example:
        torch = LazyModule("torch")

        # Module not imported yet

        print(torch.cuda.is_available())  # Import happens here
    """

    def __init__(
        self,
        name: str,
        fallback: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        self._name = name
        self._fallback = fallback
        self._error_message = error_message
        self._module: Optional[Any] = None
        self._loaded = False
        self._lock = threading.Lock()

    def _load(self) -> Any:
        """Load the module."""
        if self._loaded:
            return self._module

        with self._lock:
            if self._loaded:
                return self._module

            checker = DependencyChecker()
            self._module = checker.get_module(
                self._name,
                fallback=self._fallback,
                required=False,
            )

            if self._module is None:
                error_msg = self._error_message or f"Module '{self._name}' not available"
                if self._fallback:
                    error_msg += f" (fallback '{self._fallback}' also unavailable)"
                raise ImportError(error_msg)

            self._loaded = True
            return self._module

    def __getattr__(self, name: str) -> Any:
        module = self._load()
        return getattr(module, name)

    def __dir__(self) -> List[str]:
        module = self._load()
        return dir(module)


def lazy_import(
    name: str,
    fallback: Optional[str] = None,
    error_message: Optional[str] = None,
) -> LazyModule:
    """
    Create a lazy-loaded module.

    Example:
        torch = lazy_import("torch")
        vllm = lazy_import("vllm", fallback="transformers")
    """
    return LazyModule(name, fallback, error_message)


# =============================================================================
# REQUIRE DECORATOR
# =============================================================================

def requires(
    *dependencies: str,
    fallback: Optional[Callable[..., T]] = None,
    error_message: Optional[str] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to require dependencies for a function.

    Example:
        @requires("vllm", "flash_attn")
        def fast_inference():
            ...

        @requires("vllm", fallback=slow_inference)
        def inference():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            checker = DependencyChecker()

            missing = []
            for dep in dependencies:
                if not checker.is_available(dep):
                    missing.append(dep)

            if missing:
                if fallback:
                    logger.info(f"Using fallback for {func.__name__} (missing: {missing})")
                    return fallback(*args, **kwargs)

                msg = error_message or f"Missing dependencies: {missing}"
                raise ImportError(msg)

            return func(*args, **kwargs)

        return wrapper
    return decorator


def optional_import(
    name: str,
    package: Optional[str] = None,
    fallback: Any = None,
) -> Any:
    """
    Import a module optionally, returning fallback if unavailable.

    Example:
        vllm = optional_import("vllm")
        LlamaForCausalLM = optional_import(
            "LlamaForCausalLM",
            package="transformers",
            fallback=None
        )
    """
    try:
        if package:
            module = importlib.import_module(package)
            return getattr(module, name, fallback)
        return importlib.import_module(name)
    except ImportError:
        return fallback


# =============================================================================
# TORCH UTILITIES
# =============================================================================

class TorchBackend:
    """
    Unified interface for PyTorch backends with automatic selection.

    Handles:
    - CUDA for NVIDIA GPUs
    - MPS for Apple Silicon
    - CPU fallback
    - Mixed precision support
    - Memory management
    """

    def __init__(self):
        self._checker = DependencyChecker()
        self._device: Optional[str] = None
        self._dtype: Optional[Any] = None

    @property
    def device(self) -> str:
        """Get optimal device."""
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    @property
    def dtype(self) -> Any:
        """Get optimal dtype for current device."""
        if self._dtype is None:
            self._dtype = self._detect_dtype()
        return self._dtype

    def _detect_device(self) -> str:
        """Detect best available device."""
        torch = self._checker.get_module("torch")
        if torch is None:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _detect_dtype(self) -> Any:
        """Detect optimal dtype."""
        torch = self._checker.get_module("torch")
        if torch is None:
            return None

        if self.device == "cuda":
            # Check for bfloat16 support
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16

        if self.device == "mps":
            return torch.float16

        return torch.float32

    def get_device(self, device_id: Optional[int] = None) -> Any:
        """Get torch device object."""
        torch = self._checker.get_module("torch", required=True)

        if self.device == "cuda" and device_id is not None:
            return torch.device(f"cuda:{device_id}")

        return torch.device(self.device)

    def to_device(self, tensor_or_model: Any, device_id: Optional[int] = None) -> Any:
        """Move tensor or model to optimal device."""
        device = self.get_device(device_id)
        return tensor_or_model.to(device)

    def empty_cache(self) -> None:
        """Clear device cache."""
        torch = self._checker.get_module("torch")
        if torch is None:
            return

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for current device."""
        torch = self._checker.get_module("torch")
        if torch is None:
            return {}

        if self.device == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
        elif self.device == "mps":
            if hasattr(torch.mps, "current_allocated_memory"):
                return {
                    "allocated": torch.mps.current_allocated_memory(),
                }

        return {}

    def supports_flash_attention(self) -> bool:
        """Check if flash attention is supported."""
        if self.device != "cuda":
            return False
        return self._checker.is_available("flash_attn")

    def supports_bitsandbytes(self) -> bool:
        """Check if bitsandbytes quantization is supported."""
        if self.device != "cuda":
            return False
        return self._checker.is_available("bitsandbytes")

    def get_quantization_config(self, bits: int = 4) -> Optional[Any]:
        """Get appropriate quantization config for current platform."""
        if not self.supports_bitsandbytes():
            return None

        try:
            from transformers import BitsAndBytesConfig
            import torch

            if bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif bits == 8:
                return BitsAndBytesConfig(load_in_8bit=True)

        except ImportError:
            pass

        return None


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Singleton instances
_dependency_checker: Optional[DependencyChecker] = None
_torch_backend: Optional[TorchBackend] = None


def get_dependency_checker() -> DependencyChecker:
    """Get global dependency checker instance."""
    global _dependency_checker
    if _dependency_checker is None:
        _dependency_checker = DependencyChecker()
    return _dependency_checker


def get_torch_backend() -> TorchBackend:
    """Get global torch backend instance."""
    global _torch_backend
    if _torch_backend is None:
        _torch_backend = TorchBackend()
    return _torch_backend


# =============================================================================
# STARTUP VALIDATION
# =============================================================================

def validate_environment(strict: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate the runtime environment.

    Args:
        strict: If True, fail on any missing dependency

    Returns:
        (success, report) tuple
    """
    checker = get_dependency_checker()

    # Check core dependencies
    core_ok, core_missing = checker.check_core()

    if not core_ok:
        missing_names = [d.name for d in core_missing]
        logger.error(f"Missing core dependencies: {missing_names}")

        if strict:
            return False, {"error": "Missing core dependencies", "missing": missing_names}

    # Get full report
    report = checker.get_status_report()

    # Add recommendations
    recommendations = []

    if checker.accelerator == AcceleratorType.CPU:
        recommendations.append("Consider using a GPU for better performance")

    if checker.platform == Platform.MACOS_ARM:
        if not checker.is_available("mlx"):
            recommendations.append("Consider installing mlx for Apple Silicon optimization")

    if checker.platform == Platform.LINUX:
        if not checker.is_available("flash_attn"):
            recommendations.append("Consider installing flash-attn for faster attention")
        if not checker.is_available("vllm"):
            recommendations.append("Consider installing vLLM for faster inference")

    report["recommendations"] = recommendations

    success = core_ok or not strict
    return success, report


def print_environment_status() -> None:
    """Print environment status to console."""
    success, report = validate_environment()

    print("\n" + "=" * 60)
    print("AGI OS - Reactor Core Environment Status")
    print("=" * 60)

    print(f"\nPlatform: {report['platform']}")
    print(f"Accelerator: {report['accelerator']}")

    print(f"\nDependencies:")
    print(f"  Available: {report['summary']['available']}")
    print(f"  Missing: {report['summary']['missing']}")
    print(f"  Optional Missing: {report['summary']['optional_missing']}")
    print(f"  Platform Unsupported: {report['summary']['platform_unsupported']}")

    if report['issues']['missing']:
        print("\n⚠️  Missing Core Dependencies:")
        for dep in report['issues']['missing']:
            print(f"  - {dep['name']}")

    if report['issues']['import_errors']:
        print("\n⚠️  Import Errors:")
        for dep in report['issues']['import_errors']:
            print(f"  - {dep['name']}: {dep['import_error']}")

    if report.get('recommendations'):
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

    print("\n" + "=" * 60)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DependencyStatus",
    "Platform",
    "AcceleratorType",
    # Data structures
    "DependencyInfo",
    # Core classes
    "DependencyChecker",
    "LazyModule",
    "TorchBackend",
    # Functions
    "detect_platform",
    "detect_accelerator",
    "lazy_import",
    "requires",
    "optional_import",
    "get_dependency_checker",
    "get_torch_backend",
    "validate_environment",
    "print_environment_status",
    # Constants
    "CORE_DEPENDENCIES",
    "OPTIONAL_DEPENDENCIES",
    "VERSION_REQUIREMENTS",
]
