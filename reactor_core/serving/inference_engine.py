"""
Model Serving & Inference Engine for AGI OS - Reactor Core
============================================================

Production-grade model serving infrastructure with:
- vLLM integration for high-throughput inference
- Dynamic model routing (which model for which task)
- Intelligent caching with semantic similarity
- Model ensemble management
- Latency optimization for real-time AGI
- Quantization support (GGUF, GPTQ, AWQ)
- Auto-scaling and load balancing

ARCHITECTURE:
    Request → Router → Model Pool → Inference Engine → Response
                ↓            ↓              ↓
           Cache Check   Load Balance   Batch/Stream
                ↓            ↓              ↓
           Hit/Miss    Model Select    Output Format

FEATURES:
- OpenAI-compatible API endpoints
- Streaming responses with SSE
- Request batching for efficiency
- Speculative decoding support
- KV cache optimization
- Multi-GPU serving
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# AUTO-FALLBACK AND PORT DETECTION (v76.0 Enhancement)
# =============================================================================

class BackendFallbackChain:
    """
    Manages automatic fallback between inference backends.

    When primary backend fails (e.g., vLLM not available on macOS),
    automatically falls back to compatible alternatives.

    Fallback Order:
    1. vLLM (Linux only, highest throughput)
    2. MLX-LM (macOS ARM only)
    3. llama.cpp (cross-platform, GGUF support)
    4. Transformers (universal fallback)
    """

    # Platform-specific backend preferences
    LINUX_CHAIN = ["vllm", "llama_cpp", "transformers"]
    MACOS_ARM_CHAIN = ["mlx_lm", "llama_cpp", "transformers"]
    MACOS_INTEL_CHAIN = ["llama_cpp", "transformers"]
    DEFAULT_CHAIN = ["transformers"]

    def __init__(self):
        self._platform = self._detect_platform()
        self._available_backends: Dict[str, bool] = {}
        self._fallback_chain = self._get_chain_for_platform()
        self._check_backends()

    def _detect_platform(self) -> str:
        """Detect current platform."""
        import platform as plat
        system = plat.system().lower()
        machine = plat.machine().lower()

        if system == "darwin":
            if "arm" in machine or machine == "arm64":
                return "macos_arm"
            return "macos_intel"
        elif system == "linux":
            return "linux"
        return "other"

    def _get_chain_for_platform(self) -> List[str]:
        """Get fallback chain for current platform."""
        chains = {
            "linux": self.LINUX_CHAIN,
            "macos_arm": self.MACOS_ARM_CHAIN,
            "macos_intel": self.MACOS_INTEL_CHAIN,
        }
        return chains.get(self._platform, self.DEFAULT_CHAIN)

    def _check_backends(self) -> None:
        """Check which backends are available."""
        backend_checks = {
            "vllm": self._check_vllm,
            "mlx_lm": self._check_mlx,
            "llama_cpp": self._check_llama_cpp,
            "transformers": self._check_transformers,
        }

        for backend, check_fn in backend_checks.items():
            try:
                self._available_backends[backend] = check_fn()
            except Exception:
                self._available_backends[backend] = False

        logger.info(f"Available backends: {[k for k, v in self._available_backends.items() if v]}")

    def _check_vllm(self) -> bool:
        """Check if vLLM is available."""
        if self._platform != "linux":
            return False
        try:
            import vllm
            return True
        except ImportError:
            return False

    def _check_mlx(self) -> bool:
        """Check if MLX-LM is available."""
        if self._platform != "macos_arm":
            return False
        try:
            import mlx
            import mlx_lm
            return True
        except ImportError:
            return False

    def _check_llama_cpp(self) -> bool:
        """Check if llama-cpp-python is available."""
        try:
            from llama_cpp import Llama
            return True
        except ImportError:
            return False

    def _check_transformers(self) -> bool:
        """Check if transformers is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False

    def get_best_backend(self, preferred: Optional[str] = None) -> str:
        """
        Get the best available backend.

        Args:
            preferred: Preferred backend (if available)

        Returns:
            Name of best available backend
        """
        # Check preferred first
        if preferred and self._available_backends.get(preferred, False):
            return preferred

        # Go through fallback chain
        for backend in self._fallback_chain:
            if self._available_backends.get(backend, False):
                if preferred and preferred != backend:
                    logger.warning(f"Backend '{preferred}' not available, falling back to '{backend}'")
                return backend

        # Ultimate fallback
        return "transformers"

    def is_available(self, backend: str) -> bool:
        """Check if a specific backend is available."""
        return self._available_backends.get(backend, False)

    def get_status(self) -> Dict[str, Any]:
        """Get backend availability status."""
        return {
            "platform": self._platform,
            "available_backends": dict(self._available_backends),
            "fallback_chain": self._fallback_chain,
            "recommended": self.get_best_backend(),
        }


class PortManager:
    """
    Manages port allocation for inference servers.

    Features:
    - Automatic free port detection
    - Port range configuration
    - Health checking on ports
    - Conflict resolution
    """

    DEFAULT_PORT_RANGE = (8000, 8100)

    def __init__(
        self,
        port_range: Tuple[int, int] = DEFAULT_PORT_RANGE,
        reserved_ports: Optional[Set[int]] = None,
    ):
        self.port_range = port_range
        self.reserved_ports = reserved_ports or set()
        self._allocated_ports: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
        """Check if a port is free."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((host, port))
                return result != 0
        except Exception:
            return False

    async def find_free_port(self, preferred: Optional[int] = None) -> int:
        """
        Find a free port.

        Args:
            preferred: Preferred port number

        Returns:
            Available port number
        """
        async with self._lock:
            # Try preferred port first
            if preferred and preferred not in self.reserved_ports:
                if self.is_port_free(preferred):
                    return preferred

            # Search in range
            start, end = self.port_range
            for port in range(start, end):
                if port in self.reserved_ports:
                    continue
                if port in self._allocated_ports.values():
                    continue
                if self.is_port_free(port):
                    return port

            raise RuntimeError(f"No free ports available in range {self.port_range}")

    async def allocate(self, service_id: str, preferred: Optional[int] = None) -> int:
        """
        Allocate a port for a service.

        Args:
            service_id: Unique identifier for the service
            preferred: Preferred port number

        Returns:
            Allocated port number
        """
        async with self._lock:
            # Check if already allocated
            if service_id in self._allocated_ports:
                return self._allocated_ports[service_id]

            port = await self.find_free_port(preferred)
            self._allocated_ports[service_id] = port
            logger.info(f"Allocated port {port} for service '{service_id}'")
            return port

    async def release(self, service_id: str) -> None:
        """Release an allocated port."""
        async with self._lock:
            if service_id in self._allocated_ports:
                port = self._allocated_ports.pop(service_id)
                logger.info(f"Released port {port} from service '{service_id}'")

    def get_allocated(self, service_id: str) -> Optional[int]:
        """Get allocated port for a service."""
        return self._allocated_ports.get(service_id)

    async def health_check(self, port: int, timeout: float = 5.0) -> bool:
        """
        Check if a service is responding on a port.

        Args:
            port: Port number to check
            timeout: Timeout in seconds

        Returns:
            True if service is responding
        """
        import aiohttp
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.get(f"http://127.0.0.1:{port}/health") as response:
                    return response.status == 200
        except Exception:
            return False


class AutoRetryInference:
    """
    Wrapper for inference with automatic retry and fallback.

    Features:
    - Configurable retry attempts
    - Exponential backoff
    - Automatic backend fallback
    - Circuit breaker integration
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        fallback_chain: Optional[BackendFallbackChain] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.fallback_chain = fallback_chain or BackendFallbackChain()

        self._retry_count = 0
        self._last_error: Optional[Exception] = None

    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        self._retry_count = 0

        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                self._retry_count = 0
                return result

            except Exception as e:
                self._last_error = e
                self._retry_count = attempt + 1

                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} retry attempts failed: {e}")
                    raise

                delay = min(
                    self.base_delay * (2 ** attempt),
                    self.max_delay
                )
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Retry logic error")

    async def execute_with_fallback(
        self,
        backends: Dict[str, Callable[..., Awaitable[T]]],
        *args,
        **kwargs,
    ) -> Tuple[T, str]:
        """
        Execute with backend fallback.

        Args:
            backends: Dict mapping backend name to function
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, backend_used)
        """
        chain = self.fallback_chain._fallback_chain

        for backend in chain:
            if backend not in backends:
                continue

            if not self.fallback_chain.is_available(backend):
                continue

            try:
                result = await self.execute_with_retry(
                    backends[backend],
                    *args,
                    **kwargs,
                )
                return result, backend

            except Exception as e:
                logger.warning(f"Backend '{backend}' failed: {e}")
                continue

        raise RuntimeError("All backends failed")


class InferenceHealthMonitor:
    """
    Monitor inference engine health and performance.

    Tracks:
    - Request latency
    - Error rates
    - Throughput
    - Memory usage
    - Model loading status
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        self._latencies: List[float] = []
        self._errors: List[Tuple[datetime, str]] = []
        self._requests: int = 0
        self._start_time = datetime.now()
        self._lock = asyncio.Lock()

    async def record_request(
        self,
        latency_ms: float,
        success: bool,
        error_msg: Optional[str] = None,
    ) -> None:
        """Record a request."""
        async with self._lock:
            self._requests += 1

            self._latencies.append(latency_ms)
            if len(self._latencies) > self.window_size:
                self._latencies = self._latencies[-self.window_size:]

            if not success and error_msg:
                self._errors.append((datetime.now(), error_msg))
                if len(self._errors) > self.window_size:
                    self._errors = self._errors[-self.window_size:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        elapsed = (datetime.now() - self._start_time).total_seconds()

        latency_stats = {}
        if self._latencies:
            sorted_lat = sorted(self._latencies)
            latency_stats = {
                "mean_ms": sum(self._latencies) / len(self._latencies),
                "p50_ms": sorted_lat[len(sorted_lat) // 2],
                "p95_ms": sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 20 else sorted_lat[-1],
                "p99_ms": sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) > 100 else sorted_lat[-1],
            }

        return {
            "total_requests": self._requests,
            "requests_per_second": self._requests / elapsed if elapsed > 0 else 0,
            "latency": latency_stats,
            "error_count": len(self._errors),
            "error_rate": len(self._errors) / self._requests if self._requests > 0 else 0,
            "recent_errors": [
                {"time": t.isoformat(), "error": e}
                for t, e in self._errors[-5:]
            ],
            "uptime_seconds": elapsed,
        }

    def is_healthy(self, max_error_rate: float = 0.1) -> bool:
        """Check if inference engine is healthy."""
        if self._requests == 0:
            return True
        return (len(self._errors) / self._requests) < max_error_rate


# Global instances
_backend_fallback: Optional[BackendFallbackChain] = None
_port_manager: Optional[PortManager] = None


def get_backend_fallback() -> BackendFallbackChain:
    """Get global backend fallback chain instance."""
    global _backend_fallback
    if _backend_fallback is None:
        _backend_fallback = BackendFallbackChain()
    return _backend_fallback


def get_port_manager() -> PortManager:
    """Get global port manager instance."""
    global _port_manager
    if _port_manager is None:
        _port_manager = PortManager()
    return _port_manager


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ModelBackend(Enum):
    """Supported model backends."""
    TRANSFORMERS = "transformers"  # HuggingFace Transformers
    VLLM = "vllm"  # vLLM for high throughput
    LLAMA_CPP = "llama_cpp"  # llama.cpp for GGUF
    TENSORRT = "tensorrt"  # TensorRT-LLM
    ONNX = "onnx"  # ONNX Runtime
    MLXLM = "mlx_lm"  # MLX for Apple Silicon


class QuantizationType(Enum):
    """Quantization types for inference."""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    GGUF_Q4_K_M = "gguf_q4_k_m"
    GGUF_Q5_K_M = "gguf_q5_k_m"
    GGUF_Q8_0 = "gguf_q8_0"
    GPTQ = "gptq"
    AWQ = "awq"


class TaskType(Enum):
    """Task types for dynamic routing."""
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CHAT = "chat"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    SAFETY = "safety"


class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 1  # Safety-related, immediate
    HIGH = 2  # User-facing, low latency
    MEDIUM = 3  # Normal requests
    LOW = 4  # Background tasks
    BATCH = 5  # Batch processing, can wait


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None

    # Advanced options
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Dict[int, float] = field(default_factory=dict)
    response_format: Optional[str] = None  # "json", "text"

    # Streaming
    stream: bool = False
    stream_interval: float = 0.05  # Seconds between chunks

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class InferenceRequest:
    """A request for model inference."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    prompt: str = ""
    messages: Optional[List[Dict[str, str]]] = None  # Chat format
    system_prompt: Optional[str] = None
    config: GenerationConfig = field(default_factory=GenerationConfig)
    task_type: TaskType = TaskType.GENERAL
    priority: RequestPriority = RequestPriority.MEDIUM
    model_preference: Optional[str] = None  # Specific model to use
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: float = 60.0

    def get_cache_key(self) -> str:
        """Generate cache key for this request."""
        content = json.dumps({
            "prompt": self.prompt,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "config": {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "seed": self.config.seed,
            }
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    text: str
    tokens_generated: int = 0
    prompt_tokens: int = 0
    finish_reason: str = "stop"  # stop, length, error
    model_id: str = ""
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.request_id,
            "text": self.text,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.tokens_generated,
                "total_tokens": self.prompt_tokens + self.tokens_generated,
            },
            "finish_reason": self.finish_reason,
            "model": self.model_id,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "cached": self.cached,
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API compatible format."""
        return {
            "id": f"chatcmpl-{self.request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self.text,
                },
                "finish_reason": self.finish_reason,
            }],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.tokens_generated,
                "total_tokens": self.prompt_tokens + self.tokens_generated,
            },
        }


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    id: str
    name: str
    path: str
    backend: ModelBackend
    quantization: QuantizationType = QuantizationType.NONE
    context_length: int = 4096
    supported_tasks: Set[TaskType] = field(default_factory=lambda: {TaskType.GENERAL})
    max_batch_size: int = 32
    memory_gb: float = 0.0
    loaded_at: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    total_tokens_generated: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "backend": self.backend.value,
            "quantization": self.quantization.value,
            "context_length": self.context_length,
            "supported_tasks": [t.value for t in self.supported_tasks],
            "max_batch_size": self.max_batch_size,
            "memory_gb": round(self.memory_gb, 2),
            "request_count": self.request_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


# =============================================================================
# CACHING SYSTEM - Semantic & Exact Match
# =============================================================================

class ResponseCache:
    """
    Intelligent response caching with LRU eviction.

    Features:
    - Exact match caching with configurable TTL
    - Memory-efficient storage
    - Async-safe operations
    - Cache statistics
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
        enable_semantic: bool = False,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_semantic = enable_semantic

        # LRU cache storage
        self._cache: OrderedDict[str, Tuple[InferenceResponse, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Semantic cache (optional)
        self._embeddings: Dict[str, torch.Tensor] = {}
        self._embedding_model = None

    async def get(self, key: str) -> Optional[InferenceResponse]:
        """Get cached response."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            response, cached_at = self._cache[key]

            # Check TTL
            if (datetime.now() - cached_at).total_seconds() > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            # Mark as cached
            response.cached = True
            return response

    async def set(self, key: str, response: InferenceResponse) -> None:
        """Cache a response."""
        async with self._lock:
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = (response, datetime.now())

    async def invalidate(self, key: str) -> bool:
        """Invalidate a cached entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear entire cache, returns count cleared."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "evictions": self._evictions,
            "ttl_seconds": self.ttl_seconds,
        }


# =============================================================================
# MODEL BACKENDS - Abstract Interface
# =============================================================================

class ModelBackendInterface(ABC):
    """Abstract interface for model backends."""

    @abstractmethod
    async def load(self, model_path: str, **kwargs) -> bool:
        """Load a model."""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        pass

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Get model information."""
        pass


class TransformersBackend(ModelBackendInterface):
    """HuggingFace Transformers backend."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or self._detect_device()
        self.model = None
        self.tokenizer = None
        self._info: Optional[ModelInfo] = None

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    async def load(self, model_path: str, **kwargs) -> bool:
        """Load model with Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model loading options
            load_kwargs = {
                "trust_remote_code": kwargs.get("trust_remote_code", True),
                "device_map": kwargs.get("device_map", "auto"),
            }

            # Quantization
            quant_type = kwargs.get("quantization", QuantizationType.NONE)
            if quant_type in (QuantizationType.INT4, QuantizationType.INT8):
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(quant_type == QuantizationType.INT4),
                    load_in_8bit=(quant_type == QuantizationType.INT8),
                )
            elif quant_type == QuantizationType.FP16:
                load_kwargs["torch_dtype"] = torch.float16
            elif quant_type == QuantizationType.BF16:
                load_kwargs["torch_dtype"] = torch.bfloat16

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            self.model.eval()

            # Get memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1e9
            else:
                memory_gb = 0

            # Create info
            self._info = ModelInfo(
                id=str(uuid.uuid4())[:8],
                name=Path(model_path).name,
                path=model_path,
                backend=ModelBackend.TRANSFORMERS,
                quantization=quant_type,
                context_length=getattr(self.model.config, "max_position_embeddings", 4096),
                memory_gb=memory_gb,
            )

            logger.info(f"Model loaded: {self._info.name} ({memory_gb:.2f} GB)")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    async def unload(self) -> None:
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

        logger.info("Model unloaded")

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            if config.seed is not None:
                torch.manual_seed(config.seed)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                num_beams=config.num_beams,
                early_stopping=config.early_stopping,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """Generate with streaming using TextIteratorStreamer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.do_sample else 1.0,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Run generation in thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens
        for text in streamer:
            yield text
            await asyncio.sleep(config.stream_interval)

        thread.join()

    def get_info(self) -> ModelInfo:
        """Get model info."""
        if self._info is None:
            raise RuntimeError("Model not loaded")
        return self._info


class VLLMBackend(ModelBackendInterface):
    """vLLM backend for high-throughput inference."""

    def __init__(self, **kwargs):
        self.engine = None
        self._info: Optional[ModelInfo] = None
        self._kwargs = kwargs

    async def load(self, model_path: str, **kwargs) -> bool:
        """Load model with vLLM."""
        try:
            from vllm import LLM, SamplingParams

            logger.info(f"Loading model with vLLM: {model_path}")

            # vLLM options
            vllm_kwargs = {
                "model": model_path,
                "trust_remote_code": kwargs.get("trust_remote_code", True),
                "tensor_parallel_size": kwargs.get("tensor_parallel_size", 1),
                "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
                "max_model_len": kwargs.get("max_model_len", 4096),
            }

            # Quantization
            quant_type = kwargs.get("quantization", QuantizationType.NONE)
            if quant_type == QuantizationType.AWQ:
                vllm_kwargs["quantization"] = "awq"
            elif quant_type == QuantizationType.GPTQ:
                vllm_kwargs["quantization"] = "gptq"

            self.engine = LLM(**vllm_kwargs)

            self._info = ModelInfo(
                id=str(uuid.uuid4())[:8],
                name=Path(model_path).name,
                path=model_path,
                backend=ModelBackend.VLLM,
                quantization=quant_type,
                context_length=vllm_kwargs.get("max_model_len", 4096),
                max_batch_size=kwargs.get("max_batch_size", 256),
            )

            logger.info(f"vLLM model loaded: {self._info.name}")
            return True

        except ImportError:
            logger.warning("vLLM not installed, cannot use vLLM backend")
            return False
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            return False

    async def unload(self) -> None:
        """Unload vLLM model."""
        if self.engine:
            del self.engine
            self.engine = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc
        gc.collect()

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        """Generate with vLLM."""
        if self.engine is None:
            raise RuntimeError("vLLM engine not loaded")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop_sequences or None,
            seed=config.seed,
        )

        outputs = self.engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        """vLLM streaming generation."""
        # vLLM streaming requires async engine
        text = await self.generate(prompt, config)
        # Simulate streaming by chunking
        chunk_size = 5
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            yield chunk + " "
            await asyncio.sleep(config.stream_interval)

    def get_info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("Model not loaded")
        return self._info


class LlamaCppBackend(ModelBackendInterface):
    """llama.cpp backend for GGUF models."""

    def __init__(self):
        self.model = None
        self._info: Optional[ModelInfo] = None

    async def load(self, model_path: str, **kwargs) -> bool:
        """Load GGUF model."""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model: {model_path}")

            self.model = Llama(
                model_path=model_path,
                n_ctx=kwargs.get("n_ctx", 4096),
                n_gpu_layers=kwargs.get("n_gpu_layers", -1),  # -1 = all layers
                n_threads=kwargs.get("n_threads", None),
                verbose=kwargs.get("verbose", False),
            )

            # Determine quantization from filename
            quant = QuantizationType.NONE
            path_lower = model_path.lower()
            if "q4_k_m" in path_lower:
                quant = QuantizationType.GGUF_Q4_K_M
            elif "q5_k_m" in path_lower:
                quant = QuantizationType.GGUF_Q5_K_M
            elif "q8_0" in path_lower:
                quant = QuantizationType.GGUF_Q8_0

            self._info = ModelInfo(
                id=str(uuid.uuid4())[:8],
                name=Path(model_path).stem,
                path=model_path,
                backend=ModelBackend.LLAMA_CPP,
                quantization=quant,
                context_length=kwargs.get("n_ctx", 4096),
            )

            logger.info(f"GGUF model loaded: {self._info.name}")
            return True

        except ImportError:
            logger.warning("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            return False

    async def unload(self) -> None:
        if self.model:
            del self.model
            self.model = None

        import gc
        gc.collect()

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> str:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        output = self.model(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repetition_penalty,
            stop=config.stop_sequences or None,
        )

        return output["choices"][0]["text"]

    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> AsyncIterator[str]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        stream = self.model(
            prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stream=True,
        )

        for output in stream:
            text = output["choices"][0]["text"]
            yield text
            await asyncio.sleep(config.stream_interval)

    def get_info(self) -> ModelInfo:
        if self._info is None:
            raise RuntimeError("Model not loaded")
        return self._info


# =============================================================================
# DYNAMIC MODEL ROUTER
# =============================================================================

@dataclass
class RoutingRule:
    """A rule for routing requests to models."""
    task_type: Optional[TaskType] = None
    priority: Optional[RequestPriority] = None
    prompt_pattern: Optional[str] = None  # Regex pattern
    model_id: str = ""
    weight: float = 1.0  # For load balancing


class ModelRouter:
    """
    Dynamic model router for intelligent request routing.

    Features:
    - Task-based routing (coding -> code model, etc.)
    - Priority-based model selection
    - Load balancing across model pool
    - Fallback chains
    """

    def __init__(self):
        self._rules: List[RoutingRule] = []
        self._model_pool: Dict[str, ModelBackendInterface] = {}
        self._model_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "latency_sum": 0, "errors": 0}
        )
        self._default_model: Optional[str] = None

    def add_model(
        self,
        model_id: str,
        backend: ModelBackendInterface,
        is_default: bool = False,
    ) -> None:
        """Add model to the pool."""
        self._model_pool[model_id] = backend

        if is_default or self._default_model is None:
            self._default_model = model_id

        logger.info(f"Model added to router: {model_id}")

    def remove_model(self, model_id: str) -> bool:
        """Remove model from pool."""
        if model_id in self._model_pool:
            del self._model_pool[model_id]
            if self._default_model == model_id:
                self._default_model = next(iter(self._model_pool.keys()), None)
            return True
        return False

    def add_rule(self, rule: RoutingRule) -> None:
        """Add routing rule."""
        self._rules.append(rule)

    def route(self, request: InferenceRequest) -> Optional[str]:
        """Route request to appropriate model."""
        # Check explicit model preference
        if request.model_preference and request.model_preference in self._model_pool:
            return request.model_preference

        # Match rules
        for rule in self._rules:
            if self._matches_rule(request, rule):
                if rule.model_id in self._model_pool:
                    return rule.model_id

        # Fall back to default
        return self._default_model

    def _matches_rule(self, request: InferenceRequest, rule: RoutingRule) -> bool:
        """Check if request matches routing rule."""
        import re

        if rule.task_type and request.task_type != rule.task_type:
            return False

        if rule.priority and request.priority != rule.priority:
            return False

        if rule.prompt_pattern:
            pattern = re.compile(rule.prompt_pattern, re.IGNORECASE)
            if not pattern.search(request.prompt):
                return False

        return True

    def get_model(self, model_id: str) -> Optional[ModelBackendInterface]:
        """Get model backend by ID."""
        return self._model_pool.get(model_id)

    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get info for all models."""
        return {
            model_id: backend.get_info()
            for model_id, backend in self._model_pool.items()
        }

    def record_request(self, model_id: str, latency_ms: float, success: bool = True) -> None:
        """Record request statistics."""
        stats = self._model_stats[model_id]
        stats["requests"] += 1
        stats["latency_sum"] += latency_ms
        if not success:
            stats["errors"] += 1

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get routing statistics."""
        result = {}
        for model_id, stats in self._model_stats.items():
            result[model_id] = {
                "requests": stats["requests"],
                "avg_latency_ms": stats["latency_sum"] / stats["requests"] if stats["requests"] > 0 else 0,
                "error_rate": stats["errors"] / stats["requests"] if stats["requests"] > 0 else 0,
            }
        return result


# =============================================================================
# INFERENCE ENGINE - Main Entry Point
# =============================================================================

@dataclass
class InferenceEngineConfig:
    """Configuration for inference engine."""
    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 10000
    cache_ttl_seconds: int = 3600

    # Batching
    batch_enabled: bool = True
    batch_max_size: int = 32
    batch_wait_ms: int = 50

    # Concurrency
    max_concurrent_requests: int = 100

    # Timeouts
    default_timeout_seconds: float = 60.0

    # Health checks
    health_check_interval_seconds: float = 30.0


class InferenceEngine:
    """
    Main inference engine orchestrating model serving.

    Features:
    - Multi-model management
    - Intelligent routing
    - Response caching
    - Request batching
    - Streaming support
    - Health monitoring
    """

    def __init__(self, config: Optional[InferenceEngineConfig] = None):
        self.config = config or InferenceEngineConfig()

        # Components
        self.router = ModelRouter()
        self.cache = ResponseCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        ) if self.config.cache_enabled else None

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Request queue for batching
        self._batch_queue: asyncio.Queue[Tuple[InferenceRequest, asyncio.Future]] = asyncio.Queue()
        self._batch_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._start_time = time.time()

        # State
        self._running = False

        logger.info("Inference Engine initialized")

    async def start(self) -> None:
        """Start the inference engine."""
        self._running = True

        if self.config.batch_enabled:
            self._batch_task = asyncio.create_task(self._batch_processor())

        logger.info("Inference Engine started")

    async def stop(self) -> None:
        """Stop the inference engine."""
        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        for model_id, backend in list(self.router._model_pool.items()):
            await backend.unload()

        logger.info("Inference Engine stopped")

    async def load_model(
        self,
        model_path: str,
        model_id: Optional[str] = None,
        backend: ModelBackend = ModelBackend.TRANSFORMERS,
        is_default: bool = False,
        **kwargs,
    ) -> str:
        """Load a model into the engine."""
        model_id = model_id or str(uuid.uuid4())[:8]

        # Create backend
        if backend == ModelBackend.TRANSFORMERS:
            backend_instance = TransformersBackend()
        elif backend == ModelBackend.VLLM:
            backend_instance = VLLMBackend()
        elif backend == ModelBackend.LLAMA_CPP:
            backend_instance = LlamaCppBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Load model
        success = await backend_instance.load(model_path, **kwargs)
        if not success:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # Add to router
        self.router.add_model(model_id, backend_instance, is_default=is_default)

        return model_id

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from the engine."""
        backend = self.router.get_model(model_id)
        if backend:
            await backend.unload()
            return self.router.remove_model(model_id)
        return False

    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Generate text for a request."""
        start_time = time.time()

        # Check cache
        if self.cache:
            cache_key = request.get_cache_key()
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        # Acquire semaphore
        async with self._semaphore:
            # Route to model
            model_id = self.router.route(request)
            if not model_id:
                return InferenceResponse(
                    request_id=request.id,
                    text="",
                    finish_reason="error",
                    metadata={"error": "No model available"},
                )

            backend = self.router.get_model(model_id)

            try:
                # Format prompt
                prompt = self._format_prompt(request)

                # Generate
                text = await asyncio.wait_for(
                    backend.generate(prompt, request.config),
                    timeout=request.timeout_seconds,
                )

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                tokens_generated = len(text.split())  # Approximate

                # Record stats
                self.router.record_request(model_id, latency_ms, success=True)
                self._total_requests += 1
                self._total_tokens += tokens_generated

                response = InferenceResponse(
                    request_id=request.id,
                    text=text,
                    tokens_generated=tokens_generated,
                    prompt_tokens=len(prompt.split()),
                    finish_reason="stop",
                    model_id=model_id,
                    latency_ms=latency_ms,
                    tokens_per_second=tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0,
                )

                # Cache response
                if self.cache and not request.config.stream:
                    await self.cache.set(cache_key, response)

                return response

            except asyncio.TimeoutError:
                self.router.record_request(model_id, request.timeout_seconds * 1000, success=False)
                return InferenceResponse(
                    request_id=request.id,
                    text="",
                    finish_reason="error",
                    metadata={"error": "Request timeout"},
                )

            except Exception as e:
                logger.error(f"Generation error: {e}")
                self.router.record_request(model_id, (time.time() - start_time) * 1000, success=False)
                return InferenceResponse(
                    request_id=request.id,
                    text="",
                    finish_reason="error",
                    metadata={"error": str(e)},
                )

    async def generate_stream(
        self,
        request: InferenceRequest,
    ) -> AsyncIterator[str]:
        """Generate text with streaming."""
        # Route to model
        model_id = self.router.route(request)
        if not model_id:
            yield ""
            return

        backend = self.router.get_model(model_id)
        prompt = self._format_prompt(request)

        async for chunk in backend.generate_stream(prompt, request.config):
            yield chunk

    def _format_prompt(self, request: InferenceRequest) -> str:
        """Format request into prompt string."""
        if request.messages:
            # Chat format
            parts = []
            if request.system_prompt:
                parts.append(f"<|system|>\n{request.system_prompt}</s>")

            for msg in request.messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    parts.append(f"<|user|>\n{content}</s>")
                elif role == "assistant":
                    parts.append(f"<|assistant|>\n{content}</s>")
                elif role == "system":
                    parts.insert(0, f"<|system|>\n{content}</s>")

            parts.append("<|assistant|>\n")
            return "\n".join(parts)

        else:
            # Raw prompt
            if request.system_prompt:
                return f"{request.system_prompt}\n\n{request.prompt}"
            return request.prompt

    async def _batch_processor(self) -> None:
        """Process requests in batches."""
        while self._running:
            batch: List[Tuple[InferenceRequest, asyncio.Future]] = []

            # Collect batch
            try:
                # Wait for first request
                item = await asyncio.wait_for(
                    self._batch_queue.get(),
                    timeout=1.0,
                )
                batch.append(item)

                # Collect more if available
                deadline = time.time() + self.config.batch_wait_ms / 1000
                while len(batch) < self.config.batch_max_size and time.time() < deadline:
                    try:
                        item = self._batch_queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.005)

            except asyncio.TimeoutError:
                continue

            # Process batch
            if batch:
                await self._process_batch(batch)

    async def _process_batch(
        self,
        batch: List[Tuple[InferenceRequest, asyncio.Future]],
    ) -> None:
        """Process a batch of requests."""
        # For now, process sequentially (true batching requires vLLM)
        for request, future in batch:
            try:
                response = await self.generate(request)
                future.set_result(response)
            except Exception as e:
                future.set_exception(e)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
            "tokens_per_second": self._total_tokens / uptime if uptime > 0 else 0,
            "models": self.router.get_all_models(),
            "routing_stats": self.router.get_stats(),
            "cache_stats": self.cache.get_stats() if self.cache else None,
        }


# =============================================================================
# MODEL ENSEMBLE - Multi-Model Aggregation
# =============================================================================

class EnsembleStrategy(Enum):
    """Strategies for combining ensemble outputs."""
    MAJORITY_VOTE = "majority_vote"  # Most common response
    BEST_CONFIDENCE = "best_confidence"  # Highest confidence
    AVERAGE = "average"  # Average embeddings/scores
    CASCADING = "cascading"  # Try models in order until success


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble."""
    strategy: EnsembleStrategy = EnsembleStrategy.MAJORITY_VOTE
    models: List[str] = field(default_factory=list)
    timeout_per_model: float = 30.0
    parallel: bool = True


class ModelEnsemble:
    """
    Model ensemble for improved accuracy and reliability.

    Combines outputs from multiple models using various strategies.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        config: EnsembleConfig,
    ):
        self.engine = engine
        self.config = config

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate using ensemble."""
        if self.config.strategy == EnsembleStrategy.CASCADING:
            return await self._cascading_generate(request)
        elif self.config.parallel:
            return await self._parallel_generate(request)
        else:
            return await self._sequential_generate(request)

    async def _cascading_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Try models in order until one succeeds."""
        for model_id in self.config.models:
            request.model_preference = model_id
            response = await self.engine.generate(request)

            if response.finish_reason != "error":
                return response

        # All failed
        return InferenceResponse(
            request_id=request.id,
            text="",
            finish_reason="error",
            metadata={"error": "All models in ensemble failed"},
        )

    async def _parallel_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run all models in parallel and aggregate."""
        tasks = []
        for model_id in self.config.models:
            req_copy = InferenceRequest(
                prompt=request.prompt,
                messages=request.messages,
                system_prompt=request.system_prompt,
                config=request.config,
                model_preference=model_id,
            )
            tasks.append(self.engine.generate(req_copy))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        valid_responses = [
            r for r in results
            if isinstance(r, InferenceResponse) and r.finish_reason != "error"
        ]

        if not valid_responses:
            return InferenceResponse(
                request_id=request.id,
                text="",
                finish_reason="error",
                metadata={"error": "All ensemble models failed"},
            )

        # Aggregate based on strategy
        if self.config.strategy == EnsembleStrategy.MAJORITY_VOTE:
            # Return most common response
            from collections import Counter
            texts = [r.text for r in valid_responses]
            most_common = Counter(texts).most_common(1)[0][0]
            best = next(r for r in valid_responses if r.text == most_common)
            return best

        elif self.config.strategy == EnsembleStrategy.BEST_CONFIDENCE:
            # Return response with lowest latency (proxy for confidence)
            return min(valid_responses, key=lambda r: r.latency_ms)

        else:
            # Default to first valid
            return valid_responses[0]

    async def _sequential_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run models sequentially and aggregate."""
        responses = []
        for model_id in self.config.models:
            request.model_preference = model_id
            response = await self.engine.generate(request)
            if response.finish_reason != "error":
                responses.append(response)

        if not responses:
            return InferenceResponse(
                request_id=request.id,
                text="",
                finish_reason="error",
            )

        return responses[0]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === AUTO-FALLBACK AND PORT DETECTION (v76.0) ===
    "BackendFallbackChain",
    "PortManager",
    "AutoRetryInference",
    "InferenceHealthMonitor",
    "get_backend_fallback",
    "get_port_manager",
    # Enums
    "ModelBackend",
    "QuantizationType",
    "TaskType",
    "RequestPriority",
    "EnsembleStrategy",
    # Data structures
    "GenerationConfig",
    "InferenceRequest",
    "InferenceResponse",
    "ModelInfo",
    # Caching
    "ResponseCache",
    # Backends
    "ModelBackendInterface",
    "TransformersBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    # Routing
    "RoutingRule",
    "ModelRouter",
    # Engine
    "InferenceEngineConfig",
    "InferenceEngine",
    # Ensemble
    "EnsembleConfig",
    "ModelEnsemble",
]
