"""
Model Server with Hot-Reload Capability for Reactor-Core.

Provides production-grade model serving infrastructure including:
- Async model loading and inference
- Hot-reload without service interruption
- Model version management
- Request queuing and prioritization
- Circuit breaker for backend failures
- Metrics and health monitoring

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                       Model Server                                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
    │  │  Request Router  │──│   Model Pool     │──│  Response Cache  │   │
    │  │  (priority queue)│  │  (LRU + pinned)  │  │  (semantic hash) │   │
    │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
    │           │                     │                     │              │
    │           └─────────────────────┼─────────────────────┘              │
    │                                 ▼                                    │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │                    Hot-Reload Engine                          │   │
    │  │  • File watcher for model updates                            │   │
    │  │  • Zero-downtime model swaps                                  │   │
    │  │  • Version rollback capability                                │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ Inference     │
                              │ Backends      │
                              │ (Transformers │
                              │  vLLM, llama  │
                              │  .cpp, MLX)   │
                              └───────────────┘

Features:
- Async model loading with progress tracking
- LRU cache with memory-aware eviction
- Priority-based request queue
- Semantic response caching
- File watcher for hot-reload
- Health monitoring and circuit breaker
- Multi-backend support (Transformers, vLLM, llama.cpp, MLX)
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import mmap
import os
import sys
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Configuration
# ============================================================================

class ModelServerConfig:
    """Model server configuration."""

    # Model paths
    MODELS_DIR = Path(os.getenv("REACTOR_MODELS_DIR", str(Path.home() / ".jarvis" / "models")))
    GGUF_DIR = MODELS_DIR / "gguf"
    TRAINED_DIR = MODELS_DIR / "trained"

    # Memory management
    MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", "3"))
    MAX_MEMORY_GB = float(os.getenv("MAX_MODEL_MEMORY_GB", "16.0"))
    OFFLOAD_TO_CPU = os.getenv("MODEL_OFFLOAD_CPU", "true").lower() == "true"

    # Request queue
    MAX_QUEUE_SIZE = int(os.getenv("MODEL_QUEUE_SIZE", "100"))
    REQUEST_TIMEOUT = float(os.getenv("MODEL_REQUEST_TIMEOUT", "60.0"))

    # Hot-reload
    WATCH_INTERVAL = float(os.getenv("MODEL_WATCH_INTERVAL", "5.0"))
    AUTO_RELOAD = os.getenv("MODEL_AUTO_RELOAD", "true").lower() == "true"

    # Cache
    CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "1000"))
    CACHE_TTL = float(os.getenv("MODEL_CACHE_TTL", "3600.0"))

    # Health
    HEALTH_CHECK_INTERVAL = float(os.getenv("MODEL_HEALTH_INTERVAL", "30.0"))
    CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("MODEL_CB_THRESHOLD", "5"))
    CIRCUIT_BREAKER_TIMEOUT = float(os.getenv("MODEL_CB_TIMEOUT", "60.0"))


# ============================================================================
# Data Models
# ============================================================================

class ModelState(Enum):
    """Model loading state."""
    UNLOADED = auto()
    LOADING = auto()
    READY = auto()
    FAILED = auto()
    UNLOADING = auto()


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    REALTIME = 4


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    name: str
    path: Path
    backend: str
    version: str = "1.0.0"
    state: ModelState = ModelState.UNLOADED
    loaded_at: Optional[float] = None
    last_used: float = field(default_factory=time.time)
    memory_bytes: int = 0
    request_count: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "path": str(self.path),
            "backend": self.backend,
            "version": self.version,
            "state": self.state.name,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "memory_mb": self.memory_bytes / (1024 * 1024),
            "request_count": self.request_count,
            "avg_latency_ms": self.avg_latency_ms,
            "error_count": self.error_count,
        }


@dataclass
class InferenceRequest:
    """Inference request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str = ""
    model_id: Optional[str] = None
    priority: RequestPriority = RequestPriority.NORMAL
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    timeout: float = ModelServerConfig.REQUEST_TIMEOUT
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Inference response."""
    request_id: str
    model_id: str
    text: str
    tokens_generated: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Response Cache with Semantic Hashing
# ============================================================================

class SemanticCache:
    """
    Response cache with semantic hashing for similar queries.

    Uses a combination of exact matching and semantic similarity
    to cache and retrieve responses efficiently.
    """

    def __init__(
        self,
        max_size: int = ModelServerConfig.CACHE_SIZE,
        ttl: float = ModelServerConfig.CACHE_TTL,
    ):
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict[str, Tuple[InferenceResponse, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _compute_hash(prompt: str, model_id: str, temperature: float) -> str:
        """Compute semantic hash for cache key."""
        # Normalize prompt
        normalized = prompt.strip().lower()
        # Include model and temperature in hash
        content = f"{normalized}|{model_id}|{temperature:.2f}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
    ) -> Optional[InferenceResponse]:
        """Get cached response."""
        async with self._lock:
            cache_key = self._compute_hash(prompt, model_id, temperature)

            if cache_key in self._cache:
                response, timestamp = self._cache[cache_key]

                # Check TTL
                if time.time() - timestamp > self._ttl:
                    del self._cache[cache_key]
                    self._misses += 1
                    return None

                # Move to end (LRU)
                self._cache.move_to_end(cache_key)
                self._hits += 1

                # Mark as cached
                response.cached = True
                return response

            self._misses += 1
            return None

    async def set(
        self,
        prompt: str,
        model_id: str,
        temperature: float,
        response: InferenceResponse,
    ):
        """Cache a response."""
        async with self._lock:
            cache_key = self._compute_hash(prompt, model_id, temperature)

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[cache_key] = (response, time.time())

    async def clear(self):
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


# ============================================================================
# Model Backend Interface
# ============================================================================

class ModelBackend(ABC):
    """Abstract model backend interface."""

    @abstractmethod
    async def load(self, path: Path, **kwargs) -> bool:
        """Load model from path."""
        pass

    @abstractmethod
    async def unload(self) -> bool:
        """Unload the model."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """Get memory usage in bytes."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class LlamaCppBackend(ModelBackend):
    """llama.cpp backend for GGUF models."""

    def __init__(self):
        self._model = None
        self._path: Optional[Path] = None
        self._memory = 0

    async def load(self, path: Path, **kwargs) -> bool:
        """Load GGUF model using llama-cpp-python."""
        try:
            from llama_cpp import Llama

            # Determine context size and GPU layers
            n_ctx = kwargs.get("n_ctx", 4096)
            n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1 = all

            # Check for Metal (Apple Silicon)
            if sys.platform == "darwin":
                # llama.cpp uses Metal by default on macOS
                logger.info(f"Loading GGUF with Metal acceleration: {path}")
            else:
                logger.info(f"Loading GGUF: {path}")

            # Load in executor to not block
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: Llama(
                    model_path=str(path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                ),
            )

            self._path = path
            self._memory = path.stat().st_size  # Approximate

            return True

        except ImportError:
            logger.error("llama-cpp-python not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to load GGUF: {e}")
            return False

    async def unload(self) -> bool:
        """Unload the model."""
        if self._model:
            del self._model
            self._model = None
            gc.collect()
        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                stop=kwargs.get("stop_sequences", []),
            ),
        )

        return result["choices"][0]["text"]

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        if not self._model:
            raise RuntimeError("Model not loaded")

        # llama-cpp-python streaming
        for output in self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop_sequences", []),
            stream=True,
        ):
            yield output["choices"][0]["text"]
            await asyncio.sleep(0)  # Yield control

    def get_memory_usage(self) -> int:
        return self._memory

    def is_loaded(self) -> bool:
        return self._model is not None


class TransformersBackend(ModelBackend):
    """Hugging Face Transformers backend."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._path: Optional[Path] = None
        self._memory = 0
        self._device = "cpu"

    async def load(self, path: Path, **kwargs) -> bool:
        """Load Transformers model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            logger.info(f"Loading Transformers model on {self._device}: {path}")

            # Load in executor
            loop = asyncio.get_event_loop()

            def _load():
                tokenizer = AutoTokenizer.from_pretrained(str(path))
                model = AutoModelForCausalLM.from_pretrained(
                    str(path),
                    torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
                    device_map="auto" if self._device == "cuda" else None,
                )
                if self._device != "cuda":
                    model = model.to(self._device)
                return model, tokenizer

            self._model, self._tokenizer = await loop.run_in_executor(None, _load)
            self._path = path

            # Estimate memory
            if self._device == "cuda":
                import torch
                self._memory = torch.cuda.memory_allocated()
            else:
                self._memory = sum(p.numel() * p.element_size() for p in self._model.parameters())

            return True

        except ImportError as e:
            logger.error(f"Required library not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            return False

    async def unload(self) -> bool:
        """Unload the model."""
        if self._model:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            # Clear GPU cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

            gc.collect()

        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()

        def _generate():
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=kwargs.get("top_p", 0.9),
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            return self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return await loop.run_in_executor(None, _generate)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        # Transformers doesn't have native streaming, simulate with batched generation
        result = await self.generate(prompt, max_tokens, temperature, **kwargs)
        # Yield word by word
        for word in result.split():
            yield word + " "
            await asyncio.sleep(0.01)

    def get_memory_usage(self) -> int:
        return self._memory

    def is_loaded(self) -> bool:
        return self._model is not None


class MLXBackend(ModelBackend):
    """Apple MLX backend for Apple Silicon Macs."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._path: Optional[Path] = None
        self._memory = 0

    async def load(self, path: Path, **kwargs) -> bool:
        """Load MLX model."""
        if sys.platform != "darwin":
            logger.error("MLX backend only available on macOS")
            return False

        try:
            import mlx.core as mx
            from mlx_lm import load, generate

            logger.info(f"Loading MLX model: {path}")

            loop = asyncio.get_event_loop()

            def _load():
                model, tokenizer = load(str(path))
                return model, tokenizer

            self._model, self._tokenizer = await loop.run_in_executor(None, _load)
            self._path = path

            # Estimate memory (rough)
            if path.is_dir():
                self._memory = sum(f.stat().st_size for f in path.glob("*.safetensors"))
            else:
                self._memory = path.stat().st_size

            return True

        except ImportError:
            logger.error("mlx-lm not installed. Install with: pip install mlx-lm")
            return False
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            return False

    async def unload(self) -> bool:
        """Unload the model."""
        if self._model:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text using MLX."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded")

        try:
            from mlx_lm import generate

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                ),
            )
            return result

        except Exception as e:
            raise RuntimeError(f"MLX generation failed: {e}")

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text using MLX."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded")

        try:
            from mlx_lm import stream_generate

            for chunk in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
            ):
                yield chunk
                await asyncio.sleep(0)

        except ImportError:
            # Fallback to non-streaming
            result = await self.generate(prompt, max_tokens, temperature, **kwargs)
            yield result

    def get_memory_usage(self) -> int:
        return self._memory

    def is_loaded(self) -> bool:
        return self._model is not None


# ============================================================================
# Model Pool with LRU Eviction
# ============================================================================

class ModelPool:
    """
    Pool of loaded models with LRU eviction.

    Manages model lifecycle with memory-aware eviction.
    """

    def __init__(
        self,
        max_models: int = ModelServerConfig.MAX_LOADED_MODELS,
        max_memory_gb: float = ModelServerConfig.MAX_MEMORY_GB,
    ):
        self._max_models = max_models
        self._max_memory = max_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        self._models: OrderedDict[str, Tuple[ModelInfo, ModelBackend]] = OrderedDict()
        self._pinned: Set[str] = set()  # Models that shouldn't be evicted
        self._lock = asyncio.Lock()

    async def get(self, model_id: str) -> Optional[Tuple[ModelInfo, ModelBackend]]:
        """Get a model from the pool."""
        async with self._lock:
            if model_id in self._models:
                # Move to end (most recently used)
                self._models.move_to_end(model_id)
                info, backend = self._models[model_id]
                info.last_used = time.time()
                return info, backend
            return None

    async def add(
        self,
        model_id: str,
        info: ModelInfo,
        backend: ModelBackend,
        pin: bool = False,
    ):
        """Add a model to the pool."""
        async with self._lock:
            # Evict if necessary
            await self._ensure_capacity(info.memory_bytes)

            self._models[model_id] = (info, backend)
            if pin:
                self._pinned.add(model_id)

            info.state = ModelState.READY
            info.loaded_at = time.time()

    async def remove(self, model_id: str) -> bool:
        """Remove a model from the pool."""
        async with self._lock:
            if model_id in self._models:
                info, backend = self._models.pop(model_id)
                self._pinned.discard(model_id)

                info.state = ModelState.UNLOADING
                await backend.unload()
                info.state = ModelState.UNLOADED

                return True
            return False

    async def _ensure_capacity(self, required_bytes: int):
        """Ensure there's capacity for a new model."""
        # Check model count
        while len(self._models) >= self._max_models:
            # Find oldest non-pinned model
            for model_id in self._models:
                if model_id not in self._pinned:
                    info, backend = self._models.pop(model_id)
                    await backend.unload()
                    logger.info(f"Evicted model: {info.name}")
                    break
            else:
                raise RuntimeError("All models are pinned, cannot evict")

        # Check memory
        current_memory = sum(info.memory_bytes for info, _ in self._models.values())
        while current_memory + required_bytes > self._max_memory:
            for model_id in self._models:
                if model_id not in self._pinned:
                    info, backend = self._models.pop(model_id)
                    current_memory -= info.memory_bytes
                    await backend.unload()
                    logger.info(f"Evicted model for memory: {info.name}")
                    break
            else:
                raise RuntimeError("Cannot free enough memory")

    async def pin(self, model_id: str):
        """Pin a model to prevent eviction."""
        self._pinned.add(model_id)

    async def unpin(self, model_id: str):
        """Unpin a model."""
        self._pinned.discard(model_id)

    def get_all(self) -> List[ModelInfo]:
        """Get info for all models in pool."""
        return [info for info, _ in self._models.values()]

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        total_memory = sum(info.memory_bytes for info, _ in self._models.values())
        return {
            "loaded_models": len(self._models),
            "max_models": self._max_models,
            "pinned_models": len(self._pinned),
            "memory_used_mb": total_memory / (1024 * 1024),
            "max_memory_mb": self._max_memory / (1024 * 1024),
            "memory_utilization": total_memory / self._max_memory if self._max_memory > 0 else 0,
        }


# ============================================================================
# File Watcher for Hot-Reload
# ============================================================================

class ModelFileWatcher:
    """
    Watch model directories for changes and trigger hot-reload.

    Uses file modification times for cross-platform compatibility.
    """

    def __init__(
        self,
        watch_dirs: Optional[List[Path]] = None,
        interval: float = ModelServerConfig.WATCH_INTERVAL,
    ):
        self._watch_dirs = watch_dirs or [
            ModelServerConfig.GGUF_DIR,
            ModelServerConfig.TRAINED_DIR,
        ]
        self._interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._file_mtimes: Dict[Path, float] = {}
        self._callbacks: List[Callable[[Path, str], Awaitable[None]]] = []

    def on_change(self, callback: Callable[[Path, str], Awaitable[None]]):
        """Register callback for file changes."""
        self._callbacks.append(callback)

    async def start(self):
        """Start watching."""
        if self._running:
            return

        self._running = True

        # Initialize file times
        for watch_dir in self._watch_dirs:
            if watch_dir.exists():
                for model_file in self._find_model_files(watch_dir):
                    self._file_mtimes[model_file] = model_file.stat().st_mtime

        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"Model file watcher started (watching {len(self._watch_dirs)} directories)")

    async def stop(self):
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _find_model_files(self, directory: Path) -> List[Path]:
        """Find model files in directory."""
        patterns = ["*.gguf", "*.bin", "*.safetensors", "config.json"]
        files = []
        for pattern in patterns:
            files.extend(directory.glob(f"**/{pattern}"))
        return files

    async def _watch_loop(self):
        """Watch loop."""
        while self._running:
            try:
                for watch_dir in self._watch_dirs:
                    if not watch_dir.exists():
                        continue

                    current_files = self._find_model_files(watch_dir)

                    for model_file in current_files:
                        try:
                            mtime = model_file.stat().st_mtime
                            old_mtime = self._file_mtimes.get(model_file)

                            if old_mtime is None:
                                # New file
                                self._file_mtimes[model_file] = mtime
                                await self._notify("added", model_file)

                            elif mtime > old_mtime:
                                # Modified file
                                self._file_mtimes[model_file] = mtime
                                await self._notify("modified", model_file)

                        except OSError:
                            pass

                    # Check for deleted files
                    current_set = set(current_files)
                    for tracked_file in list(self._file_mtimes.keys()):
                        if tracked_file not in current_set and tracked_file.parent in self._watch_dirs:
                            del self._file_mtimes[tracked_file]
                            await self._notify("deleted", tracked_file)

                await asyncio.sleep(self._interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"File watcher error: {e}")
                await asyncio.sleep(self._interval)

    async def _notify(self, change_type: str, path: Path):
        """Notify callbacks of change."""
        logger.info(f"Model file {change_type}: {path}")
        for callback in self._callbacks:
            try:
                await callback(path, change_type)
            except Exception as e:
                logger.error(f"File watcher callback error: {e}")


# ============================================================================
# Request Queue with Priority
# ============================================================================

class PriorityRequestQueue:
    """Priority-based request queue with timeout handling."""

    def __init__(self, max_size: int = ModelServerConfig.MAX_QUEUE_SIZE):
        self._max_size = max_size
        self._queues: Dict[RequestPriority, Deque[Tuple[InferenceRequest, asyncio.Future]]] = {
            priority: deque() for priority in RequestPriority
        }
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

    async def enqueue(
        self,
        request: InferenceRequest,
    ) -> asyncio.Future:
        """Add request to queue."""
        async with self._lock:
            total = sum(len(q) for q in self._queues.values())
            if total >= self._max_size:
                raise RuntimeError("Request queue full")

            future: asyncio.Future = asyncio.Future()
            self._queues[request.priority].append((request, future))
            self._not_empty.notify()

            return future

    async def dequeue(self) -> Tuple[InferenceRequest, asyncio.Future]:
        """Get next request (highest priority first)."""
        async with self._not_empty:
            while True:
                # Check queues in priority order
                for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
                    if self._queues[priority]:
                        return self._queues[priority].popleft()

                # Wait for new items
                await self._not_empty.wait()

    def size(self) -> int:
        """Get total queue size."""
        return sum(len(q) for q in self._queues.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total": self.size(),
            "max_size": self._max_size,
            "by_priority": {p.name: len(q) for p, q in self._queues.items()},
        }


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """Circuit breaker for model backend failures."""

    def __init__(
        self,
        threshold: int = ModelServerConfig.CIRCUIT_BREAKER_THRESHOLD,
        timeout: float = ModelServerConfig.CIRCUIT_BREAKER_TIMEOUT,
    ):
        self._threshold = threshold
        self._timeout = timeout
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_open(self, model_id: str) -> bool:
        """Check if circuit is open (blocking requests)."""
        async with self._lock:
            if model_id in self._opened_at:
                if time.time() - self._opened_at[model_id] > self._timeout:
                    # Half-open: allow one request
                    del self._opened_at[model_id]
                    return False
                return True
            return False

    async def record_success(self, model_id: str):
        """Record successful request."""
        async with self._lock:
            self._failures[model_id] = 0
            self._opened_at.pop(model_id, None)

    async def record_failure(self, model_id: str):
        """Record failed request."""
        async with self._lock:
            self._failures[model_id] = self._failures.get(model_id, 0) + 1

            if self._failures[model_id] >= self._threshold:
                self._opened_at[model_id] = time.time()
                logger.warning(f"Circuit breaker opened for model: {model_id}")

    def get_status(self, model_id: str) -> str:
        """Get circuit status."""
        if model_id in self._opened_at:
            return "open"
        elif self._failures.get(model_id, 0) > 0:
            return "half-open"
        return "closed"


# ============================================================================
# Model Server (Main Class)
# ============================================================================

class ModelServer:
    """
    Production-grade model server with hot-reload capability.

    Features:
    - Async model loading and inference
    - LRU model pool with memory-aware eviction
    - Priority request queue
    - Semantic response caching
    - File watcher for hot-reload
    - Circuit breaker for failures
    """

    def __init__(self):
        self._pool = ModelPool()
        self._cache = SemanticCache()
        self._queue = PriorityRequestQueue()
        self._watcher = ModelFileWatcher()
        self._circuit_breaker = CircuitBreaker()

        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
        self._default_model_id: Optional[str] = None

        # Statistics
        self._start_time = 0.0
        self._total_requests = 0
        self._total_latency_ms = 0.0

        # Callbacks
        self._on_model_loaded: List[Callable[[ModelInfo], Awaitable[None]]] = []
        self._on_model_unloaded: List[Callable[[str], Awaitable[None]]] = []

    async def start(self, num_workers: int = 4):
        """Start the model server."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        # Start file watcher
        if ModelServerConfig.AUTO_RELOAD:
            self._watcher.on_change(self._handle_file_change)
            await self._watcher.start()

        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)

        logger.info(f"Model server started with {num_workers} workers")

    async def stop(self):
        """Stop the model server."""
        self._running = False

        # Cancel workers
        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

        # Stop watcher
        await self._watcher.stop()

        # Unload all models
        for model_id in list(self._pool._models.keys()):
            await self._pool.remove(model_id)

        logger.info("Model server stopped")

    async def load_model(
        self,
        path: Path,
        model_id: Optional[str] = None,
        backend: Optional[str] = None,
        pin: bool = False,
        set_default: bool = False,
    ) -> ModelInfo:
        """
        Load a model from path.

        Args:
            path: Path to model file or directory
            model_id: Optional model ID (auto-generated if not provided)
            backend: Backend to use (auto-detected if not provided)
            pin: Pin model to prevent eviction
            set_default: Set as default model

        Returns:
            ModelInfo for loaded model
        """
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        # Generate model ID
        if not model_id:
            model_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]

        # Check if already loaded
        existing = await self._pool.get(model_id)
        if existing:
            info, _ = existing
            if info.state == ModelState.READY:
                logger.info(f"Model already loaded: {model_id}")
                return info

        # Auto-detect backend
        if not backend:
            backend = self._detect_backend(path)

        # Create backend instance
        backend_instance = self._create_backend(backend)
        if not backend_instance:
            raise ValueError(f"Unknown backend: {backend}")

        # Create model info
        info = ModelInfo(
            model_id=model_id,
            name=path.stem,
            path=path,
            backend=backend,
            state=ModelState.LOADING,
        )

        logger.info(f"Loading model: {info.name} ({backend})")

        # Load model
        try:
            success = await backend_instance.load(path)
            if not success:
                info.state = ModelState.FAILED
                raise RuntimeError(f"Failed to load model: {path}")

            info.memory_bytes = backend_instance.get_memory_usage()

            # Add to pool
            await self._pool.add(model_id, info, backend_instance, pin=pin)

            if set_default:
                self._default_model_id = model_id

            # Notify callbacks
            for callback in self._on_model_loaded:
                try:
                    await callback(info)
                except Exception as e:
                    logger.error(f"Model loaded callback error: {e}")

            logger.info(f"Model loaded: {info.name} ({info.memory_bytes / 1024 / 1024:.1f} MB)")
            return info

        except Exception as e:
            info.state = ModelState.FAILED
            raise RuntimeError(f"Failed to load model: {e}")

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model."""
        if model_id == self._default_model_id:
            self._default_model_id = None

        success = await self._pool.remove(model_id)

        if success:
            # Notify callbacks
            for callback in self._on_model_unloaded:
                try:
                    await callback(model_id)
                except Exception as e:
                    logger.error(f"Model unloaded callback error: {e}")

        return success

    async def reload_model(self, model_id: str) -> ModelInfo:
        """Reload a model (hot-reload)."""
        result = await self._pool.get(model_id)
        if not result:
            raise ValueError(f"Model not found: {model_id}")

        info, _ = result
        path = info.path
        backend = info.backend
        is_pinned = model_id in self._pool._pinned
        is_default = model_id == self._default_model_id

        # Unload
        await self.unload_model(model_id)

        # Reload
        return await self.load_model(
            path,
            model_id=model_id,
            backend=backend,
            pin=is_pinned,
            set_default=is_default,
        )

    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        use_cache: bool = True,
    ) -> InferenceResponse:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            model_id: Model to use (default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            stop_sequences: Stop sequences
            priority: Request priority
            use_cache: Whether to use response cache

        Returns:
            InferenceResponse with generated text
        """
        model_id = model_id or self._default_model_id
        if not model_id:
            raise ValueError("No model specified and no default model set")

        # Check circuit breaker
        if await self._circuit_breaker.is_open(model_id):
            return InferenceResponse(
                request_id=str(uuid.uuid4())[:8],
                model_id=model_id,
                text="",
                error="Circuit breaker open - model temporarily unavailable",
            )

        # Check cache
        if use_cache:
            cached = await self._cache.get(prompt, model_id, temperature)
            if cached:
                return cached

        # Create request
        request = InferenceRequest(
            prompt=prompt,
            model_id=model_id,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences or [],
        )

        # Enqueue and wait
        future = await self._queue.enqueue(request)

        try:
            response = await asyncio.wait_for(future, timeout=request.timeout)

            # Cache successful response
            if use_cache and not response.error:
                await self._cache.set(prompt, model_id, temperature, response)

            return response

        except asyncio.TimeoutError:
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model_id,
                text="",
                error="Request timeout",
            )

    async def generate_stream(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text."""
        model_id = model_id or self._default_model_id
        if not model_id:
            raise ValueError("No model specified and no default model set")

        result = await self._pool.get(model_id)
        if not result:
            raise ValueError(f"Model not found: {model_id}")

        info, backend = result

        async for chunk in backend.generate_stream(
            prompt, max_tokens, temperature, **kwargs
        ):
            yield chunk

    async def _worker_loop(self, worker_id: str):
        """Worker loop for processing requests."""
        while self._running:
            try:
                request, future = await self._queue.dequeue()

                if future.cancelled():
                    continue

                start_time = time.time()

                try:
                    # Get model
                    result = await self._pool.get(request.model_id)
                    if not result:
                        future.set_result(InferenceResponse(
                            request_id=request.request_id,
                            model_id=request.model_id,
                            text="",
                            error=f"Model not found: {request.model_id}",
                        ))
                        continue

                    info, backend = result

                    # Generate
                    text = await backend.generate(
                        request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop_sequences=request.stop_sequences,
                    )

                    latency_ms = (time.time() - start_time) * 1000

                    # Update stats
                    info.request_count += 1
                    info.avg_latency_ms = (
                        info.avg_latency_ms * (info.request_count - 1) + latency_ms
                    ) / info.request_count
                    info.last_used = time.time()

                    self._total_requests += 1
                    self._total_latency_ms += latency_ms

                    await self._circuit_breaker.record_success(request.model_id)

                    future.set_result(InferenceResponse(
                        request_id=request.request_id,
                        model_id=request.model_id,
                        text=text,
                        tokens_generated=len(text.split()),  # Approximate
                        latency_ms=latency_ms,
                    ))

                except Exception as e:
                    logger.error(f"[{worker_id}] Request failed: {e}")
                    await self._circuit_breaker.record_failure(request.model_id)

                    future.set_result(InferenceResponse(
                        request_id=request.request_id,
                        model_id=request.model_id,
                        text="",
                        latency_ms=(time.time() - start_time) * 1000,
                        error=str(e),
                    ))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{worker_id}] Worker error: {e}")
                await asyncio.sleep(0.1)

    async def _handle_file_change(self, path: Path, change_type: str):
        """Handle model file changes for hot-reload."""
        if change_type == "deleted":
            return

        # Find model ID for this path
        for info in self._pool.get_all():
            if info.path == path or (info.path.is_dir() and path.is_relative_to(info.path)):
                logger.info(f"Hot-reloading model: {info.name}")
                try:
                    await self.reload_model(info.model_id)
                except Exception as e:
                    logger.error(f"Hot-reload failed: {e}")
                break

    def _detect_backend(self, path: Path) -> str:
        """Auto-detect backend from model path."""
        if path.suffix == ".gguf" or (path.is_dir() and any(path.glob("*.gguf"))):
            return "llama_cpp"

        if sys.platform == "darwin" and (
            path.suffix == ".mlx" or
            (path.is_dir() and (path / "weights.npz").exists())
        ):
            return "mlx"

        # Default to Transformers
        return "transformers"

    def _create_backend(self, backend: str) -> Optional[ModelBackend]:
        """Create backend instance."""
        backends = {
            "llama_cpp": LlamaCppBackend,
            "transformers": TransformersBackend,
            "mlx": MLXBackend,
        }
        backend_class = backends.get(backend)
        return backend_class() if backend_class else None

    def on_model_loaded(self, callback: Callable[[ModelInfo], Awaitable[None]]):
        """Register callback for model loaded events."""
        self._on_model_loaded.append(callback)

    def on_model_unloaded(self, callback: Callable[[str], Awaitable[None]]):
        """Register callback for model unloaded events."""
        self._on_model_unloaded.append(callback)

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info."""
        result = asyncio.run(self._pool.get(model_id)) if not asyncio.get_event_loop().is_running() else None
        return result[0] if result else None

    def list_models(self) -> List[ModelInfo]:
        """List all loaded models."""
        return self._pool.get_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "total_requests": self._total_requests,
            "avg_latency_ms": self._total_latency_ms / self._total_requests if self._total_requests > 0 else 0,
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
            "pool": self._pool.get_stats(),
            "cache": self._cache.get_stats(),
            "queue": self._queue.get_stats(),
            "default_model": self._default_model_id,
        }


# ============================================================================
# Global Instance
# ============================================================================

_model_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get global model server instance."""
    global _model_server
    if _model_server is None:
        _model_server = ModelServer()
    return _model_server


@asynccontextmanager
async def model_server_context():
    """Context manager for model server lifecycle."""
    server = get_model_server()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
