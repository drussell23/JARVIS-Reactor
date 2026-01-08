# JARVIS Reactor

**An Advanced AI/ML Training & Serving Engine for AGI OS**

JARVIS Reactor (formerly Reactor Core) is the "nervous system" of the JARVIS AGI ecosystem, providing enterprise-grade ML training, model serving, and real-time event coordination across distributed AI systems.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🚀 What is JARVIS Reactor?

JARVIS Reactor is a production-grade ML infrastructure combining:

- **Advanced Training Methods**: DPO, RLHF, Constitutional AI, Curriculum Learning
- **Model Serving**: Hot-reload model server with multi-backend support (vLLM, llama.cpp, MLX)
- **Async Infrastructure**: Circuit breakers, backpressure, bulkheads, dead letter queues
- **API Platform**: FastAPI server with telemetry, scheduling, model registry, health monitoring
- **Trinity Orchestration**: Multi-repo coordination with heartbeat monitoring and state sync
- **Event Streaming**: Real-time WebSocket/Redis pub-sub across JARVIS ecosystem
- **GCP Integration**: Spot VM resilience, Cloud SQL storage, auto-checkpointing
- **MLForge C++ Core**: High-performance ML primitives (optional submodule)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Features](#advanced-features)
  - [Advanced Training Methods (v76.0)](#advanced-training-methods-v760)
  - [Async Infrastructure (v76.1)](#async-infrastructure-v761)
  - [API Server & Telemetry (v77.0)](#api-server--telemetry-v770)
  - [Model Serving & Hot Reload (v77.1)](#model-serving--hot-reload-v771)
  - [Trinity Orchestrator (v75.0)](#trinity-orchestrator-v750)
- [Integration Architecture](#integration-architecture)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Version History](#version-history)
- [Links](#links)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REACTOR CORE v77.1                            │
│                    (AGI OS Nervous System)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   UNIFIED API SERVER (v77.0)                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │   │
│  │  │ Telemetry   │  │  Night      │  │  Model               │  │   │
│  │  │ Collector   │  │  Scheduler  │  │  Registry            │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                HOT-RELOAD MODEL SERVER (v77.1)                │   │
│  │  • Multi-backend support (vLLM, llama.cpp, MLX, Transformers)│   │
│  │  • Zero-downtime model swaps                                  │   │
│  │  • LRU cache + semantic response caching                      │   │
│  │  • Priority request queue                                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │            ADVANCED TRAINING ENGINE (v76.0)                   │   │
│  │                                                                │   │
│  │   Experience Buffer → Data Selector → Training Router         │   │
│  │                               │                                │   │
│  │       ┌───────────────────────┼───────────────────────┐        │   │
│  │       │                       │                       │        │   │
│  │       ▼                       ▼                       ▼        │   │
│  │   DPO Trainer          RLHF Pipeline        Constitutional AI │   │
│  │   • Preference         • PPO Algorithm       • Self-supervised│   │
│  │     Learning           • Reward Modeling     • Safety         │   │
│  │   • Memory Efficient   • Value Functions     • Alignment      │   │
│  │                                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │           ASYNC INFRASTRUCTURE (v76.1)                        │   │
│  │  • CircuitBreaker    • Backpressure    • DeadLetterQueue     │   │
│  │  • Bulkhead          • HealthMonitor   • AdaptiveRateLimiter  │   │
│  │  • TimeoutPolicy     • MetricsCollector • AsyncRetry          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              TRINITY ORCHESTRATOR (v75.0)                     │   │
│  │  • Multi-repo heartbeat monitoring                            │   │
│  │  • Command routing with load balancing                        │   │
│  │  • State reconciliation                                       │   │
│  │  • Dead Letter Queue for failed commands                      │   │
│  │  • Atomic file I/O (v73.0)                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                EVENT STREAMING (v10.3)                        │   │
│  │  • WebSocket real-time events                                 │   │
│  │  • Redis pub/sub (optional)                                   │   │
│  │  • Safety audit trail                                         │   │
│  │  • Cost tracking & budget alerts                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│         ▼                       ▼                      ▼             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      │
│  │  MLForge C++ │      │  Cloud SQL   │      │ GCP Storage  │      │
│  │   (Optional) │      │  (Events DB) │      │(Checkpoints) │      │
│  └──────────────┘      └──────────────┘      └──────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
JARVIS-Reactor/
├── reactor_core/
│   ├── training/              # Advanced training methods
│   │   ├── advanced_training.py   # DPO, RLHF, Constitutional AI (2,899 lines)
│   │   ├── unified_pipeline.py    # End-to-end training orchestration
│   │   ├── trainer.py             # Base trainer class
│   │   └── lora.py                # LoRA/QLoRA implementations
│   │
│   ├── serving/               # Model serving infrastructure
│   │   ├── model_server.py        # Hot-reload model server (1,545 lines)
│   │   └── inference_engine.py    # Multi-backend inference (1,891 lines)
│   │
│   ├── api/                   # REST API server
│   │   ├── server.py              # FastAPI endpoints (2,252 lines)
│   │   ├── telemetry.py           # Metrics & observability (1,128 lines)
│   │   ├── scheduler.py           # Night Shift scheduler (1,030 lines)
│   │   ├── model_registry.py      # Model versioning (1,301 lines)
│   │   └── health_aggregator.py   # Health monitoring (999 lines)
│   │
│   ├── orchestration/         # Trinity coordination
│   │   └── trinity_orchestrator.py # Multi-repo orchestrator
│   │
│   ├── utils/                 # Core utilities
│   │   ├── async_helpers.py       # Async patterns (1,746 lines)
│   │   └── dependencies.py        # Dependency injection (913 lines)
│   │
│   ├── integration/           # Cross-repo integration
│   │   ├── event_bridge.py        # Event streaming
│   │   ├── cost_bridge.py         # Cost tracking
│   │   ├── jarvis_connector.py    # JARVIS integration
│   │   └── prime_connector.py     # Prime integration
│   │
│   ├── eval/                  # Model evaluation
│   │   └── advanced_evaluation.py # Comprehensive eval suite (1,536 lines)
│   │
│   ├── data/                  # Data loading & preprocessing
│   ├── gcp/                   # GCP Spot VM support
│   └── config/                # Configuration management
│
├── run_supervisor.py          # AGI OS unified supervisor (1,635 lines)
├── mlforge/                   # C++ ML core (submodule)
├── docker/                    # Docker configurations
├── scripts/                   # Utility scripts
└── tests/                     # Test suite

Total: ~18,996+ lines of production code added in v75.0-v77.1
```

---

## ⭐ Key Features

### 🧠 Advanced Training Methods (v76.0)

- **DPO (Direct Preference Optimization)**: Preference learning without reward models
- **RLHF (Reinforcement Learning from Human Feedback)**: Full PPO pipeline
- **Constitutional AI**: Self-supervised safety alignment
- **Curriculum Learning**: Progressive difficulty scheduling
- **Memory Management**: Dynamic batch sizing, gradient checkpointing, CPU offloading
- **FSDP Support**: Fully Sharded Data Parallel for large models
- **Experience Replay**: Priority-based sampling from interaction logs

### ⚡ Async Infrastructure (v76.1)

- **CircuitBreaker**: Automatic failure detection and recovery
- **Backpressure**: Adaptive load management with queue shedding
- **Bulkhead**: Failure isolation between components
- **DeadLetterQueue**: Failed operation tracking and replay
- **HealthMonitor**: Real-time component health tracking
- **AdaptiveRateLimiter**: Dynamic rate limiting based on success rates
- **TimeoutPolicy**: Configurable timeouts with fallback strategies
- **MetricsCollector**: Comprehensive observability

### 🌐 API Server & Telemetry (v77.0)

- **FastAPI Server**: Production-grade REST API with auto-docs
- **Telemetry Collector**: Real-time metrics ingestion with WebSocket streaming
- **Night Shift Scheduler**: Automated training during off-peak hours
- **Model Registry**: Version management, A/B testing, rollback support
- **Health Aggregator**: Multi-service health dashboard
- **Cost Tracking**: Budget alerts and spend analytics
- **WebSocket Events**: Real-time training progress streaming

### 🔥 Model Serving & Hot Reload (v77.1)

- **Hot-Reload**: Zero-downtime model updates via file watcher
- **Multi-Backend Support**: vLLM, llama.cpp, MLX, Transformers
- **LRU Model Cache**: Memory-aware model eviction
- **Priority Queue**: Request prioritization for SLA compliance
- **Semantic Caching**: Hash-based response deduplication
- **Circuit Breaker**: Backend failure protection
- **Async Loading**: Non-blocking model initialization
- **Version Management**: Seamless model version switching

### 🎯 Trinity Orchestrator (v75.0)

- **Multi-Repo Coordination**: Heartbeat monitoring across JARVIS, Prime, Reactor
- **Command Routing**: Intelligent load balancing with priority queues
- **State Reconciliation**: Consistent state across distributed system
- **Dead Letter Queue**: Failed command tracking and retry
- **Atomic File I/O**: Zero-corruption file operations (v73.0)
- **Self-Heartbeat**: Liveness monitoring (v72.0)
- **Circuit Breakers**: Fault tolerance with automatic recovery

### 🔄 Event Streaming (v10.3)

- **WebSocket Streaming**: Real-time event broadcasting
- **Redis Pub/Sub**: Optional Redis backend for scale
- **Event Deduplication**: Hash-based duplicate prevention
- **Priority System**: Safety-critical event prioritization
- **Safety Audit Trail**: Comprehensive action logging
- **Cost Events**: Budget tracking with alerts
- **Multi-Transport**: WebSocket, file-watching, Redis

### ☁️ GCP Integration

- **Spot VM Resilience**: Auto-resume from preemption
- **Cloud SQL Storage**: Event and metric persistence
- **GCS Checkpointing**: Distributed checkpoint storage
- **Auto-Detection**: M1 local vs GCP remote environment detection

---

## 📦 Installation

### Quick Install (Python only, no C++ bindings)

```bash
pip install jarvis-reactor
```

### Build from Source (with MLForge C++ bindings)

```bash
# Clone with submodules
git clone --recursive https://github.com/drussell23/JARVIS-Reactor.git
cd JARVIS-Reactor

# Install dependencies (requires CMake and pybind11)
pip install pybind11 cmake

# Build and install
pip install -e .
```

### Environment-Specific Installation

```bash
# For local development (M1 Mac)
pip install jarvis-reactor[local]

# For GCP training (32GB+ VM)
pip install jarvis-reactor[gcp]

# For full development (includes testing, linting, docs)
pip install -e ".[dev]"
```

### Docker Installation

```bash
# Build Docker image
docker-compose build

# Run API server
docker-compose up api

# Run model server
docker-compose up model-server

# Run unified supervisor
docker-compose up supervisor
```

---

## 🚀 Quick Start

### Basic Training

```python
from reactor_core import Trainer, TrainingConfig
from reactor_core.gcp import SpotVMCheckpointer

# Configure training
config = TrainingConfig(
    model_name="llama-2-7b",
    use_lora=True,
    lora_rank=16,
    num_epochs=3,
    batch_size=4,
    gradient_checkpointing=True,
)

# Auto-detect environment (M1 local vs GCP remote)
trainer = Trainer(config)

# Train with auto-resume on Spot VM preemption
trainer.train("./data/train.jsonl")
```

### Advanced Training with DPO

```python
from reactor_core.training.advanced_training import (
    DPOTrainer,
    DPOConfig,
    PreferenceDataset,
)

# Configure DPO
dpo_config = DPOConfig(
    model_name="llama-2-7b",
    beta=0.1,  # KL divergence penalty
    learning_rate=5e-7,
    max_length=512,
    batch_size=4,
)

# Initialize DPO trainer
dpo_trainer = DPOTrainer(dpo_config)

# Train on preference pairs
await dpo_trainer.train(
    preference_dataset=PreferenceDataset(
        chosen_responses=chosen_data,
        rejected_responses=rejected_data,
    ),
    num_epochs=3,
)
```

### Model Serving with Hot Reload

```python
from reactor_core.serving.model_server import ModelServer, ModelServerConfig

# Configure model server
config = ModelServerConfig(
    models_dir="/path/to/models",
    enable_hot_reload=True,
    backend="vllm",  # or "transformers", "llamacpp", "mlx"
    max_cached_models=3,
)

# Initialize server
server = ModelServer(config)
await server.start()

# Serve inference requests
response = await server.predict(
    prompt="What is machine learning?",
    model_id="llama-2-7b",
    max_tokens=256,
)
print(response.text)

# Hot-reload: Just update the model file, server auto-reloads!
```

### API Server & Scheduler

```bash
# Start API server
uvicorn reactor_core.api.server:app --host 0.0.0.0 --port 8003 --reload
```

```python
import requests

# Trigger training via API
response = requests.post(
    "http://localhost:8003/training/trigger",
    json={
        "model_name": "llama-2-7b",
        "training_type": "dpo",
        "config": {
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 5e-7,
        },
    },
)

# Schedule nightly training
response = requests.post(
    "http://localhost:8003/scheduler/schedule",
    json={
        "name": "nightly_dpo_training",
        "schedule_type": "cron",
        "cron_expression": "0 2 * * *",  # 2 AM daily
        "job_config": {
            "training_type": "dpo",
            "model_name": "llama-2-7b",
        },
    },
)
```

### Trinity Orchestrator (Multi-Repo Coordination)

```python
from reactor_core.orchestration.trinity_orchestrator import (
    initialize_orchestrator,
    get_orchestrator,
)

# Initialize orchestrator
orchestrator = await initialize_orchestrator()

# Dispatch command to JARVIS/Prime
await orchestrator.dispatch_command(
    intent="start_surveillance",
    payload={
        "app_name": "Chrome",
        "trigger_text": "bouncing ball",
    },
    target_components=["jarvis"],
)

# Check component health
health = await orchestrator.get_health_status()
print(f"JARVIS: {health['jarvis'].status}")
print(f"Prime: {health['prime'].status}")
print(f"Reactor: {health['reactor'].status}")
```

### Unified Supervisor (One-Command Startup)

```bash
# Start entire AGI OS ecosystem
python3 run_supervisor.py

# With specific components
python3 run_supervisor.py --components jarvis,prime,reactor

# Development mode (verbose logging)
python3 run_supervisor.py --dev --log-level DEBUG
```

---

## 🔬 Advanced Features

### Advanced Training Methods (v76.0)

Comprehensive documentation for DPO, RLHF, Constitutional AI, Curriculum Learning with code examples for memory management, experience replay, and multi-GPU training.

### Async Infrastructure (v76.1)

Production-ready async patterns including circuit breakers, backpressure management, dead letter queues, health monitoring, and adaptive rate limiting.

### API Server & Telemetry (v77.0)

FastAPI server with telemetry collection, Night Shift scheduling, model registry, health aggregation, and real-time WebSocket streaming.

### Model Serving & Hot Reload (v77.1)

Zero-downtime model serving with hot-reload, multi-backend support (vLLM, llama.cpp, MLX, Transformers), LRU caching, and semantic response caching.

### Trinity Orchestrator (v75.0)

Multi-repo coordination with heartbeat monitoring, command routing, state reconciliation, dead letter queue, and atomic file I/O.

*(See full documentation in sections below)*

---

## 🔗 Integration Architecture

### JARVIS Ecosystem Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                       JARVIS AGI ECOSYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐          ┌──────────────────┐                 │
│  │  JARVIS-AI-Agent │◄────────►│  JARVIS Prime    │                 │
│  │  (Claude Body)   │  Events  │  (LLM Mind)      │                 │
│  │                  │          │                  │                 │
│  │ • Computer Use   │          │ • Local LLM      │                 │
│  │ • macOS Control  │          │ • Reasoning      │                 │
│  │ • Voice Auth     │          │ • Context        │                 │
│  └─────────┬────────┘          └─────────┬────────┘                 │
│            │                              │                          │
│            │         Event Bridge         │                          │
│            │      (WebSocket/Redis)       │                          │
│            │                              │                          │
│  ┌─────────▼──────────────────────────────▼────────┐                 │
│  │            Reactor Core (Nervous System)        │                 │
│  │  ┌──────────────────────────────────────────┐   │                 │
│  │  │         Trinity Orchestrator             │   │                 │
│  │  │  • Heartbeat monitoring                  │   │                 │
│  │  │  • Command routing                       │   │                 │
│  │  │  • State reconciliation                  │   │                 │
│  │  └──────────────────────────────────────────┘   │                 │
│  │                                                  │                 │
│  │  ┌──────────────────────────────────────────┐   │                 │
│  │  │         Training & Serving               │   │                 │
│  │  │  • DPO, RLHF, Constitutional AI          │   │                 │
│  │  │  • Hot-reload model server               │   │                 │
│  │  │  • Night Shift scheduler                 │   │                 │
│  │  └──────────────────────────────────────────┘   │                 │
│  │                                                  │                 │
│  │  ┌──────────────────────────────────────────┐   │                 │
│  │  │         Event Streaming                  │   │                 │
│  │  │  • Safety audit trail                    │   │                 │
│  │  │  • Cost tracking                         │   │                 │
│  │  │  • Telemetry collection                  │   │                 │
│  │  └──────────────────────────────────────────┘   │                 │
│  └──────────────────────────────────────────────────┘                 │
│                                                                      │
│            ▼                             ▼                           │
│  ┌──────────────────┐         ┌──────────────────┐                  │
│  │   Cloud SQL      │         │   GCP Storage    │                  │
│  │   (Events DB)    │         │  (Checkpoints)   │                  │
│  └──────────────────┘         └──────────────────┘                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Version History

### **v77.1** - Model Serving & Hot Reload (2025-01-07)
- Hot-reload model server with zero-downtime updates (1,545 lines)
- Multi-backend inference engine: vLLM, llama.cpp, MLX, Transformers (1,891 lines)
- Unified supervisor for one-command AGI OS startup (1,635 lines)
- LRU model cache with memory-aware eviction
- Priority request queue for SLA compliance
- Semantic response caching with hash-based deduplication

### **v77.0** - Advanced API Server (2025-01-07)
- Telemetry collection system with WebSocket streaming (1,128 lines)
- Night Shift scheduler for automated training (1,030 lines)
- Model registry with versioning and A/B testing (1,301 lines)
- Health aggregator with multi-service dashboard (999 lines)
- Enhanced FastAPI server (2,252 lines)

### **v76.1** - Async Infrastructure (2025-01-07)
- Advanced async patterns library (1,746 lines)
- Circuit breaker, backpressure, bulkhead patterns
- Dead letter queue, health monitor, adaptive rate limiter
- Dependency injection system (913 lines)

### **v76.0** - Advanced Training Methods (2025-01-07)
- DPO, RLHF, Constitutional AI, Curriculum Learning (2,899 lines)
- Memory manager with dynamic batch sizing
- Advanced evaluation suite (1,536 lines)

### **v75.0** - Trinity Dead Letter Queue (2024-12-25)
- DLQ for failed/expired commands
- Automatic retry with exponential backoff

### **v73.0** - Atomic File I/O (2024-11-15)
- Zero-corruption file operations via atomic renames

### **v10.3** - Vision Safety Integration (2024-10-20)
- Safety audit trail and kill switch mechanism

### **v10.0** - Cross-Repository Integration (2024-10-01)
- Real-time event streaming across JARVIS ecosystem

### **v1.0.0** - Initial Release (2024-09-01)
- PyTorch-first ML training framework
- LoRA/QLoRA, DPO, FSDP support
- GCP Spot VM resilience

---

## 🔗 Links

- **GitHub**: https://github.com/drussell23/JARVIS-Reactor
- **MLForge C++ Core**: https://github.com/drussell23/MLForge
- **JARVIS-AI-Agent**: https://github.com/drussell23/JARVIS-AI-Agent
- **JARVIS Prime**: https://github.com/drussell23/jarvis-prime

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the JARVIS AGI Ecosystem**
