# JARVIS Reactor (Reactor-Core)

**The Nerves of the AGI OS — training, fine-tuning, experience collection, and model deployment**

JARVIS Reactor (Reactor-Core) is the **training and learning layer** of the JARVIS AGI ecosystem. It provides ML training (DPO, RLHF, curriculum, meta-learning, world models, causal reasoning), model serving with hot-reload, experience collection from JARVIS Body, model deployment to JARVIS-Prime, and **Trinity Protocol** integration for cross-repo coordination. It is started either **standalone** (`run_reactor.py`) or by the **unified supervisor** in JARVIS (`python3 unified_supervisor.py`).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Version](https://img.shields.io/badge/version-2.10.0-blue.svg)](https://github.com/drussell23/JARVIS-Reactor)

---

## What is JARVIS Reactor? (Trinity Role)

| Role | Repository | Responsibility |
|------|------------|----------------|
| **Body** | [JARVIS (JARVIS-AI-Agent)](https://github.com/drussell23/JARVIS-AI-Agent) | macOS integration, computer use, unified supervisor, voice/vision |
| **Mind** | [JARVIS-Prime](https://github.com/drussell23/jarvis-prime) | LLM inference, Neural Orchestrator Core, OpenAI-compatible API |
| **Nerves** | **Reactor-Core (this repo)** | Training, fine-tuning, experience collection, model deployment, Trinity coordination |

Reactor-Core is the **nervous system**: it trains and improves models, collects experience from JARVIS, and deploys models to JARVIS-Prime. The **unified supervisor** in JARVIS discovers and starts Reactor-Core (default port **8090**) alongside JARVIS-Prime (8000) and the JARVIS backend (8010).

---

## 🚀 What is JARVIS Reactor? (Features)

JARVIS Reactor is a production-grade ML infrastructure combining:

- **Advanced Training Methods**: DPO, RLHF, Constitutional AI, Curriculum Learning, Meta-Learning, World Models, Causal Reasoning
- **Model Serving**: Hot-reload model server with multi-backend support (vLLM, llama.cpp, MLX, Transformers)
- **Async Infrastructure**: Circuit breakers, backpressure, bulkheads, dead letter queues, structured concurrency
- **API Platform**: FastAPI server with telemetry, scheduling, model registry, health monitoring
- **Trinity Orchestration**: Multi-repo coordination with heartbeat monitoring and state sync
- **Event Streaming**: Real-time WebSocket/Redis pub-sub across JARVIS ecosystem
- **GCP Integration**: Spot VM resilience, Cloud SQL storage, auto-checkpointing
- **MLForge C++ Core**: High-performance ML primitives (optional submodule)
- **Unified Supervisor**: One-command startup for entire AGI OS ecosystem (`python3 run_supervisor.py`)
- **Connection Pooling**: Efficient HTTP/Redis connection management with automatic lifecycle
- **Dynamic Configuration**: Zero hardcoding, XDG-compliant paths, environment-driven config
- **Structured Concurrency**: Python 3.11+ TaskGroup patterns for robust async operations

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Unified Supervisor (One-Command Startup)](#unified-supervisor-one-command-startup)
- [Advanced Features](#advanced-features)
  - [Advanced Training Methods (v76.0-v80.0)](#advanced-training-methods-v760-v800)
  - [Async Infrastructure (v76.1, v92.0)](#async-infrastructure-v761-v920)
  - [API Server & Telemetry (v77.0)](#api-server--telemetry-v770)
  - [Model Serving & Hot Reload (v77.1)](#model-serving--hot-reload-v771)
  - [Trinity Orchestrator (v75.0)](#trinity-orchestrator-v750)
  - [Online Learning & Data Versioning (v91.0)](#online-learning--data-versioning-v910)
  - [Distributed Training (v91.0)](#distributed-training-v910)
- [Integration Architecture](#integration-architecture)
- [Configuration Guide](#configuration-guide)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Development Guide](#development-guide)
- [Version History](#version-history)
- [Roadmap](#roadmap--next-phases)
  - [v242.0 — Training Data Pipeline Activation](#v2420--training-data-pipeline-activation-planned)
  - [v243.0 — Ouroboros Training Support](#v2430--ouroboros-training-support-planned)
  - [v244.0 — Continuous Learning Loop](#v2440--continuous-learning-loop-planned)
  - [v245.0 — Distributed Training on GCP](#v2450--distributed-training-on-gcp-planned)
- [Links](#links)

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AGI OS UNIFIED SUPERVISOR v92.0                   │
│                    (Central Coordination Hub)                       │
│                    python3 run_supervisor.py                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
    ┌─────────┐       ┌─────────────┐       ┌─────────┐
    │ JARVIS  │◄─────►│   TRINITY   │◄─────►│ J-PRIME │
    │ (Body)  │Events │ ORCHESTRATOR│Events │ (Mind)  │
    │         │       │             │       │         │
    │ macOS   │       │ Heartbeats  │       │ LLM     │
    │ Actions │       │ Commands    │       │ Inference
    └─────────┘       │ State Sync  │       └─────────┘
                      └─────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │REACTOR CORE │  │  ONLINE     │  │ DISTRIBUTED │
      │  (Nerves)   │  │  LEARNING   │  │  TRAINING   │
      │             │  │             │  │             │
      │ Training    │  │ Experience  │  │ Multi-VM    │
      │ Learning    │  │ Replay      │  │ Gradient    │
      │ Serving     │  │ EWC/Drift   │  │ Sync        │
      └─────────────┘  └─────────────┘  └─────────────┘
```

### Reactor Core Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REACTOR CORE v2.10.0                         │
│                    (AGI OS Nervous System)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              UNIFIED API SERVER (v77.0)                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐  │   │
│  │  │ Telemetry   │  │  Night      │  │  Model               │  │   │
│  │  │ Collector   │  │  Scheduler  │  │  Registry            │  │   │
│  │  │ + WebSocket │  │ + Cron Jobs │  │ + A/B Testing         │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐                             │   │
│  │  │ Health      │  │ Cost       │                             │   │
│  │  │ Aggregator  │  │ Tracker    │                             │   │
│  │  └─────────────┘  └─────────────┘                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         HOT-RELOAD MODEL SERVER (v77.1)                      │   │
│  │  • Multi-backend: vLLM, llama.cpp, MLX, Transformers        │   │
│  │  • Zero-downtime model swaps via file watcher                │   │
│  │  • LRU cache with memory-aware eviction                      │   │
│  │  • Priority request queue for SLA compliance                 │   │
│  │  • Semantic response caching (hash-based deduplication)      │   │
│  │  • Circuit breaker for backend failure protection            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │      ADVANCED TRAINING ENGINE (v76.0-v80.0)                   │   │
│  │                                                                │   │
│  │   Experience Buffer → Data Selector → Training Router         │   │
│  │                               │                                │   │
│  │   ┌───────────────────────────┼───────────────────────────┐   │   │
│  │   │                           │                           │   │
│  │   ▼                           ▼                           ▼   │   │
│  │   DPO Trainer          RLHF Pipeline        Constitutional AI │   │
│  │   • Preference         • PPO Algorithm       • Self-supervised│   │
│  │     Learning           • Reward Modeling     • Safety         │   │
│  │   • Memory Efficient   • Value Functions     • Alignment      │   │
│  │                                                                │   │
│  │   Curriculum Learning  Meta-Learning        World Models     │   │
│  │   • Progressive        • MAML/Reptile       • Latent dynamics │   │
│  │     difficulty         • Few-shot learning   • Planning        │   │
│  │   • Adaptive           • Task adaptation     • Counterfactual │   │
│  │     scheduling                                 reasoning       │   │
│  │                                                                │   │
│  │   Causal Reasoning    FSDP Training        Federated Learning│   │
│  │   • SCMs              • Multi-GPU/Node      • Cross-repo      │   │
│  │   • Do-calculus       • Gradient sharding   • Byzantine-robust│   │
│  │   • Causal discovery  • Memory efficient    • Privacy-preserving│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │    ASYNC INFRASTRUCTURE (v76.1, v92.0)                        │   │
│  │  • CircuitBreaker      • Backpressure      • DeadLetterQueue │   │
│  │  • Bulkhead            • HealthMonitor     • AdaptiveRateLimiter│   │
│  │  • TimeoutPolicy       • MetricsCollector  • AsyncRetry       │   │
│  │  • StructuredTaskGroup • ConnectionPool     • AsyncBarrier     │   │
│  │  • AsyncContextGroup   • AsyncLatch        • ScatterGather    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         TRINITY ORCHESTRATOR (v75.0)                           │   │
│  │  • Multi-repo heartbeat monitoring (JARVIS, Prime, Reactor)  │   │
│  │  • Command routing with intelligent load balancing            │   │
│  │  • State reconciliation across distributed system             │   │
│  │  • Dead Letter Queue for failed commands with auto-retry      │   │
│  │  • Atomic file I/O (zero-corruption operations)              │   │
│  │  • Circuit breakers for fault tolerance                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │    ONLINE LEARNING & DATA VERSIONING (v91.0)                  │   │
│  │  • Prioritized experience replay with importance sampling     │   │
│  │  • Elastic Weight Consolidation (EWC) - prevents forgetting  │   │
│  │  • Concept Drift Detection (Page-Hinkley test)                │   │
│  │  • Data Versioning: Content-addressed storage (DVC compatible)│   │
│  │  • Lineage tracking and reproducibility                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │    DISTRIBUTED TRAINING (v91.0)                               │   │
│  │  • Multi-VM coordination with gradient compression            │   │
│  │  • GCP Spot VM checkpointing with predictive preemption       │   │
│  │  • Dynamic resource allocation with cost-aware decisions      │   │
│  │  • Gradient checksum validation                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         EVENT STREAMING (v10.3)                               │   │
│  │  • WebSocket real-time events with priority queues            │   │
│  │  • Redis pub/sub (optional) for scale                          │   │
│  │  • Safety audit trail with kill switch                        │   │
│  │  • Cost tracking & budget alerts                              │   │
│  │  • Multi-transport: WebSocket, file-watching, Redis            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│         ▼                       ▼                      ▼             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      │
│  │  MLForge C++ │      │  Cloud SQL   │      │ GCP Storage  │      │
│  │   (Optional) │      │  (Events DB) │      │(Checkpoints) │      │
│  │  pybind11    │      │  PostgreSQL  │      │  GCS Bucket  │      │
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

### Start Reactor-Core (Recommended: via JARVIS)

```bash
# From JARVIS-AI-Agent repo — starts Body + Prime + Reactor-Core
cd /path/to/JARVIS-AI-Agent
python3 unified_supervisor.py
```

Reactor-Core will start on port **8090** and register with Trinity. Health: `http://localhost:8090/health`.

### Start Reactor-Core Standalone

```bash
# From Reactor-Core repo
cd /path/to/Reactor-Core
python3 run_reactor.py --port 8090
```

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

### Entry Points

| Entry Point | Purpose | When to Use |
|-------------|---------|-------------|
| **Unified Supervisor (JARVIS)** | `python3 unified_supervisor.py` in **JARVIS-AI-Agent** | **Recommended** — starts Body + Prime + Reactor-Core with Trinity coordination; discovers Reactor via `REACTOR_CORE_REPO_PATH` or default path |
| **`run_reactor.py`** | Trinity-integrated Reactor entry point | Standalone Reactor, or when supervisor calls it (e.g. `python3 run_reactor.py --port 8090`) |
| **`run_supervisor.py`** (in this repo) | Legacy/alternative supervisor in Reactor repo | When running orchestration from the Reactor repo instead of JARVIS |

The **unified supervisor lives in JARVIS-AI-Agent**. It starts Reactor-Core by running `run_reactor.py` (or the configured script) in this repo, typically on port **8090**. Reactor exposes `/health` for supervisor health checks and Trinity state sync.

### Unified Supervisor (One-Command Startup)

The unified supervisor is in **JARVIS-AI-Agent** (`unified_supervisor.py`). It is the **single entry point** for the entire AGI OS ecosystem and automatically discovers, starts, and coordinates JARVIS (Body), JARVIS-Prime (Mind), and Reactor-Core (Nerves).

```bash
# From JARVIS-AI-Agent repo — start entire AGI OS ecosystem (recommended)
python3 unified_supervisor.py

# With options (see JARVIS-AI-Agent unified_supervisor.py for full CLI)
# python3 unified_supervisor.py --mode supervisor --skip-trinity ...
```

**What the Supervisor Does (in JARVIS-AI-Agent):**

1. **Component Discovery**: Automatically finds JARVIS, JARVIS Prime, and Reactor Core repos
2. **Health Monitoring**: Continuous health checks with automatic recovery
3. **Event Bridge**: Sets up real-time event streaming between components
4. **Trinity Orchestration**: Initializes multi-repo coordination
5. **Service Startup**: Starts all Reactor Core services (API, Model Server, Training, etc.)
6. **Experience Collection**: Continuous learning from JARVIS interactions
7. **Graceful Shutdown**: Clean shutdown of all components on Ctrl+C

**Startup Phases:**

```
Phase 1: Initialize Trinity Orchestrator
Phase 2: Initialize Event Bridge
Phase 3: Discover Components
Phase 4: Start Reactor Core Services
Phase 5: Initialize v91.0 Advanced Services
Phase 6: Start JARVIS (Body)
Phase 7: Start J-Prime (Mind)
Phase 8: Start Background Tasks
Phase 9: Wait for Component Health
```

**Output Example:**

```
======================================================================
           AGI OS UNIFIED SUPERVISOR - PROJECT TRINITY
======================================================================

[Phase 1] Initializing Trinity Orchestrator...
[OK] Trinity Orchestrator running

[Phase 2] Initializing Event Bridge...
[OK] Event Bridge running

[Phase 3] Discovering components...
  Found JARVIS at /path/to/JARVIS-AI-Agent
  Found J-Prime at /path/to/jarvis-prime
  Reactor Core at /path/to/reactor-core

[Phase 4] Starting Reactor Core services...
  [OK] Telemetry Collector started
  [OK] Model Registry initialized (5 models)
  [OK] Health Aggregator started
  [OK] Scheduler started (daily/weekly training)
  [OK] Model Server started

[Phase 5] Initializing v91.0 Advanced Services...
  [OK] Online Learning Engine started
  [OK] Distributed Coordinator started
  [OK] Data Version Controller started
  [OK] Spot VM Checkpointer started

[Phase 6] Starting JARVIS (Body)...
[OK] JARVIS started (PID: 12345)

[Phase 7] Starting J-Prime (Mind)...
[OK] J-Prime started (PID: 12346)

[Phase 8] Starting background services...
[OK] Health monitoring started
[OK] Experience collection started
[OK] Event processing started

[Phase 9] Waiting for component health...
======================================================================
            AGI OS READY - All Systems Operational
======================================================================

Component Status:
  JARVIS:      ✅ Running (http://localhost:8000)
  J-Prime:     ✅ Running (http://localhost:8001)
  Reactor API: ✅ Running (http://localhost:8003)
  Model Server: ✅ Running (http://localhost:8004)

Background Services:
  Health Monitor:      ✅ Active
  Experience Collector: ✅ Active (0 experiences collected)
  Event Processor:     ✅ Active
  Trinity Experience Receiver: ✅ Active

Press Ctrl+C to shutdown gracefully...
```

---

## 🔬 Advanced Features

### Advanced Training Methods (v76.0-v80.0)

#### DPO (Direct Preference Optimization)

Train models on preference pairs without reward models:

```python
from reactor_core.training.advanced_training import DPOTrainer, DPOConfig

config = DPOConfig(
    model_name="llama-2-7b",
    beta=0.1,  # KL divergence penalty
    learning_rate=5e-7,
    max_length=512,
)

trainer = DPOTrainer(config)
await trainer.train(
    preference_dataset=PreferenceDataset(
        chosen_responses=chosen_data,
        rejected_responses=rejected_data,
    ),
    num_epochs=3,
)
```

**Variants Supported:**
- Standard DPO
- IPO (Identity Preference Optimization)
- KTO (Kahneman-Tversky Optimization)
- ORPO (Odds Ratio Preference Optimization)

#### RLHF (Reinforcement Learning from Human Feedback)

Full PPO pipeline with reward modeling:

```python
from reactor_core.training.advanced_training import RLHFTrainer, RLHFConfig

config = RLHFConfig(
    model_name="llama-2-7b",
    reward_model_name="reward-model",
    ppo_config={
        "clip_epsilon": 0.2,
        "value_coef": 0.1,
        "entropy_coef": 0.01,
    },
)

trainer = RLHFTrainer(config)
await trainer.train(
    preference_dataset=preference_data,
    num_epochs=3,
)
```

#### Curriculum Learning (v79.0)

Progressive difficulty scheduling for faster convergence:

```python
from reactor_core.training.curriculum_learning import CurriculumLearner

curriculum = CurriculumLearner(
    model=model,
    dataset=dataset,
    difficulty_metric="perplexity",
    progression_strategy="exponential",  # or "linear", "adaptive"
)

# Automatic difficulty progression
await curriculum.train(num_epochs=10)
```

**Benefits:** 30-50% faster convergence, better generalization

#### Meta-Learning (v79.0)

Few-shot learning with MAML, Reptile, Meta-SGD:

```python
from reactor_core.training.meta_learning import MAMLTrainer

maml = MAMLTrainer(
    model=model,
    inner_lr=0.01,
    outer_lr=0.001,
    adaptation_steps=5,
)

# Learn to learn from few examples
await maml.meta_train(
    tasks=task_distribution,
    meta_batch_size=4,
    num_meta_iterations=1000,
)
```

#### World Model Training (v80.0)

Learn latent dynamics for planning and counterfactual reasoning:

```python
from reactor_core.training.world_model_training import WorldModelTrainer

world_model = WorldModelTrainer(
    latent_dim=512,
    action_dim=128,
    reward_dim=1,
)

await world_model.train(
    trajectories=trajectory_data,
    num_epochs=100,
)

# Counterfactual reasoning: "What if I had done X?"
counterfactual = await world_model.imagine_rollout(
    initial_state=state,
    alternative_action=action,
    horizon=10,
)
```

#### Causal Reasoning (v80.0)

Understand cause-effect relationships:

```python
from reactor_core.training.causal_reasoning import CausalReasoner

reasoner = CausalReasoner(
    model=model,
    causal_graph=graph,
)

# Do-calculus: P(Y | do(X))
interventional_prob = await reasoner.interventional_inference(
    intervention={"X": value},
    query="Y",
)

# Causal discovery
discovered_graph = await reasoner.discover_causality(data)
```

### Async Infrastructure (v76.1, v92.0)

#### Structured Concurrency (v92.0+)

Python 3.11+ compatible structured concurrency with TaskGroup:

```python
from reactor_core.utils.async_helpers import StructuredTaskGroup, run_in_task_group

# Structured task group with automatic error handling
async with StructuredTaskGroup(
    name="training_pipeline",
    max_concurrent=5,
    cancel_on_error=True,
    timeout_seconds=3600.0,
) as tg:
    tg.create_task(load_data(), name="data_loading")
    tg.create_task(preprocess_data(), name="preprocessing")
    tg.create_task(train_model(), name="training")
    tg.create_task(validate_model(), name="validation")

# Get results
results = tg.results
for result in results:
    if result.success:
        print(f"{result.name}: {result.result}")
    else:
        print(f"{result.name}: {result.exception}")

# Convenience function
results = await run_in_task_group(
    [fetch_url(url) for url in urls],
    names=[f"fetch_{i}" for i in range(len(urls))],
    max_concurrent=10,
)
```

#### Connection Pooling (v92.0+)

Efficient HTTP and Redis connection management:

```python
from reactor_core.config.unified_config import (
    HTTPConnectionPool,
    RedisConnectionPool,
    ConnectionPoolConfig,
)

# HTTP connection pool
pool = await HTTPConnectionPool.get_instance("api_client")
async with pool.request("GET", "https://api.example.com/data") as response:
    data = await response.json()

# Redis connection pool
redis_pool = await RedisConnectionPool.get_instance()
client = await redis_pool.get_client(host="localhost", port=6379)
await client.set("key", "value")
```

**Features:**
- Singleton pattern per configuration
- Automatic session lifecycle management
- Connection reuse with keepalive
- Configurable pool sizes via environment variables

#### Circuit Breaker

Automatic failure detection and recovery:

```python
from reactor_core.utils.async_helpers import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_max_calls=3,
)

@breaker.protect
async def risky_operation():
    # This will be protected by circuit breaker
    return await external_api_call()

# Circuit states: CLOSED → OPEN → HALF_OPEN → CLOSED
```

#### Backpressure Control

Prevents memory exhaustion under high load:

```python
from reactor_core.utils.async_helpers import BackpressureController

controller = BackpressureController(
    max_queue_size=1000,
    queue_full_strategy="reject",  # or "block", "drop_oldest"
)

async def process_item(item):
    await controller.acquire()
    try:
        await process(item)
    finally:
        controller.release()
```

#### Dead Letter Queue

Failed operation tracking and automatic retry:

```python
from reactor_core.utils.async_helpers import DeadLetterQueue

dlq = DeadLetterQueue(
    name="training_operations",
    persist_path=Path("/tmp/dlq"),
    auto_retry_interval=300.0,  # Retry every 5 minutes
)

# Register operation for retry
dlq.register_operation("publish_model_ready", publish_model_ready_func)

# Add failed operation
await dlq.add(
    operation="publish_model_ready",
    args=(model_name, model_path),
    kwargs={},
    exception=exception,
)

# Automatic retry with exponential backoff
```

### Online Learning & Data Versioning (v91.0)

#### Prioritized Experience Replay

Learn continuously from JARVIS interactions:

```python
from reactor_core.training.online_learning import OnlineLearningEngine

engine = OnlineLearningEngine(
    buffer_size=100000,
    importance_sampling=True,
    ewc_lambda=100.0,  # Elastic Weight Consolidation
)

# Add experiences from JARVIS
await engine.add_experience({
    "user_input": "Hello",
    "assistant_output": "Hi there!",
    "feedback": "positive",
})

# Trigger incremental update
await engine.incremental_update(
    model=model,
    batch_size=32,
    num_steps=100,
)
```

#### Concept Drift Detection

Automatic model adaptation when data distribution changes:

```python
from reactor_core.training.online_learning import DriftDetector

detector = DriftDetector(
    threshold=0.1,
    window_size=1000,
    test_type="page_hinkley",  # or "adwin", "kswin"
)

# Monitor for drift
drift_detected = await detector.check_drift(
    current_batch=recent_data,
    reference_batch=historical_data,
)

if drift_detected:
    # Trigger model retraining
    await retrain_model()
```

#### Data Versioning

Content-addressed storage with lineage tracking:

```python
from reactor_core.data.versioning import DataVersionController

controller = DataVersionController(
    version_store_path=Path("/data/versions"),
)

# Version a dataset
version = await controller.create_version(
    dataset_path=Path("/data/train.jsonl"),
    metadata={"source": "jarvis_interactions", "date": "2025-01-15"},
)

# Get version lineage
lineage = await controller.get_lineage(version.id)
print(f"Version {version.id} derived from {lineage.parent_id}")

# Reproduce exact dataset
dataset = await controller.load_version(version.id)
```

### Distributed Training (v91.0)

#### Multi-VM Coordination

Train across multiple GCP Spot VMs with gradient compression:

```python
from reactor_core.training.distributed_coordinator import DistributedCoordinator

coordinator = DistributedCoordinator(
    num_workers=8,
    gradient_compression="fp16",  # or "int8", "sparse"
    checkpoint_interval=300,  # seconds
)

# Start distributed training
await coordinator.start_training(
    model=model,
    dataset=dataset,
    num_epochs=10,
)

# Automatic checkpoint/resume on VM preemption
```

#### GCP Spot VM Checkpointing

Predictive preemption detection and automatic resume:

```python
from reactor_core.gcp.checkpointer import SpotVMCheckpointer

checkpointer = SpotVMCheckpointer(
    gcs_bucket="my-checkpoints",
    checkpoint_interval=300,
    enable_preemption_prediction=True,
)

# Automatic checkpointing during training
async with checkpointer.protect_training():
    await train_model()

# Resume from latest checkpoint
await checkpointer.resume_training()
```

**Preemption Signals Monitored:**
- GCP metadata API warnings
- System load spikes
- Network latency increases
- Memory pressure indicators

---

## Cross-Repo Integration (Trinity)

Reactor-Core is the **Nerves** in the three-repo Trinity architecture. It is **started and monitored** by the JARVIS unified supervisor and **coordinates with JARVIS-Prime** for inference and model deployment.

**How JARVIS (Body) uses Reactor-Core:**

- **Discovery:** Supervisor resolves `REACTOR_CORE_REPO_PATH` (or default `~/Documents/repos/Reactor-Core`).
- **Startup:** Supervisor runs `run_reactor.py` (or configured script) with port **8090**; Reactor starts HTTP server and health endpoint.
- **Health:** Supervisor polls `GET /health` on port 8090; Reactor reports training readiness and Trinity connection state.
- **State:** Reactor reads/writes shared state under `~/.jarvis/` (e.g. Trinity state, experience queue) for coordination.

**How Reactor-Core uses JARVIS-Prime:**

- **Inference:** Reactor can call Prime’s OpenAI-compatible API for generation during training, evaluation, or distillation.
- **Model deployment:** Trained/updated models can be deployed to Prime (e.g. hot swap, model registry).
- **Trinity Protocol:** Events and heartbeats flow via file IPC and/or WebSocket; Reactor participates in Trinity state sync and experience collection from JARVIS Body.
- **DPO Training from Telemetry (v238.0+):** JARVIS Body's `TelemetryEmitter` captures every interaction — query, complexity classification, response, latency, and source. Reactor-Core uses this telemetry to build DPO preference pairs (e.g., chosen: `"10"`, rejected: `"Of course, the sum of five and five is ten..."`) for fine-tuning Mistral-7B. This training loop makes the v236.0/v238.0 adaptive prompt system's conciseness enforcement **permanent** — encoding terse-vs-detailed behavior in the model's weights instead of relying on prompt instructions. See the JARVIS-Prime README for the full training loop architecture.

**run_reactor.py:**

- **Trinity-integrated entry point** for Reactor-Core. Designed to be started by the unified supervisor (`python3 run_reactor.py --port 8090`).
- Exposes **health** (`/health`) for supervisor monitoring and **training/API** endpoints for the ecosystem.
- Environment: `REACTOR_PORT` (default 8090), `JARVIS_PRIME_URL`, `TRINITY_ENABLED`, `MODEL_OUTPUT_DIR`, `LOG_LEVEL`.

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

## ⚙️ Configuration Guide

### Environment Variables

JARVIS Reactor uses environment variables for all configuration (zero hardcoding):

```bash
# Path Configuration (XDG-compliant defaults)
export JARVIS_EVENTS_DIR="/custom/path/events"
export TRINITY_EVENTS_DIR="/custom/path/trinity/events"
export EXPERIENCE_QUEUE_DIR="/custom/path/experience_queue"
export MODEL_REGISTRY_PATH="/custom/path/models"
export DATA_VERSION_PATH="/custom/path/data_versions"

# API Configuration
export AGI_API_PORT=8003
export AGI_SERVING_PORT=8001
export AGI_JPRIME_PORT=8000

# Connection Pooling
export HTTP_POOL_SIZE=100
export HTTP_POOL_PER_HOST=10
export HTTP_KEEPALIVE_TIMEOUT=30.0
export REDIS_POOL_SIZE=10

# Training Configuration
export REACTOR_EXPERIENCE_BUFFER_THRESHOLD=100
export REACTOR_AUTO_TRAINING_THRESHOLD=1000
export REACTOR_CHECKPOINT_INTERVAL=300

# GCP Configuration
export GCP_PROJECT_ID="my-project"
export GCP_CHECKPOINT_BUCKET="my-checkpoints"
export GCP_SPOT_VM_ENABLED=true

# Feature Flags
export REACTOR_ENABLE_ONLINE_LEARNING=true
export REACTOR_ENABLE_DISTRIBUTED_TRAINING=true
export REACTOR_ENABLE_DATA_VERSIONING=true
```

### Configuration Files

Configuration is loaded in this priority order:
1. Environment variables (highest priority)
2. `~/.jarvis/reactor/config.json` (user config)
3. `reactor_core/config/default_config.json` (defaults)

Example config file:

```json
{
  "api": {
    "port": 8003,
    "host": "0.0.0.0"
  },
  "training": {
    "default_model": "llama-2-7b",
    "use_lora": true,
    "lora_rank": 16
  },
  "serving": {
    "max_cached_models": 5,
    "enable_hot_reload": true,
    "default_backend": "auto"
  },
  "trinity": {
    "heartbeat_interval": 5.0,
    "health_check_timeout": 10.0
  }
}
```

### Dynamic Path Resolution

All paths are resolved dynamically with XDG compliance:

1. **Environment Variable** (if set)
2. **base_config.resolve_path()** (if available)
3. **XDG_DATA_HOME/jarvis/** (fallback)

No hardcoded `Path.home()` calls - fully portable across systems.

## 🔧 Troubleshooting

### Common Issues

#### Issue: Components fail to start

**Symptoms:** `run_supervisor.py` shows component failures

**Solutions:**
```bash
# Check component paths
python3 run_supervisor.py --dev --log-level DEBUG

# Verify component health
curl http://localhost:8003/health

# Check logs
tail -f ~/.jarvis/reactor/logs/supervisor.log
```

#### Issue: Training fails with OOM

**Symptoms:** Out of memory errors during training

**Solutions:**
```python
# Enable gradient checkpointing
config = TrainingConfig(
    gradient_checkpointing=True,
    use_qlora=True,  # 4-bit quantization
    cpu_offload=True,  # Offload to CPU
)

# Use smaller batch size
config.batch_size = 1
config.gradient_accumulation_steps = 8
```

#### Issue: Model server not hot-reloading

**Symptoms:** Model updates don't appear in server

**Solutions:**
```python
# Verify file watcher is enabled
config = ModelServerConfig(
    enable_hot_reload=True,
    watch_directories=["/path/to/models"],
)

# Check file permissions
ls -la /path/to/models

# Verify model format
# Server supports: .gguf, .safetensors, .bin
```

#### Issue: Cross-repo events not working

**Symptoms:** Events not flowing between JARVIS, Prime, Reactor

**Solutions:**
```bash
# Check event bridge status
curl http://localhost:8003/api/v1/events/status

# Verify event directories exist
ls -la ~/.jarvis/events/
ls -la ~/.jarvis/trinity/events/

# Check WebSocket connection
# Open browser console: ws://localhost:8003/ws
```

#### Issue: Distributed training hangs

**Symptoms:** Training stuck at barrier synchronization

**Solutions:**
```python
# Check network connectivity
await coordinator.check_connectivity()

# Verify all workers are healthy
health = await coordinator.get_worker_health()

# Enable gradient checksum validation
coordinator.enable_gradient_verification = True
```

### Debug Mode

Enable comprehensive debugging:

```bash
# Set debug environment
export REACTOR_DEBUG=true
export REACTOR_LOG_LEVEL=DEBUG

# Run with debug flags
python3 run_supervisor.py --dev --log-level DEBUG

# Check debug logs
tail -f ~/.jarvis/reactor/logs/debug.log
```

## 🛠️ Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone --recursive https://github.com/drussell23/JARVIS-Reactor.git
cd JARVIS-Reactor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
black reactor_core/
ruff check reactor_core/
```

### Code Structure

```
reactor_core/
├── training/          # Training methods and pipelines
├── serving/           # Model serving infrastructure
├── api/              # REST API endpoints
├── orchestration/    # Trinity coordination
├── integration/      # Cross-repo integration
├── utils/            # Utilities (async_helpers, etc.)
├── config/           # Configuration management
├── data/             # Data processing and versioning
├── eval/             # Model evaluation
└── gcp/              # GCP-specific features
```

### Adding New Features

1. **Create feature branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Follow code style:**
   - Use `black` for formatting
   - Follow type hints (use `mypy`)
   - Add docstrings (Google style)

3. **Write tests:**
   ```python
   # tests/test_my_feature.py
   import pytest
   from reactor_core.my_module import MyFeature
   
   @pytest.mark.asyncio
   async def test_my_feature():
       feature = MyFeature()
       result = await feature.do_something()
       assert result is not None
   ```

4. **Update documentation:**
   - Add to README.md
   - Update API docs
   - Add examples

5. **Submit PR:**
   - Ensure all tests pass
   - Update version in `__init__.py`
   - Add to CHANGELOG.md

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_training.py

# Run with coverage
pytest --cov=reactor_core --cov-report=html

# Run integration tests
pytest tests/integration/ -v
```

### Code Quality

```bash
# Format code
black reactor_core/

# Lint code
ruff check reactor_core/

# Type checking
mypy reactor_core/

# Security scanning
bandit -r reactor_core/
```

## 📈 Version History

### **v238.0** - Ecosystem: Degenerate Response Elimination (2026-02-08, JARVIS Body-side)
- JARVIS Body v238.0 fixes degenerate LLM responses ("...") via 3-layer defense-in-depth
- SIMPLE classification narrowed: "what is X?" queries promoted to MODERATE (512 tokens)
- Backend degenerate response detection with safe retry using MODERATE parameters
- Client-side degenerate suppression with zombie timeout re-arming
- `requestId` echo in WebSocket responses enables frontend deduplication
- Reactor-Core's DPO training pipeline receives improved telemetry (complexity + source fields) for preference pair generation
- **Note:** v238.0 changes are in JARVIS (Body) and documented here for ecosystem coherence

### **v92.0** - Reliability & Robustness (2025-01-15)
- **Structured Concurrency**: Python 3.11+ TaskGroup patterns for robust async operations
- **Connection Pooling**: Efficient HTTP/Redis connection management with automatic lifecycle
- **Dynamic Path Resolution**: Zero hardcoding, XDG-compliant paths, environment-driven config
- **Atomic File Writes**: Prevents checkpoint corruption from partial writes
- **Circuit Breaker Pattern**: Protects external service calls with auto-recovery
- **Backpressure Control**: Prevents memory exhaustion under high load
- **Proper Async Patterns**: Deadlock-free async/await with timeouts
- **Gradient Verification**: Checksum validation for distributed training
- **Memory Pressure Awareness**: Adaptive behavior under resource constraints
- **Unified Error Handling**: Centralized error classification and routing

### **v91.0** - Advanced Learning & Distributed Training (2025-01-10)
- **Online/Incremental Learning**: Prioritized experience replay with importance sampling
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting during updates
- **Concept Drift Detection**: Page-Hinkley test for automatic model adaptation
- **Data Versioning**: Content-addressed storage with lineage tracking (DVC compatible)
- **GCP Spot VM Checkpointing**: Predictive preemption with multi-signal detection
- **Distributed Training**: Multi-VM coordination with gradient compression
- **Dynamic Resource Allocation**: Auto-scaling with cost-aware decisions
- **MLForge C++ Bindings**: High-performance matrix/neural ops with pybind11

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

### **v80.0** - World Models & Causal Reasoning (2024-12-20)
- World model training with latent dynamics and planning
- Causal reasoning with SCMs and do-calculus
- Advanced data preprocessing with quality gates
- Synthetic data generation (3-10x augmentation)
- Active learning for efficient labeling

### **v79.0** - Curriculum & Meta-Learning (2024-12-15)
- Curriculum learning with progressive difficulty
- Meta-learning (MAML, Reptile, Meta-SGD)
- Dependency injection framework

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

## 🗺️ Roadmap — Next Phases

### v242.0 — Training Data Pipeline Activation (Planned)

**Status:** Infrastructure ~80% built. The pieces exist across all three repos but the connection points are not wired. This is the single highest-impact next step for the entire JARVIS ecosystem.

**What Reactor Core needs to do:**

1. **Ingest telemetry from JARVIS Body**
   - `TelemetryIngestor` reads JSONL from `~/.jarvis/telemetry/`
   - Expected format: `{event_type, timestamp, properties: {user_input, output, task_type}, metrics: {model_id, latency_ms, tokens}}`
   - JARVIS Body's `TelemetryEmitter` writes to this path — but the schema alignment is unverified
   - **Fix needed:** Validate field names match between emitter and ingestor. Add schema versioning.

2. **Ingest interaction logs from J-Prime**
   - `TrinityExperienceReceiver` watches `~/.jarvis/` for event files from J-Prime
   - J-Prime's `TrainingDataPipeline` captures conversations and calls `ReactorCoreBridge.upload_training_data()`
   - **Broken link:** `upload_training_data()` is called but **not fully implemented** on the Reactor Core side. This means J-Prime's locally captured conversations never arrive.
   - **Fix needed:** Implement the upload endpoint in Reactor Core's API server, or have J-Prime write directly to `~/.jarvis/telemetry/` in the expected JSONL format.

3. **Generate DPO preference pairs automatically**
   - v241.1's multi-model routing creates implicit quality comparisons:
     ```
     Query: "solve 5x+3=18" routed to Mistral-7B → "x=11" (wrong)
     Same query type routed to Qwen-Math-7B → "x=3" (correct)
     → Automatic DPO pair: {prompt, chosen: "x=3", rejected: "x=11"}
     ```
   - **Multi-model routing IS the labeling mechanism.** No human annotation needed.
   - `model_id` in telemetry enables per-model performance tracking and automatic pair generation.
   - **Fix needed:** Build the comparison logic in `UnifiedTrainingPipeline` that groups interactions by query type and generates preference pairs from model_id divergences.

4. **Fine-tune and export**
   - `UnifiedTrainingPipeline` already supports DPO training with LoRA/QLoRA
   - Training requires full-precision FP16 base models (~14 GB for 7B), not the GGUFs
   - Output: quantized GGUF files deployed to J-Prime's golden image
   - Elastic Weight Consolidation (EWC) prevents catastrophic forgetting

5. **Deploy to J-Prime**
   - `HotSwapManager` in J-Prime accepts fine-tuned GGUF files with zero-downtime swap
   - Reactor Core's `ModelDeploymentManager` handles GGUF export and deployment signaling
   - **Fix needed:** Verify the deployment signal path (Reactor Core → Trinity Protocol → J-Prime HotSwapManager) is end-to-end functional.

**When this works, the full loop is:**
```
User → JARVIS Body → J-Prime (inference + telemetry capture)
  → ~/.jarvis/telemetry/ (JSONL logs with model_id)
    → Reactor Core TelemetryIngestor
      → DPO pair generation (cross-model comparison)
        → LoRA fine-tuning (DPO on preference data)
          → GGUF quantization → deploy to J-Prime golden image
            → Models improve at being JARVIS, automatically
```

### Architectural Status Report — Cross-Repo Audit (February 2026)

A comprehensive audit of the JARVIS ecosystem identified critical integration gaps that affect Reactor Core's role as the training and learning layer:

#### Training Data Pipeline Status

The training data pipeline from JARVIS Body → J-Prime → Reactor Core is **~80% built but not end-to-end functional**. The infrastructure exists at each node but the connections between them are broken:

| Component | Location | Status | Issue |
|-----------|----------|--------|-------|
| `TelemetryEmitter` | JARVIS Body | Built | JSONL output format may not match `TelemetryIngestor` expectations |
| `TelemetryIngestor` | Reactor Core | Built | Reads from `~/.jarvis/telemetry/` but no data is being written there in the correct format |
| `ReactorCoreBridge.upload_training_data()` | J-Prime | **Not implemented** | The method is called but has no implementation — J-Prime's captured conversations never reach Reactor Core |
| `UnifiedTrainingPipeline` | Reactor Core | Built | DPO/LoRA training works in isolation but has never run on real production data |
| `HotSwapManager` | J-Prime | Built | Accepts GGUF files for hot swap but the deployment signal path (Reactor → Trinity → J-Prime) is unverified |
| `ModelDeploymentManager` | Reactor Core | Built | GGUF export and deployment signaling exists but is untested end-to-end |

**Root Cause:** Each repo built its side of the pipeline independently, but nobody wired the handoff points. The JSONL format, the upload API, and the deployment signals need explicit cross-repo contract verification.

#### Google Workspace Fixes Impact (v245.0)

The v245.0 Google Workspace fixes in JARVIS Body have a direct impact on Reactor Core's future training data:

- **Draft email body generation now works** — Previously silent failures meant no email body generation telemetry was captured. Now, every draft email request generates a real LLM inference call (with `X-Model-Id`), producing training-relevant interaction data.
- **Agent singleton fix eliminates noise** — The 49s recreation bug caused timeout errors that would have polluted training data with failed interactions. Clean request/response pairs are now the norm.
- **Task-type metadata flows correctly** — Workspace commands now carry proper task-type metadata, enabling Reactor Core to generate per-model DPO pairs from workspace interactions.

#### LangGraph in JARVIS Body

All 9 LangGraph reasoning graphs in JARVIS Body are **dead code** because `langgraph` is not installed. This means:

- The reasoning engine uses linear fallback (analysis → planning → validation → execution → reflection → learning) instead of conditional graph routing
- The `route_after_reflection()` loop-back (for iterative reasoning on low confidence) has **never executed**
- Training data from the reasoning engine reflects single-pass linear thinking, not the intended iterative, graph-based reasoning
- **Impact on Reactor Core:** When the training pipeline activates, the quality of reasoning traces available for fine-tuning will be lower than designed until LangGraph is installed (v246.0 in JARVIS Body)

#### Planned: Unified Agent Runtime (v247.0 in JARVIS Body)

The JARVIS Body Unified Agent Runtime will generate a new class of training data for Reactor Core:

- **Multi-step goal traces** — Complete autonomous workflows (sense → think → act → verify → reflect) with sub-step decomposition, producing rich sequential decision-making data
- **Cross-agent coordination traces** — When the Runtime dispatches work to Neural Mesh agents, the coordination patterns become training data for improving multi-agent orchestration
- **Failure recovery traces** — When autonomous goals fail and the Runtime retries or replans, the recovery patterns become training data for improving resilience
- **Human escalation signals** — When the Runtime escalates to the user for approval, the decision boundary becomes a training signal for the safety classifier

**New training data types Reactor Core should prepare for:**

| Data Type | Source | Training Method |
|-----------|--------|----------------|
| Goal decomposition traces | Agent Runtime THINK phase | Supervised fine-tuning on planning |
| Sub-step success/failure | Agent Runtime VERIFY phase | DPO pairs (successful vs. failed approaches) |
| Escalation decisions | Agent Runtime escalation protocol | Constitutional AI for safety boundaries |
| Multi-agent coordination | Neural Mesh dispatch logs | Curriculum learning on orchestration complexity |

---

### v243.0 — Ouroboros Training Support (Planned)

Support the training side of JARVIS self-programming:

- [ ] **Code quality evaluation** — Evaluate generated code diffs for correctness, style, security. Feed scores back as DPO signals.
- [ ] **Self-programming telemetry** — Capture Ouroboros cycles (architect plan → generated code → verifier review → human decision) as training data.
- [ ] **Architect/Implementer specialization** — Fine-tune DeepSeek-R1-14B on architectural reasoning traces and Qwen-Coder-14B on code generation from plans, using Ouroboros interaction data.
- [ ] **Constitutional AI for code** — Apply Constitutional AI training to code generation: "Is this code safe? Does it follow the existing patterns? Does it handle errors?"

### v244.0 — Continuous Learning Loop (Planned)

- [ ] **Night Shift automation** — `NightShiftScheduler` already exists. Wire it to trigger DPO training runs during off-peak hours using accumulated telemetry.
- [ ] **Concept drift detection** — `PageHinkleyDriftDetector` already exists. Monitor model performance metrics and trigger retraining when quality degrades.
- [ ] **A/B model testing** — `ModelRegistry` supports versioned models and A/B testing. Deploy fine-tuned models alongside originals, compare performance, promote winners.
- [ ] **Curriculum learning** — Start fine-tuning on easy tasks (general chat), progressively add harder tasks (math, code, reasoning) using curriculum learning infrastructure already built in v79.0.

### v245.0 — Distributed Training on GCP (Planned)

- [ ] **Multi-VM gradient aggregation** — v91.0 built distributed training with gradient compression. Activate for 14B model fine-tuning which exceeds single-VM memory.
- [ ] **Spot VM resilience** — Predictive preemption with checkpoint save already built. Test with real training runs.
- [ ] **Cost-aware scheduling** — Train on spot VMs during cheap hours, pause during expensive hours. `DynamicResourceAllocator` has the framework.

### v246.0 — Agent Runtime Training Data Ingestion (Planned)

Prepare Reactor Core to ingest and process training data from the JARVIS Body Unified Agent Runtime:

- [ ] **Goal trace schema** — Define JSONL schema for multi-step autonomous goal traces (goal → sub-steps → outcomes → reflections) compatible with `TelemetryIngestor`
- [ ] **Sequential DPO pairs** — Generate preference pairs from goal execution sequences: successful multi-step approaches vs. failed approaches for the same goal type
- [ ] **Escalation boundary training** — Use human escalation decisions (approve/reject) as Constitutional AI training signals for the safety classifier
- [ ] **Multi-agent coordination curriculum** — Build progressive difficulty curriculum from simple single-agent tasks to complex multi-agent workflows
- [ ] **Failure recovery fine-tuning** — Fine-tune reasoning models on recovery traces: when a sub-step fails, what replanning strategies worked vs. didn't
- [ ] **Cross-model comparison at scale** — With the Agent Runtime generating higher request volume across all specialist models, DPO pair generation becomes more statistically significant

### v247.0 — End-to-End Pipeline Verification (Planned)

Verify and fix all cross-repo handoff points in the training pipeline:

- [ ] **JSONL format contract** — Define and enforce a shared schema between `TelemetryEmitter` (JARVIS Body), `TelemetryIngestor` (Reactor Core), and `TrainingDataPipeline` (J-Prime)
- [ ] **Implement `ReactorCoreBridge.upload_training_data()`** — The broken link in J-Prime that prevents locally captured conversations from reaching Reactor Core
- [ ] **Deployment signal verification** — Test the Reactor Core → Trinity Protocol → J-Prime `HotSwapManager` path end-to-end with a dummy GGUF
- [ ] **Integration test suite** — Automated test that writes a telemetry event in JARVIS Body format, ingests it in Reactor Core, generates a DPO pair, runs a mock training step, exports a GGUF, and signals J-Prime for hot swap
- [ ] **Monitoring dashboard** — Track pipeline health: events written/day, events ingested/day, DPO pairs generated, training runs completed, models deployed

---

## 📚 API Documentation

### REST API Endpoints

Once the API server is running (`python3 run_supervisor.py`), access:

- **API Base URL**: `http://localhost:8003`
- **Interactive Docs**: `http://localhost:8003/docs` (Swagger UI)
- **ReDoc**: `http://localhost:8003/redoc`
- **Health Check**: `http://localhost:8003/health`

### Key Endpoints

#### Training

```bash
# Trigger training
POST /api/v1/training/trigger
{
  "model_name": "llama-2-7b",
  "training_type": "dpo",
  "config": {
    "num_epochs": 3,
    "batch_size": 4
  }
}

# Get training status
GET /api/v1/training/status/{job_id}

# Cancel training
POST /api/v1/training/cancel/{job_id}
```

#### Model Registry

```bash
# List models
GET /api/v1/models

# Get model info
GET /api/v1/models/{model_id}

# Register model
POST /api/v1/models/register
{
  "model_id": "my-model-v1",
  "model_path": "/path/to/model",
  "metadata": {...}
}
```

#### Scheduler

```bash
# Schedule job
POST /api/v1/scheduler/schedule
{
  "name": "nightly_training",
  "schedule_type": "cron",
  "cron_expression": "0 2 * * *",
  "job_config": {...}
}

# List scheduled jobs
GET /api/v1/scheduler/jobs
```

#### Telemetry

```bash
# Submit telemetry
POST /api/v1/telemetry/submit
{
  "event_type": "interaction",
  "data": {...}
}

# Query metrics
GET /api/v1/telemetry/metrics?start_time=...&end_time=...
```

### WebSocket Events

Connect to `ws://localhost:8003/ws` for real-time events:

```javascript
const ws = new WebSocket('ws://localhost:8003/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Event:', data.type, data.payload);
};

// Subscribe to training events
ws.send(JSON.stringify({
  type: 'subscribe',
  channels: ['training:progress', 'training:complete']
}));
```

### Model Server API

Model server runs on port 8001 (configurable):

```bash
# Inference
POST http://localhost:8001/predict
{
  "prompt": "What is machine learning?",
  "model_id": "llama-2-7b",
  "max_tokens": 256,
  "temperature": 0.7
}

# List loaded models
GET http://localhost:8001/models

# Load model
POST http://localhost:8001/models/load
{
  "model_id": "my-model",
  "model_path": "/path/to/model",
  "backend": "vllm"
}
```

## 🔗 Links & Resources

### Repositories (Trinity)

| Role | Repository | URL |
|------|------------|-----|
| **Body** | JARVIS (JARVIS-AI-Agent) | https://github.com/drussell23/JARVIS-AI-Agent |
| **Mind** | JARVIS-Prime | https://github.com/drussell23/jarvis-prime |
| **Nerves** | **Reactor-Core (this repo)** | https://github.com/drussell23/JARVIS-Reactor |
| **C++ Core** | MLForge | https://github.com/drussell23/MLForge |

### Documentation

- **Architecture Docs**: See `ARCHITECTURE_ADVANCED.md`
- **Trinity Integration**: See `TRINITY_INTEGRATION_COMPLETE.md`
- **Version History**: See `CHANGELOG.md` (if available)

### Community

- **Issues**: https://github.com/drussell23/JARVIS-Reactor/issues
- **Discussions**: https://github.com/drussell23/JARVIS-Reactor/discussions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our code style
4. Add tests for new features
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- **Formatting**: Black (line length 100)
- **Linting**: Ruff
- **Type Hints**: Required for all functions
- **Docstrings**: Google style

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with ❤️ for the JARVIS AGI Ecosystem

**Special Thanks:**
- PyTorch team for the excellent ML framework
- Hugging Face for transformers and PEFT
- FastAPI for the amazing async web framework
- All contributors and users of the JARVIS ecosystem

---

**Version**: 2.11.0 (v92.0)  
**Last Updated**: February 2026  
**Status**: ✅ Production Ready (training infrastructure built; cross-repo pipeline pending activation)

### Known Gaps (In Roadmap)

- **Training data pipeline not end-to-end** — Infrastructure exists at each node (emitter, ingestor, trainer, deployer) but cross-repo handoffs are broken (v247.0 target)
- **`ReactorCoreBridge.upload_training_data()` not implemented** — J-Prime's captured conversations never reach Reactor Core (v247.0 target)
- **No real production training data yet** — `UnifiedTrainingPipeline` has never run on actual user interaction data (v242.0 target)
- **LangGraph reasoning traces unavailable** — JARVIS Body's reasoning engine produces linear fallback traces, not rich graph-based reasoning data (depends on JARVIS Body v246.0)
- **Agent Runtime training data schema undefined** — When autonomous goal pursuit generates multi-step traces, Reactor Core needs a new ingestion schema (v246.0 target)
