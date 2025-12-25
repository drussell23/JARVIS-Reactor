# Reactor Core

**An AI/ML Training Engine with Cross-Repository Event Streaming & Safety Integration**

Reactor Core is a hybrid ML training framework that combines:
- High-performance C++ ML engine (MLForge)
- Python-first API for PyTorch, LoRA, DPO, FSDP
- GCP Spot VM resilience with auto-checkpointing
- Environment-aware compute (M1 local vs GCP remote)
- **Real-time event streaming** across JARVIS-AI-Agent, JARVIS Prime, and Reactor Core
- **Vision safety integration** with comprehensive audit trails and kill switch mechanisms

## Architecture

```
Reactor Core
├── MLForge C++ Core (submodule)
│   └── High-performance ML primitives
└── Python Layer
    ├── training/       # LoRA, DPO, FSDP training
    ├── data/           # Data loading & preprocessing
    ├── eval/           # Model evaluation
    ├── serving/        # Model serving utilities
    ├── gcp/            # GCP Spot VM integration
    ├── integration/    # Cross-repo event streaming (NEW)
    │   ├── event_bridge.py       # Real-time event synchronization
    │   ├── jarvis_connector.py   # JARVIS-AI-Agent integration
    │   ├── prime_connector.py    # JARVIS Prime integration
    │   └── cost_bridge.py        # Unified cost tracking
    ├── api/            # REST API server (NEW)
    │   └── server.py             # FastAPI endpoints for JARVIS
    └── utils/          # Common utilities
```

## Installation

### Quick Install (Python only, no C++ bindings)
```bash
pip install reactor-core
```

### Build from Source (with MLForge C++ bindings)
```bash
# Clone with submodules
git clone --recursive https://github.com/drussell23/reactor-core.git
cd reactor-core

# Install dependencies (requires CMake and pybind11)
pip install pybind11 cmake

# Build and install
pip install -e .
```

### For Local Development (M1 Mac)
```bash
pip install reactor-core[local]
```

### For GCP Training (32GB VM)
```bash
pip install reactor-core[gcp]
```

## Quick Start

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

# Auto-detect environment
trainer = Trainer(config)

# Train (auto-resumes on GCP Spot preemption)
trainer.train("./data/train.jsonl")
```

## Environment Detection

Reactor Core automatically detects your environment:

| Environment | Mode | Features |
|-------------|------|----------|
| M1 Mac 16GB | Lightweight | Inference-only, quantized models |
| GCP 32GB VM | Full Training | LoRA, DPO, FSDP, auto-resume |

## GCP Spot VM Support

Built-in checkpoint/resume for preemptible VMs:

```python
from reactor_core.gcp import SpotVMCheckpointer

# Automatically saves checkpoints every N steps
# Resumes from last checkpoint on VM restart
trainer = Trainer(config, checkpointer=SpotVMCheckpointer(
    checkpoint_interval=500,
    gcs_bucket="gs://my-training-checkpoints"
))
```

## Cross-Repository Event Streaming

Reactor Core provides a sophisticated event bridge for real-time synchronization across the JARVIS ecosystem.

### Event Bridge Architecture

```python
from reactor_core.integration import EventBridge, EventSource, EventType

# Initialize event bridge
bridge = EventBridge(
    source=EventSource.REACTOR_CORE,
    websocket_url="ws://localhost:8000/ws/events",
    fallback_to_file=True,
    redis_url="redis://localhost:6379"  # Optional
)

# Subscribe to events from other services
await bridge.subscribe([
    EventType.INTERACTION_START,
    EventType.TRAINING_START,
    EventType.COST_UPDATE
])

# Emit events to other services
await bridge.publish(
    EventType.TRAINING_PROGRESS,
    {"epoch": 2, "loss": 0.123, "progress": 0.67},
    priority=2
)

# Listen for events
async for event in bridge.listen():
    print(f"Received: {event.event_type} from {event.source}")
```

### Supported Event Types

#### Training Events
- `TRAINING_START`: Training pipeline initiated
- `TRAINING_PROGRESS`: Epoch/batch progress updates
- `TRAINING_COMPLETE`: Training successfully completed
- `TRAINING_FAILED`: Training failed with error details

#### Cost Tracking Events (v10.0+)
- `COST_UPDATE`: Real-time cost updates from any service
- `COST_ALERT`: Budget threshold exceeded
- `COST_REPORT`: Periodic cost summaries
- `INFERENCE_METRICS`: Token usage and API call metrics

#### Infrastructure Events (v10.0+)
- `RESOURCE_CREATED`: GCP resource provisioned
- `RESOURCE_DESTROYED`: GCP resource terminated
- `ORPHAN_DETECTED`: Unused resource identified
- `ORPHAN_CLEANED`: Orphaned resource cleaned up
- `SQL_STOPPED`/`SQL_STARTED`: Cloud SQL state changes

#### Safety Events (v10.3+)
- `SAFETY_AUDIT`: Action plan audited for safety risks
- `SAFETY_BLOCKED`: High-risk action automatically blocked
- `SAFETY_CONFIRMED`: User confirmed risky action
- `SAFETY_DENIED`: User rejected risky action
- `KILL_SWITCH_TRIGGERED`: Emergency halt activated
- `VISUAL_CLICK_PREVIEW`: Click action previewed to user
- `VISUAL_CLICK_VETOED`: Click cancelled during preview

### Event Priority System

Events are prioritized for training optimization:
- **Priority 1 (Highest)**: Safety blocks, kill switch, critical errors
- **Priority 2 (High)**: Safety audits, confirmations, cost alerts
- **Priority 3 (Normal)**: Training progress, visual previews
- **Priority 4 (Low)**: Informational events, metrics

### Automatic Deduplication

The event bridge prevents duplicate processing:
```python
# Events are deduplicated based on content hash
# Duplicate events within 60 seconds are automatically filtered
await bridge.publish(EventType.COST_UPDATE, {"total": 15.43})
await bridge.publish(EventType.COST_UPDATE, {"total": 15.43})  # Deduplicated
```

## Vision Safety Integration

Reactor Core v10.3+ provides comprehensive safety event tracking for vision-enabled AI actions.

### Safety Event Emission

```python
from reactor_core.integration import EventBridge, EventSource

bridge = EventBridge(source=EventSource.JARVIS_AGENT)

# Emit safety audit when plan is generated
await bridge.emit_safety_audit(
    goal="Update production database schema",
    plan_steps=5,
    verdict="REQUIRES_CONFIRMATION",
    risk_level="HIGH",
    risky_steps=[
        {
            "step": 3,
            "action": "DROP TABLE users",
            "risk": "Data loss",
            "severity": "CRITICAL"
        }
    ],
    confirmation_required=True
)

# Emit when action is blocked
await bridge.emit_safety_blocked(
    action="rm -rf /",
    reason="Destructive filesystem operation detected",
    safety_tier="TIER_1_CRITICAL",
    auto_blocked=True
)

# Emit user confirmation/denial
await bridge.emit_safety_confirmation(
    action="Delete 1,000 files in ~/Downloads",
    risk_level="MEDIUM",
    confirmed=True,
    confirmation_method="voice",
    user_response="yes, proceed"
)

# Emit kill switch activation
await bridge.emit_kill_switch_triggered(
    trigger_method="mouse_corner",
    halted_action="Sending 500 emails",
    response_time_ms=147.3
)

# Emit visual click preview/veto
await bridge.emit_visual_click_event(
    x=450,
    y=320,
    button="left",
    vetoed=True,
    preview_duration_ms=2500,
    veto_reason="User moved mouse to kill corner"
)
```

### Safety Audit Trail

All safety events are logged with full context for training and analysis:

```python
# Query safety events from event history
safety_events = await bridge.get_events_by_type([
    EventType.SAFETY_AUDIT,
    EventType.SAFETY_BLOCKED,
    EventType.KILL_SWITCH_TRIGGERED
], time_range="last_24h")

# Analyze safety patterns
for event in safety_events:
    print(f"{event.timestamp}: {event.event_type}")
    print(f"  Action: {event.data['action']}")
    print(f"  Risk: {event.data.get('risk_level', 'N/A')}")
    print(f"  Outcome: {event.data.get('verdict', 'N/A')}")
```

### Integration with JARVIS Computer Use

Safety events are automatically emitted during computer use actions:

```python
# In JARVIS-AI-Agent with Computer Use
from anthropic import Anthropic
from reactor_core.integration import EventBridge

client = Anthropic()
bridge = EventBridge(source=EventSource.JARVIS_AGENT)

# Before executing computer use action
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    tools=[computer_tool],
    messages=[{"role": "user", "content": "Delete old files"}]
)

# Safety audit triggered automatically
# User shown preview if action is risky
# Event emitted: SAFETY_AUDIT, VISUAL_CLICK_PREVIEW, etc.

# If user cancels during preview
await bridge.emit_visual_click_event(
    x=100, y=200,
    button="left",
    vetoed=True,
    preview_duration_ms=1500,
    veto_reason="User moved cursor to kill corner"
)
```

## Cost Tracking Integration

Unified cost tracking across all JARVIS repositories.

### Cost Bridge Usage

```python
from reactor_core.integration import CostBridge

# Initialize cost bridge
cost_bridge = CostBridge(
    event_bridge=bridge,
    budget_limit=100.0,  # $100 daily budget
    alert_threshold=0.8  # Alert at 80% of budget
)

# Automatic cost tracking for training
async with cost_bridge.track_training("lora-fine-tune"):
    trainer.train()  # Costs automatically tracked

# Query current costs
daily_cost = await cost_bridge.get_daily_cost()
print(f"Today's spend: ${daily_cost:.2f}")

# Get cost breakdown by service
breakdown = await cost_bridge.get_cost_breakdown()
for service, cost in breakdown.items():
    print(f"{service}: ${cost:.2f}")
```

### Automatic Budget Alerts

```python
# Budget alerts emitted automatically
@bridge.on_event(EventType.COST_ALERT)
async def handle_cost_alert(event):
    print(f"⚠️  Budget Alert: ${event.data['current']:.2f} / ${event.data['limit']:.2f}")
    print(f"Threshold: {event.data['percentage']:.1f}%")

    # Optionally pause training
    if event.data['percentage'] > 95:
        await trainer.pause()
```

## REST API Server

Reactor Core provides a FastAPI server for external integrations.

### Starting the API Server

```bash
# Start with uvicorn
uvicorn reactor_core.api.server:app --host 0.0.0.0 --port 8003 --reload

# Or use the module directly
python -m reactor_core.api.server
```

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Trigger Training
```bash
POST /training/trigger
{
  "model_name": "llama-2-7b",
  "training_type": "lora",
  "config": {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4
  }
}
```

#### Submit Experience Log
```bash
POST /experience/submit
{
  "interaction_id": "uuid-here",
  "user_message": "Fix the authentication bug",
  "assistant_response": "I'll fix that...",
  "outcome": "success",
  "metadata": {
    "tokens_used": 1234,
    "model": "claude-3-7-sonnet"
  }
}
```

#### Get JARVIS Status
```bash
GET /jarvis/status
```

Response:
```json
{
  "broadcaster_active": true,
  "uptime_seconds": 3847.2,
  "last_interaction": "2025-12-25T10:30:45Z",
  "event_bridge_connected": true,
  "pending_events": 3
}
```

## Features

### Core Training Features
- **PyTorch-First**: Full PyTorch compatibility
- **LoRA/QLoRA**: Memory-efficient fine-tuning
- **DPO Support**: Direct Preference Optimization
- **FSDP**: Fully Sharded Data Parallel for large models
- **Resume Training**: Auto-resume from checkpoints
- **Async-Safe**: Non-blocking training loops
- **C++ Acceleration**: Optional MLForge backend for speed

### Integration & Event Streaming (v10.0+)
- **Cross-Repository Events**: Real-time event synchronization across JARVIS ecosystem
- **WebSocket Streaming**: Low-latency event broadcasting
- **Event Deduplication**: Prevents duplicate event processing
- **Automatic Reconnection**: Resilient connections with exponential backoff
- **Multi-Transport Support**: WebSocket, file-based watching, Redis pub/sub
- **Cost Tracking**: Unified cost analytics across all repos with budget alerts
- **Infrastructure Events**: Resource lifecycle tracking and orphan detection

### Vision Safety Integration (v10.3+)
- **Safety Audit Trail**: Comprehensive logging of all safety decisions
- **Action Blocking**: Automatic blocking of high-risk actions
- **User Confirmation**: Multi-factor confirmation for risky operations
- **Kill Switch**: Dead man's switch for emergency halts
- **Visual Click Previews**: Pre-execution click verification
- **Veto Mechanism**: Real-time action cancellation with preview window

## Version History

### **v10.3** - Vision Safety Integration (Current)
- Added comprehensive safety event types for vision-enabled AI
- Safety audit trail with `SAFETY_AUDIT`, `SAFETY_BLOCKED`, `SAFETY_CONFIRMED`, `SAFETY_DENIED`
- Kill switch mechanism with `KILL_SWITCH_TRIGGERED` event
- Visual click preview and veto system (`VISUAL_CLICK_PREVIEW`, `VISUAL_CLICK_VETOED`)
- Convenience methods for all safety event emissions
- Priority-based event processing for safety-critical events

### **v10.0** - Cross-Repository Integration
- Real-time event streaming across JARVIS ecosystem
- WebSocket-based event bridge with auto-reconnection
- Unified cost tracking with budget alerts
- Infrastructure lifecycle events (resources, orphans, SQL)
- Multi-transport support (WebSocket, file-based, Redis)
- Event deduplication and priority system
- REST API server for external integrations
- JARVIS connector for bidirectional communication
- Prime connector for training pipeline integration

### **v1.0.0** - Initial Release
- PyTorch-first ML training framework
- LoRA/QLoRA memory-efficient fine-tuning
- DPO (Direct Preference Optimization) support
- FSDP for large model training
- GCP Spot VM resilience with auto-checkpointing
- MLForge C++ core integration
- Environment-aware compute (M1 local vs GCP remote)

## Integration Architecture

### JARVIS Ecosystem Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    JARVIS Ecosystem                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │ JARVIS Agent │◄───────►│ JARVIS Prime │                  │
│  │  (Claude)    │  Events │  (Training)  │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                           │
│         │     Event Bridge       │                           │
│         │   (WebSocket/Redis)    │                           │
│         │                        │                           │
│  ┌──────▼────────────────────────▼───────┐                  │
│  │        Reactor Core                   │                  │
│  │  ┌─────────────────────────────────┐  │                  │
│  │  │   Event Bridge Core             │  │                  │
│  │  │   - Safety Events               │  │                  │
│  │  │   - Cost Tracking               │  │                  │
│  │  │   - Infrastructure Events       │  │                  │
│  │  │   - Training Events              │  │                  │
│  │  └─────────────────────────────────┘  │                  │
│  │                                        │                  │
│  │  ┌─────────────┐   ┌──────────────┐  │                  │
│  │  │  REST API   │   │  MLForge C++ │  │                  │
│  │  │  Server     │   │  Engine      │  │                  │
│  │  └─────────────┘   └──────────────┘  │                  │
│  └────────────────────────────────────────┘                  │
│                                                               │
│         ▼                        ▼                            │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  Cloud SQL   │         │  GCP Storage │                  │
│  │  (Events DB) │         │ (Checkpoints)│                  │
│  └──────────────┘         └──────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Event Flow Example

```
User: "Delete old logs" (to JARVIS Agent)
    │
    ▼
JARVIS Agent generates plan with Computer Use
    │
    ▼
Safety Audit triggered
    │
    ├─► Reactor Core: SAFETY_AUDIT event emitted
    │   └─► Stored in Cloud SQL for training
    │
    ├─► Risk detected: HIGH (bulk file deletion)
    │
    ▼
User confirmation required
    │
    ├─► Visual preview shown (VISUAL_CLICK_PREVIEW event)
    │
User confirms or denies
    │
    ├─► SAFETY_CONFIRMED or SAFETY_DENIED event
    │   └─► Training data captured for learning
    │
    ▼
Action executed (if confirmed)
    │
    └─► All events synced across JARVIS ecosystem
```

## Use Cases

### 1. Real-Time Safety Monitoring
Monitor all AI actions across your JARVIS ecosystem with comprehensive audit trails:

```python
# In JARVIS Agent
await bridge.emit_safety_audit(
    goal="Update system configurations",
    plan_steps=8,
    verdict="SAFE_WITH_MONITORING",
    risk_level="MEDIUM",
    risky_steps=[],
    confirmation_required=False
)

# In Reactor Core (receives event automatically)
# Event stored in database for analysis and training
```

### 2. Multi-Repository Cost Optimization
Track costs across all services with unified budgets:

```python
# JARVIS Agent tracks inference costs
await bridge.publish(EventType.COST_UPDATE, {
    "service": "claude-api",
    "amount": 0.47,
    "tokens": 1834
})

# JARVIS Prime tracks training costs
await bridge.publish(EventType.COST_UPDATE, {
    "service": "gcp-compute",
    "amount": 2.15,
    "duration_minutes": 47
})

# Reactor Core aggregates and alerts
# Alert triggered at 80% of $100 daily budget
```

### 3. Training from Safety Events
Use safety decision history to improve AI behavior:

```python
# Query safety events for training
safety_data = await bridge.get_events_by_type([
    EventType.SAFETY_CONFIRMED,
    EventType.SAFETY_DENIED
], time_range="last_30_days")

# Extract user preferences
for event in safety_data:
    if event.data["confirmed"]:
        # User accepted this action - train as positive example
        training_examples.append({
            "action": event.data["action"],
            "risk_level": event.data["risk_level"],
            "user_decision": "approved"
        })
```

### 4. Emergency Kill Switch Integration
Implement dead man's switch across all services:

```python
# In JARVIS Agent computer use wrapper
@on_mouse_corner_trigger
async def emergency_halt():
    await bridge.emit_kill_switch_triggered(
        trigger_method="mouse_corner",
        halted_action=current_action,
        response_time_ms=time_since_trigger
    )

    # Kill switch event propagates to all services
    # All in-progress actions halted immediately
```

## Environment Variables

```bash
# Reactor Core Configuration
REACTOR_CORE_HOST=0.0.0.0
REACTOR_CORE_PORT=8003

# Event Bridge Configuration
EVENT_BRIDGE_WS_URL=ws://localhost:8000/ws/events
EVENT_BRIDGE_REDIS_URL=redis://localhost:6379  # Optional
EVENT_BRIDGE_FALLBACK_FILE=/tmp/events.jsonl

# JARVIS Integration
JARVIS_API_URL=http://localhost:8000
JARVIS_PRIME_URL=http://localhost:8001

# Cost Tracking
COST_DAILY_BUDGET=100.0
COST_ALERT_THRESHOLD=0.8

# GCP Configuration
GCP_PROJECT_ID=jarvis-473803
GCS_CHECKPOINT_BUCKET=gs://jarvis-training-checkpoints

# Safety Configuration
SAFETY_AUDIT_ENABLED=true
SAFETY_PREVIEW_DURATION_MS=2000
KILL_SWITCH_ENABLED=true
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run integration tests
pytest tests/integration/

# Run safety event tests
pytest tests/integration/test_event_bridge.py -k safety
```

### Building MLForge C++ Bindings

```bash
# Build C++ extensions
cd mlforge
mkdir build && cd build
cmake ..
make -j$(nproc)

# Install Python bindings
cd ../..
pip install -e .
```

## License

MIT License

## Links

- **MLForge C++ Core**: https://github.com/drussell23/MLForge
- **JARVIS Prime**: https://github.com/drussell23/jarvis-prime (uses Reactor Core)
- **JARVIS-AI-Agent**: https://github.com/drussell23/jarvis-ai-agent (integrates with Reactor Core)

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For issues and questions:
- **GitHub Issues**: https://github.com/drussell23/reactor-core/issues
- **Documentation**: See `/docs` folder for detailed guides
