# Reactor Core

**An AI/ML Training Engine with Python Bindings to MLForge C++ Core**

Reactor Core is a hybrid ML training framework that combines:
- High-performance C++ ML engine (MLForge)
- Python-first API for PyTorch, LoRA, DPO, FSDP
- GCP Spot VM resilience with auto-checkpointing
- Environment-aware compute (M1 local vs GCP remote)

## Architecture

```
Reactor Core
├── MLForge C++ Core (submodule)
│   └── High-performance ML primitives
└── Python Layer
    ├── training/    # LoRA, DPO, FSDP training
    ├── data/        # Data loading & preprocessing
    ├── eval/        # Model evaluation
    ├── serving/     # Model serving utilities
    ├── gcp/         # GCP Spot VM integration
    └── utils/       # Common utilities
```

## Installation

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

## Features

- **PyTorch-First**: Full PyTorch compatibility
- **LoRA/QLoRA**: Memory-efficient fine-tuning
- **DPO Support**: Direct Preference Optimization
- **FSDP**: Fully Sharded Data Parallel for large models
- **Resume Training**: Auto-resume from checkpoints
- **Async-Safe**: Non-blocking training loops
- **C++ Acceleration**: Optional MLForge backend for speed

## Version

**v1.0.0** - Initial release

## License

MIT License

## Links

- **MLForge C++ Core**: https://github.com/drussell23/MLForge
- **JARVIS Prime**: https://github.com/drussell23/jarvis-prime (uses Reactor Core)
