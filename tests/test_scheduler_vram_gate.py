"""VRAM admission gate for the Night Shift scheduler.

Regression cover for a gate that read the wrong number and then did not
check it:

  * ``_get_gpu_metrics`` reported nvidia-smi's ``utilization.memory`` —
    the percent of time the memory bus was busy, NOT how much VRAM is
    occupied. Measured on the RTX 5090 with a resident 32B model:
    ``utilization.memory=0`` while 29078/32607 MiB (89.2%) was held.
  * ``ResourceSnapshot.is_training_allowed`` collected
    ``gpu_memory_percent`` and never consulted it, so a training job was
    admitted onto a card with ~3 GiB free and OOMed.

``scheduler`` is loaded by path: ``reactor_core/__init__`` eagerly imports
the training stack (torch/peft/trl), which these stdlib-only assertions
do not need and CI runners without a GPU do not have.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCHEDULER_PATH = (
    Path(__file__).resolve().parents[1] / "reactor_core" / "api" / "scheduler.py"
)


def _load_scheduler() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_scheduler_under_test", _SCHEDULER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scheduler = _load_scheduler()
ResourceSnapshot = scheduler.ResourceSnapshot
SchedulerConfig = scheduler.SchedulerConfig


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_idle_gpu_holding_vram_is_refused() -> None:
    """The exact live measurement: 0% utilization, 89.2% VRAM held."""
    snap = ResourceSnapshot(
        cpu_percent=5.0,
        memory_percent=30.0,
        gpu_percent=0.0,
        gpu_memory_percent=100.0 * 29078 / 32607,
    )
    allowed, reason = snap.is_training_allowed()
    assert allowed is False
    assert "GPU memory too high" in reason
    assert "89.2%" in reason


def test_free_card_is_admitted() -> None:
    snap = ResourceSnapshot(
        cpu_percent=5.0,
        memory_percent=30.0,
        gpu_percent=2.0,
        gpu_memory_percent=4.0,
    )
    allowed, reason = snap.is_training_allowed()
    assert allowed is True
    assert reason == "Resources available"


def test_threshold_is_85_percent_by_default() -> None:
    assert SchedulerConfig.GPU_MEMORY_THRESHOLD == pytest.approx(85.0)


@pytest.mark.parametrize(
    ("vram_pct", "expected_allowed"),
    [(84.9, True), (85.0, True), (85.1, False), (99.0, False)],
)
def test_gate_boundary(vram_pct: float, expected_allowed: bool) -> None:
    snap = ResourceSnapshot(gpu_percent=0.0, gpu_memory_percent=vram_pct)
    assert snap.is_training_allowed()[0] is expected_allowed


def test_unknown_vram_does_not_block() -> None:
    """No GPU / no nvidia-smi ⇒ None ⇒ the gate abstains (CPU-only hosts
    must still be able to train)."""
    snap = ResourceSnapshot(gpu_percent=None, gpu_memory_percent=None)
    assert snap.is_training_allowed()[0] is True


def test_utilization_gate_still_independent() -> None:
    """A busy-but-empty card is still refused by the older gate."""
    snap = ResourceSnapshot(gpu_percent=99.0, gpu_memory_percent=1.0)
    allowed, reason = snap.is_training_allowed()
    assert allowed is False
    assert "GPU usage too high" in reason


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, stdout: bytes, returncode: int = 0) -> None:
        self._stdout = stdout
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, b""


@pytest.mark.asyncio
async def test_metrics_report_occupancy_not_bandwidth(monkeypatch) -> None:
    """nvidia-smi is asked for memory.used/memory.total, and the ratio is
    what reaches the snapshot."""
    seen_args = {}

    async def _fake_exec(*args, **kwargs):
        seen_args["args"] = args
        return _FakeProc(b"0, 29078, 32607\n")

    monkeypatch.setattr(
        scheduler.asyncio, "create_subprocess_exec", _fake_exec,
    )
    monitor = scheduler.ResourceMonitor()
    gpu_util, vram_pct = await monitor._get_gpu_metrics()

    assert "--query-gpu=utilization.gpu,memory.used,memory.total" in seen_args["args"]
    assert gpu_util == pytest.approx(0.0)
    assert vram_pct == pytest.approx(89.18, abs=0.05)


@pytest.mark.asyncio
async def test_zero_total_memory_does_not_divide_by_zero(monkeypatch) -> None:
    async def _fake_exec(*args, **kwargs):
        return _FakeProc(b"0, 0, 0\n")

    monkeypatch.setattr(
        scheduler.asyncio, "create_subprocess_exec", _fake_exec,
    )
    monitor = scheduler.ResourceMonitor()
    _gpu, vram_pct = await monitor._get_gpu_metrics()
    assert vram_pct is None


@pytest.mark.asyncio
async def test_missing_nvidia_smi_degrades_to_none(monkeypatch) -> None:
    async def _boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(
        scheduler.asyncio, "create_subprocess_exec", _boom,
    )
    monitor = scheduler.ResourceMonitor()
    assert await monitor._get_gpu_metrics() == (None, None)
