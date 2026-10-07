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


# Memory admission is memory_guard.check_admission's verdict, carried on the
# snapshot. The snapshot used to hold its own 85% VRAM threshold, which
# admitted training beside the resident 30B serving model (26.7/32.6 GiB =
# 82%) and passed an unreadable card as free.


def test_guard_refusal_is_the_snapshot_refusal() -> None:
    """The live case: 30B resident at 82% -- under the old 85% bar, refused now."""
    reason = "VRAM occupancy 82.0% > 55.0% -- something is already resident"
    snap = ResourceSnapshot(cpu_percent=5.0, memory_percent=30.0, gpu_percent=0.0,
                            gpu_memory_percent=82.0, memory_admission=(False, reason))
    assert snap.is_training_allowed() == (False, reason)


def test_guard_admission_is_the_snapshot_admission() -> None:
    snap = ResourceSnapshot(cpu_percent=5.0, memory_percent=30.0, gpu_percent=2.0,
                            gpu_memory_percent=4.0,
                            memory_admission=(True, "resources available"))
    assert snap.is_training_allowed() == (True, "resources available")


def test_no_guard_verdict_refuses() -> None:
    """Unknown is not free: a snapshot whose admission was never taken refuses.
    (A CPU-only host still trains -- the guard itself admits when it can read
    host memory and there is no card; only an UNTAKEN verdict refuses.)"""
    snap = ResourceSnapshot(gpu_percent=None, gpu_memory_percent=None)
    allowed, reason = snap.is_training_allowed()
    assert allowed is False and "blind" in reason


def test_scheduler_carries_no_vram_threshold_of_its_own() -> None:
    assert not hasattr(SchedulerConfig, "GPU_MEMORY_THRESHOLD")


def test_memory_admission_delegates_to_the_guard(monkeypatch) -> None:
    class Guard:
        @staticmethod
        def check_admission():
            return type("A", (), {"allowed": False, "reason": "occupied"})()
    monkeypatch.setattr(scheduler, "_load_memory_guard", lambda: Guard)
    assert asyncio.run(scheduler.memory_admission()) == (False, "occupied")
    monkeypatch.setattr(scheduler, "_load_memory_guard", lambda: None)
    assert asyncio.run(scheduler.memory_admission()) is None


def test_utilization_gate_still_independent() -> None:
    """A busy-but-empty card is still refused by the older gate."""
    snap = ResourceSnapshot(gpu_percent=99.0, gpu_memory_percent=1.0,
                            memory_admission=(True, "resources available"))
    allowed, reason = snap.is_training_allowed()
    assert allowed is False
    assert "GPU usage too high" in reason


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


class _FakeRun:
    """What ``subprocess.run`` returns to the shared probe."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _guard():
    """The single memory reader the scheduler now delegates to.

    ``_get_gpu_metrics`` no longer shells out itself: the "occupancy, not
    bandwidth" decision lives in ``training.memory_guard`` and is shared
    with the GRPO runner's admission gate, so there is one implementation
    to keep right rather than two that can drift back apart. These tests
    therefore patch the probe's subprocess seam; the assertions are
    unchanged, because the contract is.
    """
    probe = scheduler._load_memory_guard()
    assert probe is not None, "memory_guard must be loadable"
    return probe


@pytest.mark.asyncio
async def test_metrics_report_occupancy_not_bandwidth(monkeypatch) -> None:
    """nvidia-smi is asked for memory.used/memory.total, and the ratio is
    what reaches the snapshot."""
    probe = _guard()
    seen_args = {}

    def _fake_run(args, **kwargs):
        seen_args["args"] = args
        return _FakeRun("0, 29078, 32607\n")

    monkeypatch.setattr(probe.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run", _fake_run)
    monitor = scheduler.ResourceMonitor()
    gpu_util, vram_pct = await monitor._get_gpu_metrics()

    assert "--query-gpu=utilization.gpu,memory.used,memory.total" in seen_args["args"]
    assert gpu_util == pytest.approx(0.0)
    assert vram_pct == pytest.approx(89.18, abs=0.05)


@pytest.mark.asyncio
async def test_zero_total_memory_does_not_divide_by_zero(monkeypatch) -> None:
    probe = _guard()
    monkeypatch.setattr(probe.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda *a, **k: _FakeRun("0, 0, 0\n"))
    monitor = scheduler.ResourceMonitor()
    _gpu, vram_pct = await monitor._get_gpu_metrics()
    assert vram_pct is None


@pytest.mark.asyncio
async def test_missing_nvidia_smi_degrades_to_none(monkeypatch) -> None:
    probe = _guard()
    monkeypatch.setattr(probe.shutil, "which", lambda _n: None)
    monitor = scheduler.ResourceMonitor()
    assert await monitor._get_gpu_metrics() == (None, None)


@pytest.mark.asyncio
async def test_a_raising_probe_degrades_to_none(monkeypatch) -> None:
    """A wedged nvidia-smi must not stall the monitor, and must not be
    reported as an idle card."""
    probe = _guard()
    monkeypatch.setattr(probe.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")

    def _boom(*_a, **_k):
        raise OSError("nvidia-smi went away")

    monkeypatch.setattr(probe.subprocess, "run", _boom)
    monitor = scheduler.ResourceMonitor()
    assert await monitor._get_gpu_metrics() == (None, None)


def test_scheduler_no_longer_carries_its_own_nvidia_smi_call() -> None:
    """DRY, enforced: a second parser here is how the metric drifted back
    to ``utilization.memory`` the first time.

    Comments and docstrings are stripped before the check — the surviving
    prose *explains* the distinction and must stay readable, while a
    literal ``--query-gpu`` in executable code means the duplication is
    back.
    """
    import io
    import tokenize

    source = _SCHEDULER_PATH.read_text(encoding="utf-8")
    code_tokens = [
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.STRING)
    ]
    code = " ".join(code_tokens)
    assert "--query-gpu" not in code
    assert "utilization.memory" not in code
