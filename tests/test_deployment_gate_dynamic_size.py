"""The deployment gate's size ceiling must fit the machine, not a constant.

``DEFAULT_MAX_FILE_SIZE`` was a hardcoded 20 GB. That silently became a
policy rather than a sanity check the moment the target card grew: a
Q5/Q6 quantization of a 27B lands around 20-22 GB and would be REJECTED
by the gate on a 32 GB card with ~10 GB of measured headroom -- the gate
refusing quality the hardware can actually serve.

What the gate deliberately does NOT do is compute a KV-cache budget. KV
size depends on the context window, which is chosen at serve time by the
num_ctx negotiator. That is the negotiator's math, and a second copy here
would be free to drift from it.

Loaded by path: reactor_core/__init__ eagerly imports the training stack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GIB = 1024 ** 3


def _load(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vram = _load("_vram_uut", "reactor_core/deployment/vram.py")
gate = _load("_gate_uut", "reactor_core/deployment/gate.py")


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "REACTOR_GATE_MAX_FILE_SIZE_BYTES",
        "REACTOR_GATE_VRAM_WEIGHT_FRACTION",
        "REACTOR_VRAM_TOTAL_MIB",
    ):
        monkeypatch.delenv(k, raising=False)


def _fake_vram(monkeypatch: pytest.MonkeyPatch, total_bytes) -> None:
    """Patch the lazily-imported probe at its source module."""
    fake = type(sys)("reactor_core.deployment.vram")
    fake.total_vram_bytes = lambda: total_bytes  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reactor_core.deployment.vram", fake)


# ---------------------------------------------------------------------------
# Ceiling resolution
# ---------------------------------------------------------------------------


def test_ceiling_scales_to_measured_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """32,607 MiB is 31.84 GiB, so 0.75 of it is 23.88 GiB."""
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    size, provenance = gate.resolve_max_file_size()
    assert provenance == "measured_vram"
    assert 23.5 * _GIB < size < 24.5 * _GIB


def test_unprobeable_gpu_falls_back_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown capacity must not read as an unlimited one."""
    _fake_vram(monkeypatch, None)
    size, provenance = gate.resolve_max_file_size()
    assert size == gate.DEFAULT_MAX_FILE_SIZE
    assert provenance == "static_fallback_unprobeable_gpu"


def test_operator_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    monkeypatch.setenv("REACTOR_GATE_MAX_FILE_SIZE_BYTES", str(40 * _GIB))
    size, provenance = gate.resolve_max_file_size()
    assert size == 40 * _GIB
    assert provenance == "operator_override"


def test_garbage_override_is_ignored_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    monkeypatch.setenv("REACTOR_GATE_MAX_FILE_SIZE_BYTES", "twenty gigs")
    size, provenance = gate.resolve_max_file_size()
    assert provenance == "measured_vram"
    assert size > 0


@pytest.mark.parametrize(
    ("fraction", "expect_lo", "expect_hi"),
    [("0.5", 15.5, 16.5), ("0.9", 28.0, 29.0)],
)
def test_fraction_is_tunable(
    monkeypatch: pytest.MonkeyPatch, fraction: str, expect_lo: float,
    expect_hi: float,
) -> None:
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    monkeypatch.setenv("REACTOR_GATE_VRAM_WEIGHT_FRACTION", fraction)
    size, _ = gate.resolve_max_file_size()
    assert expect_lo * _GIB < size < expect_hi * _GIB


@pytest.mark.parametrize("bad", ["0", "-1", "5", "abc"])
def test_fraction_is_clamped(
    monkeypatch: pytest.MonkeyPatch, bad: str,
) -> None:
    """A fraction outside (0,1) would make the gate meaningless."""
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    monkeypatch.setenv("REACTOR_GATE_VRAM_WEIGHT_FRACTION", bad)
    size, _ = gate.resolve_max_file_size()
    assert 0 < size <= int(0.95 * 32_607 * 1024 * 1024)


# ---------------------------------------------------------------------------
# The behaviour that motivated the change
# ---------------------------------------------------------------------------


def test_q6_27b_now_admitted_where_the_static_cap_refused_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q6_27b = int(22.0 * _GIB)
    assert q6_27b > gate.DEFAULT_MAX_FILE_SIZE, "premise: static cap refused it"

    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    size, _ = gate.resolve_max_file_size()
    assert q6_27b < size, "a 32GB card should serve a Q6 27B"


def test_q8_27b_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """~29GB of weights leaves no room for a KV cache on this card."""
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    size, _ = gate.resolve_max_file_size()
    assert int(29.0 * _GIB) > size


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_gate_defaults_to_the_dynamic_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    g = gate.DeploymentGate()
    assert g.max_file_size_provenance == "measured_vram"
    assert 23.5 * _GIB < g.max_file_size_bytes < 24.5 * _GIB


def test_explicit_value_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deploy aimed at different hardware must stay caller-controlled."""
    _fake_vram(monkeypatch, 32_607 * 1024 * 1024)
    g = gate.DeploymentGate(max_file_size_bytes=8 * _GIB)
    assert g.max_file_size_bytes == 8 * _GIB
    assert g.max_file_size_provenance == "caller_supplied"


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_probe_override_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_VRAM_TOTAL_MIB", "16384")
    assert vram.total_vram_bytes() == 16384 * 1024 * 1024


def test_probe_reports_none_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vram.shutil, "which", lambda _: None)
    assert vram.total_vram_bytes() is None
    assert vram.free_vram_bytes() is None
    assert vram.used_fraction() is None
