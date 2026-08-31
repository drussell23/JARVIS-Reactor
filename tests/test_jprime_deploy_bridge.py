"""GPU lease + Ollama deployment tail for the Trinity flywheel.

The properties under test are the ones that protect a live soak:

  * A lease that cannot be enforced across processes is NOT a lease.
    Reactor's own bridge falls back to an ``asyncio.Lock`` and returns
    True when Redis is down; anything built on that would collide with a
    soak holding 29 of 32.6 GiB. So an in-process backend must read as
    NOT held.
  * No ollama command may run before the gate passes AND the lease is
    held -- ``ollama create`` writes blobs and the verify probe loads the
    model, both of which contend for VRAM.
  * A deploy is complete when the tag is SERVED, not when the CLI exits 0.

Modules are loaded by path: ``reactor_core/__init__`` eagerly imports
torch/peft/trl, and the deployment tail must run on a serving-only box.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lease_mod = _load("_gpu_lease_uut", "reactor_core/deployment/gpu_lease.py")
deploy_mod = _load(
    "_ollama_deploy_uut", "reactor_core/deployment/ollama_deployer.py"
)

OllamaDeployer = deploy_mod.OllamaDeployer


# ===========================================================================
# GPU lease
# ===========================================================================


def _fake_bridge(
    *, acquired: bool, backend: str, token: int = 7, raises: bool = False
):
    """Stand-in for JARVIS's acquire_trinity_lock context manager."""

    @asynccontextmanager
    async def _acquire(name, repo="", timeout=0.0, ttl=0.0):
        if raises:
            raise RuntimeError("lock backend exploded")
        yield acquired, SimpleNamespace(backend=backend, fencing_token=token)

    return _acquire


@pytest.mark.asyncio
async def test_no_jarvis_repo_means_no_lease(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_REPO_PATH", raising=False)
    monkeypatch.delenv("TRINITY_GPU_LEASE_ALLOW_UNSAFE", raising=False)
    async with lease_mod.gpu_lease(reason="t") as lease:
        assert lease.held is False
        assert lease.backend == "unavailable"
        assert "JARVIS_REPO_PATH" in lease.reason


@pytest.mark.asyncio
async def test_unsafe_override_is_explicit_and_labelled(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_REPO_PATH", raising=False)
    monkeypatch.setenv("TRINITY_GPU_LEASE_ALLOW_UNSAFE", "1")
    async with lease_mod.gpu_lease(reason="t") as lease:
        assert lease.held is True
        # The backend name must never claim an exclusion it does not have.
        assert lease.backend == "none-unsafe-override"


@pytest.mark.asyncio
async def test_file_backend_grants(monkeypatch) -> None:
    monkeypatch.setattr(
        lease_mod, "_load_jarvis_bridge",
        lambda: _fake_bridge(acquired=True, backend="file", token=3),
    )
    async with lease_mod.gpu_lease(reason="qlora") as lease:
        assert lease.held is True
        assert lease.backend == "file"
        assert lease.fencing_token == 3
        assert bool(lease) is True


@pytest.mark.asyncio
async def test_contended_lock_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        lease_mod, "_load_jarvis_bridge",
        lambda: _fake_bridge(acquired=False, backend="file"),
    )
    async with lease_mod.gpu_lease(reason="qlora") as lease:
        assert lease.held is False
        assert "another Trinity process" in lease.reason


@pytest.mark.parametrize("backend", ["local", "asyncio", "memory", "inprocess"])
@pytest.mark.asyncio
async def test_in_process_backend_is_not_a_lease(monkeypatch, backend) -> None:
    """The defect this module exists to refuse: reactor's DistributedLock
    returns True from an asyncio.Lock when Redis is down."""
    monkeypatch.setattr(
        lease_mod, "_load_jarvis_bridge",
        lambda: _fake_bridge(acquired=True, backend=backend),
    )
    async with lease_mod.gpu_lease(reason="qlora") as lease:
        assert lease.held is False
        assert "cannot exclude another process" in lease.reason


@pytest.mark.asyncio
async def test_lock_error_refuses_rather_than_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        lease_mod, "_load_jarvis_bridge",
        lambda: _fake_bridge(acquired=True, backend="file", raises=True),
    )
    async with lease_mod.gpu_lease(reason="qlora") as lease:
        assert lease.held is False
        assert lease.backend == "error"


# ===========================================================================
# Ollama deployer
# ===========================================================================


@dataclass
class _FakeGate:
    passed: bool = True

    async def validate(self, path):
        return SimpleNamespace(
            passed=self.passed,
            summary=lambda: "APPROVED (2/2)" if self.passed
            else "REJECTED (1/2): Invalid GGUF magic",
        )


def _lease_factory(*, held: bool, backend: str = "file"):
    @asynccontextmanager
    async def _f(*, reason: str):
        yield lease_mod.LeaseVerdict(
            held=held, backend=backend,
            reason=reason if held else "GPU busy",
        )

    return _f


class _Recorder:
    """Captures ollama CLI invocations and scripts their results."""

    def __init__(self, rc_by_verb=None, tags_before=None, tags_after=None):
        self.calls: List[Tuple[str, ...]] = []
        self.rc_by_verb = rc_by_verb or {}
        self.tags_before = tags_before if tags_before is not None else []
        self.tags_after = tags_after if tags_after is not None else []
        self._tag_calls = 0
        self.modelfiles: List[str] = []

    async def run(self, args, timeout_s):
        self.calls.append(tuple(args))
        if args[0] == "create":
            mf = Path(args[3])
            if mf.is_file():
                self.modelfiles.append(mf.read_text(encoding="utf-8"))
        rc = self.rc_by_verb.get(args[0], 0)
        return rc, "" if rc == 0 else f"boom rc={rc}"

    async def tags(self):
        self._tag_calls += 1
        return self.tags_before if self._tag_calls == 1 else self.tags_after

    @property
    def verbs(self) -> List[str]:
        return [c[0] for c in self.calls]


def _deployer(rec: _Recorder, *, gate_ok=True, lease_held=True, **kw):
    d = OllamaDeployer(
        gate=_FakeGate(passed=gate_ok),
        lease_factory=_lease_factory(held=lease_held),
        **kw,
    )
    d._run = rec.run          # type: ignore[method-assign]
    d._tags = rec.tags        # type: ignore[method-assign]
    return d


@pytest.fixture()
def gguf(tmp_path: Path) -> Path:
    p = tmp_path / "model-q4_k_m.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 64)
    return p


@pytest.mark.asyncio
async def test_missing_gguf_is_refused(tmp_path: Path) -> None:
    rec = _Recorder()
    res = await _deployer(rec).deploy(tmp_path / "nope.gguf")
    assert res.ok is False
    assert res.stage == "input"
    assert rec.calls == []


@pytest.mark.asyncio
async def test_gate_rejection_runs_no_ollama_command(gguf: Path) -> None:
    rec = _Recorder()
    res = await _deployer(rec, gate_ok=False).deploy(gguf)
    assert res.ok is False
    assert res.stage == "gate"
    assert "REJECTED" in res.gate_summary
    assert rec.calls == [], "a rejected model must never reach ollama"


@pytest.mark.asyncio
async def test_unheld_lease_touches_nothing(gguf: Path) -> None:
    """The contention guarantee: no blob writes, no model load."""
    rec = _Recorder()
    res = await _deployer(rec, lease_held=False).deploy(gguf)
    assert res.ok is False
    assert res.stage == "gpu_lease"
    assert "deferring" in res.reason
    assert rec.calls == []


@pytest.mark.asyncio
async def test_happy_path_snapshots_creates_and_verifies(gguf: Path) -> None:
    rec = _Recorder(
        tags_before=["jprime-latest:latest", "qwen2.5-coder:32b"],
        tags_after=["jprime-latest:latest", "jprime-previous:latest"],
    )
    res = await _deployer(rec).deploy(gguf)

    assert res.ok is True, res.reason
    assert res.stage == "complete"
    # Snapshot must precede create, or a bad model is unrecoverable.
    assert rec.verbs == ["cp", "create"]
    assert rec.calls[0] == ("cp", "jprime-latest", "jprime-previous")
    assert "snapshot:jprime-previous" in res.checks
    assert "verify:served" in res.checks


@pytest.mark.asyncio
async def test_first_deploy_has_nothing_to_snapshot(gguf: Path) -> None:
    rec = _Recorder(
        tags_before=["qwen2.5-coder:32b"],
        tags_after=["jprime-latest:latest"],
    )
    res = await _deployer(rec).deploy(gguf)
    assert res.ok is True
    assert rec.verbs == ["create"]
    assert "snapshot:none-first-deploy" in res.checks


@pytest.mark.asyncio
async def test_create_failure_is_reported(gguf: Path) -> None:
    rec = _Recorder(rc_by_verb={"create": 1}, tags_before=[], tags_after=[])
    res = await _deployer(rec).deploy(gguf)
    assert res.ok is False
    assert res.stage == "create"
    assert "rc=1" in res.reason


@pytest.mark.asyncio
async def test_exit_zero_without_a_served_tag_is_a_failure(gguf: Path) -> None:
    """ollama create writes a manifest; only /api/tags proves servability."""
    rec = _Recorder(tags_before=[], tags_after=["qwen2.5-coder:32b"])
    res = await _deployer(rec).deploy(gguf)
    assert res.ok is False
    assert res.stage == "verify"
    assert "not in /api/tags" in res.reason


@pytest.mark.asyncio
async def test_modelfile_pins_native_context_and_zero_temperature(
    gguf: Path,
) -> None:
    rec = _Recorder(tags_before=[], tags_after=["jprime-latest:latest"])
    d = _deployer(rec)
    await d.deploy(gguf)

    assert len(rec.modelfiles) == 1
    mf = rec.modelfiles[0]
    assert f"FROM {gguf.resolve()}" in mf
    assert "PARAMETER num_ctx 32768" in mf
    assert "PARAMETER temperature 0" in mf


@pytest.mark.asyncio
async def test_num_ctx_is_configurable(gguf: Path) -> None:
    rec = _Recorder(tags_before=[], tags_after=["jprime-latest:latest"])
    d = _deployer(rec, num_ctx=8192)
    await d.deploy(gguf)
    assert "PARAMETER num_ctx 8192" in rec.modelfiles[0]


@pytest.mark.asyncio
async def test_rollback_without_snapshot_refuses(gguf: Path) -> None:
    rec = _Recorder(tags_before=["jprime-latest:latest"])
    res = await _deployer(rec).rollback()
    assert res.ok is False
    assert "no snapshot" in res.reason
    assert rec.calls == []


@pytest.mark.asyncio
async def test_rollback_restores_previous(gguf: Path) -> None:
    rec = _Recorder(
        tags_before=["jprime-latest:latest", "jprime-previous:latest"],
    )
    res = await _deployer(rec).rollback()
    assert res.ok is True
    assert rec.calls == [("cp", "jprime-previous", "jprime-latest")]


@pytest.mark.asyncio
async def test_rollback_respects_the_lease(gguf: Path) -> None:
    rec = _Recorder(
        tags_before=["jprime-latest:latest", "jprime-previous:latest"],
    )
    res = await _deployer(rec, lease_held=False).rollback()
    assert res.ok is False
    assert "deferring" in res.reason
    assert rec.calls == []
