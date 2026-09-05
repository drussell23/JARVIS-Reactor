"""An adapter is published by LAYERING it over a base tag, never by merging.

Merging a LoRA into the base to make one deployable model needs the base in
bf16 -- 16 shards, ~57 GiB -- against a WSL guest capped at 47 GiB, and on
this host a host-RAM blowout is not a failed job, it is the desktop (the
commit-limit arc). Ollama applies the adapter at load time instead, so the
whole deployment costs the size of the adapter: ~27 MB for a LoRA over 192
attention projections.

Two adapter-specific preconditions, both fail CLOSED, because the failure
they prevent is silent:

* a missing base tag -- layering a LoRA over the wrong weights yields a
  model that loads and answers WRONGLY, worse than one that fails to load;
* a base that is not served -- ``ollama create`` would try to pull ~18 GB,
  turning a deploy into a download that competes for the card.

Everything else is the model path's machinery, asserted here to be the SAME
machinery: gate, GPU lease, rollback snapshot, create, verify-by-serving.
"""
from __future__ import annotations

import importlib.util
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List, Optional, Tuple

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.deployment.ollama_deployer import OllamaDeployer  # noqa: E402


class _FakeGate:
    def __init__(self, passed: bool = True) -> None:
        self.passed = passed
        self.seen: List[Path] = []

    async def validate(self, path: Path):
        self.seen.append(Path(path))
        return SimpleNamespace(passed=self.passed,
                               summary=lambda: "ok" if self.passed else "too small")


def _lease_factory(*, held: bool = True, backend: str = "file"):
    @asynccontextmanager
    async def _f(*, reason: str):
        yield SimpleNamespace(held=held, backend=backend,
                              reason=reason if held else "GPU busy")
    return _f


class _Recorder:
    """Records ollama invocations; serves a scripted sequence of tag lists."""

    def __init__(self, tag_sequence: Optional[List[List[str]]] = None,
                 rc_by_verb=None) -> None:
        self.calls: List[Tuple[str, ...]] = []
        self.modelfiles: List[str] = []
        self.rc_by_verb = rc_by_verb or {}
        self._tags_seq = list(tag_sequence or [[]])
        self._i = 0

    async def run(self, args, timeout_s):
        self.calls.append(tuple(args))
        if args[0] == "create":
            mf = Path(args[3])
            if mf.is_file():
                self.modelfiles.append(mf.read_text(encoding="utf-8"))
        rc = self.rc_by_verb.get(args[0], 0)
        return rc, "" if rc == 0 else f"boom rc={rc}"

    async def tags(self):
        out = self._tags_seq[min(self._i, len(self._tags_seq) - 1)]
        self._i += 1
        return list(out)

    @property
    def verbs(self) -> List[str]:
        return [c[0] for c in self.calls]


BASE = "qwen3-coder:30b"


def _deployer(rec, *, gate_ok=True, lease_held=True, **kw):
    d = OllamaDeployer(gate=_FakeGate(passed=gate_ok),
                       lease_factory=_lease_factory(held=lease_held), **kw)
    d._run = rec.run      # type: ignore[method-assign]
    d._tags = rec.tags    # type: ignore[method-assign]
    return d


@pytest.fixture()
def adapter(tmp_path: Path) -> Path:
    p = tmp_path / "ov-lora-20260905.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 4096)
    return p


# ---------------------------------------------------------------------------
# The Modelfile: pure, and the whole point
# ---------------------------------------------------------------------------


def test_the_modelfile_layers_the_adapter_over_a_TAG(adapter) -> None:
    mf = OllamaDeployer().build_adapter_modelfile(adapter, base_tag=BASE)
    lines = mf.strip().split("\n")
    assert lines[0] == f"FROM {BASE}", "FROM names a served tag, not a file"
    assert lines[1] == f"ADAPTER {adapter.resolve()}"
    assert any(l.startswith("PARAMETER num_ctx") for l in lines)
    assert "PARAMETER temperature 0" in lines


def test_the_base_weights_are_never_named_as_a_path(adapter) -> None:
    """The proof that nothing is merged: no filesystem path but the adapter."""
    mf = OllamaDeployer().build_adapter_modelfile(adapter, base_tag=BASE)
    paths = [w for w in mf.split() if "/" in w or "\\" in w]
    assert paths == [str(adapter.resolve())]


def test_a_draft_model_still_reaches_the_modelfile(adapter) -> None:
    mf = OllamaDeployer().build_adapter_modelfile(
        adapter, base_tag=BASE, draft_model="qwen2.5-coder:7b")
    assert "DRAFT qwen2.5-coder:7b" in mf


# ---------------------------------------------------------------------------
# Fail-closed preconditions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_base_tag_refuses(adapter, monkeypatch) -> None:
    monkeypatch.delenv("REACTOR_OLLAMA_BASE_TAG", raising=False)
    rec = _Recorder([[BASE]])
    res = await _deployer(rec).deploy_adapter(adapter)
    assert not res.ok and res.stage == "input"
    assert "REACTOR_OLLAMA_BASE_TAG" in res.reason
    assert rec.verbs == [], "nothing was run"


@pytest.mark.asyncio
async def test_an_unserved_base_refuses_rather_than_pulling(adapter) -> None:
    rec = _Recorder([["something-else:7b"]])
    res = await _deployer(rec).deploy_adapter(adapter, base_tag=BASE)
    assert not res.ok and res.stage == "base_missing"
    assert "not served" in res.reason
    assert "create" not in rec.verbs, "a deploy must not become an 18 GB download"


@pytest.mark.asyncio
async def test_a_missing_adapter_file_refuses(tmp_path) -> None:
    rec = _Recorder([[BASE]])
    res = await _deployer(rec).deploy_adapter(tmp_path / "nope.gguf", base_tag=BASE)
    assert not res.ok and res.stage == "input"
    assert "create" not in rec.verbs


@pytest.mark.asyncio
async def test_a_busy_card_defers(adapter) -> None:
    rec = _Recorder([[BASE], [BASE], [BASE]])
    res = await _deployer(rec, lease_held=False).deploy_adapter(adapter, base_tag=BASE)
    assert not res.ok and res.stage == "gpu_lease"
    assert "create" not in rec.verbs


@pytest.mark.asyncio
async def test_a_rejected_gate_never_publishes(adapter) -> None:
    rec = _Recorder([[BASE], [BASE], [BASE]])
    res = await _deployer(rec, gate_ok=False).deploy_adapter(adapter, base_tag=BASE)
    assert not res.ok and res.stage == "gate"
    assert "adapter" in res.reason
    assert "create" not in rec.verbs


# ---------------------------------------------------------------------------
# The landing: same machinery as a model deploy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_clean_adapter_deploy_runs_every_stage(adapter) -> None:
    tag = "qwen3-coder-ov:30b"
    rec = _Recorder([[BASE], [BASE, tag], [BASE, tag]])
    res = await _deployer(rec).deploy_adapter(adapter, base_tag=BASE, tag=tag)
    assert res.ok, res.reason
    assert res.stage == "complete" and res.tag == tag
    assert "gate:passed" in res.checks
    assert any(c.startswith("lease:") for c in res.checks)
    assert "create:ok" in res.checks
    assert "verify:served" in res.checks
    assert f"base:{BASE}" in res.checks
    assert "create" in rec.verbs
    assert rec.modelfiles and rec.modelfiles[0].startswith(f"FROM {BASE}")


@pytest.mark.asyncio
async def test_an_existing_tag_is_snapshotted_before_being_overwritten(adapter) -> None:
    tag = "qwen3-coder-ov:30b"
    rec = _Recorder([[BASE], [BASE, tag], [BASE, tag]])
    res = await _deployer(rec).deploy_adapter(adapter, base_tag=BASE, tag=tag)
    assert res.ok
    assert "cp" in rec.verbs, "the outgoing adapter tag must be recoverable"
    assert any(c.startswith("snapshot:") for c in res.checks)


@pytest.mark.asyncio
async def test_create_returning_zero_is_not_enough(adapter) -> None:
    """Exit 0 means a manifest was written, not that the tag is servable."""
    tag = "qwen3-coder-ov:30b"
    rec = _Recorder([[BASE], [BASE], [BASE]])   # tag never appears
    res = await _deployer(rec).deploy_adapter(adapter, base_tag=BASE, tag=tag)
    assert not res.ok and res.stage == "verify"


@pytest.mark.asyncio
async def test_a_failed_create_reports_the_stage(adapter) -> None:
    rec = _Recorder([[BASE], [BASE], [BASE]], rc_by_verb={"create": 1})
    res = await _deployer(rec).deploy_adapter(adapter, base_tag=BASE)
    assert not res.ok and res.stage == "create"


@pytest.mark.asyncio
async def test_a_rejected_draft_retries_without_it(adapter) -> None:
    """Speculative decoding is an optimisation; losing it must not lose the
    deployment. Same fallback the model path has, exercised on an adapter."""
    tag = "qwen3-coder-ov:30b"
    calls = {"n": 0}

    class _Rec(_Recorder):
        async def run(self, args, timeout_s):
            await super().run(args, timeout_s)
            if args[0] == "create":
                calls["n"] += 1
                if calls["n"] == 1:
                    return 1, "unknown instruction DRAFT"
            return 0, ""

    rec = _Rec([[BASE], [BASE, tag], [BASE, tag]])
    d = _deployer(rec, draft_model="qwen2.5-coder:7b")
    res = await d.deploy_adapter(adapter, base_tag=BASE, tag=tag)
    assert res.ok, res.reason
    assert any(c.startswith("draft:rejected:") for c in res.checks)
    assert len(rec.modelfiles) == 2
    assert "DRAFT" in rec.modelfiles[0] and "DRAFT" not in rec.modelfiles[1]
    assert rec.modelfiles[1].startswith(f"FROM {BASE}"), "still an adapter deploy"


# ---------------------------------------------------------------------------
# Size floor: an adapter is 1000x smaller than the model the floor was for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_gate_floor_suits_an_adapter_not_a_model(adapter, monkeypatch) -> None:
    """DEFAULT_MIN_FILE_SIZE (100 MB) catches a truncated 18 GB model and
    would reject every healthy 27 MB adapter."""
    seen = {}

    class _Gate:
        def __init__(self, **kw):
            seen.update(kw)
            self.passed = True

        async def validate(self, path):
            return SimpleNamespace(passed=True, summary=lambda: "ok")

    import reactor_core.deployment.gate as gate_mod
    monkeypatch.setattr(gate_mod, "DeploymentGate", _Gate)
    tag = "qwen3-coder-ov:30b"
    rec = _Recorder([[BASE], [BASE, tag], [BASE, tag]])
    d = OllamaDeployer(lease_factory=_lease_factory(held=True))
    d._run = rec.run       # type: ignore[method-assign]
    d._tags = rec.tags     # type: ignore[method-assign]
    res = await d.deploy_adapter(adapter, base_tag=BASE, tag=tag)
    assert res.ok, res.reason
    assert seen.get("min_file_size_bytes", 0) < 100 * 1024 * 1024
    assert seen.get("min_file_size_bytes", 0) > 0, "still catches an empty file"


@pytest.mark.asyncio
async def test_the_adapter_floor_is_configurable(adapter, monkeypatch) -> None:
    seen = {}

    class _Gate:
        def __init__(self, **kw):
            seen.update(kw)

        async def validate(self, path):
            return SimpleNamespace(passed=True, summary=lambda: "ok")

    import reactor_core.deployment.gate as gate_mod
    monkeypatch.setattr(gate_mod, "DeploymentGate", _Gate)
    monkeypatch.setenv("REACTOR_GATE_ADAPTER_MIN_FILE_SIZE_BYTES", "999")
    tag = "t:1"
    rec = _Recorder([[BASE], [BASE, tag], [BASE, tag]])
    d = OllamaDeployer(lease_factory=_lease_factory(held=True))
    d._run = rec.run       # type: ignore[method-assign]
    d._tags = rec.tags     # type: ignore[method-assign]
    await d.deploy_adapter(adapter, base_tag=BASE, tag=tag)
    assert seen.get("min_file_size_bytes") == 999


@pytest.mark.asyncio
async def test_a_model_deploy_keeps_the_model_floor(tmp_path, monkeypatch) -> None:
    """The adapter floor must not leak onto the model path."""
    seen = {}

    class _Gate:
        def __init__(self, **kw):
            seen.update(kw)

        async def validate(self, path):
            return SimpleNamespace(passed=True, summary=lambda: "ok")

    import reactor_core.deployment.gate as gate_mod
    monkeypatch.setattr(gate_mod, "DeploymentGate", _Gate)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    rec = _Recorder([["jprime-latest"], ["jprime-latest"]])
    d = OllamaDeployer(lease_factory=_lease_factory(held=True))
    d._run = rec.run       # type: ignore[method-assign]
    d._tags = rec.tags     # type: ignore[method-assign]
    await d.deploy(gguf)
    assert "min_file_size_bytes" not in seen, "the model path keeps the gate's own default"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_both_paths_share_one_deploy_implementation() -> None:
    import inspect
    for name in ("deploy", "deploy_adapter"):
        src = inspect.getsource(getattr(OllamaDeployer, name))
        assert "self._deploy(" in src, f"{name} must delegate, not duplicate"
    shared = inspect.getsource(OllamaDeployer._deploy)
    for stage in ("gate", "lease", "cp", "create", "verify"):
        assert stage in shared
