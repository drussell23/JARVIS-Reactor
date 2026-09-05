"""Adapter placement is decided by the checkpoint, and never lands on experts.

``resolve_target_modules`` is load-bearing and was, until this file,
untested. Its own docstring says the router exclusion is "asserted here
so a future reader does not have to re-derive it" — but an assertion in
prose is not a regression test, and the failure it guards against is
silent and expensive rather than loud:

On a Qwen3 MoE the names ``gate_proj`` / ``up_proj`` / ``down_proj``
exist ONLY inside the experts, so the dense-model list that reads
naturally means "one adapter per projection per expert per layer" —
48 layers x 128 experts x 3 = 18,432 adapted projections against 192 for
attention. At r=16 that is ~830 M trainable parameters, whose bf16
adapter plus AdamW's three fp32 states is ~11.8 GiB ON TOP of the ~18 GiB
4-bit base, on a 32.6 GiB card, before a single activation.

The reason that must be pinned rather than trusted is the *shape* of the
failure: the cost is invisible to the rung ladder. The ladder relieves
memory by moving ``num_generations`` and ``max_completion_length``, and
adapter plus optimiser state is independent of both — so every rung OOMs
identically and the run exits ladder-exhausted having learned nothing.
A regression here does not surface as a bad number, it surfaces as a
wasted GPU-hour that looks like a hardware limit.

The module is loaded by path: it is stdlib-only at module scope, and
``reactor_core/__init__`` would drag the whole ML stack in behind it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "reactor_core" / "training" / "grpo_pipeline.py"
)

#: The real target's geometry, from the checkpoint's own config.
_QWEN3_MOE_LAYERS = 48
_QWEN3_MOE_EXPERTS = 128
_MLP_PROJECTIONS_PER_EXPERT = 3
_ATTENTION_PROJECTIONS_PER_LAYER = 4

EXPERT_PROJECTION_COUNT = (
    _QWEN3_MOE_LAYERS * _QWEN3_MOE_EXPERTS * _MLP_PROJECTIONS_PER_EXPERT
)
ATTENTION_PROJECTION_COUNT = _QWEN3_MOE_LAYERS * _ATTENTION_PROJECTIONS_PER_LAYER


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_pipeline_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gp = _load()

_EXPERT_NAMES = {"gate_proj", "up_proj", "down_proj"}
_ATTENTION_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    """Both escape hatches off.

    Without this the suite would pass or fail according to the developer's
    shell, and the pin would be worthless on the one machine that matters.
    """
    monkeypatch.delenv("REACTOR_GRPO_TARGET_MODULES", raising=False)
    monkeypatch.delenv("REACTOR_GRPO_ADAPT_EXPERTS", raising=False)


def _topology(monkeypatch, **kw) -> None:
    """Mock the CHECKPOINT, not the resolver.

    ``describe_expert_topology`` is the only thing standing between this
    function and a network round-trip through ``AutoConfig``; patching it
    keeps the test isolated while leaving every branch under test real.
    """
    base = {"is_moe": False, "num_experts": 0, "experts_per_tok": 0,
            "model_type": "", "known": True}
    base.update(kw)
    monkeypatch.setattr(gp, "describe_expert_topology", lambda _m: base)


# ---------------------------------------------------------------------------
# The MoE branch: the one that would OOM
# ---------------------------------------------------------------------------


def test_moe_excludes_every_expert_projection(monkeypatch) -> None:
    """The 18,432 must not be adapted. This is the whole point of the file."""
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS,
              experts_per_tok=8, model_type="qwen3_moe")
    chosen = gp.resolve_target_modules("Qwen/Qwen3-Coder-30B-A3B-Instruct")

    assert _EXPERT_NAMES.isdisjoint(chosen), (
        f"expert projections {sorted(_EXPERT_NAMES & set(chosen))} would "
        f"place adapters on {EXPERT_PROJECTION_COUNT} modules "
        f"(~830M params, ~11.8 GiB of optimiser state) instead of "
        f"{ATTENTION_PROJECTION_COUNT}"
    )


def test_moe_selects_exactly_the_attention_projections(monkeypatch) -> None:
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS,
              model_type="qwen3_moe")
    assert set(gp.resolve_target_modules("qwen3-moe")) == _ATTENTION_NAMES


def test_moe_detected_by_model_type_alone(monkeypatch) -> None:
    """A vendor that names no expert count is still MoE, not dense.

    ``is_moe`` is true when EITHER the count or the type says so; a
    checkpoint that only spells it in ``model_type`` must not fall
    through to the dense branch and pick up the expert projections.
    """
    _topology(monkeypatch, is_moe=True, num_experts=0, model_type="qwen3_moe")
    assert _EXPERT_NAMES.isdisjoint(gp.resolve_target_modules("m"))


def test_router_is_never_targeted(monkeypatch) -> None:
    """``mlp.gate`` is the router; adapting it destabilises expert choice.

    peft matches on the module SUFFIX and ``gate`` never suffix-matches
    ``gate_proj``, so the router is safe in both branches. Pinned in both
    directions because the guarantee is a property of the names chosen,
    and a future edit adding a bare ``gate`` would break it silently.
    """
    for moe in (True, False):
        _topology(monkeypatch, is_moe=moe, num_experts=128 if moe else 0)
        chosen = gp.resolve_target_modules("m")
        assert "gate" not in chosen
        assert "mlp.gate" not in chosen


# ---------------------------------------------------------------------------
# The dense branch still gets its MLP
# ---------------------------------------------------------------------------


def test_dense_model_keeps_the_mlp_projections(monkeypatch) -> None:
    """The exclusion is conditional, not a blanket ban.

    On a dense model those three names are one MLP per layer — cheap, and
    worth adapting. A fix that dropped them everywhere would quietly
    degrade every non-MoE run.
    """
    _topology(monkeypatch, is_moe=False, model_type="llama")
    chosen = gp.resolve_target_modules("meta-llama/Llama-3.2-3B")
    assert _ATTENTION_NAMES.issubset(chosen)
    assert _EXPERT_NAMES.issubset(chosen)


def test_unreadable_config_assumes_dense(monkeypatch) -> None:
    """An unknown architecture takes the documented branch.

    ``known=False`` means the config could not be read at all. The module
    warns and treats it as dense; pinned so the behaviour is a decision
    rather than an accident of control flow.
    """
    _topology(monkeypatch, is_moe=False, known=False)
    assert _EXPERT_NAMES.issubset(gp.resolve_target_modules("mystery-model"))


# ---------------------------------------------------------------------------
# The escape hatches, which must work AND must not be the default
# ---------------------------------------------------------------------------


def test_expert_adaptation_is_off_by_default(monkeypatch) -> None:
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS)
    assert _EXPERT_NAMES.isdisjoint(gp.resolve_target_modules("m"))


def test_expert_adaptation_can_be_opted_into(monkeypatch) -> None:
    """Deliberate opt-in still reaches the experts.

    The env flag is the difference between "this placement is wrong" and
    "this placement is unavailable"; someone with a bigger card must
    still be able to ask for it.
    """
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS)
    monkeypatch.setenv("REACTOR_GRPO_ADAPT_EXPERTS", "1")
    assert _EXPERT_NAMES.issubset(gp.resolve_target_modules("m"))


def test_kwarg_beats_the_env_flag(monkeypatch) -> None:
    """An explicit argument is not overridable by ambient state."""
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS)
    monkeypatch.setenv("REACTOR_GRPO_ADAPT_EXPERTS", "1")
    chosen = gp.resolve_target_modules("m", adapt_experts=False)
    assert _EXPERT_NAMES.isdisjoint(chosen)


def test_explicit_pin_short_circuits_detection(monkeypatch) -> None:
    """The pin is read before the checkpoint is consulted.

    It exists so an operator can override a misdetection without editing
    code; if detection ran first and won, the escape hatch would be a
    no-op exactly when it was needed.
    """
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS)
    monkeypatch.setenv("REACTOR_GRPO_TARGET_MODULES", "q_proj, v_proj")
    assert gp.resolve_target_modules("m") == ["q_proj", "v_proj"]


def test_blank_pin_is_ignored_not_obeyed(monkeypatch) -> None:
    """An empty env var must not resolve to an empty target list.

    ``REACTOR_GRPO_TARGET_MODULES=""`` is how a shell exports a variable
    it means to leave unset. Honouring it would attach zero adapters and
    train nothing while reporting success.
    """
    _topology(monkeypatch, is_moe=True, num_experts=_QWEN3_MOE_EXPERTS)
    monkeypatch.setenv("REACTOR_GRPO_TARGET_MODULES", "   ")
    assert set(gp.resolve_target_modules("m")) == _ATTENTION_NAMES
