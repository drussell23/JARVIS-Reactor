"""The trainability gate must answer from a venv that has no torch.

That is the entire reason `grpo_preflight.py` is a COMMAND and not an
import. JARVIS and Reactor-Core are separate repositories with separate
virtualenvs; the soak-side one has no ML stack, and `reactor_core/__init__`
imports torch eagerly at module scope.

The isolation used to rest on a promise: `_load` read each training module
from a bare file spec, which worked only while every one of them was
stdlib-only at its top level. Nothing enforced that. `grpo_pipeline` later
gained `from reactor_core.training.prompt_budget import ...` at module
scope, Python resolved it through the REAL package, and the gate began
dying with `No module named 'reactor_core'` -- reporting an ERROR where its
whole job is to say trainable or not. A gate that cannot run is worse than
no gate: the caller reads a fault where it expected a verdict.

So these tests do not check that the modules are written a particular way.
They check the property that matters -- the gate runs, and the ML stack is
never imported -- in a way that a future absolute sibling import cannot
quietly break.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_GATE = _REPO / "scripts" / "grpo_preflight.py"

#: What must never be imported. `reactor_core/__init__` pulls all of it in,
#: and a venv without them is the ONLY venv this gate is promised to run in.
FORBIDDEN = ("torch", "peft", "trl", "transformers", "bitsandbytes")

#: Exit codes the gate documents. 2 is "I looked and the answer is no",
#: which must stay distinguishable from 1, "I broke".
EXIT_TRAINABLE, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


def _run(args, *, cwd=None):
    return subprocess.run(
        [sys.executable, str(_GATE), *args],
        capture_output=True, text=True, cwd=str(cwd or _REPO), timeout=180,
    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_the_gate_answers_on_an_empty_corpus(tmp_path) -> None:
    """No rows is a legitimate answer, not a fault. This is the exact
    invocation that regressed: before the fix it printed
    `{"error": "ModuleNotFoundError: No module named 'reactor_core'"}`."""
    proc = _run(["--telemetry-dir", str(tmp_path)])
    payload = json.loads(proc.stdout)
    assert "error" not in payload, payload.get("error")
    assert payload["rows"] == 0
    assert proc.returncode == EXIT_REFUSED, (
        "an empty corpus is a REFUSAL (2), never an error (1)")


def test_a_real_group_is_read_and_reported(tmp_path) -> None:
    prompt = "Fix the timezone handling in backend/api/clock.py"
    rows = [
        {
            "event_type": "interaction",
            "user_input": prompt,
            "assistant_output": f"def now():\n    return {i}\n",
            "metadata": {
                "op_id": "op-1", "candidate_hash": f"h{i}",
                "draw_kind": "primary" if i == 0 else "sibling",
                "should_train": True, "attempt_index": i,
            },
        }
        for i in range(2)
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    proc = _run(["--telemetry-dir", str(tmp_path)])
    payload = json.loads(proc.stdout)
    assert "error" not in payload, payload.get("error")
    assert payload["rows"] == 2
    assert payload["prompts"] == 1


def test_the_ml_stack_is_never_imported(tmp_path) -> None:
    """The property, asserted directly rather than inferred from how the
    modules happen to be written today.

    A stub package that raises on import stands in for every forbidden
    module. If the gate reaches for any of them the run fails loudly, and
    it fails HERE rather than months later on a machine that has no GPU.
    """
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name in FORBIDDEN:
        (stubs / f"{name}.py").write_text(
            f"raise ImportError('{name} must never be imported by the gate')",
            encoding="utf-8",
        )
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    import os
    env = dict(os.environ)
    # PREPENDED, so the stub wins over a real install. On a machine that
    # HAS torch the unstubbed run would pass while proving nothing.
    env["PYTHONPATH"] = str(stubs) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_GATE), "--telemetry-dir", str(corpus)],
        capture_output=True, text=True, cwd=str(_REPO), env=env, timeout=180,
    )
    payload = json.loads(proc.stdout)
    assert "error" not in payload, (
        f"the gate reached for the ML stack: {payload.get('error')}")
    assert proc.returncode == EXIT_REFUSED


def test_every_loaded_module_survives_the_stubs() -> None:
    """`_load` is the seam, so exercise it on every module the gate uses.

    An import that only fires on a populated corpus would slip past a test
    that runs on an empty one.
    """
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import grpo_preflight as pf
    finally:
        sys.path.pop(0)
    for name in ("grpo_pipeline", "grpo_verifier", "grpo_reward"):
        mod = pf._load(name)
        assert mod is not None
        assert getattr(mod, "__name__", "") == f"reactor_core.training.{name}"


def test_a_module_has_ONE_identity() -> None:
    """The previous loader registered modules under private `_pf_` names.
    A module imported both ways would exist twice, and a dataclass or
    isinstance check spanning the two copies would compare unrelated
    types -- silently, and only in the venv that had both."""
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import grpo_preflight as pf
    finally:
        sys.path.pop(0)
    first = pf._load("grpo_pipeline")
    second = pf._load("grpo_pipeline")
    assert first is second


def test_a_real_reactor_core_is_left_alone() -> None:
    """In the trainer venv the package imports normally. The gate must not
    shadow it with its lightweight stand-in."""
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import grpo_preflight as pf
    finally:
        sys.path.pop(0)
    import types
    real = types.ModuleType("reactor_core")
    real.__path__ = []          # type: ignore[attr-defined]
    real.SENTINEL = object()    # type: ignore[attr-defined]
    saved = sys.modules.get("reactor_core")
    sys.modules["reactor_core"] = real
    try:
        pf._install_light_packages()
        assert sys.modules["reactor_core"] is real
    finally:
        if saved is None:
            sys.modules.pop("reactor_core", None)
        else:
            sys.modules["reactor_core"] = saved


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_a_missing_module_is_an_ImportError_not_a_silent_None() -> None:
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import grpo_preflight as pf
    finally:
        sys.path.pop(0)
    with pytest.raises(ImportError):
        pf._load("no_such_training_module")


def test_a_torn_line_does_not_stop_the_read(tmp_path) -> None:
    """A harvest against a LIVE soak can catch the row being appended."""
    good = {
        "event_type": "interaction", "user_input": "p",
        "assistant_output": "x = 1\n",
        "metadata": {"op_id": "o", "candidate_hash": "h", "should_train": True},
    }
    (tmp_path / "e.jsonl").write_text(
        json.dumps(good) + "\n{\"event_type\": \"inter", encoding="utf-8")
    proc = _run(["--telemetry-dir", str(tmp_path)])
    payload = json.loads(proc.stdout)
    assert "error" not in payload, payload.get("error")
    assert payload["rows"] == 1
