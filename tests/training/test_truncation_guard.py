"""A step whose loss has no tokens in it must not run for hours.

With ``mask_truncated_completions=True`` TRL zeroes the completion mask for
every row whose last token is not EOS or pad, and ``num_items_in_batch`` is
that mask's sum. Measured 2026-09-05 on Qwen3-Coder-30B:
``completions/clipped_ratio`` was 1.0 at BOTH 8 and 256 completion tokens
and ``mean_terminated_length`` was 0, so every row was masked and the
policy loss had zero contributing tokens.

The reason it went unnoticed for three profile runs is the important part:
the step still logged a plausible ``loss`` of 0.008183, which is exactly
``aux_loss * 1e-3`` -- the MoE router's auxiliary term, which survives
masking. A confident number attached to no learning.

The guard reads the ratio TRL already publishes rather than re-deriving
truncation, so there is one definition of truncated and it belongs to the
trainer.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("_runner_trunc_under_test", _REPO / "scripts" / "run_grpo_training.py")


def _ctl():
    return SimpleNamespace(should_training_stop=False)


def _args(max_completion: int = 512):
    return SimpleNamespace(max_completion_length=max_completion,
                           mask_truncated_completions=True)


def _state(step: int = 1):
    return SimpleNamespace(global_step=step)


def _feed(cb, ratios, args=None):
    ctl = _ctl()
    args = args or _args()
    for i, r in enumerate(ratios, start=1):
        cb.on_log(args, _state(i), ctl, logs={"completions/clipped_ratio": r})
    return ctl


def test_a_fully_clipped_run_is_stopped_after_patience() -> None:
    cb = runner._make_truncation_callback(mask_truncated=True, patience=2)
    ctl = _feed(cb, [1.0])
    assert ctl.should_training_stop is False, "one step is not a verdict"
    assert cb.tripped is None
    ctl = _feed(cb, [1.0])
    assert ctl.should_training_stop is True
    assert "policy gradient is empty" in cb.tripped
    assert "512" in cb.tripped, "the message must name the ceiling to raise"


def test_any_terminating_completion_clears_the_counter() -> None:
    """One row that reaches EOS means the loss has tokens in it."""
    cb = runner._make_truncation_callback(mask_truncated=True, patience=2)
    ctl = _feed(cb, [1.0, 0.9375, 1.0])   # 15/16 clipped in the middle step
    assert ctl.should_training_stop is False
    assert cb.tripped is None
    assert cb.consecutive == 1


def test_the_guard_is_inert_when_truncation_is_not_masked() -> None:
    """With masking off a clipped completion still carries gradient, so a
    full clip ratio is a budget observation, not a stall."""
    cb = runner._make_truncation_callback(mask_truncated=False, patience=1)
    ctl = _feed(cb, [1.0, 1.0, 1.0])
    assert ctl.should_training_stop is False
    assert cb.tripped is None


def test_a_healthy_run_never_trips() -> None:
    cb = runner._make_truncation_callback(mask_truncated=True, patience=2)
    ctl = _feed(cb, [0.0, 0.25, 0.5, 0.75, 0.9])
    assert ctl.should_training_stop is False
    assert cb.last_ratio == pytest.approx(0.9)


def test_it_stops_once_not_once_per_step() -> None:
    cb = runner._make_truncation_callback(mask_truncated=True, patience=2)
    _feed(cb, [1.0, 1.0])
    first = cb.tripped
    _feed(cb, [1.0, 1.0])
    assert cb.tripped == first, "the reason is recorded once, not overwritten"


def test_missing_or_unparseable_metrics_are_ignored() -> None:
    """A log line without the metric must not be read as healthy OR as a
    stall -- it carries no information about truncation."""
    cb = runner._make_truncation_callback(mask_truncated=True, patience=1)
    ctl = _ctl()
    cb.on_log(_args(), _state(), ctl, logs={"loss": 0.5})
    cb.on_log(_args(), _state(), ctl, logs={"completions/clipped_ratio": None})
    cb.on_log(_args(), _state(), ctl, logs={"completions/clipped_ratio": "n/a"})
    cb.on_log(_args(), _state(), ctl, logs=None)
    assert ctl.should_training_stop is False
    assert cb.consecutive == 0


def test_patience_is_env_tunable(monkeypatch) -> None:
    monkeypatch.setenv("REACTOR_TRAIN_CLIPPED_PATIENCE", "1")
    cb = runner._make_truncation_callback(mask_truncated=True)
    ctl = _feed(cb, [1.0])
    assert ctl.should_training_stop is True


def test_garbage_patience_env_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv("REACTOR_TRAIN_CLIPPED_PATIENCE", "not-a-number")
    assert runner._env_int("REACTOR_TRAIN_CLIPPED_PATIENCE",
                           runner.DEFAULT_CLIPPED_PATIENCE) == 2


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------


def test_the_ladder_arms_the_guard_from_the_trainers_own_config() -> None:
    """Never a second opinion about the config the trainer was built with."""
    import inspect
    src = inspect.getsource(runner.train_with_ladder)
    assert "_make_truncation_callback(" in src
    assert 'trainer.args, "mask_truncated_completions"' in src
    assert "trainer.add_callback(truncation)" in src
    assert 'attempt["truncation_tripped"]' in src


def test_the_completion_ceiling_default_is_512() -> None:
    """256 is the value that produced clipped_ratio 1.0 and a masked loss."""
    import argparse
    parser = None
    src = (_REPO / "scripts" / "run_grpo_training.py").read_text(encoding="utf-8")
    import re
    m = re.search(r'"--max-completion-length", type=int, default=(\d+)', src)
    assert m is not None and int(m.group(1)) == 512
