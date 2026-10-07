"""The time budget is ENFORCED at a step boundary, not only predicted.

The step count is fitted from a calibrated estimate; a run whose steps prove
slower than measured would reach its caller's timeout mid-step and save
nothing (``save_strategy="no"``: the adapter is written after ``train()``
returns). Live 2026-10-07 the first step of handoff-20261007-081918 ran past
its 1129 s estimate, which is exactly the case this guards.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("_runner_deadline_under_test", _REPO / "scripts" / "run_grpo_training.py")


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _drive(cb, clock, durations, *, num_iterations=2):
    """Run steps of the given durations; return how many ran before a stop."""
    args = SimpleNamespace(num_iterations=num_iterations)
    control = SimpleNamespace(should_training_stop=False)
    for i, d in enumerate(durations, 1):
        cb.on_step_begin(args, SimpleNamespace(global_step=i - 1), control)
        clock.t += d
        cb.on_step_end(args, SimpleNamespace(global_step=i), control)
        if control.should_training_stop:
            return i
    return len(durations)


def test_stops_when_the_next_step_cannot_finish():
    clock = Clock()
    cb = runner._make_deadline_callback(1000.0, clock=clock)
    # 300 s steps: after step 3 (t=900) a 4th would end at 1200 > 1000.
    assert _drive(cb, clock, [300] * 10) == 3
    assert "past the deadline" in cb.tripped


def test_assumes_the_slow_kind_of_step_when_steps_alternate():
    # Generation is reused for num_iterations=2 steps: slow, fast, slow, fast.
    clock = Clock()
    cb = runner._make_deadline_callback(2000.0, clock=clock)
    ran = _drive(cb, clock, [900, 100, 900, 100], num_iterations=2)
    # After step 2 (t=1000) the next may be a 900 s one -> 1900 <= 2000, go.
    # After step 3 (t=1900) the slowest recent is 900 -> 2800 > 2000, stop.
    assert ran == 3


def test_a_run_inside_its_budget_is_untouched():
    clock = Clock()
    cb = runner._make_deadline_callback(10_000.0, clock=clock)
    assert _drive(cb, clock, [300] * 5) == 5 and cb.tripped is None


def test_the_deadline_reaches_every_rung_child_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--model", "m", "--time-budget-s", "100"])
    argv = runner.child_argv(0, "/tmp/r.json", {"--deadline-epoch": "1791400000.5"})
    assert argv[argv.index("--deadline-epoch") + 1] == "1791400000.5"
    assert argv.count("--deadline-epoch") == 1
