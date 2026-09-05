"""The profiler measures the run the runner will actually launch.

A memory profile is evidence about one configuration. If the profiler and
the runner disagree about that configuration, the profile is not a weaker
answer -- it is an answer to a different question, presented as if it were
the right one.

The divergence this pins had already opened: ``profile_grpo_vram`` took
``--num-generations`` default 4 while ``run_grpo_training`` launched at
``default_num_generations()`` = 16. ``num_generations`` is the dimension
the rollout multiplies (every completion in a group is generated AND
backpropagated), so that understates the peak by the largest factor in
the measurement -- and in the reassuring direction. The observable result
is a clean profile followed by an OOM on the real step, which is the
precise false negative the profiler exists to prevent.

Both scripts are loaded by path because each is stdlib-only at module
scope, so collecting this file costs nothing. Note that RESOLVING the
default does import ``reactor_core.training.grpo_pipeline``, and
``reactor_core/__init__`` pulls the ML stack in behind it -- that import
is the behaviour under test, not an accident of the harness.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


profiler = _load("_profiler_under_test", _REPO / "scripts" / "profile_grpo_vram.py")
runner = _load("_runner_defaults_under_test", _REPO / "scripts" / "run_grpo_training.py")
pipeline = _load(
    "_pipeline_defaults_under_test",
    _REPO / "reactor_core" / "training" / "grpo_pipeline.py",
)


def test_profiler_group_size_matches_the_runner() -> None:
    """The two must not drift; 4-vs-16 is how the false negative happened."""
    assert (profiler._default_num_generations()
            == runner._default_num_generations())


def test_profiler_group_size_comes_from_the_pipeline() -> None:
    """One source of truth, and it is not a literal in either script."""
    assert (profiler._default_num_generations()
            == pipeline.default_num_generations())


def test_group_size_default_is_the_documented_sixteen() -> None:
    """Pins the value itself.

    The equality tests above would still pass if all three collapsed to a
    shared wrong number, so the constant is asserted once, here.
    """
    assert pipeline.DEFAULT_NUM_GENERATIONS == 16
    assert profiler._default_num_generations() == 16


def test_group_size_is_env_overridable_everywhere(monkeypatch) -> None:
    """A rung override must reach the profiler too, or it cannot measure one."""
    monkeypatch.setenv("REACTOR_GRPO_NUM_GENERATIONS", "6")
    assert pipeline.default_num_generations() == 6
    assert profiler._default_num_generations() == 6


def test_profiler_step_geometry_is_the_runners() -> None:
    """One sequence per micro-step, accumulation from the shared helper.

    The collapsed geometry (per_device = group, accum = 1) measured a
    forward the runner never runs -- ~20 GiB of checkpointed inputs for
    16 x 6.3k-token sequences -- and OOM'd the 30B at 2026-09-04 23:30
    while the real launch shape would not have.
    """
    guard = _load(
        "_guard_for_geometry",
        _REPO / "reactor_core" / "training" / "memory_guard.py",
    )
    for n in (4, 8, 16):
        per_device, accum, prompts = profiler._step_geometry(guard, n)
        assert per_device == 1
        assert accum == guard.accumulation_for_groups(
            n, per_device_batch=1, requested_accum=8, device_count=1)
        assert (per_device * accum) % n == 0, "whole groups, as TRL demands"
        assert prompts == (per_device * accum) // n
    assert profiler.RUNNER_REQUESTED_ACCUM == 8


def test_runner_default_accumulation_matches_the_profilers_mirror() -> None:
    """Read the runner's parser rather than trusting the constant."""
    src = (_REPO / "scripts" / "run_grpo_training.py").read_text(encoding="utf-8")
    import re  # noqa: PLC0415
    m = re.search(r'"--gradient-accumulation-steps",\s*type=int,\s*default=(\d+)', src)
    assert m is not None, "the runner's accumulation default moved"
    assert int(m.group(1)) == profiler.RUNNER_REQUESTED_ACCUM


def test_degrades_instead_of_breaking_help(monkeypatch) -> None:
    """A missing training extra must not make ``--help`` unavailable.

    argparse evaluates this default at parse time, so an exception here
    would take out the whole CLI -- including the invocation someone runs
    precisely because their environment is broken.
    """
    import builtins

    real_import = builtins.__import__

    def _no_pipeline(name, *a, **k):
        if "grpo_pipeline" in name:
            raise ImportError("training extra not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_pipeline)
    assert profiler._default_num_generations() == 8
