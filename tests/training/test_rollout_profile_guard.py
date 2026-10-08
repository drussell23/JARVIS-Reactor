"""The rollout profiler stops cleanly and keeps what it measured.

2026-10-07: the first rollout-profile run was killed from outside mid-load
(its launcher reaped it under host-memory pressure) and left nothing behind --
no report, and a GPU lease nobody returned. The measuring child (calibration
and profiling) was also the one GPU path in this runner with no memory
watchdog. Pinned here:

* the child arms the SAME ``MemoryWatchdog`` a rung runs under, before the
  model loads;
* a breach or a signal stops ``generate`` at the next token, the unfinished
  context is dropped (never reported from a partial decode), and every
  context already finished has been flushed to disk;
* SIGTERM becomes a stop request, not a death; a second one ends it;
* the report file is replaced atomically, so a kill mid-write never leaves
  half a JSON document.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "run_grpo_training.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("_runner_profile_guard_under_test", _SCRIPT)


class _Prof:
    """torch.profiler.profile stand-in: no CUDA, a fixed op table."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def key_averages(self):
        return [SimpleNamespace(key="aten::mm", count=1, self_cpu_time_total=10.0, self_device_time_total=5.0)]


class _Probe:
    """RolloutProbe stand-in. ``gen`` runs ``on_gen`` (which may trip the
    stop) and then honours the stop exactly as generate's criterion would."""

    on_gen = staticmethod(lambda length, new: None)

    def __init__(self, trainer):
        self.model, self.g = SimpleNamespace(), 16

    def gen(self, length, new, *, stop=None):
        return lambda: _Probe.on_gen(length, new)

    def prompts_summary(self):
        return {"rows": 3, "max": 900, "mean": 400}


@pytest.fixture
def harness(monkeypatch):
    import contextlib

    import torch.profiler as tp
    from reactor_core.training import grpo_pipeline

    monkeypatch.setattr(tp, "profile", _Prof)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(grpo_pipeline, "rollout_mode", lambda m: contextlib.nullcontext())
    monkeypatch.setattr(runner, "RolloutProbe", _Probe)
    monkeypatch.setattr(_Probe, "on_gen", staticmethod(lambda length, new: None))
    return monkeypatch


def test_an_unstopped_run_measures_every_context(harness):
    flushed = []
    out = runner.profile_rollout(None, contexts=[100, 200, 300], short=4, long=20, top=5,
                                 stop=lambda: None, sink=lambda o: flushed.append(sorted(o["contexts"])))
    assert sorted(out["contexts"]) == ["100", "200", "300"] and out["interrupted"] is None
    # flushed after EVERY completed context, not only at the end
    assert flushed == [["100"], ["100", "200"], ["100", "200", "300"]]


def test_a_breach_mid_decode_drops_that_context_and_keeps_the_rest(harness):
    state = {"reason": None}

    def on_gen(length, new):
        if length == 200 and new == 20:          # during the 2nd context's long decode
            state["reason"] = "host memory 3.9 GiB <= floor 4.0 GiB"
    harness.setattr(_Probe, "on_gen", staticmethod(on_gen))
    flushed = []
    out = runner.profile_rollout(None, contexts=[100, 200, 300], short=4, long=20, top=5,
                                 stop=lambda: state["reason"], sink=lambda o: flushed.append(dict(o)))
    assert sorted(out["contexts"]) == ["100"]                 # 200 never reported from a partial decode
    assert out["interrupted"].startswith("at context 200: host memory")
    assert len(flushed) == 1


def test_a_stop_before_anything_measures_nothing(harness):
    out = runner.profile_rollout(None, contexts=[100], short=4, long=20, top=5, stop=lambda: "signal SIGTERM")
    assert out["contexts"] == {} and "SIGTERM" in out["interrupted"]


def test_the_stop_reaches_inside_generate():
    """A decode at a long context runs for minutes; the stop is a stopping
    criterion so it lands at the next token, with one verdict per sequence."""
    probe = runner.RolloutProbe.__new__(runner.RolloutProbe)
    probe.torch, probe.g, probe.pad, probe._base_gc = torch, 2, 0, None
    probe._copy = lambda x: x
    seen = {}

    def fake_generate(ids, **kw):
        seen.update(kw)
        crit = kw["stopping_criteria"]
        return crit(torch.zeros((2, 3), dtype=torch.long), None)
    probe.model = SimpleNamespace(generate=fake_generate)
    probe.ids = lambda length, batch: None
    verdict = probe.gen(10, 4, stop=lambda: "breach")()
    assert verdict.dtype == torch.bool and verdict.tolist() == [True, True]
    # no stop source: generate is called exactly as before
    seen.clear()
    probe.model = SimpleNamespace(generate=lambda ids, **kw: seen.update(kw))
    probe.gen(10, 4)()
    assert "stopping_criteria" not in seen


def test_the_measuring_child_arms_the_watchdog_before_the_model_loads():
    src = _SCRIPT.read_text(encoding="utf-8")
    block = src[src.index("if args.calibrate or args.profile_rollout:"):]
    block = block[:block.index("return EXIT_OK")]
    arm = block.index("with watchdog:")
    assert block.index("guard.MemoryWatchdog(") < arm < block.index("build_trainer(")
    assert re.search(r"stop=lambda: watchdog\.breached or signalled\(\)", block)
    assert "_trap_stop_signals()" in block


def test_sigterm_is_a_stop_request_and_a_second_one_ends_it(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(
        "import importlib.util, os, signal, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('r', {str(_SCRIPT)!r})\n"
        "r = importlib.util.module_from_spec(spec); sys.modules['r'] = r; spec.loader.exec_module(r)\n"
        "got = r._trap_stop_signals()\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "time.sleep(0.2)\n"
        "print(got(), flush=True)\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "time.sleep(5)\n"
        "print('survived', flush=True)\n")
    proc = subprocess.run([sys.executable, str(child)], capture_output=True, text=True, timeout=120,
                          cwd=str(_REPO), env={**os.environ, "PYTHONPATH": str(_REPO)})
    assert proc.stdout.splitlines()[0] == "signal SIGTERM"
    assert "survived" not in proc.stdout and proc.returncode == -signal.SIGTERM


def test_the_report_is_replaced_atomically(tmp_path):
    out = tmp_path / "r" / "profile.json"
    runner._write(str(out), {"a": 1}, echo=False)
    runner._write(str(out), {"a": 2}, echo=False)
    assert json.loads(out.read_text())["a"] == 2
    assert [p.name for p in out.parent.iterdir()] == ["profile.json"]
