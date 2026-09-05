"""The profiler is guarded the same way the runner is.

The runner had two gates and a watchdog. The profiler had none, and the
profiler is what gets pointed at a new checkpoint first -- so on
2026-09-04 at 22:03 it was the profiler that loaded the 30B straight
through Windows' commit limit and took the desktop down at 22:09. The
guard code was right; the script people ran did not call it. Same
family as ``test_scheduler_vram_gate``: pin the CALL SITE, not just the
function.

Three things are pinned:

* admission runs before torch is imported, and a refusal returns the
  runner's exit code 2 without touching CUDA;
* the watchdog is armed before ``load_training_model`` (the spill happens
  inside ``from_pretrained``, before any trainer exists);
* the allocator cap is applied in the process that will allocate.
"""
from __future__ import annotations

import builtins
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_PROFILER = _REPO / "scripts" / "profile_grpo_vram.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


profiler = _load("_profiler_guard_under_test", _PROFILER)


class _Refusal:
    allowed = False
    reason = "Windows commit headroom 2.0 GiB < 16.0 GiB (test)"


class _Guard:
    """A stand-in memory_guard whose admission refuses."""

    def __init__(self) -> None:
        self.admissions = 0
        self.watchdogs = 0

    def check_admission(self):
        self.admissions += 1
        return _Refusal()

    def MemoryWatchdog(self, **_kw):  # noqa: N802 -- mirrors the module API
        self.watchdogs += 1
        raise AssertionError("watchdog must not be built after a refusal")


def test_refusal_returns_exit_2_without_importing_torch(monkeypatch) -> None:
    """A refused profile is a measurement, and it costs zero CUDA bytes."""
    guard = _Guard()
    monkeypatch.setattr(profiler, "_load_guard", lambda: guard)

    real_import = builtins.__import__

    def _no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch imported before admission refused")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    rc = profiler.main(["--model", "whatever"])
    assert rc == profiler.EXIT_REFUSED == 2
    assert guard.admissions == 1
    assert guard.watchdogs == 0


def test_skip_admission_still_arms_the_watchdog(monkeypatch) -> None:
    """``--skip-admission`` skips the gate, never the watchdog."""
    events: list = []

    class _WD:
        def __init__(self, **kw):
            events.append(("build", kw.get("label")))

        def __enter__(self):
            events.append(("enter", None))
            return self

        def __exit__(self, *_exc):
            events.append(("exit", None))

    class _G:
        def check_admission(self):
            raise AssertionError("admission must be skipped")

        MemoryWatchdog = _WD

    monkeypatch.setattr(profiler, "_load_guard", lambda: _G())

    def _fake_profile(args, guard, watchdog):
        events.append(("profile", args.model))
        return 0

    monkeypatch.setattr(profiler, "_profile", _fake_profile)
    rc = profiler.main(["--model", "m", "--skip-admission"])
    assert rc == 0
    assert events == [("build", "profile"), ("enter", None),
                      ("profile", "m"), ("exit", None)]


def _main_source() -> str:
    src = _PROFILER.read_text(encoding="utf-8")
    start = src.index("\ndef main(")
    return src[start:]


def test_source_order_admission_before_torch_and_watchdog_before_load() -> None:
    """The order in the file IS the guarantee; pin it against a refactor
    that moves ``import torch`` back above the gate or the load out from
    under the watchdog."""
    src = _main_source()
    i_adm = src.index("check_admission(")
    i_wd = src.index("MemoryWatchdog(")
    i_torch = re.search(r"^\s+import torch\s*$", src, re.M).start()
    i_cap = src.index("cap_cuda_allocator(")
    i_load = src.index("load_training_model(\n")
    assert i_adm < i_torch, "admission must run before torch is imported"
    assert i_wd < i_torch, "the watchdog must be armed before torch is imported"
    assert i_torch < i_cap < i_load, (
        "the allocator cap must sit between `import torch` and the load")


def test_profile_body_is_a_separate_function() -> None:
    """``_profile`` exists so the ``with watchdog:`` block owns every
    return path; a stray ``return`` inside ``main`` after the load would
    otherwise escape the watchdog's ``__exit__``."""
    assert callable(getattr(profiler, "_profile", None))
    src = _main_source()
    body = src[: src.index("\ndef _profile(")]
    assert "load_training_model(" not in body
    assert "trainer.train()" not in body
