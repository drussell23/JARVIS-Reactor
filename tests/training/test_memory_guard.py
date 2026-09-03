"""The memory probe never guesses, and the ladder never proposes a rung
TRL would reject.

Two failure families are pinned here because both have already cost a run
on this box:

* **A confidently-wrong number.** The Night Shift gate read nvidia-smi's
  ``utilization.memory`` (bus-busy percent) as occupancy and measured 0%
  while 29078/32607 MiB was held. Every unreadable probe below must
  return ``None``, never ``0.0`` — a zero here is indistinguishable from
  an idle card and admits a job onto a full one.
* **A fallback that cannot be tried.** TRL requires the global batch to be
  a whole number of groups, so a rung that halves ``num_generations`` into
  an indivisible value fails in the config constructor rather than at the
  allocation it was meant to relieve.

``memory_guard`` is loaded by path: it is stdlib-only at module scope, and
``reactor_core/__init__`` would drag the whole ML stack in behind it.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "reactor_core" / "training" / "memory_guard.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_memguard_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mg = _load()


def _sample(**kw):
    base = dict(
        ts=0.0, gpu_util_pct=0.0, vram_used_mib=None, vram_total_mib=None,
        host_available_gib=None, host_total_gib=None,
    )
    base.update(kw)
    return mg.MemorySample(**base)


# ---------------------------------------------------------------------------
# Occupancy is a ratio, and an unknown is a None
# ---------------------------------------------------------------------------


def test_occupancy_is_used_over_total() -> None:
    """The exact live measurement the scheduler gate was built around."""
    s = _sample(vram_used_mib=29078.0, vram_total_mib=32607.0)
    assert s.vram_occupancy_pct == pytest.approx(89.177, abs=1e-3)


def test_unreadable_vram_is_none_not_zero() -> None:
    s = _sample(vram_used_mib=None, vram_total_mib=None)
    assert s.vram_occupancy_pct is None
    assert s.readable is False


def test_zero_total_does_not_divide() -> None:
    """A total of 0 is unusable, not a card of infinite size."""
    s = _sample(vram_used_mib=100.0, vram_total_mib=0.0)
    assert s.vram_occupancy_pct is None


def test_sample_gpu_without_nvidia_smi_returns_nones(monkeypatch) -> None:
    monkeypatch.setattr(mg.shutil, "which", lambda _n: None)
    assert mg.sample_gpu() == (None, None, None)


def test_sample_gpu_on_unparseable_output_returns_nones(monkeypatch) -> None:
    class _Done:
        returncode = 0
        stdout = "not,a,number\n"
        stderr = ""

    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(mg.subprocess, "run", lambda *a, **k: _Done())
    assert mg.sample_gpu() == (None, None, None)


def test_sample_gpu_on_nonzero_exit_returns_nones(monkeypatch) -> None:
    class _Failed:
        returncode = 9
        stdout = ""
        stderr = "no devices"

    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(mg.subprocess, "run", lambda *a, **k: _Failed())
    assert mg.sample_gpu() == (None, None, None)


def test_gpu_occupancy_pct_reports_none_when_unreadable(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample_gpu", lambda: (None, None, None))
    util, occ = mg.gpu_occupancy_pct()
    assert occ is None


def test_gpu_occupancy_pct_is_a_percentage(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample_gpu", lambda: (12.0, 16000.0, 32000.0))
    util, occ = mg.gpu_occupancy_pct()
    assert util == 12.0
    assert occ == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def _patch_sample(monkeypatch, snap) -> None:
    monkeypatch.setattr(mg, "sample", lambda: snap)


def test_resident_model_is_refused(monkeypatch) -> None:
    """A soak's ollama model and a training run cannot share this card."""
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=29078.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
    ))
    adm = mg.check_admission()
    assert adm.allowed is False
    assert "occupancy" in adm.reason


def test_free_card_is_admitted(monkeypatch) -> None:
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=44.0, host_total_gib=47.0,
    ))
    adm = mg.check_admission()
    assert adm.allowed is True


def test_host_memory_floor_is_enforced(monkeypatch) -> None:
    """48.2 GiB anon-rss against WSL's 47 is how the profiler died."""
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=2.0, host_total_gib=47.0,
    ))
    adm = mg.check_admission()
    assert adm.allowed is False
    assert "host memory" in adm.reason


def test_unreadable_refuses_by_default(monkeypatch) -> None:
    _patch_sample(monkeypatch, _sample())
    assert mg.check_admission().allowed is False


def test_unreadable_can_be_overridden(monkeypatch) -> None:
    _patch_sample(monkeypatch, _sample())
    adm = mg.check_admission(require_readable=False)
    assert adm.allowed is True


def test_thresholds_are_env_overridable(monkeypatch) -> None:
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=20000.0, vram_total_mib=32607.0,
        host_available_gib=44.0, host_total_gib=47.0,
    ))
    assert mg.check_admission().allowed is False        # 61% > 55% default
    monkeypatch.setenv("REACTOR_TRAIN_MAX_VRAM_OCCUPANCY_PCT", "70")
    assert mg.check_admission().allowed is True


def test_garbage_env_threshold_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("REACTOR_TRAIN_MAX_VRAM_OCCUPANCY_PCT", "not-a-number")
    assert mg._env_float("REACTOR_TRAIN_MAX_VRAM_OCCUPANCY_PCT", 55.0) == 55.0


# ---------------------------------------------------------------------------
# The watchdog
# ---------------------------------------------------------------------------


def test_watchdog_records_peak_and_trips_once(monkeypatch) -> None:
    readings = [
        _sample(vram_used_mib=10000.0, vram_total_mib=32000.0,
                host_available_gib=40.0),
        _sample(vram_used_mib=31900.0, vram_total_mib=32000.0,
                host_available_gib=39.0),
        _sample(vram_used_mib=31950.0, vram_total_mib=32000.0,
                host_available_gib=38.0),
    ]
    idx = {"i": 0}

    def _next():
        s = readings[min(idx["i"], len(readings) - 1)]
        idx["i"] += 1
        return s

    monkeypatch.setattr(mg, "sample", _next)
    fired = []
    wd = mg.MemoryWatchdog(
        interval_s=0.25, vram_ceiling_pct=99.0, host_floor_gib=1.0,
        on_breach=lambda reason, snap: fired.append(reason),
    )
    with wd:
        deadline = time.time() + 5.0
        while wd.breached is None and time.time() < deadline:
            time.sleep(0.05)
    report = wd.report()
    assert wd.breached is not None
    assert len(fired) == 1, "on_breach must fire once, not once per sample"
    assert report["peak_vram_occupancy_pct"] >= 99.0
    assert report["min_host_available_gib"] <= 40.0


def test_watchdog_does_not_trip_below_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample", lambda: _sample(
        vram_used_mib=16000.0, vram_total_mib=32000.0, host_available_gib=40.0,
    ))
    with mg.MemoryWatchdog(interval_s=0.25, vram_ceiling_pct=99.0,
                           host_floor_gib=1.0) as wd:
        time.sleep(0.7)
        assert wd.breached is None


def test_watchdog_survives_a_raising_probe(monkeypatch) -> None:
    """A sampler exception must not escape the thread."""
    def _boom():
        raise OSError("nvidia-smi went away")

    monkeypatch.setattr(mg, "sample", _boom)
    with mg.MemoryWatchdog(interval_s=0.2) as wd:
        time.sleep(0.5)
        assert wd.breached is None
    assert wd.report()["samples"] == 0


def test_watchdog_stop_is_idempotent_and_joins(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample", lambda: _sample(
        vram_used_mib=1000.0, vram_total_mib=32000.0, host_available_gib=40.0))
    before = threading.active_count()
    wd = mg.MemoryWatchdog(interval_s=0.2).start()
    wd.stop()
    wd.stop()
    assert threading.active_count() <= before


def test_watchdog_breach_callback_exception_is_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample", lambda: _sample(
        vram_used_mib=31999.0, vram_total_mib=32000.0, host_available_gib=40.0))

    def _bad(_reason, _snap):
        raise RuntimeError("callback is broken")

    with mg.MemoryWatchdog(interval_s=0.2, on_breach=_bad) as wd:
        deadline = time.time() + 3.0
        while wd.breached is None and time.time() < deadline:
            time.sleep(0.05)
    assert wd.breached is not None


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_ladder_starts_at_what_was_asked_for() -> None:
    rungs = mg.build_ladder(num_generations=4, max_completion_length=256)
    assert rungs[0].num_generations == 4
    assert rungs[0].max_completion_length == 256


def test_ladder_is_monotonically_cheaper() -> None:
    rungs = mg.build_ladder(num_generations=4, max_completion_length=256)
    costs = [r.num_generations * r.max_completion_length for r in rungs]
    assert costs == sorted(costs, reverse=True)
    assert len(set(costs)) == len(costs), "no rung may repeat a cost"


def test_ladder_never_proposes_an_indivisible_group() -> None:
    """TRL rejects a global batch that is not a whole number of groups.

    Including the FIRST rung: a ladder whose top rung cannot even be
    constructed fails in GRPOConfig before allocating anything, which is
    a config error wearing a memory strategy's clothes.
    """
    for global_batch in (2, 4, 6, 8, 12):
        for rung in mg.build_ladder(num_generations=4, global_batch=global_batch):
            assert global_batch % rung.num_generations == 0, (
                f"{rung.name}: {global_batch} % {rung.num_generations} != 0"
            )


def test_indivisible_request_is_clamped_not_obeyed() -> None:
    rungs = mg.build_ladder(num_generations=4, global_batch=2)
    assert rungs[0].num_generations == 2


def test_ladder_never_goes_below_two_generations() -> None:
    """A group of one has no sibling to form an advantage against."""
    for rung in mg.build_ladder(num_generations=2, global_batch=8):
        assert rung.num_generations >= 2


def test_ladder_shrinks_completions_before_the_group() -> None:
    """Shorter completions truncate some candidates; a smaller group
    degrades every gradient. Cheap first, damaging last."""
    rungs = mg.build_ladder(num_generations=4, max_completion_length=256,
                            global_batch=8)
    names = [r.name for r in rungs]
    assert names.index("short-completions") < names.index("small-group")


def test_divisible_generations_picks_the_largest_that_fits() -> None:
    assert mg._divisible_generations(4, 8) == 4
    assert mg._divisible_generations(3, 8) == 2
    assert mg._divisible_generations(6, 12) == 6
    assert mg._divisible_generations(5, 12) == 4


# ---------------------------------------------------------------------------
# OOM classification
# ---------------------------------------------------------------------------


def test_is_oom_matches_torch_class_by_name() -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    assert mg.is_oom(OutOfMemoryError("CUDA out of memory")) is True


def test_is_oom_matches_by_message_for_foreign_classes() -> None:
    """bitsandbytes, Triton and the GPTQ kernels each raise their own."""
    assert mg.is_oom(RuntimeError("CUDA error: out of memory")) is True
    assert mg.is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")) is True


def test_is_oom_rejects_unrelated_failures() -> None:
    assert mg.is_oom(ValueError("no trainable prompts in corpus")) is False
    assert mg.is_oom(KeyError("prompt")) is False
