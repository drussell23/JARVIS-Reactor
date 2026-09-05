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
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    monkeypatch.setattr(mg, "sample", lambda **_k: snap)


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

    def _next(**_k):
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
    monkeypatch.setattr(mg, "sample", lambda **_k: _sample(
        vram_used_mib=16000.0, vram_total_mib=32000.0, host_available_gib=40.0,
    ))
    with mg.MemoryWatchdog(interval_s=0.25, vram_ceiling_pct=99.0,
                           host_floor_gib=1.0) as wd:
        time.sleep(0.7)
        assert wd.breached is None


def test_watchdog_survives_a_raising_probe(monkeypatch) -> None:
    """A sampler exception must not escape the thread."""
    def _boom(**_k):
        raise OSError("nvidia-smi went away")

    monkeypatch.setattr(mg, "sample", _boom)
    with mg.MemoryWatchdog(interval_s=0.2) as wd:
        time.sleep(0.5)
        assert wd.breached is None
    assert wd.report()["samples"] == 0


def test_watchdog_stop_is_idempotent_and_joins(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample", lambda **_k: _sample(
        vram_used_mib=1000.0, vram_total_mib=32000.0, host_available_gib=40.0))
    before = threading.active_count()
    wd = mg.MemoryWatchdog(interval_s=0.2).start()
    wd.stop()
    wd.stop()
    assert threading.active_count() <= before


def test_watchdog_breach_callback_exception_is_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(mg, "sample", lambda **_k: _sample(
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


# ---------------------------------------------------------------------------
# Windows commit: the dimension the guest cannot see
# ---------------------------------------------------------------------------
#
# The guard already refused a full CARD and a full GUEST. It could not
# see a full HOST, and that is the one whose breach damages the machine:
# on 2026-09-04 vmmemWSL held 83 GiB of a 101 GiB Windows commit limit
# while /proc/meminfo inside the guest still reported gigabytes free.
# Windows began refusing allocations and sethc.exe, Code.exe and dwm.exe
# died with STATUS_COMMITMENT_LIMIT (0xc000012d). The trainer never
# OOM'd -- it took the desktop down instead.
#
# Every probe below follows this file's founding rule: an unreadable
# dimension is None, never a number, because a fabricated zero here would
# refuse every run and a fabricated maximum would admit the fatal one.


@pytest.fixture(autouse=True)
def _clear_commit_cache():
    """The reading is TTL-cached; a stale entry would leak across tests."""
    mg._win_commit_cache = (0.0, None, None)
    yield
    mg._win_commit_cache = (0.0, None, None)


class _Commit:
    """A successful powershell.exe interop call. Values are kB, as WMI reports."""

    returncode = 0

    def __init__(self, free_kb: str = "18000000", total_kb: str = "105000000"):
        self.stdout = f"{free_kb} {total_kb}\r\n"
        self.stderr = ""


def test_windows_commit_parses_kb_into_gib(monkeypatch) -> None:
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/mnt/c/powershell.exe")
    # 104,857,600 kB is exactly 100 GiB; 16,777,216 kB exactly 16 GiB.
    monkeypatch.setattr(mg.subprocess, "run",
                        lambda *a, **k: _Commit("16777216", "104857600"))
    avail, limit = mg.sample_windows_commit()
    assert avail == pytest.approx(16.0)
    assert limit == pytest.approx(100.0)


def test_windows_commit_is_none_off_wsl(monkeypatch) -> None:
    """A native Linux box has no such thing; it must not be invented."""
    monkeypatch.setattr(mg, "under_wsl", lambda: False)
    assert mg.sample_windows_commit() == (None, None)


def test_windows_commit_is_none_without_interop(monkeypatch) -> None:
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: None)
    assert mg.sample_windows_commit() == (None, None)


def test_windows_commit_never_raises_on_garbage(monkeypatch) -> None:
    """Interop returning something unparseable is unknown, not fatal."""
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/mnt/c/powershell.exe")

    class _Junk:
        returncode = 0
        stdout = "Get-CimInstance : Access denied\r\n"
        stderr = ""

    monkeypatch.setattr(mg.subprocess, "run", lambda *a, **k: _Junk())
    assert mg.sample_windows_commit() == (None, None)


def test_windows_commit_never_raises_on_timeout(monkeypatch) -> None:
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/mnt/c/powershell.exe")

    def _boom(*_a, **_k):
        raise mg.subprocess.TimeoutExpired(cmd="powershell.exe", timeout=8.0)

    monkeypatch.setattr(mg.subprocess, "run", _boom)
    assert mg.sample_windows_commit() == (None, None)


def test_zero_commit_limit_is_unreadable_not_zero(monkeypatch) -> None:
    """A zero limit would make the headroom ratio meaningless."""
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/mnt/c/powershell.exe")
    monkeypatch.setattr(mg.subprocess, "run", lambda *a, **k: _Commit("100", "0"))
    assert mg.sample_windows_commit() == (None, None)


def test_windows_commit_reading_is_cached(monkeypatch) -> None:
    """The watchdog polls faster than a process spawn costs."""
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/mnt/c/powershell.exe")
    calls = {"n": 0}

    def _counted(*_a, **_k):
        calls["n"] += 1
        return _Commit()

    monkeypatch.setattr(mg.subprocess, "run", _counted)
    mg.sample_windows_commit()
    mg.sample_windows_commit()
    mg.sample_windows_commit()
    assert calls["n"] == 1


# --- admission ------------------------------------------------------------


def test_exhausted_host_commit_is_refused(monkeypatch) -> None:
    """The 2026-09-04 geometry exactly: healthy guest, dying host."""
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
        win_commit_available_gib=2.0, win_commit_limit_gib=101.0,
    ))
    adm = mg.check_admission()
    assert adm.allowed is False
    assert "commit" in adm.reason.lower()


def test_healthy_commit_is_admitted(monkeypatch) -> None:
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
        win_commit_available_gib=60.0, win_commit_limit_gib=101.0,
    ))
    assert mg.check_admission().allowed is True


def test_unknown_commit_does_not_block_a_linux_box(monkeypatch) -> None:
    """None is unknown. Off WSL2 this dimension must not gate anything."""
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
        win_commit_available_gib=None, win_commit_limit_gib=None,
    ))
    assert mg.check_admission().allowed is True


def test_commit_floor_is_env_overridable(monkeypatch) -> None:
    monkeypatch.setenv("REACTOR_TRAIN_MIN_WINDOWS_COMMIT_GIB", "4")
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
        win_commit_available_gib=8.0, win_commit_limit_gib=101.0,
    ))
    assert mg.check_admission().allowed is True


def test_commit_refusal_survives_an_unreadable_guest(monkeypatch) -> None:
    """The reason string interpolates guest memory, which may be None."""
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=550.0, vram_total_mib=32607.0,
        host_available_gib=None, host_total_gib=None,
        win_commit_available_gib=1.0, win_commit_limit_gib=101.0,
    ))
    adm = mg.check_admission()
    assert adm.allowed is False
    assert "unreadable" in adm.reason


# ---------------------------------------------------------------------------
# The watchdog and Windows commit: the one breach that kills
# ---------------------------------------------------------------------------
#
# Admission is one-shot. The 2026-09-04 22:03 run PASSED admission (81 GiB
# free) and then moved Windows commit by ~60 GiB in the next minute, from
# inside ``from_pretrained`` -- before any trainer existed to carry a
# ``should_training_stop`` flag. So the watchdog must (a) read commit
# fresh on every tick, (b) treat a floor breach as a kill, not a flag,
# and (c) do it exactly once. ``_kill_self`` is replaced in every test
# below; a real SIGKILL would take pytest with it.


def _disarm(monkeypatch) -> list:
    killed: list = []
    monkeypatch.setattr(mg, "_kill_self", lambda: killed.append(True))
    return killed


def _healthy_but_host_at(commit_gib):
    return _sample(
        vram_used_mib=5000.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
        win_commit_available_gib=commit_gib, win_commit_limit_gib=101.0,
    )


def test_commit_floor_hard_aborts_exactly_once(monkeypatch, tmp_path) -> None:
    killed = _disarm(monkeypatch)
    abort = tmp_path / "abort.json"
    monkeypatch.setenv("REACTOR_TRAIN_ABORT_FILE", str(abort))
    _patch_sample(monkeypatch, _healthy_but_host_at(9.0))
    fired: list = []
    with mg.MemoryWatchdog(
        interval_s=0.2, win_commit_floor_gib=14.0,
        on_breach=lambda reason, snap: fired.append(reason),
    ) as wd:
        time.sleep(0.8)  # several ticks; the kill must not repeat
    assert killed == [True]
    assert wd.commit_breached is not None and "commit" in wd.commit_breached
    assert wd.breached == wd.commit_breached
    rep = wd.report()
    assert rep["hard_aborted"] is True
    assert rep["min_win_commit_available_gib"] == 9.0
    assert rep["win_commit_floor_gib"] == 14.0
    assert fired == [wd.commit_breached], "on_breach fires once, before the kill"
    # The last words were written BEFORE the trigger was pulled.
    data = json.loads(abort.read_text(encoding="utf-8"))
    assert data["reason"] == wd.commit_breached
    assert data["sample"]["win_commit_available_gib"] == 9.0
    assert data["watchdog"]["hard_aborted"] is True


def test_unknown_commit_never_aborts(monkeypatch) -> None:
    """None is unknown, and a native-Linux box must never be killed for it."""
    killed = _disarm(monkeypatch)
    _patch_sample(monkeypatch, _sample(
        vram_used_mib=5000.0, vram_total_mib=32607.0,
        host_available_gib=40.0, host_total_gib=47.0,
    ))
    with mg.MemoryWatchdog(interval_s=0.2, win_commit_floor_gib=14.0) as wd:
        time.sleep(0.5)
    assert killed == []
    assert wd.commit_breached is None
    assert wd.report()["min_win_commit_available_gib"] is None


def test_commit_above_floor_never_aborts(monkeypatch) -> None:
    killed = _disarm(monkeypatch)
    _patch_sample(monkeypatch, _healthy_but_host_at(60.0))
    with mg.MemoryWatchdog(interval_s=0.2, win_commit_floor_gib=14.0) as wd:
        time.sleep(0.5)
    assert killed == []
    assert wd.commit_breached is None
    assert wd.report()["hard_aborted"] is False


def test_commit_breach_still_kills_after_a_soft_breach(monkeypatch) -> None:
    """A run that tripped the VRAM ceiling is waiting for step-end. If the
    host collapses in that window it must still be killed."""
    killed = _disarm(monkeypatch)
    readings = [
        _sample(vram_used_mib=32500.0, vram_total_mib=32607.0,
                host_available_gib=40.0, host_total_gib=47.0,
                win_commit_available_gib=50.0, win_commit_limit_gib=101.0),
        _healthy_but_host_at(5.0),
    ]
    idx = {"i": 0}

    def _next(**_k):
        s = readings[min(idx["i"], len(readings) - 1)]
        idx["i"] += 1
        return s

    monkeypatch.setattr(mg, "sample", _next)
    with mg.MemoryWatchdog(interval_s=0.2, vram_ceiling_pct=99.0,
                           win_commit_floor_gib=14.0) as wd:
        time.sleep(0.8)
    assert killed == [True]
    assert "VRAM" in (wd.breached or ""), "the first (soft) breach is kept"
    assert "commit" in (wd.commit_breached or "")


def test_hard_abort_off_records_without_killing(monkeypatch) -> None:
    killed = _disarm(monkeypatch)
    _patch_sample(monkeypatch, _healthy_but_host_at(5.0))
    with mg.MemoryWatchdog(interval_s=0.2, win_commit_floor_gib=14.0,
                           hard_abort=False) as wd:
        time.sleep(0.5)
    assert killed == []
    assert "commit" in (wd.commit_breached or "")
    assert wd.report()["hard_aborted"] is False


def test_commit_floor_is_env_overridable(monkeypatch) -> None:
    killed = _disarm(monkeypatch)
    monkeypatch.setenv("REACTOR_TRAIN_WIN_COMMIT_FLOOR_GIB", "2")
    _patch_sample(monkeypatch, _healthy_but_host_at(9.0))
    with mg.MemoryWatchdog(interval_s=0.2) as wd:
        time.sleep(0.5)
    assert killed == []
    assert wd.report()["win_commit_floor_gib"] == 2.0


def test_default_floor_sits_below_admission_and_above_the_sentinel() -> None:
    """The three floors are an ordering, and the order is the design:
    admission refuses first, the in-process kill comes second, the host
    sentinel (12, then 8) is the backstop. Pin it so nobody 'tidies' them
    into the same number."""
    assert mg.DEFAULT_MIN_WINDOWS_COMMIT_GIB > mg.DEFAULT_WIN_COMMIT_FLOOR_GIB > 12.0


def test_watchdog_asks_for_a_fresh_commit_reading(monkeypatch) -> None:
    """The cache TTL is sized for admission; the watchdog passes its own
    interval so a reading is never older than one tick."""
    _disarm(monkeypatch)
    seen: list = []

    def _capture(**kw):
        seen.append(kw.get("commit_max_age_s"))
        return _healthy_but_host_at(60.0)

    monkeypatch.setattr(mg, "sample", _capture)
    with mg.MemoryWatchdog(interval_s=0.3):
        time.sleep(0.45)
    assert seen and all(age == 0.3 for age in seen)


def test_sample_windows_commit_max_age_bypasses_the_cache(monkeypatch) -> None:
    monkeypatch.setattr(mg, "under_wsl", lambda: True)
    monkeypatch.setattr(mg.shutil, "which", lambda _n: "/mnt/c/powershell.exe")
    calls = {"n": 0}

    def _counted(*_a, **_k):
        calls["n"] += 1
        return _Commit()

    monkeypatch.setattr(mg.subprocess, "run", _counted)
    mg.sample_windows_commit()
    mg.sample_windows_commit(max_age_s=0.0)
    mg.sample_windows_commit(max_age_s=0.0)
    assert calls["n"] == 3
    mg.sample_windows_commit()  # default TTL: served from the cache
    assert calls["n"] == 3


def test_write_abort_report_never_raises(monkeypatch, tmp_path) -> None:
    """The caller is about to SIGKILL itself; a bad path must not stop it."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("REACTOR_TRAIN_ABORT_FILE", str(blocker / "abort.json"))
    out = mg.write_abort_report("why", _healthy_but_host_at(1.0), {})
    assert out is None


# --- the allocator cap ----------------------------------------------------


def _fake_torch(available: bool, total_gib: float = 32.0):
    calls: list = []
    cuda = SimpleNamespace(
        is_available=lambda: available,
        set_per_process_memory_fraction=lambda f: calls.append(f),
        get_device_properties=lambda _i: SimpleNamespace(
            total_memory=int(total_gib * mg.GIB)),
    )
    return SimpleNamespace(cuda=cuda), calls


def test_cap_cuda_allocator_applies_the_default_fraction(monkeypatch) -> None:
    torch, calls = _fake_torch(available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert mg.cap_cuda_allocator() == mg.DEFAULT_CUDA_ALLOCATOR_FRACTION
    assert calls == [mg.DEFAULT_CUDA_ALLOCATOR_FRACTION]


def test_cap_cuda_allocator_is_env_overridable(monkeypatch) -> None:
    torch, calls = _fake_torch(available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("REACTOR_TRAIN_CUDA_ALLOCATOR_FRACTION", "0.8")
    assert mg.cap_cuda_allocator() == 0.8
    assert calls == [0.8]


def test_cap_cuda_allocator_rejects_nonsense_without_touching_torch(monkeypatch) -> None:
    torch, calls = _fake_torch(available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert mg.cap_cuda_allocator(1.5) is None
    assert mg.cap_cuda_allocator(0.0) is None
    assert calls == []


def test_cap_cuda_allocator_is_none_without_cuda(monkeypatch) -> None:
    torch, calls = _fake_torch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert mg.cap_cuda_allocator() is None
    assert calls == []


# --- the page-cache valve -------------------------------------------------
#
# The loader keeps every shard mmapped; under WSL2 each mapped page is
# Windows commit on vmmemWSL. Measured 2026-09-04 23:16: Mapped: grew one
# shard per sample and never fell, MemAvailable never moved, and the 30B
# load headed for ~97 of 100.7 GiB. The valve unmaps from inside the
# process, which is the only place mapped pages can be released from.

linux_only = pytest.mark.skipif(not Path("/proc/self/maps").exists(),
                                reason="needs /proc/self/maps")


@linux_only
def test_valve_sees_a_mapped_checkpoint_and_unmaps_it(tmp_path) -> None:
    import mmap

    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"\x5a" * (8 << 20))
    other = tmp_path / "notes.txt"
    other.write_bytes(b"\x00" * (1 << 20))
    with open(shard, "rb") as fh, open(other, "rb") as oh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        om = mmap.mmap(oh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            assert mm[:4] == b"\x5a" * 4  # fault the pages in
            valve = mg.PageCacheValve(interval_s=0.2)
            ranges = valve.mapped_ranges()
            paths = {p for _, _, p in ranges}
            assert str(shard) in paths, "the shard mapping is found by suffix"
            assert str(other) not in paths, "only checkpoint suffixes qualify"
            size = sum(e - s for s, e, p in ranges if p == str(shard))
            assert size >= 8 << 20
            valve._tick()
            rep = valve.report()
            assert rep["enabled"] is True and rep["ticks"] == 1
            assert rep["advised_gib"] > 0.0 and rep["peak_mapped_gib"] > 0.0
            assert rep["files"] == 1
            # Correctness after PAGEOUT: a read-only mapping simply refaults.
            assert mm[-4:] == b"\x5a" * 4
        finally:
            mm.close()
            om.close()


@linux_only
def test_valve_runs_as_a_context_manager_and_takes_a_final_pass(tmp_path) -> None:
    with mg.PageCacheValve(interval_s=0.1) as valve:
        time.sleep(0.35)
        assert valve._thread is not None
    assert valve._thread is None
    assert valve.report()["ticks"] >= 3, "periodic ticks plus the final pass"


def test_valve_is_a_reporting_noop_without_proc_maps(monkeypatch) -> None:
    real_exists = mg.os.path.exists
    monkeypatch.setattr(mg.os.path, "exists",
                        lambda p: False if p == "/proc/self/maps" else real_exists(p))
    with mg.PageCacheValve(interval_s=0.1) as valve:
        time.sleep(0.2)
    rep = valve.report()
    assert rep["enabled"] is False and rep["ticks"] == 0


def test_valve_recognises_hub_blobs_and_large_files_not_libraries() -> None:
    """The kernel reports the RESOLVED path; a hub shard resolves to
    ``blobs/<sha256>`` with no suffix. That is how the first valve saw
    nothing while 44 GiB sat mapped."""
    valve = mg.PageCacheValve()
    big = 4 << 30
    assert valve.is_checkpoint_mapping("/home/u/.cache/huggingface/hub/models--x/blobs/abc123", "r--s", big)
    assert valve.is_checkpoint_mapping("/data/model-00001-of-00016.safetensors", "r--p", 1 << 20)
    assert valve.is_checkpoint_mapping("/data/weights.bin", "r--p", 1 << 20)
    assert valve.is_checkpoint_mapping("/data/anything-huge", "r--p", big), "large + not executable"
    assert not valve.is_checkpoint_mapping("/usr/lib/libtorch_cuda.so", "r-xp", big), "executable"
    assert not valve.is_checkpoint_mapping("/usr/lib/libcudnn.so.9", "r--p", big), "a library's data segment"
    assert not valve.is_checkpoint_mapping("/data/small.dat", "r--p", 1 << 20), "small unknown file"


def test_valve_survives_a_bad_maps_line(monkeypatch, tmp_path) -> None:
    """A mapping line it cannot parse is skipped, never fatal."""
    fake = tmp_path / "maps"
    fake.write_text("garbage\n7f0000000000-7f0000001000 r--s 0 08:30 1 /x/a.safetensors\n",
                    encoding="utf-8")
    real_open = open

    def _open(path, *a, **k):
        return real_open(fake if path == "/proc/self/maps" else path, *a, **k)

    monkeypatch.setattr("builtins.open", _open)
    valve = mg.PageCacheValve()
    valve._enabled = True
    assert valve.mapped_ranges() == [(0x7F0000000000, 0x7F0000001000, "/x/a.safetensors")]


def test_default_fraction_leaves_room_for_what_the_cap_cannot_see() -> None:
    """Context, cuBLAS workspace and bitsandbytes scratch bypass the
    caching allocator. The default must leave them somewhere to go."""
    assert 0.9 <= mg.DEFAULT_CUDA_ALLOCATOR_FRACTION < 1.0
