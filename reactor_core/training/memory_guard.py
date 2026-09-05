"""Memory admission, live watchdog, and the degradation ladder for GRPO.

A GRPO step on this box has three memory consumers and only one of them
is negotiable:

1. the 4-bit base weights (~17 GiB for the 30B MoE) — **fixed**, they are
   the model;
2. the generation KV cache — ``num_generations`` sequences, each holding
   the whole prompt plus ``max_completion_length`` tokens. O+V prompts are
   ~24 KB of largely-boilerplate context, so this term is *large* and it
   scales linearly in both knobs;
3. backprop activations through the LoRA path, already suppressed by
   ``gradient_checkpointing``.

Measured 2026-09-03 on the RTX 5090 (32607 MiB) with the 30B GPTQ
checkpoint at ``num_generations=4 / max_completion_length=256``: the card
sat at **32051-32081 MiB, 98.4% occupancy**, host RSS ~3 GiB. There is no
headroom at that setting. That is the number this module exists to
defend, and (2) is the only lever with real travel in it.

## Why not FSDP

The obvious-sounding rung is "shard it with FSDP", and it is wrong here
for two independent reasons:

* FSDP shards parameters **across ranks**. ``world_size == 1`` has nothing
  to shard, so on this single card it costs a wrapper and buys zero bytes.
* The checkpoint is pre-quantized GPTQ. FSDP's flat-parameter machinery
  assumes unpacked floating-point tensors and cannot shard 4-bit packed
  weights with their scales and zero-points.

So ``fsdp_rungs()`` yields rungs only when a second device actually
exists. On one card the honest ladder is: shrink the KV cache, then
shrink it again, then refuse. Refusing is a real outcome — an OOM-killed
run that took the GPU down with it is strictly worse than a job that
declined to start and said why.

## Why the numbers are read the way they are

VRAM occupancy is ``memory.used / memory.total``, never nvidia-smi's
``utilization.memory`` — that is the percent of time the memory *bus* was
busy, and the two diverge exactly where it matters. Measured on this card
with a resident 32B: ``utilization.memory=0`` while 29078/32607 MiB
(89.2%) was held. Gating on the bandwidth figure admits a training job
onto a card with 3 GiB free. ``reactor_core.api.scheduler.ResourceMonitor``
had that defect; it now delegates here so there is one reader, not two.

Host memory is read from ``/proc/meminfo``'s **MemAvailable**, not
``MemFree``. The profiler was once OOM-killed at ``anon-rss:48216856kB``
against WSL's 47 GiB while MemFree looked comfortable, because page cache
is reclaimable and MemFree does not say so.

An unreadable probe returns ``None``, never ``0.0``. A confidently-wrong
zero is the failure family this module was written to end, and a guard
that cannot see is a guard that must say so rather than wave the job
through.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MIB = 1024 ** 2
GIB = 1024 ** 3

#: Refuse to start above this VRAM occupancy. 55% leaves room for the
#: ~17 GiB of base weights on a 32 GiB card; anything higher means
#: something else is resident (ollama serving the soak's 30B is the
#: recurring case) and the load would OOM partway through.
DEFAULT_MAX_VRAM_OCCUPANCY_PCT = 55.0

#: Refuse to start below this much reclaimable host memory. The GPTQ
#: shard stream and Triton's JIT scratch both land here.
DEFAULT_MIN_HOST_AVAILABLE_GIB = 12.0

#: The watchdog trips above this occupancy. Higher than the admission
#: bar on purpose: admission asks "is the card free enough to begin",
#: the watchdog asks "are we about to die".
DEFAULT_VRAM_CEILING_PCT = 99.0

#: ...and below this much host memory.
DEFAULT_HOST_FLOOR_GIB = 4.0

#: Refuse to start below this much WINDOWS COMMIT headroom when running
#: under WSL2. This is a DIFFERENT number from host availability above,
#: and the distinction is the entire point of it existing:
#:
#: ``/proc/meminfo`` inside WSL2 describes the GUEST. ``MemTotal`` is the
#: ``memory=`` line of .wslconfig, not the machine. Windows meanwhile
#: charges the VM's whole reservation (guest RAM + swap + VM overhead)
#: against a system-wide COMMIT LIMIT of physical RAM + pagefile. The
#: guest can therefore report gigabytes available while Windows has none
#: left, and that combination is not hypothetical: on 2026-09-04
#: vmmemWSL held 83 GiB of a 101 GiB commit limit, Windows began
#: refusing allocations, and sethc.exe / Code.exe / dwm.exe died with
#: STATUS_COMMITMENT_LIMIT (0xc000012d) while the guest looked healthy.
#: Resource-Exhaustion-Detector logged event 2004 eleven times that day.
#:
#: A run admitted in that state does not fail cleanly -- it takes the
#: desktop down with it, which is strictly worse than refusing to start.
DEFAULT_MIN_WINDOWS_COMMIT_GIB = 16.0

#: The watchdog HARD-ABORTS its own process below this much Windows
#: commit headroom. Admission above asks "may we begin"; this asks "is
#: Windows about to die", and it is answered by killing the run, not by
#: asking the trainer to stop at the end of the step.
#:
#: Why a kill and not a flag: the two soft breaches (VRAM ceiling, guest
#: floor) are delivered through ``should_training_stop`` because the
#: trainer's traceback is worth keeping and the machine survives the
#: wait. Commit does not work that way. The 2026-09-04 22:03 profile run
#: pushed vmmemWSL from ~20 GB to 84 GB in ONE MINUTE while it was still
#: inside ``from_pretrained`` -- no trainer, no callback, nothing to set
#: a flag on -- and Windows took the desktop down at 22:09. The only
#: action that returns commit fast enough is ending the process that
#: holds it. SIGKILL is used deliberately: the CUDA context, the pinned
#: host buffers and every page the loader touched are released by the
#: kernel and the driver, and no Python ``finally`` block gets to run in
#: a host that has nothing left to give it.
#:
#: Sits between admission (16) and the Windows-side sentinel
#: (``scripts/host/wsl_commit_sentinel.ps1``, 12 then 8), so the
#: in-process guard fires first and the host sentinel is the backstop
#: for a guest that has stopped answering.
DEFAULT_WIN_COMMIT_FLOOR_GIB = 14.0

#: ``torch.cuda.set_per_process_memory_fraction`` argument. On this box
#: the WDDM driver backs CUDA allocations with HOST memory once the card
#: is full ("sysmem fallback": torch measured 61.75 GiB allocated on a
#: 32 GiB card), and every byte of that spill lands on vmmemWSL's Windows
#: commit charge. Capping torch's caching allocator at a fraction of
#: DEVICE memory makes it raise an honest OOM at the true ceiling instead
#: of paging the model into RAM. 0.95 leaves ~1.6 GiB for the CUDA
#: context, cuBLAS workspaces and bitsandbytes' own scratch, which do not
#: go through the caching allocator and so are not counted by the cap.
DEFAULT_CUDA_ALLOCATOR_FRACTION = 0.95

#: Where the hard-abort path leaves its last words. SIGKILL means the
#: caller's own report never gets written, so the guard writes this one
#: BEFORE it pulls the trigger. Env-overridable so a test can redirect it.
DEFAULT_ABORT_REPORT = os.path.join(
    os.path.expanduser("~"), ".jarvis", "trinity", "memguard_abort.json")

_NVIDIA_SMI_TIMEOUT_S = 5.0

#: Windows commit is read through WSL interop, which costs a process
#: spawn (~0.3-0.8s). The watchdog polls far faster than that, so the
#: reading is cached behind a TTL; admission is one-shot and pays it once.
_WIN_COMMIT_TIMEOUT_S = 8.0
_WIN_COMMIT_TTL_S = 15.0


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("[memguard] %s=%r is not a number; using %s",
                       name, raw, default)
        return default


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemorySample:
    """One instantaneous reading. ``None`` means *could not read*."""

    ts: float
    gpu_util_pct: Optional[float]
    vram_used_mib: Optional[float]
    vram_total_mib: Optional[float]
    host_available_gib: Optional[float]
    host_total_gib: Optional[float]
    #: WINDOWS commit, not guest memory -- see
    #: DEFAULT_MIN_WINDOWS_COMMIT_GIB for why the two are not the same
    #: question. None off WSL2, and None means unknown, never healthy.
    #: Defaulted so existing constructors (and their tests) still bind.
    win_commit_available_gib: Optional[float] = None
    win_commit_limit_gib: Optional[float] = None

    @property
    def vram_occupancy_pct(self) -> Optional[float]:
        if not self.vram_used_mib or not self.vram_total_mib:
            # `not` rather than `is None`: a total of 0 is also unusable,
            # and dividing by it would manufacture a number.
            return None
        return 100.0 * self.vram_used_mib / self.vram_total_mib

    @property
    def readable(self) -> bool:
        return self.vram_used_mib is not None or self.host_available_gib is not None

    def describe(self) -> str:
        occ = self.vram_occupancy_pct
        vram = (
            f"{self.vram_used_mib:.0f}/{self.vram_total_mib:.0f} MiB ({occ:.1f}%)"
            if occ is not None else "vram=unreadable"
        )
        host = (
            f"{self.host_available_gib:.1f} GiB avail"
            if self.host_available_gib is not None else "host=unreadable"
        )
        # Only ever mentioned when actually read, so the line stays
        # truthful (and unchanged) on a native-Linux box.
        if self.win_commit_available_gib is not None:
            limit = (f"/{self.win_commit_limit_gib:.0f}"
                     if self.win_commit_limit_gib else "")
            return (f"{vram}, {host}, win-commit "
                    f"{self.win_commit_available_gib:.1f}{limit} GiB free")
        return f"{vram}, {host}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "gpu_util_pct": self.gpu_util_pct,
            "vram_used_mib": self.vram_used_mib,
            "vram_total_mib": self.vram_total_mib,
            "vram_occupancy_pct": self.vram_occupancy_pct,
            "host_available_gib": self.host_available_gib,
            "host_total_gib": self.host_total_gib,
            "win_commit_available_gib": self.win_commit_available_gib,
            "win_commit_limit_gib": self.win_commit_limit_gib,
        }


def sample_gpu() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """``(gpu_util_pct, vram_used_mib, vram_total_mib)``, or Nones.

    Deliberately shells out rather than importing torch: this runs BEFORE
    the training stack is imported, and it must also report memory held by
    *other* processes — which is the whole point, and which
    ``torch.cuda.mem_get_info`` cannot be trusted for under WSL2, where the
    driver reports host-RAM fallback as if it were free device memory.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("[memguard] nvidia-smi failed: %s", exc)
        return None, None, None
    if out.returncode != 0:
        logger.debug("[memguard] nvidia-smi rc=%s: %s",
                     out.returncode, (out.stderr or "").strip()[:200])
        return None, None, None
    line = (out.stdout or "").strip().splitlines()
    if not line:
        return None, None, None
    try:
        util, used, total = (float(p.strip()) for p in line[0].split(",")[:3])
    except (TypeError, ValueError):
        logger.debug("[memguard] unparseable nvidia-smi row: %r", line[0])
        return None, None, None
    return util, used, total


def sample_host() -> Tuple[Optional[float], Optional[float]]:
    """``(available_gib, total_gib)`` from /proc/meminfo, or Nones."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            fields = {}
            for raw in fh:
                key, _, rest = raw.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        fields[key.strip()] = float(parts[0])  # kB
                    except ValueError:
                        continue
    except OSError:
        # Non-Linux, or a container without /proc. psutil is optional here
        # precisely so this module imports on a machine with neither.
        try:
            import psutil  # noqa: PLC0415
            vm = psutil.virtual_memory()
            return vm.available / GIB, vm.total / GIB
        except Exception:  # noqa: BLE001
            return None, None
    avail = fields.get("MemAvailable")
    total = fields.get("MemTotal")
    return (
        (avail * 1024.0 / GIB) if avail is not None else None,
        (total * 1024.0 / GIB) if total is not None else None,
    )


_win_commit_lock = threading.Lock()
_win_commit_cache: Tuple[float, Optional[float], Optional[float]] = (0.0, None, None)


def under_wsl() -> bool:
    """Are we running inside a WSL guest?

    Checked two ways because neither alone is reliable: the env var is
    absent under systemd services and bare ``wsl -e`` invocations, and
    the osrelease string has changed spelling across WSL versions.
    """
    if (os.environ.get("WSL_DISTRO_NAME") or "").strip():
        return True
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def sample_windows_commit(
    max_age_s: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """``(commit_available_gib, commit_limit_gib)`` for the WINDOWS host.

    ``(None, None)`` when not under WSL, when interop is unavailable, or
    when the reading cannot be trusted -- never an exception, and never a
    fabricated number. A caller that gets None must treat the dimension
    as unknown rather than as healthy.

    ``Win32_OperatingSystem`` is the cheapest source that reports both
    halves: ``FreeVirtualMemory`` is commit AVAILABLE and
    ``TotalVirtualMemorySize`` is the commit LIMIT, both in kB. Note that
    "virtual memory" in that class means commit, not address space --
    it is the same pair the Resource-Exhaustion-Detector arbitrates.

    ``max_age_s`` overrides the default TTL for one call. The watchdog
    passes its own poll interval: a shard stream moves commit at ~1 GiB/s
    on this box, so a reading 15 s old describes a host that no longer
    exists. Admission keeps the default -- it samples once.
    """
    global _win_commit_cache

    if not under_wsl():
        return None, None

    ttl = _WIN_COMMIT_TTL_S if max_age_s is None else max(0.0, float(max_age_s))
    now = time.time()
    with _win_commit_lock:
        stamp, avail, limit = _win_commit_cache
        if now - stamp < ttl:
            return avail, limit

    exe = shutil.which("powershell.exe")
    if not exe:
        # Interop disabled, or a distro without the Windows PATH. Cache
        # the miss so a stopped-interop box does not pay a `which` and a
        # failed spawn on every watchdog poll.
        with _win_commit_lock:
            _win_commit_cache = (now, None, None)
        return None, None

    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem;"
             "Write-Output ('{0} {1}' -f "
             "$o.FreeVirtualMemory,$o.TotalVirtualMemorySize)"],
            capture_output=True, text=True, timeout=_WIN_COMMIT_TIMEOUT_S,
        )
        parts = (proc.stdout or "").strip().split()
        if proc.returncode != 0 or len(parts) < 2:
            raise ValueError(
                f"rc={proc.returncode} stdout={(proc.stdout or '')[:80]!r}")
        # kB -> GiB. Guard the limit: a zero would make the headroom
        # ratio meaningless and is a clearer signal as "unreadable".
        avail_gib = float(parts[0]) * 1024.0 / GIB
        limit_gib = float(parts[1]) * 1024.0 / GIB
        if limit_gib <= 0.0:
            raise ValueError(f"commit limit {limit_gib}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memguard] windows commit unreadable: %s", exc)
        with _win_commit_lock:
            _win_commit_cache = (now, None, None)
        return None, None

    with _win_commit_lock:
        _win_commit_cache = (now, avail_gib, limit_gib)
    return avail_gib, limit_gib


def sample(commit_max_age_s: Optional[float] = None) -> MemorySample:
    """One reading of both devices. Never raises.

    ``commit_max_age_s`` is forwarded to :func:`sample_windows_commit`;
    see there for why the watchdog needs a fresher reading than admission.
    """
    util, used, total = sample_gpu()
    avail, host_total = sample_host()
    win_avail, win_limit = sample_windows_commit(max_age_s=commit_max_age_s)
    return MemorySample(
        ts=time.time(), gpu_util_pct=util, vram_used_mib=used,
        vram_total_mib=total, host_available_gib=avail,
        host_total_gib=host_total,
        win_commit_available_gib=win_avail, win_commit_limit_gib=win_limit,
    )


def gpu_occupancy_pct() -> Tuple[Optional[float], Optional[float]]:
    """``(gpu_util_pct, vram_occupancy_pct)`` — the scheduler's contract.

    ``ResourceMonitor._get_gpu_metrics`` delegates here so the "occupancy,
    not bandwidth" decision has exactly one implementation to keep right.
    """
    util, used, total = sample_gpu()
    if used is None or not total:
        return util, None
    return util, 100.0 * used / total


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reason: str
    sample: MemorySample


def check_admission(
    *,
    max_vram_occupancy_pct: Optional[float] = None,
    min_host_available_gib: Optional[float] = None,
    min_windows_commit_gib: Optional[float] = None,
    require_readable: bool = True,
) -> Admission:
    """Is this box free enough to start a training run right now?

    ``require_readable`` decides the unknown case. Default True: if
    neither probe can be read we refuse, because the alternative is
    starting a multi-hour job blind on a card that may already be full.
    Set False only where a caller has its own protection.
    """
    max_occ = (
        max_vram_occupancy_pct if max_vram_occupancy_pct is not None
        else _env_float("REACTOR_TRAIN_MAX_VRAM_OCCUPANCY_PCT",
                        DEFAULT_MAX_VRAM_OCCUPANCY_PCT)
    )
    min_host = (
        min_host_available_gib if min_host_available_gib is not None
        else _env_float("REACTOR_TRAIN_MIN_HOST_AVAILABLE_GIB",
                        DEFAULT_MIN_HOST_AVAILABLE_GIB)
    )
    min_commit = (
        min_windows_commit_gib if min_windows_commit_gib is not None
        else _env_float("REACTOR_TRAIN_MIN_WINDOWS_COMMIT_GIB",
                        DEFAULT_MIN_WINDOWS_COMMIT_GIB)
    )
    snap = sample()

    if not snap.readable:
        if require_readable:
            return Admission(
                False,
                "cannot read GPU or host memory — refusing to start blind "
                "(set require_readable=False to override)",
                snap,
            )
        return Admission(True, "memory unreadable; admitted by request", snap)

    occ = snap.vram_occupancy_pct
    if occ is not None and occ > max_occ:
        return Admission(
            False,
            f"VRAM occupancy {occ:.1f}% > {max_occ:.1f}% — something is "
            f"already resident ({snap.describe()}). A soak's ollama model "
            "and a training run cannot share this card.",
            snap,
        )
    if snap.host_available_gib is not None and snap.host_available_gib < min_host:
        return Admission(
            False,
            f"host memory {snap.host_available_gib:.1f} GiB available < "
            f"{min_host:.1f} GiB — the shard stream and Triton JIT scratch "
            "land here; this is how the profiler was OOM-killed at 48.2 GiB.",
            snap,
        )
    # Checked LAST and reported separately because it is the only
    # dimension whose breach damages the machine rather than the run: a
    # guest-memory refusal costs a training slot, a commit refusal
    # prevents Windows from killing the desktop out from under it.
    if (snap.win_commit_available_gib is not None
            and snap.win_commit_available_gib < min_commit):
        # host_available_gib may be None; formatting that with :.1f would
        # raise inside a module whose contract is that it never does.
        guest = (f"{snap.host_available_gib:.1f} GiB avail"
                 if snap.host_available_gib is not None else "unreadable")
        return Admission(
            False,
            f"Windows commit headroom {snap.win_commit_available_gib:.1f} GiB "
            f"< {min_commit:.1f} GiB — the WSL2 guest still looks healthy "
            f"({guest}) because /proc/meminfo "
            "describes the GUEST, but the HOST is out of commit. Starting "
            "here does not OOM the trainer, it takes Windows down "
            "(STATUS_COMMITMENT_LIMIT). Lower .wslconfig memory=/swap= or "
            "raise the pagefile; a guest-side drop_caches cannot return a "
            "host reservation.",
            snap,
        )
    return Admission(True, f"resources available ({snap.describe()})", snap)


# ---------------------------------------------------------------------------
# The live watchdog
# ---------------------------------------------------------------------------


class MemoryWatchdog:
    """Polls memory on a daemon thread, records the peak, trips on breach.

    Thread-safety contract: the sampler thread only ever mutates state
    under ``_lock``; readers take the same lock. The thread never raises
    into the interpreter — an exception inside it would otherwise be
    delivered at shutdown as an unretrieved-exception panic, which is
    noise on top of whatever real failure was in progress.

    Tripping on VRAM or guest memory does not kill anything. It records
    that a ceiling was crossed and calls ``on_breach`` once; the training
    thread decides what that means. A watchdog that unilaterally killed
    the process would lose the traceback that says which allocation
    actually failed.

    Tripping on WINDOWS COMMIT is the one exception, and it kills. See
    ``DEFAULT_WIN_COMMIT_FLOOR_GIB`` for the argument; the short form is
    that the alternative to losing a traceback is losing the desktop, and
    that breach arrives while the model is still loading, before any
    trainer exists to carry a stop flag. ``hard_abort=False`` turns the
    kill into a recorded breach for callers that own their own exit.
    """

    def __init__(
        self,
        *,
        interval_s: float = 2.0,
        vram_ceiling_pct: Optional[float] = None,
        host_floor_gib: Optional[float] = None,
        win_commit_floor_gib: Optional[float] = None,
        hard_abort: bool = True,
        on_breach: Optional[Callable[[str, MemorySample], None]] = None,
        label: str = "train",
    ) -> None:
        self._interval = max(0.25, float(interval_s))
        self._ceiling = (
            vram_ceiling_pct if vram_ceiling_pct is not None
            else _env_float("REACTOR_TRAIN_VRAM_CEILING_PCT",
                            DEFAULT_VRAM_CEILING_PCT)
        )
        self._floor = (
            host_floor_gib if host_floor_gib is not None
            else _env_float("REACTOR_TRAIN_HOST_FLOOR_GIB",
                            DEFAULT_HOST_FLOOR_GIB)
        )
        self._commit_floor = (
            win_commit_floor_gib if win_commit_floor_gib is not None
            else _env_float("REACTOR_TRAIN_WIN_COMMIT_FLOOR_GIB",
                            DEFAULT_WIN_COMMIT_FLOOR_GIB)
        )
        self._hard_abort = bool(hard_abort)
        self._on_breach = on_breach
        self._label = label
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_occ: Optional[float] = None
        self._peak_sample: Optional[MemorySample] = None
        self._min_host: Optional[float] = None
        self._min_commit: Optional[float] = None
        self._breach: Optional[str] = None
        self._commit_breach: Optional[str] = None
        self._hard_aborted = False
        self._samples = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "MemoryWatchdog":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"memguard-{self._label}", daemon=True,
        )
        self._thread.start()
        logger.info(
            "[memguard] watchdog armed: ceiling %.1f%% VRAM, floor %.1f GiB "
            "host, %s below %.1f GiB Windows commit, every %.1fs",
            self._ceiling, self._floor,
            "HARD ABORT" if self._hard_abort else "record",
            self._commit_floor, self._interval,
        )
        return self

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            # Bounded: the thread's own loop waits on the same Event, so it
            # returns within one interval. A join with no timeout here would
            # hang a shutdown on a wedged nvidia-smi.
            t.join(timeout=self._interval + _NVIDIA_SMI_TIMEOUT_S + 1.0)

    def __enter__(self) -> "MemoryWatchdog":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — see the class docstring
                logger.debug("[memguard] sampler tick failed", exc_info=True)
            self._stop.wait(self._interval)

    def _tick(self) -> None:
        # Fresh commit every tick: the cache TTL is sized for admission,
        # and a shard stream moves this number by the whole TTL's worth
        # of headroom in seconds.
        snap = sample(commit_max_age_s=self._interval)
        occ = snap.vram_occupancy_pct
        breach: Optional[str] = None
        hard: Optional[str] = None
        commit = snap.win_commit_available_gib
        with self._lock:
            self._samples += 1
            if occ is not None and (self._peak_occ is None or occ > self._peak_occ):
                self._peak_occ, self._peak_sample = occ, snap
            if snap.host_available_gib is not None and (
                self._min_host is None or snap.host_available_gib < self._min_host
            ):
                self._min_host = snap.host_available_gib
            if commit is not None and (
                self._min_commit is None or commit < self._min_commit
            ):
                self._min_commit = commit
            # Checked FIRST and independently of the soft breach: a run
            # that already tripped the VRAM ceiling and is waiting for
            # step-end must still be killed if the host collapses under
            # it in the meantime. Unknown (None) is not a breach -- a
            # native-Linux box has no such dimension.
            if (self._commit_breach is None and commit is not None
                    and commit < self._commit_floor):
                hard = self._commit_breach = (
                    f"Windows commit {commit:.1f} GiB < floor "
                    f"{self._commit_floor:.1f} GiB ({snap.describe()})"
                )
                if self._breach is None:
                    self._breach = hard
            if self._breach is None:
                if occ is not None and occ >= self._ceiling:
                    breach = self._breach = (
                        f"VRAM occupancy {occ:.1f}% >= ceiling "
                        f"{self._ceiling:.1f}% ({snap.describe()})"
                    )
                elif (
                    snap.host_available_gib is not None
                    and snap.host_available_gib <= self._floor
                ):
                    breach = self._breach = (
                        f"host memory {snap.host_available_gib:.1f} GiB <= floor "
                        f"{self._floor:.1f} GiB ({snap.describe()})"
                    )
        # Fire outside the lock: a callback that samples us must not deadlock.
        for reason in (hard, breach):
            if reason is None:
                continue
            logger.warning("[memguard] BREACH: %s", reason)
            if self._on_breach is not None:
                try:
                    self._on_breach(reason, snap)
                except Exception:  # noqa: BLE001
                    logger.debug("[memguard] on_breach raised", exc_info=True)
        if hard is not None:
            self._abort(hard, snap)

    def _abort(self, reason: str, snap: MemorySample) -> None:
        """Leave a report, then end the process. Only for the commit breach."""
        if not self._hard_abort:
            logger.error("[memguard] commit floor crossed; hard_abort is off, "
                         "recording only: %s", reason)
            return
        with self._lock:
            self._hard_aborted = True
        logger.critical(
            "[memguard] HARD ABORT (%s): %s -- ending pid %d with SIGKILL so "
            "Windows gets its commit back before it kills the desktop",
            self._label, reason, os.getpid(),
        )
        write_abort_report(reason, snap, self.report())
        _kill_self()

    # -- readers -----------------------------------------------------------

    @property
    def breached(self) -> Optional[str]:
        with self._lock:
            return self._breach

    @property
    def commit_breached(self) -> Optional[str]:
        with self._lock:
            return self._commit_breach

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "samples": self._samples,
                "peak_vram_occupancy_pct": self._peak_occ,
                "peak_sample": self._peak_sample.to_dict() if self._peak_sample else None,
                "min_host_available_gib": self._min_host,
                "min_win_commit_available_gib": self._min_commit,
                "breach": self._breach,
                "commit_breach": self._commit_breach,
                "hard_aborted": self._hard_aborted,
                "vram_ceiling_pct": self._ceiling,
                "host_floor_gib": self._floor,
                "win_commit_floor_gib": self._commit_floor,
            }


def write_abort_report(reason: str, snap: MemorySample,
                       watchdog: Dict[str, Any]) -> Optional[str]:
    """Best-effort JSON at ``REACTOR_TRAIN_ABORT_FILE`` (or the default).

    Returns the path written, or None. Never raises: the caller is about
    to SIGKILL itself and a failure to write must not change that.
    """
    path = (os.environ.get("REACTOR_TRAIN_ABORT_FILE") or "").strip() \
        or DEFAULT_ABORT_REPORT
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "ts": time.time(),
                "pid": os.getpid(),
                "reason": reason,
                "sample": snap.to_dict(),
                "watchdog": watchdog,
            }, fh, indent=2)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memguard] abort report not written: %s", exc)
        return None


def _kill_self() -> None:
    """SIGKILL this process. Module-level so a test can replace it.

    ``os._exit`` would also skip ``finally`` blocks, but it still returns
    through the C runtime; SIGKILL is delivered by the kernel and cannot
    be caught, masked, or delayed by a thread holding the GIL.
    """
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass
    os.kill(os.getpid(), signal.SIGKILL)


class PageCacheValve:
    """Stop a checkpoint stream from pinning the guest's RAM against Windows.

    Measured 2026-09-04 23:16 on the 30B: transformers' loader keeps every
    safetensors shard MAPPED for the whole load. ``Mapped:`` in
    ``/proc/meminfo`` grew one shard (~3.7 GiB) every 3 s and never fell,
    while ``MemAvailable`` stayed at 46 GiB because clean file pages are
    "available" -- and Windows charged every one of them to vmmemWSL. With
    the guest capped at 47 GiB and WDDM reserving a further ~30 GiB of
    backing store for the VRAM the model occupies, the load lands at ~97 of
    a 100.7 GiB commit limit, and the in-process guard has to kill it.

    A root-free ``posix_fadvise(DONTNEED)`` from another process returned
    43 GiB to Windows in one pass -- but only for UNMAPPED cache. Mapped
    pages belong to the loader, so this runs inside it: every tick it
    ``madvise(MADV_PAGEOUT)``s each mapped shard range (the mapping is
    read-only, a later touch simply refaults from the file), then fadvises
    the files so the pages actually leave the cache. The loader reads each
    tensor once, so nothing is refaulted in practice.

    Linux only (``/proc/self/maps``); a no-op that reports so elsewhere.
    Never raises out of the thread.
    """

    #: <sys/mman.h>, Linux >= 5.4. PAGEOUT reclaims the pages -- a clean file
    #: page is dropped, a modified private page is written to swap first --
    #: so unlike DONTNEED it can never discard a write. The loader maps its
    #: shards rw-private, so that distinction is the safety margin.
    MADV_PAGEOUT = 21

    def __init__(self, *, suffixes: Tuple[str, ...] = (".safetensors", ".bin", ".pt", ".gguf"),
                 interval_s: float = 2.0, label: str = "load") -> None:
        self._suffixes = tuple(suffixes)
        self._interval = max(0.25, float(interval_s))
        self._label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ticks = 0
        self._advised_bytes = 0
        self._peak_mapped_bytes = 0
        self._files: set = set()
        self._libc: Any = None
        self._enabled = os.path.exists("/proc/self/maps")

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "PageCacheValve":
        if self._thread is not None or not self._enabled:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"pagecache-valve-{self._label}",
                                        daemon=True)
        self._thread.start()
        logger.info("[memguard] page-cache valve open: unmapping %s every %.1fs",
                    "/".join(s.lstrip(".") for s in self._suffixes), self._interval)
        return self

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=self._interval + 5.0)
        # One last pass so the pages the final shard left behind go too.
        if self._enabled:
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                logger.debug("[memguard] valve final pass failed", exc_info=True)

    def __enter__(self) -> "PageCacheValve":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                logger.debug("[memguard] valve tick failed", exc_info=True)
            self._stop.wait(self._interval)

    #: A file mapping at least this large that is not executable and not a
    #: shared library is a checkpoint shard whatever it is called.
    LARGE_FILE_BYTES = 256 * 1024 * 1024

    def is_checkpoint_mapping(self, path: str, perms: str, length: int) -> bool:
        """Does this ``/proc/self/maps`` entry belong to a checkpoint file?

        Three tests, because the path alone lied: the hub stores shards as
        ``snapshots/<rev>/model-00001-of-00016.safetensors`` -> symlink ->
        ``blobs/<sha256>``, and the kernel reports the RESOLVED blob path,
        which carries no suffix at all. Measured 2026-09-04 23:19: the
        suffix-only valve saw 0 mappings while 44 GiB of shards sat mapped.
        """
        if path.endswith(self._suffixes):
            return True
        if "/blobs/" in path:  # huggingface_hub cache layout
            return True
        if "x" in perms or ".so" in path.rsplit("/", 1)[-1]:
            return False
        return length >= self.LARGE_FILE_BYTES and path.startswith("/")

    def mapped_ranges(self) -> List[Tuple[int, int, str]]:
        """``(start, end, path)`` for every mapping of a checkpoint file."""
        out: List[Tuple[int, int, str]] = []
        try:
            with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split(None, 5)
                    if len(parts) < 6:
                        continue
                    path = parts[5].strip()
                    if not path.startswith("/") or path.endswith(" (deleted)"):
                        continue
                    try:
                        start_s, end_s = parts[0].split("-")
                        start, end = int(start_s, 16), int(end_s, 16)
                    except ValueError:
                        continue
                    if self.is_checkpoint_mapping(path, parts[1], end - start):
                        out.append((start, end, path))
        except OSError:
            pass
        return out

    def _madvise(self, start: int, length: int) -> bool:
        if self._libc is None:
            import ctypes  # noqa: PLC0415
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.madvise.restype = ctypes.c_int
            self._libc = libc
        return self._libc.madvise(start, length, self.MADV_PAGEOUT) == 0

    def _tick(self) -> None:
        ranges = self.mapped_ranges()
        mapped = sum(end - start for start, end, _ in ranges)
        advised = 0
        for start, end, _path in ranges:
            if self._madvise(start, end - start):
                advised += end - start
        for path in {p for _, _, p in ranges}:
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                self._files.add(path)
            except (OSError, AttributeError):
                pass
        with self._lock:
            self._ticks += 1
            self._advised_bytes += advised
            self._peak_mapped_bytes = max(self._peak_mapped_bytes, mapped)

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "ticks": self._ticks,
                "advised_gib": round(self._advised_bytes / GIB, 2),
                "peak_mapped_gib": round(self._peak_mapped_bytes / GIB, 2),
                "files": len(self._files),
            }


def cap_cuda_allocator(fraction: Optional[float] = None) -> Optional[float]:
    """Refuse CUDA allocations past ``fraction`` of DEVICE memory.

    Returns the fraction applied, or None when torch/CUDA is unavailable.
    Never raises. Call it once, after ``import torch`` and before the
    first allocation, in every process that loads a model on this box --
    see ``DEFAULT_CUDA_ALLOCATOR_FRACTION`` for why: the driver here will
    otherwise page a model that does not fit into Windows' commit, and
    that is the spill that took the desktop down on 2026-09-04.

    The cap is an allocator policy, not a driver setting, so it holds
    regardless of the NVIDIA "Sysmem Fallback Policy" -- which should ALSO
    be set to "Prefer No Sysmem Fallback", because cuBLAS, NCCL and any
    non-torch cudaMalloc bypass this cap.
    """
    frac = (
        fraction if fraction is not None
        else _env_float("REACTOR_TRAIN_CUDA_ALLOCATOR_FRACTION",
                        DEFAULT_CUDA_ALLOCATOR_FRACTION)
    )
    if not (0.0 < frac <= 1.0):
        logger.warning("[memguard] allocator fraction %r out of (0, 1]; "
                       "not applied", frac)
        return None
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return None
        torch.cuda.set_per_process_memory_fraction(frac)
        total = torch.cuda.get_device_properties(0).total_memory / GIB
        logger.info("[memguard] CUDA caching allocator capped at %.0f%% of "
                    "device memory (%.1f of %.1f GiB); allocations past it "
                    "raise OOM instead of spilling to host RAM",
                    frac * 100.0, total * frac, total)
        return frac
    except Exception as exc:  # noqa: BLE001
        logger.warning("[memguard] allocator cap not applied: %s", exc)
        return None


# ---------------------------------------------------------------------------
# The degradation ladder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rung:
    """One attempt's memory settings, and why they are what they are."""

    name: str
    num_generations: int
    max_completion_length: int
    note: str

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "num_generations": self.num_generations,
            "max_completion_length": self.max_completion_length,
        }


def _divisible_generations(n: int, global_batch: int) -> int:
    """Largest g <= n, g >= 2, that divides the global batch.

    TRL requires the global batch to be a whole number of groups. A rung
    that halved ``num_generations`` into an indivisible value would fail
    in the config constructor rather than at the allocation it was meant
    to relieve — a fallback that cannot even be tried.
    """
    for g in range(min(n, global_batch), 1, -1):
        if global_batch % g == 0:
            return g
    return 2


def accumulation_for_groups(
    num_generations: int,
    *,
    per_device_batch: int = 1,
    requested_accum: int = 8,
    device_count: int = 1,
) -> int:
    """Smallest accumulation >= ``requested_accum`` whose generation batch
    holds a whole number of groups.

    The COMPANION of :func:`_divisible_generations`. That one bends the
    group to fit a fixed batch, which is what a memory fallback needs --
    the batch is the thing being held down. This one bends the batch to
    fit a chosen group, which is what raising ``num_generations`` for
    sample efficiency needs: the group size is the thing being asked for.

    TRL's constraint is the same in both directions -- the generation
    batch (``per_device_batch * accumulation * device_count``) must be an
    exact multiple of ``num_generations``, or ``GRPOConfig`` raises in its
    constructor. Raising ``num_generations`` to 16 against the historical
    accumulation of 8 gives a generation batch of 8, and 8 % 16 != 0: the
    run would die at construction, before a single byte of VRAM was
    allocated, with a message about batch shapes rather than about the
    knob that was actually changed.

    NEVER raises; a nonsensical input returns ``requested_accum``.
    """
    import math  # noqa: PLC0415
    try:
        g = int(num_generations)
        per = max(1, int(per_device_batch)) * max(1, int(device_count))
        accum = max(1, int(requested_accum))
        if g <= 1:
            return accum
        # need: per * accum % g == 0  ->  accum % (g / gcd(per, g)) == 0
        step = g // math.gcd(per, g)
        fitted = ((accum + step - 1) // step) * step
        if fitted != accum:
            logger.info(
                "[memguard] accumulation %d -> %d so the generation batch "
                "(%d x %d = %d) is a whole number of %d-completion groups",
                accum, fitted, per, fitted, per * fitted, g,
            )
        return fitted
    except Exception:  # noqa: BLE001 -- a sizing helper must never break a run
        logger.debug("[memguard] accumulation_for_groups fell back",
                     exc_info=True)
        return max(1, int(requested_accum or 1))


def build_ladder(
    *,
    num_generations: int = 4,
    max_completion_length: int = 256,
    global_batch: int = 8,
    device_count: int = 1,
) -> List[Rung]:
    """Attempts in order, each strictly cheaper than the last.

    Only the KV-cache terms move. The base weights are the model and the
    activations are already checkpointed, so ``num_generations`` and
    ``max_completion_length`` are the two knobs with real travel — see the
    module docstring for why FSDP is not on this list at
    ``device_count == 1``.

    Shrinking ``num_generations`` is placed AFTER shrinking the completion
    budget because it costs signal: GRPO's advantage is computed within a
    group, and a group of 2 estimates the baseline from a single sibling.
    Shorter completions truncate some candidates; a smaller group degrades
    every gradient. Cheaper first, but *this* cheap only when it must be.
    """
    mcl = max(64, int(max_completion_length))
    requested = max(2, int(num_generations))
    # The FIRST rung is subject to the divisibility rule too. A ladder whose
    # top rung cannot be constructed does not degrade gracefully — it fails
    # in GRPOConfig before a single byte is allocated, and the operator sees
    # a config error where they asked for a memory strategy.
    ng = _divisible_generations(requested, global_batch)
    if ng != requested:
        logger.warning(
            "[memguard] num_generations %d does not divide the global batch "
            "of %d; using %d. TRL needs a whole number of groups per batch.",
            requested, global_batch, ng,
        )
    rungs = [Rung("as-configured", ng, mcl, "the requested settings")]

    half = max(64, mcl // 2)
    if half < mcl:
        rungs.append(Rung(
            "short-completions", ng, half,
            "halve the completion budget: the KV cache is linear in it and "
            "it costs only the longest candidates",
        ))
    quarter = max(64, mcl // 4)
    if quarter < half:
        rungs.append(Rung(
            "shorter-completions", ng, quarter,
            "quarter the completion budget",
        ))

    smaller = _divisible_generations(max(2, ng // 2), global_batch)
    if smaller < ng:
        rungs.append(Rung(
            "small-group", smaller, quarter,
            f"drop to {smaller} generations per prompt — degrades the "
            "advantage baseline, so it is the last rung before refusing",
        ))
    if device_count > 1:
        # Left deliberately empty of an FSDP rung even here: this repo has
        # never run multi-GPU, and an untested sharding path presented as a
        # fallback is worse than an honest refusal. Recorded so the next
        # reader knows it was considered, not forgotten.
        logger.info(
            "[memguard] %d devices visible; FSDP sharding is a real option "
            "at this world size but is not implemented or tested here",
            device_count,
        )
    return rungs


def is_oom(exc: BaseException) -> bool:
    """Is this the allocator giving up?

    Matches by message as well as type: bitsandbytes, Triton and the GPTQ
    kernels each raise their own class on an allocation failure, and only
    torch's is a ``torch.cuda.OutOfMemoryError``.
    """
    name = type(exc).__name__
    if name in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    text = f"{name}: {exc}".lower()
    return any(s in text for s in (
        "out of memory", "cuda error: out of memory", "cublas_status_alloc_failed",
        "no kernel image", "failed to allocate",
    ))


def free_cuda_memory() -> None:
    """Return cached blocks between rungs.

    Without this the next attempt starts against an allocator still
    holding the previous one's reserved-but-unused segments, so a rung
    that would have fit reports OOM and the ladder walks past a working
    configuration.
    """
    try:
        import gc  # noqa: PLC0415
        import torch  # noqa: PLC0415
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        logger.debug("[memguard] could not free CUDA memory", exc_info=True)


__all__ = [
    "Admission",
    "MemorySample",
    "MemoryWatchdog",
    "Rung",
    "build_ladder",
    "check_admission",
    "free_cuda_memory",
    "gpu_occupancy_pct",
    "is_oom",
    "sample",
    "sample_gpu",
    "sample_host",
    "sample_windows_commit",
    "under_wsl",
    "cap_cuda_allocator",
    "PageCacheValve",
    "write_abort_report",
    "DEFAULT_WIN_COMMIT_FLOOR_GIB",
    "DEFAULT_CUDA_ALLOCATOR_FRACTION",
]
