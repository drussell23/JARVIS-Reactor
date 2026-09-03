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

import logging
import os
import shutil
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

_NVIDIA_SMI_TIMEOUT_S = 5.0


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


def sample() -> MemorySample:
    """One reading of both devices. Never raises."""
    util, used, total = sample_gpu()
    avail, host_total = sample_host()
    return MemorySample(
        ts=time.time(), gpu_util_pct=util, vram_used_mib=used,
        vram_total_mib=total, host_available_gib=avail,
        host_total_gib=host_total,
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

    Tripping does not kill anything. It records that a ceiling was crossed
    and calls ``on_breach`` once; the training thread decides what that
    means. A watchdog that unilaterally killed the process would lose the
    traceback that says which allocation actually failed.
    """

    def __init__(
        self,
        *,
        interval_s: float = 2.0,
        vram_ceiling_pct: Optional[float] = None,
        host_floor_gib: Optional[float] = None,
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
        self._on_breach = on_breach
        self._label = label
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_occ: Optional[float] = None
        self._peak_sample: Optional[MemorySample] = None
        self._min_host: Optional[float] = None
        self._breach: Optional[str] = None
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
            "host, every %.1fs", self._ceiling, self._floor, self._interval,
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
        snap = sample()
        occ = snap.vram_occupancy_pct
        breach: Optional[str] = None
        with self._lock:
            self._samples += 1
            if occ is not None and (self._peak_occ is None or occ > self._peak_occ):
                self._peak_occ, self._peak_sample = occ, snap
            if snap.host_available_gib is not None and (
                self._min_host is None or snap.host_available_gib < self._min_host
            ):
                self._min_host = snap.host_available_gib
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
        if breach is not None:
            logger.warning("[memguard] BREACH: %s", breach)
            if self._on_breach is not None:
                try:
                    self._on_breach(breach, snap)
                except Exception:  # noqa: BLE001
                    logger.debug("[memguard] on_breach raised", exc_info=True)

    # -- readers -----------------------------------------------------------

    @property
    def breached(self) -> Optional[str]:
        with self._lock:
            return self._breach

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "samples": self._samples,
                "peak_vram_occupancy_pct": self._peak_occ,
                "peak_sample": self._peak_sample.to_dict() if self._peak_sample else None,
                "min_host_available_gib": self._min_host,
                "breach": self._breach,
                "vram_ceiling_pct": self._ceiling,
                "host_floor_gib": self._floor,
            }


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
]
