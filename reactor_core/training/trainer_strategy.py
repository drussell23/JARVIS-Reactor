"""Which trainer runs a cycle is a CHOICE, resolved at run time.

``UnifiedTrainingPipeline`` hardcoded ``AsyncTrainer`` (supervised
fine-tuning) as the only way to train, with a DPO refinement bolted on
after it. GRPO -- the method the flywheel actually needs, because the
corpus is graded rollouts rather than matched preference pairs -- lives in
``scripts/run_grpo_training.py`` and nothing in the pipeline could reach
it. Two trainers, one automated loop, and they were not connected.

## The shape

A strategy is a name bound to an async callable taking a
:class:`TrainerRequest` and returning a :class:`TrainerOutcome`. The
registry is the only thing that knows the mapping, so adding a method is a
registration rather than a branch in the pipeline, and the pipeline never
names a trainer.

Resolution order, each step deliberate:

1. an explicit ``strategy`` on the request -- a caller that knows;
2. ``REACTOR_TRAINING_STRATEGY`` -- the operator, per box;
3. :data:`DEFAULT_STRATEGY` -- SFT, so an unconfigured pipeline behaves
   exactly as it did before this module existed.

An unknown name RAISES rather than falling back. A typo that silently
trained the wrong way would be discovered only in the artifact, days
later, and the whole point of naming a method is that the name is honoured.

## Why GRPO composes the RUNNER rather than importing its internals

``run_grpo_training.py`` is not a thin CLI. It owns the admission gate,
the corpus gate, the contrast filter, the prompt budget, the degradation
ladder and -- since 2026-09-05 -- the subprocess isolation that gives each
rung a clean CUDA context. Importing ``train_with_ladder`` directly would
bypass the first two and re-implement the rest inside a process that also
holds the pipeline. Running it as a child gets all of it, and the ladder's
own isolation keeps a failed rung from poisoning the pipeline's process.
The contract between them is the one both halves already speak: exit codes
plus the ``--json-out`` report.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: Env override for the strategy name (see module docstring for the order).
ENV_STRATEGY = "REACTOR_TRAINING_STRATEGY"
#: Env override for the GRPO runner's location, for a non-standard checkout.
ENV_GRPO_RUNNER = "REACTOR_GRPO_RUNNER"
#: Extra argv for the GRPO runner, whitespace-separated. The pipeline passes
#: model/telemetry/output; everything else a box needs is expressed here
#: rather than by widening this module every time the runner gains a flag.
ENV_GRPO_ARGS = "REACTOR_GRPO_EXTRA_ARGS"

#: SFT: what the pipeline did before strategies existed, and what it still
#: does when nothing is configured.
STRATEGY_SFT = "lora_sft"
STRATEGY_GRPO = "grpo"
DEFAULT_STRATEGY = STRATEGY_SFT

#: The runner's exit codes, mirrored so the outcome can say WHY without the
#: caller parsing prose. Kept in sync by `tests/test_trainer_strategy.py`,
#: which imports the runner and compares.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_LADDER_EXHAUSTED = 3


@dataclass(frozen=True)
class TrainerRequest:
    """Everything a strategy needs, and nothing about HOW it trains."""

    base_model: str
    output_dir: Path
    telemetry_dir: Optional[Path] = None
    #: An already-built dataset, for strategies that take one (SFT does;
    #: GRPO reads the corpus itself, because its gate selects the groups).
    train_dataset: Any = None
    eval_dataset: Any = None
    strategy: str = ""
    #: Strategy-specific knobs. A strategy ignores what it does not know,
    #: so one config can serve several without conditionals at the caller.
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerOutcome:
    """What happened, in terms every caller can act on."""

    ok: bool
    strategy: str
    adapter_path: Optional[Path] = None
    exit_code: Optional[int] = None
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    #: True when the box DECLINED (busy card, thin corpus) rather than
    #: failed. A scheduler should retry a refusal and investigate a failure.
    refused: bool = False

    def summary(self) -> str:
        head = "trained" if self.ok else ("refused" if self.refused else "failed")
        where = f" -> {self.adapter_path}" if self.adapter_path else ""
        return f"[{self.strategy}] {head}{where}: {self.reason}".strip()


TrainerFn = Callable[[TrainerRequest], Awaitable[TrainerOutcome]]

_REGISTRY: Dict[str, TrainerFn] = {}


def register(name: str, fn: TrainerFn, *, replace: bool = False) -> None:
    """Bind a strategy name. Re-registering without ``replace`` raises:
    two trainers quietly answering to one name is a coin toss."""
    key = name.strip().lower()
    if not key:
        raise ValueError("a strategy needs a name")
    if key in _REGISTRY and not replace:
        raise ValueError(f"strategy {key!r} is already registered")
    _REGISTRY[key] = fn


def available() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_name(request: Optional[TrainerRequest] = None) -> str:
    """Request, then env, then the default."""
    if request is not None and request.strategy.strip():
        return request.strategy.strip().lower()
    env = (os.environ.get(ENV_STRATEGY, "") or "").strip().lower()
    return env or DEFAULT_STRATEGY


def resolve(name: str) -> TrainerFn:
    key = (name or "").strip().lower()
    fn = _REGISTRY.get(key)
    if fn is None:
        raise KeyError(
            f"unknown training strategy {key!r}; registered: {available()}"
        )
    return fn


async def run(request: TrainerRequest) -> TrainerOutcome:
    """Resolve and run. The pipeline's whole interface to trainer choice."""
    name = resolve_name(request)
    fn = resolve(name)
    logger.info("[TrainerStrategy] running %r for %s", name, request.base_model)
    return await fn(request)


# ---------------------------------------------------------------------------
# GRPO — composes the runner as a child process
# ---------------------------------------------------------------------------


def _runner_path() -> Path:
    override = (os.environ.get(ENV_GRPO_RUNNER, "") or "").strip()
    if override:
        return Path(override)
    # reactor_core/training/trainer_strategy.py -> <repo>/scripts/...
    return Path(__file__).resolve().parents[2] / "scripts" / "run_grpo_training.py"


def _extra_args() -> Tuple[str, ...]:
    return tuple((os.environ.get(ENV_GRPO_ARGS, "") or "").split())


def build_grpo_argv(request: TrainerRequest, *, report_path: Path) -> Tuple[str, ...]:
    """The child's command line. Separated so a test can read it without
    starting a 30B."""
    argv = [
        sys.executable, "-u", str(_runner_path()),
        "--model", request.base_model,
        "--output-dir", str(request.output_dir),
        "--json-out", str(report_path),
    ]
    if request.telemetry_dir is not None:
        argv += ["--telemetry-dir", str(request.telemetry_dir)]
    for flag, key in (
        ("--num-generations", "num_generations"),
        ("--max-completion-length", "max_completion_length"),
        ("--max-prompt-tokens", "max_prompt_tokens"),
        ("--epochs", "epochs"),
        ("--max-steps", "max_steps"),
        ("--gradient-accumulation-steps", "gradient_accumulation_steps"),
    ):
        value = request.options.get(key)
        if value is not None:
            argv += [flag, str(value)]
    if request.options.get("train_truncated"):
        argv.append("--train-truncated")
    argv += list(_extra_args())
    return tuple(argv)


def _adapter_from_report(report: Dict[str, Any], fallback: Path) -> Optional[Path]:
    """Where the child says it saved. An attempt's ``saved_to`` is written
    only after a step actually ran, so it is preferred over the configured
    output dir, which exists whether or not anything landed in it."""
    for attempt in reversed(report.get("attempts") or []):
        saved = attempt.get("saved_to")
        if saved:
            return Path(saved)
    top = report.get("output_dir")
    if top:
        return Path(top)
    return fallback if fallback.exists() else None


async def grpo_strategy(request: TrainerRequest) -> TrainerOutcome:
    """Run the GRPO runner and translate its exit code into an outcome."""
    runner = _runner_path()
    if not runner.is_file():
        return TrainerOutcome(
            ok=False, strategy=STRATEGY_GRPO,
            reason=f"GRPO runner not found at {runner} (set {ENV_GRPO_RUNNER})",
        )

    handle, tmp = tempfile.mkstemp(prefix="grpo-strategy-", suffix=".json")
    os.close(handle)
    report_path = Path(tmp)
    argv = build_grpo_argv(request, report_path=report_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Drain as it arrives: a child whose pipe fills stops making
        # progress, and a training run writes for hours.
        assert proc.stdout is not None
        tail: list = []
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                logger.info("[grpo] %s", line[:400])
                tail.append(line)
                del tail[:-40]
        code = await proc.wait()

        report: Dict[str, Any] = {}
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a child killed by the guard writes none
            logger.debug("[grpo] no parseable report at %s", report_path,
                         exc_info=True)

        metrics = {
            k: report.get(k)
            for k in ("status", "corpus", "contrast_filtered", "batching",
                      "admission", "ladder")
            if k in report
        }
        if code == EXIT_OK:
            adapter = _adapter_from_report(report, request.output_dir)
            if adapter is None:
                return TrainerOutcome(
                    ok=False, strategy=STRATEGY_GRPO, exit_code=code,
                    metrics=metrics,
                    reason="runner exited 0 but reported no saved adapter",
                )
            return TrainerOutcome(
                ok=True, strategy=STRATEGY_GRPO, adapter_path=adapter,
                exit_code=code, metrics=metrics,
                reason=f"adapter saved to {adapter}",
            )
        if code == EXIT_REFUSED:
            return TrainerOutcome(
                ok=False, strategy=STRATEGY_GRPO, exit_code=code, refused=True,
                metrics=metrics,
                reason=report.get("refused")
                or "a gate declined (busy card, or a corpus with no contrast)",
            )
        if code == EXIT_LADDER_EXHAUSTED:
            return TrainerOutcome(
                ok=False, strategy=STRATEGY_GRPO, exit_code=code, refused=True,
                metrics=metrics,
                reason="every rung ran out of memory; this model does not fit here",
            )
        return TrainerOutcome(
            ok=False, strategy=STRATEGY_GRPO, exit_code=code, metrics=metrics,
            reason=(report.get("error") or "; ".join(tail[-3:])
                    or f"runner exited {code}")[:400],
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — a strategy reports, never explodes
        logger.exception("[grpo] strategy failed to run the child")
        return TrainerOutcome(
            ok=False, strategy=STRATEGY_GRPO,
            reason=f"{type(exc).__name__}: {exc}"[:400],
        )
    finally:
        try:
            report_path.unlink()
        except OSError:
            pass


register(STRATEGY_GRPO, grpo_strategy)
