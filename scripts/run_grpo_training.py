#!/usr/bin/env python3
"""Run one GRPO pass over the O+V corpus — the missing driver.

``grpo_pipeline.build_trainer`` has existed and been tested for days with
**no caller anywhere in the repository**. Gate 3 passing on soak 19 is
what makes a driver worth having, and this is it.

Everything load-bearing is imported, never re-implemented:

* ``grpo_pipeline.build_trainer`` — dataset, config, model, adapter and
  reward wiring. This script does not assemble a trainer; it asks for one.
* ``grpo_preflight.analyse`` — the Gate 3 verdict, the same function and
  therefore the same answer the operator gets from the command line. A
  runner with its own opinion about corpus quality would drift from the
  gate and eventually train on something the gate refuses.
* ``memory_guard`` — admission, the live watchdog and the degradation
  ladder, shared with ``api.scheduler``'s VRAM gate.

## Order of operations, and why

Both gates run BEFORE the ML stack is imported. A refusal should cost a
second, not the forty it takes to bring torch, trl and transformers into
the process — and on a box where the corpus lives next to a running soak,
refusing is the common case, not the exception.

## Exit codes

``0`` trained; ``1`` error; ``2`` refused (corpus or hardware); ``3`` the
ladder was exhausted without a configuration that fit.

2 and 3 are distinct from 1 deliberately, the same way ``grpo_preflight``
separates them: "I looked and the answer is no" and "every rung OOMed"
are operational facts, while 1 means the runner itself broke. A caller
that cannot tell them apart cannot automate on top of this.

## What this will NOT do

It will not pick a model for you. The 30B MoE target must be the BASE
bf16 checkpoint (``Qwen/Qwen3-Coder-30B-A3B-Instruct``) so bnb-NF4 can
quantize it at load. The pre-quantized GPTQ mirror is not a substitute:
on transformers 5.16.1 its expert tensors load as MISSING and are newly
initialized (measured 2026-09-03 — ``model.layers.{0..47}.mlp.experts.*``),
and the loader dequantises to bf16 anyway (59.38 GiB for a "4-bit" 30B).
Training that checkpoint would train randomly-initialized experts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _default_num_generations() -> int:
    """The group size default, owned by grpo_pipeline.

    Wrapped so that a missing training extra degrades to TRL's own default
    instead of making ``--help`` unavailable.
    """
    try:
        from reactor_core.training.grpo_pipeline import (  # noqa: PLC0415
            default_num_generations,
        )
        return default_num_generations()
    except Exception:  # noqa: BLE001
        return 8


_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

logger = logging.getLogger("run_grpo_training")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_LADDER_EXHAUSTED = 3

#: Columns whose presence in the training dataset would mean the corpus's
#: own answers had been handed to the trainer. GRPO must generate its own
#: completions and be graded on those; a reference completion riding along
#: is train-on-test, and it would be invisible in the loss curve.
_LEAKAGE_COLUMNS = frozenset({
    "completion", "completions", "assistant_output", "response", "responses",
    "chosen", "rejected", "labels", "label", "target", "answer",
    "candidate", "candidates", "reference",
})


def _load_by_path(mod_name: str, path: Path) -> Any:
    """Import a module by path, bypassing ``reactor_core/__init__``.

    Same reason ``grpo_preflight`` does it: the package __init__ eagerly
    imports the ML stack, and both gates below must run before that cost
    is paid.
    """
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Gate 1 — the corpus
# ---------------------------------------------------------------------------


def corpus_gate(
    telemetry_dir: Path,
    *,
    min_group: int,
    min_spread: float,
    min_groups: int,
    trainable_only: bool,
) -> Dict[str, Any]:
    """Ask ``grpo_preflight`` whether this corpus is worth a GPU-hour."""
    import grpo_preflight  # noqa: PLC0415 — see _load_by_path's docstring

    verdict = grpo_preflight.analyse(
        telemetry_dir,
        min_group=min_group,
        trainable_only=trainable_only,
        min_spread=min_spread,
    )
    verdict["min_groups_required"] = min_groups
    verdict["passes"] = verdict["trainable_groups"] >= min_groups
    return verdict


# ---------------------------------------------------------------------------
# Gate 2 — the dataset actually handed to TRL
# ---------------------------------------------------------------------------


def validate_dataset(dataset: Any, *, reward_columns: Any) -> Dict[str, Any]:
    """Shape and leakage checks on the built dataset. Raises on failure.

    The corpus gate scores rows; this checks the *object* the trainer will
    iterate. They are different questions and both have been wrong before
    — a corpus can pass Gate 3 and still build a dataset whose reward
    columns are absent, in which case ``candidate_reward``'s historical
    nudge silently reads defaults for every row and the run looks healthy.
    """
    cols = set(getattr(dataset, "column_names", []) or [])
    report: Dict[str, Any] = {"columns": sorted(cols), "rows": len(dataset)}

    if "prompt" not in cols:
        raise ValueError(
            f"dataset has no 'prompt' column (got {sorted(cols)}) — GRPO "
            "generates from prompts and has nothing to do without it"
        )

    leaked = sorted(cols & _LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(
            f"dataset carries reference-answer column(s) {leaked}. GRPO "
            "must generate its own completions and be graded on those; "
            "shipping the corpus's own answers alongside is data leakage."
        )
    report["leakage_columns"] = leaked

    missing_reward = sorted(set(reward_columns) - cols)
    if missing_reward:
        raise ValueError(
            f"dataset is missing reward column(s) {missing_reward}. "
            "candidate_reward reads these through TRL's **kwargs; absent, "
            "every row silently scores on defaults."
        )
    report["reward_columns_present"] = sorted(set(reward_columns) & cols)

    if not len(dataset):
        raise ValueError("dataset is empty")

    prompts = dataset["prompt"]
    blank = sum(1 for p in prompts if not str(p or "").strip())
    if blank:
        raise ValueError(f"{blank} of {len(prompts)} prompts are blank")
    distinct = len(set(prompts))
    if distinct != len(prompts):
        # build_prompt_dataset deduplicates; a duplicate here means that
        # contract broke, and a repeated prompt silently reweights the epoch.
        raise ValueError(
            f"{len(prompts) - distinct} duplicate prompt(s) survived "
            "deduplication — the epoch would weight them twice"
        )
    report["distinct_prompts"] = distinct
    report["prompt_chars"] = {
        "min": min(len(str(p)) for p in prompts),
        "max": max(len(str(p)) for p in prompts),
        "mean": round(sum(len(str(p)) for p in prompts) / len(prompts), 1),
    }
    return report


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _make_guard_callback(watchdog: Any) -> Any:
    """A TrainerCallback that stops the run when the watchdog trips.

    This is the proactive half of the memory strategy. Waiting for the
    allocator to raise means the failure lands mid-backward with whatever
    fragmentation that leaves; stopping at the ceiling lets the runner
    descend the ladder from a clean allocator instead. The callback only
    ever asks the Trainer to stop — it does not kill the process, so the
    traceback and the step counter both survive.
    """
    from transformers import TrainerCallback  # noqa: PLC0415

    class _MemoryGuardCallback(TrainerCallback):
        def __init__(self) -> None:
            self.tripped: Optional[str] = None

        def _check(self, control: Any) -> Any:
            breach = watchdog.breached
            if breach and self.tripped is None:
                self.tripped = breach
                logger.warning(
                    "[runner] memory ceiling reached — asking the trainer to "
                    "stop so the next rung starts from a clean allocator: %s",
                    breach,
                )
                control.should_training_stop = True
            return control

        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            return self._check(control)

        def on_substep_end(self, args, state, control, **kwargs):  # noqa: ANN001
            return self._check(control)

    return _MemoryGuardCallback()


#: Consecutive fully-clipped steps before the truncation guard stops a run.
#: One could be an unlucky batch; two is a configuration verdict. At 11-18
#: minutes per step, stopping at two costs one step where continuing costs
#: the whole run.
DEFAULT_CLIPPED_PATIENCE = 2


def _env_int(name: str, default: int) -> int:
    """An int from the environment, or the default. Never raises."""
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except (TypeError, ValueError):
        return default


def _make_truncation_callback(*, mask_truncated: bool, patience: int = 0) -> Any:
    """Refuse to spend hours on steps whose loss has no tokens in it.

    With ``mask_truncated_completions=True`` TRL zeroes the completion mask
    for every row whose last token is not EOS or pad
    (``grpo_trainer.py``: ``completion_mask * (~is_truncated)``), and
    ``num_items_in_batch`` is that mask's sum. Measured 2026-09-05:
    ``completions/clipped_ratio`` was **1.0** at BOTH 8 and 256 completion
    tokens, and ``mean_terminated_length`` was 0 -- no rollout ever emitted
    EOS. Every row was therefore masked, the policy loss had zero
    contributing tokens, and the only gradient came from the MoE router's
    auxiliary term.

    That is the failure mode this exists for, and note WHY it was invisible:
    the step still logged a perfectly plausible ``loss`` (0.008183, which is
    exactly ``aux_loss * 1e-3``), so nothing looked wrong. A number attached
    to no learning is worse than an error.

    The guard therefore reads the ratio TRL already publishes rather than
    re-deriving truncation itself -- one definition, and it is the
    trainer's. It only arms when masking is actually on, because with
    ``mask_truncated_completions=False`` a clipped completion still carries
    gradient and a full clip ratio is merely a budget observation.
    """
    from transformers import TrainerCallback  # noqa: PLC0415

    limit = patience if patience > 0 else _env_int(
        "REACTOR_TRAIN_CLIPPED_PATIENCE", DEFAULT_CLIPPED_PATIENCE)

    class _TruncationCallback(TrainerCallback):
        def __init__(self) -> None:
            self.tripped: Optional[str] = None
            self.consecutive = 0
            self.last_ratio: Optional[float] = None

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not mask_truncated or not logs:
                return control
            ratio = logs.get("completions/clipped_ratio")
            try:
                ratio = float(ratio)
            except (TypeError, ValueError):
                return control
            self.last_ratio = ratio
            # 1.0 exactly: EVERY completion hit the ceiling, so every row is
            # masked. Anything less leaves at least one row contributing.
            if ratio < 1.0:
                self.consecutive = 0
                return control
            self.consecutive += 1
            logger.warning(
                "[runner] step %s: clipped_ratio=1.0 — every completion hit "
                "the %s-token ceiling, so mask_truncated_completions masks "
                "them ALL and the policy loss has no tokens (%d/%d)",
                getattr(state, "global_step", "?"),
                getattr(args, "max_completion_length", "?"),
                self.consecutive, limit,
            )
            if self.consecutive >= limit and self.tripped is None:
                self.tripped = (
                    f"{self.consecutive} consecutive step(s) with "
                    f"clipped_ratio=1.0 and mask_truncated_completions=True: "
                    f"every completion is masked out of the loss, so the "
                    f"policy gradient is empty. Raise --max-completion-length "
                    f"past {getattr(args, 'max_completion_length', '?')} until "
                    f"completions terminate, or set "
                    f"mask_truncated_completions=False to train on truncated "
                    f"text."
                )
                logger.error("[runner] STOPPING: %s", self.tripped)
                control.should_training_stop = True
            return control

    return _TruncationCallback()


def _dataset_len(trainer: Any) -> int:
    """Rows the trainer will actually iterate. NEVER raises."""
    try:
        return int(len(trainer.train_dataset))
    except Exception:  # noqa: BLE001
        return -1


def describe_adapter(trainer: Any) -> Dict[str, Any]:
    """Trainable-parameter census -- the adapter half of the memory profile.

    Peak VRAM from the watchdog answers "did it fit". This answers "what
    is it training", which is the question the MoE targeting bug turned
    on: a run whose adapters landed on 18,432 expert projections and one
    whose adapters landed on 192 attention projections both LOAD, and are
    told apart only by this census.

    ``adamw_states_gib`` is the optimiser's own footprint, which does not
    exist yet at dry-run time -- the optimiser is constructed on the first
    training step. It is projected here because it is the term that
    decides whether the real run fits, and a dry-run that omitted it would
    understate the peak it exists to predict.

    NEVER raises; a model that cannot be walked returns ``{"error": ...}``.
    """
    try:
        model = getattr(trainer, "model", None)
        if model is None:
            return {"error": "trainer has no model"}
        total = 0
        trainable = 0
        adapted: Dict[str, int] = {}
        for name, param in model.named_parameters():
            count = int(param.numel())
            total += count
            if not param.requires_grad:
                continue
            trainable += count
            if "lora_" not in name:
                continue
            parts = name.split(".")
            for index, segment in enumerate(parts):
                if segment.startswith("lora_") and index:
                    key = parts[index - 1]
                    adapted[key] = adapted.get(key, 0) + 1
                    break
        gib = float(1 << 30)
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100.0 * trainable / max(1, total), 4),
            "adapter_bf16_gib": round(trainable * 2 / gib, 3),
            # AdamW: exp_avg + exp_avg_sq + fp32 master copy, 4 bytes each.
            "projected_adamw_gib": round(trainable * 12 / gib, 3),
            "adapted_projections": dict(sorted(adapted.items())),
            "adapted_module_count": sum(adapted.values()) // 2,
        }
    except Exception as exc:  # noqa: BLE001 -- profiling must not kill a run
        logger.debug("[runner] adapter census failed", exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def train_with_ladder(
    *,
    model_id: str,
    telemetry_dir: Path,
    output_dir: Path,
    ladder: List[Any],
    guard: Any,
    trainable_only: bool,
    max_prompts: Optional[int],
    only_prompts: Optional[List[str]],
    gptq_backend: str,
    use_qlora: bool,
    config_overrides: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    """Walk the ladder until one rung completes. Returns a report."""
    from reactor_core.training import grpo_pipeline  # noqa: PLC0415

    attempts: List[Dict[str, Any]] = []
    for index, rung in enumerate(ladder):
        logger.info(
            "[runner] rung %d/%d '%s': num_generations=%d "
            "max_completion_length=%d — %s",
            index + 1, len(ladder), rung.name, rung.num_generations,
            rung.max_completion_length, rung.note,
        )
        attempt: Dict[str, Any] = {
            "rung": rung.name,
            "num_generations": rung.num_generations,
            "max_completion_length": rung.max_completion_length,
        }
        watchdog = guard.MemoryWatchdog(label=f"rung{index}")
        started = time.time()
        try:
            with watchdog:
                trainer = grpo_pipeline.build_trainer(
                    model_id,
                    telemetry_dir,
                    str(output_dir),
                    trainable_only=trainable_only,
                    max_prompts=max_prompts,
                    only_prompts=only_prompts,
                    use_qlora=use_qlora,
                    gptq_backend=gptq_backend,
                    **rung.as_kwargs(),
                    **config_overrides,
                )
                shape = validate_dataset(
                    trainer.train_dataset,
                    reward_columns=grpo_pipeline._REWARD_COLUMNS,
                )
                attempt["dataset"] = shape
                logger.info(
                    "[runner] dataset validated: %d prompt(s), columns %s",
                    shape["rows"], shape["columns"],
                )
                if dry_run:
                    attempt["status"] = "dry-run"
                    attempt["watchdog"] = watchdog.report()
                    attempt["adapter"] = describe_adapter(trainer)
                    attempt["dataset_rows"] = _dataset_len(trainer)
                    attempts.append(attempt)
                    logger.info("[runner] dry-run adapter: %s",
                                attempt["adapter"])
                    return {"status": "dry-run", "attempts": attempts}

                callback = _make_guard_callback(watchdog)
                trainer.add_callback(callback)
                # The trainer's OWN view of whether truncation is masked --
                # never a second opinion about the config it was built with.
                truncation = _make_truncation_callback(
                    mask_truncated=bool(getattr(
                        trainer.args, "mask_truncated_completions", False)),
                )
                trainer.add_callback(truncation)
                result = trainer.train()
                attempt["metrics"] = getattr(result, "metrics", None)
                attempt["global_step"] = int(
                    getattr(trainer.state, "global_step", 0) or 0
                )
                attempt["guard_tripped"] = callback.tripped
                attempt["truncation_tripped"] = truncation.tripped
                attempt["clipped_ratio"] = truncation.last_ratio
        except Exception as exc:  # noqa: BLE001
            attempt["error"] = f"{type(exc).__name__}: {exc}"[:500]
            attempt["watchdog"] = watchdog.report()
            attempt["elapsed_s"] = round(time.time() - started, 1)
            attempts.append(attempt)
            if not guard.is_oom(exc):
                logger.error("[runner] rung '%s' failed, and not for memory — "
                             "descending the ladder would not help", rung.name)
                raise
            logger.warning("[runner] rung '%s' ran out of memory: %s",
                           rung.name, attempt["error"])
            guard.free_cuda_memory()
            continue

        attempt["watchdog"] = watchdog.report()
        attempt["elapsed_s"] = round(time.time() - started, 1)

        if attempt.get("guard_tripped"):
            # Stopped early at the ceiling rather than OOMing. That is a
            # rung that does not fit, not a completed run — descending is
            # the whole point of having tripped.
            attempt["status"] = "stopped-at-ceiling"
            attempts.append(attempt)
            guard.free_cuda_memory()
            continue

        if not attempt["global_step"]:
            # `trainer.train()` returns happily after zero steps when the
            # epoch iterator is empty. Saving that adapter would ship
            # untrained weights labelled as a training result.
            attempt["status"] = "no-steps"
            attempts.append(attempt)
            raise RuntimeError(
                "trainer.train() completed without taking a single "
                "optimisation step — the dataset filled no batch"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
        attempt["status"] = "trained"
        attempt["saved_to"] = str(output_dir)
        attempts.append(attempt)
        return {"status": "trained", "attempts": attempts}

    return {"status": "ladder-exhausted", "attempts": attempts}


# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="BASE bf16 checkpoint; see the module docstring on "
                         "why the GPTQ mirror is not a substitute")
    ap.add_argument("--telemetry-dir", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--num-generations", type=int,
                    default=_default_num_generations(),
                    help="completions per prompt; the within-group contrast "
                         "IS the GRPO signal, so this is the main lever on a "
                         "small corpus")
    ap.add_argument(
        "--max-completion-length", type=int, default=512,
        help="tokens per completion. 512, not 256, because at 256 EVERY "
             "rollout hit the ceiling (clipped_ratio 1.0, "
             "mean_terminated_length 0) and mask_truncated_completions then "
             "masked all of them out of the loss. Generation is ~1.645s per "
             "decode step and 64%% of the step, so this knob is the run's "
             "wall clock: the truncation callback stops the run if the "
             "ceiling is still never reached.",
    )
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--min-spread", type=float, default=0.01,
                    help="Gate 3's floor on a group's reward spread")
    ap.add_argument("--min-groups", type=int, default=1)
    ap.add_argument("--min-group-size", type=int, default=2)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--include-untrainable", action="store_true")
    ap.add_argument("--gptq-backend", default="")
    ap.add_argument("--no-qlora", action="store_true")
    ap.add_argument("--skip-corpus-gate", action="store_true",
                    help="train regardless of Gate 3. For wiring smoke "
                         "tests only — a flat corpus amplifies noise to "
                         "full scale, it does not train weakly.")
    ap.add_argument(
        "--all-prompts", action="store_true",
        help="train on every prompt the row filter admits, not only the "
             "contrast-bearing groups the corpus gate selected. Measured "
             "2026-09-05: 242 prompts vs 27, which at 656.8s per optimiser "
             "step is 44.2 hours against 4.9. The default narrows to the "
             "gate's selection; this restores the old behaviour.",
    )
    ap.add_argument("--skip-admission", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate everything, take no step")
    ap.add_argument("--json-out", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    guard = _load_by_path(
        "_runner_memory_guard",
        _REPO / "reactor_core" / "training" / "memory_guard.py",
    )
    import grpo_preflight  # noqa: PLC0415

    telemetry_dir = (
        Path(args.telemetry_dir) if args.telemetry_dir
        else grpo_preflight._default_telemetry_dir()
    )
    output_dir = Path(
        args.output_dir or (Path.home() / "grpo-runs" /
                            time.strftime("grpo-%Y%m%d-%H%M%S"))
    )
    report: Dict[str, Any] = {
        "model": args.model,
        "telemetry_dir": str(telemetry_dir),
        "output_dir": str(output_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    # --- Gate 1: the corpus -------------------------------------------------
    verdict = corpus_gate(
        telemetry_dir,
        min_group=args.min_group_size,
        min_spread=args.min_spread,
        min_groups=args.min_groups,
        trainable_only=not args.include_untrainable,
    )
    # Popped BEFORE the verdict becomes the report: 27 prompts of ~24k chars
    # would make the JSON unreadable, and the COUNT is the evidence anyone
    # actually reads. The prompts themselves go to the dataset loader.
    contrast_prompts: Optional[List[str]] = (
        verdict.pop("trainable_prompts", None) or None
    )
    report["corpus"] = verdict
    logger.info(
        "[runner] corpus: %d rows, %d prompts, %d trainable group(s) at "
        "--min-spread %.4f (%d flat, %d below min size)",
        verdict["rows"], verdict["prompts"], verdict["trainable_groups"],
        args.min_spread, verdict["flat_groups"], verdict["groups_below_min"],
    )
    if args.all_prompts:
        contrast_prompts = None
        logger.info(
            "[runner] --all-prompts: training over every admitted prompt, not "
            "only the gate's contrast-bearing groups",
        )
    elif contrast_prompts:
        logger.info(
            "[runner] narrowing the training set to the %d contrast-bearing "
            "group(s) the gate selected (of %d admitted prompts)",
            len(contrast_prompts), verdict["prompts"],
        )
    report["contrast_filtered"] = bool(contrast_prompts)
    report["training_prompts"] = (
        len(contrast_prompts) if contrast_prompts else verdict["prompts"]
    )
    if not verdict["passes"] and not args.skip_corpus_gate:
        logger.error(
            "[runner] REFUSED: %d trainable group(s) < required %d. GRPO "
            "learns from within-group contrast; there is not enough here.",
            verdict["trainable_groups"], args.min_groups,
        )
        report["refused"] = "corpus"
        _write(args.json_out, report)
        return EXIT_REFUSED

    # --- Gate 2: the hardware ----------------------------------------------
    if not args.skip_admission:
        admission = guard.check_admission()
        report["admission"] = {
            "allowed": admission.allowed,
            "reason": admission.reason,
            "sample": admission.sample.to_dict(),
        }
        logger.info("[runner] admission: %s", admission.reason)
        if not admission.allowed:
            logger.error("[runner] REFUSED: %s", admission.reason)
            report["refused"] = "hardware"
            _write(args.json_out, report)
            return EXIT_REFUSED

    # --- the ladder ---------------------------------------------------------
    device_count = 0
    try:
        import torch  # noqa: PLC0415
        device_count = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        pass
    # Before the first allocation, in the process that will make it. The
    # driver on this box pages a model that does not fit into HOST memory
    # instead of failing, and that spill is charged to Windows' commit
    # limit -- see memory_guard.DEFAULT_CUDA_ALLOCATOR_FRACTION.
    report["cuda_allocator_fraction"] = guard.cap_cuda_allocator()
    # Reconcile FIRST. build_grpo_config enforces the same invariant as a
    # last line of defence, but the ladder sizes its fallback rungs against
    # global_batch -- so it has to see the accumulation that will really be
    # used, not the one that was requested.
    accumulation = guard.accumulation_for_groups(
        args.num_generations,
        per_device_batch=1,
        requested_accum=max(1, args.gradient_accumulation_steps),
        device_count=max(1, device_count),
    )
    if accumulation != args.gradient_accumulation_steps:
        logger.info(
            "[runner] gradient_accumulation_steps %d -> %d so the generation "
            "batch holds whole %d-completion groups",
            args.gradient_accumulation_steps, accumulation,
            args.num_generations,
        )
        args.gradient_accumulation_steps = accumulation
    global_batch = accumulation
    ladder = guard.build_ladder(
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        global_batch=global_batch,
        device_count=device_count,
    )
    report["ladder"] = [
        {"name": r.name, "num_generations": r.num_generations,
         "max_completion_length": r.max_completion_length, "note": r.note}
        for r in ladder
    ]

    overrides: Dict[str, Any] = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
    }
    report["batching"] = {
        "num_generations": args.num_generations,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "per_device_train_batch_size": 1,
        "generation_batch": global_batch,
        "groups_per_generation_batch": (
            global_batch // args.num_generations
            if args.num_generations else 0
        ),
    }
    if args.max_steps > 0:
        overrides["max_steps"] = args.max_steps

    try:
        outcome = train_with_ladder(
            model_id=args.model,
            telemetry_dir=telemetry_dir,
            output_dir=output_dir,
            ladder=ladder,
            guard=guard,
            trainable_only=not args.include_untrainable,
            max_prompts=args.max_prompts or None,
            only_prompts=contrast_prompts,
            gptq_backend=args.gptq_backend,
            use_qlora=not args.no_qlora,
            config_overrides=overrides,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[runner] training failed")
        report["error"] = f"{type(exc).__name__}: {exc}"[:500]
        _write(args.json_out, report)
        return EXIT_ERROR

    report.update(outcome)
    report["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write(args.json_out, report)

    if outcome["status"] == "ladder-exhausted":
        logger.error(
            "[runner] every rung ran out of memory. The smallest tried was "
            "%d generations x %d tokens; this model does not fit here.",
            ladder[-1].num_generations, ladder[-1].max_completion_length,
        )
        return EXIT_LADDER_EXHAUSTED
    logger.info("[runner] %s", outcome["status"])
    return EXIT_OK


def _write(path: str, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
