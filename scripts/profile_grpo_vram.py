#!/usr/bin/env python3
"""Measure what a GRPO step actually costs on this card.

Reports MEASURED numbers at four points — idle, weights loaded, adapter
attached, and peak during a real training step — because the question
"does the 30B MoE fit in 32 GiB" is answered by the PEAK during
backpropagation, not by the weight footprint everyone quotes.

Run it against a small model first to validate the wiring without a
60 GB download:

    python scripts/profile_grpo_vram.py --model Qwen/Qwen2.5-0.5B-Instruct

and against the real target once its weights are local:

    python scripts/profile_grpo_vram.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct

NOTE ON THE MODEL SOURCE: ollama serves GGUF, which transformers cannot
QLoRA-train. Profiling the real target needs the HF-format safetensors,
which is a separate (large) download — this script does not fetch
anything implicitly, it fails with a clear message instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GIB = 1024 ** 3

#: Same meaning as run_grpo_training.EXIT_REFUSED: the box declined, the
#: script did not fail. A refusal here IS a measurement -- it says the
#: profile cannot be taken without damaging the host.
EXIT_REFUSED = 2


def _load_guard():
    """``memory_guard`` by path, exactly as the runner loads it.

    ``reactor_core/__init__`` imports the ML stack, and the guard has to
    run BEFORE torch does -- admission is the only gate that fires before
    a CUDA byte is touched. See ``run_grpo_training._load_by_path``.
    """
    import importlib.util  # noqa: PLC0415
    path = REPO / "reactor_core" / "training" / "memory_guard.py"
    spec = importlib.util.spec_from_file_location("_profiler_memory_guard", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _default_num_generations() -> int:
    """The group size default, owned by grpo_pipeline — same source the
    runner reads.

    This must not be a literal. A profile is only evidence about the run
    it mirrors, and ``num_generations`` is the dimension the rollout
    multiplies: every completion in a group is generated AND
    backpropagated. Profiling at 4 while ``run_grpo_training`` launches
    at 16 understates the peak by the largest factor in the measurement,
    and it does so in the reassuring direction — a clean profile followed
    by an OOM on the real step is exactly the false negative this script
    exists to prevent.

    Wrapped like the runner's copy so a missing training extra degrades
    to TRL's own default rather than making ``--help`` unavailable.
    """
    try:
        from reactor_core.training.grpo_pipeline import (  # noqa: PLC0415
            default_num_generations,
        )
        return default_num_generations()
    except Exception:  # noqa: BLE001
        return 8


def vram() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"alloc": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "alloc": torch.cuda.memory_allocated() / GIB,
        "reserved": torch.cuda.memory_reserved() / GIB,
        "peak": torch.cuda.max_memory_allocated() / GIB,
    }


def free_total() -> tuple:
    import torch
    f, t = torch.cuda.mem_get_info()
    return f / GIB, t / GIB


def mark(label: str, marks: list) -> None:
    v = vram()
    f, t = free_total()
    marks.append({"stage": label, **v, "device_free": f})
    print(f"  {label:<34} alloc={v['alloc']:6.2f}  reserved={v['reserved']:6.2f}  "
          f"peak={v['peak']:6.2f}  free={f:6.2f} / {t:.1f} GiB")


def _pad_to(ds, n: int):
    """Repeat rows until the dataset can fill one step.

    Only ever used to make the MEMORY measurement possible — repeating a
    prompt teaches nothing, but the profile is asking what a step costs,
    not what it learns.
    """
    from datasets import Dataset
    rows = [ds[i] for i in range(len(ds))]
    while len(rows) < n:
        rows.append(rows[len(rows) % len(ds)])
    return Dataset.from_list(rows)


def _step_ran(trainer) -> bool:
    """Did a real optimisation step happen?

    `trainer.train()` returns happily after zero steps when the epoch
    iterator is empty, so the peak must not be trusted without this.
    """
    try:
        return int(getattr(trainer.state, "global_step", 0) or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--telemetry-dir", default=str(Path.home() / ".jarvis" / "trinity" / "events"))
    ap.add_argument("--num-generations", type=int,
                    default=_default_num_generations(),
                    help="completions per prompt. Defaults to the value "
                         "run_grpo_training will actually launch with, so "
                         "the profile answers the question that was asked; "
                         "lower it only to measure a specific rung.")
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--no-qlora", action="store_true")
    ap.add_argument(
        "--pre-quantized", action="store_true",
        help="the checkpoint is ALREADY 4-bit (GPTQ/AWQ): keep the LoRA "
             "adapter but do NOT attach a BitsAndBytesConfig. Quantizing an "
             "already-quantized checkpoint is not a no-op -- it either "
             "errors or silently produces a model whose memory profile "
             "belongs to neither format.",
    )
    ap.add_argument(
        "--gptq-backend", default="triton",
        help="GPTQ kernel for a pre-quantized checkpoint. Default "
             "'triton' compiles through Triton's own LLVM at runtime, so "
             "it needs no nvcc, and it stays in 4-bit. 'torch' also needs "
             "no compiler but DEQUANTISES to bf16, which destroys the very "
             "number this script measures. 'marlin' is fastest but "
             "JIT-builds a CUDA extension, which needs a toolkit that can "
             "target this card.",
    )
    ap.add_argument(
        "--no-warmup", action="store_true",
        help="skip the kernel warm-up. The measured peak then includes "
             "Triton JIT compilation workspace, and the report says so.",
    )
    ap.add_argument(
        "--skip-admission", action="store_true",
        help="measure even when memory_guard would refuse to start. The "
             "watchdog still runs and will still hard-abort on Windows "
             "commit; this only skips the pre-torch gate.",
    )
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    # The guard first, torch second. On 2026-09-04 22:03 this script --
    # which then had NO guard, while the runner had two -- loaded the 30B
    # straight through Windows' commit limit and took the desktop down at
    # 22:09. The profiler is the thing people actually run against a new
    # checkpoint, so it gets the same admission gate, the same watchdog
    # (armed BEFORE the load, which is where the spill happens) and the
    # same allocator cap as the runner.
    guard = _load_guard()
    if not args.skip_admission:
        adm = guard.check_admission()
        print(f"  admission: {adm.reason}")
        if not adm.allowed:
            print(f"\nREFUSED (exit {EXIT_REFUSED}): {adm.reason}")
            return EXIT_REFUSED

    with guard.MemoryWatchdog(label="profile") as watchdog:
        return _profile(args, guard, watchdog)


#: The runner's CLI default for --gradient-accumulation-steps and the
#: pipeline's build_grpo_config default. Mirrored, not imported: loading
#: the runner by path just to read an argparse default would import its
#: whole CLI. test_profile_defaults pins the two against each other.
RUNNER_REQUESTED_ACCUM = 8


def _step_geometry(guard, num_generations: int) -> tuple:
    """``(per_device_train_batch_size, gradient_accumulation_steps,
    prompts_per_step)`` exactly as ``run_grpo_training`` will launch.

    One sequence per micro-step; accumulation is whatever makes the
    generation batch a whole number of groups, decided by the SAME
    ``memory_guard.accumulation_for_groups`` the runner calls.
    """
    per_device = 1
    accum = guard.accumulation_for_groups(
        num_generations, per_device_batch=per_device,
        requested_accum=RUNNER_REQUESTED_ACCUM, device_count=1,
    )
    return per_device, accum, max(1, (per_device * accum) // max(1, num_generations))


def _profile(args, guard, watchdog) -> int:
    """Everything that touches CUDA. Runs inside the watchdog."""
    import torch
    if not torch.cuda.is_available():
        print("CUDA unavailable — nothing to profile."); return 1
    allocator_fraction = guard.cap_cuda_allocator()
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  sm_{p.major}{p.minor}  {p.total_memory / GIB:.1f} GiB")
    print(f"torch : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"sm_120 in build: {'sm_120' in torch.cuda.get_arch_list()}\n")

    torch.cuda.reset_peak_memory_stats()
    marks: list = []
    mark("0. idle", marks)

    # Import late so the idle mark is honest.
    from trl import GRPOTrainer
    from transformers import AutoConfig, AutoModelForCausalLM, GPTQConfig
    from reactor_core.training.grpo_pipeline import (
        build_grpo_config, build_lora_config, build_qlora_config,
        build_prompt_dataset, detect_quantization, load_training_model,
    )
    from reactor_core.training.grpo_reward import candidate_reward

    # The corpus may legitimately be too thin; fall back to a synthetic
    # prompt so a memory profile is still obtainable. Labelled, so the
    # report never implies it trained on real data when it did not.
    try:
        ds = build_prompt_dataset(Path(args.telemetry_dir), trainable_only=True)
        source = f"corpus ({len(ds)} prompt(s))"
    except Exception as exc:
        print(f"  ! corpus unusable ({exc}); using a synthetic prompt for the profile")
        from datasets import Dataset
        ds = Dataset.from_list([{
            "prompt": "Refactor this function for clarity:\n\ndef f(a):\n    return a*2\n",
            "outcome": "unknown", "confidence": 0.5, "latency_ms": 1000.0,
            "model_id": args.model, "task_type": "code_repair",
        }] * max(2, args.num_generations))
        source = "SYNTHETIC (corpus unusable)"
    print(f"  dataset source: {source}\n")

    # THE RUNNER'S GEOMETRY, not a collapsed one. This script used to put
    # the whole group in one micro-batch (per_device = num_generations,
    # accum = 1) so a thin corpus could still take a step. That measures a
    # forward the runner never runs: on 2026-09-04 23:30 the 30B loaded at
    # 15.6 GiB and then OOM'd in the TRAINING forward with 16 sequences of
    # ~6.3k tokens in one micro-batch -- ~20 GiB of checkpointed layer
    # inputs -- while the runner trains ONE sequence per micro-step and
    # accumulates. Same generation batch, same group, one sixteenth of the
    # activation footprint. The geometry is taken from the same helper the
    # runner calls, so the two cannot drift again.
    per_device, accum, prompts_per_step = _step_geometry(guard, args.num_generations)
    print(f"  geometry: per_device_train_batch_size={per_device} "
          f"gradient_accumulation_steps={accum} -> generation batch "
          f"{per_device * accum} = {prompts_per_step} prompt(s) x "
          f"{args.num_generations} completions per optimiser step")
    need = max(2, prompts_per_step)
    if len(ds) < need:
        ds = _pad_to(ds, need)
        print(f"  padded dataset to {len(ds)} row(s) so one full step can run")

    cfg = build_grpo_config(
        output_dir="/tmp/grpo_profile",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        num_train_epochs=1,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
    )
    kw = {}
    if not args.no_qlora:
        # A pre-quantized checkpoint carries its own quantization_config in
        # config.json; transformers reads it from there. Passing a second
        # one describes a conversion that is not happening.
        if not args.pre_quantized:
            kw["quantization_config"] = build_qlora_config()
        # model_id: the profiler exists to measure the real footprint, and
        # adapter placement is now part of it.
        kw["peft_config"] = build_lora_config(args.model)
    quant = ("pre-quantized (checkpoint)" if args.pre_quantized
             else "bnb-nf4 (at load)" if not args.no_qlora else "none (bf16)")
    print(f"  quantization: {quant}")

    print("loading (this is the long part) ...")
    # Load EXPLICITLY rather than handing GRPOTrainer a model string.
    #
    # The string path lets transformers load with no `device_map`, so every
    # shard is materialised in HOST ram and only then moved. For this
    # checkpoint that also drags GPTQModel's Marlin repack through CPU, and
    # it was measured killing the run outright:
    #
    #   Out of memory: Killed process (python) anon-rss:48216856kB
    #
    # 48.2 GiB against WSL's 47. The card was never the constraint -- the
    # profile died before it could report one. `device_map` streams shard
    # by shard straight to the GPU, which is also the placement the real
    # pipeline will use.
    # DRY: the adaptive loader lives in the pipeline, because the REAL
    # training path had both defects this script found (a model string,
    # so shards materialise in host RAM; and a bnb config attached to an
    # already-quantized checkpoint). Fixing them only here would have
    # left `build_trainer` broken while the measurement looked healthy.
    model_obj = load_training_model(
        args.model,
        use_qlora=not args.no_qlora,
        gptq_backend=args.gptq_backend if args.pre_quantized else "",
    )
    quant_detected = detect_quantization(args.model)["method"] or "none(base)"
    print(f"  checkpoint quantization: {quant_detected}")
    trainer = GRPOTrainer(
        model=model_obj, reward_funcs=candidate_reward,
        args=cfg, train_dataset=ds, **kw,
    )
    mark("1. weights + adapter resident", marks)

    if not args.no_warmup:
        # Triton compiles its kernels on FIRST USE, through its own LLVM.
        # That compile allocates scratch, and it would otherwise land
        # inside the window this script exists to measure -- reporting
        # compiler workspace as if it were training memory, on the single
        # step we take. Force it to resolve here, then reset the peak so
        # mark 2 is a clean training measurement.
        #
        # This is NOT a race fix: Triton compiles synchronously inside the
        # forward pass and nothing else is running. It buys measurement
        # fidelity, and it keeps an unbounded first-step compile out of a
        # step someone may have wrapped in a timeout.
        print("\nwarming up (compiling kernels; first pass is the slow one) ...")
        t0 = time.time()
        try:
            tok = trainer.processing_class
            m = trainer.model
            enc = tok("def f():\n    return 1\n", return_tensors="pt")
            enc = {k: v.to(m.device) for k, v in enc.items()}
            with torch.no_grad():
                m(**enc)
            warm = {"ok": True, "seconds": round(time.time() - t0, 1)}
            print(f"  kernels compiled in {warm['seconds']}s")
        except Exception as exc:  # noqa: BLE001
            # A warm-up must never decide the run. If it fails the step
            # still runs; the peak then includes compilation, and the
            # report SAYS so rather than passing a polluted number off as
            # a clean one.
            warm = {"ok": False, "seconds": round(time.time() - t0, 1),
                    "error": f"{type(exc).__name__}: {exc}"[:160]}
            print(f"  warm-up FAILED ({warm['error']}) -- "
                  f"mark 2 will INCLUDE kernel compilation")
        mark("1b. after warm-up", marks)
        torch.cuda.reset_peak_memory_stats()
    else:
        warm = {"ok": False, "seconds": 0.0, "error": "skipped (--no-warmup)"}

    print("\nrunning one real GRPO step (generate x N, reward, backward) ...")
    trainer.train()
    mark("2. peak during training step", marks)

    steps = int(getattr(trainer.state, "global_step", 0) or 0)
    peak = max(m["peak"] for m in marks)
    total = p.total_memory / GIB
    print(f"\n=== VERDICT ===")
    print(f"  optimisation steps run : {steps}")
    if not _step_ran(trainer):
        # The failure mode this guard exists for: train() returns cleanly
        # after zero steps, and the "peak" is then just resident weights.
        # Reporting that as a memory profile would be a fabricated result.
        print("  *** NO STEP RAN — the numbers below are RESIDENT WEIGHTS ONLY,")
        print("      not a training peak. Do not read them as a memory profile.")
    print(f"  peak allocated : {peak:.2f} GiB")
    print(f"  card total     : {total:.1f} GiB")
    print(f"  headroom       : {total - peak:.2f} GiB")
    print(f"  fits           : "
          f"{('YES' if peak < total * 0.92 else 'TIGHT/NO') if _step_ran(trainer) else 'UNKNOWN (no step)'}")
    print(f"  config         : n_gen={args.num_generations} "
          f"max_completion={args.max_completion_length} "
          f"qlora={not args.no_qlora}")
    wd = watchdog.report()
    if wd.get("min_win_commit_available_gib") is not None:
        print(f"  windows commit : min {wd['min_win_commit_available_gib']:.1f} GiB "
              f"free during the run (floor {wd['win_commit_floor_gib']:.1f})")
    if wd.get("breach"):
        print(f"  watchdog breach: {wd['breach']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model": args.model, "device": p.name,
            "total_gib": total, "peak_gib": peak,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "qlora": not args.no_qlora, "quantization": quant_detected,
            "dataset_source": source, "steps_run": steps, "marks": marks,
            "warmup": warm,
            "cuda_allocator_fraction": allocator_fraction,
            "watchdog": wd,
        }, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
