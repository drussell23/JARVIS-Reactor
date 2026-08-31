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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GIB = 1024 ** 3


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
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--no-qlora", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    import torch
    if not torch.cuda.is_available():
        print("CUDA unavailable — nothing to profile."); return 1
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  sm_{p.major}{p.minor}  {p.total_memory / GIB:.1f} GiB")
    print(f"torch : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"sm_120 in build: {'sm_120' in torch.cuda.get_arch_list()}\n")

    torch.cuda.reset_peak_memory_stats()
    marks: list = []
    mark("0. idle", marks)

    # Import late so the idle mark is honest.
    from trl import GRPOTrainer
    from reactor_core.training.grpo_pipeline import (
        build_grpo_config, build_lora_config, build_qlora_config,
        build_prompt_dataset,
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

    # A GRPO step consumes `per_device_train_batch_size *
    # gradient_accumulation_steps` prompts, and that product must be
    # divisible by num_generations. The shipping config uses accum=8 for
    # memory; with a thin corpus that needs 8 prompts and, short of them,
    # transformers logs "not a single sample in your epoch_iterator" and
    # exits at step 0 — reporting a peak that is only the loaded weights.
    # A profile that silently measures nothing is worse than no profile,
    # so the batch is collapsed to exactly one real step and the dataset
    # padded to fill it.
    need = args.num_generations
    if len(ds) < need:
        ds = _pad_to(ds, need)
        print(f"  padded dataset to {len(ds)} row(s) so one full step can run")

    cfg = build_grpo_config(
        output_dir="/tmp/grpo_profile",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        num_train_epochs=1,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
    )
    kw = {}
    if not args.no_qlora:
        kw["quantization_config"] = build_qlora_config()
        kw["peft_config"] = build_lora_config()

    print("loading (this is the long part) ...")
    trainer = GRPOTrainer(
        model=args.model, reward_funcs=candidate_reward,
        args=cfg, train_dataset=ds, **kw,
    )
    mark("1. weights + adapter resident", marks)

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

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "model": args.model, "device": p.name,
            "total_gib": total, "peak_gib": peak,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "qlora": not args.no_qlora,
            "dataset_source": source, "steps_run": steps, "marks": marks,
        }, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
