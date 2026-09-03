"""GRPO training over O+V telemetry — config, in-memory dataset, trainer.

Three things this module owns:

1. a ``GRPOConfig`` tuned for a 30B MoE inside 32 GiB (QLoRA + the knobs
   that GRPO actually exposes — see the API-reality note below);
2. an IN-MEMORY Arrow dataset built straight from the trajectory corpus,
   with no intermediate JSONL written; and
3. the trainer wiring, with ``grpo_reward.candidate_reward`` as the
   reward function.

## API reality check (verified against the installed trl 1.12.0)

Several knobs that exist for DPO do NOT exist for GRPO, and pretending
otherwise would produce a config that silently ignores them:

* ``precompute_ref_log_probs`` — **DPOConfig only**. GRPO does not need
  it: ``beta`` defaults to ``0.0``, which drops the reference model
  entirely rather than merely precomputing its log-probs. That is
  strictly better for VRAM than the DPO trick it replaces.
* ``activation_offloading`` — **DPOConfig only**. Not available here;
  ``gradient_checkpointing`` (already default ``True``) plus 4-bit QLoRA
  plus a smaller ``num_generations`` are the levers GRPO does have.
* ``sigmoid_norm`` / SimPO — a **DPO** loss type. GRPO's loss types are
  ``grpo`` / ``dapo`` / ``dr_grpo`` / ``sapo``. The length-bias fix in
  this family is **Dr. GRPO** (``loss_type="dr_grpo"``), which removes
  the length normalisation term that biases the standard GRPO objective
  toward long completions. That matters here specifically: candidates are
  whole source files, and the 2026-08-30 sweep measured newer models
  emitting 3-4x longer ones (18-22k chars vs 5-8.5k).
* ``router_aux_loss_coef`` IS a GRPOConfig field and already defaults to
  ``0.001`` — the MoE load-balancing loss is on by default, so setting it
  is a statement of intent rather than a change.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Fields the reward function reads off the dataset. They travel as extra
#: columns and reach `candidate_reward` through TRL's `**kwargs`.
_REWARD_COLUMNS = ("outcome", "confidence", "latency_ms", "model_id", "task_type")


# ---------------------------------------------------------------------------
# Phase 3 — telemetry -> Arrow, in memory
# ---------------------------------------------------------------------------


#: ``metadata.draw_kind`` values that are a genuine answer to the op's
#: GENERATE prompt. Mirrors ``trajectory_recorder.GENUINE_DRAW_KINDS`` on
#: the jarvis side; the empty string is a pre-discriminator corpus row.
GENUINE_DRAW_KINDS = frozenset({"primary", "sibling", "unknown", ""})


def iter_trajectory_rows(
    telemetry_dir: Path,
    *,
    trainable_only: bool = True,
    genuine_only: bool = True,
) -> Iterable[Dict[str, Any]]:
    """Stream recorder rows out of the corpus.

    ``genuine_only`` keeps only rows whose ``metadata.draw_kind`` is a
    genuine primary/sibling draw (or absent -- a corpus written before the
    discriminator existed is treated as genuine, never silently emptied).
    An L2 repair iteration answered a different prompt and is not a
    sibling of the draw it repaired; pairing them is what made soak 17's
    "twins" 1.0000 alike. Rows are also deduplicated by
    ``(op_id, candidate_hash)``, deterministically, first seen wins.

    A generator, not a list, because the corpus is append-only and grows
    without bound across soaks; materialising every historical row to
    select a few hundred prompts is the I/O cost this path exists to
    avoid.
    """
    seen_keys: set = set()
    for f in sorted(Path(telemetry_dir).glob("*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("[GRPO] unreadable telemetry %s: %s", f.name, exc)
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Recorder rows only; the same directory also carries Trinity
            # bus envelopes, which have no generation content at all.
            if row.get("event_type") != "interaction":
                continue
            if not row.get("user_input") or not row.get("assistant_output"):
                continue
            if trainable_only and not row.get("metadata", {}).get("should_train", False):
                continue
            meta = row.get("metadata") or {}
            if genuine_only and str(meta.get("draw_kind", "") or "") not in GENUINE_DRAW_KINDS:
                continue
            key = (str(meta.get("op_id", "") or ""), str(meta.get("candidate_hash", "") or ""))
            if key[1]:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            yield row


def build_prompt_dataset(
    telemetry_dir: Path,
    *,
    trainable_only: bool = True,
    max_prompts: Optional[int] = None,
) -> Any:
    """An in-memory Arrow ``Dataset`` of PROMPTS. No intermediate file.

    GRPO needs only a ``prompt`` column — it generates the completions
    itself — so this deduplicates the corpus down to distinct prompts and
    carries the reward-relevant fields alongside. Nothing is written to
    disk: `Dataset.from_list` builds the Arrow table in memory, which is
    the whole point of routing telemetry straight in.

    Deduplication is by prompt text, keeping the FIRST occurrence's
    metadata. Feeding the same prompt N times would not add information —
    GRPO already draws `num_generations` samples from each one — it would
    just weight that prompt N times in the epoch.
    """
    from datasets import Dataset  # noqa: PLC0415 — heavy import, call-time

    seen: set = set()
    records: List[Dict[str, Any]] = []
    for row in iter_trajectory_rows(telemetry_dir, trainable_only=trainable_only):
        prompt = str(row.get("user_input") or "")
        key = prompt.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        meta = row.get("metadata") or {}
        records.append({
            "prompt": prompt,
            "outcome": str(row.get("outcome") or "unknown"),
            "confidence": float(row.get("confidence") or 0.0),
            "latency_ms": float(row.get("latency_ms") or 0.0),
            "model_id": str(row.get("model_id") or ""),
            "task_type": str(row.get("task_type") or ""),
            "op_id": str(meta.get("op_id") or ""),
        })
        if max_prompts and len(records) >= max_prompts:
            break

    if not records:
        raise ValueError(
            f"no trainable prompts in {telemetry_dir} — with "
            f"trainable_only={trainable_only}. A corpus of governance-caged "
            "ops is correctly marked should_train=false and yields nothing."
        )
    logger.info("[GRPO] built in-memory dataset: %d distinct prompt(s)", len(records))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Phase 2 — the VRAM / MoE configuration
# ---------------------------------------------------------------------------


def build_grpo_config(
    output_dir: str,
    *,
    num_generations: int = 4,
    max_completion_length: int = 1024,
    length_unbiased: bool = True,
    use_liger: bool = False,
    **overrides: Any,
) -> Any:
    """A ``GRPOConfig`` sized for one 32 GiB card.

    ``num_generations`` defaults to 4, not TRL's 8: every completion in a
    group is generated AND backpropagated, so the group size multiplies
    activation memory directly. 4 still gives the sibling comparison GRPO
    needs while halving that cost, and it matches the n>=3 the generation
    lane produces.

    ``length_unbiased`` selects ``dr_grpo``, which drops the length
    normalisation that biases standard GRPO toward long completions. Set
    it False to keep TRL's ``dapo`` default.
    """
    from trl import GRPOConfig  # noqa: PLC0415

    cfg: Dict[str, Any] = dict(
        output_dir=output_dir,
        # --- the group ---
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        # --- memory ---
        # beta=0.0 means NO reference model is held at all. This is GRPO's
        # answer to the memory problem DPO solves with
        # precompute_ref_log_probs, and it is a bigger saving: nothing to
        # keep resident and nothing to precompute.
        beta=0.0,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        # --- MoE ---
        # Load-balancing auxiliary loss for the router. Already the
        # default; stated because a MoE trained without it drifts toward
        # a few overused experts, and the next reader should see that this
        # was a decision rather than an omission.
        router_aux_loss_coef=0.001,
        # --- objective ---
        loss_type="dr_grpo" if length_unbiased else "dapo",
        # A truncated completion is an artefact of the token budget, not a
        # judgement about the model's choice; leaving it unmasked teaches
        # the model that running out of room is a quality signal.
        mask_truncated_completions=True,
        # --- optimisation ---
        learning_rate=1e-5,   # adapter-appropriate, per TRL's PEFT note
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    if use_liger:
        cfg["use_liger_kernel"] = True
    cfg.update(overrides)
    return GRPOConfig(**cfg)


def build_qlora_config(**overrides: Any) -> Any:
    """4-bit NF4 quantization — the base model's weights are frozen anyway."""
    import torch  # noqa: PLC0415
    from transformers import BitsAndBytesConfig  # noqa: PLC0415

    kw: Dict[str, Any] = dict(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kw.update(overrides)
    return BitsAndBytesConfig(**kw)


def build_lora_config(**overrides: Any) -> Any:
    """LoRA over attention + MLP projections.

    ``target_modules`` deliberately names the attention and MLP
    projections and NOT the MoE router (``gate``): adapting the router
    changes which experts fire, which is the one part of a MoE whose
    balance ``router_aux_loss_coef`` is simultaneously trying to hold
    steady.

    ``lora_dropout`` is 0.0 because on THIS architecture it is not a free
    hyperparameter. A Qwen3 MoE keeps its expert MLPs as fused
    ``nn.Parameter`` tensors, so peft adapts them through ``ParamWrapper``
    rather than a Linear layer, and ParamWrapper rejects dropout outright:

        ValueError: lora.ParamWrapper does not work with lora_dropout != 0.

    That is a hard constructor failure, not a warning -- 0.05 meant the
    trainer could never be built for the model this pipeline exists to
    train. Overridable for dense models, where dropout does work.
    """
    from peft import LoraConfig  # noqa: PLC0415

    kw: Dict[str, Any] = dict(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    kw.update(overrides)
    return LoraConfig(**kw)


def detect_quantization(model_id: str) -> Dict[str, Any]:
    """What quantization does this checkpoint ALREADY carry?

    Returns ``{"method": <str|"">, "config": <dict>}``. ``method`` is empty
    for an unquantized checkpoint. Never raises -- an unreadable config
    means "assume base weights", which is the conservative answer because
    it attaches a quantizer rather than skipping one.
    """
    try:
        from transformers import AutoConfig  # noqa: PLC0415
        raw = getattr(AutoConfig.from_pretrained(model_id),
                      "quantization_config", None) or {}
        if not isinstance(raw, dict):
            raw = getattr(raw, "to_dict", dict)()
        return {"method": str(raw.get("quant_method", "") or ""), "config": raw}
    except Exception:  # noqa: BLE001
        logger.debug("[GRPO] no readable quantization_config on %s", model_id,
                     exc_info=True)
        return {"method": "", "config": {}}


def load_training_model(
    model_id: str,
    *,
    use_qlora: bool = True,
    device_map: Optional[Any] = None,
    gptq_backend: str = "",
) -> Any:
    """Load a model for training, adapting to what the checkpoint IS.

    Two failure modes this exists to prevent, both measured on this box:

    **Host-RAM OOM.** Handing `GRPOTrainer` a model STRING lets transformers
    load with no ``device_map``, so every shard is materialised in host RAM
    before anything reaches the GPU. For a 30B that was
    ``Killed process (python) anon-rss:48216856kB`` -- 48.2 GiB against 47.
    A ``device_map`` streams shard-by-shard straight to the device, which is
    also the placement training will use anyway.

    **Double quantization.** Attaching a BitsAndBytes config to a checkpoint
    that is ALREADY 4-bit (GPTQ/AWQ) describes a conversion that is not
    happening: it either errors or produces a model whose memory profile
    belongs to neither format. The checkpoint is asked what it is, and its
    own config is honoured; only the runtime KERNEL may be overridden.

    NOTE for GPTQ specifically: on transformers 5.16.1 + gptqmodel 7.3.5,
    both the `torch` and `triton` backends DEQUANTISE to bf16 at load --
    59.38 GiB resident for a 30B "4-bit" checkpoint, measured. A GPTQ
    checkpoint is therefore NOT a route to a small footprint on this stack;
    bnb-NF4 over base bf16 weights is.
    """
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    quant = detect_quantization(model_id)
    kw: Dict[str, Any] = {
        # Never None: the whole point is to avoid a host-RAM materialisation.
        "device_map": device_map if device_map is not None else {"": 0},
        "low_cpu_mem_usage": True,
    }

    if quant["method"]:
        # Pre-quantized. Honour the checkpoint; override only the kernel.
        if gptq_backend and quant["method"].lower() == "gptq":
            from transformers import GPTQConfig  # noqa: PLC0415
            src = quant["config"]
            kw["quantization_config"] = GPTQConfig(
                bits=int(src.get("bits", 4)),
                group_size=int(src.get("group_size", 128)),
                desc_act=bool(src.get("desc_act", False)),
                sym=bool(src.get("sym", True)),
                backend=gptq_backend,
            )
        logger.info("[GRPO] %s is pre-quantized (%s); not re-quantizing",
                    model_id, quant["method"])
    elif use_qlora:
        kw["quantization_config"] = build_qlora_config()
        logger.info("[GRPO] %s is base weights; attaching bnb-NF4", model_id)

    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


def build_trainer(
    model_id: str,
    telemetry_dir: Path,
    output_dir: str,
    *,
    num_generations: int = 4,
    max_prompts: Optional[int] = None,
    trainable_only: bool = True,
    use_qlora: bool = True,
    device_map: Optional[Any] = None,
    gptq_backend: str = "",
    **config_overrides: Any,
) -> Any:
    """Assemble the GRPOTrainer over the live corpus."""
    from trl import GRPOTrainer  # noqa: PLC0415

    from reactor_core.training.grpo_reward import candidate_reward  # noqa: PLC0415

    dataset = build_prompt_dataset(
        telemetry_dir, trainable_only=trainable_only, max_prompts=max_prompts,
    )
    args = build_grpo_config(
        output_dir, num_generations=num_generations, **config_overrides,
    )
    kwargs: Dict[str, Any] = {}
    if use_qlora:
        kwargs["peft_config"] = build_lora_config()
    # A model OBJECT, never the id: see load_training_model for the host-RAM
    # OOM and the double-quantization this avoids.
    return GRPOTrainer(
        model=load_training_model(
            model_id, use_qlora=use_qlora,
            device_map=device_map, gptq_backend=gptq_backend,
        ),
        reward_funcs=candidate_reward,
        args=args,
        train_dataset=dataset,
        **kwargs,
    )


__all__ = [
    "detect_quantization",
    "load_training_model",
    "build_grpo_config",
    "build_lora_config",
    "build_prompt_dataset",
    "build_qlora_config",
    "build_trainer",
    "iter_trajectory_rows",
]
