"""GRPO reward over O+V sibling groups — the group IS the sibling draw.

## Why GRPO and not more DPO

Five farming soaks produced **zero** DPO pairs. Every one failed the same
way: a pair needs two responses to ONE prompt plus a verdict that
separates them, and the pipeline kept not producing both at once.

GRPO does not need pairs. It needs prompts, generates ``num_generations``
completions itself, and normalises reward within the group:

    Â_i = (r_i - mean(r)) / std(r)

The group is the n>=3 sibling draw. The reward is what
:mod:`reactor_core.training.grpo_verifier` can establish about each
completion. The pair requirement — the thing that blocked every run —
stops existing.

## The reward measures the COMPLETION, and only the completion

The first version of this module made `_score_candidate` the primary
signal. That was wrong, and a live profiling run proved it: its inputs
(`outcome`, `confidence`, `latency_ms`) are columns of the DATASET ROW,
so TRL hands the SAME metadata to every completion in a group. The
scorer therefore returned N identical values by construction and every
group was flat before the model had said anything. Those fields describe
a historical generation, not the one being scored.

Verification is primary. The historical scorer survives only as a small,
env-tunable nudge for the case where two completions verify identically
— weak evidence about the PROMPT, never enough to outrank a verification
difference.

## The zero-variance rule

When every sibling scores the same, std is 0 and the advantage is 0/0.
Shifting the whole group by a penalty is arithmetically inert: subtract
the same constant from every member and the variance is exactly where it
was. Only a term that DIFFERS between siblings can help, which is why the
verifier grades in BANDS rather than pass/fail — two completions that
both failed usually failed differently, and that difference is real
signal a boolean discards.

When they are genuinely indistinguishable, the group is DROPPED (TRL's
contract: "None excludes that sample"). Manufacturing a winner between
identical answers is noise with a gradient attached — the same refusal
the trajectory recorder makes when it labels an unseen outcome
``unknown`` rather than guessing. A quiet group is a fact about the
group.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Below this spread a group is degenerate. Not exactly 0.0: float noise
#: produces spreads like 1e-17 that are not signal but would pass an
#: `== 0` test and then explode when divided by a near-zero std.
_FLAT_EPS = 1e-6

#: Weight of the HISTORICAL scorer, demoted from primary signal to a
#: nudge. Its inputs describe a past generation, not the completion being
#: scored — see the module docstring.
_ENV_HISTORY_WEIGHT = "REACTOR_GRPO_HISTORY_WEIGHT"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _load_scorer() -> Any:
    """Get ``(DPOConfig, DPOPairGenerator, ResponseCandidate)``.

    Tries the package import, then falls back to loading
    ``dpo_pair_generator`` BY PATH. The fallback earns its place: this
    package's ``__init__`` was unimportable for a long time, and the
    training modules must not become unusable again if it regresses.
    ``dpo_pair_generator`` is stdlib-only and loads cleanly alone.

    ONE loader, so the scorer is still imported from exactly one place.
    """
    try:
        from reactor_core.training.dpo_pair_generator import (  # noqa: PLC0415
            DPOConfig, DPOPairGenerator, ResponseCandidate,
        )
        return DPOConfig, DPOPairGenerator, ResponseCandidate
    except Exception:  # noqa: BLE001
        import importlib.util as _ilu  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415

        name = "_reactor_dpo_pair_generator"
        cached = _sys.modules.get(name)
        if cached is not None:
            return cached.DPOConfig, cached.DPOPairGenerator, cached.ResponseCandidate
        spec = _ilu.spec_from_file_location(
            name, pathlib.Path(__file__).with_name("dpo_pair_generator.py"),
        )
        mod = _ilu.module_from_spec(spec)
        # Register BEFORE exec: @dataclass resolves cls.__module__ through
        # sys.modules, and a module absent from it raises on the first
        # decorated class.
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.DPOConfig, mod.DPOPairGenerator, mod.ResponseCandidate


def _is_flat(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return True
    try:
        return (max(values) - min(values)) < _FLAT_EPS
    except Exception:  # noqa: BLE001
        return True


def history_rewards(
    completions: Sequence[str],
    *,
    outcome: Optional[Sequence[str]] = None,
    confidence: Optional[Sequence[float]] = None,
    latency_ms: Optional[Sequence[float]] = None,
    model_id: Optional[Sequence[str]] = None,
    task_type: Optional[Sequence[str]] = None,
) -> List[float]:
    """The EXISTING DPO scorer, applied to each row's historical metadata.

    Reused rather than restated: a second copy of its weights (outcome
    .50 / confidence .25 / specialist .15 / latency .10) would rank
    candidates differently the first time either moved.

    Note what this can and cannot do. Within one group these values are
    usually identical, because they come from the dataset row rather than
    the completion — which is exactly why this is a nudge and not the
    signal.
    """
    DPOConfig, DPOPairGenerator, ResponseCandidate = _load_scorer()
    gen = DPOPairGenerator(DPOConfig())

    def _at(seq: Optional[Sequence[Any]], i: int, default: Any) -> Any:
        try:
            return seq[i] if seq is not None and i < len(seq) else default
        except Exception:  # noqa: BLE001
            return default

    out: List[float] = []
    for i, text in enumerate(completions):
        cand = ResponseCandidate(
            response=str(text or ""),
            model_id=_at(model_id, i, None),
            confidence=float(_at(confidence, i, 0.5) or 0.0),
            outcome=str(_at(outcome, i, "unknown")),
            latency_ms=float(_at(latency_ms, i, 0.0) or 0.0),
            task_type=_at(task_type, i, None),
            timestamp="", event_id="",
        )
        try:
            out.append(float(gen._score_candidate(cand)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scorer fault on candidate %d: %s", i, exc)
            out.append(0.5)
    return out


async def candidate_reward(
    completions: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[Any]] = None,  # noqa: ARG001 — TRL passes it
    **kwargs: Any,
) -> Optional[List[Optional[float]]]:
    """TRL-compatible reward over one group of sibling candidates.

    Keyword-argument signature per TRL's contract, returning one float (or
    None) per completion. Extra dataset columns arrive through
    ``**kwargs`` and are exactly the fields the trajectory corpus carries.

    1. Verify every completion — the primary and usually the only signal.
    2. Add a small historical nudge, if configured.
    3. If the group is still flat, return None per sibling and let TRL
       drop it rather than fabricate a winner.
    """
    texts = [str(c or "") for c in (completions or [])]
    if not texts:
        return []

    log_metric = kwargs.get("log_metric")
    log_extra = kwargs.get("log_extra")

    # Grades the CODE inside the envelope, in bands, across a tiered
    # ladder (envelope -> shape -> syntax -> substance -> authority).
    from reactor_core.training.grpo_verifier import verify_batch  # noqa: PLC0415

    verdicts = await verify_batch(texts)
    rewards: List[float] = [v.score for v in verdicts]
    _emit(log_extra, "verify_reason", [v.reason for v in verdicts])
    _emit(log_extra, "verify_tier", [v.tier for v in verdicts])
    _emit(log_metric, "verify/mean_tier",
          sum(v.tier for v in verdicts) / max(1, len(verdicts)))
    _emit(log_metric, "verify/authoritative",
          1.0 if any(v.authoritative for v in verdicts) else 0.0)

    hist_w = _env_float(_ENV_HISTORY_WEIGHT, 0.10)
    if hist_w > 0.0:
        try:
            hist = history_rewards(
                texts,
                outcome=kwargs.get("outcome"),
                confidence=kwargs.get("confidence"),
                latency_ms=kwargs.get("latency_ms"),
                model_id=kwargs.get("model_id"),
                task_type=kwargs.get("task_type"),
            )
            rewards = [r + hist_w * h for r, h in zip(rewards, hist)]
        except Exception as exc:  # noqa: BLE001 — the nudge is optional
            logger.debug("history nudge unavailable: %s", exc)

    if not _is_flat(rewards):
        _emit(log_metric, "reward/flat_group", 0.0)
        return list(rewards)

    # Every sibling verified identically. There is no preference signal in
    # this group, and there is nothing further to consult -- the verifier
    # already graded structure in bands and, when configured, asked the
    # authoritative validator. Dropping is the honest answer.
    _emit(log_metric, "reward/flat_group", 1.0)
    _emit(log_metric, "reward/dropped_flat_group", 1.0)
    logger.info(
        "[GRPO] group of %d verified identically (%s) — dropped rather than "
        "assigned a fabricated advantage",
        len(texts), verdicts[0].reason if verdicts else "?",
    )
    return [None] * len(texts)


def _emit(fn: Any, name: str, value: Any) -> None:
    """Call TRL's log_metric / log_extra when present. NEVER raises."""
    try:
        if callable(fn):
            fn(name, value)
    except Exception:  # noqa: BLE001
        pass


__all__ = ["candidate_reward", "history_rewards"]
