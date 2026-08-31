"""GRPO reward functions over O+V candidates — the group IS the sibling draw.

## Why GRPO and not more DPO

Five farming soaks produced **zero** DPO pairs. Every one failed the same
way: a pair needs two responses to ONE prompt plus a verdict that
separates them, and the pipeline kept not producing both at once.

GRPO does not need pairs. It needs prompts, generates ``num_generations``
completions per prompt itself, and scores each with a reward function,
normalising within the group:

    Â_i = (r_i - mean(r)) / std(r)

That is the shape O+V already produces. The group is the n>=3 sibling
draw; the reward is the per-candidate VALIDATE verdict the orchestrator
already computes. The pair requirement — the thing that has blocked every
run — simply stops existing.

## The zero-variance trap, and why this does NOT fabricate variance

When every sibling gets the same reward (all three failed, the common
case here) the group standard deviation is 0 and the advantage is 0/0.
The tempting fix is to shift the whole group by some penalty. **That does
nothing**: subtracting the same constant from every member leaves the
variance exactly where it was. Only a term that DIFFERS between siblings
can produce a non-zero advantage.

So the tiebreaker has to be a real per-candidate MEASUREMENT, and
:func:`structural_severity` is one — it grades how badly a candidate is
broken, and two candidates that both "failed" usually fail differently
(one does not parse at all, one parses but is a no-op, one truncates
mid-function). That difference is genuine signal the coarse
success/failure label threw away.

And when it does not differ — three byte-identical failures, say — there
is **no preference signal in that group at all**, and this returns
``None`` to drop it. TRL's contract allows exactly that ("None excludes
that sample from that reward function"). Manufacturing an advantage out
of a tie would teach the model that one of two indistinguishable answers
is better, which is noise presented as gradient. It is the same principle
the trajectory recorder already states about labels: a mislabelled sample
is worse than a missing one. A quiet group is a fact about the group, not
a defect to paper over.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Below this spread a group is treated as degenerate. Not exactly 0.0:
#: float noise in the latency term produces spreads like 1e-17 that are
#: not real signal but would pass an `== 0` test and yield an advantage of
#: ~1e17 after division by a near-zero std.
_FLAT_EPS = 1e-6

_ENV_TIEBREAK = "REACTOR_GRPO_STRUCTURAL_TIEBREAK"
_ENV_TIEBREAK_WEIGHT = "REACTOR_GRPO_TIEBREAK_WEIGHT"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Structural severity — the per-candidate measurement that breaks flat groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Severity:
    """How badly one candidate is broken, and why.

    ``score`` is in [0.0, 1.0], HIGHER IS BETTER, so it composes with the
    reward directly rather than needing a sign flip at the call site.
    """

    score: float
    reason: str

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"Severity({self.score:.3f}, {self.reason!r})"


def structural_severity(text: str) -> Severity:
    """Grade a candidate's structural health from the text alone.

    Deliberately a LADDER rather than a boolean, because the whole job of
    this function is to separate candidates that a binary pass/fail
    already declared equal. "Does not parse" and "parses but is empty" are
    both failures and are not equally bad, and a file that dies on line 3
    is more broken than one that dies on line 300 — the second got
    almost everything right.

    Cheap: one `ast.parse`, no imports, no execution. Pure; NEVER raises.
    """
    try:
        if text is None:
            return Severity(0.0, "none")
        body = str(text)
        if not body.strip():
            return Severity(0.0, "empty")

        try:
            tree = ast.parse(body)
        except SyntaxError as exc:
            # Where the parse died is a real gradient: a file that fails
            # at the last line is nearly right; one that fails at line 1
            # is not code at all. Normalised over the candidate's own
            # length so long and short files are comparable.
            total = max(1, body.count("\n") + 1)
            line = int(getattr(exc, "lineno", 1) or 1)
            reached = max(0.0, min(1.0, (line - 1) / total))
            # Capped below the "parses" band so a syntax error can never
            # outrank a candidate that actually compiles.
            return Severity(0.05 + 0.25 * reached,
                            f"syntax_error:line{line}/{total}")
        except (ValueError, RecursionError) as exc:
            return Severity(0.02, f"unparseable:{type(exc).__name__}")

        # It compiles. Now grade what it CONTAINS -- a syntactically
        # perfect empty module is a valid parse and a useless patch.
        defs = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        stmts = len(getattr(tree, "body", []) or [])
        if stmts == 0:
            return Severity(0.35, "parses:empty_module")
        # A body that is only a docstring / `pass` / `...` parses and says
        # nothing -- the Quine-class no-op the BG/SPEC filter also hunts.
        inert = all(
            isinstance(n, ast.Pass)
            or (isinstance(n, ast.Expr)
                and isinstance(getattr(n, "value", None), ast.Constant))
            for n in tree.body
        )
        if inert:
            return Severity(0.40, "parses:inert_body")
        if defs == 0:
            return Severity(0.60, f"parses:{stmts}stmt_no_defs")
        # Saturating: past a handful of definitions "more" stops meaning
        # "better", and rewarding size would reintroduce the length bias
        # the SimPO loss is configured to remove.
        return Severity(min(1.0, 0.70 + 0.06 * min(defs, 5)),
                        f"parses:{defs}defs/{stmts}stmt")
    except Exception as exc:  # noqa: BLE001 — a grader must never break training
        logger.debug("structural_severity failed: %s", exc, exc_info=True)
        return Severity(0.5, "grader_fault")


async def structural_severities(texts: Sequence[str]) -> List[Severity]:
    """Grade a whole group, off the event loop.

    `ast.parse` on a multi-KB source file is CPU-bound, and a group is
    `num_generations` of them. Awaiting them on the loop would stall the
    trainer's other coroutines for the duration; a thread offload is what
    makes this honestly asynchronous rather than an `async def` that
    blocks anyway.
    """
    loop = asyncio.get_running_loop()
    return await asyncio.gather(*(
        loop.run_in_executor(None, structural_severity, t) for t in texts
    ))


# ---------------------------------------------------------------------------
# The reward function TRL calls
# ---------------------------------------------------------------------------


def _is_flat(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return True
    try:
        return (max(values) - min(values)) < _FLAT_EPS
    except Exception:  # noqa: BLE001
        return True


def _load_scorer() -> Any:
    """Get ``(DPOConfig, DPOPairGenerator, ResponseCandidate)``.

    Tries the ordinary package import first, then falls back to loading
    ``dpo_pair_generator`` BY PATH. The fallback is not defensive
    paranoia: ``reactor_core/__init__`` imports ``PreprocessingPipeline``
    from ``reactor_core.data``, which holds only ``lineage.py`` — so the
    package raises ImportError on import and every module inside it is
    unreachable through the normal path. ``dpo_pair_generator`` is
    stdlib-only and loads cleanly on its own; the audit scripts and the
    reactor tests already use this same by-path trick for the same
    reason.

    Kept as ONE loader so the scorer is still imported from exactly one
    place — the point of reusing it at all.
    """
    try:
        from reactor_core.training.dpo_pair_generator import (  # noqa: PLC0415
            DPOConfig, DPOPairGenerator, ResponseCandidate,
        )
        return DPOConfig, DPOPairGenerator, ResponseCandidate
    except Exception:  # noqa: BLE001 — broken package __init__, not our bug
        import importlib.util as _ilu  # noqa: PLC0415
        import pathlib  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415

        mod_name = "_reactor_dpo_pair_generator"
        cached = _sys.modules.get(mod_name)
        if cached is not None:
            return cached.DPOConfig, cached.DPOPairGenerator, cached.ResponseCandidate
        path = pathlib.Path(__file__).with_name("dpo_pair_generator.py")
        spec = _ilu.spec_from_file_location(mod_name, path)
        mod = _ilu.module_from_spec(spec)
        # Register BEFORE exec: @dataclass resolves cls.__module__ through
        # sys.modules, and a module absent from it raises AttributeError
        # on the first decorated class.
        _sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.DPOConfig, mod.DPOPairGenerator, mod.ResponseCandidate


def base_rewards(
    completions: Sequence[str],
    *,
    outcome: Optional[Sequence[str]] = None,
    confidence: Optional[Sequence[float]] = None,
    latency_ms: Optional[Sequence[float]] = None,
    model_id: Optional[Sequence[str]] = None,
    task_type: Optional[Sequence[str]] = None,
) -> List[float]:
    """Score each completion with the EXISTING DPO scorer.

    Reuses `DPOPairGenerator._score_candidate` rather than restating its
    weights (outcome .50 / confidence .25 / specialist .15 / latency .10).
    A second copy of that formula is a second thing to keep in step, and
    the two would rank candidates differently the first time either moved.
    """
    DPOConfig, DPOPairGenerator, ResponseCandidate = _load_scorer()
    gen = DPOPairGenerator(DPOConfig())
    n = len(completions)

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
            timestamp="",
            event_id="",
        )
        try:
            out.append(float(gen._score_candidate(cand)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("scorer fault on candidate %d: %s", i, exc)
            out.append(0.5)
    _ = n
    return out


async def candidate_reward(
    completions: Optional[Sequence[str]] = None,
    prompts: Optional[Sequence[Any]] = None,  # noqa: ARG001 — TRL passes it
    **kwargs: Any,
) -> Optional[List[Optional[float]]]:
    """TRL-compatible reward function over one group of sibling candidates.

    Signature follows TRL's contract: keyword arguments, returning a list
    of floats (or None entries) one per completion. Extra dataset columns
    — ``outcome``, ``confidence``, ``latency_ms``, ``model_id``,
    ``task_type`` — arrive through ``**kwargs`` and are exactly the fields
    the trajectory corpus already carries.

    Flow:

    1. Score every sibling with the existing DPO scorer.
    2. If the group is FLAT, add a structural-severity term. That term is
       a per-candidate measurement, so it can separate siblings the coarse
       success/failure label called equal — which is the only way to get a
       non-zero advantage honestly.
    3. If it is STILL flat, return ``None`` per sibling so TRL drops the
       group. See the module docstring: inventing a winner between
       indistinguishable answers is noise with a gradient attached.
    """
    texts = [str(c or "") for c in (completions or [])]
    if not texts:
        return []

    rewards = base_rewards(
        texts,
        outcome=kwargs.get("outcome"),
        confidence=kwargs.get("confidence"),
        latency_ms=kwargs.get("latency_ms"),
        model_id=kwargs.get("model_id"),
        task_type=kwargs.get("task_type"),
    )
    log_metric = kwargs.get("log_metric")
    log_extra = kwargs.get("log_extra")

    if not _is_flat(rewards):
        _emit(log_metric, "reward/flat_group", 0.0)
        return list(rewards)

    # ── Degenerate group: every sibling scored the same. ──
    if not _env_flag(_ENV_TIEBREAK, True):
        _emit(log_metric, "reward/flat_group", 1.0)
        return [None] * len(texts)

    sev = await structural_severities(texts)
    weight = _env_float(_ENV_TIEBREAK_WEIGHT, 0.30)
    adjusted = [r + weight * s.score for r, s in zip(rewards, sev)]
    _emit(log_extra, "severity_reason", [s.reason for s in sev])

    if _is_flat(adjusted):
        # Genuinely indistinguishable. Dropping is the honest answer; the
        # log names it so a corpus full of these is diagnosable rather
        # than silently absent from training.
        _emit(log_metric, "reward/dropped_flat_group", 1.0)
        logger.info(
            "[GRPO] group of %d has no separable signal (all %s) — dropped "
            "rather than assigned a fabricated advantage",
            len(texts), sev[0].reason if sev else "?",
        )
        return [None] * len(texts)

    _emit(log_metric, "reward/flat_group", 1.0)
    _emit(log_metric, "reward/tiebreak_rescued", 1.0)
    logger.debug(
        "[GRPO] flat group rescued by structural severity: %s",
        [f"{s.score:.2f}:{s.reason}" for s in sev],
    )
    return adjusted


def _emit(fn: Any, name: str, value: Any) -> None:
    """Call TRL's log_metric / log_extra when present. NEVER raises."""
    try:
        if callable(fn):
            fn(name, value)
    except Exception:  # noqa: BLE001
        pass


__all__ = [
    "Severity",
    "base_rewards",
    "candidate_reward",
    "structural_severities",
    "structural_severity",
]
