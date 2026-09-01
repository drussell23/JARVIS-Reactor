#!/usr/bin/env python3
"""Is this corpus worth a training run? Answer in JSON, from ANY venv.

The process boundary exists for the same reason `verify_candidate.py` does:
JARVIS and Reactor-Core are separate repositories with separate
virtualenvs, and the soak-side venv has no torch. A cross-repo import is
impossible, so the contract is a COMMAND and a JSON document, never shared
code.

## Why this is not a second opinion

Everything load-bearing here is reactor's own implementation, loaded by
path:

  * ``grpo_pipeline.iter_trajectory_rows`` -- the corpus reader, including
    the ``event_type == "interaction"`` and ``metadata.should_train``
    filters. Those are not obvious and a second copy would drift; a gate
    that counted rows the trainer would discard is worse than no gate.
  * ``grpo_verifier.verify_static`` -- the same grader the reward uses, so
    "differentiated" here means differentiated THERE.
  * ``grpo_reward._is_flat`` / ``_FLAT_EPS`` -- the exact predicate that
    drops a group inside the trainer.

If the trainer's notion of a usable group changes, this gate changes with
it, because it is the same code.

## What it answers

A GRPO group needs two things at once, and the corpus has repeatedly had
one without the other:

  1. >= 2 responses to the SAME prompt, and
  2. rewards that are not all equal.

A corpus can carry 74 rows and 14 multi-response prompts and still be
worth nothing, because every row inherited one op-level verdict and the
whole group scores identically. That state is indistinguishable from a
healthy corpus by row count alone, which is exactly how an automated
trigger burns an hour of GPU to produce a checkpoint trained on nothing.

## Exit codes

  0  trainable  -- at least ``--min-groups`` groups survive
  2  refused    -- corpus read fine, but there is nothing to learn from
  1  error      -- could not answer

2 is distinct from 1 on purpose: "I looked and the answer is no" must not
be indistinguishable from "I broke", or a caller cannot tell a healthy
refusal from a fault.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_TRAINING = _REPO / "reactor_core" / "training"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _load(mod_name: str):
    """Import one training module BY PATH.

    ``reactor_core/__init__`` eagerly imports the ML stack, so a normal
    import drags torch/peft/trl into a venv that may have none of them.
    Each module loaded here is stdlib-only at module scope; the heavy
    imports inside them are lazy and never reached on this path.
    """
    path = _TRAINING / f"{mod_name}.py"
    spec = importlib.util.spec_from_file_location(f"_pf_{mod_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses needs it registered
    spec.loader.exec_module(mod)
    return mod


def _default_telemetry_dir() -> Path:
    """Same directory the recorder writes and the trainer reads.

    Resolution order mirrors the rest of the flywheel: the DPO variable
    first (that is what the generator honours), then the recorder's own,
    then the canonical Trinity path.
    """
    for var in ("DPO_TELEMETRY_DIR", "JARVIS_TRAJECTORY_RECORDER_DIR"):
        val = _env(var)
        if val:
            return Path(val)
    return Path.home() / ".jarvis" / "trinity" / "events"


def analyse(
    telemetry_dir: Path,
    *,
    min_group: int,
    trainable_only: bool,
) -> Dict[str, Any]:
    """Group the corpus by prompt and score each group. Never raises."""
    pipeline = _load("grpo_pipeline")
    verifier = _load("grpo_verifier")
    reward = _load("grpo_reward")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_seen = 0
    for row in pipeline.iter_trajectory_rows(
        telemetry_dir, trainable_only=trainable_only,
    ):
        rows_seen += 1
        groups[str(row.get("user_input") or "")].append(row)

    trainable: List[Dict[str, Any]] = []
    flat: List[Dict[str, Any]] = []
    singleton = 0
    verdict_sources: Dict[str, int] = defaultdict(int)

    for prompt, rows in groups.items():
        for r in rows:
            src = str((r.get("metadata") or {}).get("verdict_source") or "")
            verdict_sources[src or "__unset__"] += 1
        if len(rows) < max(2, min_group):
            singleton += 1
            continue
        scores = [
            float(verifier.verify_static(str(r.get("assistant_output") or "")).score)
            for r in rows
        ]
        entry = {
            "prompt_head": prompt[:80],
            "responses": len(rows),
            "scores": [round(s, 4) for s in scores],
            "spread": round(max(scores) - min(scores), 4) if scores else 0.0,
        }
        (flat if reward._is_flat(scores) else trainable).append(entry)

    return {
        "telemetry_dir": str(telemetry_dir),
        "rows": rows_seen,
        "prompts": len(groups),
        "groups_below_min": singleton,
        "flat_groups": len(flat),
        "trainable_groups": len(trainable),
        "flat_eps": getattr(reward, "_FLAT_EPS", None),
        "min_group": min_group,
        "trainable_only": trainable_only,
        "verdict_sources": dict(verdict_sources),
        "examples": trainable[:5],
        "flat_examples": flat[:3],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--telemetry-dir", default="")
    ap.add_argument(
        "--min-groups", type=int,
        default=_env_int("TRINITY_GRPO_MIN_TRAINABLE_GROUPS", 1),
        help="how many differentiated groups justify a run "
             "(env TRINITY_GRPO_MIN_TRAINABLE_GROUPS)",
    )
    ap.add_argument(
        "--min-group-size", type=int,
        default=_env_int("TRINITY_GRPO_MIN_GROUP_SIZE", 2),
        help="responses a prompt needs before it can be scored at all "
             "(env TRINITY_GRPO_MIN_GROUP_SIZE)",
    )
    ap.add_argument(
        "--include-untrainable", action="store_true",
        help="count rows the classifier excluded from training. Diagnostic "
             "only -- it deliberately disagrees with the trainer.",
    )
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    tdir = Path(args.telemetry_dir) if args.telemetry_dir else _default_telemetry_dir()
    try:
        report = analyse(
            tdir,
            min_group=args.min_group_size,
            trainable_only=not args.include_untrainable,
        )
    except Exception as exc:  # noqa: BLE001 — a gate must explain itself
        err = {"error": f"{type(exc).__name__}: {exc}", "telemetry_dir": str(tdir)}
        print(json.dumps(err, indent=2))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(err, indent=2), encoding="utf-8")
        return 1

    report["min_groups_required"] = args.min_groups
    report["trainable"] = report["trainable_groups"] >= args.min_groups
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8")
    return 0 if report["trainable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
