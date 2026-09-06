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


def _install_light_packages() -> None:
    """Make ``reactor_core.training.*`` importable WITHOUT running either
    package ``__init__``.

    ``reactor_core/__init__`` eagerly imports the ML stack, so a plain
    import drags torch/peft/trl into a venv that has none of them. The
    previous approach loaded each module from a bare file spec under a
    private ``_pf_`` name, and it worked only while every training module
    was stdlib-only at module scope -- an invariant stated in a docstring
    and enforced by nothing.

    It stopped being true. `grpo_pipeline` gained
    ``from reactor_core.training.prompt_budget import ...`` at module
    scope, Python resolved that through the REAL package, and the gate
    died with ``No module named 'reactor_core'`` from the soak venv --
    reporting an ERROR (exit 1) where its whole purpose is to answer
    "trainable or not". A gate that cannot run is worse than no gate,
    because the caller reads a fault where it expected a verdict.

    So the boundary is made structural instead of promised. Two namespace
    packages are placed in ``sys.modules`` with nothing but a
    ``__path__``; ordinary import machinery then finds every sibling by
    absolute name, and neither ``__init__`` is ever executed. Any future
    absolute sibling import just works, and no module has to remember a
    rule about what it may import at the top of the file.

    Loading is normal after this, so a module has ONE identity. The old
    ``_pf_`` aliasing gave a module two, and a dataclass or isinstance
    check spanning them would have compared unrelated types.

    A real ``reactor_core`` already imported (the trainer venv, running
    with torch present) is left completely alone.
    """
    if "reactor_core" in sys.modules:
        return
    import types  # noqa: PLC0415 — stdlib, only needed on this path

    pkg = types.ModuleType("reactor_core")
    pkg.__path__ = [str(_REPO / "reactor_core")]      # type: ignore[attr-defined]
    sub = types.ModuleType("reactor_core.training")
    sub.__path__ = [str(_TRAINING)]                   # type: ignore[attr-defined]
    pkg.training = sub                                # type: ignore[attr-defined]
    sys.modules["reactor_core"] = pkg
    sys.modules["reactor_core.training"] = sub


def _load(mod_name: str):
    """Import one training module without the package ``__init__``.

    The heavy imports inside these modules are lazy and never reached on
    this path; see :func:`_install_light_packages` for why the isolation
    is done with namespace packages rather than by-path specs.
    """
    _install_light_packages()
    if not (_TRAINING / f"{mod_name}.py").is_file():
        raise ImportError(f"cannot load {_TRAINING / (mod_name + '.py')}")
    return importlib.import_module(f"reactor_core.training.{mod_name}")


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
    min_spread: float = 0.0,
) -> Dict[str, Any]:
    """Group the corpus by prompt and score each group. Never raises."""
    pipeline = _load("grpo_pipeline")
    verifier = _load("grpo_verifier")
    reward = _load("grpo_reward")

    # Keyed on the TASK, not on the exact prompt bytes. Ambient memory
    # sections advance between draw 1 and draw 2 of the same op, which
    # split genuine sibling groups into singletons and made the corpus
    # report a sibling failure that never happened. `task_group_key` is
    # reactor's own function, so the gate and the trainer cannot hold two
    # notions of "the same prompt".
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_seen = 0
    for row in pipeline.iter_trajectory_rows(
        telemetry_dir, trainable_only=trainable_only,
    ):
        rows_seen += 1
        groups[pipeline.task_group_key(row.get("user_input"))].append(row)

    trainable: List[Dict[str, Any]] = []
    #: The FULL prompt text of every trainable group, in group order.
    #: ``examples`` keeps only 80-char heads because a report carrying 27
    #: prompts of ~24k chars each is not a report. The trainer needs the
    #: whole string to select on, so it travels separately and the runner
    #: pops it before writing JSON. This is the gate's own verdict reused,
    #: which is the point: the runner must not form a second opinion about
    #: which prompts carry contrast.
    trainable_prompts: List[str] = []
    flat: List[Dict[str, Any]] = []
    singleton = 0
    verdict_sources: Dict[str, int] = defaultdict(int)

    for _key, rows in groups.items():
        # The group is KEYED on the task, but everything downstream needs a
        # real prompt: the verifier must grade in the context the trainer
        # will generate from, and `build_prompt_dataset(only_prompts=...)`
        # matches on the FULL text's `.strip()`. Handing either of them the
        # key would silently select nothing. First row wins, deterministic
        # because `iter_trajectory_rows` walks files in sorted order.
        prompt = str(rows[0].get("user_input") or "")
        for r in rows:
            src = str((r.get("metadata") or {}).get("verdict_source") or "")
            verdict_sources[src or "__unset__"] += 1
        if len(rows) < max(2, min_group):
            singleton += 1
            continue
        # Grouped, not per-row: Gate 3 must measure what the TRAINER will
        # optimise, and the trainer grades a group in the context it
        # shares (task intent + the code each sibling chose differently).
        # Scoring rows independently here and jointly there would make the
        # gate an answer about a reward nothing uses.
        texts = [str(r.get("assistant_output") or "") for r in rows]
        scores = [
            float(v.score)
            for v in verifier.verify_group(texts, prompt=prompt)
        ]
        entry = {
            "prompt_head": prompt[:80],
            "responses": len(rows),
            "scores": [round(s, 6) for s in scores],
            # 6dp, not 4: _FLAT_EPS is 1e-06, so a group can be legitimately
            # non-flat and still display as "0.0" at 4dp -- a report that
            # contradicts its own verdict.
            "spread": round(max(scores) - min(scores), 6) if scores else 0.0,
        }
        # Two thresholds, not one. `_is_flat` is the TRAINER's guard and
        # answers "is there any difference at all" (_FLAT_EPS = 1e-06).
        # That is necessary and not sufficient: GRPO normalises advantage by
        # the group's standard deviation, so a spread of 3e-05 yields the
        # SAME magnitude advantage as a spread of 0.5. Admitting such a
        # group does not train on a weak signal -- it amplifies measurement
        # noise to full scale and backpropagates it.
        #
        # `min_spread` is therefore a separate, operator-set floor on what
        # counts as a MEANINGFUL difference. Default 0.0 preserves the
        # trainer's own semantics exactly; raise it to demand real
        # separation.
        span = (max(scores) - min(scores)) if scores else 0.0
        if reward._is_flat(scores) or span < min_spread:
            entry["excluded_by"] = (
                "flat" if reward._is_flat(scores) else "below_min_spread"
            )
            flat.append(entry)
        else:
            trainable.append(entry)
            trainable_prompts.append(prompt)

    return {
        "telemetry_dir": str(telemetry_dir),
        "rows": rows_seen,
        "prompts": len(groups),
        "groups_below_min": singleton,
        "flat_groups": len(flat),
        "trainable_groups": len(trainable),
        "trainable_prompts": trainable_prompts,
        "flat_eps": getattr(reward, "_FLAT_EPS", None),
        "min_group": min_group,
        "min_spread": min_spread,
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
        "--min-spread", type=float,
        default=float(_env("TRINITY_GRPO_MIN_SPREAD", "0") or 0.0),
        help="minimum MEANINGFUL reward spread. _is_flat only asks whether "
             "a difference exists; GRPO divides advantage by group std, so "
             "a 3e-05 spread trains as hard as a 0.5 one "
             "(env TRINITY_GRPO_MIN_SPREAD)",
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
            min_spread=args.min_spread,
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
