"""Turn a devtest session into the control the promotion gate compares against.

The gate refuses to promote an adapter without a MEASURED base-model
baseline. Nothing produced one: the harness writes a session summary and
the gate reads a JSON record, and no code connected them. This is that
connection, and it is deliberately the only place that decides what
"better" means.

## What the session actually contains, and the trap in it

`summary.json` carries `stats.{attempted,completed,failed}`,
`substance.{substantive,total}`, `branch_stats.commits`, and a per-op list.
Measured on session bt-2026-09-05-071918: `stats.completed == 1`, and that
one completion has `terminal_reason_code == "noop"` with
`files_changed == 0`.

So a raw completion count rewards a model that decides "no change needed"
every time. That is not a hypothetical: NO_OP terminations are documented
as common, and an op that declines to act completes just as cleanly as one
that ships a fix. A headline score built on `stats.completed` alone would
rank a model that does nothing above one that tries and sometimes fails.

`score_v1` therefore counts completions that DID something, and adds the
rarer, harder outcomes on top.

## Why the formula never changes shape

A composite that switches definition when a component is zero -- "no
applies, so fall back to repair accuracy" -- produces numbers that cannot
be compared. The gate's entire integrity is comparing like with like; it
carries a `metric` field and refuses on mismatch for exactly this reason.
A conditional formula would slip past that check while silently violating
what it protects.

The concern behind that idea is real, though: a short baseline with zero
applies must not collapse to a meaningless zero. The fix is a FIXED
formula whose components are simply zero when absent, so substantive
completions still carry signal at zero applies. Same shape every time,
comparable across runs, and never a binary.

`METRIC` encodes the formula version. Change the weights and the name
changes, so the gate refuses to compare across the change rather than
reporting a formula revision as a model improvement.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The formula's identity. The gate compares only records whose metric
#: matches, so bumping this makes an old baseline refuse rather than be
#: silently compared against a differently-computed number.
METRIC = "devtest_chain_closure_v1"

#: Weights. A substantive completion is the unit; an apply is worth more
#: because reaching APPLY means the candidate survived VALIDATE and GATE;
#: a commit more still, because it survived VERIFY. Ordered by how much of
#: the chain each one proves, not by taste.
W_SUBSTANTIVE = 1.0
W_APPLY = 3.0
W_COMMIT = 5.0

#: A completion that changed nothing and reported one of these is a
#: DECLINE, not a delivery. Counted separately so the score cannot be
#: farmed by declining.
_NOOP_CODES = frozenset({"noop", "no_op", "no_op_cosmetic", "noop_cosmetic"})


@dataclass
class SessionMetrics:
    """What one devtest session did, in terms the score can use."""

    session_id: str = ""
    attempted: int = 0
    completed: int = 0            # includes no-ops; kept for provenance
    failed: int = 0
    substantive: int = 0          # completed AND changed something
    noop_completions: int = 0
    applies: int = 0              # ops that changed >=1 file
    files_changed: int = 0
    commits: int = 0
    duration_s: float = 0.0
    stop_reason: str = ""
    session_outcome: str = ""

    def as_detail(self) -> Dict[str, Any]:
        return asdict(self)


def score_v1(m: SessionMetrics) -> float:
    """The headline. One formula, always the same shape.

    Substantive completions are the base; applies and commits add on. A run
    with zero applies still scores from its substantive completions rather
    than collapsing to zero, and a run that only ever declines scores zero
    no matter how many operations "completed".
    """
    return (
        W_SUBSTANTIVE * m.substantive
        + W_APPLY * m.applies
        + W_COMMIT * m.commits
    )


def _op_is_noop(op: Dict[str, Any]) -> bool:
    code = str(op.get("terminal_reason_code") or "").strip().lower()
    return code in _NOOP_CODES


def read_session(session_dir: Path) -> SessionMetrics:
    """Parse a session's ``summary.json``. Raises only on an unreadable file.

    Every field is defaulted, because a session killed mid-write still has
    a partial summary and a partial answer beats an exception here -- the
    caller is deciding whether to record a control, and "this run did
    almost nothing" is itself a finding.
    """
    path = Path(session_dir)
    if path.is_dir():
        path = path / "summary.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object")

    stats = raw.get("stats") or {}
    branch = raw.get("branch_stats") or {}
    ops: List[Dict[str, Any]] = [
        o for o in (raw.get("operations") or []) if isinstance(o, dict)
    ]

    completed_ops = [o for o in ops if str(o.get("status")) == "completed"]
    noops = [o for o in completed_ops if _op_is_noop(o)]
    # Substantive: completed, and it either touched a file or terminated for
    # a reason that is not a decline. Both signals are used because a
    # completion can be real without changing a file (a verified no-change
    # conclusion is rare but legitimate), and files_changed alone would miss
    # an op whose work landed elsewhere.
    substantive = [
        o for o in completed_ops
        if int(o.get("files_changed") or 0) > 0 or not _op_is_noop(o)
    ]
    applies = [o for o in ops if int(o.get("files_changed") or 0) > 0]

    return SessionMetrics(
        session_id=str(raw.get("session_id") or Path(session_dir).name),
        attempted=int(stats.get("attempted") or 0),
        completed=int(stats.get("completed") or len(completed_ops)),
        failed=int(stats.get("failed") or 0),
        substantive=len(substantive),
        noop_completions=len(noops),
        applies=len(applies),
        files_changed=sum(int(o.get("files_changed") or 0) for o in ops),
        commits=int(branch.get("commits") or 0),
        duration_s=float(raw.get("duration_s") or 0.0),
        stop_reason=str(raw.get("stop_reason") or ""),
        session_outcome=str(raw.get("session_outcome") or ""),
    )


def build_record(
    metrics: SessionMetrics,
    *,
    base_model: str,
    harness: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """A :class:`promotion_gate.BaselineRecord` from a session's metrics.

    ``harness`` is stamped so a baseline measured against a different
    VALIDATE configuration is visible in the record rather than inferred.
    """
    from reactor_core.deployment.promotion_gate import (  # noqa: PLC0415
        BaselineRecord,
    )

    detail = metrics.as_detail()
    detail["weights"] = {
        "substantive": W_SUBSTANTIVE, "apply": W_APPLY, "commit": W_COMMIT,
    }
    detail.update(extra or {})
    return BaselineRecord(
        score=score_v1(metrics),
        metric=METRIC,
        base_model=base_model,
        measured_at=datetime.now(tz=timezone.utc).isoformat(),
        harness=harness,
        session_id=metrics.session_id,
        detail=detail,
    )


def record_from_session(
    session_dir: Path,
    *,
    base_model: str,
    harness: str = "",
    path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Any:
    """Read a session, build the record, write it. Returns the record."""
    from reactor_core.deployment.promotion_gate import (  # noqa: PLC0415
        save_baseline,
    )

    metrics = read_session(Path(session_dir))
    record = build_record(metrics, base_model=base_model, harness=harness,
                          extra=extra)
    save_baseline(record, path=path)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Record a devtest session as the promotion baseline.",
    )
    ap.add_argument("session_dir", help="a .ouroboros/sessions/<id> directory")
    ap.add_argument("--base-model", required=True,
                    help="the tag/id this control was measured on")
    ap.add_argument("--harness", default="",
                    help="what produced it, e.g. 'devtest@<sha>'")
    ap.add_argument("--out", default="",
                    help="baseline path (default: REACTOR_BASELINE_PATH)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the record without writing it")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        metrics = read_session(Path(args.session_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSING: cannot read {args.session_dir}: {exc}", file=sys.stderr)
        return 1

    record = build_record(metrics, base_model=args.base_model,
                          harness=args.harness)
    print(json.dumps(asdict(record), indent=2))
    if metrics.attempted == 0:
        # A control measured on a session that never ran is not a control.
        print("REFUSING: the session attempted 0 operations — nothing to "
              "measure", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    from reactor_core.deployment.promotion_gate import save_baseline  # noqa: PLC0415
    save_baseline(record, path=Path(args.out) if args.out else None)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
