"""Score a candidate session and ask the gate, in one place.

The candidate half of the comparison. `devtest_baseline` reads a session
and writes the CONTROL; this reads a session the same way and hands the
number to the gate, without writing a baseline -- overwriting the control
with the candidate would destroy the comparison and leave a record
claiming the adapter is its own control.

The scoring is imported, not reimplemented. Two scorers that agree today
drift tomorrow, and the gate would not notice: it compares `metric` names,
and both halves would carry the same name while computing different
things. One function, one formula, one name.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

EXIT_PROMOTE = 0
#: The gate compared and said no. Retrain.
EXIT_HOLD = 1
#: The gate could not compare. Go measure -- a different fix entirely, so a
#: different code, or a scheduler cannot tell them apart.
EXIT_CANNOT_ANSWER = 2
EXIT_ERROR = 3


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a candidate devtest session and evaluate promotion.",
    )
    ap.add_argument("session_dir", help="the candidate's .ouroboros/sessions/<id>")
    ap.add_argument("--candidate-tag", required=True,
                    help="the ollama tag that was evaluated")
    ap.add_argument("--base-model", required=True,
                    help="the base the control was measured on")
    ap.add_argument("--baseline", default="",
                    help="baseline path (default: REACTOR_BASELINE_PATH)")
    ap.add_argument("--json-out", default="",
                    help="write the full comparison here")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from reactor_core.deployment import devtest_baseline as db
    from reactor_core.deployment import promotion_gate as pg

    try:
        metrics = db.read_session(Path(args.session_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSING: cannot read {args.session_dir}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    score = db.score_v1(metrics)
    print("=== candidate ===")
    print(f"  tag         : {args.candidate_tag}")
    print(f"  session     : {metrics.session_id}")
    print(f"  attempted   : {metrics.attempted}")
    print(f"  completed   : {metrics.completed} "
          f"({metrics.noop_completions} no-op, {metrics.substantive} substantive)")
    print(f"  applies     : {metrics.applies}  files: {metrics.files_changed}")
    print(f"  commits     : {metrics.commits}")
    print(f"  score       : {score:.4f}  [{db.METRIC}]")

    path = Path(args.baseline) if args.baseline else None
    baseline = pg.load_baseline(path=path)
    if baseline is not None:
        print("=== control ===")
        print(f"  base model  : {baseline.base_model}")
        print(f"  measured    : {baseline.measured_at}  ({baseline.harness})")
        print(f"  score       : {baseline.score:.4f}  [{baseline.metric}]")

    verdict = pg.evaluate_promotion(
        candidate_score=score,
        candidate_metric=db.METRIC,
        base_model=args.base_model,
        baseline=baseline,
        path=path,
    )
    print("=== verdict ===")
    print(f"  {verdict.summary()}")

    if args.json_out:
        payload = {
            "candidate": {
                "tag": args.candidate_tag,
                "score": score,
                "metric": db.METRIC,
                "metrics": asdict(metrics),
            },
            "baseline": asdict(baseline) if baseline else None,
            "verdict": {
                "promote": verdict.promote,
                "unanswerable": verdict.unanswerable,
                "reason": verdict.reason,
                "margin": verdict.margin,
            },
        }
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
        print(f"  written: {args.json_out}")

    if verdict.promote:
        print(f"\n  To serve it:  export JARVIS_LOCAL_MODEL_NAME={args.candidate_tag}")
        return EXIT_PROMOTE
    return EXIT_CANNOT_ANSWER if verdict.unanswerable else EXIT_HOLD


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
