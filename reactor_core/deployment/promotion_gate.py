"""An adapter takes the live tag only if something MEASURED says it should.

Without this, a night-shift cycle that trains and deploys is a machine that
replaces the model O+V generates with every night, on the strength of
having produced a file. Loss went down is not evidence: GRPO's loss is
dominated by the router auxiliary term, and a run can complete, save, and
convert while having learned nothing the pipeline can see.

So promotion asks one question -- is this adapter at least as good as the
base model, on the same harness? -- and REFUSES when it cannot answer.
That refusal is the whole point. The failure this prevents is not a bad
score; it is a promotion decided by a score that does not exist, is stale,
or was measured against something else.

## What "cannot answer" covers, and why each one refuses

* **No baseline on disk.** Nothing to compare against. A first cycle on a
  fresh box must not promote by default -- "no data" is not "no
  regression".
* **A baseline that does not parse.** A truncated or half-written record
  is indistinguishable from a good one if you only check existence, so it
  is read and validated, not stat'd.
* **A stale baseline.** The harness changes. A number measured against a
  VALIDATE stage with different walls is not a control for today's run,
  and silently comparing across that is how a harness fix looks like a
  model improvement.
* **A baseline for a different base model.** Comparing an adapter over
  Qwen3-Coder against a baseline measured on Qwen2.5 says nothing.
* **No candidate score.** An evaluation that did not run leaves nothing to
  judge; promoting anyway is promoting on hope.

## What it does NOT do

It does not decide what to measure. `score` is whatever the harness
reports -- today the devtest's completed-operation count, tomorrow
something better -- and this module only compares like with like, which is
why `metric` is recorded and mismatches refuse.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Where the baseline record lives. One file, so the number that governs
#: promotion has exactly one home and can be inspected by hand.
ENV_BASELINE_PATH = "REACTOR_BASELINE_PATH"
DEFAULT_BASELINE_PATH = "~/.jarvis/baselines/devtest_baseline.json"

#: How long a baseline stays a valid control, in hours. The harness moves;
#: a number measured against different walls is not a control for today.
ENV_BASELINE_MAX_AGE_H = "REACTOR_BASELINE_MAX_AGE_H"
DEFAULT_BASELINE_MAX_AGE_H = 336.0          # 14 days

#: How much WORSE than baseline still promotes, as a fraction. Default 0:
#: an adapter must not regress. A positive value buys tolerance for a noisy
#: metric and should be set from measured run-to-run variance, not taste.
ENV_REGRESSION_TOLERANCE = "REACTOR_PROMOTION_TOLERANCE"
DEFAULT_REGRESSION_TOLERANCE = 0.0

#: Emergency override. Present so an operator who has looked at the numbers
#: can ship without editing code -- and named so it can never be mistaken
#: for a default.
ENV_FORCE_PROMOTE = "REACTOR_PROMOTE_WITHOUT_BASELINE"


@dataclass
class BaselineRecord:
    """A measured control. Every field is needed to know it still applies."""

    score: float
    metric: str
    base_model: str
    measured_at: str                      # ISO-8601, UTC
    harness: str = ""                     # what produced it
    session_id: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def age_hours(self, *, now: Optional[datetime] = None) -> Optional[float]:
        try:
            when = datetime.fromisoformat(self.measured_at)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        current = now or datetime.now(tz=timezone.utc)
        return (current - when).total_seconds() / 3600.0


@dataclass
class PromotionVerdict:
    promote: bool
    reason: str
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    margin: Optional[float] = None
    #: True when the gate could not ANSWER, as opposed to answering "no".
    #: A scheduler should surface these differently: one is a regression,
    #: the other is a missing measurement, and they need opposite fixes.
    unanswerable: bool = False

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else (
            "CANNOT ANSWER" if self.unanswerable else "HOLD")
        if self.margin is not None:
            return (f"[promotion] {head}: {self.reason} "
                    f"(candidate {self.candidate_score:.4f} vs baseline "
                    f"{self.baseline_score:.4f}, margin {self.margin:+.4f})")
        return f"[promotion] {head}: {self.reason}"


def baseline_path() -> Path:
    raw = (os.environ.get(ENV_BASELINE_PATH, "") or "").strip() \
        or DEFAULT_BASELINE_PATH
    return Path(os.path.expanduser(raw))


def _max_age_h() -> float:
    raw = (os.environ.get(ENV_BASELINE_MAX_AGE_H, "") or "").strip()
    try:
        v = float(raw)
        return v if v > 0 else DEFAULT_BASELINE_MAX_AGE_H
    except ValueError:
        return DEFAULT_BASELINE_MAX_AGE_H


def _tolerance() -> float:
    raw = (os.environ.get(ENV_REGRESSION_TOLERANCE, "") or "").strip()
    try:
        v = float(raw)
        return v if v >= 0 else DEFAULT_REGRESSION_TOLERANCE
    except ValueError:
        return DEFAULT_REGRESSION_TOLERANCE


def save_baseline(record: BaselineRecord, *, path: Optional[Path] = None) -> Path:
    """Write the control. Atomic, because a half-written baseline that still
    parses is worse than none."""
    target = Path(path) if path is not None else baseline_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    os.replace(tmp, target)
    logger.info("[promotion] baseline recorded: %s=%.4f on %s (%s)",
                record.metric, record.score, record.base_model, target)
    return target


def load_baseline(*, path: Optional[Path] = None) -> Optional[BaselineRecord]:
    """Read and VALIDATE the control. None when it cannot be trusted."""
    target = Path(path) if path is not None else baseline_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("[promotion] baseline at %s is unreadable: %s", target, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("[promotion] baseline at %s is not an object", target)
        return None
    try:
        return BaselineRecord(
            score=float(raw["score"]),
            metric=str(raw["metric"]),
            base_model=str(raw["base_model"]),
            measured_at=str(raw["measured_at"]),
            harness=str(raw.get("harness", "")),
            session_id=str(raw.get("session_id", "")),
            detail=dict(raw.get("detail") or {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        # A record missing a field is a record whose provenance is unknown.
        logger.warning("[promotion] baseline at %s is incomplete: %s", target, exc)
        return None


def evaluate_promotion(
    *,
    candidate_score: Optional[float],
    candidate_metric: str,
    base_model: str,
    baseline: Optional[BaselineRecord] = None,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> PromotionVerdict:
    """Decide whether an adapter may take the live tag.

    Refuses -- with ``unanswerable=True`` -- whenever the comparison cannot
    be made honestly. Answers "hold" only when it CAN compare and the
    candidate is worse.
    """
    if baseline is None:
        baseline = load_baseline(path=path)

    if baseline is None:
        if (os.environ.get(ENV_FORCE_PROMOTE, "") or "").strip().lower() in (
            "1", "true", "yes", "on",
        ):
            logger.warning(
                "[promotion] no baseline, but %s is set — promoting on the "
                "operator's word", ENV_FORCE_PROMOTE,
            )
            return PromotionVerdict(
                promote=True,
                reason=f"no baseline; {ENV_FORCE_PROMOTE} set by the operator",
                candidate_score=candidate_score,
            )
        return PromotionVerdict(
            promote=False, unanswerable=True, candidate_score=candidate_score,
            reason=(
                f"no usable baseline at {Path(path) if path else baseline_path()} — "
                "run the base-model control first; 'no data' is not "
                "'no regression'"
            ),
        )

    age = baseline.age_hours(now=now)
    if age is None:
        return PromotionVerdict(
            promote=False, unanswerable=True, candidate_score=candidate_score,
            baseline_score=baseline.score,
            reason=f"baseline timestamp {baseline.measured_at!r} is unreadable",
        )
    limit = _max_age_h()
    if age > limit:
        return PromotionVerdict(
            promote=False, unanswerable=True, candidate_score=candidate_score,
            baseline_score=baseline.score,
            reason=(
                f"baseline is {age:.0f}h old (limit {limit:.0f}h) — the "
                "harness has moved; re-measure the control"
            ),
        )

    if baseline.base_model and base_model and baseline.base_model != base_model:
        return PromotionVerdict(
            promote=False, unanswerable=True, candidate_score=candidate_score,
            baseline_score=baseline.score,
            reason=(
                f"baseline was measured on {baseline.base_model!r}, this "
                f"adapter is over {base_model!r} — not a control"
            ),
        )

    if baseline.metric and candidate_metric and baseline.metric != candidate_metric:
        return PromotionVerdict(
            promote=False, unanswerable=True, candidate_score=candidate_score,
            baseline_score=baseline.score,
            reason=(
                f"baseline measures {baseline.metric!r}, candidate reports "
                f"{candidate_metric!r} — comparing unlike things"
            ),
        )

    if candidate_score is None:
        return PromotionVerdict(
            promote=False, unanswerable=True, baseline_score=baseline.score,
            reason="the candidate has no score — evaluation did not run",
        )

    tol = _tolerance()
    floor = baseline.score - abs(baseline.score) * tol
    margin = candidate_score - baseline.score
    if candidate_score >= floor:
        return PromotionVerdict(
            promote=True, baseline_score=baseline.score,
            candidate_score=candidate_score, margin=margin,
            reason=(
                f"at or above the control"
                + (f" within a {tol:.0%} tolerance" if tol else "")
            ),
        )
    return PromotionVerdict(
        promote=False, baseline_score=baseline.score,
        candidate_score=candidate_score, margin=margin,
        reason=f"regression against the base model (floor {floor:.4f})",
    )
