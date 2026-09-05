"""A devtest session becomes the control the promotion gate compares against.

The trap this module exists to avoid is in the data. Measured on session
bt-2026-09-05-071918: `stats.completed == 1`, and that one completion has
`terminal_reason_code == "noop"` with `files_changed == 0`. NO_OP
terminations are documented as common, and an op that declines to act
completes as cleanly as one that ships a fix — so a headline built on raw
completions would rank a model that does nothing above one that tries.

The second property under test is that the formula never changes SHAPE. A
composite that switches definition when a component is zero produces
numbers that cannot be compared, which is precisely what the gate's
`metric` field exists to prevent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.deployment import devtest_baseline as db  # noqa: E402
from reactor_core.deployment import promotion_gate as pg  # noqa: E402

BASE = "qwen3-coder:30b"


def _session(tmp_path: Path, *, ops, commits=0, attempted=None, name="bt-test"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for o in ops if o.get("status") == "completed")
    payload = {
        "session_id": name,
        "schema_version": 2,
        "duration_s": 1640.4,
        "stop_reason": "idle_timeout",
        "session_outcome": "complete",
        "stats": {
            "attempted": len(ops) if attempted is None else attempted,
            "completed": completed,
            "failed": sum(1 for o in ops if o.get("status") == "failed"),
            "cancelled": 0, "queued": 0,
        },
        "branch_stats": {"commits": commits, "files_changed": 0,
                         "insertions": 0, "deletions": 0},
        "operations": ops,
    }
    (d / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def _op(status="completed", files=0, code="", op_id="op-1"):
    return {"op_id": op_id, "status": status, "files_changed": files,
            "terminal_reason_code": code}


# ---------------------------------------------------------------------------
# The trap: a no-op is a decline, not a delivery
# ---------------------------------------------------------------------------


def test_the_real_session_shape_scores_zero(tmp_path) -> None:
    """bt-2026-09-05-071918 exactly: 1 completed, and it is a noop."""
    d = _session(tmp_path, ops=[_op("completed", 0, "noop")]
                 + [_op("failed", 0, "", f"op-{i}") for i in range(6)])
    m = db.read_session(d)
    assert m.completed == 1, "the raw counter still says 1"
    assert m.noop_completions == 1
    assert m.substantive == 0, "but nothing was delivered"
    assert db.score_v1(m) == 0.0


def test_a_model_that_only_declines_cannot_farm_the_score(tmp_path) -> None:
    d = _session(tmp_path, ops=[_op("completed", 0, "noop", f"op-{i}")
                                for i in range(50)])
    m = db.read_session(d)
    assert m.completed == 50
    assert db.score_v1(m) == 0.0, (
        "50 clean completions that changed nothing must not outrank one real fix")


def test_one_real_fix_outranks_fifty_declines(tmp_path) -> None:
    declines = _session(tmp_path, name="a",
                        ops=[_op("completed", 0, "noop", f"op-{i}")
                             for i in range(50)])
    one_fix = _session(tmp_path, name="b",
                       ops=[_op("completed", 1, "applied")])
    assert db.score_v1(db.read_session(one_fix)) > \
        db.score_v1(db.read_session(declines))


def test_a_completion_that_changed_a_file_is_substantive(tmp_path) -> None:
    d = _session(tmp_path, ops=[_op("completed", 3, "applied")])
    m = db.read_session(d)
    assert m.substantive == 1 and m.applies == 1 and m.files_changed == 3


def test_a_non_noop_completion_counts_even_with_no_files(tmp_path) -> None:
    """A verified no-change conclusion is rare but legitimate; only the
    explicit decline codes are excluded."""
    d = _session(tmp_path, ops=[_op("completed", 0, "verified")])
    assert db.read_session(d).substantive == 1


# ---------------------------------------------------------------------------
# The formula never changes shape
# ---------------------------------------------------------------------------


def test_zero_applies_still_scores_from_substantive_work(tmp_path) -> None:
    """The real concern behind a 'fallback' formula, solved without one."""
    d = _session(tmp_path, ops=[_op("completed", 0, "verified", f"op-{i}")
                                for i in range(4)])
    m = db.read_session(d)
    assert m.applies == 0
    assert db.score_v1(m) == 4.0, "not a binary zero"


def test_the_weights_order_the_chain_by_what_it_proves(tmp_path) -> None:
    sub = db.score_v1(db.SessionMetrics(substantive=1))
    app = db.score_v1(db.SessionMetrics(substantive=1, applies=1))
    com = db.score_v1(db.SessionMetrics(substantive=1, applies=1, commits=1))
    assert sub < app < com, "an apply proves more than a completion; a commit more"


def test_the_score_is_a_pure_function_of_the_metrics() -> None:
    m = db.SessionMetrics(substantive=2, applies=1, commits=1)
    assert db.score_v1(m) == db.score_v1(m)
    assert db.score_v1(m) == 2 * db.W_SUBSTANTIVE + db.W_APPLY + db.W_COMMIT


def test_the_metric_name_is_versioned() -> None:
    """Change the weights and the name must change, so the gate refuses to
    compare across the change instead of reporting a formula revision as a
    model improvement."""
    assert db.METRIC.endswith("_v1")
    import inspect
    src = inspect.getsource(db)
    assert "METRIC = " in src


# ---------------------------------------------------------------------------
# The record the gate reads
# ---------------------------------------------------------------------------


def test_the_record_carries_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(pg.ENV_BASELINE_PATH, str(tmp_path / "b.json"))
    d = _session(tmp_path, ops=[_op("completed", 2, "applied")], commits=1)
    rec = db.record_from_session(d, base_model=BASE, harness="devtest@abc123")
    assert rec.metric == db.METRIC
    assert rec.base_model == BASE
    assert rec.harness == "devtest@abc123"
    assert rec.session_id == "bt-test"
    assert rec.detail["applies"] == 1 and rec.detail["commits"] == 1
    assert rec.detail["weights"]["apply"] == db.W_APPLY


def test_the_written_record_is_what_the_gate_reads(tmp_path, monkeypatch) -> None:
    """The join under test: one writes, the other reads, same schema."""
    path = tmp_path / "b.json"
    monkeypatch.setenv(pg.ENV_BASELINE_PATH, str(path))
    d = _session(tmp_path, ops=[_op("completed", 1, "applied")])
    written = db.record_from_session(d, base_model=BASE)
    back = pg.load_baseline(path=path)
    assert back is not None
    assert back.score == written.score and back.metric == db.METRIC

    verdict = pg.evaluate_promotion(
        candidate_score=written.score + 1.0, candidate_metric=db.METRIC,
        base_model=BASE, path=path,
    )
    assert verdict.promote, verdict.reason
    assert not verdict.unanswerable, "the gate can now ANSWER"


def test_a_stale_metric_name_refuses_rather_than_comparing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "b.json"
    monkeypatch.setenv(pg.ENV_BASELINE_PATH, str(path))
    d = _session(tmp_path, ops=[_op("completed", 1, "applied")])
    db.record_from_session(d, base_model=BASE)
    v = pg.evaluate_promotion(candidate_score=99.0,
                              candidate_metric="devtest_chain_closure_v2",
                              base_model=BASE, path=path)
    assert not v.promote and v.unanswerable
    assert "unlike things" in v.reason


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_a_partial_summary_still_parses(tmp_path) -> None:
    """A session killed mid-write leaves a partial summary; 'this run did
    almost nothing' is itself a finding."""
    d = tmp_path / "partial"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"session_id": "x"}),
                                    encoding="utf-8")
    m = db.read_session(d)
    assert m.attempted == 0 and m.substantive == 0
    assert db.score_v1(m) == 0.0


def test_a_session_that_attempted_nothing_is_refused_by_the_cli(tmp_path, capsys) -> None:
    d = _session(tmp_path, ops=[], attempted=0)
    rc = db.main([str(d), "--base-model", BASE, "--dry-run"])
    assert rc == 2, "a control measured on a session that never ran is not a control"


def test_the_cli_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "b.json"
    monkeypatch.setenv(pg.ENV_BASELINE_PATH, str(path))
    d = _session(tmp_path, ops=[_op("completed", 1, "applied")])
    assert db.main([str(d), "--base-model", BASE, "--dry-run"]) == 0
    assert not path.exists()


def test_the_cli_writes_the_baseline(tmp_path, monkeypatch) -> None:
    path = tmp_path / "b.json"
    d = _session(tmp_path, ops=[_op("completed", 1, "applied")])
    assert db.main([str(d), "--base-model", BASE, "--out", str(path)]) == 0
    assert pg.load_baseline(path=path) is not None


def test_a_missing_session_is_an_error_not_a_crash(tmp_path, capsys) -> None:
    assert db.main([str(tmp_path / "nope"), "--base-model", BASE]) == 1
