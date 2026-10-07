"""The corpus reader joins O+V's landing-provenance labels -- and only verified ones."""
from __future__ import annotations

import json
from pathlib import Path

from reactor_core.training import grpo_pipeline as gp


def _row(eid: str, outcome: str = "unknown", train: bool = False) -> dict:
    return {"event_id": eid, "event_type": "interaction", "user_input": f"p {eid}",
            "assistant_output": "def test_x():\n    assert 1\n", "outcome": outcome,
            "confidence": 0.5, "metadata": {"op_id": "op-1", "candidate_hash": eid,
                                             "should_train": train, "draw_kind": "primary"}}


def _ledger(events: Path, labels, *, break_chain_at: int = -1) -> None:
    d = events / "provenance"
    d.mkdir(parents=True)
    prev, lines = "genesis", []
    for i, lab in enumerate(labels):
        rec = {"payload": lab, "prev_hash": "WRONG" if i == break_chain_at else prev,
               "record_hash": f"h{i}", "mac": ""}
        prev = f"h{i}"
        lines.append(json.dumps(rec))
    (d / "landing_labels.jsonl").write_text("\n".join(lines) + "\n")


def _corpus(tmp_path: Path, rows) -> Path:
    d = tmp_path / "events"
    d.mkdir()
    (d / "experience_20261007.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _label(eid: str, surviving: bool = True) -> dict:
    return {"subject_event_id": eid, "landed": True, "surviving": surviving, "commit_sha": "abc123"}


def test_landed_surviving_row_becomes_a_trainable_success(tmp_path):
    d = _corpus(tmp_path, [_row("e1"), _row("e2")])
    _ledger(d, [_label("e1")])
    rows = {r["event_id"]: r for r in gp.iter_trajectory_rows(d)}
    assert set(rows) == {"e1"}                       # e2 was never trainable
    assert rows["e1"]["outcome"] == "success" and rows["e1"]["confidence"] == 1.0
    assert rows["e1"]["metadata"]["landed_commit"] == "abc123"
    assert rows["e1"]["metadata"]["verdict_source"] == "landing_provenance"


def test_reverted_landing_is_marked_never_promoted(tmp_path):
    d = _corpus(tmp_path, [_row("e1", outcome="failure", train=True)])
    _ledger(d, [_label("e1"), _label("e1", surviving=False)])
    (row,) = list(gp.iter_trajectory_rows(d))
    assert row["outcome"] == "failure"
    assert row["metadata"]["landed"] is True and row["metadata"]["landing_surviving"] is False


def test_labels_after_a_broken_chain_are_ignored(tmp_path):
    d = _corpus(tmp_path, [_row("e1"), _row("e2")])
    _ledger(d, [_label("e1"), _label("e2")], break_chain_at=1)
    assert [r["event_id"] for r in gp.iter_trajectory_rows(d)] == ["e1"]


def test_the_ledger_is_never_read_as_a_corpus_row(tmp_path):
    d = _corpus(tmp_path, [_row("e1")])
    _ledger(d, [_label("e1")])
    assert len(list(gp.iter_trajectory_rows(d, trainable_only=False))) == 1


def test_no_ledger_changes_nothing(tmp_path):
    d = _corpus(tmp_path, [_row("e1", outcome="success", train=True)])
    (row,) = list(gp.iter_trajectory_rows(d))
    assert "landed" not in row["metadata"]
