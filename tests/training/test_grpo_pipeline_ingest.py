"""Reactor ingestion pairs only genuine draws, and one hash is one row.

Mirrors the jarvis-side harvest filter so the preflight's Gate 3 and the
training pipeline read the SAME sample the recorder meant to offer: an L2
repair iteration is not a sibling of the draw it repaired, and a
candidate re-recorded under a second attempt is not a second candidate.
"""
from __future__ import annotations

import json
from pathlib import Path

from reactor_core.training import grpo_pipeline as gp


def _row(op: str, attempt: int, body: str, *, kind: str = "", h: str = "",
         train: bool = True) -> dict:
    meta = {"op_id": op, "attempt_index": attempt, "should_train": train}
    if kind:
        meta["draw_kind"] = kind
    if h:
        meta["candidate_hash"] = h
    return {
        "event_type": "interaction",
        "user_input": f"prompt for {op}",
        "assistant_output": body,
        "outcome": "success",
        "metadata": meta,
    }


def _corpus(tmp_path: Path, rows) -> Path:
    d = tmp_path / "events"
    d.mkdir()
    (d / "a.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


def test_repair_and_retry_rows_are_excluded_by_default(tmp_path: Path) -> None:
    d = _corpus(tmp_path, [
        _row("op-1", 0, "a = 1\n", kind="primary", h="h1"),
        _row("op-1", 1, "a = 2\n", kind="repair", h="h2"),
        _row("op-1", 2, "a = 3\n", kind="retry", h="h3"),
        _row("op-1", 3, "a = 4\n", kind="sibling", h="h4"),
    ])
    kinds = [r["metadata"]["draw_kind"] for r in gp.iter_trajectory_rows(d)]
    assert kinds == ["primary", "sibling"]


def test_genuine_only_can_be_switched_off(tmp_path: Path) -> None:
    d = _corpus(tmp_path, [
        _row("op-1", 0, "a = 1\n", kind="primary", h="h1"),
        _row("op-1", 1, "a = 2\n", kind="repair", h="h2"),
    ])
    assert len(list(gp.iter_trajectory_rows(d, genuine_only=False))) == 2


def test_legacy_rows_without_a_draw_kind_are_genuine(tmp_path: Path) -> None:
    d = _corpus(tmp_path, [_row("op-1", 0, "a = 1\n"), _row("op-1", 1, "a = 2\n")])
    assert len(list(gp.iter_trajectory_rows(d))) == 2


def test_one_hash_per_op_first_seen_wins(tmp_path: Path) -> None:
    d = _corpus(tmp_path, [
        _row("op-1", 0, "a = 1\n", kind="primary", h="same"),
        _row("op-1", 1, "a = 1\n", kind="sibling", h="same"),
        _row("op-2", 0, "a = 1\n", kind="primary", h="same"),
    ])
    out = list(gp.iter_trajectory_rows(d))
    assert [(r["metadata"]["op_id"], r["metadata"]["attempt_index"]) for r in out] == [
        ("op-1", 0), ("op-2", 0)]


def test_rows_without_a_hash_are_never_collapsed(tmp_path: Path) -> None:
    d = _corpus(tmp_path, [_row("op-1", 0, "a = 1\n"), _row("op-1", 1, "a = 1\n")])
    assert len(list(gp.iter_trajectory_rows(d))) == 2


def test_a_torn_line_is_counted_not_silently_dropped(tmp_path, caplog) -> None:
    """Readers take no lock, so a harvest against a LIVE soak can catch the
    row being appended and see it torn. That is benign — the file is
    append-only — but an uncounted skip makes a damaged corpus
    indistinguishable from a healthy one.
    """
    import logging

    d = tmp_path / "events"
    d.mkdir()
    good = _row("op-a", 0, "def f(): return 1", kind="primary", h="h1")
    torn = json.dumps(_row("op-b", 0, "def g(): return 2", kind="primary",
                           h="h2"))[:120]
    (d / "experience.jsonl").write_text(
        json.dumps(good) + "\n" + torn + "\n", encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger=gp.logger.name):
        rows = list(gp.iter_trajectory_rows(d))

    assert len(rows) == 1, "the intact row must still be ingested"
    assert any("undecodable line" in r.getMessage() for r in caplog.records), \
        "the skip must be reported"


def test_a_clean_corpus_logs_no_warning(tmp_path, caplog) -> None:
    import logging

    d = tmp_path / "events"
    d.mkdir()
    (d / "experience.jsonl").write_text(
        json.dumps(_row("op-a", 0, "def f(): return 1", kind="primary", h="h1"))
        + "\n", encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger=gp.logger.name):
        assert len(list(gp.iter_trajectory_rows(d))) == 1
    assert not [r for r in caplog.records if "undecodable" in r.getMessage()]
