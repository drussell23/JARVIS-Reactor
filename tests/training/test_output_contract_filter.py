"""Train only on prompts whose output contract O+V still asks for.

2026-10-07: 86% of the corpus prompts asked for whole-file rewrites while O+V
had moved large files to unified diffs. GRPO re-generates from the prompt, so
those prompts drove every rollout into the completion window (clipped_ratio
1.0) and the policy loss had no tokens. O+V now stamps each row with the
contract its prompt imposed (``metadata.output_schema``); the one corpus
reader admits only the contracts its caller names, and an untagged row --
written before the tag existed, contract unknown -- never passes a filter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from reactor_core.training import grpo_pipeline as gp

_REPO = Path(__file__).resolve().parents[2]


def _row(op, body, schema=None, *, h=None):
    meta = {"op_id": op, "should_train": True, "draw_kind": "primary", "candidate_hash": h or f"h-{op}-{body}"}
    if schema is not None:
        meta["output_schema"] = schema
    return {"event_type": "interaction", "event_id": f"e-{op}-{body}", "user_input": f"## Task\nprompt for {op}",
            "assistant_output": body, "outcome": "success", "confidence": 1.0, "metadata": meta}


def _corpus(tmp_path, rows):
    d = tmp_path / "events"
    d.mkdir()
    (d / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


ROWS = [
    _row("op-diff", "a = 1\n", "2b.1-diff"),
    _row("op-small", "b = 1\n", "2b.1"),
    _row("op-multi", "c = 1\n", "2c.1"),
    _row("op-legacy", "d = 1\n"),             # predates the tag
    _row("op-blank", "e = 1\n", ""),          # recorded with no contract
]


def _ops(d, **kw):
    return [r["metadata"]["op_id"] for r in gp.iter_trajectory_rows(d, **kw)]


def test_only_the_named_contracts_are_admitted(tmp_path, monkeypatch):
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1-diff,2b.1")
    assert _ops(_corpus(tmp_path, ROWS)) == ["op-diff", "op-small"]


def test_an_untagged_row_never_passes_a_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1")
    ops = _ops(_corpus(tmp_path, ROWS))
    assert "op-legacy" not in ops and "op-blank" not in ops


def test_no_filter_when_the_caller_names_none(tmp_path, monkeypatch):
    """Reactor cannot know O+V's policy; without a caller's answer it reads
    everything, as before the tag existed."""
    monkeypatch.delenv("REACTOR_GRPO_OUTPUT_SCHEMAS", raising=False)
    assert len(_ops(_corpus(tmp_path, ROWS))) == len(ROWS)


def test_an_explicit_argument_overrides_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1")
    d = _corpus(tmp_path, ROWS)
    assert _ops(d, output_schemas={"2c.1"}) == ["op-multi"]
    assert len(_ops(d, output_schemas=None)) == len(ROWS)


def test_whitespace_and_empty_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", " 2b.1-diff , ,2b.1 ")
    assert gp.accepted_output_schemas() == frozenset({"2b.1-diff", "2b.1"})
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", " , ")
    assert gp.accepted_output_schemas() is None


def test_rows_are_excluded_not_deleted(tmp_path, monkeypatch):
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1-diff")
    d = _corpus(tmp_path, ROWS)
    before = (d / "a.jsonl").read_bytes()
    list(gp.iter_trajectory_rows(d))
    assert (d / "a.jsonl").read_bytes() == before


def test_the_dataset_reads_through_the_same_filter(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1-diff")
    ds = gp.build_prompt_dataset(_corpus(tmp_path, ROWS))
    assert ds["op_id"] == ["op-diff"]


def test_the_gate_counts_only_current_contracts_and_says_which(tmp_path, monkeypatch):
    """Preflight decides whether a cycle trains at all. It must count the
    rows the trainer will read, and its report must name the contracts."""
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1-diff")
    rows = [_row("op-g", "x = 1\n", "2b.1-diff", h="h1"), _row("op-g", "x = 2\n", "2b.1-diff", h="h2"),
            _row("op-l", "y = 1\n", h="h3"), _row("op-l", "y = 2\n", h="h4")]
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import grpo_preflight as pf
    finally:
        sys.path.pop(0)
    rep = pf.analyse(_corpus(tmp_path, rows), min_group=2, trainable_only=True)
    assert rep["rows"] == 2 and rep["prompts"] == 1
    assert rep["output_schemas"] == ["2b.1-diff"]


def test_tagged_rows_accumulate_across_session_files_without_duplicates(tmp_path, monkeypatch):
    """The corpus is append-only across sessions (one file per writer, many
    sessions per file). The reader has no session boundary: rows from an
    earlier, deferred session are read again next time, and a candidate
    re-recorded later is still one row."""
    monkeypatch.setenv("REACTOR_GRPO_OUTPUT_SCHEMAS", "2b.1-diff")
    d = tmp_path / "events"
    d.mkdir()
    s1 = [_row(f"op-s1-{i}", f"v = {i}\n", "2b.1-diff") for i in range(14)]
    s2 = [_row("op-s2-0", "w = 0\n", "2b.1-diff"), s1[3]]          # s1[3] recorded again
    (d / "session1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s1), encoding="utf-8")
    (d / "session2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in s2), encoding="utf-8")
    ops = _ops(d)
    assert len(ops) == 15 and len(set(ops)) == 15
