"""Two draws of one task must group together even when the prompt drifted.

## The measurement that forced this

On the 2026-09-05 corpus the gate reported 152 singletons out of 244
prompts, and the standing theory was that L2 `repair` draws were being
mislabeled and discarded. That theory is wrong: the repair exclusion costs
exactly ONE prompt. Attributing every singleton to the filter that made it
one gives 146 of 152 with only a single row ever recorded.

Of those 146, **100 had an admissible sibling on the same `op_id` under a
DIFFERENT prompt text**. The two prompts were usually the same LENGTH and
diverged after a common prefix of ~1,770 characters -- a substitution in
the middle, not an addition at the end. Across 45 such ops, `## Task` was
byte-identical in 45 of 45. What moved was ambient memory injected at
prompt-build time: `Recent Episodes` in 38 ops, `Rust Subsystems` in 23,
`What Happened` in 12. Draw 2 is composed after draw 1 completes, so the
episodic memory has advanced by one entry.

Grouping on exact bytes therefore split real sibling groups into
singletons, and the corpus reported a sibling-generation failure that had
not happened.

## What must NOT happen

Loosening a grouping key trades a false split for a false merge, which is
strictly worse: a false merge grades two answers to DIFFERENT questions
against each other and feeds the difference to the optimiser. So the key
keeps every task-bearing section, `Target` and `Structural Index` among
them -- two draws that saw different file content stay apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
import grpo_preflight as _pf  # noqa: E402

gp = _pf._load("grpo_pipeline")

TASK = (
    "## Task\n"
    "Op-ID: op-1\n"
    "Goal: re-raise HTTPException instead of swallowing it.\n"
    "Target file: `backend/api/routes.py`\n\n"
)
TARGET = "## Target: backend/api/routes.py [SHA-256: abc123]\n\ndef f():\n    pass\n\n"
SCHEMA = "## Output Schema\n\n{\"file_path\": str, \"full_content\": str}\n"


def _prompt(episodes: str) -> str:
    """A prompt in the real shape: ambient memory between task sections."""
    return (
        "You are generating code.\n\n"
        + TASK
        + "## Recent Episodes (your short-term memory)\n\n"
        + episodes
        + "\n"
        + TARGET
        + SCHEMA
    )


# ---------------------------------------------------------------------------
# The split this closes
# ---------------------------------------------------------------------------


def test_drifted_ambient_memory_no_longer_splits_a_group() -> None:
    a = _prompt("- op=1 applied\n- op=2 applied\n")
    b = _prompt("- op=1 applied\n- op=2 applied\n- op=3 applied\n")
    assert a != b, "the fixture must actually differ, or this proves nothing"
    assert gp.task_group_key(a) == gp.task_group_key(b)


def test_the_ambient_text_is_absent_from_the_key() -> None:
    key = gp.task_group_key(_prompt("- op=1 applied\n"))
    assert "Recent Episodes" not in key
    assert "op=1 applied" not in key
    assert "Goal: re-raise HTTPException" in key


# ---------------------------------------------------------------------------
# The false merges it must refuse
# ---------------------------------------------------------------------------


def test_a_different_task_keeps_a_different_key() -> None:
    a = _prompt("- x\n")
    b = a.replace("re-raise HTTPException instead of swallowing it",
                  "use datetime.now(timezone.utc)")
    assert gp.task_group_key(a) != gp.task_group_key(b)


def test_a_different_target_file_keeps_a_different_key() -> None:
    a = _prompt("- x\n")
    b = a.replace("backend/api/routes.py", "backend/api/clock.py")
    assert gp.task_group_key(a) != gp.task_group_key(b)


def test_a_changed_source_snapshot_keeps_a_different_key() -> None:
    """`Target` carries the file's SHA and body. Two draws that saw
    different code answered different questions; pairing them is the error
    the `repair` exclusion exists to prevent."""
    a = _prompt("- x\n")
    b = a.replace("def f():\n    pass", "def f():\n    return 1")
    assert gp.task_group_key(a) != gp.task_group_key(b)


def test_a_changed_output_schema_keeps_a_different_key() -> None:
    a = _prompt("- x\n")
    b = a.replace("\"full_content\": str", "\"patch\": str")
    assert gp.task_group_key(a) != gp.task_group_key(b)


# ---------------------------------------------------------------------------
# Edges — the key must never empty the corpus
# ---------------------------------------------------------------------------


def test_a_prompt_with_no_recognised_section_falls_back_to_itself() -> None:
    """An unrecognised shape must never collapse onto an unrelated one."""
    a = "just some prose with no headings at all"
    b = "different prose with no headings at all"
    assert gp.task_group_key(a) == a
    assert gp.task_group_key(a) != gp.task_group_key(b)


def test_only_ambient_sections_also_falls_back() -> None:
    a = "## Recent Episodes\n\n- one\n"
    b = "## Recent Episodes\n\n- two\n"
    assert gp.task_group_key(a) != gp.task_group_key(b), (
        "with nothing task-bearing to key on, two prompts must stay apart "
        "rather than collapse into one meaningless group")


@pytest.mark.parametrize("bad", ["", None, 0, [], {}])
def test_it_never_raises(bad) -> None:
    assert isinstance(gp.task_group_key(bad), str)


def test_it_is_deterministic() -> None:
    p = _prompt("- x\n")
    assert gp.task_group_key(p) == gp.task_group_key(p)


def test_the_escape_hatch_restores_byte_exact_grouping(monkeypatch) -> None:
    monkeypatch.setenv("REACTOR_GRPO_TASK_GROUPING", "0")
    a = _prompt("- op=1\n")
    b = _prompt("- op=1\n- op=2\n")
    assert gp.task_group_key(a) != gp.task_group_key(b)
    assert gp.task_group_key(a) == a.strip()


# ---------------------------------------------------------------------------
# The downstream contract
# ---------------------------------------------------------------------------


def test_the_gate_hands_back_a_FULL_prompt_not_the_key(tmp_path) -> None:
    """`build_prompt_dataset(only_prompts=...)` matches on the full text's
    `.strip()`. A gate that returned its own grouping key would select
    NOTHING and the trainer would silently train on an empty dataset."""
    import json
    rows = []
    for i, ep in enumerate(("- op=1\n", "- op=1\n- op=2\n")):
        rows.append({
            "event_type": "interaction",
            "user_input": _prompt(ep),
            "assistant_output": f"def f():\n    raise ValueError({i})\n",
            "metadata": {"op_id": "op-1", "candidate_hash": f"h{i}",
                         "draw_kind": "primary" if i == 0 else "sibling",
                         "should_train": True},
        })
    (tmp_path / "e.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    report = _pf.analyse(tmp_path, min_group=2, trainable_only=True)
    assert report["prompts"] == 1, "the two drifted prompts must form ONE group"
    assert report["groups_below_min"] == 0

    corpus = {str(r["user_input"]).strip() for r in rows}
    for p in report["trainable_prompts"]:
        assert p.strip() in corpus, (
            "the gate returned something that is not a prompt in the corpus")


def test_byte_identical_prompts_still_group(tmp_path) -> None:
    """The pre-existing case must be untouched."""
    import json
    p = _prompt("- op=1\n")
    rows = [{
        "event_type": "interaction", "user_input": p,
        "assistant_output": f"x = {i}\n",
        "metadata": {"op_id": "op-1", "candidate_hash": f"h{i}",
                     "draw_kind": "sibling", "should_train": True},
    } for i in range(2)]
    (tmp_path / "e.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = _pf.analyse(tmp_path, min_group=2, trainable_only=True)
    assert report["prompts"] == 1 and report["groups_below_min"] == 0
