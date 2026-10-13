"""A diff answer is graded by where it lands, not parsed as Python.

Measured 2026-10-10 before this: a correct 2b.1-diff and one whose context
exists nowhere in the file both scored 0.25 (`syntax_error:line1`), and any
whole-file answer 0.79 -- the reward could not see placement and pushed
away from the contract O+V enforces. The applier is O+V's own
(`governance/diff_apply.py`, loaded by path), so these run against the real
file from the sibling jarvis checkout.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from reactor_core.training import grpo_verifier as gv

_APPLIER = Path(os.environ.get(
    "REACTOR_GRPO_DIFF_APPLIER",
    Path(__file__).resolve().parents[3] / "jarvis" / "backend" / "core" / "ouroboros"
    / "governance" / "diff_apply.py"))

pytestmark = pytest.mark.skipif(not _APPLIER.is_file(),
                                reason=f"jarvis diff_apply.py not found at {_APPLIER}")

BAR = "│"
SRC = ('"""Helpers."""\nimport os\n\n\ndef f(x: int) -> int:\n    """Add one."""\n'
       "    return x + 1\n\n\ndef g() -> str:\n    \"\"\"The separator.\"\"\"\n    return os.sep\n")


def _prompt(body: str = SRC) -> str:
    numbered = "\n".join(f"{i:>2}{BAR}{t}" for i, t in enumerate(body.split("\n")[:-1], 1)) + "\n"
    return (f"## Task\nMake f add two.\n\n## File: m.py [SHA-256: abc123def456] "
            f"[{len(body)} bytes, {body.count(chr(10))} lines]\n```\n{numbered}\n```\n")


def _env(diff: str) -> str:
    return json.dumps({"schema_version": "2b.1-diff", "candidates": [
        {"candidate_id": "c1", "file_path": "m.py", "unified_diff": diff, "rationale": "r"}]})


GOOD = '@@ -5,3 +5,3 @@\n def f(x: int) -> int:\n     """Add one."""\n-    return x + 1\n+    return x + 2\n'
WRONG_HEADER = GOOD.replace("@@ -5,3 +5,3 @@", "@@ -40,3 +40,3 @@")
HALF = '@@ -5,3 +5,3 @@\n def f(x: int) -> int:\n     """Add one to x."""\n-    return x + 1\n+    return x + 2\n'
INVENTED = "@@ -5,3 +5,3 @@\n def f(a, b):\n     # sum\n-    return a + b\n+    return a + b + 2\n"
BREAKS = '@@ -5,3 +5,3 @@\n def f(x: int) -> int:\n     """Add one."""\n-    return x + 1\n+    return (x +\n'


@pytest.fixture(autouse=True)
def _applier(monkeypatch):
    monkeypatch.setenv("REACTOR_GRPO_DIFF_APPLIER", str(_APPLIER))
    gv._APPLIER.clear()
    yield
    gv._APPLIER.clear()


def _score(diff: str, prompt: str = None) -> gv.Verdict:
    return gv.verify_static(_env(diff), prompt=_prompt() if prompt is None else prompt)


def test_a_landing_diff_is_graded_as_the_module_it_produces() -> None:
    v = _score(GOOD)
    assert "diff_applied" in v.reason and v.reason.startswith("quality:")
    assert v.score >= gv.TierWeights().passing_floor


def test_the_ladder_a_diff_climbs() -> None:
    """lands > lands-with-a-wrong-header > lands-and-breaks > half-right > invented."""
    good, hdr, brk, half, inv = (_score(d).score for d in (GOOD, WRONG_HEADER, BREAKS, HALF, INVENTED))
    assert good > hdr > brk > half > inv, (good, hdr, brk, half, inv)
    assert "diff_misplaced:0/1" in _score(INVENTED).reason


def test_copied_line_numbers_cost_a_little_and_still_land() -> None:
    echoed = (f"@@ -1,3 +1,3 @@\n  5{BAR}def f(x: int) -> int:\n  6{BAR}    \"\"\"Add one.\"\"\"\n"
              f"- 7{BAR}    return x + 1\n+    return x + 2\n")
    v = _score(echoed)
    assert "diff_applied" in v.reason and "echo=" in v.reason
    assert v.score < _score(GOOD).score


def test_nothing_to_check_against_is_one_neutral_grade(monkeypatch) -> None:
    a = _score(GOOD, prompt="## Task\nno source shown\n")
    b = _score(INVENTED, prompt="## Task\nno source shown\n")
    assert a.score == b.score and a.reason == "diff_unverifiable:no_source"
    monkeypatch.delenv("REACTOR_GRPO_DIFF_APPLIER")
    gv._APPLIER.clear()
    assert _score(GOOD).reason == "diff_unverifiable:no_applier"


def test_the_reward_hands_the_prompt_to_the_grade() -> None:
    from reactor_core.training.grpo_reward import candidate_reward
    rewards = asyncio.run(candidate_reward([_env(GOOD), _env(INVENTED), _env(HALF)],
                                           prompts=[_prompt()] * 3))
    assert rewards[0] > rewards[2] > rewards[1], rewards


def test_the_group_footprint_reads_the_patched_modules() -> None:
    other = GOOD.replace("x + 2", "x + 1 + 1")
    out = gv.verify_group([_env(GOOD), _env(other)], prompt=_prompt())
    assert all("fp=" in v.reason for v in out), [v.reason for v in out]


# --- Phase B: the hallucinated-context penalty matrix -------------------------
#
# Each pair below fails to apply at the same place; they differ only in WHAT
# is wrong with the context, which is what the matrix must rank.

# A real line from g(), cited inside f's hunk: displaced, not invented.
DISPLACED = ('@@ -5,3 +5,3 @@\n def f(x: int) -> int:\n     return os.sep\n'
             '-    return x + 1\n+    return x + 2\n')
# The same slot holding a line that exists nowhere: fabricated.
FABRICATED = ('@@ -5,3 +5,3 @@\n def f(x: int) -> int:\n     y = compute(x)\n'
              '-    return x + 1\n+    return x + 2\n')
# Fabricated as a REMOVAL instead of context: deleting code that is not there.
FAB_REMOVAL = ('@@ -5,3 +5,3 @@\n def f(x: int) -> int:\n     """Add one."""\n'
               '-    y = compute(x)\n+    return x + 2\n')


def _rep(**classes):
    return [{"classes": classes, "checkable": True, "placed": False}]


def test_a_displaced_line_outranks_an_invented_one() -> None:
    d, f = _score(DISPLACED), _score(FABRICATED)
    assert d.score > f.score, (d.reason, f.reason)
    assert "fab=0/" in d.reason and "fab=1/" in f.reason


def test_deleting_a_phantom_line_costs_more_than_citing_one() -> None:
    ctx = gv._context_fidelity(_rep(context={"exact": 2, "displaced": 0, "fabricated": 1},
                                    removal={"exact": 1, "displaced": 0, "fabricated": 0}))
    rem = gv._context_fidelity(_rep(context={"exact": 3, "displaced": 0, "fabricated": 0},
                                    removal={"exact": 0, "displaced": 0, "fabricated": 1}))
    assert ctx[0] > rem[0]


def test_the_first_invented_line_costs_more_than_its_share() -> None:
    """Concave: one fabricated line in twenty must cost clearly more than the
    5% a linear penalty would charge."""
    clean = gv._context_fidelity(_rep(context={"exact": 20, "displaced": 0, "fabricated": 0},
                                      removal={"exact": 0, "displaced": 0, "fabricated": 0}))[0]
    one = gv._context_fidelity(_rep(context={"exact": 19, "displaced": 0, "fabricated": 1},
                                    removal={"exact": 0, "displaced": 0, "fabricated": 0}))[0]
    assert clean == 1.0
    assert clean - one > 0.10, one


def test_landing_still_outranks_every_non_landing_diff() -> None:
    worst_landing = _score(BREAKS).score
    for d in (DISPLACED, FABRICATED, FAB_REMOVAL, HALF, INVENTED):
        assert _score(d).score < worst_landing, d


def test_an_applier_without_classes_keeps_the_align_grade() -> None:
    assert gv._context_fidelity([{"present": 1.0, "in_order": 1.0}]) is None
