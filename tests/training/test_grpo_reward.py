"""GRPO rewards over O+V sibling groups, and the zero-variance trap.

GRPO normalises rewards WITHIN a group: ``Â_i = (r_i - mean(r)) / std(r)``.
When every sibling scores identically the standard deviation is 0 and the
advantage is 0/0 — the degenerate case, and the common one here, because
"all three candidates failed" is the modal outcome of a farming soak.

The tempting fix is to subtract a penalty from the group. **That is
arithmetically inert**: shifting every member by the same constant leaves
the variance exactly where it was. Only a term that DIFFERS between
siblings can produce a non-zero advantage.

So the tiebreaker is a real per-candidate measurement — structural
severity — and when even that ties, the group is DROPPED rather than
given a fabricated winner. Inventing a preference between
indistinguishable answers is noise with a gradient attached, and it is
the same failure the recorder already refuses when it labels an unseen
outcome ``unknown`` instead of guessing.

Loaded by path: ``reactor_core/__init__`` imports ``PreprocessingPipeline``
from ``reactor_core.data``, which contains only ``lineage.py`` — so the
package raises ImportError and nothing inside it is reachable normally.
"""

from __future__ import annotations

import asyncio
import importlib.util
import statistics
import sys
from pathlib import Path
from typing import List, Optional

import pytest

_SRC = Path(__file__).resolve().parents[2] / "reactor_core" / "training" / "grpo_reward.py"
_spec = importlib.util.spec_from_file_location("grpo_reward_under_test", _SRC)
grpo_reward = importlib.util.module_from_spec(_spec)
sys.modules["grpo_reward_under_test"] = grpo_reward
_spec.loader.exec_module(grpo_reward)

candidate_reward = grpo_reward.candidate_reward
structural_severity = grpo_reward.structural_severity


def _advantage(rewards: List[Optional[float]]):
    """GRPO's group-relative advantage, as TRL computes it."""
    vals = [r for r in rewards if r is not None]
    if len(vals) < 2:
        return None
    sd = statistics.pstdev(vals)
    if sd < 1e-12:
        return None  # degenerate: 0/0
    m = statistics.mean(vals)
    return [(v - m) / sd for v in vals]


def _run(**kw):
    n = len(kw["completions"])
    kw.setdefault("outcome", ["failure"] * n)
    kw.setdefault("confidence", [0.0] * n)
    kw.setdefault("latency_ms", [1000.0] * n)
    kw.setdefault("model_id", ["qwen3-coder:30b"] * n)
    kw.setdefault("task_type", ["code_repair"] * n)
    return asyncio.run(candidate_reward(**kw))


# --------------------------------------------------------------------------
# The severity ladder — the measurement that breaks ties
# --------------------------------------------------------------------------


def test_severity_is_a_ladder_not_a_boolean() -> None:
    """A binary pass/fail is what CREATED the flat group.

    Grading has to be ordered, or it cannot separate two candidates that
    the coarse label already called equal.
    """
    empty = structural_severity("")
    garbage = structural_severity("!!! not python")
    inert = structural_severity('"""only a docstring"""\n')
    stmts = structural_severity("x = 1\ny = 2\n")
    real = structural_severity("def page(items, n):\n    return items[:n]\n")

    assert empty.score < garbage.score < inert.score < stmts.score < real.score


def test_a_late_syntax_error_outranks_an_early_one() -> None:
    """Where the parse dies is real signal.

    A file that fails on its last line got almost everything right; one
    that fails on line 1 is not code. Both are "syntax_error" to a
    boolean check.
    """
    late = structural_severity("def f():\n    return 1\n" * 20 + "def broken(:\n")
    early = structural_severity("def broken(:\n" + "x = 1\n" * 40)
    assert late.score > early.score
    assert "syntax_error" in late.reason and "syntax_error" in early.reason


def test_no_syntax_error_can_outrank_a_parsing_candidate() -> None:
    """The bands must not overlap.

    A nearly-complete broken file is still broken; ranking it above
    working code would invert the thing being taught.
    """
    best_broken = structural_severity("x = 1\n" * 500 + "def broken(:\n")
    worst_parsing = structural_severity("pass\n")
    assert best_broken.score < worst_parsing.score


def test_a_parsing_no_op_does_not_score_as_working_code() -> None:
    """The Quine-class candidate: valid syntax, says nothing."""
    assert structural_severity('"""doc"""\n').score < structural_severity(
        "def f():\n    return 1\n"
    ).score


def test_severity_never_raises() -> None:
    for bad in ("", "\x00\x01", "def f(:\n" * 500, ""):
        s = structural_severity(bad)
        assert 0.0 <= s.score <= 1.0


# --------------------------------------------------------------------------
# The zero-variance interceptor
# --------------------------------------------------------------------------


def test_a_normal_group_passes_through_untouched() -> None:
    """When outcomes already differ, no tiebreak is needed or applied."""
    r = _run(
        completions=["def f(): return 1", "def g(): return 2", "broken(:"],
        outcome=["success", "failure", "failure"],
        confidence=[1.0, 0.0, 0.0],
    )
    assert all(x is not None for x in r)
    assert _advantage(r) is not None


def test_all_siblings_failed_but_differently_yields_a_real_advantage() -> None:
    """THE trap. Same coarse label, genuinely different quality.

    Without the tiebreak these three are identical to the scorer and the
    group is wasted. With it, the difference is recovered from a
    measurement rather than invented.
    """
    r = _run(completions=[
        "def page(items, n):\n    return items[:n]\n" * 3 + "def x(:\n",  # late break
        "def x(:\n" + "y = 1\n" * 40,                                     # early break
        '"""nothing at all"""\n',                                         # inert
    ])
    assert all(x is not None for x in r), "group was dropped despite a real difference"
    adv = _advantage(r)
    assert adv is not None, "still degenerate — the tiebreak did not separate them"
    assert max(adv) - min(adv) > 0.5


def test_identical_failures_are_dropped_not_fabricated() -> None:
    """No signal exists, so none is invented.

    Three byte-identical answers cannot have a best one. Returning a
    manufactured ordering would train the model on noise.
    """
    same = "def x(:\n"
    r = _run(completions=[same, same, same])
    assert r == [None, None, None]


def test_identical_successes_are_also_dropped() -> None:
    """Symmetry: a flat GOOD group is just as empty of preference."""
    ok = "def page(items, n):\n    return items[:n]\n"
    r = _run(completions=[ok, ok, ok], outcome=["success"] * 3, confidence=[1.0] * 3)
    assert r == [None, None, None]


def test_a_uniform_shift_would_not_have_worked() -> None:
    """Pins the arithmetic the design rests on.

    If the interceptor subtracted a constant instead of a per-candidate
    measurement, variance would stay zero and the group would still be
    degenerate. This asserts the property directly so nobody
    "simplifies" the tiebreak into a flat penalty later.
    """
    flat = [0.5, 0.5, 0.5]
    shifted = [v - 0.3 for v in flat]
    assert _advantage(shifted) is None, (
        "a uniform penalty leaves std=0 — only a per-candidate term helps"
    )


def test_tiebreak_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off -> flat groups are dropped without the structural pass."""
    monkeypatch.setenv("REACTOR_GRPO_STRUCTURAL_TIEBREAK", "false")
    r = _run(completions=["def x(:\n", '"""doc"""\n', "y = 1\n"])
    assert r == [None, None, None]


def test_empty_group_is_handled() -> None:
    assert asyncio.run(candidate_reward(completions=[])) == []


def test_single_completion_group_is_degenerate_by_definition() -> None:
    """n=1 cannot support a group-relative advantage at all.

    Worth pinning: it is exactly the state the generation lane was in
    before sibling drawing, and it is why GRPO needs n>=2.
    """
    r = _run(completions=["def f(): return 1"])
    assert _advantage(r) is None
