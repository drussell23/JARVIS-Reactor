"""The reward had the signal and was diluting it away.

## The measurement

Grouping by task (see `test_task_group_key`) leaves 114 groups of 2+.
Only 30 cleared the operative `--min-spread 0.01`. Attributing the rest:

  * 23 groups are FLAT at any resolution because their payloads are
    BYTE-IDENTICAL after extraction. Nothing can separate those and
    nothing should try -- they are the same answer.
  * 61 groups had real but tiny spread. Their sub-metric readings looked
    like this::

        docs=1.00 types=0.40 den=0.33 simp=0.43 con=0.10 flat=0.27
        docs=1.00 types=0.40 den=0.33 simp=0.44 con=0.10 flat=0.30
        docs=1.00 types=0.40 den=0.33 simp=0.48 con=0.10 flat=0.33

    Three of six axes are identical across every sibling and carry 2.6 of
    the 4.6 total weight. They cannot say which candidate is better, and
    they sit in the denominator anyway, shrinking the axes that can by the
    ratio of total to varying weight. Here 3.3x. The surviving q spread of
    0.021 then maps through the 0.35-wide passing band to a score spread
    of 0.0074, against a 0.01 floor.

So the reward was measuring the difference and then dividing it away.

## Two fixes, and what each was worth

  * `concision` divided by TOP-LEVEL statements while `defs` walked the
    whole tree. Characters-per-top-level-statement is ~780 for a real
    module against a target of 90, so `_soft` pinned the axis near its
    floor: 109 of 271 corpus sources at or below 0.11, median 0.123.
    Against all statements the same corpus reads median 0.495, nothing
    saturated. **Measured worth on this corpus: zero extra groups.** It is
    fixed because a metric whose docstring says "per statement" must
    divide by statements, not because it bought anything here.
  * `_discriminating_weights` re-aims the grade onto the axes the group is
    not unanimous on. **30 -> 47 trainable groups at 0.01.**

`density` keeps the top-level denominator. It asks whether the module is
organised into definitions, which is a question about the top level;
measured over all statements it collapses, with 239 of 271 sources under
0.11. The two denominators differ because the two questions differ.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import grpo_preflight as _pf  # noqa: E402

vf = _pf._load("grpo_verifier")


def _payload(src: str) -> str:
    """A candidate in the corpus's own shape: a fenced code block."""
    return "```python\n" + src + "\n```"


# A module with several axes deliberately held constant across variants,
# so a test can state exactly which axis is supposed to move.
_BASE = '''
import os


def alpha(a: int) -> int:
    """Doc."""
    if a > 0:
        return a
    return 0


def beta(b: int) -> int:
    """Doc."""
    return b
'''

_DEEPER = '''
import os


def alpha(a: int) -> int:
    """Doc."""
    if a > 0:
        if a > 1:
            if a > 2:
                return a
    return 0


def beta(b: int) -> int:
    """Doc."""
    return b
'''


# ---------------------------------------------------------------------------
# The aggregation: order-preserving, bounded, honest
# ---------------------------------------------------------------------------


def _parts(src: str):
    tree = ast.parse(src)
    stmts = list(tree.body)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return vf._quality_parts(tree, src, defs, stmts)


def test_a_constant_axis_is_dropped_and_a_varying_one_is_kept() -> None:
    a = {"docs": 1.0, "types": 0.4, "simplicity": 0.43}
    b = {"docs": 1.0, "types": 0.4, "simplicity": 0.48}
    base = {"docs": 1.0, "types": 1.0, "simplicity": 0.8}
    w, n = vf._discriminating_weights([a, b], base)
    assert n == 1
    assert w["simplicity"] == 0.8
    assert w["docs"] == 0.0 and w["types"] == 0.0


def test_narrowing_preserves_order_exactly() -> None:
    """The claim that makes this resolution rather than re-ranking."""
    base = dict(vf._Q_WEIGHTS)
    members = [
        {"docs": 1.0, "types": 0.4, "density": 0.33, "simplicity": s,
         "concision": 0.5, "flatness": f}
        for s, f in ((0.43, 0.27), (0.48, 0.33), (0.44, 0.30), (0.41, 0.25))
    ]
    w, _n = vf._discriminating_weights(members, base)
    before = [vf._aggregate_q(m, base) for m in members]
    after = [vf._aggregate_q(m, w) for m in members]
    assert (sorted(range(len(before)), key=lambda i: before[i])
            == sorted(range(len(after)), key=lambda i: after[i]))


def test_the_amplification_is_exactly_the_dilution_it_removes() -> None:
    base = dict(vf._Q_WEIGHTS)
    members = [
        {"docs": 1.0, "types": 0.4, "density": 0.33, "simplicity": s,
         "concision": 0.5, "flatness": f}
        for s, f in ((0.43, 0.27), (0.48, 0.33))
    ]
    w, _n = vf._discriminating_weights(members, base)
    before = [vf._aggregate_q(m, base) for m in members]
    after = [vf._aggregate_q(m, w) for m in members]
    expected = sum(base.values()) / sum(v for v in w.values() if v > 0)
    got = (max(after) - min(after)) / (max(before) - min(before))
    assert got == pytest.approx(expected, rel=1e-9)


def test_amplification_is_bounded_by_the_weight_policy() -> None:
    """No clamp: the bound falls out of the operator's own weights."""
    base = dict(vf._Q_WEIGHTS)
    bound = sum(base.values()) / min(v for v in base.values() if v > 0)
    for axis in base:
        members = [{k: (0.2 if k == axis else 0.5) for k in base},
                   {k: (0.9 if k == axis else 0.5) for k in base}]
        w, _n = vf._discriminating_weights(members, base)
        before = [vf._aggregate_q(m, base) for m in members]
        after = [vf._aggregate_q(m, w) for m in members]
        amp = (max(after) - min(after)) / (max(before) - min(before))
        assert amp <= bound + 1e-9


# ---------------------------------------------------------------------------
# The safety property: identical stays identical
# ---------------------------------------------------------------------------


def test_identical_members_are_never_separated() -> None:
    m = {"docs": 1.0, "types": 0.4, "density": 0.33,
         "simplicity": 0.5, "concision": 0.5, "flatness": 0.5}
    w, n = vf._discriminating_weights([dict(m), dict(m)], dict(vf._Q_WEIGHTS))
    assert n == 0
    assert w == dict(vf._Q_WEIGHTS), "nothing varies, so nothing is re-aimed"


def test_identical_candidates_still_score_identically() -> None:
    """End to end, through the real group path."""
    t = _payload(_BASE)
    scores = [float(v.score) for v in vf.verify_group([t, t, t], prompt="fix it")]
    assert max(scores) - min(scores) == 0.0


def test_a_difference_below_the_axis_epsilon_is_not_a_difference() -> None:
    base = {"docs": 1.0, "simplicity": 0.8}
    a = {"docs": 0.5, "simplicity": 0.5}
    b = {"docs": 0.5 + vf._AXIS_EPS / 10.0, "simplicity": 0.5}
    _w, n = vf._discriminating_weights([a, b], base)
    assert n == 0


# ---------------------------------------------------------------------------
# It never leaves the band, and never breaks training
# ---------------------------------------------------------------------------


def test_the_refined_score_stays_inside_the_passing_band() -> None:
    w = vf.TierWeights()
    vs = vf.verify_group([_payload(_BASE), _payload(_DEEPER)], prompt="guard it")
    for v in vs:
        assert w.passing_floor - 1e-9 <= float(v.score) <= w.substance + 1e-9


def test_a_real_difference_now_separates() -> None:
    """Two modules identical on every axis but nesting depth."""
    vs = vf.verify_group([_payload(_BASE), _payload(_DEEPER)], prompt="guard it")
    scores = [float(v.score) for v in vs]
    assert scores[0] > scores[1], "the flatter module must win"
    assert max(scores) - min(scores) > 0.01


def test_the_reason_says_how_many_axes_carried_the_grade() -> None:
    vs = vf.verify_group([_payload(_BASE), _payload(_DEEPER)], prompt="guard it")
    assert any("axes=" in str(v.reason) for v in vs)


def test_the_master_switch_restores_the_fixed_weight_mean(monkeypatch) -> None:
    base = dict(vf._Q_WEIGHTS)
    members = [{k: 0.5 for k in base}, {k: 0.5 for k in base}]
    members[1]["simplicity"] = 0.9
    monkeypatch.setattr(vf, "_DISCRIMINATING_WEIGHTS", False)
    w, n = vf._discriminating_weights(members, base)
    assert n == 0 and w == base


def test_an_ungradeable_member_disables_the_layer() -> None:
    """A None reading is not a zero. If any member cannot be measured the
    group has no common axis set, and the base weights stand."""
    m = {k: 0.5 for k in vf._Q_WEIGHTS}
    m2 = dict(m); m2["simplicity"] = 0.9
    w, n = vf._discriminating_weights([m, None, m2], dict(vf._Q_WEIGHTS))
    assert n == 0 and w == dict(vf._Q_WEIGHTS)


def test_a_varying_axis_with_zero_weight_is_not_reweighted_onto() -> None:
    """The operator said that axis does not matter. Honour it rather than
    making it the whole grade because it happens to be the one that moved."""
    base = {"docs": 1.0, "types": 1.0, "concision": 0.0}
    a = {"docs": 0.5, "types": 0.5, "concision": 0.1}
    b = {"docs": 0.5, "types": 0.5, "concision": 0.9}
    w, n = vf._discriminating_weights([a, b], base)
    assert n == 0 and w == base


@pytest.mark.parametrize("bad", ([], [None], [{}, {}], [{"docs": 1.0}]))
def test_it_never_raises(bad) -> None:
    w, n = vf._discriminating_weights(bad, dict(vf._Q_WEIGHTS))
    assert isinstance(w, dict) and isinstance(n, int)


# ---------------------------------------------------------------------------
# The concision denominator
# ---------------------------------------------------------------------------


def test_concision_is_measured_per_statement_not_per_top_level_statement() -> None:
    """The target is 90 characters. That is a LINE. Divided by the top
    level it read ~780 for a real module and pinned the axis at its
    floor."""
    src = _BASE
    p = _parts(src)
    tree = ast.parse(src)
    top = len(tree.body)
    allst = sum(1 for n in ast.walk(tree) if isinstance(n, ast.stmt))
    assert allst > top, "the fixture must have nested statements"
    assert p["concision"] == pytest.approx(
        vf._soft(len(src.strip()) / allst, vf._CONCISION_TARGET))


def test_density_keeps_the_top_level_denominator() -> None:
    """A different question, so a different denominator. Over all
    statements it collapses -- 239 of 271 corpus sources under 0.11."""
    src = _BASE
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    assert _parts(src)["density"] == pytest.approx(
        min(1.0, len(defs) / len(tree.body)))


def test_whitespace_is_still_presentation() -> None:
    """The fence contract: stripped and padded code score the SAME."""
    a = vf.verify_static(_payload(_BASE))
    b = vf.verify_static(_payload("\n\n" + _BASE + "\n\n   "))
    assert float(a.score) == float(b.score)


# ---------------------------------------------------------------------------
# The torch-free boundary
# ---------------------------------------------------------------------------


def test_the_grader_imports_nothing_heavy() -> None:
    """This module runs inside the orchestrator. An ML import here would
    put torch in the memory footprint of every soak."""
    import subprocess
    code = (
        "import sys, importlib.util, types;"
        "pkg = types.ModuleType('reactor_core'); pkg.__path__ = [r'%s'];"
        "sub = types.ModuleType('reactor_core.training'); sub.__path__ = [r'%s'];"
        "sys.modules['reactor_core'] = pkg; sys.modules['reactor_core.training'] = sub;"
        "import reactor_core.training.grpo_verifier as v;"
        "bad = [m for m in ('torch','peft','trl','transformers') if m in sys.modules];"
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    ) % (_REPO / "reactor_core", _REPO / "reactor_core" / "training")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=120)
    assert "CLEAN" in out.stdout, out.stdout + out.stderr
