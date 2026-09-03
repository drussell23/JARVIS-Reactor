"""Context-aware grading: the task's intent, and the code that DIFFERS.

Soaks 13-18 all failed `--min-spread 0.01` with a provably healthy
generator. Soak 18's sample was clean by every structural measure — draw
kinds separated, no duplicate hashes, 15 of 15 groups pairable — and its
best group still spread 0.00585, because a candidate is a whole-file
rewrite and ~90% of every sub-metric's input is shared across siblings.
Measured on that corpus, the one group whose siblings genuinely disagree
scores q = 0.523 / 0.526 / 0.509 over the whole file and 0.426 / 0.428 /
0.367 over the code they actually chose.

These tests pin the two properties that make that refinement honest
rather than a spread-manufacturing device:

  * it MEASURES — identical siblings stay identical and the group is
    still correctly dropped as flat;
  * it cannot REORDER the ladder — an authoritative verdict, a syntax
    failure and a declined answer all keep the score their band assigned.
"""
from __future__ import annotations

import ast

import pytest

from reactor_core.training import grpo_verifier as V


# --------------------------------------------------------------------------
# Fixtures shaped like the real corpus
# --------------------------------------------------------------------------

_SHARED = '''\
import logging

logger = logging.getLogger(__name__)


def helper(value: str) -> str:
    """Untouched by every sibling."""
    return value.strip()


def other(value: int) -> int:
    """Also untouched."""
    return value + 1
'''

# Two siblings differ ONLY in the one function the task is about. One
# guards with a plain depth check, the other builds a nested tangle.
_FLAT_FIX = _SHARED + '''

def clean(payload: dict, depth: int = 0) -> str:
    """Bounded, flat, annotated."""
    if depth > 5:
        return ""
    return str(payload)
'''

_NESTED_FIX = _SHARED + '''

def clean(payload, depth=0):
    if payload:
        if depth < 5:
            if isinstance(payload, dict):
                for k in payload:
                    if k:
                        return str(payload[k])
    return ""
'''


def _goal(text: str) -> str:
    """A prompt shaped like the composer's, boilerplate included."""
    return (
        "## Human Instructions\n\nFollow the house style.\n\n---\n\n"
        f"## Task\nOp-ID: op-test\nGoal: {text}\n\n"
        "## Strategic Direction (Manifesto v4)\n\n"
        # Boilerplate identical across every op in a soak, and stuffed
        # with words that would swamp a naive whole-prompt keyword count.
        + ("performance optimise cache latency throughput refactor "
           "simplify docstring document annotate typing pytest test file\n"
           * 40)
    )


# --------------------------------------------------------------------------
# Task intent — deterministic, and blind to the boilerplate
# --------------------------------------------------------------------------


def test_intent_reads_the_task_not_the_manifesto() -> None:
    """The 24 KB of identical boilerplate must not decide every task."""
    p = _goal("add a bounded recursion guard and handle the list edge case")
    assert V._task_intent(p) == "robustness"


def test_intent_follows_the_goal_when_the_goal_changes() -> None:
    assert V._task_intent(_goal("add docstrings to every public function")) == "docs"
    assert V._task_intent(_goal("refactor and simplify the branch tangle")) == "refactor"


def test_no_markers_means_no_intent_not_a_guess() -> None:
    assert V._task_intent(_goal("rename the module")) == ""
    assert V._task_intent("") == ""


def test_a_tie_declines_rather_than_picking_alphabetically() -> None:
    """Equal evidence for two axes is not evidence for either."""
    p = "## Task\nGoal: add a docstring and a type hint\n"
    hits = {
        n: sum(p.lower().count(m) for m in ms)
        for n, ms in V._INTENT_MARKERS.items()
    }
    assert hits["docs"] == hits["typing"] > 0, "fixture must actually tie"
    assert V._task_intent(p) == ""


def test_intent_weights_scale_the_operator_policy_not_replace_it() -> None:
    base = dict(V._Q_WEIGHTS)
    w, intent = V._intent_weights(_goal("simplify and refactor this tangle"))
    assert intent == "refactor"
    assert w["simplicity"] == pytest.approx(base["simplicity"] * 1.8)
    assert w["docs"] == pytest.approx(base["docs"]), "unemphasised axes untouched"


def test_intent_weights_are_the_base_when_there_is_no_intent() -> None:
    w, intent = V._intent_weights("nothing to see here")
    assert intent == "" and w == dict(V._Q_WEIGHTS)


# --------------------------------------------------------------------------
# The differential footprint
# --------------------------------------------------------------------------


def test_footprint_is_only_the_code_the_siblings_chose_differently() -> None:
    keys_a = {k for k, _ in V._statement_keys(_FLAT_FIX)}
    keys_b = {k for k, _ in V._statement_keys(_NESTED_FIX)}
    common = keys_a & keys_b
    diff = [n for k, n in V._statement_keys(_FLAT_FIX) if k not in common]
    names = [getattr(n, "name", None) for n in diff]
    assert names == ["clean"], f"shared helpers leaked into the footprint: {names}"


def test_footprint_separates_what_the_whole_file_dilutes() -> None:
    whole = [V._grade_source(s).q for s in (_FLAT_FIX, _NESTED_FIX)]
    fp = V._footprint_q([_FLAT_FIX, _NESTED_FIX])
    assert all(x is not None for x in fp)
    assert abs(fp[0] - fp[1]) > abs(whole[0] - whole[1]), (
        "the footprint must not report LESS separation than the whole file "
        f"when the difference is real: whole={whole} fp={fp}"
    )


def test_identical_siblings_have_no_footprint() -> None:
    """None, not zero: unmeasurable is not the same as worthless."""
    assert V._footprint_q([_FLAT_FIX, _FLAT_FIX]) == [None, None]


def test_a_docstring_only_difference_is_not_a_choice() -> None:
    reworded = _FLAT_FIX.replace("Bounded, flat, annotated.", "Bounds the depth.")
    assert V._footprint_q([_FLAT_FIX, reworded]) == [None, None]


def test_a_footprint_that_defines_nothing_declines() -> None:
    a = "import os\nX = 1\n"
    b = "import os\nX = 2\n"
    assert V._footprint_q([a, b]) == [None, None]


def test_footprint_needs_someone_to_differ_from() -> None:
    assert V._footprint_q([_FLAT_FIX]) == [None]
    assert V._footprint_q([]) == []


@pytest.mark.parametrize("junk", ["", "def (", "\x00\x01", "class", "@" * 500])
def test_footprint_never_raises(junk: str) -> None:
    assert V._footprint_q([junk, _FLAT_FIX]) == [None, None]


def test_grade_footprint_survives_a_module_that_will_not_unparse() -> None:
    broken = ast.FunctionDef(name="x", args=None, body=[], decorator_list=[])
    assert V._grade_footprint([broken]) is None


# --------------------------------------------------------------------------
# The group seam — refinement that cannot reorder the ladder
# --------------------------------------------------------------------------


def _group(texts, prompt=""):
    return V.verify_group(texts, prompt=prompt)


def test_a_real_disagreement_separates_further_than_whole_file_grading() -> None:
    texts = [_FLAT_FIX, _NESTED_FIX]
    before = [V.verify_any(t).score for t in texts]
    after = [v.score for v in _group(texts, _goal("add a bounded guard"))]
    assert abs(after[0] - after[1]) > abs(before[0] - before[1])


def test_identical_candidates_still_score_identically() -> None:
    """The doctrine: a flat group must stay flat and be dropped, not
    rescued with a fabricated difference."""
    scores = [v.score for v in _group([_FLAT_FIX, _FLAT_FIX, _FLAT_FIX])]
    assert scores[0] == scores[1] == scores[2]


def test_refinement_stays_inside_the_passing_band() -> None:
    w = V.TierWeights()
    for v in _group([_FLAT_FIX, _NESTED_FIX], _goal("simplify")):
        assert w.passing_floor <= v.score <= w.substance


def test_an_authoritative_verdict_is_evidence_and_is_left_alone() -> None:
    """Tests beat style, and a refinement that moved this would invert
    the one contract the whole ladder is built on."""
    w = V.TierWeights()
    verdicts = [
        V.Verdict(w.authority, 4, "authority_accepted", "", authoritative=True),
        V.Verdict(w.syntax, 3, "authority_rejected", "", authoritative=True),
    ]
    out = V.refine_group([_FLAT_FIX, _NESTED_FIX], verdicts, prompt=_goal("guard"))
    assert [v.score for v in out] == [w.authority, w.syntax]


def test_a_syntax_failure_keeps_the_score_its_band_assigned() -> None:
    broken = "def clean(:\n    pass\n"
    out = _group([broken, _FLAT_FIX], _goal("guard"))
    assert V._reason_kind(out[0].reason) == "syntax_error"
    assert out[0].score == V.verify_any(broken).score


def test_a_declined_answer_is_never_refined_below_a_delivery() -> None:
    noop = '{"schema_version": "2b.1-noop", "reason": "already done"}'
    out = _group([noop, _FLAT_FIX, _NESTED_FIX], _goal("guard"))
    assert out[0].score == V.verify_any(noop).score


def test_one_passing_candidate_in_a_group_is_not_refined() -> None:
    """Nothing to differ from. Refining anyway would grade a footprint
    equal to the whole file and quietly change the score for no reason."""
    broken = "def clean(:\n    pass\n"
    out = _group([broken, _FLAT_FIX])
    assert out[1].score == V.verify_any(_FLAT_FIX).score


def test_a_singleton_group_is_returned_untouched() -> None:
    v = V.verify_any(_FLAT_FIX)
    assert [x.score for x in V.refine_group([_FLAT_FIX], [v])] == [v.score]


def test_mismatched_lengths_are_refused_not_zipped() -> None:
    v = V.verify_any(_FLAT_FIX)
    out = V.refine_group([_FLAT_FIX, _NESTED_FIX], [v])
    assert [x.score for x in out] == [v.score]


def test_the_master_switch_restores_whole_file_grading_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A soak's spread is only attributable if the previous behaviour is
    reproducible byte-for-byte."""
    texts = [_FLAT_FIX, _NESTED_FIX]
    monkeypatch.setattr(V, "_CONTEXT_METRICS", False)
    off = [v.score for v in _group(texts, _goal("guard"))]
    assert off == [V.verify_any(t).score for t in texts]


def test_the_reason_records_what_was_measured() -> None:
    out = _group([_FLAT_FIX, _NESTED_FIX], _goal("add a bounded recursion guard"))
    assert V._reason_kind(out[0].reason) == "quality", "reason KIND must be stable"
    assert "fp=" in out[0].reason and "intent=robustness" in out[0].reason


@pytest.mark.parametrize("junk", ["", "not json", "\x00", "[]", "{}"])
def test_the_group_seam_never_raises(junk: str) -> None:
    assert len(_group([junk, junk, _FLAT_FIX])) == 3


# --------------------------------------------------------------------------
# The prompt a group shares
# --------------------------------------------------------------------------


def test_group_prompt_reads_plain_and_chat_shaped_prompts() -> None:
    from reactor_core.training.grpo_reward import _group_prompt

    assert _group_prompt(["a goal", "a goal"]) == "a goal"
    chat = [[{"role": "user", "content": "fix the guard"}]]
    assert _group_prompt(chat) == "fix the guard"
    assert _group_prompt(None) == ""
    assert _group_prompt(["", "  ", "real"]) == "real"
