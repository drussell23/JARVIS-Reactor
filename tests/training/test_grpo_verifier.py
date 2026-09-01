"""In-loop verification grades the CODE, not the envelope it arrives in.

`grpo_reward.structural_severity` ran `ast.parse` on the raw completion.
A completion is not Python — it is the O+V response envelope, a JSON
object carrying the candidate file inside a `full_content` string. And a
JSON object is ALSO a valid Python dict literal, so the parse succeeded
on every well-formed envelope no matter what was inside it. Measured
before the fix:

    envelope carrying BROKEN python : Severity(0.600, 'parses:1stmt_no_defs')
    envelope carrying GOOD   python : Severity(0.600, 'parses:1stmt_no_defs')

Identical. The grader answered "is this JSON" while claiming to answer
"is this code correct", which is why every GRPO group came out flat: the
reward could not see the only thing that distinguishes siblings.

These tests pin the ladder's ORDERING, because ordering is the contract.
A reward that ranks a broken patch above a correct decline teaches the
model to emit garbage rather than say "already done" — an inversion that
would be invisible in aggregate loss and expensive in behaviour.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "reactor_core" / "training" / "grpo_verifier.py"
_spec = importlib.util.spec_from_file_location("grpo_verifier_under_test", _SRC)
gv = importlib.util.module_from_spec(_spec)
sys.modules["grpo_verifier_under_test"] = gv
_spec.loader.exec_module(gv)


GOOD = "def page(items, n):\n    return items[:n]\n"
BROKEN_EARLY = "def broken(:\n" + "x = 1\n" * 40
BROKEN_LATE = "def a():\n    return 1\n" * 20 + "def broken(:\n"
INERT = '"""just a docstring"""\n'


def env(code: str, ver: str = gv.SCHEMA_FULL, key: str = "full_content") -> str:
    return json.dumps({
        "schema_version": ver,
        "candidates": [{"candidate_id": "c1", "file_path": "m.py",
                        "rationale": "r", key: code}],
    })


# --------------------------------------------------------------------------
# The defect
# --------------------------------------------------------------------------


def test_envelope_carrying_broken_code_scores_below_one_carrying_good_code() -> None:
    """THE regression. These were identical (0.600) before the fix."""
    good = gv.verify_static(env(GOOD))
    broken = gv.verify_static(env(BROKEN_EARLY))
    assert good.score > broken.score
    assert good.tier > broken.tier


def test_extraction_reaches_through_the_envelope() -> None:
    ver, sources, reason = gv.extract_sources(env(GOOD))
    assert ver == gv.SCHEMA_FULL
    assert sources == [GOOD]
    assert reason == ""


# --------------------------------------------------------------------------
# Reward RANGE — passing siblings must be separable
# --------------------------------------------------------------------------


BARE = 'def page(items, n):\n    return items[:n]\n'
RICH = 'def page(items: list, n: int) -> list:\n    """Return the first n items."""\n    return items[:n]\n'
BRANCHY = 'def page(items, n):\n    out = []\n    for i, x in enumerate(items):\n        if i < n:\n            if x is not None:\n                out.append(x)\n    return out\n'


def test_passing_siblings_do_not_all_score_the_same() -> None:
    """THE regression. This spread was exactly 0.0 before.

    The old score was `min(1.0, 0.7 + 0.06 * min(defs, 5))` -- five
    reachable values, pinned above five defs. Siblings with the same def
    count scored IDENTICALLY however different their code, so a GRPO group
    of PASSING candidates was flat and the equal-reward guard dropped it.
    Measured on the live corpus: 19 multi-response groups, 19 flat, 0
    trainable. The reward ranked failure finely and success not at all.
    """
    scores = [gv.verify_static(env(c)).score for c in (BARE, RICH, BRANCHY)]
    assert len(set(round(s, 6) for s in scores)) == 3
    assert max(scores) - min(scores) > 0.01


def test_documented_and_typed_outranks_bare_which_outranks_branchy() -> None:
    """The ORDER is the claim, not merely that they differ."""
    rich = gv.verify_static(env(RICH)).score
    bare = gv.verify_static(env(BARE)).score
    branchy = gv.verify_static(env(BRANCHY)).score
    assert rich > bare > branchy


def test_identical_code_still_scores_identically() -> None:
    """Variance must be MEASURED, never manufactured.

    Two byte-identical candidates are identical, and dropping that group
    is the correct answer -- the same refusal the recorder makes when it
    labels an unseen outcome `unknown` rather than guessing.
    """
    a = gv.verify_static(env(RICH)).score
    b = gv.verify_static(env(RICH)).score
    assert a == b


def test_surrounding_whitespace_cannot_move_the_score() -> None:
    """Presentation is not quality.

    `concision` first divided by `len(src)`, which counts the whitespace
    around the code -- so a trailing newline moved the reward and the
    fence-stripped path scored differently from the identical clean one.
    That is the defect the fence stripper exists to remove, reintroduced
    one metric deeper. Caught by the fence tests, fixed at the metric.
    """
    base = gv.verify_static(env(RICH)).score
    nl = chr(10)
    for variant in (RICH + nl + nl, nl + RICH, "   " + nl + RICH + "  " + nl):
        assert gv.verify_static(env(variant)).score == base


def test_padding_a_file_does_not_buy_reward() -> None:
    """`concision` decays past a target, so bloat cannot be farmed.

    Same behaviour, same definitions, more characters -- must not score
    higher. This is the length bias `loss_type="dr_grpo"` was chosen to
    remove, and it must not re-enter through the reward instead.
    """
    padded = RICH.replace(
        '"""Return the first n items."""',
        '"""Return the first n items. ' + ("padding words " * 40) + '"""',
    )
    assert gv.verify_static(env(padded)).score < gv.verify_static(env(RICH)).score


def test_passing_code_never_outranked_by_a_lower_tier() -> None:
    """Widening the passing band must not break the non-overlap contract."""
    noop = gv.verify_static(json.dumps({
        "schema_version": gv.SCHEMA_NOOP, "reason": "already complete"}))
    broken = gv.verify_static(env(BROKEN_EARLY))
    for c in (BARE, RICH, BRANCHY):
        assert gv.verify_static(env(c)).score > noop.score > broken.score


def test_weights_are_policy_not_constants(monkeypatch) -> None:
    """Emphasis is env-tunable: a docs batch and a refactor batch differ."""
    import importlib
    monkeypatch.setenv("REACTOR_GRPO_Q_W_DOCS", "0")
    spec = importlib.util.spec_from_file_location("gv_reweighted", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gv_reweighted"] = mod
    spec.loader.exec_module(mod)
    # with docs weighted to zero, the documented variant loses its edge
    assert mod._Q_WEIGHTS["docs"] == 0.0

# --------------------------------------------------------------------------
# Reason KINDS dispatch exactly — ordering is not the contract
# --------------------------------------------------------------------------


def test_reason_kind_is_the_text_before_the_first_colon() -> None:
    assert gv._reason_kind("syntax_error:line3/9") == "syntax_error"
    assert gv._reason_kind("no_source_by_shape") == "no_source_by_shape"
    assert gv._reason_kind("no_content:full_content") == "no_content"


def test_a_declined_noop_outranks_broken_code() -> None:
    """THE inversion this ladder exists to prevent.

    A noop is a WELL-FORMED answer that happens to contain no code. If it
    scores at or below broken Python, the reward is telling the model that
    emitting garbage is no worse than correctly saying "already done" --
    and GRPO will act on exactly that.
    """
    noop = gv.verify_static(json.dumps({
        "schema_version": gv.SCHEMA_NOOP,
        "reason": "target file is already present and complete",
    }))
    assert gv._reason_kind(noop.reason) == "no_source_by_shape"
    for broken in (BROKEN_EARLY, BROKEN_LATE):
        assert noop.score > gv.verify_static(env(broken)).score
    # ...but delivering still beats declining.
    assert gv.verify_static(env(GOOD)).score > noop.score

def test_a_new_no_prefixed_reason_cannot_swallow_the_noop() -> None:
    """The regression that the old prefix test allowed.

    `no_source_by_shape` and `no_content:...` both begin `no_`. Under the
    old `reason.startswith("no_")` branch the ONLY thing keeping them
    apart was that the specific check sat above the generic one -- so
    reordering two `if`s, or adding any further `no_*` reason, silently
    scored a correct decline beneath broken code. Kinds are exact, so
    membership is the contract and order is not.
    """
    assert "no_content" in gv._SHAPE_FAULTS
    assert "no_source_by_shape" not in gv._SHAPE_FAULTS
    # a hypothetical future reason sharing the prefix stays out of both
    assert gv._reason_kind("no_candidates_at_all:x") == "no_candidates_at_all"
    assert "no_candidates_at_all" not in gv._SHAPE_FAULTS


def test_missing_content_is_a_shape_fault_not_a_syntax_fault() -> None:
    payload = json.dumps({
        "schema_version": gv.SCHEMA_FULL,
        "candidates": [{"candidate_id": "c1", "file_path": "m.py",
                        "rationale": "r"}],
    })
    ver, sources, reason = gv.extract_sources(payload)
    assert sources == []
    assert gv._reason_kind(reason) == "no_content"
    assert gv.verify_static(payload).tier == 1

# --------------------------------------------------------------------------
# The fence: a presentation fault must not be scored as a code fault
# --------------------------------------------------------------------------


FENCE = '```'


def test_fenced_good_code_scores_exactly_like_unfenced_good_code() -> None:
    """The second defect of the same shape as grading the envelope.

    A model that wraps `full_content` in a markdown fence is sending
    CORRECT Python. Before the fix the backticks reached `ast.parse` and
    it scored `syntax_error:line1` == 0.250 -- the SAME number genuinely
    broken code gets. The reward could not tell a correct patch from a
    broken one, so it graded FORMATTING.
    """
    clean = gv.verify_static(env(GOOD))
    fenced = gv.verify_static(env(FENCE + "python\n" + GOOD + FENCE))
    assert fenced.score == clean.score
    assert fenced.tier == clean.tier
    assert fenced.reason == clean.reason


def test_a_fence_does_not_rescue_broken_code() -> None:
    """Stripping the fence must REVEAL the real fault, never hide it."""
    fenced_broken = gv.verify_static(
        env(FENCE + "python\n" + BROKEN_EARLY + FENCE))
    assert fenced_broken.score < gv.verify_static(env(GOOD)).score


def test_bare_fence_without_a_language_tag_is_stripped() -> None:
    fenced = gv.verify_static(env(FENCE + "\n" + GOOD + FENCE))
    assert fenced.score == gv.verify_static(env(GOOD)).score


def test_unfenced_source_is_returned_byte_identical() -> None:
    """The guard on the fix itself.

    An unconditional `.strip()` here silently ate a trailing newline and
    broke extraction's round-trip (caught by the suite, not by review).
    Only a genuinely fenced value may be rewritten; every other value is
    the model's bytes, untouched.
    """
    for src in (GOOD, INERT, BROKEN_LATE, "   \n\nx = 1\n\n   "):
        assert gv.extract_sources(env(src))[1] == [src]


def test_backticks_inside_a_docstring_are_not_a_fence() -> None:
    """Only a value that BEGINS with a fence is touched."""
    src = ("def f():\n"
           '    """see ' + FENCE + 'code' + FENCE + ' here."""\n'
           "    return 1\n")
    assert gv.extract_sources(env(src))[1] == [src]

# --------------------------------------------------------------------------
# Ordering — the contract
# --------------------------------------------------------------------------


def test_the_ladder_is_monotonic() -> None:
    """Every rung must be strictly better than the one below it."""
    s = lambda t: gv.verify_static(t).score  # noqa: E731
    not_json = s("I think the answer is...")
    unknown = s(json.dumps({"schema_version": "9z", "x": 1}))
    malformed = s(json.dumps({"schema_version": gv.SCHEMA_FULL}))
    broken = s(env(BROKEN_EARLY))
    late = s(env(BROKEN_LATE))
    inert = s(env(INERT))
    good = s(env(GOOD))

    assert not_json < unknown < malformed < broken < late < inert < good


def test_a_correct_decline_outranks_broken_code() -> None:
    """Declining beats breaking.

    `no_source_by_shape` and `no_full_content` both start with `no_`, so
    a generic prefix check swallowed the noop case and scored a
    well-formed decline BENEATH a syntax error — teaching the model that
    emitting garbage is preferable to saying "already done". That is the
    exact inversion the reward exists to prevent, and it is the noop-spam
    qwen3-coder:30b produced 209 times in one soak, rewarded.
    """
    noop = gv.verify_static(json.dumps({"schema_version": gv.SCHEMA_NOOP, "reason": "done"}))
    tool = gv.verify_static(json.dumps({
        "schema_version": gv.SCHEMA_TOOL,
        "tool_calls": [{"name": "read_file", "arguments": {"path": "m.py"}}],
    }))
    broken = gv.verify_static(env(BROKEN_EARLY))
    malformed = gv.verify_static(json.dumps({"schema_version": gv.SCHEMA_FULL}))

    assert noop.score > broken.score
    assert tool.score > broken.score
    assert noop.score > malformed.score


def test_delivering_beats_declining() -> None:
    """...and the other half of the ordering.

    If a decline outranked working code, the model would learn to never
    attempt anything.
    """
    noop = gv.verify_static(json.dumps({"schema_version": gv.SCHEMA_NOOP, "reason": "done"}))
    assert gv.verify_static(env(GOOD)).score > noop.score
    assert gv.verify_static(env(INERT)).score > noop.score


def test_a_late_break_outranks_an_early_one() -> None:
    """How far the parse got is real signal a boolean discards."""
    assert gv.verify_static(env(BROKEN_LATE)).score > gv.verify_static(env(BROKEN_EARLY)).score


# --------------------------------------------------------------------------
# Shapes and resilience
# --------------------------------------------------------------------------


def test_fenced_json_is_not_punished_as_unparseable() -> None:
    """A markdown fence is a presentation fault, not a code fault."""
    fenced = "```json\n" + env(GOOD) + "\n```"
    assert gv.verify_static(fenced).score == gv.verify_static(env(GOOD)).score


def test_diff_shape_is_understood() -> None:
    """2b.1-diff carries `unified_diff`, not `full_content`."""
    ver, sources, reason = gv.extract_sources(
        env("--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n", gv.SCHEMA_DIFF, "unified_diff")
    )
    assert ver == gv.SCHEMA_DIFF and sources and reason == ""


def test_multi_file_candidate_is_graded_on_its_WORST_file() -> None:
    """APPLY is all-or-nothing, so one broken file spoils the candidate."""
    payload = json.dumps({
        "schema_version": gv.SCHEMA_FULL,
        "candidates": [{
            "candidate_id": "c1", "file_path": "a.py", "rationale": "r",
            "full_content": GOOD,
            "files": [{"file_path": "b.py", "full_content": BROKEN_EARLY}],
        }],
    })
    v = gv.verify_static(payload)
    assert v.score < gv.verify_static(env(GOOD)).score
    assert "syntax_error" in v.reason


@pytest.mark.parametrize("junk", [
    "", "   ", "null", "[]", "{}", '{"schema_version": null}',
    "```", "```json\n```", '{"schema_version":"2b.1","candidates":"nope"}',
    '{"schema_version":"2b.1","candidates":[{"full_content":""}]}',
    '{"schema_version":"2b.1","candidates":[null]}',
])
def test_malformed_input_never_raises(junk: str) -> None:
    v = gv.verify_static(junk)
    assert 0.0 <= v.score <= 1.0


# --------------------------------------------------------------------------
# Tier 4 — across the venv boundary
# --------------------------------------------------------------------------


def test_no_configured_verifier_means_NO_ANSWER_not_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent authority must not punish every completion equally.

    "the validator says this is broken" and "no validator ran" collapsing
    into one reward would make an unconfigured tier look like a model that
    always writes bad code.
    """
    monkeypatch.delenv("REACTOR_GRPO_VERIFY_CMD", raising=False)
    assert asyncio.run(gv.verify_authoritative(GOOD)) is None


def test_authority_accepts_and_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_GRPO_VERIFY_CMD", "true")
    gv._BUDGET = gv.AdaptiveBudget()
    assert asyncio.run(gv.verify_authoritative(GOOD)) is True

    monkeypatch.setenv("REACTOR_GRPO_VERIFY_CMD", "false")
    gv._BUDGET = gv.AdaptiveBudget()
    assert asyncio.run(gv.verify_authoritative(GOOD)) is False


def test_authority_rejection_does_not_sink_below_parsing_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected-but-parsing candidate must still outrank one that does not parse."""
    monkeypatch.setenv("REACTOR_GRPO_VERIFY_CMD", "false")
    gv._BUDGET = gv.AdaptiveBudget()
    rejected = asyncio.run(gv.verify(env(GOOD)))
    broken = gv.verify_static(env(BROKEN_EARLY))
    assert rejected.reason == "authority_rejected"
    assert rejected.score > broken.score


def test_authority_acceptance_reaches_the_top_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REACTOR_GRPO_VERIFY_CMD", "true")
    gv._BUDGET = gv.AdaptiveBudget()
    v = asyncio.run(gv.verify(env(GOOD)))
    assert v.authoritative and v.tier == 4
    assert v.score > gv.verify_static(env(GOOD)).score


def test_a_wedged_verifier_cannot_wedge_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout yields NO ANSWER — a hang says nothing about the code."""
    monkeypatch.setenv("REACTOR_GRPO_VERIFY_CMD", "sleep 30")
    monkeypatch.setenv("REACTOR_GRPO_VERIFY_TIMEOUT_S", "0.4")
    gv._BUDGET = gv.AdaptiveBudget()
    assert asyncio.run(gv.verify_authoritative(GOOD)) is None


def test_authority_is_not_consulted_for_code_that_does_not_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spending budget to learn what tier 2 established for free."""
    calls = {"n": 0}

    async def _spy(*a, **k):
        calls["n"] += 1
        return True

    monkeypatch.setattr(gv, "verify_authoritative", _spy)
    asyncio.run(gv.verify(env(BROKEN_EARLY)))
    assert calls["n"] == 0
    asyncio.run(gv.verify(env(GOOD)))
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# The adaptive budget
# --------------------------------------------------------------------------


def test_budget_throttles_when_exhausted() -> None:
    """A fixed rate cannot be right for a validator of unknown cost."""
    b = gv.AdaptiveBudget(share=0.0, burst=1.0)

    async def _go():
        asyncio.get_running_loop()
        await b.record(10.0)          # observed: expensive
        first = await b.try_acquire()  # 1.0 token vs 10s estimate
        return first

    assert asyncio.run(_go()) is False


def test_budget_admits_when_affordable() -> None:
    b = gv.AdaptiveBudget(share=1.0, burst=60.0)

    async def _go():
        await b.record(0.05)
        return await b.try_acquire()

    assert asyncio.run(_go()) is True


def test_budget_cost_estimate_follows_measurement() -> None:
    """EWMA, so one slow outlier does not permanently close the tier."""
    b = gv.AdaptiveBudget()

    async def _go():
        seed = b.estimated_cost_s
        await b.record(20.0)
        spiked = b.estimated_cost_s
        for _ in range(20):
            await b.record(0.05)
        return seed, spiked, b.estimated_cost_s

    seed, spiked, settled = asyncio.run(_go())
    assert spiked > seed
    assert settled < spiked


def test_batch_grades_a_whole_group() -> None:
    out = asyncio.run(gv.verify_batch([env(GOOD), env(BROKEN_EARLY), "garbage"]))
    assert len(out) == 3
    assert out[0].score > out[1].score > out[2].score
