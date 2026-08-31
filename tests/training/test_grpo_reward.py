"""GRPO reward over sibling groups: verification is the signal.

The first version of this reward made `_score_candidate` primary. A live
profiling run proved that wrong: its inputs (`outcome`, `confidence`,
`latency_ms`) are columns of the DATASET ROW, so TRL hands the SAME
metadata to every completion in a group. The scorer returned N identical
values by construction, every group was flat before the model had said
anything, and the run logged `rewards/candidate_reward/mean: None`,
`loss: 0`, `grad_norm: 0`.

Those fields describe a historical generation, not the completion being
scored. Verification of the completion is now the signal; the historical
scorer survives only as a small nudge.

The structural LADDER lives in `grpo_verifier` and is tested in
`test_grpo_verifier.py`. It is deliberately not duplicated here — it WAS
duplicated once (as `structural_severity`), and two graders of the same
candidate are two things free to disagree.

Loaded by path so these tests do not depend on the package `__init__`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
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

GOOD = "def page(items, n):\n    return items[:n]\n"
BROKEN = "def broken(:\n" + "x = 1\n" * 20
INERT = '"""just a docstring"""\n'


def env(code: str) -> str:
    """What the model ACTUALLY emits: the 2b.1 envelope, code inside.

    Feeding bare Python would not resemble a completion, and testing a
    grader on input it will never see is how the envelope-blind defect
    survived in the first place.
    """
    return json.dumps({"schema_version": "2b.1", "candidates": [
        {"candidate_id": "c1", "file_path": "m.py", "rationale": "r",
         "full_content": code}]})


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


def _run(codes: List[str], **kw):
    """One group, with metadata IDENTICAL across siblings — the real case."""
    n = len(codes)
    kw.setdefault("outcome", ["partial"] * n)
    kw.setdefault("confidence", [0.5] * n)
    kw.setdefault("latency_ms", [115537.9] * n)
    kw.setdefault("model_id", ["qwen3-coder:30b"] * n)
    kw.setdefault("task_type", ["code_repair"] * n)
    return asyncio.run(candidate_reward(completions=[env(c) for c in codes], **kw))


# --------------------------------------------------------------------------
# The defect this reward was rebuilt around
# --------------------------------------------------------------------------


def test_identical_metadata_still_separates_on_code_quality() -> None:
    """THE regression.

    All three siblings carry the same outcome/confidence/latency, because
    that is what TRL passes. Separation must come from the completions.
    """
    r = _run([GOOD, BROKEN, INERT])
    assert all(x is not None for x in r), "group dropped despite differing code"
    adv = _advantage(r)
    assert adv is not None
    assert adv[0] == max(adv), "best code did not get the highest advantage"
    assert adv[1] == min(adv), "broken code did not get the lowest advantage"


def test_history_alone_cannot_carry_a_group() -> None:
    """With identical metadata the historical term is a constant.

    A constant cannot create variance, so when verification ties the
    group is dropped no matter what the history says.
    """
    assert _run([GOOD, GOOD, GOOD]) == [None, None, None]


def test_history_nudge_cannot_overturn_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nudge must stay a nudge.

    Here history FAVOURS the broken candidate (success/1.0) and opposes
    the good one (failure/0.0). Better code must still win, or the reward
    is back to scoring a historical generation.
    """
    monkeypatch.setenv("REACTOR_GRPO_HISTORY_WEIGHT", "0.10")
    r = _run([GOOD, BROKEN], outcome=["failure", "success"], confidence=[0.0, 1.0])
    assert r[0] > r[1], "history overturned a verification difference"


def test_history_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REACTOR_GRPO_HISTORY_WEIGHT", "0")
    r = _run([GOOD, BROKEN])
    assert all(x is not None for x in r)
    assert r[0] > r[1]


# --------------------------------------------------------------------------
# The zero-variance rule
# --------------------------------------------------------------------------


def test_indistinguishable_group_is_dropped_not_fabricated() -> None:
    """No signal exists, so none is invented."""
    assert _run([BROKEN, BROKEN, BROKEN]) == [None, None, None]


def test_a_uniform_shift_would_not_have_worked() -> None:
    """Pins the arithmetic the whole design rests on.

    A flat penalty leaves variance at zero. Only a per-candidate term
    helps — which is why the verifier grades in BANDS rather than
    pass/fail. Asserted directly so nobody "simplifies" it later.
    """
    flat = [0.5, 0.5, 0.5]
    assert _advantage([v - 0.3 for v in flat]) is None


def test_empty_group_is_handled() -> None:
    assert asyncio.run(candidate_reward(completions=[])) == []


def test_single_completion_cannot_support_an_advantage() -> None:
    """n=1 is degenerate by definition — the state before sibling drawing."""
    assert _advantage(_run([GOOD])) is None


@pytest.mark.parametrize("junk", [
    [""], ["not json"], ["{}"], ["```"], ["null", "[]"],
    ['{"schema_version": "2b.1"}', '{"candidates": []}'],
])
def test_malformed_completions_never_raise(junk: List[str]) -> None:
    """A reward function that raises kills the whole training run."""
    n = len(junk)
    out = asyncio.run(candidate_reward(
        completions=junk, outcome=["unknown"] * n, confidence=[0.5] * n,
        latency_ms=[0.0] * n, model_id=["m"] * n, task_type=["t"] * n,
    ))
    assert len(out) == n
