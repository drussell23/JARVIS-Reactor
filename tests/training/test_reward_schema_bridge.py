"""The reward grades what the model WROTE, whatever wrapper it arrived in.

The trainer's reward ran `verify_static`, which grades the O+V JSON
envelope and returns one tier-0 constant for anything else. Measured
2026-09-05 against Qwen3-Coder-30B on a real corpus prompt: four sampled
completions, none an envelope, all four scoring 0.0200 -- spread
0.000000. `candidate_reward` then correctly declines to invent a winner,
TRL drops the group, and the only gradient left is the MoE router's
auxiliary loss. No observed step ever produced a reward signal.

The same four through `verify_any` scored 0.2707 / 0.3586 / 0.2561 /
0.2700 -- spread 0.102511.

The bridge is therefore: the reward's static tier is `verify_any`
(envelope first, source second), and the source fallback grades the FENCED
CODE rather than the prose around it.

The line this must not cross is fabricating contrast. A flat group is a
real answer -- `candidate_reward` returns None and TRL drops it, which is
correct when the siblings really are identical. These tests pin BOTH
directions: different code must separate, and identical code must not.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("_verifier_bridge_under_test",
          _REPO / "reactor_core" / "training" / "grpo_verifier.py")

NL = chr(10)
F = "```"


def _fenced(code: str, lang: str = "python", prose: bool = True) -> str:
    """What the 30B actually emits: prose, a fenced block, more prose."""
    head = "## Summary of Problem" + NL + "The function lacks a guard." + NL + NL
    tail = NL + NL + "This resolves the issue."
    block = F + lang + NL + code + NL + F
    return (head if prose else "") + block + (tail if prose else "")


ENVELOPE = json.dumps({
    "schema_version": "2b.1",
    "candidates": [{"candidate_id": "c1", "file_path": "m.py",
                    "full_content": "def f(a, b):" + NL + '    """Add."""' + NL
                                    + "    return a + b" + NL,
                    "rationale": "fix"}],
})


def _spread(verdicts) -> float:
    s = [v.score for v in verdicts]
    return max(s) - min(s)


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------


def test_fenced_blocks_pulls_the_code_out_of_prose() -> None:
    text = _fenced("def f():" + NL + "    return 1")
    assert V._fenced_blocks(text) == ["def f():" + NL + "    return 1"]


def test_fenced_blocks_drops_the_language_tag() -> None:
    for lang in ("python", "py", "diff", ""):
        got = V._fenced_blocks(_fenced("x = 1", lang=lang))
        assert got == ["x = 1"], lang


def test_fenced_blocks_returns_every_closed_block() -> None:
    text = (F + "python" + NL + "a = 1" + NL + F + NL + "prose" + NL
            + F + "python" + NL + "b = 2" + NL + F)
    assert V._fenced_blocks(text) == ["a = 1", "b = 2"]


def test_an_unclosed_fence_yields_nothing() -> None:
    """A truncated block is not evidence about code never finished."""
    assert V._fenced_blocks("prose" + NL + F + "python" + NL + "a = 1") == []


def test_no_fence_yields_nothing_so_the_caller_keeps_its_behaviour() -> None:
    assert V._fenced_blocks("just prose, no code at all") == []
    assert V._fenced_blocks("") == []


# ---------------------------------------------------------------------------
# The fallback grades code, not prose
# ---------------------------------------------------------------------------


def test_verify_any_grades_the_fenced_code_not_the_prose() -> None:
    clean = _fenced("def f(a, b):" + NL + '    """Add."""' + NL + "    return a + b")
    v = V.verify_any(clean)
    assert v.tier >= 2, f"clean fenced code should reach the graded band, got {v}"
    assert "syntax_error" not in v.reason, v.reason


def test_prose_wrapping_does_not_change_the_grade() -> None:
    """The fence and its prose are presentation; the code is the candidate."""
    code = "def f(a, b):" + NL + "    return a + b"
    bare = V.verify_any(code)
    wrapped = V.verify_any(_fenced(code))
    assert bare.score == pytest.approx(wrapped.score), (bare, wrapped)


def test_broken_code_still_scores_below_clean_code() -> None:
    good = V.verify_any(_fenced("def f(a, b):" + NL + "    return a + b"))
    bad = V.verify_any(_fenced("def f(a, b)" + NL + "    return a+b"))
    assert bad.score < good.score


def test_the_worst_block_decides_a_multi_block_answer() -> None:
    """Same rule the envelope path uses across a candidate's files."""
    good = "def f():" + NL + "    return 1"
    broken = "def g(" + NL + "    return"
    both = (F + "python" + NL + good + NL + F + NL + "and also" + NL
            + F + "python" + NL + broken + NL + F)
    assert V.verify_any(both).score == pytest.approx(
        V.verify_any(_fenced(broken, prose=False)).score)


# ---------------------------------------------------------------------------
# The reward path itself
# ---------------------------------------------------------------------------


def _batch(texts):
    return asyncio.run(V.verify_batch(texts))


def test_the_reward_path_now_separates_different_answers() -> None:
    """The shape the 30B actually produced, with genuinely different code."""
    group = [
        _fenced("def f(a, b):" + NL + '    """Add."""' + NL + "    return a + b"),
        _fenced("def f(a, b):" + NL + "    return a + b"),
        _fenced("def f(a, b)" + NL + "    return a+b"),          # syntax error
        "no code at all, only an explanation of the change",
    ]
    assert _spread(_batch(group)) > 0.01, "the reward must see these as different"


def test_identical_answers_still_produce_a_flat_group(monkeypatch) -> None:
    """The line this must not cross. A flat group is a real answer, and
    manufacturing variance to avoid it would be the defect, not the fix."""
    same = _fenced("def f(a, b):" + NL + "    return a + b")
    assert _spread(_batch([same] * 8)) == pytest.approx(0.0)


def test_the_envelope_still_outranks_fenced_prose() -> None:
    """The incentive is preserved: emitting the envelope is still better."""
    env = _batch([ENVELOPE])[0]
    fenced = _batch([_fenced("def f(a, b):" + NL + '    """Add."""' + NL
                             + "    return a + b")])[0]
    assert env.score > fenced.score, (env, fenced)
    assert env.schema_version == V.SCHEMA_FULL


def test_envelope_only_env_var_restores_the_old_grader(monkeypatch) -> None:
    monkeypatch.setenv("REACTOR_GRPO_REWARD_ENVELOPE_ONLY", "1")
    group = [
        _fenced("def f(a, b):" + NL + "    return a + b"),
        _fenced("def f(a, b)" + NL + "    return a+b"),
        "prose only",
    ]
    assert _spread(_batch(group)) == pytest.approx(0.0), (
        "the escape hatch must reproduce the flat, envelope-only behaviour")


def test_a_real_envelope_is_still_graded_as_an_envelope(monkeypatch) -> None:
    """Envelope FIRST: a JSON object must never fall through to the source
    grader, which would grade it as the Python dict literal it also is."""
    v = _batch([ENVELOPE])[0]
    assert v.schema_version == V.SCHEMA_FULL
    assert "envelope_unparseable" not in v.reason


# ---------------------------------------------------------------------------
# The footprint must describe the same candidate the grade came from
# ---------------------------------------------------------------------------


def test_sources_for_mirrors_the_fence_decision() -> None:
    code = "def f():" + NL + "    return 1"
    assert V._sources_for(_fenced(code)) == [code]
    assert V._sources_for("plain prose") == ["plain prose"]


def test_refine_group_sees_the_code_not_the_prose() -> None:
    """refine_group compares siblings; if it compared prose it would report
    two different answers as ~identical because the prose is boilerplate."""
    a = _fenced("def f():" + NL + "    return 1")
    b = _fenced("def f():" + NL + "    return 2" + NL + NL + "def g():" + NL + "    return 3")
    verdicts = V.verify_group([a, b], prompt="fix f")
    assert _spread(verdicts) > 0.0
