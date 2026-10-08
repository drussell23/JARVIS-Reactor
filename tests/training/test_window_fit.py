"""Train only on prompts whose answers can FINISH inside the completion window.

The schema of a prompt is not the measure of its answer: on 2026-10-08 the
full-content (2b.1) rows of the live soak had a median of 2,316 completion
tokens against a 1,955-token window. With truncated completions masked out of
the loss, a group teaches only through rollouts that terminate, and a contrast
needs two of them. So each contrast-bearing prompt is judged by the answers
the corpus already holds, measured by the engine that produced them.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reactor_core.training import memory_guard as guard

_REPO = Path(__file__).resolve().parents[2]


def _admit(answers, *, window=1955, g=16, need=2):
    return guard.admit_by_completion_window(answers, window=window, num_generations=g, min_terminated=need)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_a_prompt_is_kept_when_enough_rollouts_are_expected_to_finish():
    # 16 rollouts x 1/4 of answers fitting = 4 expected finishes >= 2
    fit = _admit([[900, 2400, 2500, 3000]])
    assert fit.keep == (True,) and fit.admitted == 1


def test_a_prompt_whose_answers_all_overrun_the_window_is_dropped():
    fit = _admit([[2316, 3012, 2600]])
    assert fit.keep == (False,) and fit.too_long == 1


def test_the_threshold_moves_with_the_group_size_not_a_constant():
    answers = [[900] + [3000] * 9]                 # 1 in 10 fits
    assert _admit(answers, g=16).keep == (False,)   # 16 x 0.1 = 1.6 < 2
    assert _admit(answers, g=32).keep == (True,)    # 32 x 0.1 = 3.2 >= 2


def test_the_contrast_floor_is_the_callers():
    answers = [[900, 3000]]                         # half fit: 16 x 0.5 = 8
    assert _admit(answers, need=8).keep == (True,)
    assert _admit(answers, need=9).keep == (False,)


def test_an_unmeasured_prompt_is_excluded_never_guessed():
    fit = _admit([[None, None], [], [0]])
    assert fit.keep == (False, False, False) and fit.unmeasured == 3 and fit.too_long == 0


def test_unmeasured_answers_do_not_dilute_the_measured_share():
    # one measured answer, and it fits: 16 x 1.0 >= 2
    assert _admit([[None, 800, None]]).keep == (True,)


def test_an_answer_exactly_at_the_window_finishes():
    assert _admit([[1955, 1955]]).keep == (True,)


def test_the_report_states_the_rule_and_the_counts():
    rep = _admit([[900], [5000], [None]]).report()
    assert (rep["admitted"], rep["too_long"], rep["unmeasured"]) == (1, 1, 1)
    assert rep["window"] == 1955 and "num_generations" in rep["rule"]


# ---------------------------------------------------------------------------
# Preflight hands each group's measured answers to the trainer
# ---------------------------------------------------------------------------

def _preflight():
    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        import grpo_preflight as pf
    finally:
        sys.path.pop(0)
    return pf


@pytest.mark.parametrize("row, expected", [
    ({"completion_tokens": 812, "tokens_estimated": False}, 812),
    ({"completion_tokens": 812, "tokens_estimated": True}, None),   # inferred from length: not a measurement
    ({"completion_tokens": 0}, None),
    ({}, None),
    ({"completion_tokens": "x"}, None),
])
def test_only_an_engine_reported_length_counts(row, expected):
    assert _preflight()._measured_completion(row) == expected


def test_answer_lengths_travel_parallel_to_the_selected_prompts(tmp_path, monkeypatch):
    monkeypatch.delenv("REACTOR_GRPO_OUTPUT_SCHEMAS", raising=False)
    d = tmp_path / "events"
    d.mkdir()
    rows = []
    for i, (body, ct) in enumerate([("x = 1\n", 400), ("x = 2  # longer\n", 2600)]):
        rows.append({"event_type": "interaction", "event_id": f"e{i}", "user_input": "## Task\nfix x",
                     "assistant_output": body, "outcome": "success" if i else "failure",
                     "confidence": 1.0 if i else 0.0, "completion_tokens": ct, "tokens_estimated": False,
                     "metadata": {"op_id": "op-1", "should_train": True, "draw_kind": "primary",
                                  "candidate_hash": f"h{i}"}})
    (d / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    rep = _preflight().analyse(d, min_group=2, trainable_only=True)
    assert len(rep["trainable_prompts"]) == len(rep["trainable_answer_tokens"])
    if rep["trainable_prompts"]:
        assert sorted(rep["trainable_answer_tokens"][0]) == [400, 2600]


# ---------------------------------------------------------------------------
# The runner applies it where the window is known, parent and rung alike
# ---------------------------------------------------------------------------

def _runner():
    spec = importlib.util.spec_from_file_location("_runner_window_fit_under_test",
                                                  _REPO / "scripts" / "run_grpo_training.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path, **kw):
    base = dict(num_generations=16, min_group_size=2, min_groups=1, skip_corpus_gate=False,
                json_out=str(tmp_path / "r.json"))
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_runner_narrows_to_what_fits(tmp_path):
    r = _runner()
    report = {}
    kept, refused = r._gate_by_window(["a", "b", "c"], [[900], [3000], [None]], 1955, report,
                                      args=_args(tmp_path), guard=guard)
    assert kept == ["a"] and refused is None
    assert report["window_fit"]["applied"] is True and report["training_prompts"] == 1


def test_nothing_fitting_is_a_refusal_named_for_the_window(tmp_path):
    r = _runner()
    report = {}
    kept, refused = r._gate_by_window(["a"], [[3000]], 1955, report, args=_args(tmp_path), guard=guard)
    assert refused == r.EXIT_REFUSED and report["refused"] == "window_fit"
    assert json.loads((tmp_path / "r.json").read_text())["refused"] == "window_fit"


def test_without_a_contrast_selection_nothing_is_narrowed_and_it_says_so(tmp_path):
    r = _runner()
    report = {}
    kept, refused = r._gate_by_window(None, None, 1955, report, args=_args(tmp_path), guard=guard)
    assert kept is None and refused is None and report["window_fit"]["applied"] is False


def test_mismatched_lengths_are_never_zipped(tmp_path):
    r = _runner()
    report = {}
    kept, refused = r._gate_by_window(["a", "b"], [[900]], 1955, report, args=_args(tmp_path), guard=guard)
    assert kept == ["a", "b"] and report["window_fit"]["applied"] is False


def test_the_gate_runs_before_the_step_count_and_in_every_rung():
    src = (_REPO / "scripts" / "run_grpo_training.py").read_text(encoding="utf-8")
    auto = src[src.index("fit = guard.fit_completion_window("):src.index("steps, step_s = guard.fit_steps_to_time(")]
    assert "_gate_by_window(" in auto and "rows = len(contrast_prompts)" in auto
    explicit = src[src.index("args.max_completion_length = int(args.max_completion_length)"):
                   src.index("ladder = guard.build_ladder(")]
    assert "_gate_by_window(" in explicit
