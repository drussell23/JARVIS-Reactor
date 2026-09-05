"""The trainer iterates the groups the GATE selected, not every admitted row.

Two definitions of "trainable" are live at once and they answer different
questions. ``build_prompt_dataset(trainable_only=True)`` filters ROWS to
genuine draws. ``grpo_preflight.analyse`` additionally demands two or more
responses whose grades differ. Measured on the live corpus 2026-09-05: 242
prompts survive the first and 27 survive the second. At 656.8s per optimiser
step that is 44.2 hours against 4.9 -- the largest single lever available,
and it costs nothing but selecting the right rows.

The selection is PASSED from the gate to the loader, never recomputed. A
second implementation of "which prompts carry contrast" is precisely the
drift the runner's docstring already forbids for corpus quality, and the
profiler's group-size bug showed what it costs when two halves of the same
run disagree about the configuration.

What these pin:

* the gate publishes one full prompt per trainable group, and the count
  matches its own verdict;
* the loader narrows to exactly those, matching on the same strip() key it
  dedupes with, so whitespace cannot silently drop a selected prompt;
* the un-narrowed path stays byte-identical;
* a total miss raises a WIRING error, not "empty corpus", because those send
  the reader to opposite places;
* the runner keeps the prompts out of its JSON report and honours
  ``--all-prompts``.
"""
from __future__ import annotations

import importlib.util
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


runner = _load("_runner_contrast_under_test", _REPO / "scripts" / "run_grpo_training.py")
pipeline = _load(
    "_pipeline_contrast_under_test",
    _REPO / "reactor_core" / "training" / "grpo_pipeline.py",
)


def _row(prompt: str, out: str = "ok", conf: float = 0.5) -> dict:
    return {
        "user_input": prompt,
        "assistant_output": out,
        "outcome": "success",
        "confidence": conf,
        "latency_ms": 1.0,
        "model_id": "m",
        "task_type": "code_repair",
        "metadata": {"op_id": "op-1"},
    }


@pytest.fixture
def corpus(monkeypatch):
    """Four distinct prompts through the row filter."""
    rows = [
        _row("alpha prompt"), _row("alpha prompt"),   # duplicate -> deduped
        _row("beta prompt"),
        _row("  gamma prompt  "),                     # padded on both sides
        _row("delta prompt"),
    ]
    monkeypatch.setattr(pipeline, "iter_trajectory_rows",
                        lambda _d, **_k: iter(rows))
    return rows


# ---------------------------------------------------------------------------
# The loader narrows to what it is given
# ---------------------------------------------------------------------------


def test_unfiltered_keeps_every_distinct_prompt(corpus, tmp_path) -> None:
    ds = pipeline.build_prompt_dataset(tmp_path)
    assert len(ds) == 4, "dedup only; the default path must not change"


def test_only_prompts_narrows_to_the_selection(corpus, tmp_path) -> None:
    ds = pipeline.build_prompt_dataset(
        tmp_path, only_prompts=["beta prompt", "delta prompt"])
    assert sorted(r["prompt"] for r in ds) == ["beta prompt", "delta prompt"]


def test_selection_matches_on_the_dedup_key_not_raw_text(corpus, tmp_path) -> None:
    """The gate groups on the raw string, the loader dedupes on .strip().
    A prompt padded in the corpus must still match its selected form."""
    ds = pipeline.build_prompt_dataset(tmp_path, only_prompts=["gamma prompt"])
    assert len(ds) == 1
    assert ds[0]["prompt"].strip() == "gamma prompt"


def test_empty_and_blank_entries_in_the_selection_are_ignored(corpus, tmp_path) -> None:
    ds = pipeline.build_prompt_dataset(
        tmp_path, only_prompts=["beta prompt", "", "   ", None and ""])
    assert len(ds) == 1


def test_a_total_miss_is_a_wiring_error_not_an_empty_corpus(corpus, tmp_path) -> None:
    with pytest.raises(ValueError, match="matched the"):
        pipeline.build_prompt_dataset(tmp_path, only_prompts=["nothing like this"])


def test_empty_corpus_still_reports_an_empty_corpus(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline, "iter_trajectory_rows", lambda _d, **_k: iter([]))
    with pytest.raises(ValueError, match="no trainable prompts"):
        pipeline.build_prompt_dataset(tmp_path)


def test_partial_miss_warns_but_trains(corpus, tmp_path, caplog) -> None:
    """A selected prompt the loader cannot find is a real signal; it must be
    said out loud rather than silently shrinking the epoch."""
    import logging
    with caplog.at_level(logging.WARNING, logger=pipeline.logger.name):
        ds = pipeline.build_prompt_dataset(
            tmp_path, only_prompts=["beta prompt", "ghost prompt"])
    assert len(ds) == 1
    assert any("not found by the loader" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The gate publishes what the loader consumes
# ---------------------------------------------------------------------------


def test_gate_publishes_one_full_prompt_per_trainable_group(monkeypatch, tmp_path) -> None:
    preflight = _load("_preflight_contrast_under_test",
                      _REPO / "scripts" / "grpo_preflight.py")
    long_a = "A" * 200
    long_b = "B" * 200
    rows = [
        # contrast: two responses graded differently
        _row(long_a, out="def f():\n    return 1\n"),
        _row(long_a, out="not code at all"),
        # flat: identical responses
        _row(long_b, out="def g():\n    return 2\n"),
        _row(long_b, out="def g():\n    return 2\n"),
        # singleton: one response, below min group size
        _row("C" * 200, out="def h():\n    return 3\n"),
    ]
    # preflight resolves its dependencies lazily through ``_load(name)``,
    # so that is the seam -- the corpus reader is stubbed and the real
    # verifier and reward are left alone, because the grades they produce
    # are what the group verdict is being tested on.
    from types import SimpleNamespace
    real_load = preflight._load
    stub = SimpleNamespace(iter_trajectory_rows=lambda _d, **_k: iter(rows))
    monkeypatch.setattr(
        preflight, "_load",
        lambda name: stub if name == "grpo_pipeline" else real_load(name),
    )
    v = preflight.analyse(tmp_path, min_group=2, trainable_only=True, min_spread=0.0)
    published = v["trainable_prompts"]
    assert len(published) == v["trainable_groups"]
    assert all(len(p) == 200 for p in published), "FULL prompts, not 80-char heads"
    assert v["groups_below_min"] == 1
    # And the report's own examples stay truncated, so the JSON stays small.
    assert all(len(e["prompt_head"]) <= 80 for e in v["examples"])


# ---------------------------------------------------------------------------
# The runner wires the two together
# ---------------------------------------------------------------------------


def test_runner_forwards_the_selection_to_the_trainer() -> None:
    import inspect
    src = inspect.getsource(runner.train_with_ladder)
    assert "only_prompts=only_prompts" in src, (
        "the ladder must hand the gate's selection to build_trainer")
    assert "only_prompts" in inspect.signature(runner.train_with_ladder).parameters


def test_runner_keeps_the_prompts_out_of_the_report() -> None:
    """27 prompts of ~24k chars would make the report unreadable, and the
    count is what anyone reads."""
    import inspect
    src = inspect.getsource(runner.main)
    pop = src.index('verdict.pop("trainable_prompts"')
    assign = src.index('report["corpus"] = verdict')
    assert pop < assign, "pop before the verdict becomes the report"
    assert 'report["training_prompts"]' in src


def test_all_prompts_flag_restores_the_old_behaviour() -> None:
    import inspect
    src = inspect.getsource(runner.main)
    assert "if args.all_prompts:" in src
    assert "contrast_prompts = None" in src


def test_build_trainer_accepts_and_forwards_only_prompts() -> None:
    import inspect
    assert "only_prompts" in inspect.signature(pipeline.build_trainer).parameters
    src = inspect.getsource(pipeline.build_trainer)
    assert "only_prompts=only_prompts" in src
