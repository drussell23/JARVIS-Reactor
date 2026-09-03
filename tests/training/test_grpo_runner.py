"""The runner refuses before it allocates, and never trains on leakage.

``run_grpo_training`` is the first caller ``build_trainer`` has ever had,
so the behaviours worth pinning are the ones that decide whether a
GPU-hour is spent at all:

* both gates return their verdict BEFORE the ML stack is imported, and a
  refusal is a distinct exit code from an error — ``2`` means "I looked
  and the answer is no", ``1`` means the runner broke. A caller that
  cannot tell those apart cannot automate on top of this;
* the dataset handed to TRL carries prompts and reward columns and
  **nothing else**. GRPO generates its own completions; a reference
  answer riding along in a column is train-on-test and would be
  invisible in the loss curve.

The module is loaded by path so importing it costs no torch.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PATH = _REPO / "scripts" / "run_grpo_training.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_runner_under_test", _PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load()

_REWARD_COLUMNS = ("outcome", "confidence", "latency_ms", "model_id", "task_type")


class _FakeDataset:
    """Only the surface ``validate_dataset`` touches."""

    def __init__(self, rows):
        self._rows = rows

    @property
    def column_names(self):
        return sorted(self._rows[0].keys()) if self._rows else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]


def _row(prompt: str, **extra):
    row = {
        "prompt": prompt,
        "outcome": "success",
        "confidence": 0.5,
        "latency_ms": 1000.0,
        "model_id": "qwen3-coder:30b",
        "task_type": "code",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Dataset shape and leakage
# ---------------------------------------------------------------------------


def test_a_clean_dataset_validates() -> None:
    ds = _FakeDataset([_row("fix the re-raise"), _row("use timezone.utc")])
    report = runner.validate_dataset(ds, reward_columns=_REWARD_COLUMNS)
    assert report["rows"] == 2
    assert report["distinct_prompts"] == 2
    assert report["leakage_columns"] == []
    assert report["prompt_chars"]["max"] >= report["prompt_chars"]["min"]


@pytest.mark.parametrize("leak", [
    "completion", "assistant_output", "chosen", "rejected", "labels",
    "response", "candidate", "reference",
])
def test_reference_answer_columns_are_refused(leak: str) -> None:
    """GRPO is graded on what IT generated, not on the corpus's answer."""
    ds = _FakeDataset([_row("p1", **{leak: "def f(): ..."})])
    with pytest.raises(ValueError, match="leakage"):
        runner.validate_dataset(ds, reward_columns=_REWARD_COLUMNS)


def test_missing_prompt_column_is_refused() -> None:
    ds = _FakeDataset([{"outcome": "success"}])
    with pytest.raises(ValueError, match="no 'prompt' column"):
        runner.validate_dataset(ds, reward_columns=_REWARD_COLUMNS)


def test_missing_reward_columns_are_refused() -> None:
    """Absent, candidate_reward's nudge silently scores every row on
    defaults and the run still looks healthy."""
    ds = _FakeDataset([{"prompt": "p1", "outcome": "success"}])
    with pytest.raises(ValueError, match="missing reward column"):
        runner.validate_dataset(ds, reward_columns=_REWARD_COLUMNS)


def test_duplicate_prompts_are_refused() -> None:
    """build_prompt_dataset deduplicates; a survivor reweights the epoch."""
    ds = _FakeDataset([_row("same"), _row("same")])
    with pytest.raises(ValueError, match="duplicate prompt"):
        runner.validate_dataset(ds, reward_columns=_REWARD_COLUMNS)


def test_blank_prompts_are_refused() -> None:
    ds = _FakeDataset([_row("real"), _row("   ")])
    with pytest.raises(ValueError, match="blank"):
        runner.validate_dataset(ds, reward_columns=_REWARD_COLUMNS)


def test_empty_dataset_is_refused() -> None:
    class _Empty(_FakeDataset):
        @property
        def column_names(self):
            return ["prompt", *_REWARD_COLUMNS]

    with pytest.raises(ValueError, match="empty"):
        runner.validate_dataset(_Empty([]), reward_columns=_REWARD_COLUMNS)


# ---------------------------------------------------------------------------
# The gates, and the exit codes that distinguish refusal from failure
# ---------------------------------------------------------------------------


def _verdict(trainable_groups: int) -> dict:
    return {
        "rows": 49, "prompts": 29, "groups_below_min": 16,
        "flat_groups": 9, "trainable_groups": trainable_groups,
        "min_spread": 0.01, "trainable_only": True,
    }


def _fake_guard(*, admitted: bool = True) -> SimpleNamespace:
    sample = SimpleNamespace(to_dict=lambda: {"vram_occupancy_pct": 1.7})
    return SimpleNamespace(
        check_admission=lambda **kw: SimpleNamespace(
            allowed=admitted,
            reason="ok" if admitted else "VRAM occupancy 89.2% > 55.0%",
            sample=sample,
        ),
        build_ladder=lambda **kw: [],
        MemoryWatchdog=object,
        is_oom=lambda exc: False,
        free_cuda_memory=lambda: None,
    )


def _patch_common(monkeypatch, *, groups: int, admitted: bool = True):
    monkeypatch.setattr(runner, "corpus_gate",
                        lambda *a, **k: {**_verdict(groups),
                                         "passes": groups >= k["min_groups"],
                                         "min_groups_required": k["min_groups"]})
    monkeypatch.setattr(runner, "_load_by_path",
                        lambda *a, **k: _fake_guard(admitted=admitted))


def test_a_thin_corpus_is_refused_not_failed(monkeypatch, tmp_path) -> None:
    _patch_common(monkeypatch, groups=0)
    rc = runner.main([
        "--model", "does/not/matter", "--telemetry-dir", str(tmp_path),
        "--min-groups", "1",
    ])
    assert rc == runner.EXIT_REFUSED


def test_an_occupied_card_is_refused_not_failed(monkeypatch, tmp_path) -> None:
    """A soak's resident model and a training run cannot share the card."""
    _patch_common(monkeypatch, groups=4, admitted=False)
    rc = runner.main([
        "--model", "does/not/matter", "--telemetry-dir", str(tmp_path),
    ])
    assert rc == runner.EXIT_REFUSED


def test_refusal_codes_are_distinct_from_error() -> None:
    codes = {runner.EXIT_OK, runner.EXIT_ERROR, runner.EXIT_REFUSED,
             runner.EXIT_LADDER_EXHAUSTED}
    assert len(codes) == 4


def test_corpus_gate_runs_before_any_admission(monkeypatch, tmp_path) -> None:
    """Order matters: a refusal should cost a second, not the forty it
    takes to bring torch into the process."""
    calls = []
    monkeypatch.setattr(runner, "corpus_gate",
                        lambda *a, **k: (calls.append("corpus"),
                                         {**_verdict(0), "passes": False,
                                          "min_groups_required": 1})[1])

    guard = _fake_guard()
    original = guard.check_admission
    guard.check_admission = lambda **kw: (calls.append("admission"),
                                          original(**kw))[1]
    monkeypatch.setattr(runner, "_load_by_path", lambda *a, **k: guard)

    runner.main(["--model", "m", "--telemetry-dir", str(tmp_path)])
    assert calls == ["corpus"], "admission must not run once the corpus fails"


def test_skip_corpus_gate_reaches_admission(monkeypatch, tmp_path) -> None:
    _patch_common(monkeypatch, groups=0, admitted=False)
    rc = runner.main([
        "--model", "m", "--telemetry-dir", str(tmp_path),
        "--skip-corpus-gate",
    ])
    # Corpus waved through, so the refusal below must be the hardware one.
    assert rc == runner.EXIT_REFUSED


def test_leakage_column_set_covers_the_recorder_field() -> None:
    """`assistant_output` is what the corpus literally calls the candidate;
    it is the column most likely to be added by accident."""
    assert "assistant_output" in runner._LEAKAGE_COLUMNS
