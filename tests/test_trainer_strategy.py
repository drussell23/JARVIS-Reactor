"""Which trainer runs a cycle is resolved, never hardcoded.

The pipeline hardcoded `AsyncTrainer` (SFT) as the only trainer, so GRPO --
the method the flywheel needs, because the corpus is graded rollouts rather
than matched preference pairs -- was unreachable from the automated loop
even though its runner worked. Two trainers, one loop, not connected.

The contract pinned here:
  * resolution order is request, then env, then SFT;
  * an unknown name RAISES (a typo must not silently train the wrong way);
  * SFT stays the default, so an unconfigured pipeline is unchanged;
  * GRPO composes the RUNNER as a child, because the runner owns the
    admission gate, corpus gate, prompt budget and isolated ladder;
  * every runner exit code maps to an outcome a scheduler can act on, and
    a refusal is distinguishable from a failure.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.training import trainer_strategy as ts  # noqa: E402


def _req(tmp_path: Path, **kw: Any) -> ts.TrainerRequest:
    kw.setdefault("base_model", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    kw.setdefault("output_dir", tmp_path / "out")
    return ts.TrainerRequest(**kw)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_the_default_is_sft_so_an_unconfigured_pipeline_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv(ts.ENV_STRATEGY, raising=False)
    assert ts.resolve_name(_req(tmp_path)) == ts.STRATEGY_SFT
    assert ts.DEFAULT_STRATEGY == ts.STRATEGY_SFT


def test_env_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv(ts.ENV_STRATEGY, "grpo")
    assert ts.resolve_name(_req(tmp_path)) == ts.STRATEGY_GRPO


def test_an_explicit_request_beats_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ts.ENV_STRATEGY, "grpo")
    assert ts.resolve_name(_req(tmp_path, strategy="lora_sft")) == ts.STRATEGY_SFT


def test_names_are_case_and_space_insensitive(monkeypatch, tmp_path):
    monkeypatch.setenv(ts.ENV_STRATEGY, "  GRPO  ")
    assert ts.resolve_name(_req(tmp_path)) == ts.STRATEGY_GRPO


def test_an_unknown_strategy_raises_rather_than_falling_back():
    """A typo that silently trained the wrong way would surface days later
    in the artifact, which is the whole reason a method has a name."""
    with pytest.raises(KeyError) as exc:
        ts.resolve("grpoo")
    assert "grpoo" in str(exc.value)
    assert "grpo" in str(exc.value), "the error names what IS registered"


def test_registering_a_name_twice_raises():
    async def _noop(request):
        return ts.TrainerOutcome(ok=True, strategy="dupe")

    ts.register("dupe-test", _noop)
    try:
        with pytest.raises(ValueError):
            ts.register("dupe-test", _noop)
        ts.register("dupe-test", _noop, replace=True)   # explicit is fine
    finally:
        ts._REGISTRY.pop("dupe-test", None)


def test_grpo_is_registered():
    assert ts.STRATEGY_GRPO in ts.available()


# ---------------------------------------------------------------------------
# The GRPO child's command line
# ---------------------------------------------------------------------------


def test_the_argv_carries_model_output_and_report(tmp_path):
    argv = ts.build_grpo_argv(_req(tmp_path), report_path=tmp_path / "r.json")
    assert argv[0] == sys.executable
    assert "-u" in argv, "an unbuffered child reports as it runs, not at exit"
    assert argv[argv.index("--model") + 1].endswith("Qwen3-Coder-30B-A3B-Instruct")
    assert argv[argv.index("--json-out") + 1] == str(tmp_path / "r.json")
    assert argv[argv.index("--output-dir") + 1] == str(tmp_path / "out")


def test_options_reach_the_runner_as_flags(tmp_path):
    argv = ts.build_grpo_argv(
        _req(tmp_path, options={
            "num_generations": 16, "max_completion_length": 256,
            "max_prompt_tokens": 4096, "train_truncated": True,
        }),
        report_path=tmp_path / "r.json",
    )
    assert argv[argv.index("--num-generations") + 1] == "16"
    assert argv[argv.index("--max-completion-length") + 1] == "256"
    assert argv[argv.index("--max-prompt-tokens") + 1] == "4096"
    assert "--train-truncated" in argv


def test_unknown_options_are_ignored_not_forwarded(tmp_path):
    """One config serves several strategies; a knob GRPO does not know must
    not become a flag the runner rejects."""
    argv = ts.build_grpo_argv(
        _req(tmp_path, options={"lora_alpha": 128, "beta": 0.1}),
        report_path=tmp_path / "r.json",
    )
    assert "--lora-alpha" not in argv and "128" not in argv


def test_extra_args_are_an_env_escape_hatch(tmp_path, monkeypatch):
    monkeypatch.setenv(ts.ENV_GRPO_ARGS, "--skip-admission --all-prompts")
    argv = ts.build_grpo_argv(_req(tmp_path), report_path=tmp_path / "r.json")
    assert "--skip-admission" in argv and "--all-prompts" in argv


def test_the_runner_path_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv(ts.ENV_GRPO_RUNNER, str(tmp_path / "elsewhere.py"))
    argv = ts.build_grpo_argv(_req(tmp_path), report_path=tmp_path / "r.json")
    assert str(tmp_path / "elsewhere.py") in argv


def test_the_default_runner_path_exists():
    """The composition is real: the file this module shells out to is the
    runner in this checkout."""
    monkeypatched = ts._runner_path()
    assert monkeypatched.name == "run_grpo_training.py"
    assert monkeypatched.is_file(), monkeypatched


def test_exit_codes_match_the_runners():
    """Two modules agreeing on integers by comment is how they drift."""
    spec = importlib.util.spec_from_file_location(
        "_runner_codes", _REPO / "scripts" / "run_grpo_training.py")
    assert spec and spec.loader
    mod: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules["_runner_codes"] = mod
    spec.loader.exec_module(mod)
    assert ts.EXIT_OK == mod.EXIT_OK
    assert ts.EXIT_ERROR == mod.EXIT_ERROR
    assert ts.EXIT_REFUSED == mod.EXIT_REFUSED
    assert ts.EXIT_LADDER_EXHAUSTED == mod.EXIT_LADDER_EXHAUSTED


# ---------------------------------------------------------------------------
# Exit code -> outcome
# ---------------------------------------------------------------------------


def _fake_child(monkeypatch, *, code: int, report: dict | None = None, out: str = ""):
    """Stand in for the runner: writes a report, exits with a code."""
    class _Proc:
        returncode = code

        def __init__(self, argv):
            self.argv = argv
            self.stdout = self._lines()

        async def _lines(self):
            for line in (out.splitlines() or []):
                yield (line + "\n").encode()

        async def wait(self):
            return code

    seen = {}

    async def _exec(*argv, **kw):
        seen["argv"] = argv
        if report is not None:
            path = Path(argv[argv.index("--json-out") + 1])
            path.write_text(json.dumps(report), encoding="utf-8")
        return _Proc(argv)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    return seen


@pytest.mark.asyncio
async def test_a_trained_run_reports_the_saved_adapter(monkeypatch, tmp_path):
    saved = tmp_path / "run-1"
    saved.mkdir()
    _fake_child(monkeypatch, code=ts.EXIT_OK, report={
        "status": "trained",
        "output_dir": str(tmp_path / "configured"),
        "attempts": [{"rung": "as-configured", "saved_to": str(saved)}],
    })
    out = await ts.grpo_strategy(_req(tmp_path))
    assert out.ok and out.strategy == ts.STRATEGY_GRPO
    assert out.adapter_path == saved, "saved_to beats the configured dir"
    assert not out.refused


@pytest.mark.asyncio
async def test_exit_zero_with_no_saved_adapter_is_a_failure(monkeypatch, tmp_path):
    """Untrained weights labelled as a training result is the worst shape."""
    _fake_child(monkeypatch, code=ts.EXIT_OK, report={"status": "trained",
                                                      "attempts": []})
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and "no saved adapter" in out.reason


@pytest.mark.asyncio
async def test_a_refusal_is_not_a_failure(monkeypatch, tmp_path):
    """A scheduler retries a refusal and investigates a failure."""
    _fake_child(monkeypatch, code=ts.EXIT_REFUSED,
                report={"refused": "GPU busy: 29.1 GiB held"})
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and out.refused
    assert "29.1 GiB" in out.reason


@pytest.mark.asyncio
async def test_ladder_exhaustion_is_a_refusal_with_its_own_reason(monkeypatch, tmp_path):
    _fake_child(monkeypatch, code=ts.EXIT_LADDER_EXHAUSTED, report={
        "status": "ladder-exhausted"})
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and out.refused
    assert "does not fit" in out.reason
    assert out.metrics.get("status") == "ladder-exhausted"


@pytest.mark.asyncio
async def test_an_error_carries_the_runners_own_message(monkeypatch, tmp_path):
    _fake_child(monkeypatch, code=ts.EXIT_ERROR,
                report={"error": "OutOfMemoryError: tried to allocate 1.57 GiB"})
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and not out.refused
    assert "1.57 GiB" in out.reason


@pytest.mark.asyncio
async def test_a_child_that_wrote_no_report_still_reports(monkeypatch, tmp_path):
    """A process killed by the memory guard leaves nothing behind."""
    _fake_child(monkeypatch, code=137, report=None, out="loading\nkilled")
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and out.exit_code == 137
    assert out.reason, "silence is not an acceptable explanation"


@pytest.mark.asyncio
async def test_a_missing_runner_refuses_before_spawning(monkeypatch, tmp_path):
    monkeypatch.setenv(ts.ENV_GRPO_RUNNER, str(tmp_path / "absent.py"))
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and "not found" in out.reason


@pytest.mark.asyncio
async def test_the_strategy_reports_instead_of_exploding(monkeypatch, tmp_path):
    async def _boom(*a, **k):
        raise OSError("no fork for you")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
    out = await ts.grpo_strategy(_req(tmp_path))
    assert not out.ok and "OSError" in out.reason


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch, tmp_path):
    async def _cancel(*a, **k):
        raise asyncio.CancelledError()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _cancel)
    with pytest.raises(asyncio.CancelledError):
        await ts.grpo_strategy(_req(tmp_path))


@pytest.mark.asyncio
async def test_run_dispatches_through_the_registry(monkeypatch, tmp_path):
    called = {}

    async def _fake(request):
        called["model"] = request.base_model
        return ts.TrainerOutcome(ok=True, strategy="probe", reason="ok")

    ts.register("probe", _fake, replace=True)
    try:
        out = await ts.run(_req(tmp_path, strategy="probe"))
        assert out.ok and called["model"].endswith("Instruct")
    finally:
        ts._REGISTRY.pop("probe", None)


def test_the_outcome_summary_says_what_happened(tmp_path):
    ok = ts.TrainerOutcome(ok=True, strategy="grpo", adapter_path=tmp_path,
                           reason="saved")
    assert "trained" in ok.summary() and str(tmp_path) in ok.summary()
    refused = ts.TrainerOutcome(ok=False, strategy="grpo", refused=True,
                                reason="busy")
    assert "refused" in refused.summary()
    failed = ts.TrainerOutcome(ok=False, strategy="grpo", reason="boom")
    assert "failed" in failed.summary()


# ---------------------------------------------------------------------------
# The pipeline consults the registry
# ---------------------------------------------------------------------------


def test_the_pipeline_no_longer_names_a_trainer_in_train_model():
    import inspect
    from reactor_core.training.unified_pipeline import UnifiedTrainingPipeline
    src = inspect.getsource(UnifiedTrainingPipeline._train_model)
    assert "trainer_strategy" in src, "the choice must be resolved, not hardcoded"
    assert "AsyncTrainer" not in src, "the SFT trainer moved to _train_sft"
    sft = inspect.getsource(UnifiedTrainingPipeline._train_sft)
    assert "AsyncTrainer" in sft, "SFT is unchanged, just reached differently"


def test_the_config_carries_the_choice():
    from reactor_core.training.unified_pipeline import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.training_strategy == "", "empty resolves through env then default"
    assert cfg.trainer_options == {}
