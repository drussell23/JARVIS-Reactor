"""The night-shift loop reaches the trainer that works and ships what it makes.

Three joins were missing, and each broke the loop in a different place:

* **Training** hardcoded `AsyncTrainer` and required formatted JSON files on
  disk. GRPO reads the telemetry corpus itself through its own gate, so the
  file check would raise "No training data found" before any trainer ran.
* **Quantization** assumed the artifact was a full HF model. GRPO produces a
  PEFT adapter, and turning one into a full model needs the base in bf16 --
  ~57 GiB against a 47 GiB guest, which on this host is the desktop.
* **Deployment** copied files into a models directory. An adapter is not
  servable as a file: it must be layered over a base tag at load time, so
  that copy would "succeed" and serve nothing.

The through-line is that each stage ASSUMED what the one before it made.
Each now ASKS, and the question has a structural answer.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.orchestration.pipeline import (  # noqa: E402
    NightShiftPipeline,
    PipelineConfig,
    PipelineStage,
    PipelineState,
)
from reactor_core.quantization import adapter_gguf as ag  # noqa: E402
from reactor_core.training import trainer_strategy as ts  # noqa: E402


def _adapter_dir(tmp_path: Path, name: str = "adapter") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"\x00" * 32)
    return d


def _model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "full-model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00" * 32)
    return d


def _pipeline(tmp_path: Path, **cfg: Any) -> NightShiftPipeline:
    config = PipelineConfig(work_dir=tmp_path / "work",
                            output_dir=tmp_path / "out", **cfg)
    p = NightShiftPipeline(config)
    p._state = PipelineState(run_id="t", stage=PipelineStage.IDLE,
                             started_at=__import__("datetime").datetime.now())
    p._save_state = lambda: None            # type: ignore[method-assign]
    return p


# ---------------------------------------------------------------------------
# Classification: the question every later stage asks
# ---------------------------------------------------------------------------


def test_a_peft_adapter_is_recognised(tmp_path) -> None:
    assert ag.classify_artifact(_adapter_dir(tmp_path)) is ag.ArtifactKind.ADAPTER


def test_a_full_model_is_recognised(tmp_path) -> None:
    assert ag.classify_artifact(_model_dir(tmp_path)) is ag.ArtifactKind.MODEL


def test_adapter_markers_win_over_a_copied_base_config(tmp_path) -> None:
    """PEFT dirs often carry a config.json copied from the base. Checking
    model markers first would classify every adapter as a model and send it
    into a conversion that needs 57 GiB."""
    d = _adapter_dir(tmp_path)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\x00" * 8)
    assert ag.classify_artifact(d) is ag.ArtifactKind.ADAPTER


def test_half_an_adapter_is_unknown_not_an_adapter(tmp_path) -> None:
    """A run that died mid-save must stop here, not at the converter."""
    d = tmp_path / "partial"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert ag.classify_artifact(d) is ag.ArtifactKind.UNKNOWN


def test_a_gguf_file_and_a_missing_path(tmp_path) -> None:
    g = tmp_path / "x.gguf"
    g.write_bytes(b"GGUF")
    assert ag.classify_artifact(g) is ag.ArtifactKind.GGUF
    assert ag.classify_artifact(tmp_path / "nope") is ag.ArtifactKind.UNKNOWN
    assert ag.classify_artifact(None) is ag.ArtifactKind.UNKNOWN


def test_the_converter_argv_reads_only_the_adapter(tmp_path, monkeypatch) -> None:
    d = _adapter_dir(tmp_path)
    out = tmp_path / "o.gguf"
    argv = ag.build_convert_argv(d, out, converter=Path("/llama/convert_lora_to_gguf.py"))
    assert str(d) in argv and str(out) in argv
    assert "--outtype" in argv
    assert argv[argv.index("--outtype") + 1] == "f16", (
        "an adapter is a delta; quantising 27 MB saves nothing and costs precision")


def test_the_outtype_is_configurable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ag.ENV_ADAPTER_OUTTYPE, "q8_0")
    argv = ag.build_convert_argv(_adapter_dir(tmp_path), tmp_path / "o.gguf",
                                 converter=Path("/c.py"))
    assert argv[argv.index("--outtype") + 1] == "q8_0"


@pytest.mark.asyncio
async def test_converting_a_non_adapter_refuses(tmp_path) -> None:
    out = await ag.convert_adapter_to_gguf(_model_dir(tmp_path), tmp_path / "o.gguf")
    assert not out.success and "not a PEFT adapter" in (out.error or "")


@pytest.mark.asyncio
async def test_a_missing_converter_refuses_clearly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(ag.ENV_LLAMA_CPP_DIR, str(tmp_path / "absent"))
    out = await ag.convert_adapter_to_gguf(_adapter_dir(tmp_path), tmp_path / "o.gguf")
    assert not out.success and ag.ENV_LLAMA_CPP_DIR in (out.error or "")


@pytest.mark.asyncio
async def test_exit_zero_with_no_file_is_a_failure(tmp_path, monkeypatch) -> None:
    """The shape that would ship nothing while reporting success."""
    conv = tmp_path / "llama"
    conv.mkdir()
    (conv / "convert_lora_to_gguf.py").write_text("", encoding="utf-8")
    monkeypatch.setenv(ag.ENV_LLAMA_CPP_DIR, str(conv))

    class _P:
        async def _lines(self):
            for _ in ():
                yield b""

        def __init__(self):
            self.stdout = self._lines()

        async def wait(self):
            return 0

    async def _exec(*a, **k):
        return _P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    out = await ag.convert_adapter_to_gguf(_adapter_dir(tmp_path), tmp_path / "o.gguf")
    assert not out.success and "wrote no file" in (out.error or "")


# ---------------------------------------------------------------------------
# Join 1 — training reaches the resolved strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_non_sft_strategy_runs_without_files_on_disk(tmp_path, monkeypatch) -> None:
    """GRPO sources its own corpus. The file check must not run first."""
    saved = _adapter_dir(tmp_path, "run-1")

    async def _fake(request):
        return ts.TrainerOutcome(ok=True, strategy="grpo", adapter_path=saved,
                                 reason="saved")

    ts.register("grpo", _fake, replace=True)
    try:
        p = _pipeline(tmp_path, training_strategy="grpo")
        assert not (tmp_path / "work" / "formatted").exists()
        result = await p._run_training()
        assert result["success"] and result["strategy"] == "grpo"
        assert result["adapter_path"] == str(saved)
        assert result["model_path"] == str(saved), (
            "an adapter IS the artifact; claiming a merged model sends "
            "quantization into a 57 GiB merge")
        assert p._state.adapter_path == str(saved)
    finally:
        ts.register("grpo", ts.grpo_strategy, replace=True)


@pytest.mark.asyncio
async def test_a_refusal_is_recorded_apart_from_a_failure(tmp_path) -> None:
    async def _refused(request):
        return ts.TrainerOutcome(ok=False, strategy="grpo", refused=True,
                                 reason="GPU busy: 29.1 GiB held")

    ts.register("grpo", _refused, replace=True)
    try:
        p = _pipeline(tmp_path, training_strategy="grpo")
        with pytest.raises(RuntimeError, match="refused"):
            await p._run_training()
        assert p._state.training_refused is True
    finally:
        ts.register("grpo", ts.grpo_strategy, replace=True)


@pytest.mark.asyncio
async def test_a_failure_is_not_marked_as_a_refusal(tmp_path) -> None:
    async def _failed(request):
        return ts.TrainerOutcome(ok=False, strategy="grpo", reason="boom")

    ts.register("grpo", _failed, replace=True)
    try:
        p = _pipeline(tmp_path, training_strategy="grpo")
        with pytest.raises(RuntimeError, match="failed"):
            await p._run_training()
        assert p._state.training_refused is False
    finally:
        ts.register("grpo", ts.grpo_strategy, replace=True)


def test_the_strategy_comes_from_config_or_env(monkeypatch) -> None:
    monkeypatch.delenv("REACTOR_TRAINING_STRATEGY", raising=False)
    assert PipelineConfig().training_strategy == ""
    monkeypatch.setenv("REACTOR_TRAINING_STRATEGY", "grpo")
    assert PipelineConfig().training_strategy == "grpo"


# ---------------------------------------------------------------------------
# Join 2 — quantization converts what it was actually given
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_adapter_goes_through_the_adapter_converter(tmp_path, monkeypatch) -> None:
    adapter = _adapter_dir(tmp_path)
    produced = tmp_path / "out" / "adapter-adapter.gguf"
    seen = {}

    async def _convert(src, dst):
        seen["src"], seen["dst"] = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"GGUF" + b"\x00" * 16)
        return ag.ConversionOutcome(success=True, output_path=dst,
                                    quantized_size_mb=27.0,
                                    kind=ag.ArtifactKind.ADAPTER)

    monkeypatch.setattr(ag, "convert_adapter_to_gguf", _convert)
    p = _pipeline(tmp_path)
    p._state.model_path = str(adapter)
    out = await p._run_quantization()
    assert out == str(produced), out
    assert seen["src"] == adapter
    assert p._state.quantized_path == str(produced)


@pytest.mark.asyncio
async def test_a_failed_adapter_conversion_reports_empty(tmp_path, monkeypatch) -> None:
    async def _convert(src, dst):
        return ag.ConversionOutcome(success=False, error="converter exited 1",
                                    kind=ag.ArtifactKind.ADAPTER)

    monkeypatch.setattr(ag, "convert_adapter_to_gguf", _convert)
    p = _pipeline(tmp_path)
    p._state.model_path = str(_adapter_dir(tmp_path))
    assert await p._run_quantization() == ""
    assert p._state.quantized_path == ""


@pytest.mark.asyncio
async def test_an_already_gguf_artifact_is_passed_through(tmp_path) -> None:
    g = tmp_path / "already.gguf"
    g.write_bytes(b"GGUF")
    p = _pipeline(tmp_path)
    p._state.model_path = str(g)
    assert await p._run_quantization() == str(g)


@pytest.mark.asyncio
async def test_an_unrecognisable_artifact_refuses_to_guess(tmp_path) -> None:
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "readme.txt").write_text("hi", encoding="utf-8")
    p = _pipeline(tmp_path)
    p._state.model_path = str(junk)
    assert await p._run_quantization() == ""


@pytest.mark.asyncio
async def test_skip_quantization_still_wins(tmp_path) -> None:
    p = _pipeline(tmp_path, skip_quantization=True)
    p._state.model_path = str(_adapter_dir(tmp_path))
    assert await p._run_quantization() == ""


# ---------------------------------------------------------------------------
# Join 3 — an adapter is deployed by layering, never by copying
# ---------------------------------------------------------------------------


def _adapter_gguf(tmp_path: Path) -> Path:
    g = tmp_path / "run-adapter.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 16)
    return g


@pytest.mark.asyncio
async def test_an_adapter_is_published_through_the_deployer(tmp_path, monkeypatch) -> None:
    g = _adapter_gguf(tmp_path)
    seen = {}

    class _Deployer:
        def __init__(self, **kw):
            seen["ctor"] = kw

        async def deploy_adapter(self, path, *, base_tag=None, tag=None):
            seen["path"], seen["base"], seen["tag"] = Path(path), base_tag, tag
            return SimpleNamespace(ok=True, tag=tag or "jprime-adapter-latest",
                                   stage="complete", reason="live",
                                   summary=lambda: "ok")

    import reactor_core.deployment.ollama_deployer as dep
    monkeypatch.setattr(dep, "OllamaDeployer", _Deployer)

    p = _pipeline(tmp_path, deploy_via_ollama=True,
                  ollama_base_tag="qwen3-coder:30b",
                  ollama_adapter_tag="qwen3-coder-ov:30b",
                  require_gatekeeper=False)
    p._state.quantized_path = str(g)
    await p._run_deployment()
    assert seen["path"] == g
    assert seen["base"] == "qwen3-coder:30b"
    assert seen["tag"] == "qwen3-coder-ov:30b"
    assert p._state.deployed_tag == "qwen3-coder-ov:30b"


@pytest.mark.asyncio
async def test_a_failed_adapter_deploy_is_loud(tmp_path, monkeypatch) -> None:
    g = _adapter_gguf(tmp_path)

    class _Deployer:
        def __init__(self, **kw):
            pass

        async def deploy_adapter(self, path, *, base_tag=None, tag=None):
            return SimpleNamespace(ok=False, tag="t", stage="base_missing",
                                   reason="base tag is not served",
                                   summary=lambda: "no")

    import reactor_core.deployment.ollama_deployer as dep
    monkeypatch.setattr(dep, "OllamaDeployer", _Deployer)

    p = _pipeline(tmp_path, deploy_via_ollama=True, require_gatekeeper=False)
    p._state.quantized_path = str(g)
    with pytest.raises(RuntimeError, match="base_missing"):
        await p._run_deployment()
    assert not p._state.deployed_tag


@pytest.mark.asyncio
async def test_a_full_model_still_takes_the_file_path(tmp_path, monkeypatch) -> None:
    """The historical deployment is untouched for the artifact it was for."""
    g = tmp_path / "model-q4_k_m.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 16)
    models = tmp_path / "jprime-models"
    models.mkdir()
    monkeypatch.setenv("JPRIME_MODELS_DIR", str(models))

    called = {"deployer": False}

    class _Deployer:
        def __init__(self, **kw):
            called["deployer"] = True

        async def deploy_adapter(self, *a, **k):
            raise AssertionError("a full model must not take the adapter path")

    import reactor_core.deployment.ollama_deployer as dep
    monkeypatch.setattr(dep, "OllamaDeployer", _Deployer)

    p = _pipeline(tmp_path, require_gatekeeper=False)   # deploy_via_ollama off
    p._state.quantized_path = str(g)
    await p._run_deployment()
    assert not called["deployer"]
    assert (models / g.name).is_file(), "copied into the models directory as before"


@pytest.mark.asyncio
async def test_an_adapter_takes_the_ollama_path_even_without_the_flag(
    tmp_path, monkeypatch,
) -> None:
    """A config that forgot the flag must not silently copy an adapter into
    a directory where nothing can serve it."""
    g = _adapter_gguf(tmp_path)
    seen = {}

    class _Deployer:
        def __init__(self, **kw):
            pass

        async def deploy_adapter(self, path, *, base_tag=None, tag=None):
            seen["path"] = Path(path)
            return SimpleNamespace(ok=True, tag="t", stage="complete",
                                   reason="live", summary=lambda: "ok")

    import reactor_core.deployment.ollama_deployer as dep
    monkeypatch.setattr(dep, "OllamaDeployer", _Deployer)

    p = _pipeline(tmp_path, require_gatekeeper=False)
    p._state.adapter_path = str(tmp_path / "run-1")     # training said adapter
    p._state.quantized_path = str(g)
    await p._run_deployment()
    assert seen.get("path") == g


@pytest.mark.asyncio
async def test_the_gatekeeper_still_guards_the_adapter_path(tmp_path) -> None:
    p = _pipeline(tmp_path, deploy_via_ollama=True, require_gatekeeper=True)
    p._state.quantized_path = str(_adapter_gguf(tmp_path))
    p._state.gatekeeper_passed = False
    with pytest.raises(RuntimeError, match="[Gg]atekeeper"):
        await p._run_deployment()


# ---------------------------------------------------------------------------
# The loop, end to end
# ---------------------------------------------------------------------------


def test_state_carries_the_new_fields_across_a_resume(tmp_path) -> None:
    import datetime as _dt
    s = PipelineState(run_id="r", stage=PipelineStage.IDLE,
                      started_at=_dt.datetime.now())
    s.deployed_tag = "qwen3-coder-ov:30b"
    s.training_refused = True
    back = PipelineState.from_dict(s.to_dict())
    assert back.deployed_tag == "qwen3-coder-ov:30b"
    assert back.training_refused is True


def test_no_stage_names_a_trainer_or_assumes_an_artifact() -> None:
    import inspect
    train = inspect.getsource(NightShiftPipeline._run_training)
    assert "trainer_strategy" in train
    assert train.index("resolve_name") < train.index("distilled_dir"), (
        "the strategy must be resolved BEFORE the file-based data check")
    quant = inspect.getsource(NightShiftPipeline._run_quantization)
    assert "classify_artifact" in quant
    deploy = inspect.getsource(NightShiftPipeline._run_deployment)
    assert "_is_adapter_deploy" in deploy
