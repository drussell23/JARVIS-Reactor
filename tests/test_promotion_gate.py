"""An adapter takes the live tag only if something MEASURED says it should.

Without this gate, a night-shift cycle that trains and deploys is a machine
that replaces the model O+V generates with every night on the strength of
having produced a file. Loss going down is not evidence: GRPO's loss is
dominated by the router auxiliary term, and a run can complete, save and
convert while having learned nothing the pipeline can see.

The gate distinguishes two refusals, and the distinction is the point:
  * HOLD          — it compared, and the candidate is worse.
  * CANNOT ANSWER — it could not compare at all.
They need opposite fixes (retrain vs. go measure), so a scheduler that
collapsed them would send the operator to the wrong place.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.deployment import promotion_gate as pg  # noqa: E402

BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def _record(**kw) -> pg.BaselineRecord:
    kw.setdefault("score", 10.0)
    kw.setdefault("metric", "overall_score")
    kw.setdefault("base_model", BASE)
    kw.setdefault("measured_at", datetime.now(tz=timezone.utc).isoformat())
    return pg.BaselineRecord(**kw)


def _verdict(tmp_path, *, score, baseline=None, metric="overall_score",
             base_model=BASE, now=None):
    path = tmp_path / "baseline.json"
    if baseline is not None:
        pg.save_baseline(baseline, path=path)
    return pg.evaluate_promotion(
        candidate_score=score, candidate_metric=metric,
        base_model=base_model, path=path, now=now,
    )


# ---------------------------------------------------------------------------
# It refuses when it cannot answer
# ---------------------------------------------------------------------------


def test_no_baseline_refuses_and_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(pg.ENV_FORCE_PROMOTE, raising=False)
    v = _verdict(tmp_path, score=12.0)
    assert not v.promote and v.unanswerable
    assert "no usable baseline" in v.reason
    assert "not" in v.reason and "regression" in v.reason


def test_an_unparseable_baseline_refuses(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(pg.ENV_FORCE_PROMOTE, raising=False)
    p = tmp_path / "baseline.json"
    p.write_text("{ truncated", encoding="utf-8")
    v = pg.evaluate_promotion(candidate_score=12.0, candidate_metric="overall_score",
                              base_model=BASE, path=p)
    assert not v.promote and v.unanswerable


def test_an_incomplete_record_refuses(tmp_path, monkeypatch) -> None:
    """Existence is not validity: a record missing a field has unknown
    provenance, and stat() cannot tell the difference."""
    monkeypatch.delenv(pg.ENV_FORCE_PROMOTE, raising=False)
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"score": 10.0}), encoding="utf-8")
    assert pg.load_baseline(path=p) is None
    v = pg.evaluate_promotion(candidate_score=12.0, candidate_metric="overall_score",
                              base_model=BASE, path=p)
    assert not v.promote and v.unanswerable


def test_a_stale_baseline_refuses(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(pg.ENV_BASELINE_MAX_AGE_H, "24")
    old = (datetime.now(tz=timezone.utc) - timedelta(hours=72)).isoformat()
    v = _verdict(tmp_path, score=99.0, baseline=_record(measured_at=old))
    assert not v.promote and v.unanswerable
    assert "old" in v.reason and "harness has moved" in v.reason


def test_a_baseline_for_a_different_base_model_refuses(tmp_path) -> None:
    v = _verdict(tmp_path, score=99.0,
                 baseline=_record(base_model="Qwen/Qwen2.5-Coder-32B"))
    assert not v.promote and v.unanswerable
    assert "not a control" in v.reason


def test_a_different_metric_refuses(tmp_path) -> None:
    v = _verdict(tmp_path, score=99.0, metric="apply_count",
                 baseline=_record(metric="overall_score"))
    assert not v.promote and v.unanswerable
    assert "unlike things" in v.reason


def test_a_candidate_with_no_score_refuses(tmp_path) -> None:
    v = _verdict(tmp_path, score=None, baseline=_record())
    assert not v.promote and v.unanswerable
    assert "evaluation did not run" in v.reason


# ---------------------------------------------------------------------------
# It answers when it can
# ---------------------------------------------------------------------------


def test_better_than_the_control_promotes(tmp_path) -> None:
    v = _verdict(tmp_path, score=12.0, baseline=_record(score=10.0))
    assert v.promote and not v.unanswerable
    assert v.margin == pytest.approx(2.0)


def test_equal_to_the_control_promotes(tmp_path) -> None:
    assert _verdict(tmp_path, score=10.0, baseline=_record(score=10.0)).promote


def test_worse_than_the_control_HOLDS_and_is_not_unanswerable(tmp_path) -> None:
    """A regression and a missing measurement need opposite fixes."""
    v = _verdict(tmp_path, score=8.0, baseline=_record(score=10.0))
    assert not v.promote
    assert not v.unanswerable, "it compared; that is an answer"
    assert "regression" in v.reason
    assert v.margin == pytest.approx(-2.0)


def test_tolerance_is_configurable_and_defaults_to_no_regression(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.delenv(pg.ENV_REGRESSION_TOLERANCE, raising=False)
    assert not _verdict(tmp_path, score=9.9, baseline=_record(score=10.0)).promote
    monkeypatch.setenv(pg.ENV_REGRESSION_TOLERANCE, "0.05")
    assert _verdict(tmp_path, score=9.6, baseline=_record(score=10.0)).promote
    assert not _verdict(tmp_path, score=9.4, baseline=_record(score=10.0)).promote


def test_the_operator_override_is_explicit_and_named_so(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(pg.ENV_FORCE_PROMOTE, "1")
    v = _verdict(tmp_path, score=12.0)
    assert v.promote
    assert pg.ENV_FORCE_PROMOTE in v.reason, "the record must say it was forced"
    assert "PROMOTE_WITHOUT_BASELINE" in pg.ENV_FORCE_PROMOTE, (
        "the name must be unmistakable in a config file")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_baseline_round_trips(tmp_path) -> None:
    rec = _record(score=7.5, session_id="bt-2026-09-05", harness="devtest")
    p = pg.save_baseline(rec, path=tmp_path / "b.json")
    back = pg.load_baseline(path=p)
    assert back is not None
    assert back.score == 7.5 and back.session_id == "bt-2026-09-05"
    assert back.base_model == BASE


def test_the_write_is_atomic(tmp_path) -> None:
    """A half-written baseline that still parses is worse than none."""
    import inspect
    assert "os.replace" in inspect.getsource(pg.save_baseline)


def test_the_path_is_configurable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(pg.ENV_BASELINE_PATH, str(tmp_path / "custom.json"))
    assert pg.baseline_path() == tmp_path / "custom.json"


def test_the_summary_says_which_kind_of_no(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(pg.ENV_FORCE_PROMOTE, raising=False)
    assert "CANNOT ANSWER" in _verdict(tmp_path, score=1.0).summary()
    assert "HOLD" in _verdict(tmp_path, score=8.0,
                              baseline=_record(score=10.0)).summary()
    assert "PROMOTE" in _verdict(tmp_path, score=12.0,
                                 baseline=_record(score=10.0)).summary()


# ---------------------------------------------------------------------------
# Wiring: the deploy stage consults it
# ---------------------------------------------------------------------------


def test_the_deploy_stage_gates_on_it() -> None:
    import inspect
    from reactor_core.orchestration.pipeline import NightShiftPipeline, PipelineConfig
    src = inspect.getsource(NightShiftPipeline._deploy_adapter_to_ollama)
    assert "_promotion_verdict()" in src
    assert "promotion refused" in src
    gate = inspect.getsource(NightShiftPipeline._promotion_verdict)
    assert "evaluate_promotion" in gate
    assert PipelineConfig().require_promotion_baseline is True, (
        "guarded by default: an unguarded cycle replaces the model nightly "
        "on the strength of having produced a file")


@pytest.mark.asyncio
async def test_an_adapter_deploy_is_blocked_without_a_baseline(
    tmp_path, monkeypatch,
) -> None:
    import datetime as _dt
    from reactor_core.orchestration.pipeline import (
        NightShiftPipeline, PipelineConfig, PipelineStage, PipelineState,
    )
    monkeypatch.delenv(pg.ENV_FORCE_PROMOTE, raising=False)
    monkeypatch.setenv(pg.ENV_BASELINE_PATH, str(tmp_path / "absent.json"))

    g = tmp_path / "run-adapter.gguf"
    g.write_bytes(b"GGUF" + b"\x00" * 16)

    class _Deployer:
        def __init__(self, **kw):
            raise AssertionError("the deployer must never be reached")

    import reactor_core.deployment.ollama_deployer as dep
    monkeypatch.setattr(dep, "OllamaDeployer", _Deployer)

    p = NightShiftPipeline(PipelineConfig(
        work_dir=tmp_path / "w", output_dir=tmp_path / "o",
        deploy_via_ollama=True, require_gatekeeper=False))
    p._state = PipelineState(run_id="t", stage=PipelineStage.IDLE,
                             started_at=_dt.datetime.now())
    p._save_state = lambda: None            # type: ignore[method-assign]
    p._state.quantized_path = str(g)

    with pytest.raises(RuntimeError, match="promotion refused"):
        await p._run_deployment()
    assert not p._state.deployed_tag
