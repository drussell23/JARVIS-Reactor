"""DPO pair generation over single-model self-improvement telemetry.

Two defects made this module produce nothing from a real corpus:

  1. ``_group_by_prompt`` assigned ``_original_prompt`` onto a plain
     ``list``. ``list`` has no ``__dict__``, so the assignment raised
     ``AttributeError`` for EVERY corpus containing two or more usable
     interactions -- i.e. every corpus that could produce a pair.
  2. ``_generate_pairs_from_group`` skipped any two responses from the
     same ``model_id``, even when one succeeded and the other failed.
     A single local model generating a good patch and a broken one for
     the same prompt is exactly the self-improvement signal, and it was
     discarded wholesale.

``dpo_pair_generator`` is loaded by path: ``reactor_core/__init__``
eagerly imports torch/peft/trl, which this stdlib-only module and these
tests do not need.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

_DPO_PATH = (
    Path(__file__).resolve().parents[2]
    / "reactor_core" / "training" / "dpo_pair_generator.py"
)


def _load_dpo() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_dpo_under_test", _DPO_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dpo = _load_dpo()

_PROMPT = "repair _should_use_lean_prompt in providers.py"
_GOOD = "def should_use_lean_prompt(ctx):\n    return bool(ctx.lean)\n"
_BAD = "def should_use_lean_prompt(ctx:\n    return\n"


def _event(
    *,
    model_id: str,
    outcome: str,
    body: str,
    confidence: float,
    prompt: str = _PROMPT,
) -> dict:
    return {
        "event_id": f"e-{abs(hash(body)) % 10_000}",
        "schema_version": "1.0",
        "event_type": "interaction",
        "source": "jarvis_body",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": prompt,
        "assistant_output": body,
        "outcome": outcome,
        "confidence": confidence,
        "model_id": model_id,
        "latency_ms": 8000.0,
        "tokens_used": 1000,
        "task_type": "code_debug",
        "metadata": {"should_train": True},
    }


def _write(tmp_path: Path, *events: dict) -> Path:
    d = tmp_path / "events"
    d.mkdir(exist_ok=True)
    (d / "experience.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return d


def _generator(events_dir: Path):
    return dpo.DPOPairGenerator(dpo.DPOConfig(telemetry_dir=events_dir))


# ---------------------------------------------------------------------------
# 1. The crash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_interactions_do_not_raise(tmp_path: Path) -> None:
    """Regression: any corpus with 2+ usable rows used to raise
    AttributeError inside _group_by_prompt."""
    events_dir = _write(
        tmp_path,
        _event(model_id="m-a", outcome="success", body=_GOOD, confidence=1.0),
        _event(model_id="m-b", outcome="failure", body=_BAD, confidence=0.0),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()
    assert pairs, "expected at least one pair, got none"


@pytest.mark.asyncio
async def test_prompt_survives_onto_the_pair(tmp_path: Path) -> None:
    """The original (un-normalized) prompt must reach DPOPair.prompt —
    it is the training input."""
    events_dir = _write(
        tmp_path,
        _event(model_id="m-a", outcome="success", body=_GOOD, confidence=1.0),
        _event(model_id="m-b", outcome="failure", body=_BAD, confidence=0.0),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()
    assert pairs[0].prompt == _PROMPT


# ---------------------------------------------------------------------------
# 2. Single-model self-improvement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_model_opposite_outcomes_yields_a_pair(
    tmp_path: Path,
) -> None:
    """The O+V shape: ONE local model, one good patch, one broken."""
    model = "qwen2.5-coder:32b"
    events_dir = _write(
        tmp_path,
        _event(model_id=model, outcome="success", body=_GOOD, confidence=1.0),
        _event(model_id=model, outcome="failure", body=_BAD, confidence=0.0),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.chosen == _GOOD
    assert pair.rejected == _BAD
    assert pair.chosen_model == pair.rejected_model == model
    assert pair.generation_method == "outcome_diff"


@pytest.mark.asyncio
async def test_same_model_same_outcome_is_still_skipped(
    tmp_path: Path,
) -> None:
    """Without an outcome difference the ranking is confidence/latency
    noise, not a preference — that guard is kept."""
    model = "qwen2.5-coder:32b"
    events_dir = _write(
        tmp_path,
        _event(model_id=model, outcome="success", body=_GOOD, confidence=1.0),
        _event(
            model_id=model, outcome="success",
            body="def other():\n    return 2\n", confidence=0.1,
        ),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()
    assert pairs == []


@pytest.mark.asyncio
async def test_cross_model_pairing_still_works(tmp_path: Path) -> None:
    """No regression for the multi-model case the module was built for."""
    events_dir = _write(
        tmp_path,
        _event(
            model_id="qwen2.5-coder:32b", outcome="success",
            body=_GOOD, confidence=1.0,
        ),
        _event(
            model_id="mistral-7b", outcome="failure",
            body=_BAD, confidence=0.0,
        ),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()
    assert len(pairs) == 1
    assert pairs[0].chosen_model == "qwen2.5-coder:32b"
    assert pairs[0].rejected_model == "mistral-7b"


@pytest.mark.asyncio
async def test_distinct_prompts_never_pair(tmp_path: Path) -> None:
    model = "qwen2.5-coder:32b"
    events_dir = _write(
        tmp_path,
        _event(model_id=model, outcome="success", body=_GOOD, confidence=1.0),
        _event(
            model_id=model, outcome="failure", body=_BAD, confidence=0.0,
            prompt="a completely different task",
        ),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()
    assert pairs == []


@pytest.mark.asyncio
async def test_identical_responses_never_pair(tmp_path: Path) -> None:
    model = "qwen2.5-coder:32b"
    events_dir = _write(
        tmp_path,
        _event(model_id=model, outcome="success", body=_GOOD, confidence=1.0),
        _event(model_id=model, outcome="failure", body=_GOOD, confidence=0.0),
    )
    pairs = await _generator(events_dir).generate_from_telemetry()
    assert pairs == []
