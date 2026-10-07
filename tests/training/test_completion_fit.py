"""The completion window is fitted to the card from measured slopes.

memory_guard is loaded by path: reactor_core/__init__ eagerly imports the
training stack, which these pure-arithmetic tests do not need.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "reactor_core" / "training" / "memory_guard.py"
_spec = importlib.util.spec_from_file_location("_mg_fit_under_test", _PATH)
mg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mg
_spec.loader.exec_module(mg)

GIB = 1024 ** 3
MB = 1024 ** 2          # the 2026-09-05 "2.296 MB/token" is MiB: 13.65 GiB / 6089 tokens


def cal(**kw):
    """Shaped like the 2026-09-05 measurements on the RTX 5090 / 30B NF4."""
    base = dict(cap_bytes=int(30.25 * GIB), resident_bytes=int(15.57 * GIB),
                optimizer_bytes=int(0.149 * GIB), grad_bytes=int(0.025 * GIB),
                gen_bytes_per_token=2.296 * MB, act_bytes_per_token=0.30 * MB,
                logit_bytes_per_token=1.2 * MB, decode_s_per_token=1.645,
                train_s_per_token=0.0023, num_generations=16, num_iterations=2,
                context_limit=262144)
    base.update(kw)
    return mg.MemoryCalibration(**base)


def test_the_measured_september_facts_are_reproduced():
    # ~6,100-token prompts left ~1 GiB: a window there must be small...
    tight = mg.fit_completion_window(cal(), prompt_tokens_max=6089, headroom_bytes=0)
    assert tight.binding == "generation" and 0 < tight.tokens < 600
    # ...and the 4,096-token budget the runner adopted leaves real room.
    roomy = mg.fit_completion_window(cal(), prompt_tokens_max=4096, headroom_bytes=0)
    assert roomy.tokens > tight.tokens + 1500


def test_both_peaks_stay_under_the_budget_at_the_fitted_window():
    c = cal()
    f = mg.fit_completion_window(c, prompt_tokens_max=4096, headroom_bytes=GIB)
    assert f.gen_peak_bytes <= f.budget_bytes and f.train_peak_bytes <= f.budget_bytes
    over = mg.fit_completion_window(
        mg.MemoryCalibration(**{**c.__dict__}), prompt_tokens_max=4096, headroom_bytes=GIB)
    assert over.tokens == f.tokens                       # deterministic


def test_the_training_peak_binds_when_logits_dominate():
    f = mg.fit_completion_window(cal(gen_bytes_per_token=0.1 * MB, logit_bytes_per_token=30 * MB),
                                 prompt_tokens_max=2048, headroom_bytes=0)
    assert f.binding == "training"


def test_the_model_context_binds_when_memory_is_plentiful():
    f = mg.fit_completion_window(cal(cap_bytes=1000 * GIB, context_limit=8192),
                                 prompt_tokens_max=6000, headroom_bytes=0)
    assert f.binding == "context" and f.tokens == 2192


def test_nothing_fits_is_a_refusal_not_a_guess():
    f = mg.fit_completion_window(cal(cap_bytes=int(16 * GIB)), prompt_tokens_max=4096,
                                 headroom_bytes=0, min_tokens=64)
    assert f.tokens == 0 and "no completion window fits" in f.reason


def test_more_headroom_means_a_smaller_window():
    a = mg.fit_completion_window(cal(), prompt_tokens_max=4096, headroom_bytes=0).tokens
    b = mg.fit_completion_window(cal(), prompt_tokens_max=4096, headroom_bytes=2 * GIB).tokens
    assert b < a


def test_steps_are_fitted_to_the_time_the_cycle_has():
    c = cal()
    steps, step_s = mg.fit_steps_to_time(c, completion_tokens=1500, prompt_tokens_mean=3500,
                                         accumulation=16, time_budget_s=7 * 3600, steps_per_epoch=400)
    assert step_s == pytest.approx(1.645 * 1500 / 2 + 0.0023 * 5000 * 16)
    assert steps == int(7 * 3600 // step_s) and steps < 400
    all_, _ = mg.fit_steps_to_time(c, completion_tokens=1500, prompt_tokens_mean=3500,
                                   accumulation=16, time_budget_s=0, steps_per_epoch=40)
    assert all_ == 40
    few, _ = mg.fit_steps_to_time(c, completion_tokens=1500, prompt_tokens_mean=3500,
                                  accumulation=16, time_budget_s=10_000_000, steps_per_epoch=40)
    assert few == 40                                      # never more than an epoch


def test_calibration_round_trips_through_its_json_shape():
    c = cal()
    assert mg.MemoryCalibration.from_dict({**c.__dict__, "extra": 1}) == c


# ---------------------------------------------------------------------------
# Decode cost grows with context: integrated, not single-point
# ---------------------------------------------------------------------------

def test_decode_integral_matches_the_per_token_sum():
    c = cal(decode_s_per_token=0.8, decode_ref_ctx=1044, decode_s_per_token_per_ctx=0.0001)
    brute = sum(0.8 + 0.0001 * (2978 + t - 1044) for t in range(1955))
    assert mg.decode_seconds(c, prompt_tokens=2978, completion_tokens=1955) == pytest.approx(brute)


def test_a_single_point_calibration_is_unchanged():
    c = cal(decode_s_per_token=0.8)                         # no ref ctx: no growth known
    assert mg.decode_seconds(c, prompt_tokens=2978, completion_tokens=1955) == pytest.approx(0.8 * 1955)
    legacy = mg.MemoryCalibration.from_dict({k: v for k, v in cal().__dict__.items()
                                             if k not in ("decode_ref_ctx", "decode_s_per_token_per_ctx")})
    assert legacy.decode_s_per_token_per_ctx == 0.0


def test_growth_shrinks_the_steps_that_fit_the_budget():
    """The 2026-10-07 shape: decode timed at ~1k context, rollouts at ~3k-5k."""
    flat = cal(decode_s_per_token=0.8, train_s_per_token=0.00048)
    grown = cal(decode_s_per_token=0.8, train_s_per_token=0.00048,
                decode_ref_ctx=1044, decode_s_per_token_per_ctx=0.00025)
    kw = dict(completion_tokens=1955, prompt_tokens_mean=2978.4, accumulation=16,
              time_budget_s=22500, steps_per_epoch=780)
    steps_flat, s_flat = mg.fit_steps_to_time(flat, **kw)
    steps_grown, s_grown = mg.fit_steps_to_time(grown, **kw)
    assert s_grown > s_flat and steps_grown < steps_flat
