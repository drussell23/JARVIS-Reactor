"""Rollouts sample with the KV cache: eval mode at the one generation seam.

transformers drops the cache in any layer that is gradient-checkpointed AND
training; TRL generates from inside training_step. Live 2026-10-07 a step at
a 512-token window took 1422 s (~2.8 s/token vs 0.80 s cached).
"""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")

from reactor_core.training import grpo_pipeline as gp  # noqa: E402


def test_rollout_mode_samples_in_eval_and_restores_train():
    m = torch.nn.Linear(2, 2)
    m.train()
    with gp.rollout_mode(m):
        assert not m.training
    assert m.training


def test_rollout_mode_restores_even_when_generation_raises():
    m = torch.nn.Linear(2, 2)
    m.train()
    with pytest.raises(RuntimeError):
        with gp.rollout_mode(m):
            raise RuntimeError("oom")
    assert m.training


def test_rollout_mode_leaves_an_eval_model_in_eval():
    m = torch.nn.Linear(2, 2)
    m.eval()
    with gp.rollout_mode(m):
        pass
    assert not m.training


def test_the_trainer_generates_in_eval_mode(monkeypatch):
    from trl import GRPOTrainer
    seen = {}

    def fake(self, *a, **kw):
        seen["training"] = self.model.training
        return "ids"

    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", fake)
    cls = gp.rollout_trainer_class()
    trainer = object.__new__(cls)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.model.train()
    assert trainer._generate_single_turn([[1, 2]], None, None) == "ids"
    assert seen["training"] is False and trainer.model.training


def test_build_trainer_uses_the_rollout_trainer():
    src = inspect.getsource(gp.build_trainer)
    assert "rollout_trainer_class()(" in src and "GRPOTrainer(" not in src.replace("rollout_trainer_class", "")
