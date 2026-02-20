"""
Tier-2 runtime orchestration for Reactor-Core training jobs.

This module activates advanced training capabilities that were previously
implemented but disconnected from the primary API execution path.

Activated capabilities:
- curriculum_learning
- meta_learning
- world_model_training
- causal_reasoning
- cognitive_modules
- online_learning
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, default)
    try:
        value = int(raw)
    except ValueError:
        return max(minimum, default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, default)
    try:
        value = float(raw)
    except ValueError:
        return max(minimum, default)
    return max(minimum, value)


@dataclass(frozen=True)
class Tier2RuntimeConfig:
    enabled: bool
    max_examples: int
    feature_dim: int
    action_dim: int
    world_model_epochs: int
    world_model_batch_size: int
    meta_task_batch: int
    module_flags: Dict[str, bool]


class _MetaDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self._features = torch.tensor(features, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.long)
        self.targets = labels.tolist()

    def __len__(self) -> int:
        return int(self._features.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._features[index], self._labels[index]


class _TransitionDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
    ):
        self._observations = torch.tensor(observations, dtype=torch.float32)
        self._actions = torch.tensor(actions, dtype=torch.float32)
        self._next_observations = torch.tensor(next_observations, dtype=torch.float32)
        self._rewards = torch.tensor(rewards, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._observations.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._observations[index],
            self._actions[index],
            self._next_observations[index],
            self._rewards[index],
        )


class _TinyOnlineModel(nn.Module):
    def __init__(self, vocab_size: int = 2048, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.head(pooled)

        if labels is not None:
            target = labels[:, 0].long().clamp(min=0, max=logits.shape[-1] - 1)
            loss = F.cross_entropy(logits, target, reduction="mean")
        else:
            loss = logits.mean() * 0.0

        return type("TinyModelOutput", (), {"loss": loss})()


class Tier2RuntimeOrchestrator:
    MODULES: Tuple[str, ...] = (
        "curriculum_learning",
        "meta_learning",
        "world_model_training",
        "causal_reasoning",
        "cognitive_modules",
        "online_learning",
    )

    def __init__(self, config: Tier2RuntimeConfig):
        self._config = config
        self._lock = asyncio.Lock()
        self._last_run: Optional[Dict[str, Any]] = None

    @classmethod
    def from_env(cls) -> "Tier2RuntimeOrchestrator":
        module_flags = {
            module: _env_bool(f"REACTOR_TIER2_{module.upper()}_ENABLED", True)
            for module in cls.MODULES
        }
        config = Tier2RuntimeConfig(
            enabled=_env_bool("REACTOR_TIER2_ENABLED", True),
            max_examples=_env_int("REACTOR_TIER2_MAX_EXAMPLES", 256, 8),
            feature_dim=_env_int("REACTOR_TIER2_FEATURE_DIM", 48, 16),
            action_dim=_env_int("REACTOR_TIER2_ACTION_DIM", 8, 2),
            world_model_epochs=_env_int("REACTOR_TIER2_WORLD_MODEL_EPOCHS", 1, 1),
            world_model_batch_size=_env_int("REACTOR_TIER2_WORLD_MODEL_BATCH_SIZE", 16, 2),
            meta_task_batch=_env_int("REACTOR_TIER2_META_TASK_BATCH", 3, 1),
            module_flags=module_flags,
        )
        return cls(config)

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "modules": dict(self._config.module_flags),
            "limits": {
                "max_examples": self._config.max_examples,
                "feature_dim": self._config.feature_dim,
                "action_dim": self._config.action_dim,
                "world_model_epochs": self._config.world_model_epochs,
                "world_model_batch_size": self._config.world_model_batch_size,
                "meta_task_batch": self._config.meta_task_batch,
            },
            "last_run": self._last_run,
        }

    async def run(
        self,
        *,
        job_id: str,
        experiences: Optional[Sequence[Dict[str, Any]]] = None,
        snapshot_path: Optional[Path] = None,
        work_dir: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        context = dict(context or {})
        config = self._apply_overrides(overrides or {})

        if not config.enabled:
            result = {
                "job_id": job_id,
                "status": "disabled",
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": 0.0,
                "context": context,
            }
            self._last_run = result
            return result

        async with self._lock:
            examples = self._load_examples(
                experiences=experiences,
                snapshot_path=snapshot_path,
                work_dir=work_dir,
                limit=config.max_examples,
            )

            if len(examples) < 6:
                result = {
                    "job_id": job_id,
                    "status": "skipped",
                    "reason": "insufficient_examples",
                    "example_count": len(examples),
                    "completed_at": datetime.now().isoformat(),
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "context": context,
                }
                self._last_run = result
                return result

            features = self._build_feature_matrix(examples, config.feature_dim)
            module_results: Dict[str, Any] = {}
            module_errors: Dict[str, str] = {}

            runners = [
                ("curriculum_learning", self._run_curriculum_learning),
                ("meta_learning", self._run_meta_learning),
                ("world_model_training", self._run_world_model_training),
                ("causal_reasoning", self._run_causal_reasoning),
                ("cognitive_modules", self._run_cognitive_modules),
                ("online_learning", self._run_online_learning),
            ]

            for module_name, runner in runners:
                if not config.module_flags.get(module_name, True):
                    module_results[module_name] = {"status": "disabled"}
                    continue

                try:
                    module_results[module_name] = await runner(
                        examples=examples,
                        features=features,
                        config=config,
                    )
                except Exception as exc:
                    module_errors[module_name] = str(exc)
                    logger.warning(
                        "[Tier2Runtime] %s failed for job %s: %s",
                        module_name,
                        job_id,
                        exc,
                    )
                    module_results[module_name] = {
                        "status": "error",
                        "error": str(exc),
                    }

            status = "ok" if not module_errors else "partial"
            result = {
                "job_id": job_id,
                "status": status,
                "example_count": len(examples),
                "modules": module_results,
                "errors": module_errors,
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "context": context,
            }
            self._last_run = result
            return result

    def _apply_overrides(self, overrides: Dict[str, Any]) -> Tier2RuntimeConfig:
        if not overrides:
            return self._config

        enabled = overrides.get("enabled", self._config.enabled)
        max_examples = overrides.get("max_examples", self._config.max_examples)
        feature_dim = overrides.get("feature_dim", self._config.feature_dim)
        action_dim = overrides.get("action_dim", self._config.action_dim)
        world_model_epochs = overrides.get("world_model_epochs", self._config.world_model_epochs)
        world_model_batch_size = overrides.get("world_model_batch_size", self._config.world_model_batch_size)
        meta_task_batch = overrides.get("meta_task_batch", self._config.meta_task_batch)

        module_overrides = overrides.get("modules", {})
        module_flags = dict(self._config.module_flags)
        if isinstance(module_overrides, dict):
            for key, value in module_overrides.items():
                if key in module_flags:
                    module_flags[key] = bool(value)

        return Tier2RuntimeConfig(
            enabled=bool(enabled),
            max_examples=max(8, int(max_examples)),
            feature_dim=max(16, int(feature_dim)),
            action_dim=max(2, int(action_dim)),
            world_model_epochs=max(1, int(world_model_epochs)),
            world_model_batch_size=max(2, int(world_model_batch_size)),
            meta_task_batch=max(1, int(meta_task_batch)),
            module_flags=module_flags,
        )

    def _load_examples(
        self,
        *,
        experiences: Optional[Sequence[Dict[str, Any]]],
        snapshot_path: Optional[Path],
        work_dir: Optional[Path],
        limit: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        if experiences:
            for item in experiences:
                normalized = self._normalize_example(item)
                if normalized:
                    candidates.append(normalized)

        if snapshot_path and snapshot_path.exists():
            candidates.extend(self._load_from_snapshot(snapshot_path))

        if work_dir and work_dir.exists():
            candidates.extend(self._load_from_work_dir(work_dir))

        # Stable de-duplication by content hash
        deduped: List[Dict[str, Any]] = []
        seen: set = set()
        for example in candidates:
            fingerprint = hashlib.sha256(
                f"{example['prompt']}||{example['response']}".encode("utf-8", errors="ignore")
            ).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(example)
            if len(deduped) >= limit:
                break

        return deduped

    def _load_from_snapshot(self, snapshot_path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            with snapshot_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    normalized = self._normalize_example(payload)
                    if normalized:
                        rows.append(normalized)
        except Exception as exc:
            logger.debug("[Tier2Runtime] Snapshot load failed (%s): %s", snapshot_path, exc)
        return rows

    def _load_from_work_dir(self, work_dir: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for subdir in ("distilled", "formatted", "jarvis_events", "scout_data"):
            candidate = work_dir / subdir
            if not candidate.exists():
                continue
            for json_file in candidate.glob("*.json"):
                try:
                    with json_file.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                except Exception:
                    continue

                # Distilled/formatted files use messages format.
                if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
                    prompt = ""
                    response = ""
                    for message in payload["messages"]:
                        if not isinstance(message, dict):
                            continue
                        role = str(message.get("role", "")).lower()
                        content = str(message.get("content", ""))
                        if role == "user":
                            prompt = content
                        elif role == "assistant":
                            response = content
                    normalized = self._normalize_example({"prompt": prompt, "response": response})
                else:
                    normalized = self._normalize_example(payload)

                if normalized:
                    rows.append(normalized)
        return rows

    def _normalize_example(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None

        prompt = (
            payload.get("prompt")
            or payload.get("user_input")
            or payload.get("query")
            or payload.get("input")
            or ""
        )
        response = (
            payload.get("response")
            or payload.get("assistant_output")
            or payload.get("output")
            or payload.get("answer")
            or payload.get("jarvis_response")
            or ""
        )

        prompt = str(prompt).strip()
        response = str(response).strip()
        if not prompt or not response:
            return None

        return {
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": dict(payload.get("metadata") or {}),
        }

    def _text_to_vector(self, text: str, dim: int) -> np.ndarray:
        vector = np.zeros(dim, dtype=np.float32)
        if not text:
            return vector
        encoded = text.encode("utf-8", errors="ignore")
        if not encoded:
            return vector

        for index, byte in enumerate(encoded):
            position = (index * 31 + byte) % dim
            vector[position] += (byte % 29 + 1) / 29.0

        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            vector /= norm
        return vector

    def _build_feature_matrix(self, examples: Sequence[Dict[str, Any]], dim: int) -> np.ndarray:
        features: List[np.ndarray] = []
        half = dim // 2
        for example in examples:
            prompt = example["prompt"]
            response = example["response"]
            prompt_vec = self._text_to_vector(prompt, half)
            response_vec = self._text_to_vector(response, dim - half)

            merged = np.concatenate([prompt_vec, response_vec], axis=0)
            length_signal = np.array(
                [
                    min(1.0, len(prompt) / 1024.0),
                    min(1.0, len(response) / 1024.0),
                    min(1.0, abs(len(response) - len(prompt)) / 1024.0),
                ],
                dtype=np.float32,
            )
            merged[: length_signal.shape[0]] = np.maximum(merged[: length_signal.shape[0]], length_signal)
            features.append(merged.astype(np.float32))

        return np.stack(features, axis=0)

    async def _run_curriculum_learning(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        features: np.ndarray,
        config: Tier2RuntimeConfig,
    ) -> Dict[str, Any]:
        from reactor_core.training.curriculum_learning import LengthDifficultyScorer, create_default_curriculum

        max_len = max(e["prompt_length"] + e["response_length"] for e in examples)
        scorer = LengthDifficultyScorer(max_length=max(32, int(max_len)))
        scores = [scorer.score(f"{e['prompt']} {e['response']}").value for e in examples]

        stages = create_default_curriculum(num_stages=3, total_epochs=3)
        stage_distribution: Dict[str, int] = {}
        for stage in stages:
            low, high = stage.difficulty_range
            count = sum(1 for score in scores if low <= score <= high)
            stage_distribution[stage.name] = count

        return {
            "status": "ok",
            "samples_scored": len(scores),
            "difficulty_mean": float(np.mean(scores)),
            "difficulty_std": float(np.std(scores)),
            "stage_distribution": stage_distribution,
        }

    async def _run_meta_learning(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        features: np.ndarray,
        config: Tier2RuntimeConfig,
    ) -> Dict[str, Any]:
        from reactor_core.training.meta_learning import MAMLConfig, MAMLTrainer, create_n_way_k_shot_task

        norms = np.linalg.norm(features, axis=1)
        quantiles = np.quantile(norms, [0.33, 0.66])
        labels = np.digitize(norms, quantiles).astype(np.int64)
        unique_labels, counts = np.unique(labels, return_counts=True)

        if len(unique_labels) < 2:
            return {"status": "skipped", "reason": "insufficient_label_diversity"}

        min_count = int(counts.min())
        if min_count < 2:
            return {"status": "skipped", "reason": "insufficient_class_examples"}

        n_way = int(min(3, len(unique_labels)))
        k_shot = int(max(1, min_count // 2))
        q_queries = int(max(1, min_count - k_shot))

        dataset = _MetaDataset(features, labels)
        model = nn.Sequential(
            nn.Linear(features.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, n_way),
        )
        trainer = MAMLTrainer(
            model=model,
            config=MAMLConfig(inner_lr=0.01, outer_lr=0.001, num_inner_steps=1, first_order=True),
        )

        losses: List[float] = []
        for _ in range(config.meta_task_batch):
            task = create_n_way_k_shot_task(
                dataset=dataset,
                n_way=n_way,
                k_shot=k_shot,
                q_queries=q_queries,
            )
            losses.append(float(trainer.meta_train_step([task])))

        return {
            "status": "ok",
            "tasks_executed": len(losses),
            "n_way": n_way,
            "k_shot": k_shot,
            "q_queries": q_queries,
            "meta_loss_mean": float(np.mean(losses)),
            "meta_loss_last": float(losses[-1]),
        }

    async def _run_world_model_training(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        features: np.ndarray,
        config: Tier2RuntimeConfig,
    ) -> Dict[str, Any]:
        from reactor_core.training.world_model_training import (
            CounterfactualReasoner,
            WorldModel,
            WorldModelConfig,
            WorldModelTrainer,
            WorldModelTrainingConfig,
        )

        if len(features) < 8:
            return {"status": "skipped", "reason": "insufficient_transitions"}

        observations = features[:-1]
        next_observations = features[1:]
        transitions = len(observations)

        action_dim = min(config.action_dim, features.shape[1])
        deltas = next_observations[:, :action_dim] - observations[:, :action_dim]
        rewards = np.array(
            [
                (examples[idx + 1]["response_length"] - examples[idx]["response_length"]) / 512.0
                for idx in range(transitions)
            ],
            dtype=np.float32,
        )

        dataset = _TransitionDataset(
            observations=observations,
            actions=deltas,
            next_observations=next_observations,
            rewards=rewards,
        )

        world_cfg = WorldModelConfig(
            observation_dim=features.shape[1],
            latent_dim=min(32, features.shape[1]),
            action_dim=action_dim,
            hidden_dims=[128, 64],
            deterministic_transition=False,
        )
        world_model = WorldModel(world_cfg)
        trainer_cfg = WorldModelTrainingConfig(
            num_epochs=config.world_model_epochs,
            batch_size=min(config.world_model_batch_size, len(dataset)),
            learning_rate=1e-4,
            device="cpu",
            checkpoint_dir=None,
        )
        trainer = WorldModelTrainer(world_model=world_model, config=trainer_cfg)
        losses = await trainer.train(dataset=dataset)

        last_total_loss = float(losses["total"][-1]) if losses["total"] else 0.0
        reasoner = CounterfactualReasoner(world_model)
        horizon = min(3, max(1, len(dataset)))
        base_obs = torch.tensor(observations[:1], dtype=torch.float32)
        factual_actions = torch.tensor(deltas[:horizon], dtype=torch.float32).unsqueeze(0)
        counterfactual_actions = torch.flip(factual_actions, dims=[1])
        counterfactual = reasoner.what_if(
            observation=base_obs,
            factual_actions=factual_actions,
            counterfactual_actions=counterfactual_actions,
            horizon=horizon,
        )

        improvement = float(counterfactual["improvement"].mean().item())
        return {
            "status": "ok",
            "transitions": len(dataset),
            "epochs": config.world_model_epochs,
            "loss_total_last": last_total_loss,
            "counterfactual_improvement_mean": improvement,
        }

    async def _run_causal_reasoning(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        features: np.ndarray,
        config: Tier2RuntimeConfig,
    ) -> Dict[str, Any]:
        from reactor_core.training.causal_reasoning import CausalDiscovery, CausalGraph, StructuralCausalModel

        prompt_lengths = np.array([e["prompt_length"] for e in examples], dtype=np.float64)
        response_lengths = np.array([e["response_length"] for e in examples], dtype=np.float64)
        diff_lengths = np.abs(response_lengths - prompt_lengths)
        token_proxy = np.array(
            [len(e["prompt"].split()) + len(e["response"].split()) for e in examples],
            dtype=np.float64,
        )

        def _ensure_variance(values: np.ndarray) -> np.ndarray:
            if np.std(values) > 0:
                return values
            offsets = np.linspace(0.0, 1e-3, num=len(values), endpoint=False)
            return values + offsets

        prompt_lengths = _ensure_variance(prompt_lengths)
        response_lengths = _ensure_variance(response_lengths)
        diff_lengths = _ensure_variance(diff_lengths)
        token_proxy = _ensure_variance(token_proxy)

        data = {
            "prompt_length": prompt_lengths,
            "response_length": response_lengths,
            "length_delta": diff_lengths,
            "token_proxy": token_proxy,
        }
        variable_names = list(data.keys())

        graph = CausalGraph()
        try:
            method = os.getenv("REACTOR_TIER2_CAUSAL_METHOD", "correlation").strip().lower()
            if method in {"pc", "ges", "notears"}:
                discovery = CausalDiscovery(method=method)
                graph = await discovery.discover(data=data, variable_names=variable_names)
            if not graph.edges:
                threshold = _env_float("REACTOR_TIER2_CAUSAL_CORR_THRESHOLD", 0.15, 0.01)
                for idx, source in enumerate(variable_names):
                    for target in variable_names[idx + 1:]:
                        corr = np.corrcoef(data[source], data[target])[0, 1]
                        if np.isnan(corr):
                            continue
                        if abs(corr) > threshold:
                            graph.add_edge(
                                source,
                                target,
                                strength=float(abs(corr)),
                                confidence=float(abs(corr)),
                            )
        except Exception as exc:
            logger.debug("[Tier2Runtime] Causal discovery fallback engaged: %s", exc)
            for variable in variable_names:
                graph.add_node(variable)
            for idx, source in enumerate(variable_names):
                for target in variable_names[idx + 1:]:
                    corr = np.corrcoef(data[source], data[target])[0, 1]
                    if np.isnan(corr):
                        continue
                    if abs(corr) > _env_float("REACTOR_TIER2_CAUSAL_CORR_THRESHOLD", 0.15, 0.01):
                        graph.add_edge(source, target, strength=float(abs(corr)), confidence=float(abs(corr)))

        scm = StructuralCausalModel(graph)
        for variable in graph.nodes:
            parents = graph.get_parents(variable)

            def _mechanism(parent_values: Dict[str, np.ndarray], noise: np.ndarray, _parents: List[str] = parents):
                if not _parents:
                    return noise
                parent_stack = np.vstack([parent_values[p] for p in _parents])
                return parent_stack.mean(axis=0) + noise * 0.05

            scm.set_mechanism(variable, _mechanism)

        intervention_var = variable_names[0]
        intervention_value = float(np.mean(data[intervention_var]) + 1.0)
        simulated = scm.do_calculus({intervention_var: intervention_value}, num_samples=64)
        baseline = float(np.mean(data[variable_names[-1]]))
        after = float(np.mean(simulated[variable_names[-1]]))

        return {
            "status": "ok",
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "intervention_variable": intervention_var,
            "intervention_shift": float(after - baseline),
        }

    async def _run_cognitive_modules(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        features: np.ndarray,
        config: Tier2RuntimeConfig,
    ) -> Dict[str, Any]:
        from reactor_core.training.cognitive_modules import (
            CognitiveModuleConfig,
            CognitiveModuleType,
            MemoryModule,
            PerceptionModule,
            PlanningModule,
            ReasoningModule,
        )

        input_dim = int(features.shape[1])
        batch_size = min(16, len(features))
        tensor_input = torch.tensor(features[:batch_size], dtype=torch.float32)

        hidden_dim = max(64, input_dim * 2)
        module_builders = {
            CognitiveModuleType.PLANNING: PlanningModule,
            CognitiveModuleType.REASONING: ReasoningModule,
            CognitiveModuleType.MEMORY: MemoryModule,
            CognitiveModuleType.PERCEPTION: PerceptionModule,
        }

        outputs: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for module_type, builder in module_builders.items():
                module_cfg = CognitiveModuleConfig(
                    module_type=module_type,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=input_dim,
                )
                module = builder(module_cfg)
                outputs[module_type.value] = module(tensor_input)

        norms = {name: float(output.norm(dim=-1).mean().item()) for name, output in outputs.items()}
        return {
            "status": "ok",
            "batch_size": batch_size,
            "active_modules": list(outputs.keys()),
            "output_norm_mean": norms,
        }

    async def _run_online_learning(
        self,
        *,
        examples: Sequence[Dict[str, Any]],
        features: np.ndarray,
        config: Tier2RuntimeConfig,
    ) -> Dict[str, Any]:
        from reactor_core.training.online_learning import FeedbackType, create_online_learning_engine

        model = _TinyOnlineModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        engine = create_online_learning_engine(
            model=model,
            optimizer=optimizer,
            buffer_size=max(256, len(examples) * 4),
            batch_size=max(4, min(16, len(examples) // 2)),
            update_frequency=max(2, min(8, len(examples) // 4)),
            tokenizer_name=os.getenv("REACTOR_TIER2_TOKENIZER", "gpt2"),
            enable_backpressure=True,
            device=torch.device("cpu"),
        )

        async def _tokenize(texts: List[str], device: Optional[torch.device] = None):
            vocab_size = 2048
            max_len = 48
            dev = device or torch.device("cpu")
            input_ids = torch.zeros((len(texts), max_len), dtype=torch.long, device=dev)
            attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long, device=dev)
            for row, text in enumerate(texts):
                encoded = text.encode("utf-8", errors="ignore")[:max_len]
                if encoded:
                    token_ids = torch.tensor([byte % vocab_size for byte in encoded], dtype=torch.long, device=dev)
                    input_ids[row, : len(token_ids)] = token_ids
                    attention_mask[row, : len(token_ids)] = 1
                else:
                    attention_mask[row, 0] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Use deterministic local tokenization to avoid dependency-specific
        # tokenizer behavior and keep online updates gradient-valid.
        engine._tokenizer.tokenize = _tokenize  # type: ignore[attr-defined]

        async def _compute_losses_patch(self, experiences_batch, weights):
            input_texts = [exp.input_text for exp in experiences_batch]
            target_texts = [exp.target_text if exp.target_text else exp.output_text for exp in experiences_batch]
            encoded_inputs = await _tokenize(input_texts, device=self.device)
            encoded_targets = await _tokenize(target_texts, device=self.device)

            outputs = self.model(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                labels=encoded_targets["input_ids"],
            )
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            return torch.ones(len(experiences_batch), device=self.device) * loss.detach()

        engine._compute_losses = _compute_losses_patch.__get__(engine, type(engine))

        accepted = 0
        for example in examples:
            reward = min(1.0, example["response_length"] / 512.0)
            feedback_type = (
                FeedbackType.EXPLICIT_POSITIVE
                if example["response_length"] >= max(8, example["prompt_length"] // 3)
                else FeedbackType.IMPLICIT_NEGATIVE
            )
            added = await engine.add_experience(
                input_text=example["prompt"],
                output_text=example["response"],
                feedback_type=feedback_type,
                reward=reward,
                confidence=reward,
                source="tier2_runtime",
                drop_if_busy=False,
            )
            accepted += int(bool(added))

        stats = engine.get_statistics()
        metrics = stats.get("metrics", {})
        return {
            "status": "ok",
            "accepted_experiences": accepted,
            "buffer_size": stats.get("buffer", {}).get("current_size", 0),
            "total_updates": metrics.get("total_updates", 0),
            "avg_loss": metrics.get("avg_loss", 0.0),
            "drift_detected": metrics.get("drift_detected", False),
        }
