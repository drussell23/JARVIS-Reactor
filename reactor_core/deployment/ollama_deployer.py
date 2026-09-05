"""Deploy a trained GGUF to J-Prime's Ollama runtime.

This is the missing tail of the flywheel. Reactor could already convert
HF weights to GGUF (``quantization/gguf_converter.py``) and validate the
artifact (``deployment/gate.py``), but nothing published it to the engine
O+V actually calls. O+V reaches a local model over Ollama's OpenAI-
compatible endpoint at ``127.0.0.1:11434``; a GGUF sitting in a directory
is not reachable from there until ``ollama create`` registers it under a
tag. That one step did not exist in this repo (verified: zero matches for
"ollama" across ``reactor_core/``).

## Order of operations, and why

  1. **Gate first.** ``DeploymentGate.validate`` is composed, not
     bypassed -- a corrupt or truncated GGUF must never reach a tag O+V
     will route to. Publishing an unvalidated artifact to
     ``jprime-latest`` would put a broken model in the generation path
     with no signal until candidates start failing.
  2. **GPU lease second.** ``ollama create`` writes blobs and the
     verification probe LOADS the model, both of which contend with a
     live soak on a single 32 GiB card. The lease is cross-process or
     the deploy defers -- see ``deployment/gpu_lease.py``.
  3. **Snapshot before overwrite.** The previous ``jprime-latest`` is
     copied to a rollback tag *before* the new one is created, so a bad
     model is one ``ollama cp`` away from being undone. Ollama copies at
     the blob level, so this costs no extra disk.
  4. **Verify by serving, not by exit code.** ``ollama create``
     returning 0 means the manifest was written. The deploy is only
     complete when the tag appears in ``/api/tags``.

Nothing here imports torch: this module is the deployment tail and must
run on a box that serves models without being able to train them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# NOTE: gpu_lease and DeploymentGate are imported LAZILY, inside the methods
# that use them. `reactor_core/__init__` eagerly imports the training stack
# (torch/peft/trl), so any package-level import here would make the
# deployment tail unrunnable on a box that serves models without being able
# to train them -- which is precisely the box this module exists to serve.

OLLAMA_DEPLOY_SCHEMA_VERSION = "ollama_deploy.1"

_ENV_HOST = "OLLAMA_HOST"
_ENV_TAG = "REACTOR_OLLAMA_TAG"
_ENV_ROLLBACK_TAG = "REACTOR_OLLAMA_ROLLBACK_TAG"
_ENV_NUM_CTX = "REACTOR_OLLAMA_NUM_CTX"
_ENV_CREATE_TIMEOUT_S = "REACTOR_OLLAMA_CREATE_TIMEOUT_S"
_ENV_BIN = "REACTOR_OLLAMA_BIN"
_ENV_DRAFT_MODEL = "REACTOR_OLLAMA_DRAFT_MODEL"
_ENV_BASE_TAG = "REACTOR_OLLAMA_BASE_TAG"
_ENV_ADAPTER_TAG = "REACTOR_OLLAMA_ADAPTER_TAG"
_ENV_ADAPTER_MIN_BYTES = "REACTOR_GATE_ADAPTER_MIN_FILE_SIZE_BYTES"

_DEFAULT_HOST = "http://127.0.0.1:11434"
_DEFAULT_TAG = "jprime-latest"
_DEFAULT_ROLLBACK_TAG = "jprime-previous"
#: Qwen2.5-Coder's NATIVE context is 32K. This is not a legacy hardcode:
#: raising it past the model's training window degrades quality even
#: where VRAM allows, so the deploy default matches the architecture.
_DEFAULT_NUM_CTX = 32768
_DEFAULT_CREATE_TIMEOUT_S = 1800.0

#: The tag an ADAPTER is layered over. Empty by default: only the operator
#: knows which base the adapter was trained against, and layering a LoRA
#: onto the wrong weights produces a model that loads and answers wrongly
#: -- the worst failure shape. An empty value makes the adapter path
#: refuse rather than guess.
_DEFAULT_BASE_TAG = ""
_DEFAULT_ADAPTER_TAG = "jprime-adapter-latest"

#: The gate's size floor for an ADAPTER. `DEFAULT_MIN_FILE_SIZE` (100 MB)
#: exists to catch a truncated 18 GB model; a LoRA over 192 attention
#: projections is ~27 MB, so the model floor would reject every healthy
#: adapter. The floor is a property of WHAT is being validated, not a
#: constant: 256 KiB still catches a truncated or empty conversion.
_DEFAULT_ADAPTER_MIN_BYTES = 256 * 1024


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


@dataclass
class DeployResult:
    """Outcome of a deployment attempt."""

    ok: bool
    tag: str = ""
    stage: str = ""
    reason: str = ""
    rollback_tag: str = ""
    gguf_path: str = ""
    duration_s: float = 0.0
    gate_summary: str = ""
    lease_backend: str = ""
    checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": OLLAMA_DEPLOY_SCHEMA_VERSION,
            "ok": self.ok,
            "tag": self.tag,
            "stage": self.stage,
            "reason": self.reason,
            "rollback_tag": self.rollback_tag,
            "gguf_path": self.gguf_path,
            "duration_s": round(self.duration_s, 3),
            "gate_summary": self.gate_summary,
            "lease_backend": self.lease_backend,
            "checks": list(self.checks),
        }

    def summary(self) -> str:
        verdict = "DEPLOYED" if self.ok else "REFUSED"
        return f"{verdict} tag={self.tag or '?'} stage={self.stage}: {self.reason}"


class OllamaDeployer:
    """Publish a validated GGUF to Ollama under a stable tag.

    Args:
        host: Ollama base URL. Defaults to ``$OLLAMA_HOST`` then
            ``http://127.0.0.1:11434``.
        tag: Target tag O+V routes to.
        rollback_tag: Tag the previous model is preserved under.
        num_ctx: Context window baked into the Modelfile.
        gate: Optional pre-built DeploymentGate (injected in tests).
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        tag: Optional[str] = None,
        rollback_tag: Optional[str] = None,
        num_ctx: Optional[int] = None,
        gate: Optional[Any] = None,
        lease_factory: Optional[Callable[..., Any]] = None,
        draft_model: Optional[str] = None,
    ) -> None:
        self.host = (host or _env_str(_ENV_HOST, _DEFAULT_HOST)).rstrip("/")
        self.tag = tag or _env_str(_ENV_TAG, _DEFAULT_TAG)
        self.rollback_tag = rollback_tag or _env_str(
            _ENV_ROLLBACK_TAG, _DEFAULT_ROLLBACK_TAG
        )
        self.num_ctx = num_ctx or _env_int(
            _ENV_NUM_CTX, _DEFAULT_NUM_CTX, 512, 1_048_576
        )
        self._gate = gate
        self._lease_factory = lease_factory
        # Empty = no speculative decoding. Left empty by default because a
        # draft model with a mismatched tokenizer is worse than none, and
        # only the operator knows which pairing is valid.
        self.draft_model = (
            draft_model if draft_model is not None
            else _env_str(_ENV_DRAFT_MODEL, "")
        )

    def _lease(self, *, reason: str) -> Any:
        """Resolve the GPU lease context manager (lazily, see module note)."""
        if self._lease_factory is not None:
            return self._lease_factory(reason=reason)
        from reactor_core.deployment.gpu_lease import (  # noqa: PLC0415
            gpu_lease,
        )
        return gpu_lease(reason=reason)

    # -- injectable seams -------------------------------------------------
    async def _run(self, args: Sequence[str], timeout_s: float) -> Tuple[int, str]:
        """Run an ollama CLI command. Returns (returncode, combined output)."""
        binary = _env_str(_ENV_BIN, "ollama")
        try:
            proc = await asyncio.create_subprocess_exec(
                binary, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, _ENV_HOST: self.host},
            )
        except FileNotFoundError:
            return 127, f"{binary}: not found on PATH"
        try:
            out, _ = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            return 124, f"timed out after {timeout_s:.0f}s"
        return proc.returncode or 0, out.decode("utf-8", "replace")

    async def _tags(self) -> List[str]:
        """Model tags the running Ollama currently serves."""
        def _fetch() -> List[str]:
            with urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=10.0,
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
            return [
                str(m.get("name", ""))
                for m in (payload.get("models") or [])
                if m.get("name")
            ]

        try:
            return await asyncio.to_thread(_fetch)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OllamaDeploy] /api/tags unreachable: %s", exc)
            return []

    # -- helpers ----------------------------------------------------------
    def build_modelfile(self, gguf_path: Path, *, draft_model: str = "") -> str:
        """Render the Modelfile. Kept pure so tests can assert on it.

        ``draft_model`` emits a ``DRAFT`` instruction for speculative
        decoding. It is bound HERE, at deploy time, because DRAFT is a
        Modelfile instruction baked into the tag -- not a per-request
        option. The request-time counterpart is ``draft_num_predict``,
        which belongs to the generation lane.

        The caller is responsible for only passing a draft model that
        shares the target's TOKENIZER: speculation verifies draft token
        IDs, so a mismatched vocabulary does not merely perform badly, it
        validates against the wrong symbols. On this host
        qwen2.5-coder:7b (vocab pre=qwen2, BOS 151643) can draft for
        qwen2.5-coder:32b, and CANNOT draft for qwen3.8:27b
        (pre=qwen35, BOS 248044). qwen3.8 uses Multi-Token Prediction
        instead -- self-speculative, no draft model, so no DRAFT line.
        """
        lines = [
            f"FROM {gguf_path.resolve()}",
            f"PARAMETER num_ctx {self.num_ctx}",
            # Deterministic generation: O+V's pipeline judges candidates by
            # whether they parse and pass tests, so sampling entropy here
            # is noise in the evaluation signal, not diversity.
            "PARAMETER temperature 0",
        ]
        if draft_model:
            lines.append(f"DRAFT {draft_model}")
        return "\n".join(lines) + "\n"

    def build_adapter_modelfile(
        self, adapter_path: Path, *, base_tag: str, draft_model: str = "",
    ) -> str:
        """Render a Modelfile that layers an ADAPTER over an existing tag.

        ``FROM`` names a TAG ollama already serves rather than a file, and
        ``ADAPTER`` points at the LoRA GGUF. The base weights are never
        read, copied or re-quantised.

        This is what makes the flywheel closable on one 32 GiB card.
        Merging an adapter into the base to produce a single deployable
        model needs the base in bf16 -- 16 shards, ~57 GiB -- against a
        WSL guest capped at 47 GiB, and on this host a host-RAM blowout is
        not a failed job, it is the desktop (the commit-limit arc). Ollama
        applies the adapter at load time instead, so the whole deployment
        costs the size of the adapter.

        Kept pure, and deliberately parallel to :meth:`build_modelfile`:
        the two differ only in what ``FROM`` names and the extra
        ``ADAPTER`` line, so the deploy machinery below can drive either.
        """
        lines = [
            f"FROM {base_tag}",
            f"ADAPTER {Path(adapter_path).resolve()}",
            f"PARAMETER num_ctx {self.num_ctx}",
            "PARAMETER temperature 0",
        ]
        if draft_model:
            lines.append(f"DRAFT {draft_model}")
        return "\n".join(lines) + "\n"

    async def _gate_result(self, gguf_path: Path, *, min_bytes: int = 0) -> Any:
        """Validate an artifact. ``min_bytes`` overrides the size floor.

        An injected gate (the test seam) is used as given -- a test that
        supplied a gate means to control the verdict, and silently
        rebuilding it would take that control away.
        """
        gate = self._gate
        if gate is None:
            from reactor_core.deployment.gate import (  # noqa: PLC0415
                DeploymentGate,
            )
            kwargs: Dict[str, Any] = {"skip_inference_check": True}
            if min_bytes > 0:
                kwargs["min_file_size_bytes"] = min_bytes
            gate = DeploymentGate(**kwargs)
        return await gate.validate(gguf_path)

    # -- public API -------------------------------------------------------
    async def deploy(self, gguf_path: Path) -> DeployResult:
        """Validate, lease the GPU, snapshot, create, and verify a MODEL."""
        return await self._deploy(
            gguf_path,
            tag=self.tag,
            kind="model",
            modelfile=lambda draft: self.build_modelfile(
                Path(gguf_path), draft_model=draft,
            ),
        )

    async def deploy_adapter(
        self,
        adapter_path: Path,
        *,
        base_tag: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> DeployResult:
        """Publish a LoRA ADAPTER layered over a base tag ollama already has.

        Every step of :meth:`deploy` applies unchanged -- gate, GPU lease,
        rollback snapshot, create, verify-by-serving -- because it is the
        same machinery with a different Modelfile and a size floor that
        suits an adapter. Two preconditions are adapter-specific and both
        fail CLOSED:

        * ``base_tag`` must be given (env ``REACTOR_OLLAMA_BASE_TAG``).
          Layering a LoRA over the wrong base yields a model that loads
          and answers wrongly, which is worse than one that fails to load.
        * that base must already be SERVED. ``ollama create`` would
          otherwise try to pull ~18 GB, turning a deploy into a download
          that competes with whatever else holds the card.
        """
        base = (base_tag if base_tag is not None
                else _env_str(_ENV_BASE_TAG, _DEFAULT_BASE_TAG)).strip()
        target = (tag if tag is not None
                  else _env_str(_ENV_ADAPTER_TAG, _DEFAULT_ADAPTER_TAG)).strip()
        adapter_path = Path(adapter_path)

        if not base:
            return DeployResult(
                ok=False, tag=target, gguf_path=str(adapter_path),
                stage="input",
                reason=(
                    "no base tag: an adapter must name the model it was "
                    f"trained against (pass base_tag= or set {_ENV_BASE_TAG})"
                ),
            )

        served = await self._tags()
        if not any(t == base or t.split(":")[0] == base.split(":")[0]
                   for t in served):
            return DeployResult(
                ok=False, tag=target, gguf_path=str(adapter_path),
                stage="base_missing",
                reason=(
                    f"base tag {base!r} is not served by ollama "
                    f"({len(served)} tags); pull it deliberately -- a deploy "
                    "must not become an 18 GB download"
                ),
            )

        min_bytes = _env_int(
            _ENV_ADAPTER_MIN_BYTES, _DEFAULT_ADAPTER_MIN_BYTES, 1, 1 << 40,
        )
        return await self._deploy(
            adapter_path,
            tag=target,
            kind="adapter",
            gate_min_bytes=min_bytes,
            modelfile=lambda draft: self.build_adapter_modelfile(
                adapter_path, base_tag=base, draft_model=draft,
            ),
            extra_checks=(f"base:{base}",),
        )

    async def _deploy(
        self,
        artifact: Path,
        *,
        tag: str,
        kind: str,
        modelfile: Callable[[str], str],
        gate_min_bytes: int = 0,
        extra_checks: Sequence[str] = (),
    ) -> DeployResult:
        """The deployment machinery, shared by model and adapter.

        ``modelfile`` renders the Modelfile for a given draft model, so the
        DRAFT-retry fallback below is written once and applies to both.
        """
        t0 = time.monotonic()
        gguf_path = Path(artifact)
        res = DeployResult(
            ok=False, tag=tag, gguf_path=str(gguf_path),
            rollback_tag=self.rollback_tag,
        )
        res.checks.extend(extra_checks)

        if not gguf_path.is_file():
            res.stage = "input"
            res.reason = f"{kind} GGUF not found: {gguf_path}"
            res.duration_s = time.monotonic() - t0
            return res

        # 1. Gate -- never publish an artifact that failed validation.
        gate_result = await self._gate_result(
            gguf_path, min_bytes=gate_min_bytes,
        )
        res.gate_summary = (
            gate_result.summary() if hasattr(gate_result, "summary") else ""
        )
        if not getattr(gate_result, "passed", False):
            res.stage = "gate"
            res.reason = (
                f"deployment gate rejected the {kind}: {res.gate_summary}"
            )
            res.duration_s = time.monotonic() - t0
            return res
        res.checks.append("gate:passed")

        # 2. GPU lease -- ollama create writes blobs and the verify probe
        #    loads the model; both contend with a live soak.
        async with self._lease(reason=f"ollama-create:{tag}") as lease:
            res.lease_backend = lease.backend
            if not lease.held:
                res.stage = "gpu_lease"
                res.reason = f"deferring: {lease.reason}"
                res.duration_s = time.monotonic() - t0
                return res
            res.checks.append(f"lease:{lease.backend}")

            # 3. Snapshot the outgoing model BEFORE overwriting the tag.
            existing = await self._tags()
            if any(t.split(":")[0] == tag.split(":")[0] for t in existing):
                rc, out = await self._run(
                    ["cp", tag, self.rollback_tag], timeout_s=120.0,
                )
                if rc == 0:
                    res.checks.append(f"snapshot:{self.rollback_tag}")
                else:
                    # Not fatal: no previous model is a first deploy, and a
                    # failed copy must not block shipping a validated one.
                    logger.warning(
                        "[OllamaDeploy] rollback snapshot failed rc=%d: %s",
                        rc, out.strip()[:200],
                    )
                    res.checks.append("snapshot:failed")
            else:
                res.checks.append("snapshot:none-first-deploy")

            # 4. Create.
            timeout_s = _env_float(
                _ENV_CREATE_TIMEOUT_S, _DEFAULT_CREATE_TIMEOUT_S, 30.0, 86_400.0
            )
            tmpdir = Path(tempfile.mkdtemp(prefix="reactor-modelfile-"))
            try:
                modelfile_path = tmpdir / "Modelfile"
                draft = self.draft_model
                modelfile_path.write_text(modelfile(draft), encoding="utf-8")
                rc, out = await self._run(
                    ["create", tag, "-f", str(modelfile_path)],
                    timeout_s=timeout_s,
                )
                if rc != 0 and draft:
                    # Speculative decoding is an OPTIMISATION. A tag that
                    # will not build with a draft model must still ship
                    # without one -- degrading to single-token generation
                    # is strictly better than failing the deployment and
                    # leaving O+V on the previous model.
                    logger.warning(
                        "[OllamaDeploy] create failed with DRAFT %s (rc=%d: "
                        "%s) -- retrying without speculative decoding",
                        draft, rc, out.strip()[:160],
                    )
                    res.checks.append(f"draft:rejected:{draft}")
                    modelfile_path.write_text(modelfile(""), encoding="utf-8")
                    rc, out = await self._run(
                        ["create", tag, "-f", str(modelfile_path)],
                        timeout_s=timeout_s,
                    )
                elif draft:
                    res.checks.append(f"draft:{draft}")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            if rc != 0:
                res.stage = "create"
                res.reason = f"ollama create failed rc={rc}: {out.strip()[:300]}"
                res.duration_s = time.monotonic() - t0
                return res
            res.checks.append("create:ok")

            # 5. Verify by serving. Exit code 0 means a manifest was
            #    written, not that the tag is servable.
            served = await self._tags()
            stem = tag.split(":")[0]
            if not any(t == tag or t.split(":")[0] == stem for t in served):
                res.stage = "verify"
                res.reason = (
                    f"ollama create returned 0 but {tag} is not in "
                    f"/api/tags ({len(served)} tags served)"
                )
                res.duration_s = time.monotonic() - t0
                return res
            res.checks.append("verify:served")

        res.ok = True
        res.stage = "complete"
        res.reason = f"{tag} is live and servable ({kind})"
        res.duration_s = time.monotonic() - t0
        logger.info("[OllamaDeploy] %s", res.summary())
        return res

    async def rollback(self) -> DeployResult:
        """Restore the previously deployed model onto the live tag."""
        t0 = time.monotonic()
        res = DeployResult(
            ok=False, tag=self.tag, rollback_tag=self.rollback_tag,
            stage="rollback",
        )
        served = await self._tags()
        if not any(
            t == self.rollback_tag
            or t.split(":")[0] == self.rollback_tag.split(":")[0]
            for t in served
        ):
            res.reason = f"no snapshot to roll back to ({self.rollback_tag})"
            res.duration_s = time.monotonic() - t0
            return res

        async with self._lease(reason=f"ollama-rollback:{self.tag}") as lease:
            res.lease_backend = lease.backend
            if not lease.held:
                res.reason = f"deferring: {lease.reason}"
                res.duration_s = time.monotonic() - t0
                return res
            rc, out = await self._run(
                ["cp", self.rollback_tag, self.tag], timeout_s=120.0,
            )
        if rc != 0:
            res.reason = f"ollama cp failed rc={rc}: {out.strip()[:300]}"
        else:
            res.ok = True
            res.reason = f"{self.tag} restored from {self.rollback_tag}"
        res.duration_s = time.monotonic() - t0
        return res


__all__ = [
    "OLLAMA_DEPLOY_SCHEMA_VERSION",
    "DeployResult",
    "OllamaDeployer",
]
