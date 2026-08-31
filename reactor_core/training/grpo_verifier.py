"""In-loop verification of a GRPO completion — grade the CODE, not the envelope.

## The defect this replaces

`grpo_reward.structural_severity` ran `ast.parse` on the raw completion.
The completion is not Python, it is the O+V response envelope: a JSON
object carrying the candidate file inside a `full_content` string. And a
JSON object is ALSO a valid Python dict literal, so `ast.parse` succeeded
on every well-formed envelope regardless of what it contained. Measured:

    envelope carrying BROKEN python : Severity(0.600, 'parses:1stmt_no_defs')
    envelope carrying GOOD   python : Severity(0.600, 'parses:1stmt_no_defs')

Identical. The grader was answering "is this JSON" while claiming to
answer "is this code correct", so it could not separate a working patch
from a broken one — which is precisely the separation the reward exists
to provide, and precisely why every group came out flat.

## The ladder

Verification is TIERED because the tiers cost different amounts and
answer different questions, and because a failure at one tier makes the
tiers above it unanswerable rather than merely unmet — a completion whose
envelope does not parse has no candidate to AST-check, and scoring it as
"AST failed" would conflate two different faults.

    0 ENVELOPE    the response parses and names a known schema_version
    1 SHAPE       the declared shape carries the keys that shape requires
    2 SYNTAX      the extracted source parses as Python
    3 SUBSTANCE   it is more than a docstring / pass / empty module
    4 AUTHORITY   the real validator agrees            (optional, budgeted)

Each tier yields a graded band, so two completions failing at the SAME
tier are still separable — where a parse dies, how much of the file was
reached. That grading is what rescues a group in which every sibling
failed, without inventing a difference that is not there.

## Tier 4 and the two-venv boundary

The authoritative validator lives in the jarvis repo, in a different venv
this process cannot import (see `reactor_core.training.grpo_reward` and
the flywheel notes: the cross-repo contract is the SCHEMA, never the
code). Reimplementing VALIDATE here would be exactly the duplication that
guarantees the two drift apart, and a training reward drifting from the
gate it is supposed to teach is worse than having no tier 4 at all.

So tier 4 SHELLS OUT to a verifier command the operator configures, and
is absent by default. The process boundary is what lets two venvs share
one implementation. It is budgeted by an adaptive token bucket whose
refill tracks the MEASURED cost of previous calls, so an expensive
validator throttles itself instead of stalling the trainer.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The shared contract
# ---------------------------------------------------------------------------
#
# Mirrors jarvis `providers._SCHEMA_VERSION` / `_CANDIDATE_KEYS` /
# `_DIFF_CANDIDATE_KEYS` / `_NOOP_KEYS` / `_TOOL_KEYS`. Restated rather
# than imported because the two repos have separate venvs and a
# cross-repo import is impossible; the SCHEMA is the integration surface,
# so this is the contract, not a duplicated implementation. Declared once
# here and projected into every tier below.

SCHEMA_FULL = "2b.1"
SCHEMA_DIFF = "2b.1-diff"
SCHEMA_NOOP = "2b.1-noop"
SCHEMA_TOOL = "2b.2-tool"

#: schema_version -> (required top-level keys, per-candidate content key)
_SHAPES: Dict[str, Tuple[frozenset, Optional[str]]] = {
    SCHEMA_FULL: (frozenset({"schema_version", "candidates"}), "full_content"),
    SCHEMA_DIFF: (frozenset({"schema_version", "candidates"}), "unified_diff"),
    SCHEMA_NOOP: (frozenset({"schema_version", "reason"}), None),
    SCHEMA_TOOL: (frozenset({"schema_version", "tool_calls"}), None),
}

_ENV_PREFIX = "REACTOR_GRPO_"


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(_ENV_PREFIX + name, str(default)))
    except (TypeError, ValueError):
        return default


def _envs(name: str, default: str = "") -> str:
    return (os.environ.get(_ENV_PREFIX + name, default) or "").strip()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One completion's graded verification result.

    ``score`` is in [0, 1], HIGHER IS BETTER, so it composes with a reward
    directly. ``tier`` is the highest tier REACHED, and ``reason`` names
    what stopped it — both carried so a flat corpus is diagnosable rather
    than merely disappointing.
    """

    score: float
    tier: int
    reason: str
    schema_version: str = ""
    authoritative: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        a = "+auth" if self.authoritative else ""
        return f"Verdict({self.score:.3f}, t{self.tier}{a}, {self.reason!r})"


@dataclass
class TierWeights:
    """Score band per tier. Bands do NOT overlap, by construction.

    A completion that fails an earlier tier can never outrank one that
    reached a later tier, however gracefully it failed — otherwise a
    beautifully-formed envelope containing garbage would outscore working
    code, and the reward would teach presentation over correctness.
    """

    envelope: float = field(default_factory=lambda: _envf("BAND_ENVELOPE", 0.10))
    shape: float = field(default_factory=lambda: _envf("BAND_SHAPE", 0.25))
    syntax: float = field(default_factory=lambda: _envf("BAND_SYNTAX", 0.45))
    substance: float = field(default_factory=lambda: _envf("BAND_SUBSTANCE", 0.70))
    authority: float = field(default_factory=lambda: _envf("BAND_AUTHORITY", 1.00))


# ---------------------------------------------------------------------------
# Tiers 0-3 — pure, fast, no external process
# ---------------------------------------------------------------------------


def extract_sources(text: str) -> Tuple[Optional[str], List[str], str]:
    """``(schema_version, [source, ...], reason)`` from a completion.

    The load-bearing correction: this reaches THROUGH the envelope to the
    candidate bodies. Grading the envelope is what made broken and
    working code indistinguishable.

    Fail-soft at every step; never raises.
    """
    body = (text or "").strip()
    if not body:
        return None, [], "empty"
    # Models fence JSON despite being told not to; the fence is a
    # presentation fault, not a code fault, and rejecting it here would
    # score a good patch as unparseable.
    if body.startswith("```"):
        nl = body.find("\n")
        body = body[nl + 1:] if nl >= 0 else body
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        body = body.strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [], f"envelope_unparseable:{exc.msg[:40]}"
    if not isinstance(obj, dict):
        return None, [], f"envelope_not_object:{type(obj).__name__}"

    ver = str(obj.get("schema_version") or "")
    if ver not in _SHAPES:
        return ver or None, [], f"unknown_schema_version:{ver or '__missing__'}"

    required, content_key = _SHAPES[ver]
    missing = sorted(required - set(obj))
    if missing:
        return ver, [], f"missing_keys:{','.join(missing)}"
    if content_key is None:
        # noop / tool are terminal-but-codeless: legitimately shaped, and
        # there is simply no source to grade.
        return ver, [], "no_source_by_shape"

    cands = obj.get("candidates")
    if not isinstance(cands, list) or not cands:
        return ver, [], "candidates_empty"

    sources: List[str] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        v = c.get(content_key)
        if isinstance(v, str) and v.strip():
            sources.append(v)
        # A multi-file candidate carries the rest under `files`; each entry
        # is graded like the primary, or a patch that split its work across
        # files would be judged on a fraction of itself.
        for f in (c.get("files") or []) if isinstance(c.get("files"), list) else []:
            if isinstance(f, dict):
                fv = f.get(content_key)
                if isinstance(fv, str) and fv.strip():
                    sources.append(fv)
    if not sources:
        return ver, [], f"no_{content_key}"
    return ver, sources, ""


def _grade_source(src: str) -> Tuple[float, str]:
    """Graded syntax/substance for ONE extracted source. In [0, 1]."""
    try:
        ast.parse(src)
    except SyntaxError as exc:
        total = max(1, src.count("\n") + 1)
        line = int(getattr(exc, "lineno", 1) or 1)
        # How far the parse got, normalised over the file's own length: a
        # candidate that dies on its last line got nearly everything right,
        # and that difference is the signal a boolean throws away.
        return max(0.0, min(1.0, (line - 1) / total)), f"syntax_error:line{line}/{total}"
    except (ValueError, RecursionError) as exc:
        return 0.0, f"uncompilable:{type(exc).__name__}"

    tree = ast.parse(src)
    stmts = list(getattr(tree, "body", []) or [])
    if not stmts:
        return 0.0, "empty_module"
    inert = all(
        isinstance(n, ast.Pass)
        or (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))
        for n in stmts
    )
    if inert:
        return 0.15, "inert_body"
    defs = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    if defs == 0:
        return 0.5, f"{len(stmts)}stmt_no_defs"
    # Saturating: rewarding "more definitions" without bound reintroduces
    # the length bias `loss_type="dr_grpo"` was chosen to remove.
    return min(1.0, 0.7 + 0.06 * min(defs, 5)), f"{defs}defs/{len(stmts)}stmt"


def verify_static(text: str, weights: Optional[TierWeights] = None) -> Verdict:
    """Tiers 0-3. Pure, sub-millisecond, never raises."""
    w = weights or TierWeights()
    try:
        ver, sources, reason = extract_sources(text)
        if ver is None:
            return Verdict(w.envelope * 0.2, 0, reason or "envelope_invalid")
        if reason.startswith("unknown_schema_version"):
            return Verdict(w.envelope, 0, reason, ver)
        if reason == "no_source_by_shape":
            # A noop or tool call is a WELL-FORMED answer that contains no
            # code. Checked BEFORE the generic `no_` prefix below, which
            # would otherwise swallow it (`no_source_by_shape` and
            # `no_full_content` both start `no_`) and score a correct
            # decline beneath broken code -- teaching the model to emit
            # garbage rather than say "already done", which is the exact
            # inversion this reward exists to prevent.
            #
            # Placed at the SYNTAX ceiling: strictly below anything that
            # parses, level with a syntax error that got almost all the way.
            # That ordering says "declining beats breaking, delivering
            # beats declining" without inviting the noop-spam that
            # qwen3-coder:30b produced 209 times in one soak.
            return Verdict(w.syntax, 1, reason, ver)
        if reason.startswith("missing_keys") or reason == "candidates_empty" \
                or reason.startswith("no_"):
            # Envelope parsed and named a shape, but does not satisfy it.
            return Verdict(w.shape * 0.6, 1, reason, ver)

        # Tier 2/3 — the whole point: grade the extracted SOURCE.
        graded = [_grade_source(s) for s in sources]
        # Worst-file semantics: a multi-file candidate is only as valid as
        # its weakest file, because APPLY is all-or-nothing.
        score, why = min(graded, key=lambda g: g[0])
        if score <= 0.0 or why.startswith(("syntax_error", "uncompilable", "empty")):
            span = w.syntax - w.shape
            return Verdict(w.shape + span * score, 2, why, ver)
        span = w.substance - w.syntax
        return Verdict(w.syntax + span * score, 3, why, ver)
    except Exception as exc:  # noqa: BLE001 — a grader must never break training
        logger.debug("verify_static fault: %s", exc, exc_info=True)
        return Verdict(w.envelope * 0.2, 0, "grader_fault")


# ---------------------------------------------------------------------------
# Tier 4 — the authoritative validator, across the venv boundary
# ---------------------------------------------------------------------------


class AdaptiveBudget:
    """Token bucket whose refill tracks the MEASURED cost of the work.

    A fixed rate cannot be right: the authoritative validator may take
    30 ms or 30 s depending on what it runs, and either guess starves the
    trainer or lets verification dominate the step. The bucket refills at
    a fraction of wall-clock (the share of time verification may consume)
    and each call debits its own EWMA-smoothed cost, so an expensive
    validator throttles itself.
    """

    def __init__(self, share: Optional[float] = None, burst: Optional[float] = None):
        self._share = share if share is not None else _envf("VERIFY_TIME_SHARE", 0.25)
        self._burst = burst if burst is not None else _envf("VERIFY_BURST_S", 30.0)
        self._tokens = self._burst
        self._last = time.monotonic()
        self._ewma: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def estimated_cost_s(self) -> float:
        return self._ewma if self._ewma is not None else _envf("VERIFY_SEED_COST_S", 2.0)

    async def try_acquire(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._burst, self._tokens + (now - self._last) * self._share
            )
            self._last = now
            need = self.estimated_cost_s
            if self._tokens < need:
                return False
            self._tokens -= need
            return True

    async def record(self, elapsed_s: float) -> None:
        async with self._lock:
            a = _envf("VERIFY_EWMA_ALPHA", 0.3)
            self._ewma = (
                elapsed_s if self._ewma is None
                else (a * elapsed_s + (1 - a) * self._ewma)
            )


_BUDGET = AdaptiveBudget()


def authoritative_command() -> str:
    """Operator-configured verifier command, or "" when unset.

    Absent by default ON PURPOSE. The real validator lives in the jarvis
    venv; wiring a guessed path would produce a tier that silently always
    fails, which is indistinguishable in the corpus from a model that
    always writes broken code.

    The command receives the candidate source on stdin and must exit 0 for
    valid. `{file}` in the command is replaced with a temp path holding
    the same source, for validators that want a file.
    """
    return _envs("VERIFY_CMD", "")


async def verify_authoritative(
    source: str, *, timeout_s: Optional[float] = None,
) -> Optional[bool]:
    """Run the operator's validator. ``None`` = no answer, not a failure.

    The distinction is load-bearing: "the validator says this is broken"
    and "no validator ran" must never collapse into the same reward, or
    an unconfigured tier would punish every completion equally.
    """
    cmd = authoritative_command()
    if not cmd:
        return None
    if not await _BUDGET.try_acquire():
        logger.debug("[GRPO] authoritative verify skipped — budget exhausted")
        return None

    import tempfile
    t0 = time.monotonic()
    path = ""
    try:
        if "{file}" in cmd:
            fd, path = tempfile.mkstemp(suffix=".py")
            os.close(fd)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            argv = shlex.split(cmd.replace("{file}", shlex.quote(path)))
        else:
            argv = shlex.split(cmd)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(
                proc.communicate(source.encode("utf-8", "replace")),
                timeout=timeout_s if timeout_s is not None
                else _envf("VERIFY_TIMEOUT_S", 60.0),
            )
        except asyncio.TimeoutError:
            # A wedged validator must not wedge training. Kill it and
            # report NO ANSWER -- a timeout says nothing about the code.
            try:
                proc.kill()
                await proc.wait()
            except Exception:  # noqa: BLE001
                pass
            logger.info("[GRPO] authoritative verify timed out — no verdict")
            return None
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        logger.debug("[GRPO] authoritative verify unavailable: %s", exc)
        return None
    finally:
        await _BUDGET.record(time.monotonic() - t0)
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


async def verify(text: str, weights: Optional[TierWeights] = None) -> Verdict:
    """Full ladder: static tiers always, tier 4 when configured + affordable.

    Tier 4 is consulted ONLY when the static tiers already reached
    SUBSTANCE. Asking an expensive validator about a candidate that does
    not parse spends budget to learn what tier 2 established for free.
    """
    w = weights or TierWeights()
    loop = asyncio.get_running_loop()
    static = await loop.run_in_executor(None, verify_static, text, w)
    if static.tier < 3:
        return static

    ver, sources, _ = await loop.run_in_executor(None, extract_sources, text)
    if not sources:
        return static
    ok = await verify_authoritative("\n".join(sources))
    if ok is None:
        return static
    if not ok:
        # The authority overrules a hopeful static pass, but not below the
        # SYNTAX band: the code did parse, and pretending otherwise would
        # rank it beneath candidates that did not.
        return Verdict(w.syntax, 3, "authority_rejected", ver, authoritative=True)
    return Verdict(w.authority, 4, "authority_accepted", ver, authoritative=True)


async def verify_batch(texts: Sequence[str], weights: Optional[TierWeights] = None):
    """Grade a whole group concurrently.

    The static tiers are CPU-bound and the authoritative tier is I/O; both
    benefit, and a group is `num_generations` of them.
    """
    return list(await asyncio.gather(*(verify(t, weights) for t in texts)))


__all__ = [
    "AdaptiveBudget", "SCHEMA_DIFF", "SCHEMA_FULL", "SCHEMA_NOOP",
    "SCHEMA_TOOL", "TierWeights", "Verdict", "authoritative_command",
    "extract_sources", "verify", "verify_authoritative", "verify_batch",
    "verify_static",
]
