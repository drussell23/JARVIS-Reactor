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

    ## The passing band is DERIVED, not a constant

    ``substance`` used to be a rigid 0.70, which made the band for code
    that WORKS ``[0.60, 0.70]`` — 0.10 wide, while everything from 0.70 to
    the authority band sat empty. Measured consequence (soaks 13 and 14):
    every passing sibling scored 0.65-0.672 and the best group spread in
    two full soaks was 0.0052, so `--min-spread 0.01` demanded 10% of the
    entire available range between two candidates and no group ever
    cleared it. The generator was producing structurally distinct answers
    by then; the reward simply had nowhere to put the difference.

    So the ceiling is now derived from the band ABOVE it rather than
    declared: passing code spans ``[passing_floor, authority -
    authority_margin]``. The margin is the whole non-overlap contract in
    one number — an authoritative (test-executed) verdict must still
    strictly outrank the most beautiful statically-graded candidate,
    because tests are evidence and style is not. Set ``BAND_SUBSTANCE``
    explicitly to pin the old behaviour.

    Every edge is env-tunable and the ordering is ENFORCED in
    ``__post_init__``: a misconfigured environment degrades to a valid
    monotonic ladder instead of silently inverting the reward.
    """

    envelope: float = field(default_factory=lambda: _envf("BAND_ENVELOPE", 0.10))
    shape: float = field(default_factory=lambda: _envf("BAND_SHAPE", 0.25))
    syntax: float = field(default_factory=lambda: _envf("BAND_SYNTAX", 0.45))
    #: Bottom of the band for code that parses AND defines something. Above
    #: `syntax` so a passing candidate can never be outranked by a graceful
    #: syntax failure.
    passing_floor: float = field(
        default_factory=lambda: _envf("BAND_PASSING_FLOOR", _PASS_FLOOR))
    #: Headroom reserved BELOW `authority` for the authoritative tier. The
    #: one number that keeps "tests passed" strictly above "reads nicely".
    authority_margin: float = field(
        default_factory=lambda: _envf("BAND_AUTHORITY_MARGIN", 0.05))
    #: Ceiling of the passing band. Negative (the default) means DERIVE it
    #: from `authority` and `authority_margin`; an explicit value pins it.
    substance: float = field(
        default_factory=lambda: _envf("BAND_SUBSTANCE", -1.0))
    authority: float = field(default_factory=lambda: _envf("BAND_AUTHORITY", 1.00))

    def __post_init__(self) -> None:
        if self.substance < 0.0:
            self.substance = self.authority - self.authority_margin
        # Monotonicity is the contract every tier mapping relies on. Repair
        # rather than raise: a grader that refuses to start would stop
        # training entirely over a typo'd env var, which is strictly worse
        # than grading on a clamped-but-ordered ladder.
        edges = ["envelope", "shape", "syntax", "passing_floor",
                 "substance", "authority"]
        prev = 0.0
        for name in edges:
            val = max(prev, min(1.0, float(getattr(self, name))))
            setattr(self, name, val)
            prev = val


# ---------------------------------------------------------------------------
# Tiers 0-3 — pure, fast, no external process
# ---------------------------------------------------------------------------


_FENCE = '```'


def _strip_code_fence(text: str) -> str:
    """Remove ONE wrapping markdown fence, if the value is entirely fenced.

    A fence is a presentation fault, not a code fault. It is stripped in
    TWO places, which is why this is a function and not the inline block
    it used to be: around the envelope, and around each extracted source.

    The second is the load-bearing one, and it was missing. A
    ``full_content`` carrying fenced-but-CORRECT Python reached
    ``ast.parse`` with the backticks still attached and scored
    ``syntax_error:line1`` -- byte-identical to the score for genuinely
    broken Python (both 0.250, measured). Correct work and broken work
    collapsed onto one number, so the reward graded FORMATTING and taught
    nothing. That is the same defect class as grading the JSON envelope
    instead of the code inside it.

    Only a value that BEGINS with a fence is touched, so a legitimate
    source that merely mentions backticks in a docstring is left alone.
    """
    body = text or ""
    if not body.lstrip().startswith(_FENCE):
        # NOT fenced: return it byte-identical. Stripping here silently ate
        # a trailing newline and broke extraction's round-trip guarantee --
        # the graded text must be the model's text wherever nothing is wrong
        # with it.
        return body
    body = body.strip()
    nl = body.find(chr(10))
    body = body[nl + 1:] if nl >= 0 else body[len(_FENCE):]
    if body.rstrip().endswith(_FENCE):
        body = body.rstrip()[:-len(_FENCE)]
    return body.strip()


def extract_sources(text: str) -> Tuple[Optional[str], List[str], str]:
    """``(schema_version, [source, ...], reason)`` from a completion.

    The load-bearing correction: this reaches THROUGH the envelope to the
    candidate bodies. Grading the envelope is what made broken and
    working code indistinguishable.

    Fail-soft at every step; never raises.
    """
    body = _strip_code_fence(text or "")
    if not body:
        return None, [], "empty"
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
            sources.append(_strip_code_fence(v))
        # A multi-file candidate carries the rest under `files`; each entry
        # is graded like the primary, or a patch that split its work across
        # files would be judged on a fraction of itself.
        for f in (c.get("files") or []) if isinstance(c.get("files"), list) else []:
            if isinstance(f, dict):
                fv = f.get(content_key)
                if isinstance(fv, str) and fv.strip():
                    sources.append(_strip_code_fence(fv))
    if not sources:
        # kind is `no_content`, NOT `no_<key>`: the old form made the kind
        # depend on the schema's content key, which is what let it collide
        # with `no_source_by_shape` under a prefix test.
        return ver, [], f"no_content:{content_key}"
    return ver, sources, ""


#: Floor of the passing band, in FINAL SCORE space. Above the structure
#: band, so a wider range for working code can never outrank a later tier.
_PASS_FLOOR = _envf("PASS_FLOOR", 0.60)

#: The three bands a graded source can land in, in ascending order. Named
#: because the ladder is now structural: each band owns a disjoint score
#: interval, so "cannot be outranked by a lower tier" is a property of the
#: intervals rather than an arithmetic coincidence between two constants.
_BAND_SYNTAX = "syntax"        # parses badly or not at all
_BAND_STRUCTURE = "structure"  # parses, but defines nothing worth grading
_BAND_PASSING = "passing"      # parses AND defines: the graded band

#: Positions WITHIN the structure band. Inert code (a body of `pass` or a
#: bare docstring) is worth strictly less than code that at least runs
#: statements, and both are worth less than anything with a definition.
_INERT_Q = _envf("INERT_Q", 0.15)
_NO_DEFS_Q = _envf("NO_DEFS_Q", 0.50)

#: Relative weights of the quality sub-metrics. Env-tunable because what
#: "better code" means is a policy, not a constant -- a docs-focused batch
#: and a refactor batch do not want the same emphasis.
_Q_WEIGHTS = {
    "docs": _envf("Q_W_DOCS", 1.0),
    "types": _envf("Q_W_TYPES", 1.0),
    "density": _envf("Q_W_DENSITY", 0.6),
    "simplicity": _envf("Q_W_SIMPLICITY", 0.8),
    "concision": _envf("Q_W_CONCISION", 0.6),
    "flatness": _envf("Q_W_FLATNESS", 0.6),
}

#: Chars-per-statement beyond which a file reads as padded. Not a hard cap:
#: the sub-metric decays smoothly so one verbose function cannot zero a file.
_CONCISION_TARGET = _envf("CONCISION_TARGET_CPS", 90.0)

#: Decision points per definition treated as "reasonably simple".
_SIMPLICITY_TARGET = _envf("SIMPLICITY_TARGET_BRANCHES", 4.0)

#: Nesting depth of branching constructs treated as "reasonably flat".
#: Decays past it rather than cutting off, for the same reason as the rest.
_DEPTH_TARGET = _envf("DEPTH_TARGET_LEVELS", 3.0)

_BRANCH_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
    ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp,
)


def _soft(value: float, target: float) -> float:
    """1.0 at zero, decaying toward 0 as `value` passes `target`.

    Hyperbolic rather than a cliff: a step function would put every
    candidate on one side or the other and reintroduce exactly the ties
    this exists to break.
    """
    if target <= 0:
        return 1.0
    return float(target) / (float(target) + max(0.0, float(value)))


def _quality(tree, src, defs, stmts):
    """Continuous quality of code that already PARSES. Returns (q, detail).

    ## Why this replaces a def-count step

    The previous score was `min(1.0, 0.7 + 0.06 * min(defs, 5))` -- five
    reachable values, pinned above five definitions. Two candidates with
    the same number of defs scored IDENTICALLY no matter how different
    they were, so a GRPO group of passing siblings had a reward spread of
    exactly 0.0 and was dropped by the equal-reward guard. Measured on the
    live corpus: 19 groups with 2+ responses, all 19 flat, 0 trainable.
    The reward could rank failure modes finely and could not rank success
    at all.

    ## What it measures, and why each one is real

    Every sub-metric is a MEASUREMENT of the candidate, never a tiebreak
    invented to manufacture difference. Two byte-identical candidates still
    score identically -- they ARE identical, and dropping that group is
    correct. Different code now almost always differs, because five
    continuous signals collide far less than one integer did.

      * docs      -- fraction of definitions carrying a docstring
      * types     -- fraction of annotatable positions (params + returns)
        that are annotated
      * density   -- definitions per statement, saturating: structure is
        good, but "more defs" without bound is the length bias
        `loss_type="dr_grpo"` was chosen to remove
      * simplicity-- decision points per definition, decaying past a
        target, so a tangle of branches scores below the same behaviour
        expressed plainly
      * concision -- characters per statement, decaying past a target, so
        padding a file cannot buy reward

    Weights are env-tunable: what "better" means is policy, not constant.
    """
    n_defs = len(defs)
    n_stmts = max(1, len(stmts))

    documented = sum(1 for d in defs if ast.get_docstring(d))
    docs = documented / n_defs

    annotatable = 0
    annotated = 0
    for d in defs:
        if not isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = d.args
        params = list(a.args) + list(a.kwonlyargs) + list(getattr(a, "posonlyargs", []))
        for arg in params:
            if arg.arg in ("self", "cls"):
                continue
            annotatable += 1
            annotated += 1 if arg.annotation is not None else 0
        annotatable += 1                                  # the return
        annotated += 1 if d.returns is not None else 0
    types = (annotated / annotatable) if annotatable else 1.0

    density = min(1.0, n_defs / n_stmts)

    branches = sum(1 for n in ast.walk(tree) if isinstance(n, _BRANCH_NODES))
    simplicity = _soft(branches / n_defs, _SIMPLICITY_TARGET)

    # NESTING is a different axis from branch COUNT, which is why it is its
    # own sub-metric rather than folded into `simplicity`: ten sequential
    # guard clauses and one ten-deep pyramid have the same branch count and
    # are not the same code. Measured on the deepest nesting of any
    # BRANCHING construct, so a module of long flat functions is not
    # punished for being long.
    flatness = _soft(_max_branch_depth(tree), _DEPTH_TARGET)

    # `src.strip()`, not `src`: surrounding whitespace is PRESENTATION, and
    # letting it move the score reintroduces the defect the fence stripper
    # exists to remove -- two candidates whose code is byte-identical would
    # score differently because one arrived with a trailing newline. Caught
    # by the fence tests, which assert stripped and clean score the SAME.
    concision = _soft(len(src.strip()) / n_stmts, _CONCISION_TARGET)

    parts = {
        "docs": docs, "types": types, "density": density,
        "simplicity": simplicity, "concision": concision,
        "flatness": flatness,
    }
    # Weight lookup tolerates a sub-metric the env has not been told about,
    # so adding a dimension can never divide by a stale total or KeyError a
    # training run mid-batch.
    weights = {k: float(_Q_WEIGHTS.get(k, 0.0)) for k in parts}
    total_w = sum(weights.values()) or 1.0
    q = sum(parts[k] * weights[k] for k in parts) / total_w
    q = max(0.0, min(1.0, q))
    detail = (
        f"q={q:.3f},docs={docs:.2f},types={types:.2f},den={density:.2f},"
        f"simp={simplicity:.2f},con={concision:.2f},flat={flatness:.2f},"
        f"defs={n_defs},stmt={n_stmts}"
    )
    return q, detail


def _max_branch_depth(tree: "ast.AST") -> int:
    """Deepest nesting of branching constructs. Iterative, never recurses.

    `ast.walk` cannot answer this — it flattens. A recursive walker would
    be the obvious alternative and is the wrong tool here: this grader runs
    on model-generated source, and deeply-nested generated code is exactly
    the input that would hit the interpreter's recursion limit. A grader
    that raises on the pathological case is a grader that stops training,
    so the traversal carries its own explicit stack.
    """
    deepest = 0
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        for child in ast.iter_child_nodes(node):
            d = depth + 1 if isinstance(child, _BRANCH_NODES) else depth
            if d > deepest:
                deepest = d
            stack.append((child, d))
    return deepest


def _grade_source(src: str) -> "SourceGrade":
    """Grade ONE extracted source: which BAND it earns, and where in it.

    Returns a position in [0, 1] WITHIN a named band rather than a single
    number spanning every outcome. That split is the fix for a real defect:
    the old return value pre-compressed passing code into the top 40% of
    [0, 1] (`_PASS_FLOOR + (1 - _PASS_FLOOR) * q`) and the caller then
    squeezed THAT into a 0.25-wide tier span, so quality variation reached
    the reward twice-attenuated — 0.10 of usable range, and in practice a
    group spread of ~1e-3 against a 0.01 floor.

    Naming the band makes the ladder structural instead of arithmetic: the
    caller maps each band onto its own interval, and the non-overlap
    contract holds because the intervals do not overlap, not because two
    magic constants happen to be ordered.
    """
    try:
        ast.parse(src)
    except SyntaxError as exc:
        total = max(1, src.count("\n") + 1)
        line = int(getattr(exc, "lineno", 1) or 1)
        # How far the parse got, normalised over the file's own length: a
        # candidate that dies on its last line got nearly everything right,
        # and that difference is the signal a boolean throws away.
        return SourceGrade(max(0.0, min(1.0, (line - 1) / total)), _BAND_SYNTAX,
                           f"syntax_error:line{line}/{total}")
    except (ValueError, RecursionError) as exc:
        return SourceGrade(0.0, _BAND_SYNTAX, f"uncompilable:{type(exc).__name__}")

    tree = ast.parse(src)
    stmts = list(getattr(tree, "body", []) or [])
    if not stmts:
        return SourceGrade(0.0, _BAND_SYNTAX, "empty_module")
    inert = all(
        isinstance(n, ast.Pass)
        or (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))
        for n in stmts
    )
    if inert:
        return SourceGrade(_INERT_Q, _BAND_STRUCTURE, "inert_body")
    defs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not defs:
        return SourceGrade(_NO_DEFS_Q, _BAND_STRUCTURE, f"{len(stmts)}stmt_no_defs")
    q, detail = _quality(tree, src, defs, stmts)
    # Raw q — the caller owns the band. No pre-compression here, or the
    # attenuation this refactor removed comes straight back.
    return SourceGrade(q, _BAND_PASSING, f"quality:{detail}")


def _reason_kind(reason: str) -> str:
    """The stable KIND of a reason string: everything before the first ':'.

    Reasons are `kind` or `kind:detail`. Dispatching on the kind is an
    EXACT match, so adding a reason can never silently change how an
    existing one is scored.

    The previous version tested string PREFIXES, and the ordering that
    made it correct was invisible: `no_source_by_shape` and
    `no_full_content` both start `no_`, so the generic test swallowed the
    noop and scored a correct decline BENEATH broken code -- teaching the
    model to emit garbage rather than say "already done". It was fixed by
    putting the specific check first, which works and stays working only
    for as long as nobody reorders two `if`s or adds another `no_*`.
    Exact kinds remove the hazard instead of documenting it.
    """
    return reason.split(":", 1)[0]


#: Envelope parsed and named a shape, but does not satisfy it.
_SHAPE_FAULTS = frozenset({"missing_keys", "candidates_empty", "no_content"})

#: The extracted source is not usable Python.
_SYNTAX_FAULTS = frozenset({"syntax_error", "uncompilable", "empty_module"})


@dataclass(frozen=True)
class SourceGrade:
    """Where one source sits: a band, and a position in [0, 1] within it."""

    q: float
    band: str
    reason: str


def _verdict_for_grade(
    grade: "SourceGrade", w: "TierWeights", schema_version: str = "",
) -> Verdict:
    """Map a banded grade onto its score interval. THE only such mapping.

    This existed twice — identically — in `verify_static` and `verify_any`,
    which is precisely the shape that drifts: a band widened in one place
    and not the other would score the same code differently depending on
    which layer read it, and nothing would fail. One function, two callers.

    The interval per band is read from `TierWeights`, whose `__post_init__`
    guarantees the edges are monotonic, so a mapped score can never escape
    its band or invert the ladder.
    """
    lo, hi, tier = {
        _BAND_SYNTAX: (w.shape, w.syntax, 2),
        _BAND_STRUCTURE: (w.syntax, w.passing_floor, 3),
        _BAND_PASSING: (w.passing_floor, w.substance, 3),
    }.get(grade.band, (w.shape, w.syntax, 2))
    q = max(0.0, min(1.0, float(grade.q)))
    return Verdict(lo + (hi - lo) * q, tier, grade.reason, schema_version)


def verify_static(text: str, weights: Optional[TierWeights] = None) -> Verdict:
    """Tiers 0-3. Pure, sub-millisecond, never raises."""
    w = weights or TierWeights()
    try:
        ver, sources, reason = extract_sources(text)
        if ver is None:
            return Verdict(w.envelope * 0.2, 0, reason or "envelope_invalid")
        kind = _reason_kind(reason)
        if kind == "unknown_schema_version":
            return Verdict(w.envelope, 0, reason, ver)
        if kind == "no_source_by_shape":
            # A noop or tool call is a WELL-FORMED answer that contains no
            # code. It is its own KIND now, so no other reason can swallow
            # it and score a correct decline beneath broken code -- the
            # exact inversion this reward exists to prevent. That used to
            # depend on this branch sitting above a `no_` prefix test.
            #
            # Placed at the SYNTAX ceiling: strictly below anything that
            # parses, level with a syntax error that got almost all the way.
            # That ordering says "declining beats breaking, delivering
            # beats declining" without inviting the noop-spam that
            # qwen3-coder:30b produced 209 times in one soak.
            return Verdict(w.syntax, 1, reason, ver)
        # Envelope parsed and named a shape, but does not satisfy it.
        if kind in _SHAPE_FAULTS:
            return Verdict(w.shape * 0.6, 1, reason, ver)

        # Tier 2/3 — the whole point: grade the extracted SOURCE.
        graded = [_grade_source(s) for s in sources]
        # Worst-file semantics: a multi-file candidate is only as valid as
        # its weakest file, because APPLY is all-or-nothing. Ranked by the
        # SCORE the band produces, not by `q` — `q` is a position within a
        # band, so comparing it across bands would rank a perfect-quality
        # file below a merely-late syntax error.
        worst = min(graded, key=lambda g: _verdict_for_grade(g, w).score)
        return _verdict_for_grade(worst, w, ver)
    except Exception as exc:  # noqa: BLE001 — a grader must never break training
        logger.debug("verify_static fault: %s", exc, exc_info=True)
        return Verdict(w.envelope * 0.2, 0, "grader_fault")


def verify_any(text: str, weights: Optional[TierWeights] = None) -> Verdict:
    """Grade a response whose LAYER is not known in advance.

    `verify_static` grades a COMPLETION -- the O+V response envelope, which
    is what a live GRPO rollout produces and what the reward sees during
    training. A CORPUS row is a different layer: the recorder stores the
    already-extracted source in `assistant_output`, so the envelope is long
    gone by the time anything reads the corpus back.

    Handing corpus text to `verify_static` therefore fails `json.loads` and
    returns the tier-0 constant for EVERY row -- measured on the live
    corpus, all 74 rows scored 0.02 / tier 0 / `envelope_unparseable`, a
    perfectly flat reward that had nothing to do with the code. The same 74
    graded as source give 67 distinct scores.

    This is the fourth appearance of one defect: grading the JSON envelope
    instead of the code, grading the markdown fence instead of the code,
    dispatching on string shape, and now grading the wrong layer entirely.
    The lesson each time is the same -- be explicit about WHAT is being
    measured, because every one of these failures was silent and produced a
    confident, uniform number.

    Envelope first (a real envelope must still be reached through, or a
    JSON object would be graded as the Python dict literal it also is);
    source only when the text is demonstrably not an envelope.
    """
    v = verify_static(text, weights)
    if _reason_kind(v.reason) != "envelope_unparseable":
        return v
    w = weights or TierWeights()
    try:
        grade = _grade_source(text or "")
    except Exception:  # noqa: BLE001 — a grader must never break training
        logger.debug("verify_any source fault", exc_info=True)
        return v
    return _verdict_for_grade(grade, w)


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
    "verify_static", "verify_any",
]
