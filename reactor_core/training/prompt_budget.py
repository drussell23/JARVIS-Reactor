"""Fit a corpus prompt into a token budget WITHOUT cutting inside a section.

## Why this exists

Measured 2026-09-05 on the 30B: prefill costs **2.296 MB per prompt token**
at a group of 16 (identical to four significant figures from 1,024 to
6,089 tokens). The corpus prompts run ~6,100 tokens, which leaves 1.03 GiB
under the allocator cap; step 3 of GRPO needs a fresh prefill on top of
optimiser state and asked for 1.57 GiB. Four ladder runs died there, at
four configurations, because the ladder moves ``num_generations`` and
``max_completion_length`` and neither one is the prompt.

## Why not clip at N tokens

A prompt clipped at a token offset ends mid-sentence, mid-code, or -- worst
-- mid-JSON-schema, and the model is then trained on a document shape that
never occurs at inference. A prompt with FEWER SECTIONS, on the other hand,
is a shape the live pipeline already emits: sensors inject sections
conditionally, so the corpus itself contains prompts with and without
``Recent Episodes``, with and without ``Strategic Direction``. Dropping a
whole optional section produces an in-distribution document.

## What the prompts actually look like

Two families, measured across the 27 contrast-bearing prompts:

* **Tool-loop** (17): ``Human Instructions``, ``Task``, ``Target``,
  ``Structural Index`` (~930 tok), ``Available Tools`` (~1,470),
  ``Output Schema`` (~320), then ambient context to the recorder cut.
  There is no source dump; the model reads files through tools.
* **Repair** (10): ``Human Instructions``, ``Task``, ambient context, then
  ``Fault Localization``, ``NEGATIVE CONSTRAINTS``, ``Structural Index``,
  and the source ``File:`` which the recorder cut mid-file.

The ~3,000 tokens of fat in both is AMBIENT injection -- manifesto,
posture, momentum, semantic themes, and memory files whose own ``##``
headings surface as top-level sections. None of it names the task.

## The rule

Sections are classified by heading against an explicit allowlist of
TASK-BEARING names (what to do, where, with what tools, in what shape,
what not to do). Everything else -- including any heading nobody has seen
before -- is ambient. That default matters: an unknown section wrongly
DROPPED costs some context; an unknown section wrongly KEPT can blow the
budget and kill the run, which is the failure this module exists to end.

Protected sections are always kept, whole. Ambient sections are added back
in document order while they fit. Order is preserved, so the result reads
as the live pipeline would have written it with fewer injections.

If the protected core ALONE exceeds the budget the prompt is refused, not
clipped -- the caller decides what to do with a task it cannot afford.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

#: Headings that carry the task. Matched case-insensitively on the H2 text
#: up to the first ``:``, ``(``, ``[`` or dash, so ``Target: backend/x.py
#: [SHA-256: ...]`` matches ``Target``. Order here is documentation only.
PROTECTED_HEADINGS: Tuple[str, ...] = (
    # what to do
    "Human Instructions",
    "Task",
    # where
    "Target",
    "Source Snapshot",
    "File",
    "Structural Index",
    "Fault Localization",
    "Recent Changes",
    # with what, in what shape
    "Available Tools",
    "Output Schema",
    "Output",
    # what not to do / repair context
    "NEGATIVE CONSTRAINTS",
    "Auto-Generated Code Documentation",
    "Function-level analysis",
    "Documentation Gaps",
    "Operation Advisory",
)

_H2 = re.compile(r"(?m)^(?=## )")
_HEAD = re.compile(r"^##\s*(?P<name>[^\n]*)")
_PROTECTED = frozenset(p.lower() for p in PROTECTED_HEADINGS)

CountTokens = Callable[[str], int]


@dataclass
class Section:
    heading: str          # raw first line, e.g. "## Task"
    name: str             # normalised key, e.g. "task"
    text: str             # the whole section including its heading
    tokens: int
    protected: bool


@dataclass
class FitReport:
    budget: int
    before: int
    after: int
    kept: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    refused: bool = False

    @property
    def trimmed(self) -> bool:
        return bool(self.dropped) and not self.refused


def _norm(heading_text: str) -> str:
    """``Target: backend/x.py [SHA-256: ...]`` -> ``target``."""
    t = heading_text.strip()
    t = re.split(r"\s*[:\(\[—–-]", t, maxsplit=1)[0]
    return t.strip().lower()


def is_protected(heading_text: str) -> bool:
    return _norm(heading_text) in _PROTECTED


def split_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """``(preamble, [(heading_text, section_text), ...])`` at H2 boundaries.

    Anything before the first ``## `` is preamble and travels with the
    first section, so a prompt that opens with prose is never split from
    its own opening.
    """
    parts = _H2.split(text)
    preamble = ""
    if parts and not parts[0].startswith("## "):
        preamble = parts.pop(0)
    out: List[Tuple[str, str]] = []
    for part in parts:
        if not part.strip():
            continue
        m = _HEAD.match(part)
        out.append((m.group("name") if m else "", part))
    return preamble, out


def classify(text: str, count: CountTokens) -> Tuple[str, List[Section]]:
    preamble, raw = split_sections(text)
    sections = [
        Section(heading=("## " + name).rstrip(), name=_norm(name), text=body,
                tokens=count(body), protected=is_protected(name))
        for name, body in raw
    ]
    return preamble, sections


def fit_prompt(text: str, budget: int, count: CountTokens) -> Tuple[str, FitReport]:
    """Return ``(fitted_text, report)``.

    ``fitted_text`` is ``text`` unchanged when it already fits. Otherwise it
    is the protected sections plus as many ambient sections as fit, in the
    original order, with no section cut. When the protected core alone
    exceeds ``budget`` the report says ``refused`` and the text is returned
    UNCHANGED -- the caller must not train on it, and must not clip it.
    """
    before = count(text)
    if before <= budget:
        kept = ["## " + name for name, _ in split_sections(text)[1]]
        return text, FitReport(budget=budget, before=before, after=before, kept=kept)

    preamble, sections = classify(text, count)
    pre_tokens = count(preamble) if preamble else 0
    core = pre_tokens + sum(s.tokens for s in sections if s.protected)
    if core > budget:
        rep = FitReport(budget=budget, before=before, after=core, refused=True,
                        kept=[s.heading for s in sections if s.protected],
                        dropped=[s.heading for s in sections if not s.protected])
        return text, rep

    keep: List[bool] = []
    used = core
    for s in sections:
        if s.protected:
            keep.append(True)
            continue
        if used + s.tokens <= budget:
            keep.append(True)
            used += s.tokens
        else:
            keep.append(False)

    body = preamble + "".join(s.text for s, k in zip(sections, keep) if k)
    after = count(body)
    rep = FitReport(
        budget=budget, before=before, after=after,
        kept=[s.heading for s, k in zip(sections, keep) if k],
        dropped=[s.heading for s, k in zip(sections, keep) if not k],
    )
    return body, rep


def summarize(reports: Sequence[FitReport]) -> str:
    """One log line for a batch: what the budget cost, on average."""
    if not reports:
        return "prompt budget: no prompts"
    n = len(reports)
    trimmed = [r for r in reports if r.trimmed]
    refused = sum(1 for r in reports if r.refused)
    b = sum(r.before for r in reports) / n
    admitted = [r for r in reports if not r.refused]
    a = (sum(r.after for r in admitted) / len(admitted)) if admitted else 0.0
    d = (sum(len(r.dropped) for r in trimmed) / len(trimmed)) if trimmed else 0.0
    return (f"prompt budget {reports[0].budget}: {n} prompt(s), "
            f"{len(trimmed)} trimmed, {refused} refused; "
            f"mean tokens {b:.0f} -> {a:.0f}; "
            f"mean sections dropped per trimmed prompt {d:.1f}")
