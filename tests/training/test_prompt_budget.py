"""A prompt fits its budget by losing whole ambient sections, never by
being cut inside one.

Measured 2026-09-05: prefill costs 2.296 MB per prompt token at a group of
16, so the ~6,100-token corpus prompts left 1.03 GiB under the cap and step
3 asked for 1.57. The ladder's two knobs never touch the prompt, so every
rung died identically. This module is the prompt-side lever, and its one
contract is that a task-bearing section is either present WHOLE or the
prompt is refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.training import prompt_budget as pb  # noqa: E402


def words(text: str) -> int:
    """A stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def sec(name: str, n_words: int) -> str:
    return f"## {name}\n" + " ".join(f"w{i}" for i in range(n_words)) + "\n\n"


# Shapes measured on the real corpus, in miniature.
TOOL_LOOP = (
    sec("Human Instructions", 20) + sec("Task", 30)
    + sec("Engineering Principles (Symbiotic AI-Native Manifesto)", 25)
    + sec("Recent Episodes (your short-term memory)", 60)
    + sec("Target: backend/api/x.py [SHA-256: abc] [9457 bytes]", 8)
    + sec("Structural Index (what already exists — DO NOT duplicate)", 120)
    + sec("Available Tools", 150) + sec("Output Schema", 40)
    + sec("Strategic Direction (Manifesto v4)", 30)
    + sec("Recent Development Momentum", 40)
    + sec("What Happened", 90)          # a memory file's own heading, promoted
    + sec("Rust Subsystems", 30)
)
REPAIR = (
    sec("Human Instructions", 20) + sec("Task", 30)
    + sec("Codebase Character", 50) + sec("Inferred Direction (hypotheses)", 60)
    + sec("Fault Localization (hierarchical, AST-narrowed)", 80)
    + sec("NEGATIVE CONSTRAINTS (do NOT repeat these mistakes)", 40)
    + sec("Structural Index (what already exists)", 70)
    + sec("Source Snapshot", 2)
    + sec("File: backend/api/x.py [SHA-256: 60b7]", 90)
)


def names(text: str):
    return [n for n, _ in pb.split_sections(text)[1]]


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_a_prompt_that_fits_is_returned_byte_for_byte() -> None:
    out, rep = pb.fit_prompt(TOOL_LOOP, budget=10_000, count=words)
    assert out == TOOL_LOOP
    assert not rep.dropped and not rep.refused and rep.after == rep.before


def test_protected_sections_survive_and_ambient_ones_go() -> None:
    out, rep = pb.fit_prompt(TOOL_LOOP, budget=420, count=words)
    kept = names(out)
    for must in ("Human Instructions", "Task", "Structural Index", "Available Tools",
                 "Output Schema"):
        assert any(k.startswith(must) for k in kept), must
    assert not any(k.startswith("What Happened") for k in kept)
    assert not any(k.startswith("Recent Development Momentum") for k in kept)
    assert words(out) <= 420


def test_no_section_is_ever_cut() -> None:
    """Every kept section is the original section, whole."""
    out, _ = pb.fit_prompt(TOOL_LOOP, budget=420, count=words)
    original = dict(pb.split_sections(TOOL_LOOP)[1])
    for name, body in pb.split_sections(out)[1]:
        assert body == original[name], name


def test_document_order_is_preserved() -> None:
    out, _ = pb.fit_prompt(TOOL_LOOP, budget=420, count=words)
    order = names(TOOL_LOOP)
    kept = names(out)
    assert kept == [n for n in order if n in kept]


def test_ambient_sections_fill_the_remaining_budget_in_document_order() -> None:
    """The protected core is 396 words (headings count under the fake
    tokenizer). At 430 there is room for the 31-word Engineering
    Principles that follows Task, and not for the 65-word Recent Episodes."""
    core = sum(words(b) for n, b in pb.split_sections(TOOL_LOOP)[1] if pb.is_protected(n))
    assert core == 396
    out, rep = pb.fit_prompt(TOOL_LOOP, budget=430, count=words)
    assert any(k.startswith("Engineering Principles") for k in names(out))
    assert any(d.startswith("## Recent Episodes") for d in rep.dropped)


def test_an_unknown_heading_is_ambient_by_default() -> None:
    """The safe default. A wrongly dropped section costs context; a wrongly
    kept one can blow the budget and kill the run."""
    text = sec("Human Instructions", 5) + sec("Task", 5) + sec("Brand New Section", 500)
    out, rep = pb.fit_prompt(text, budget=100, count=words)
    assert not any(k.startswith("Brand New") for k in names(out))
    assert rep.dropped == ["## Brand New Section"]


def test_refuses_rather_than_clips_when_the_core_does_not_fit() -> None:
    text = sec("Human Instructions", 50) + sec("Task", 50) + sec("Available Tools", 200)
    out, rep = pb.fit_prompt(text, budget=100, count=words)
    assert rep.refused
    assert out == text, "a refused prompt is returned UNCHANGED, never clipped"
    assert rep.after == words(text), "reports the core it could not afford"


def test_heading_normalisation_reaches_the_allowlist() -> None:
    assert pb.is_protected("Target: backend/api/x.py [SHA-256: abc]")
    assert pb.is_protected("File: backend/api/x.py [SHA-256: 60b7]")
    assert pb.is_protected("NEGATIVE CONSTRAINTS (do NOT repeat these mistakes)")
    assert pb.is_protected("Fault Localization (hierarchical, AST-narrowed)")
    assert pb.is_protected("Structural Index (what already exists — DO NOT duplicate)")
    assert not pb.is_protected("Recent Episodes (your short-term memory)")
    assert not pb.is_protected("What Happened")


def test_preamble_before_the_first_heading_is_kept() -> None:
    text = "Read this first.\n\n" + sec("Task", 5) + sec("Ambient", 300)
    out, _ = pb.fit_prompt(text, budget=50, count=words)
    assert out.startswith("Read this first.")


def test_both_measured_families_fit_a_realistic_budget() -> None:
    """The real numbers: cores of ~3,000 tokens against 4,096."""
    for text in (TOOL_LOOP, REPAIR):
        out, rep = pb.fit_prompt(text, budget=450, count=words)
        assert not rep.refused
        assert words(out) <= 450
    out, _ = pb.fit_prompt(REPAIR, budget=450, count=words)
    for must in ("Fault Localization", "NEGATIVE CONSTRAINTS", "File:", "Source Snapshot"):
        assert any(k.startswith(must) for k in names(out)), must


def test_summary_line_reports_what_the_budget_cost() -> None:
    _, a = pb.fit_prompt(TOOL_LOOP, budget=420, count=words)
    _, b = pb.fit_prompt(TOOL_LOOP, budget=10_000, count=words)
    line = pb.summarize([a, b])
    assert "2 prompt(s), 1 trimmed, 0 refused" in line
    assert "prompt budget 420" in line


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_the_loader_demands_a_counter_with_a_budget(tmp_path) -> None:
    pytest.importorskip("datasets")
    from reactor_core.training.grpo_pipeline import build_prompt_dataset
    with pytest.raises(ValueError, match="count_tokens"):
        build_prompt_dataset(tmp_path, max_prompt_tokens=4096)


def test_the_budget_reaches_the_loader_from_build_trainer() -> None:
    import inspect
    from reactor_core.training import grpo_pipeline as gp
    src = inspect.getsource(gp.build_trainer)
    assert "max_prompt_tokens=max_prompt_tokens" in src
    assert "count_tokens=count_tokens" in src
    assert "AutoTokenizer.from_pretrained(model_id)" in src, (
        "the budget must be counted in THIS model's tokens")
