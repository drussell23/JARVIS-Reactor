"""Each rung dies in its own process, so the next one starts clean.

Clearing CUDA state in-process is unreliable and this ladder proved it on
2026-09-05. `release_trainer` was enough to let the ladder DESCEND at all
-- before it, rung 2 loaded a second 30B on top of rung 1 and died at 28.22
GiB -- but not enough to descend cleanly:

    rung 1 (16x256)  failed step 3   28.27 GiB allocated
    rung 2 (16x128)  failed step 3   28.27 GiB
    rung 3 (16x64)   failed step 3   28.07 GiB
    rung 4 (8x64)    failed step 1   28.69 GiB

Rung 4 is the SMALLEST configuration and had the LEAST room. At its first
generation it should have held ~15.6 GiB, the model alone with no optimiser
state, and it held 28.69. So "this model does not fit here" was a verdict
about the fourth rung of a leaky descent, not about the model.

The protocol is deliberately just exit codes plus the report file both
halves already use, because a bespoke IPC channel would be one more thing
to get wrong on a path that only runs when something has already failed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = _load("_runner_iso_under_test", _REPO / "scripts" / "run_grpo_training.py")


def _rung(name: str, gens: int = 16, length: int = 256):
    return SimpleNamespace(name=name, num_generations=gens,
                           max_completion_length=length)


class _Spawner:
    """Stands in for subprocess.call, recording what the parent asked for."""

    def __init__(self, codes, reports=None) -> None:
        self.codes = list(codes)
        self.reports = list(reports or [])
        self.calls: list = []

    def __call__(self, cmd, *a, **k):
        self.calls.append(cmd)
        # The child writes its report through --json-out, as in production.
        if self.reports:
            payload = self.reports.pop(0)
            out = cmd[cmd.index("--json-out") + 1]
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        return self.codes.pop(0)


def _run(monkeypatch, codes, reports=None, ladder=None):
    import subprocess
    spawner = _Spawner(codes, reports)
    monkeypatch.setattr(subprocess, "call", spawner)
    monkeypatch.setattr(sys, "argv", ["run_grpo_training.py", "--model", "m"])
    ladder = ladder or [_rung("a"), _rung("b", length=128), _rung("c", 8, 64)]
    out = runner.train_with_isolated_ladder(ladder=ladder, report={})
    return out, spawner


# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------


def test_an_oom_child_descends_to_the_next_rung(monkeypatch) -> None:
    out, spawner = _run(monkeypatch,
                        [runner.EXIT_RUNG_OOM, runner.EXIT_RUNG_OOM, runner.EXIT_OK])
    assert out["status"] == "trained"
    assert len(spawner.calls) == 3, "one process per rung, no reuse"


def test_every_rung_ooming_is_ladder_exhausted(monkeypatch) -> None:
    out, _ = _run(monkeypatch, [runner.EXIT_RUNG_OOM] * 3)
    assert out["status"] == "ladder-exhausted"


def test_a_non_memory_failure_stops_the_descent(monkeypatch) -> None:
    """Descending cannot fix a bug, and three more identical crashes are
    noise on top of the traceback that mattered."""
    out, spawner = _run(monkeypatch, [runner.EXIT_ERROR])
    assert out["status"] == "error" and out["child_exit"] == runner.EXIT_ERROR
    assert len(spawner.calls) == 1


def test_a_refusal_stops_the_descent(monkeypatch) -> None:
    """A gate said no. A smaller rung cannot un-busy a card or thicken a
    corpus, so repeating the refusal three times only buries it."""
    out, spawner = _run(monkeypatch, [runner.EXIT_REFUSED])
    assert out["status"] == "refused"
    assert len(spawner.calls) == 1


def test_each_rung_gets_its_own_index_and_report_path(monkeypatch) -> None:
    _, spawner = _run(monkeypatch, [runner.EXIT_RUNG_OOM] * 3)
    seen = []
    outs = []
    for cmd in spawner.calls:
        seen.append(cmd[cmd.index("--rung-index") + 1])
        outs.append(cmd[cmd.index("--json-out") + 1])
    assert seen == ["0", "1", "2"]
    assert len(set(outs)) == 3, "a shared report file would overwrite itself"


def test_child_attempts_are_merged_with_their_exit_code(monkeypatch) -> None:
    reports = [
        {"attempts": [{"rung": "a", "error": "OutOfMemoryError: ..."}]},
        {"attempts": [{"rung": "b"}]},
    ]
    out, _ = _run(monkeypatch, [runner.EXIT_RUNG_OOM, runner.EXIT_OK], reports,
                  ladder=[_rung("a"), _rung("b")])
    assert [a["rung"] for a in out["attempts"]] == ["a", "b"]
    assert out["attempts"][0]["subprocess_exit"] == runner.EXIT_RUNG_OOM


def test_a_child_that_wrote_nothing_still_descends(monkeypatch) -> None:
    """A process killed by the memory guard leaves no report. Its exit code
    is still the whole answer."""
    out, _ = _run(monkeypatch, [runner.EXIT_RUNG_OOM, runner.EXIT_OK], reports=None,
                  ladder=[_rung("a"), _rung("b")])
    assert out["status"] == "trained"
    assert out["attempts"] == []


# ---------------------------------------------------------------------------
# The argv the child inherits
# ---------------------------------------------------------------------------


def test_child_inherits_the_parents_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_grpo_training.py", "--model", "M", "--train-truncated",
        "--max-completion-length", "256", "--json-out", "/parent.json"])
    argv = runner.child_argv(2, "/child.json")
    assert "--train-truncated" in argv
    assert argv[argv.index("--max-completion-length") + 1] == "256"
    assert argv[argv.index("--rung-index") + 1] == "2"
    assert argv[argv.index("--json-out") + 1] == "/child.json"
    assert "/parent.json" not in argv, "the parent's report must not be clobbered"


def test_child_argv_replaces_rather_than_repeats_rung_index(monkeypatch) -> None:
    """A child must never re-enter parent mode, and a duplicated flag would
    make argparse take the wrong one."""
    monkeypatch.setattr(sys, "argv", [
        "run_grpo_training.py", "--rung-index", "0", "--json-out=/a.json"])
    argv = runner.child_argv(3, "/b.json")
    assert argv.count("--rung-index") == 1
    assert argv[argv.index("--rung-index") + 1] == "3"
    assert not any(a.startswith("--json-out=") for a in argv)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_isolation_is_the_default_and_children_run_in_process() -> None:
    import inspect
    src = inspect.getsource(runner.main)
    assert "args.rung_index < 0 and not args.in_process_ladder" in src, (
        "isolation must be the default; a child is told apart by --rung-index")
    assert "train_with_isolated_ladder(" in src
    # A child that exhausts its single rung must report OOM, not error.
    assert "return EXIT_RUNG_OOM" in src


def test_oom_exit_code_is_distinct_from_error() -> None:
    codes = {runner.EXIT_OK, runner.EXIT_ERROR, runner.EXIT_REFUSED,
             runner.EXIT_LADDER_EXHAUSTED, runner.EXIT_RUNG_OOM}
    assert len(codes) == 5, "the parent's response to OOM is the opposite of error"
