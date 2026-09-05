"""The backward keeps the PACKED expert, not the dequantized one.

Measured 2026-09-05 with ``saved_tensors_hooks`` on a tiny MoE: the plain
``F.linear(x, stack[i])`` path saved one dequantized bf16 copy per hit
expert -- 1792 KiB of expert weights against 0 bytes of packed ones, a 4.0x
expansion -- because autograd must keep the weight to compute
``grad_input``. On the 30B that is 128 experts x 9.00 MB = ~1.12 GiB of
live bf16 per layer, which is the 1.23-1.49 GiB allocation every rung of
the degradation ladder died on.

Correctness is the first thing pinned, and it is pinned NUMERICALLY against
the dequantized reference rather than by inspection. A custom autograd
Function that silently computes the wrong gradient is worse than an OOM: an
OOM stops, a wrong gradient trains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reactor_core.training import moe_quant as mq  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA for bitsandbytes")
NL = chr(10)


def _stack(experts: int = 4, out: int = 64, inp: int = 128):
    torch.manual_seed(0)
    fused = torch.randn(experts, out, inp, dtype=torch.bfloat16, device="cuda")
    return mq.QuantizedExpertStack.from_fused(fused), fused


# ---------------------------------------------------------------------------
# Correctness FIRST
# ---------------------------------------------------------------------------


@cuda
def test_forward_matches_the_dequantized_reference() -> None:
    stack, _ = _stack()
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    for i in range(len(stack)):
        want = torch.nn.functional.linear(x, stack.dequantize(i))
        got = torch.nn.functional.linear(x, stack[i])
        assert torch.equal(got, want), f"expert {i}"


@cuda
def test_grad_input_matches_the_dequantized_reference() -> None:
    """The whole point of a custom Function is that it is INDISTINGUISHABLE
    from the obvious implementation, except in memory."""
    stack, _ = _stack()
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")

    a = x.clone().requires_grad_(True)
    ref = torch.nn.functional.linear(a, stack.dequantize(2))
    ref.sum().backward()

    b = x.clone().requires_grad_(True)
    got = torch.nn.functional.linear(b, stack[2])
    got.sum().backward()

    assert a.grad is not None and b.grad is not None
    assert torch.equal(b.grad, a.grad)


@cuda
def test_grad_flows_through_a_chain_of_experts() -> None:
    stack, _ = _stack(out=128, inp=128)
    x = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")

    def run(get):
        h = x.clone().requires_grad_(True)
        y = h
        for i in range(len(stack)):
            y = torch.nn.functional.linear(y, get(i))
        y.sum().backward()
        return h.grad

    assert torch.equal(run(lambda i: stack[i]),
                       run(lambda i: stack.dequantize(i)))


@cuda
def test_the_weight_has_the_right_metadata_but_no_storage() -> None:
    stack, fused = _stack()
    w = stack[0]
    assert isinstance(w, mq._PackedExpertWeight)
    assert tuple(w.shape) == tuple(fused.shape[1:])
    assert w.dtype == torch.bfloat16
    assert w.device.type == "cuda"


@cuda
def test_unrecognised_ops_see_the_real_weight() -> None:
    """Correctness before cleverness: anything that is not F.linear must be
    handed the actual tensor, never a hollow shell."""
    stack, _ = _stack()
    w = stack[1]
    assert torch.equal(w + 0, stack.dequantize(1))
    assert float(w.sum()) == pytest.approx(float(stack.dequantize(1).sum()))


@cuda
def test_no_grad_returns_the_plain_tensor() -> None:
    """Generation has nothing to retain, so it skips the indirection."""
    stack, _ = _stack()
    with torch.no_grad():
        w = stack[0]
    assert not isinstance(w, mq._PackedExpertWeight)
    assert torch.equal(w, stack.dequantize(0))


# ---------------------------------------------------------------------------
# ...and only THEN, memory
# ---------------------------------------------------------------------------


def _saved_bytes(fn) -> dict:
    """Bytes autograd retains, split by dtype, for one forward."""
    seen: list = []

    def pack(t):
        seen.append((str(t.dtype), t.numel() * t.element_size()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    out.sum().backward()
    by = {}
    for dt, nb in seen:
        by[dt] = by.get(dt, 0) + nb
    return by


@cuda
def test_the_backward_retains_packed_bytes_not_expanded_ones() -> None:
    stack, _ = _stack(experts=4, out=256, inp=256)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")

    def with_stack():
        h = x.clone().requires_grad_(True)
        y = h
        for i in range(len(stack)):
            y = torch.nn.functional.linear(y, stack[i])
        return y

    def with_dequant():
        h = x.clone().requires_grad_(True)
        y = h
        for i in range(len(stack)):
            y = torch.nn.functional.linear(y, stack.dequantize(i))
        return y

    packed = _saved_bytes(with_stack)
    plain = _saved_bytes(with_dequant)

    expert_bf16 = 4 * 256 * 256 * 2          # what the naive path must keep
    plain_bf16 = plain.get("torch.bfloat16", 0)
    packed_bf16 = packed.get("torch.bfloat16", 0)

    assert plain_bf16 >= expert_bf16, "the reference must retain the weights"
    assert packed_bf16 < plain_bf16 - expert_bf16 * 0.9, (
        f"expanded weights are still being retained: {packed_bf16} vs {plain_bf16}")
    assert packed.get("torch.uint8", 0) > 0, "the PACKED expert must be saved"
