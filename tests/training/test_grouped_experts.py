"""The grouped experts path: one host sync per layer, same answer as eager.

2026-10-08 profile of the served 30B (16 sequences decoding): 9,088-31,461
host<->device syncs per token, wall = 44 us x syncs, GPU idle 73-83% -- the
eager loop's per-expert ``torch.where`` / tensor-as-bool / ``.item()`` /
dequantize. These run on CPU (bitsandbytes 0.50 has a CPU NF4 path), against
a real ``Qwen3MoeExperts`` whose fused parameters are swapped for stacks
exactly as the quantizer does it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")
transformers = pytest.importorskip("transformers")

from reactor_core.training import moe_quant as mq  # noqa: E402


# ---------------------------------------------------------------------------
# The stack: fused storage, one dequantize for any subset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(6, 128, 128), (6, 128, 64), (5, 64, 64)])
def test_gather_is_exactly_the_full_dequantization_row_by_row(shape):
    """(6,128,128): 256 blocks/expert -> nested statistics. (6,128,64) and
    (5,64,64): 128 / 64 blocks -> un-nested (a nested group would straddle
    experts). Every row must equal the whole tensor's dequantization."""

    torch.manual_seed(0)
    fused = torch.randn(*shape, dtype=torch.bfloat16)
    stack = mq.QuantizedExpertStack.from_fused(fused)
    assert stack.nested == ((shape[1] * shape[2] // mq.DEFAULT_BLOCKSIZE) % mq._nested_blocksize() == 0)
    full = stack.gather(torch.arange(shape[0]))
    assert full.shape == shape and full.dtype == torch.bfloat16
    # the subset, out of order and repeated, is the same rows
    idx = torch.tensor([3, 0, 3, shape[0] - 1])
    assert torch.equal(stack.gather(idx), full[idx])
    assert torch.equal(stack.dequantize(2), full[2])
    # and NF4 is close to the source (quantization error, not a layout error)
    assert (full.float() - fused.float()).abs().mean() < 0.15


def test_a_block_that_would_straddle_two_experts_is_refused():
    with pytest.raises(ValueError, match="straddle"):
        mq.QuantizedExpertStack.from_fused(torch.randn(3, 10, 10, dtype=torch.bfloat16))


def test_the_stack_still_answers_the_per_expert_protocol():
    stack = mq.QuantizedExpertStack.from_fused(torch.randn(4, 64, 64, dtype=torch.bfloat16))
    assert len(stack) == 4 and stack.shape == torch.Size((4, 64, 64))
    with torch.no_grad():
        assert torch.equal(stack[torch.tensor(1)], stack.gather(torch.tensor([1]))[0])


# ---------------------------------------------------------------------------
# A real Qwen3MoeExperts, quantized the way the loader does it
# ---------------------------------------------------------------------------

def _experts(impl, *, num_experts=8, hidden=128, inter=64, seed=0):
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    cfg = Qwen3MoeConfig(hidden_size=hidden, moe_intermediate_size=inter, num_experts=num_experts,
                         num_experts_per_tok=2, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, intermediate_size=inter, vocab_size=64)
    cfg._experts_implementation = impl
    torch.manual_seed(seed)
    m = Qwen3MoeExperts(cfg)
    for attr in ("gate_up_proj", "down_proj"):
        fused = (torch.randn_like(getattr(m, attr), dtype=torch.float32) * 0.05).to(torch.bfloat16)
        del m._parameters[attr]
        m.add_module(attr, mq.QuantizedExpertStack.from_fused(fused))
    return m


def _same_weights(a, b):
    for attr in ("gate_up_proj", "down_proj"):
        b.add_module(attr, getattr(a, attr))


ROUTINGS = {
    "uneven": [[0, 1], [0, 2], [0, 3], [0, 1], [5, 0], [0, 7]],       # expert 0 takes almost everything
    "one_expert_hit": [[4, 4], [4, 4], [4, 4]],
    "single_token": [[6, 2]],
    "every_expert": [[0, 1], [2, 3], [4, 5], [6, 7]],
}


@pytest.mark.parametrize("name", sorted(ROUTINGS))
def test_grouped_matches_eager_under_any_routing(name):
    eager, grouped = _experts("eager"), _experts(mq.EXPERTS_IMPL)
    _same_weights(eager, grouped)
    idx = torch.tensor(ROUTINGS[name])
    x = torch.randn(idx.shape[0], 128, dtype=torch.bfloat16)
    w = torch.softmax(torch.randn(idx.shape, dtype=torch.float32), dim=-1)
    with torch.no_grad():
        a, b = eager(x, idx, w), grouped(x, idx, w)
    assert a.shape == b.shape and a.dtype == b.dtype
    torch.testing.assert_close(b.float(), a.float(), atol=2e-2, rtol=2e-2)


def test_no_tokens_is_an_empty_answer():
    m = _experts(mq.EXPERTS_IMPL)
    with torch.no_grad():
        out = m(torch.zeros(0, 128, dtype=torch.bfloat16), torch.zeros(0, 2, dtype=torch.long),
                torch.zeros(0, 2))
    assert out.shape == (0, 128)


def test_expert_parallel_sentinels_contribute_nothing():
    m = _experts(mq.EXPERTS_IMPL)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    w = torch.full((2, 2), 0.5)
    with torch.no_grad():
        with_sentinel = m(x, torch.tensor([[1, 8], [2, 8]]), w)     # 8 == num_experts: not ours
        alone = m(x, torch.tensor([[1, 1], [2, 2]]), torch.tensor([[0.5, 0.0], [0.5, 0.0]]))
    torch.testing.assert_close(with_sentinel.float(), alone.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("chunk", [1, 2, 3])
def test_chunking_does_not_change_the_answer(monkeypatch, chunk):
    m = _experts(mq.EXPERTS_IMPL)
    idx = torch.tensor(ROUTINGS["uneven"])
    x = torch.randn(idx.shape[0], 128, dtype=torch.bfloat16)
    w = torch.softmax(torch.randn(idx.shape), dim=-1)
    with torch.no_grad():
        whole = m(x, idx, w)
        monkeypatch.setattr(mq, "_dequant_chunk", lambda hit, nbytes, device: min(hit, chunk))
        parts = m(x, idx, w)
    torch.testing.assert_close(parts.float(), whole.float(), atol=1e-2, rtol=1e-2)


def test_an_oom_halves_the_chunk_and_still_answers(monkeypatch):
    m = _experts(mq.EXPERTS_IMPL)
    idx = torch.tensor(ROUTINGS["every_expert"])
    x = torch.randn(idx.shape[0], 128, dtype=torch.bfloat16)
    w = torch.softmax(torch.randn(idx.shape), dim=-1)
    with torch.no_grad():
        want = m(x, idx, w)
    real = mq.QuantizedExpertStack.gather
    seen = []

    def tight(self, index):
        seen.append(int(index.numel()))
        if index.numel() > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(self, index)
    monkeypatch.setattr(mq.QuantizedExpertStack, "gather", tight)
    with torch.no_grad():
        got = m(x, idx, w)
    torch.testing.assert_close(got.float(), want.float(), atol=1e-2, rtol=1e-2)
    assert max(seen) > 2 and seen[-1] <= 2


def test_one_host_read_per_layer_and_no_item(monkeypatch):
    """The point of the whole path: the counts are read once; nothing calls
    .item() and nothing loops over experts on device values."""
    m = _experts(mq.EXPERTS_IMPL)
    idx = torch.tensor(ROUTINGS["uneven"])
    x = torch.randn(idx.shape[0], 128, dtype=torch.bfloat16)
    w = torch.softmax(torch.randn(idx.shape), dim=-1)
    reads = {"tolist": 0}
    real_tolist = torch.Tensor.tolist

    def counted_tolist(self):
        reads["tolist"] += 1
        return real_tolist(self)

    def no_item(self):
        raise AssertionError(".item() is a host sync the grouped path must not make")
    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    monkeypatch.setattr(torch.Tensor, "item", no_item)
    with torch.no_grad():
        m(x, idx, w)
    assert reads["tolist"] == 1


def test_training_grads_match_eager_and_no_weight_is_retained():
    eager, grouped = _experts("eager"), _experts(mq.EXPERTS_IMPL)
    _same_weights(eager, grouped)
    idx = torch.tensor(ROUTINGS["uneven"])
    w = torch.softmax(torch.randn(idx.shape), dim=-1)
    base = torch.randn(idx.shape[0], 128, dtype=torch.bfloat16)
    grads = []
    for m in (eager, grouped):
        x = base.clone().requires_grad_(True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            m(x, idx, w).float().pow(2).sum().backward()
        grads.append(x.grad.float())
        if m is grouped:
            # nothing the size of a dequantized expert was kept for backward
            expert_elems = 128 * 128
            assert all(t.numel() < expert_elems or t.dtype == torch.uint8 for t in saved)
    torch.testing.assert_close(grads[1], grads[0], atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# Registration and the operator's choice
# ---------------------------------------------------------------------------

def test_the_grouped_implementation_is_registered():
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    assert ALL_EXPERTS_FUNCTIONS[mq.EXPERTS_IMPL] is mq.grouped_experts_forward


@pytest.mark.parametrize("env, want", [("", None), ("eager", "eager")])
def test_the_operator_may_choose_eager(monkeypatch, env, want):
    monkeypatch.setenv("REACTOR_MOE_EXPERTS_IMPL", env)
    assert mq.experts_implementation() == (want or mq.EXPERTS_IMPL)


def test_a_dense_kernel_is_refused(monkeypatch):
    monkeypatch.setenv("REACTOR_MOE_EXPERTS_IMPL", "grouped_mm")
    with pytest.raises(ValueError, match="packed bytes|support"):
        mq.experts_implementation()
