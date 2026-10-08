"""Per-expert NF4 for fused mixture-of-experts checkpoints.

## The problem, measured 2026-09-04

transformers 5.16 stores routed experts as two 3-D ``nn.Parameter``s per
layer -- ``experts.gate_up_proj`` ``[E, 2I, H]`` and ``experts.down_proj``
``[E, H, I]`` -- and fuses the per-expert 2-D checkpoint tensors into them
while loading. bitsandbytes' 4-bit quantizer only ever touches
``bnb.nn.Linear4bit`` (``param_needs_quantization`` is literally an
``isinstance`` check), so those parameters stay bf16. For
Qwen3-Coder-30B-A3B that is 48 x 128 x 4.7M = 29.0B of its 30.5B parameters,
~54 GiB, on a 32 GiB card. The "bnb-NF4 30B" was never 4-bit; the profile
OOM'd inside ``from_pretrained`` and, with the CUDA sysmem fallback on,
took the desktop down first.

## What this module does

It registers a quantization method, ``bnb_4bit_moe``, through transformers'
PUBLIC registry (``register_quantization_config`` / ``register_quantizer``).
Its quantizer is the stock 4-bit one plus exactly one rule: a 3-D floating
parameter that lives on an ``experts`` module is quantized too, expert by
expert, into a :class:`QuantizedExpertStack` -- a module that answers
``stack[i]`` with the dequantized 2-D weight of expert ``i``.

That is the whole trick. The models' own eager expert forward already
indexes the fused parameter that way::

    gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, -1)

so nothing in the model is patched, subclassed or re-implemented. The
parameter is replaced by a module that speaks the same one-line protocol,
and the forward pass, the routing, and the backward pass are the model's
own. Names and dimensions are DISCOVERED from the instantiated (meta)
model, never assumed: any 3-D floating parameter on any module whose name
contains "expert" qualifies, whatever it is called and however wide it is.

This is not a reshape. NF4 blocks are 64 elements along the flattened
tensor, so a 2-D ``[out, in]`` expert quantized on its own has exactly the
layout and numerics ``Linear4bit`` would give the same matrix. It IS the
3-D-to-2-D unbinding, done at the one moment the fused tensor exists on the
device and before it is retained anywhere.

## Dispatch: grouped, one host sync per layer (2026-10-08)

The eager loop paid ~5 host<->device synchronisations per ACTIVATED EXPERT
per layer -- ``torch.where`` (a nonzero), ``expert_idx == num_experts`` (a
tensor compared as a Python bool), ``int(index.item())`` here, and
bitsandbytes' per-expert dequantize -- plus ~11 kernel launches each.
Profiled live on the 30B, 16 sequences decoding: 9,088-31,461 syncs per
token, wall = 44 us x syncs at every context, the GPU idle 73-83% of the
time. The count followed how many distinct experts the group routed to per
layer, not the context length.

So the stack is stored FUSED: one packed NF4 buffer per projection, every
expert an aligned row (NF4 blocks of 64 never straddle experts, nor do the
nested-statistics blocks of 256 -- checked at quantization, and a shape that
would straddle them is quantized without nesting instead). The registered
experts implementation :func:`grouped_experts_forward`:

1. sorts the (token, expert) assignments by expert on the device;
2. reads the per-expert counts to the host ONCE per layer -- the only sync;
3. dequantizes ONLY the experts actually hit, all in one bitsandbytes call
   (their rows gathered by an on-device index -- no per-expert indexing),
   in as many chunks as free device memory allows (:func:`_dequant_chunk`),
   halving a chunk that still hits OOM;
4. runs transformers' own ``_grouped_linear`` for both projections and the
   model's own gate, then restores the order and reduces exactly as
   transformers' ``grouped_mm`` path does.

Under autograd the projection is :class:`_GroupedExpertLinear`: it keeps
only the offsets and the hit indices and re-dequantizes in the backward, so
no layer's bf16 experts are retained -- the reason ``_ExpertMatmul`` exists,
kept for the grouped path. ``REACTOR_MOE_EXPERTS_IMPL=eager`` restores the
per-expert loop (``stack[i]`` still answers it).

## Constraints that are enforced, not documented

* **A known dispatch only.** ``grouped_mm`` / ``batched_mm`` consume a dense
  3-D tensor and would read packed bytes as bf16 -- not loudly; they would
  train garbage. The quantizer pins this module's implementation (or eager)
  on every config that owns an expert module after loading.
* **Every discovered expert tensor must end up quantized.** If one is still a
  bf16 ``Parameter`` after loading, the quantizer raises. A model that is
  silently half-quantized is the failure this file exists to end.
* **Not serializable.** Only the LoRA adapter is ever saved; a base model
  with stacks in it would need a matching loader, and nothing here writes
  one.

## Memory during training

``stack[i]`` returns a fresh bf16 tensor that autograd keeps for
``grad_input``. Under gradient checkpointing (the pipeline default) that is
one layer's live experts at a time -- ~1.2 GiB for the 30B -- not all 48.
Without checkpointing it would be every expert of every layer; do not turn
it off with this quantizer.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from transformers.core_model_loading import ConversionOps
from transformers.integrations.bitsandbytes import Bnb4bitQuantize
from transformers.quantizers import register_quantization_config, register_quantizer
from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer
from transformers.utils.quantization_config import BitsAndBytesConfig

logger = logging.getLogger(__name__)

#: The name transformers' registry knows this method by.
QUANT_METHOD = "bnb_4bit_moe"

#: ``Linear4bit``'s default; ``BitsAndBytesConfig`` carries no blocksize.
DEFAULT_BLOCKSIZE = 64

#: The experts implementation this module registers, and the env that may
#: choose eager instead.
EXPERTS_IMPL = "packed_nf4_grouped"
_ENV_EXPERTS_IMPL = "REACTOR_MOE_EXPERTS_IMPL"
#: Share of the device memory free at the moment a layer runs that one
#: chunk of dequantized experts may take.
_ENV_DEQUANT_FRACTION = "REACTOR_MOE_DEQUANT_FRACTION"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def is_fused_expert_param(module_path: str, param: torch.Tensor) -> bool:
    """A 3-D floating parameter on a module whose own name says "expert".

    The MODULE name is tested, not the parameter name, because vendors
    agree on the former (``experts``) far more than the latter
    (``gate_up_proj`` / ``w13_weight`` / ``up_proj``). 2-D parameters on an
    experts module (a bias, a shared expert's Linear) are not fused
    experts; 3-D parameters elsewhere (a conv) are not experts at all.
    """
    leaf = module_path.rsplit(".", 1)[-1].lower()
    return param.ndim == 3 and param.is_floating_point() and "expert" in leaf


def discover_fused_expert_params(model: nn.Module) -> Dict[str, torch.Size]:
    """``{fully.qualified.name: shape}`` for every fused expert parameter.

    Walks the instantiated model (meta tensors are fine -- shapes and
    dtypes are real, storage is not), so it adapts to whatever the config
    produced. Deterministic order: the model's own.
    """
    found: Dict[str, torch.Size] = {}
    for module_path, module in model.named_modules():
        for attr, param in module.named_parameters(recurse=False):
            if is_fused_expert_param(module_path, param):
                found[f"{module_path}.{attr}" if module_path else attr] = param.shape
    return found


# ---------------------------------------------------------------------------
# The stack: a Parameter's one-line protocol, implemented by a Module
# ---------------------------------------------------------------------------


class _ExpertMatmul(torch.autograd.Function):
    """``F.linear(x, W)`` where W is reconstructed from 4-bit on BOTH passes.

    The point is what is NOT saved. ``ctx.save_for_backward`` gets the
    PACKED parameter; the dequantized bf16 weight is built, used and
    dropped inside each pass.

    Measured 2026-09-05 with ``saved_tensors_hooks`` on a tiny MoE: the
    plain ``F.linear(x, stack[i])`` path saved one dequantized bf16 copy
    per hit expert -- 1792 KiB of expert weights against 0 bytes of packed
    ones, an expansion of 4.0x -- because autograd must keep the weight to
    compute ``grad_input``. On the 30B that is 128 experts x 9.00 MB =
    ~1.12 GiB of live bf16 per layer, which is the 1.23-1.49 GiB
    allocation every ladder rung died on.

    The experts are FROZEN (the adapter is on attention), so no gradient
    flows back to the weight and only ``grad_input`` is returned. That is
    what makes recompute strictly cheaper than retention here: the backward
    needs W transiently and nothing needs it afterwards.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, packed: torch.Tensor,
                stack: "QuantizedExpertStack", index: int) -> torch.Tensor:
        weight = stack.dequantize(index)
        out = nn.functional.linear(x, weight)
        ctx.save_for_backward(x, packed)
        ctx.stack = stack
        ctx.index = index
        del weight
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # noqa: ANN001
        _x, _packed = ctx.saved_tensors
        weight = ctx.stack.dequantize(ctx.index)
        # grad_input only: the expert weight is frozen, so autograd needs
        # nothing back for it and the reconstruction can be dropped here.
        grad_x = grad_out @ weight
        del weight
        return grad_x, None, None, None


class _PackedExpertWeight(torch.Tensor):
    """A weight that carries a 4-bit expert and expands only when used.

    It has the expert's real shape, dtype and device but NO storage, so
    holding one costs nothing. ``F.linear`` against it is intercepted and
    routed through :class:`_ExpertMatmul`; every other operation falls
    through to the dequantized tensor, so the object still behaves like the
    weight it stands for.

    This is what keeps the model unpatched. ``Qwen3MoeExperts.forward``
    still does ``F.linear(current_state, self.gate_up_proj[expert_idx])``
    on an object it indexes out of an attribute -- exactly as before -- and
    the memory behaviour changes underneath it.
    """

    __slots__ = ["_stack", "_index"]

    @staticmethod
    def __new__(cls, stack: "QuantizedExpertStack", index: int) -> "_PackedExpertWeight":
        shape = torch.Size(tuple(stack.shape[1:]))
        obj = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, shape, dtype=stack.dtype, device=stack.device,
            requires_grad=False,
        )
        obj._stack = stack
        obj._index = int(index)
        return obj

    def materialize(self) -> torch.Tensor:
        return self._stack.dequantize(self._index)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        if func is nn.functional.linear and len(args) >= 2 and isinstance(args[1], cls):
            x, w = args[0], args[1]
            if len(args) > 2 and args[2] is not None:
                raise NotImplementedError("quantized experts carry no bias")
            return _ExpertMatmul.apply(x, w._stack.packed[w._index],
                                       w._stack, w._index)
        # Anything else gets the real tensor. Correctness first: an
        # unrecognised op must see the weight, not a hollow shell.
        return func(*cls._real(args), **{k: cls._real1(v) for k, v in kwargs.items()})

    @classmethod
    def _real1(cls, a):  # noqa: ANN001
        return a.materialize() if isinstance(a, cls) else a

    @classmethod
    def _real(cls, args):  # noqa: ANN001
        return [cls._real1(a) for a in args]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # noqa: ANN001
        """Fallback for anything that reaches the dispatcher directly.

        ``_make_wrapper_subclass`` requires this to exist. It must also be
        CORRECT rather than a stub: a path that skips ``__torch_function__``
        would otherwise operate on a tensor with no storage.
        """
        return func(*cls._real(args), **{k: cls._real1(v)
                                         for k, v in (kwargs or {}).items()})


class QuantizedExpertStack(nn.Module):
    """``stack[i]`` is the 2-D weight of expert ``i``; ``stack.gather(idx)``
    is the weights of the experts in ``idx``, ``[len(idx), out, in]``, from
    ONE dequantize call.

    Storage is fused: ``packed`` is ``[E, out*in/2]`` NF4 bytes (a frozen
    Parameter, so it moves and counts with the model), ``absmax`` the
    per-block scales ``[E, blocks]`` -- uint8 when the statistics are nested,
    with ``absmax2`` ``[E, blocks/blocksize2]`` and one ``offset`` -- and the
    codebooks. Replaces a fused ``[E, out, in]`` parameter in place, on the
    same attribute.
    """

    def __init__(self, packed: torch.Tensor, quant_state: Any, shape: torch.Size,
                 dtype: torch.dtype) -> None:
        super().__init__()
        e = int(shape[0])
        qs = quant_state
        self.packed = nn.Parameter(packed.reshape(e, -1), requires_grad=False)
        self.register_buffer("absmax", qs.absmax.reshape(e, -1))
        self.register_buffer("code", qs.code)
        self.nested = bool(getattr(qs, "nested", False) and qs.state2 is not None)
        if self.nested:
            self.register_buffer("absmax2", qs.state2.absmax.reshape(e, -1))
            self.register_buffer("code2", qs.state2.code)
            self.register_buffer("offset", torch.as_tensor(qs.offset))
            self.blocksize2 = int(qs.state2.blocksize)
            self.absmax2_dtype = qs.state2.dtype
        self.blocksize = int(qs.blocksize)
        self.quant_type = str(qs.quant_type)
        self._shape = torch.Size(shape)
        self._dtype = dtype

    @classmethod
    def from_fused(
        cls,
        fused: torch.Tensor,
        *,
        quant_type: str = "nf4",
        compress_statistics: bool = True,
        blocksize: int = DEFAULT_BLOCKSIZE,
        quant_storage: torch.dtype = torch.uint8,
    ) -> "QuantizedExpertStack":
        """Quantize a fused ``[E, out, in]`` tensor in one call, with every
        expert an aligned row of the result.

        NF4 blocks run along the flattened tensor, so an expert of ``out*in``
        elements owns whole blocks when ``out*in`` is a multiple of
        ``blocksize`` (refused otherwise: a block straddling two experts has
        no per-expert meaning). The nested statistics group ``blocksize2``
        blocks; when an expert's block count is not a multiple of it, the
        statistics are kept un-nested rather than shared across experts.
        """
        import bitsandbytes.functional as bnbf  # noqa: PLC0415

        if fused.ndim != 3:
            raise ValueError(f"expected a fused [E, out, in] tensor, got shape {tuple(fused.shape)}")
        if not fused.is_floating_point():
            raise ValueError(f"expected floating weights, got {fused.dtype}")
        if quant_storage != torch.uint8:
            raise ValueError(f"fused expert stacks pack into uint8, not {quant_storage}")
        e, out, inp = (int(s) for s in fused.shape)
        per_expert = out * inp
        if per_expert % blocksize:
            raise ValueError(f"an expert of {out}x{inp} does not divide into {blocksize}-element "
                             "blocks; a block would straddle two experts")
        nest = bool(compress_statistics)
        if nest:
            nested_block = _nested_blocksize()
            if (per_expert // blocksize) % nested_block:
                logger.info("[moe-quant] %d blocks per expert do not divide into nested groups of %d; "
                            "keeping this stack's statistics un-nested", per_expert // blocksize,
                            nested_block)
                nest = False
        flat = fused.detach().reshape(e * out, inp).contiguous()
        packed, qs = bnbf.quantize_4bit(flat, blocksize=blocksize, compress_statistics=nest,
                                        quant_type=quant_type, quant_storage=quant_storage)
        stack = cls(packed, qs, fused.shape, fused.dtype)
        stack._is_hf_initialized = True
        return stack

    # -- the protocol ------------------------------------------------------

    def gather(self, index: torch.Tensor) -> torch.Tensor:
        """The bf16 weights of the experts in ``index`` (a 1-D integer tensor
        on this stack's device), ``[len(index), out, in]``, from ONE
        dequantize call. Rows are gathered on the device; nothing here reads
        a value back to the host. Allocates."""
        import bitsandbytes.functional as bnbf  # noqa: PLC0415
        from bitsandbytes.functional import QuantState  # noqa: PLC0415

        idx = index.reshape(-1).to(device=self.packed.device, dtype=torch.long)
        h = int(idx.numel())
        out, inp = int(self._shape[1]), int(self._shape[2])
        state2 = None
        if self.nested:
            state2 = QuantState(absmax=self.absmax2.index_select(0, idx).reshape(-1), code=self.code2,
                                blocksize=self.blocksize2, dtype=self.absmax2_dtype)
        qs = QuantState(absmax=self.absmax.index_select(0, idx).reshape(-1),
                        shape=torch.Size((h * out, inp)), code=self.code, blocksize=self.blocksize,
                        quant_type=self.quant_type, dtype=self._dtype,
                        offset=self.offset if self.nested else None, state2=state2)
        packed = self.packed.index_select(0, idx).reshape(-1, 1)
        return bnbf.dequantize_4bit(packed, qs).view(h, out, inp)

    def dequantize(self, index: int) -> torch.Tensor:
        """The real 2-D bf16 weight of expert ``index``. Allocates."""
        return self.gather(torch.arange(int(index), int(index) + 1, device=self.packed.device))[0]

    @property
    def expert_bytes(self) -> int:
        """What one dequantized expert costs."""
        return int(self._shape[1]) * int(self._shape[2]) * torch.tensor([], dtype=self._dtype).element_size()

    def __getitem__(self, index: Any) -> torch.Tensor:
        """The expert's weight -- expanded lazily, and never retained.

        Returns a :class:`_PackedExpertWeight`: correct shape, dtype and
        device, no storage, and ``F.linear`` against it routes through
        :class:`_ExpertMatmul` so the backward keeps the PACKED expert
        rather than the dequantized one. Under ``no_grad`` (generation)
        there is nothing to retain, so the plain dequantized tensor is
        returned and the extra indirection is skipped.
        """
        if isinstance(index, torch.Tensor):
            # The eager loop indexes with a 0-d tensor from `nonzero()`.
            index = int(index.item())
        index = int(index)
        if not torch.is_grad_enabled():
            return self.dequantize(index)
        return _PackedExpertWeight(self, index)

    def __len__(self) -> int:
        return int(self._shape[0])

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self.packed.device

    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:  # pragma: no cover
        raise TypeError("QuantizedExpertStack is indexed per expert (stack[i]); it is not called")

    def extra_repr(self) -> str:
        return (f"experts={len(self)}, expert_shape={tuple(self._shape[1:])}, "
                f"dequantizes_to={self._dtype}")


# ---------------------------------------------------------------------------
# The grouped experts implementation
# ---------------------------------------------------------------------------


def _nested_blocksize() -> int:
    """The block size bitsandbytes groups nested statistics in, asked of
    bitsandbytes itself (a probe quantization) rather than assumed."""
    import bitsandbytes.functional as bnbf  # noqa: PLC0415

    global _NESTED_BLOCKSIZE
    if _NESTED_BLOCKSIZE is None:
        probe = torch.zeros(DEFAULT_BLOCKSIZE, 1)
        _, qs = bnbf.quantize_4bit(probe, blocksize=DEFAULT_BLOCKSIZE, compress_statistics=True)
        _NESTED_BLOCKSIZE = int(qs.state2.blocksize)
    return _NESTED_BLOCKSIZE


_NESTED_BLOCKSIZE: Optional[int] = None


class _GroupedExpertLinear(torch.autograd.Function):
    """``x_g @ W[e]^T`` for each expert's row range, W gathered from 4-bit.

    Keeps the offsets and the hit indices, never the dequantized weights:
    the backward gathers them again. The experts are frozen, so only
    ``grad_input`` flows back -- the same economy as :class:`_ExpertMatmul`,
    for a whole chunk of experts at once.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, offsets: torch.Tensor, stack: "QuantizedExpertStack",
                hit: torch.Tensor) -> torch.Tensor:
        from transformers.integrations.moe import _grouped_linear  # noqa: PLC0415

        weight = stack.gather(hit)
        out = _grouped_linear(x, weight, offsets)
        del weight
        ctx.save_for_backward(offsets, hit)
        ctx.stack = stack
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # noqa: ANN001
        from transformers.integrations.moe import _grouped_linear  # noqa: PLC0415

        offsets, hit = ctx.saved_tensors
        weight = ctx.stack.gather(hit)
        # [C, out, in] is already (input_dim=out, output_dim=in) for grad_out.
        grad_x = _grouped_linear(grad_out.contiguous(), weight, offsets, is_transposed=True)
        del weight
        return grad_x, None, None, None


def _dequant_chunk(hit: int, expert_bytes: int, device: torch.device) -> int:
    """How many hit experts one dequantize may expand at once: as many as
    the configured share of the memory free right now holds (the device's
    free memory plus what the caching allocator holds unused). Never fewer
    than one; all of them off-GPU."""
    if hit <= 0:
        return 0
    if device.type != "cuda" or not torch.cuda.is_available():
        return hit
    try:
        frac = float(os.environ.get(_ENV_DEQUANT_FRACTION, "") or 0.5)
    except ValueError:
        frac = 0.5
    free, _total = torch.cuda.mem_get_info(device)
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    budget = max(0.0, frac) * (free + max(0, cached))
    return max(1, min(hit, int(budget // max(1, expert_bytes))))


def grouped_experts_forward(self: nn.Module, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                            top_k_weights: torch.Tensor) -> torch.Tensor:
    """The experts forward with one host sync per layer (module docstring).

    Mirrors transformers' ``grouped_mm_experts_forward`` -- sort by expert,
    grouped projections, the model's own gate, routing weights, restore the
    order, reduce over top-k in fp32 -- with the weights expanded from 4-bit
    for the hit experts only, chunked to fit."""
    num_tokens, hidden_dim = hidden_states.shape[0], hidden_states.shape[-1]
    num_top_k = top_k_index.shape[-1]
    num_experts = int(self.num_experts)
    device = hidden_states.device

    expert_ids = top_k_index.reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    hidden_g = hidden_states[perm // num_top_k]
    weights_g = sample_weights[perm]

    # The one host read: how many assignments each expert received. Ids at or
    # past num_experts (expert-parallel sentinels) sort to the tail and are
    # left out of every group; their rows contribute zero.
    counts = torch.bincount(expert_ids_g, minlength=num_experts)[:num_experts]
    counts_host = counts.tolist()
    hit_host = [e for e, c in enumerate(counts_host) if c]
    routed = sum(counts_host)
    hit = torch.nonzero_static(counts, size=len(hit_host)).reshape(-1)     # no sync: size is known
    hit_counts = [counts_host[e] for e in hit_host]

    gate_up, down = self.gate_up_proj, self.down_proj
    chunk = _dequant_chunk(len(hit_host), max(gate_up.expert_bytes, down.expert_bytes), device)
    outs: List[torch.Tensor] = []
    e0 = r0 = 0
    while e0 < len(hit_host):
        e1 = min(len(hit_host), e0 + max(1, chunk))
        r1 = r0 + sum(hit_counts[e0:e1])
        try:
            offsets = torch.cumsum(counts.index_select(0, hit[e0:e1]), dim=0, dtype=torch.int32)
            h = _GroupedExpertLinear.apply(hidden_g[r0:r1], offsets, gate_up, hit[e0:e1])
            h = self._apply_gate(h)
            outs.append(_GroupedExpertLinear.apply(h, offsets, down, hit[e0:e1]))
        except torch.cuda.OutOfMemoryError:
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)                 # memory moved under us: halve and retry
            logger.warning("[moe-quant] dequant chunk OOM; retrying with %d expert(s) per chunk", chunk)
            continue
        e0, r0 = e1, r1
    if routed < expert_ids.numel():
        outs.append(torch.zeros(expert_ids.numel() - routed, hidden_dim, device=device,
                                dtype=outs[0].dtype if outs else hidden_states.dtype))
    proj_out = torch.cat(outs, dim=0) if outs else hidden_g.new_zeros(expert_ids.numel(), hidden_dim)

    weighted = proj_out * weights_g.unsqueeze(-1)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=device)
    weighted = weighted[inv_perm]
    return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def experts_implementation() -> str:
    """The implementation to pin: this module's grouped one, unless the
    operator chose eager. Anything else is refused -- a dense kernel would
    read packed bytes as weights."""
    chosen = (os.environ.get(_ENV_EXPERTS_IMPL, "") or EXPERTS_IMPL).strip()
    if chosen not in (EXPERTS_IMPL, "eager"):
        raise ValueError(f"{_ENV_EXPERTS_IMPL}={chosen!r}: quantized expert stacks support "
                         f"{EXPERTS_IMPL!r} or 'eager' only")
    return chosen


def _register_experts_implementation() -> None:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS  # noqa: PLC0415

    ALL_EXPERTS_FUNCTIONS[EXPERTS_IMPL] = grouped_experts_forward


_register_experts_implementation()


# ---------------------------------------------------------------------------
# The registered method
# ---------------------------------------------------------------------------


@register_quantization_config(QUANT_METHOD)
class MoEBitsAndBytesConfig(BitsAndBytesConfig):
    """A ``BitsAndBytesConfig`` whose ``quant_method`` routes to the MoE quantizer.

    Every NF4 knob is the parent's; nothing is re-declared here.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.pop("quant_method", None)  # to_dict() round-trips carry it
        kwargs.setdefault("load_in_4bit", True)
        super().__init__(**kwargs)
        self.quant_method = QUANT_METHOD


def as_moe_config(base: BitsAndBytesConfig) -> MoEBitsAndBytesConfig:
    """The same NF4 settings, routed to the MoE quantizer.

    Takes the pipeline's existing config so the quantization parameters
    have exactly one home (``build_qlora_config``); this only changes who
    applies them.
    """
    if base.load_in_8bit:
        raise ValueError("per-expert quantization is 4-bit only")
    return MoEBitsAndBytesConfig(**base.to_dict())


class MoEBnb4bitQuantize(ConversionOps):
    """The loader-side op: fused expert tensors become stacks, everything
    else goes to the stock 4-bit op untouched."""

    def __init__(self, hf_quantizer: "MoEBnb4BitQuantizer") -> None:
        self.hf_quantizer = hf_quantizer
        self._dense = Bnb4bitQuantize(hf_quantizer)

    def convert(
        self,
        input_dict: Dict[str, Any],
        full_layer_name: Optional[str] = None,
        model: Optional[nn.Module] = None,
        missing_keys: Optional[set] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        expert_names = self.hf_quantizer.expert_params
        dense = {k: v for k, v in input_dict.items() if k not in expert_names}
        result: Dict[str, torch.Tensor] = {}
        if dense:
            result.update(self._dense.convert(
                dense, full_layer_name=full_layer_name, model=model,
                missing_keys=missing_keys, **kwargs,
            ))
        for name, value in input_dict.items():
            if name in dense:
                continue
            value = value[0] if isinstance(value, list) else value
            self.hf_quantizer.install_expert_stack(model, name, value, missing_keys)
        # Stacks were installed on the module directly; there is no tensor
        # for the loader to assign, so none is returned for them.
        return result


@register_quantizer(QUANT_METHOD)
class MoEBnb4BitQuantizer(Bnb4BitHfQuantizer):
    """Stock bnb-4bit, plus per-expert quantization of fused expert tensors."""

    def __init__(self, quantization_config: Any, **kwargs: Any) -> None:
        super().__init__(quantization_config, **kwargs)
        self.expert_params: Dict[str, torch.Size] = {}
        self.installed: List[str] = []

    # -- before loading: find the experts -----------------------------------

    def _process_model_before_weight_loading(self, model: nn.Module, device_map: Any,
                                             **kwargs: Any) -> None:
        super()._process_model_before_weight_loading(model, device_map=device_map, **kwargs)
        self.expert_params = discover_fused_expert_params(model)
        if self.expert_params:
            total = sum(int(torch.Size(s).numel()) for s in self.expert_params.values())
            logger.info(
                "[moe-quant] %d fused expert tensor(s), %.2fB parameters, will be "
                "quantized per expert (%s)", len(self.expert_params), total / 1e9,
                self.quantization_config.bnb_4bit_quant_type,
            )
        else:
            logger.info("[moe-quant] no fused expert tensors found; behaving as "
                        "plain bnb 4-bit")

    def param_needs_quantization(self, model: nn.Module, param_name: str,
                                 **kwargs: Any) -> bool:
        if param_name in self.expert_params:
            return True
        return super().param_needs_quantization(model, param_name, **kwargs)

    def get_quantize_ops(self) -> ConversionOps:
        return MoEBnb4bitQuantize(self)

    # -- during loading: swap the parameter for a stack ----------------------

    def install_expert_stack(self, model: nn.Module, param_name: str,
                             fused: torch.Tensor, missing_keys: Optional[set]) -> None:
        expected = self.expert_params[param_name]
        if tuple(fused.shape) != tuple(expected):
            raise ValueError(
                f"{param_name}: checkpoint gives {tuple(fused.shape)}, the model "
                f"declares {tuple(expected)}")
        module_path, _, attr = param_name.rpartition(".")
        module = model.get_submodule(module_path) if module_path else model
        qc = self.quantization_config
        stack = QuantizedExpertStack.from_fused(
            fused,
            quant_type=qc.bnb_4bit_quant_type,
            compress_statistics=bool(qc.bnb_4bit_use_double_quant),
            blocksize=DEFAULT_BLOCKSIZE,
            quant_storage=qc.bnb_4bit_quant_storage,
        )
        # The attribute keeps its name; only what answers `module.attr[i]` changes.
        if attr in module._parameters:
            del module._parameters[attr]
        module.add_module(attr, stack)
        module._is_hf_initialized = True
        if missing_keys is not None:
            missing_keys.discard(param_name)
        self.installed.append(param_name)

    # -- after loading: prove it, then pin the dispatch ----------------------

    def _process_model_after_weight_loading(self, model: nn.Module, **kwargs: Any) -> nn.Module:
        model = super()._process_model_after_weight_loading(model, **kwargs)
        if not self.expert_params:
            return model
        left_over = [
            name for name in self.expert_params
            if not isinstance(_resolve(model, name), QuantizedExpertStack)
        ]
        if left_over:
            raise RuntimeError(
                f"{len(left_over)} fused expert tensor(s) are still bf16 after "
                f"loading, e.g. {left_over[0]} -- refusing a half-quantized model")
        # Only an implementation that understands the stacks may run them.
        # Pin it on every config an expert module reads (nested text configs
        # included), plus the top level.
        impl = experts_implementation()
        owners = 0
        for name in self.expert_params:
            module = _resolve(model, name.rpartition(".")[0])
            cfg = getattr(module, "config", None)
            if cfg is not None and getattr(cfg, "_experts_implementation", None) != impl:
                cfg._experts_implementation = impl
                owners += 1
        if getattr(model.config, "_experts_implementation", None) != impl:
            model.config._experts_implementation = impl
            owners += 1
        logger.info("[moe-quant] %d expert tensor(s) installed as stacks; experts "
                    "implementation pinned to %s on %d config(s)",
                    len(self.installed), impl, owners)
        return model

    def is_serializable(self, safe_serialization: Any = None) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return True


def _resolve(root: nn.Module, dotted: str) -> Any:
    obj: Any = root
    for part in dotted.split(".") if dotted else []:
        obj = getattr(obj, part)
    return obj


__all__ = [
    "QUANT_METHOD",
    "MoEBitsAndBytesConfig",
    "MoEBnb4BitQuantizer",
    "QuantizedExpertStack",
    "as_moe_config",
    "discover_fused_expert_params",
    "is_fused_expert_param",
]
