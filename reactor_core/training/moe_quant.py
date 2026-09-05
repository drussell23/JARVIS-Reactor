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

## Constraints that are enforced, not documented

* **Eager dispatch only.** ``grouped_mm`` / ``batched_mm`` consume the whole
  3-D tensor; only the eager implementation indexes per expert. The
  quantizer pins ``_experts_implementation = "eager"`` on every config that
  owns an expert module after loading. A kernel that read packed bytes as
  bf16 would not fail loudly; it would train garbage.
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
        param = stack.experts[index]
        obj = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, shape, dtype=stack.dtype, device=param.device,
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
            return _ExpertMatmul.apply(x, w._stack.experts[w._index].data,
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
    """``stack[i]`` is the 2-D weight of expert ``i``.

    Holds one ``Params4bit`` per expert. Replaces a fused ``[E, out, in]``
    parameter in place, on the same attribute, so the owning module's
    forward -- which indexes that attribute by expert -- runs unchanged.
    """

    def __init__(self, experts: List[nn.Parameter], shape: torch.Size,
                 dtype: torch.dtype) -> None:
        super().__init__()
        if len(experts) != int(shape[0]):
            raise ValueError(f"{len(experts)} experts for a stack of shape {tuple(shape)}")
        self.experts = nn.ParameterList(experts)
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
        """Quantize a fused ``[E, out, in]`` tensor expert by expert.

        Each expert is a contiguous 2-D copy while it is quantized, then
        only its packed bytes and quant state survive; the fused input is
        the caller's to drop.
        """
        import bitsandbytes as bnb  # noqa: PLC0415

        if fused.ndim != 3:
            raise ValueError(f"expected a fused [E, out, in] tensor, got shape {tuple(fused.shape)}")
        if not fused.is_floating_point():
            raise ValueError(f"expected floating weights, got {fused.dtype}")
        experts: List[nn.Parameter] = []
        for index in range(int(fused.shape[0])):
            weight = fused[index].detach().contiguous()
            param = bnb.nn.Params4bit(
                weight, requires_grad=False, quant_type=quant_type,
                compress_statistics=compress_statistics, blocksize=blocksize,
                quant_storage=quant_storage,
            ).to(weight.device)  # .to(cuda) is what triggers the quantization
            param._is_hf_initialized = True  # never re-initialised by the loader
            experts.append(param)
        stack = cls(experts, fused.shape, fused.dtype)
        stack._is_hf_initialized = True
        return stack

    # -- the protocol ------------------------------------------------------

    def dequantize(self, index: int) -> torch.Tensor:
        """The real 2-D bf16 weight of expert ``index``. Allocates."""
        import bitsandbytes.functional as bnbf  # noqa: PLC0415

        param = self.experts[int(index)]
        return bnbf.dequantize_4bit(param.data, param.quant_state)

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
        return len(self.experts)

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self.experts[0].device

    def forward(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:  # pragma: no cover
        raise TypeError("QuantizedExpertStack is indexed per expert (stack[i]); it is not called")

    def extra_repr(self) -> str:
        return (f"experts={len(self)}, expert_shape={tuple(self._shape[1:])}, "
                f"dequantizes_to={self._dtype}")


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
        # Only eager indexes per expert. Pin it on every config an expert
        # module reads (nested text configs included), plus the top level.
        owners = 0
        for name in self.expert_params:
            module = _resolve(model, name.rpartition(".")[0])
            cfg = getattr(module, "config", None)
            if cfg is not None and getattr(cfg, "_experts_implementation", "eager") != "eager":
                cfg._experts_implementation = "eager"
                owners += 1
        if getattr(model.config, "_experts_implementation", "eager") != "eager":
            model.config._experts_implementation = "eager"
            owners += 1
        logger.info("[moe-quant] %d expert tensor(s) installed as stacks; experts "
                    "implementation pinned to eager on %d config(s)",
                    len(self.installed), owners)
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
