"""Fused experts get quantized, expert by expert, and nothing else changes.

Pinned here because the failure it prevents was silent right up to the
OOM: on transformers 5.16 a ``BitsAndBytesConfig`` load of a 30B MoE
reported itself as 4-bit while 29.0B of 30.5B parameters -- the fused
``[E, out, in]`` expert tensors -- stayed bf16. bitsandbytes quantizes
``Linear4bit`` and nothing else; the experts are not Linear.

Three layers, cheapest first:

* discovery on meta models (no GPU): the rule finds exactly the fused
  expert tensors of a MoE and nothing on a dense model, with names and
  shapes taken from the model, not assumed;
* the stack on CUDA: ``stack[i]`` is expert ``i`` at NF4 precision, with
  the original 2-D shape and dtype;
* an end-to-end load of a tiny synthetic Qwen3-MoE through
  ``from_pretrained``: experts become stacks, attention becomes
  ``Linear4bit``, the forward agrees with the bf16 model, gradients reach a
  trainable parameter THROUGH the quantized experts, generation runs, and
  the dispatch is pinned to eager.

The pipeline's choice of config is tested with ``from_pretrained`` mocked,
so it runs without a GPU and without a checkpoint.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The real package path, so the registry decorators run once and the
# pipeline's lazy `from reactor_core.training.moe_quant import ...`
# resolves to the same registered classes.
from reactor_core.training import moe_quant as mq  # noqa: E402

pipeline = _load(
    "_pipeline_moe_under_test",
    _REPO / "reactor_core" / "training" / "grpo_pipeline.py",
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for bitsandbytes")


def _tiny_moe_config(**overrides):
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    kw = dict(
        hidden_size=256, intermediate_size=512, moe_intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, num_experts=8, num_experts_per_tok=2, vocab_size=512,
        max_position_embeddings=256, tie_word_embeddings=False,
        decoder_sparse_step=1, mlp_only_layers=[],
    )
    kw.update(overrides)
    return Qwen3MoeConfig(**kw)


def _meta_model(config):
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config)


# ---------------------------------------------------------------------------
# Discovery: names and shapes come from the model
# ---------------------------------------------------------------------------


def test_discovery_finds_exactly_the_fused_expert_tensors() -> None:
    cfg = _tiny_moe_config(num_experts=6, moe_intermediate_size=96, hidden_size=192)
    found = mq.discover_fused_expert_params(_meta_model(cfg))
    assert len(found) == 2 * cfg.num_hidden_layers
    for name, shape in found.items():
        assert ".experts." in name
        assert shape[0] == 6, "first dim is the expert count, read from the model"
        assert tuple(shape) in {(6, 2 * 96, 192), (6, 192, 96)}


def test_discovery_finds_nothing_on_a_dense_model() -> None:
    from transformers import Qwen2Config

    cfg = Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256)
    assert mq.discover_fused_expert_params(_meta_model(cfg)) == {}


def test_rule_needs_three_dims_and_an_experts_module() -> None:
    three_d = torch.empty(4, 8, 8, device="meta")
    two_d = torch.empty(8, 8, device="meta")
    assert mq.is_fused_expert_param("model.layers.0.mlp.experts", three_d)
    assert not mq.is_fused_expert_param("model.layers.0.mlp.experts", two_d), "a bias is not an expert"
    assert not mq.is_fused_expert_param("model.layers.0.self_attn", three_d), "3-D elsewhere is not an expert"
    assert not mq.is_fused_expert_param("model.layers.0.mlp.experts",
                                        torch.empty(4, 8, 8, dtype=torch.int8, device="meta"))


# ---------------------------------------------------------------------------
# Registry: the config routes to the quantizer, and round-trips
# ---------------------------------------------------------------------------


def test_config_routes_to_the_moe_quantizer() -> None:
    from transformers.quantizers import AutoHfQuantizer, AutoQuantizationConfig

    cfg = mq.as_moe_config(pipeline.build_qlora_config())
    assert isinstance(cfg, mq.MoEBitsAndBytesConfig)
    assert cfg.quant_method == mq.QUANT_METHOD
    assert cfg.bnb_4bit_quant_type == "nf4" and cfg.bnb_4bit_use_double_quant is True
    assert isinstance(AutoHfQuantizer.from_config(cfg), mq.MoEBnb4BitQuantizer)
    # Pinned so nobody "simplifies" the pipeline into passing a dict:
    # transformers routes ANY dict with load_in_4bit to plain bnb before it
    # reads quant_method, so the object -- not its dict -- is the contract.
    again = AutoQuantizationConfig.from_dict(cfg.to_dict())
    assert not isinstance(again, mq.MoEBitsAndBytesConfig)


def test_as_moe_config_keeps_the_pipelines_settings(monkeypatch) -> None:
    """The NF4 knobs have one home; this only changes who applies them."""
    base = pipeline.build_qlora_config(bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False)
    cfg = mq.as_moe_config(base)
    assert cfg.bnb_4bit_quant_type == "fp4"
    assert cfg.bnb_4bit_use_double_quant is False
    assert cfg.bnb_4bit_compute_dtype == base.bnb_4bit_compute_dtype


def test_as_moe_config_refuses_8bit() -> None:
    from transformers import BitsAndBytesConfig

    with pytest.raises(ValueError):
        mq.as_moe_config(BitsAndBytesConfig(load_in_8bit=True))


# ---------------------------------------------------------------------------
# The pipeline picks it for MoE checkpoints only
# ---------------------------------------------------------------------------


def _capture_from_pretrained(monkeypatch):
    from transformers import AutoModelForCausalLM

    seen = {}

    def _fake(model_id, **kw):
        seen["model_id"] = model_id
        seen.update(kw)
        return "model"

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", staticmethod(_fake))
    monkeypatch.setattr(pipeline, "detect_quantization", lambda _m: {"method": "", "config": {}})
    return seen


def test_pipeline_attaches_the_moe_config_for_a_moe(monkeypatch) -> None:
    seen = _capture_from_pretrained(monkeypatch)
    monkeypatch.setattr(pipeline, "describe_expert_topology", lambda _m: {
        "is_moe": True, "num_experts": 128, "experts_per_tok": 8,
        "model_type": "qwen3_moe", "known": True})
    assert pipeline.load_training_model("some/moe") == "model"
    assert isinstance(seen["quantization_config"], mq.MoEBitsAndBytesConfig)
    assert seen["device_map"] == {"": 0}, "still streams shard by shard to the device"


def test_pipeline_keeps_plain_bnb_for_a_dense_model(monkeypatch) -> None:
    from transformers import BitsAndBytesConfig

    seen = _capture_from_pretrained(monkeypatch)
    monkeypatch.setattr(pipeline, "describe_expert_topology", lambda _m: {
        "is_moe": False, "num_experts": 0, "experts_per_tok": 0,
        "model_type": "qwen2", "known": True})
    pipeline.load_training_model("some/dense")
    cfg = seen["quantization_config"]
    assert isinstance(cfg, BitsAndBytesConfig)
    assert not isinstance(cfg, mq.MoEBitsAndBytesConfig)


def test_pipeline_loads_inside_the_page_cache_valve(monkeypatch) -> None:
    """The unmapping only works from inside the loading process, so the
    loader call must sit inside the valve -- pinned against a refactor
    that moves ``from_pretrained`` out of the ``with``."""
    from reactor_core.training import memory_guard as guard

    events: list = []

    class _Valve:
        def __init__(self, **kw):
            events.append(("build", kw.get("label")))

        def __enter__(self):
            events.append(("enter", None))
            return self

        def __exit__(self, *_exc):
            events.append(("exit", None))

        def report(self):
            return {"enabled": True}

    monkeypatch.setattr(guard, "PageCacheValve", _Valve)
    seen = _capture_from_pretrained(monkeypatch)
    from transformers import AutoModelForCausalLM

    def _fake(model_id, **kw):
        events.append(("load", model_id))
        return "model"

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", staticmethod(_fake))
    monkeypatch.setattr(pipeline, "describe_expert_topology", lambda _m: {
        "is_moe": False, "num_experts": 0, "experts_per_tok": 0,
        "model_type": "qwen2", "known": True})
    assert pipeline.load_training_model("some/dense") == "model"
    assert events == [("build", "from_pretrained"), ("enter", None),
                      ("load", "some/dense"), ("exit", None)]
    assert seen == {} or "model_id" not in seen


def test_pipeline_does_not_quantize_when_qlora_is_off(monkeypatch) -> None:
    seen = _capture_from_pretrained(monkeypatch)
    monkeypatch.setattr(pipeline, "describe_expert_topology", lambda _m: {
        "is_moe": True, "num_experts": 8, "experts_per_tok": 2,
        "model_type": "qwen3_moe", "known": True})
    pipeline.load_training_model("some/moe", use_qlora=False)
    assert "quantization_config" not in seen


# ---------------------------------------------------------------------------
# The stack, on the card
# ---------------------------------------------------------------------------


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


@cuda
def test_stack_indexes_to_the_expert_at_nf4_precision() -> None:
    torch.manual_seed(0)
    fused = torch.randn(5, 64, 128, dtype=torch.bfloat16, device="cuda")
    stack = mq.QuantizedExpertStack.from_fused(fused)
    assert len(stack) == 5
    assert stack.shape == fused.shape and stack.dtype == torch.bfloat16
    for i in range(5):
        w = stack[i]
        assert w.shape == (64, 128) and w.dtype == torch.bfloat16
        assert _cos(w, fused[i]) > 0.99, "NF4 keeps the direction of the matrix"
        assert not w.requires_grad
    # The eager loop indexes with a 0-d tensor.
    assert torch.equal(stack[torch.tensor(2, device="cuda")], stack[2])
    packed = sum(p.numel() * p.element_size() for p in stack.parameters())
    assert packed < 0.35 * fused.numel() * fused.element_size(), "4-bit, not bf16"


@cuda
def test_stack_refuses_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        mq.QuantizedExpertStack.from_fused(torch.randn(64, 128, device="cuda"))


# ---------------------------------------------------------------------------
# End to end through from_pretrained
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

    torch.manual_seed(0)
    cfg = _tiny_moe_config()
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    where = tmp_path_factory.mktemp("tiny_moe")
    model.save_pretrained(where)
    cfg.save_pretrained(where)
    return where


@cuda
def test_end_to_end_load_quantizes_experts_and_still_trains(tiny_checkpoint) -> None:
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(
        tiny_checkpoint, dtype=torch.bfloat16, device_map={"": 0}).eval()
    model = AutoModelForCausalLM.from_pretrained(
        tiny_checkpoint, quantization_config=mq.as_moe_config(pipeline.build_qlora_config()),
        device_map={"": 0}, dtype=torch.bfloat16).eval()

    # 1. every fused expert tensor is a stack; attention is Linear4bit
    experts = model.model.layers[0].mlp.experts
    assert isinstance(experts.gate_up_proj, mq.QuantizedExpertStack)
    assert isinstance(experts.down_proj, mq.QuantizedExpertStack)
    assert len(experts.gate_up_proj) == 8
    assert isinstance(model.model.layers[0].self_attn.q_proj, bnb.nn.Linear4bit)
    leftovers = [n for n, p in model.named_parameters() if p.ndim == 3]
    assert leftovers == [], f"bf16 fused experts survived: {leftovers}"
    assert model.config._experts_implementation == "eager"
    assert getattr(model, "is_loaded_in_4bit", False) is True

    # 2. the forward agrees with the bf16 model
    torch.manual_seed(1)
    x = torch.randint(0, 512, (2, 24), device="cuda")
    with torch.no_grad():
        want = ref(input_ids=x).logits
        got = model(input_ids=x).logits
    assert torch.isfinite(got).all()
    assert _cos(got, want) > 0.9, "NF4 attention + NF4 experts vs bf16"

    # 3. gradients reach a trainable parameter THROUGH the quantized experts
    for p in model.parameters():
        p.requires_grad_(False)
    embed = model.get_input_embeddings().weight
    embed.requires_grad_(True)
    out = model(input_ids=x, labels=x)
    out.loss.backward()
    assert embed.grad is not None and torch.isfinite(embed.grad).all()
    assert float(embed.grad.abs().sum()) > 0.0

    # 4. generation runs on the eager dispatch
    with torch.no_grad():
        gen = model.generate(x[:, :4], max_new_tokens=4, do_sample=False)
    assert gen.shape == (2, 8)

    # 5. it is smaller: the experts' bytes are ~4-bit
    expert_bytes = sum(
        p.numel() * p.element_size()
        for n, p in model.named_parameters() if ".experts." in n)
    ref_expert_bytes = sum(
        p.numel() * p.element_size()
        for n, p in ref.named_parameters() if ".experts." in n)
    assert expert_bytes < 0.35 * ref_expert_bytes


@cuda
def test_end_to_end_refuses_a_half_quantized_model(tiny_checkpoint, monkeypatch) -> None:
    """If a discovered expert tensor is not a stack after loading, loading
    fails -- never a model that quietly trains bf16 experts."""
    from transformers import AutoModelForCausalLM

    real = mq.MoEBnb4BitQuantizer.install_expert_stack

    def _skip_one(self, model, name, fused, missing):
        if name.endswith("layers.1.mlp.experts.down_proj"):
            if missing is not None:
                missing.discard(name)
            return
        real(self, model, name, fused, missing)

    monkeypatch.setattr(mq.MoEBnb4BitQuantizer, "install_expert_stack", _skip_one)
    with pytest.raises(RuntimeError, match="still bf16"):
        AutoModelForCausalLM.from_pretrained(
            tiny_checkpoint, quantization_config=mq.as_moe_config(pipeline.build_qlora_config()),
            device_map={"": 0}, dtype=torch.bfloat16)
