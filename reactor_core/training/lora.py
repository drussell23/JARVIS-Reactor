"""LoRA (Low-Rank Adaptation) utilities"""
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: list = None


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Apply LoRA to model

    Args:
        model: Base model
        config: LoRA configuration

    Returns:
        Model with LoRA applied
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=config.target_modules or ["q_proj", "v_proj"],
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

    except ImportError:
        raise ImportError("Please install peft: pip install peft")
