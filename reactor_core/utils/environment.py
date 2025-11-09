"""
Environment detection for M1 local vs GCP remote
"""
import os
import platform
import psutil
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class EnvironmentType(Enum):
    """Environment types"""
    M1_LOCAL = "m1_local"
    GCP_VM = "gcp_vm"
    UNKNOWN = "unknown"


@dataclass
class EnvironmentInfo:
    """Environment information"""
    env_type: EnvironmentType
    cpu_arch: str
    total_ram_gb: float
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    is_spot_vm: bool = False
    is_m1_mac: bool = False


def detect_environment() -> EnvironmentInfo:
    """
    Detect the current execution environment

    Returns:
        EnvironmentInfo with environment details
    """
    # Get system info
    cpu_arch = platform.machine()
    total_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB

    # Check for M1 Mac
    is_m1_mac = cpu_arch == "arm64" and platform.system() == "Darwin"

    # Check for GPU
    gpu_available = False
    gpu_memory_gb = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif torch.backends.mps.is_available():
            # M1 Metal Performance Shaders
            gpu_available = True
    except ImportError:
        pass

    # Check for GCP metadata
    is_gcp_vm = _is_gcp_vm()
    is_spot_vm = _is_spot_vm() if is_gcp_vm else False

    # Determine environment type
    if is_m1_mac:
        env_type = EnvironmentType.M1_LOCAL
    elif is_gcp_vm:
        env_type = EnvironmentType.GCP_VM
    else:
        env_type = EnvironmentType.UNKNOWN

    return EnvironmentInfo(
        env_type=env_type,
        cpu_arch=cpu_arch,
        total_ram_gb=round(total_ram, 2),
        gpu_available=gpu_available,
        gpu_memory_gb=round(gpu_memory_gb, 2) if gpu_memory_gb else None,
        is_spot_vm=is_spot_vm,
        is_m1_mac=is_m1_mac,
    )


def _is_gcp_vm() -> bool:
    """Check if running on GCP VM"""
    try:
        # GCP metadata server
        import requests
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/id",
            headers={"Metadata-Flavor": "Google"},
            timeout=1
        )
        return response.status_code == 200
    except:
        return False


def _is_spot_vm() -> bool:
    """Check if running on GCP Spot (preemptible) VM"""
    try:
        import requests
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/preempted",
            headers={"Metadata-Flavor": "Google"},
            timeout=1
        )
        return response.status_code == 200
    except:
        return False


def get_recommended_config(env_info: EnvironmentInfo) -> dict:
    """
    Get recommended training configuration based on environment

    Args:
        env_info: Environment information

    Returns:
        Dictionary with recommended settings
    """
    if env_info.env_type == EnvironmentType.M1_LOCAL:
        return {
            "mode": "inference_only",
            "device": "mps" if env_info.gpu_available else "cpu",
            "batch_size": 1,
            "use_quantization": True,
            "quantization_bits": 8,
            "max_model_size": "7B",
            "enable_training": False,
            "message": "M1 Mac detected: Lightweight inference mode enabled"
        }

    elif env_info.env_type == EnvironmentType.GCP_VM:
        # Determine optimal settings based on RAM/GPU
        config = {
            "mode": "full_training",
            "device": "cuda" if env_info.gpu_available else "cpu",
            "enable_training": True,
        }

        if env_info.total_ram_gb >= 32:
            config.update({
                "batch_size": 4,
                "use_gradient_checkpointing": True,
                "use_lora": True,
                "lora_rank": 16,
                "max_model_size": "13B",
                "message": "GCP VM detected: Full training enabled"
            })
        else:
            config.update({
                "batch_size": 2,
                "use_gradient_checkpointing": True,
                "use_lora": True,
                "lora_rank": 8,
                "max_model_size": "7B",
                "message": "GCP VM detected: Memory-efficient training enabled"
            })

        if env_info.is_spot_vm:
            config.update({
                "checkpoint_interval": 500,
                "enable_auto_resume": True,
                "message": config["message"] + " (Spot VM: Auto-resume enabled)"
            })

        return config

    else:
        return {
            "mode": "unknown",
            "device": "cpu",
            "message": "Unknown environment: Using conservative settings"
        }


# Auto-detect on import
_ENV_INFO = detect_environment()


def print_environment_info():
    """Print detected environment information"""
    info = _ENV_INFO

    print("=" * 60)
    print("Reactor Core - Environment Detection")
    print("=" * 60)
    print(f"Environment Type: {info.env_type.value}")
    print(f"CPU Architecture: {info.cpu_arch}")
    print(f"Total RAM: {info.total_ram_gb} GB")
    print(f"GPU Available: {info.gpu_available}")
    if info.gpu_memory_gb:
        print(f"GPU Memory: {info.gpu_memory_gb} GB")
    print(f"M1 Mac: {info.is_m1_mac}")
    print(f"GCP Spot VM: {info.is_spot_vm}")
    print("=" * 60)

    # Print recommended config
    config = get_recommended_config(info)
    print(f"\nRecommended Config: {config['message']}")
    print(f"Mode: {config['mode']}")
    print(f"Device: {config['device']}")
    print("=" * 60)


if __name__ == "__main__":
    print_environment_info()
