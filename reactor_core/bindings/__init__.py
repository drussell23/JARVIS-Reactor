"""
MLForge C++ Bindings Package - v91.0

Provides Python bindings for MLForge C++ core library with graceful fallback
to pure-Python implementations when C++ bindings are not available.

Usage:
    from reactor_core.bindings import mlforge_bridge

    # Check if C++ bindings are available
    if mlforge_bridge.has_cpp_backend():
        # Use optimized C++ operations
        matrix_ops = mlforge_bridge.get_matrix_operations()
    else:
        # Fallback to Python/NumPy operations
        matrix_ops = mlforge_bridge.get_matrix_operations()  # Same API
"""

from .mlforge_bridge import (
    MLForgeBridge,
    get_bridge,
    has_cpp_backend,
    get_matrix_operations,
    get_neural_net_ops,
    get_serializer,
)

__all__ = [
    "MLForgeBridge",
    "get_bridge",
    "has_cpp_backend",
    "get_matrix_operations",
    "get_neural_net_ops",
    "get_serializer",
]
