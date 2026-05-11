"""Pluggable rotation strategies for fused-turboquant.

To add a new rotation kind (e.g. Rotorquant):

    from fused_turboquant.vllm_plugin.rotation import (
        RotationStrategy, register_rotation,
    )

    class MyRotation(RotationStrategy):
        name = "my_kind"
        def setup_layer(self, layer, head_size, centroids, device): ...
        def rotate_for_store(self, x_normalized, layer): ...
        def rotate_for_decode(self, q, layer): ...

    register_rotation(MyRotation.name, MyRotation)

Then select at runtime via `TURBOQUANT_KIND=my_kind`.
"""

from __future__ import annotations

from .base import (
    RotationStrategy,
    available_rotations,
    get_rotation,
    register_rotation,
)
from .matrix import MatrixRotationStrategy

# Import side-effect registers the built-in strategies.
from . import hadamard, planar  # noqa: F401

__all__ = [
    "RotationStrategy",
    "MatrixRotationStrategy",
    "available_rotations",
    "get_rotation",
    "register_rotation",
]
