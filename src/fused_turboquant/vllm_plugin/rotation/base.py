"""Abstract base for pluggable rotation strategies.

A `RotationStrategy` owns everything that's specific to a particular
orthogonal rotation kind (RHT, Planar, Rotorquant, ...): how to build
the rotation state on a layer, and how to apply the forward rotation
to K (at store time) and Q (at decode time).

The Triton store/decode kernels are rotation-agnostic — they just
bucketize against Lloyd-Max midpoints and compute a dot product
between the (pre-rotated) Q and the centroid-indexed K. So adding a
new rotation kind is a self-contained change: subclass
`RotationStrategy`, implement `setup_layer` / `rotate_for_store` /
`rotate_for_decode`, register the class via `register_rotation`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch


class RotationStrategy(ABC):
    """A pluggable orthogonal rotation for fused-turboquant.

    The contract:
      - `setup_layer` is called once per attention layer at first forward
        time, with the layer's centroid table and the target device. It
        should cache the rotation state on the layer (under attribute
        names of the implementer's choosing, but `_fused_tq_*` is the
        convention).
      - `rotate_for_store` is applied to unit-norm K vectors before the
        MSE bucketize step in the store kernel.
      - `rotate_for_decode` is applied to Q before computing
        `score = q_rot · c_vals` in the decode kernel.

    For most rotations both methods apply the same orthogonal matrix
    (since `q · K = (R·q) · (R·K)` for orthogonal R). Strategies that
    need different rotations for the two phases can override them
    independently — e.g. an asymmetric residual quantizer.
    """

    name: ClassVar[str]

    @abstractmethod
    def setup_layer(
        self,
        layer,
        head_size: int,
        centroids: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        """Build and cache rotation state on `layer`. Idempotent."""

    @abstractmethod
    def rotate_for_store(self, x_normalized: torch.Tensor, layer) -> torch.Tensor:
        """Rotate unit-norm K. Input/output shape `(..., D)`."""

    @abstractmethod
    def rotate_for_decode(self, q: torch.Tensor, layer) -> torch.Tensor:
        """Rotate Q so the score `q_rot · centroid` recovers `q · K_original`."""

    def get_centroids(self, layer) -> torch.Tensor:
        """Sorted Lloyd-Max levels. Stored by `setup_layer`."""
        return layer._fused_tq_centroids

    def get_midpoints(self, layer) -> torch.Tensor:
        """Lloyd-Max midpoints (n_centroids-1,) for the store kernel's
        binary-search bucketize."""
        return layer._fused_tq_midpoints


_STRATEGIES: dict[str, type[RotationStrategy]] = {}


def register_rotation(name: str, cls: type[RotationStrategy]) -> None:
    """Register a strategy under `TURBOQUANT_KIND=<name>`. Idempotent
    re-registration overwrites (useful for testing / shadowing)."""
    _STRATEGIES[name] = cls


def get_rotation(name: str) -> RotationStrategy:
    """Instantiate the strategy registered under `name`. Raises with a
    helpful list if the name isn't registered."""
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown rotation kind: {name!r}. "
            f"Registered kinds: {sorted(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[name]()


def available_rotations() -> list[str]:
    return sorted(_STRATEGIES.keys())
