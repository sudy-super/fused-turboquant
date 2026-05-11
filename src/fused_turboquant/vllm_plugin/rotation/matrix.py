"""Convenience base for rotations that are a (D, D) matmul.

For any orthogonal R representable as a (D, D) matrix, the forward
rotation is `x @ M` where `M` is built once per layer at setup time
and the SAME matrix is reused for both Q and K (since R is orthogonal,
`q · K = (R·q) · (R·K)` regardless of whether R == R^T).

`HadamardStrategy` and `PlanarStrategy` only have to override
`build_matrix(head_size, device)`.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from .base import RotationStrategy


class MatrixRotationStrategy(RotationStrategy):
    @abstractmethod
    def build_matrix(self, head_size: int, device: torch.device | str) -> torch.Tensor:
        """Return a (D, D) orthogonal matrix `M` (float32 on `device`)
        such that `x @ M` is the forward rotation."""

    def setup_layer(self, layer, head_size, centroids, device):
        if getattr(layer, "_fused_tq_cached", False):
            return
        layer._fused_tq_rotation = self.build_matrix(head_size, device)
        c = centroids.to(device=device, dtype=torch.float32)
        c_sorted, _ = c.sort()
        layer._fused_tq_centroids = c_sorted
        layer._fused_tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
        layer._fused_tq_cached = True

    def rotate_for_store(self, x_normalized, layer):
        return (x_normalized @ layer._fused_tq_rotation).contiguous()

    def rotate_for_decode(self, q, layer):
        return (q @ layer._fused_tq_rotation).contiguous()
