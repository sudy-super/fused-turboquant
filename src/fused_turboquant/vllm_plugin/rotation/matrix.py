"""Convenience base for rotations that are a (D, D) matmul.

For any orthogonal R representable as a (D, D) matrix, the forward
rotation is `x @ M` where `M` is built once per layer at setup time
and the SAME matrix is reused for both Q and K (since R is orthogonal,
`q · K = (R·q) · (R·K)` regardless of whether R == R^T).

`HadamardStrategy` and `PlanarStrategy` only have to override
`build_matrix(head_size, device)`.

For strategies whose `M` is strictly block-diagonal with B×B blocks
(Planar B=2, Rotor B=3, Iso B=4), use `BlockDiagonalRotationStrategy`
instead — it computes a compact `[B, D]` coefficient tensor and
dispatches to a specialized in-kernel rotation that does only O(B·D)
work per token (vs the generic kernel's O(D²)).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

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

    def launch_store(self, key, value, kv_cache, slot_mapping, layer, tq_config):
        """Default: in-kernel matrix rotation — load `M` tile into SRAM,
        matmul there. No external cuBLAS GEMM, but O(D²) work per token.
        Block-diagonal subclasses override this with a tighter kernel."""
        from fused_turboquant.vllm_plugin.triton_inkernel_store import _launch_matrix

        _launch_matrix(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            rotation_matrix=layer._fused_tq_rotation,
            midpoints=self.get_midpoints(layer),
            tq_config=tq_config,
            head_size=key.shape[-1],
        )


class BlockDiagonalRotationStrategy(MatrixRotationStrategy):
    """Subclass-of-base for rotations whose `M` is strictly block-diagonal
    with B×B blocks.

    For Planar (B=2), Rotor (B=3), Iso-Fast (B=4) and Iso-Full (B=4),
    `M[i, j] = 0` unless `i // B == j // B`. The generic O(D²) matmul
    wastes work on the zeros; the specialized kernel does only O(B·D)
    multiply-accumulates per token by gathering only the B nonzero input
    positions for each output dim.

    Mapping (proof: trivially from `M` being block-diagonal):
        y[d] = (x @ M)[d]
             = Σ_i x[i] · M[i, d]
             = Σ_{i in same block as d} x[i] · M[i, d]
             = Σ_{k=0..B-1} x[group_start + k] · M[group_start + k, d]
        where group_start = (d // B) · B.

    So defining `coeffs[k, d] = M[group_start + k, d]` recovers `x @ M`
    via B accumulations per output dim. For the last partial block when
    `D % B != 0` (only happens for Rotor on most head sizes), an input
    position `group_start + k` may be ≥ D — the kernel masks those loads
    to zero and the corresponding coeff entry is also zero.

    Subclasses only need to set `block_size` and `build_matrix`.
    """

    block_size: ClassVar[int]

    @staticmethod
    def compute_coeffs(M: torch.Tensor, B: int) -> torch.Tensor:
        """Extract `[B, D]` coefficients from a (D, D) block-diagonal `M`.

        `coeffs[k, d] = M[(d // B) * B + k, d]` for in-range partners,
        else 0. The result is contiguous `(B, D)` float32 — `coeffs[k, :]`
        is a contiguous D-vector so the kernel's
        `Coeffs_ptr + k * D + d_offs` load coalesces perfectly.
        """
        D = M.shape[-1]
        device = M.device
        d_idx = torch.arange(D, device=device)
        k_idx = torch.arange(B, device=device)
        group_start = (d_idx // B) * B
        input_pos = group_start[None, :] + k_idx[:, None]  # [B, D]
        valid = input_pos < D
        safe_pos = torch.where(valid, input_pos, torch.zeros_like(input_pos))
        d_grid = d_idx.unsqueeze(0).expand(B, D)
        coeffs = M[safe_pos, d_grid] * valid.to(M.dtype)
        return coeffs.contiguous()

    def setup_layer(self, layer, head_size, centroids, device):
        super().setup_layer(layer, head_size, centroids, device)
        if not hasattr(layer, "_fused_tq_rotation_coeffs"):
            layer._fused_tq_rotation_coeffs = self.compute_coeffs(
                layer._fused_tq_rotation, self.block_size
            )

    def launch_store(self, key, value, kv_cache, slot_mapping, layer, tq_config):
        """Block-diagonal in-kernel rotation. B gathers + B muladds per
        output dim, vs the dense kernel's D matmul."""
        from fused_turboquant.vllm_plugin.triton_inkernel_store import (
            _launch_block_diag,
        )

        _launch_block_diag(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            coeffs=layer._fused_tq_rotation_coeffs,
            midpoints=self.get_midpoints(layer),
            tq_config=tq_config,
            head_size=key.shape[-1],
            block_size=self.block_size,
        )
