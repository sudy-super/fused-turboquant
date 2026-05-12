"""Planar (2D Givens) rotation — D/2 independent pair rotations.

Each consecutive coordinate pair (v_{2p}, v_{2p+1}) is rotated by an
independent random angle θ_p. The full rotation is an orthogonal
block-diagonal matrix with 2×2 blocks `[[c, -s], [s, c]]^T`. We
materialize it as a (D, D) matrix so the rotation step reuses cuBLAS
(matmul) instead of D/2 element-wise pair launches.

Compared to RHT, Planar's mixing is purely local — distortion in any
single coordinate doesn't get smeared across the whole vector. That's
faster (less FLOPs) but quality-wise more fragile to layer-level
quantization error compounding. Empirically Planar with boundary
protection lands around 90% on GSM-8K vs RHT's 100%, and without
boundary protection it collapses to 0%. Use RHT for production; this
class is here for research and to demonstrate the strategy interface.

The angles are generated with seed=42 so all TP / DP ranks build the
same rotation matrix for a given (head_size).
"""

from __future__ import annotations

import torch

from fused_turboquant.core.planar import generate_planar_rot2

from .base import register_rotation
from .matrix import BlockDiagonalRotationStrategy

PLANAR_SEED = 42


def _build_planar_matrix(rot2: torch.Tensor) -> torch.Tensor:
    """Materialize a (D, D) block-diagonal matrix `M` such that
    `x @ M = planar_rotate(x, rot2)`.

    For each pair p with `(c, s) = rot2[p]`, the 2×2 block at rows/cols
    `[2p, 2p+1]` is `[[c, s], [-s, c]]` — the transpose of the
    column-vector rotation `[[c, -s], [s, c]]`. The transposed form is
    what lets `x @ M` (row-vector matmul) produce
    `[c*x0 - s*x1, s*x0 + c*x1]`.
    """
    n_pairs = rot2.shape[0]
    D = n_pairs * 2
    M = torch.zeros(D, D, device=rot2.device, dtype=rot2.dtype)
    c = rot2[:, 0]
    s = rot2[:, 1]
    idx = torch.arange(n_pairs, device=rot2.device)
    M[2 * idx, 2 * idx] = c
    M[2 * idx, 2 * idx + 1] = s
    M[2 * idx + 1, 2 * idx] = -s
    M[2 * idx + 1, 2 * idx + 1] = c
    return M


class PlanarStrategy(BlockDiagonalRotationStrategy):
    name = "planar"
    block_size = 2

    def build_matrix(self, head_size, device):
        rot2 = generate_planar_rot2(head_size, seed=PLANAR_SEED, device=device).to(
            dtype=torch.float32
        )
        return _build_planar_matrix(rot2)


register_rotation(PlanarStrategy.name, PlanarStrategy)
