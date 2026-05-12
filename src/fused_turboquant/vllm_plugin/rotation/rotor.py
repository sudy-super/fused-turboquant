"""RotorQuant — Cl(3,0) rotor sandwich, materialized as 3×3 block-diagonal
rotation matrix.

The original RotorQuant paper (`scrya.com/rotorquant.pdf`) decorrelates
a 3D vector by embedding it as a Cl(3,0) multivector (a 1-vector with
components on `e1, e2, e3`), applying a rotor sandwich `R x R̃` (with
rotor `R = s + p12·e12 + p13·e13 + p23·e23`), then extracting back to
3D. Quantization happens in the rotated space.

The rotor-sandwich output has non-zero grade-1 (vector) **and** grade-3
(pseudoscalar) components, but the upstream
`rotorquant/turboquant/triton_kernels.py:_rotor_full_fused_kernel`
**drops the grade-3 component on extract** (comment: "Trivector:
non-zero but NEVER READ by extract; Dropping trivector saves 25% of
indices with zero MSE impact"). The kept grade-1 output is, by direct
algebraic expansion, exactly the result of multiplying the input
3-vector by a 3×3 orthogonal matrix derived from `(s, p12, p13, p23)`.

So "rotor sandwich + extract" reduces to `x @ R₃ₓ₃`, and the full
D-dim rotation is the block-diagonal `R₃ₓ₃ ⊕ R₃ₓ₃ ⊕ …`. We
precompute that block-diagonal once per (head_size, device) and let
`MatrixRotationStrategy` reuse the same matmul-based code path as
Hadamard and Planar — no new kernels needed.

3×3 block formula. Derived from `r_i = a0·b_i + a_i·b0 + …` of the
geometric product (`turboquant/clifford.py`) substituted into
`R · v · R̃` where R = (s; 0,0,0; p12,p13,p23; 0) and v = (0; v1,v2,v3; …).
Verified against `rotor_sandwich + extract` on all three standard
basis vectors, errors at fp32 noise level for D divisible by 3.

    [s² - p12² - p13² + p23²,    2·s·p12 - 2·p13·p23,         2·s·p13 + 2·p12·p23]
    [-2·s·p12 - 2·p13·p23,       s² - p12² + p13² - p23²,     2·s·p23 - 2·p12·p13]
    [-2·s·p13 + 2·p12·p23,       -2·s·p23 - 2·p12·p13,        s² + p12² - p13² - p23²]

This matrix is in column-vector form (it acts as `M · v_col`), so for
PyTorch's row-vector matmul `x @ M_row` we use its transpose.

Compared to Planar (2×2 blocks) and Hadamard (D×D dense), RotorQuant
sits in between: 4 params per 3 dims (≈ 1.33·D) vs Planar's 2 per 2 (=
D) and RHT's D². Empirically (per the rotorquant repo's PPL table) the
extra rotation richness gives accuracy between Planar and TurboQuant.

`head_size` not divisible by 3: pad the rotation matrix up to the next
multiple of 3, then truncate to (head_size, head_size). The truncation
slightly breaks orthogonality on the last `head_size mod 3` coords,
but the deviation is small and matched between Q and K so the inner
product is approximately preserved.
"""

from __future__ import annotations

import math

import torch

from .base import register_rotation
from .matrix import BlockDiagonalRotationStrategy

ROTOR_SEED = 42


def _build_rotor_matrix(head_size: int, device) -> torch.Tensor:
    """Build the (head_size, head_size) block-diagonal rotation matrix
    `M` such that `x @ M` is the per-group rotor sandwich extracted to
    grade-1. Deterministic for a given (head_size, ROTOR_SEED).
    """
    D = head_size
    D_padded = ((D + 2) // 3) * 3  # ceil(D / 3) * 3
    n_groups = D_padded // 3

    # Generate random rotors on CPU for determinism (we don't want the
    # angle/bivector to depend on the GPU's RNG state).
    gen = torch.Generator(device="cpu").manual_seed(ROTOR_SEED)
    bv = torch.randn(n_groups, 3, generator=gen, dtype=torch.float32)
    bv = bv / bv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    angles = torch.rand(n_groups, generator=gen, dtype=torch.float32) * (2.0 * math.pi)

    half_angles = angles / 2
    s = torch.cos(half_angles)
    sin_ha = torch.sin(half_angles)
    p12 = sin_ha * bv[:, 0]
    p13 = sin_ha * bv[:, 1]
    p23 = sin_ha * bv[:, 2]

    # 3×3 block elements in column-vector form (M · v_col).
    s2, p12_2, p13_2, p23_2 = s * s, p12 * p12, p13 * p13, p23 * p23
    m00 = s2 - p12_2 - p13_2 + p23_2
    m01 = 2 * s * p12 - 2 * p13 * p23
    m02 = 2 * s * p13 + 2 * p12 * p23
    m10 = -2 * s * p12 - 2 * p13 * p23
    m11 = s2 - p12_2 + p13_2 - p23_2
    m12 = 2 * s * p23 - 2 * p12 * p13
    m20 = -2 * s * p13 + 2 * p12 * p23
    m21 = -2 * s * p23 - 2 * p12 * p13
    m22 = s2 + p12_2 - p13_2 - p23_2

    # Place column-vector blocks at (3g, 3g) then transpose at the end
    # for row-vector matmul `x @ M_row`.
    blocks_col = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )  # (n_groups, 3, 3) — column-vector form

    M = torch.zeros(D_padded, D_padded, dtype=torch.float32)
    g_idx = torch.arange(n_groups).repeat_interleave(9)
    r0_idx = torch.arange(3).repeat_interleave(3).repeat(n_groups)
    r1_idx = torch.arange(3).repeat(3 * n_groups)
    M[3 * g_idx + r0_idx, 3 * g_idx + r1_idx] = blocks_col.reshape(-1)

    # Transpose so that `x @ M` (row-vector matmul) implements rotor_sandwich.
    M = M.T.contiguous()

    return M[:D, :D].to(torch.device(device)).contiguous()


class RotorStrategy(BlockDiagonalRotationStrategy):
    """RotorQuant via 3×3 block-diagonal rotation. The block-diagonal
    in-kernel path exploits the 3×3 structure: only 3 gathers + 3 muladds
    per output dim, vs the dense kernel's D matmul."""

    name = "rotor"
    block_size = 3

    def build_matrix(self, head_size, device):
        return _build_rotor_matrix(head_size, device)


register_rotation(RotorStrategy.name, RotorStrategy)
