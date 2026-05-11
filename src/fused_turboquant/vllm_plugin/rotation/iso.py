"""IsoQuant — quaternion 4D block rotations.

The `rotorquant` repo demonstrates two flavors:

  - IsoQuant-Fast: T(v) = q_L · v        (3 DOF per 4D block, faster, isoclinic SO(3) subgroup)
  - IsoQuant-Full: T(v) = q_L · v · q̄_R   (6 DOF per 4D block, full SO(4))

Both are orthogonal transformations on each consecutive group of 4
input dims. Per the upstream PPL numbers (Llama 3 8B, iso3/iso3 with
deferred K-cache boundary protection):

    iso3/iso3:  PPL 6.91  (vs RHT turbo3/turbo3: 7.07)
    planar3:    PPL 7.05

IsoQuant beats both Planar and TurboQuant on quality, at the same
compression ratio, with 4-wide SIMD-friendly blocks.

Materializing as a 4×4 matrix.  The Hamilton product `q · v` viewed as
left multiplication by q corresponds to the 4×4 matrix

    L(q) = [w  -x  -y  -z]
           [x   w  -z   y]
           [y   z   w  -x]
           [z  -y   x   w]

(rows give `(q·v)_w, (q·v)_x, (q·v)_y, (q·v)_z` in terms of `v_w..v_z`).

The right multiplication `v · r` corresponds to

    R(r) = [w  -x  -y  -z]
           [x   w   z  -y]
           [y  -z   w   x]
           [z   y  -x   w]

For "Fast", the per-block rotation matrix is just `L(q_L)`. For "Full",
it is `L(q_L) · R(q̄_R)` (where q̄_R = conj(q_R) means `(w, -x, -y, -z)`).
Both are orthogonal because q_L, q_R are unit quaternions.

Stacking one such 4×4 block per group of 4 dims gives a (D, D)
block-diagonal rotation. Speed envelope is identical to Planar and
Rotor — single cuBLAS GEMM for the rotation step.

`head_size` not divisible by 4: pad up and truncate. For 64 / 128 /
256 / 512 (Qwen, Gemma 4), this is a no-op; we never lose
orthogonality on real models.

The seed for q_L is 42; for Full mode, q_R uses seed 10042 — same
constants as the upstream `IsoQuantMSE`, so the rotation tables agree
bit-for-bit across the two implementations.
"""

from __future__ import annotations

import math

import torch

from .base import register_rotation
from .matrix import MatrixRotationStrategy

ISO_SEED_QL = 42
ISO_SEED_QR_OFFSET = 10000  # so q_R uses seed=42+10000 (matches rotorquant)


def _random_unit_quaternion(n_groups: int, seed: int) -> torch.Tensor:
    """Sample `n_groups` random unit quaternions deterministically.
    Returns `(n_groups, 4)` float32."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n_groups, 4, generator=gen, dtype=torch.float32)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _left_multiply_matrix(q: torch.Tensor) -> torch.Tensor:
    """`L(q)` such that `L(q) · v = q · v` (column-vector form).
    `q`: `(..., 4)`. Returns `(..., 4, 4)`."""
    w, x, y, z = q.unbind(-1)
    z_t = torch.stack
    rows = [
        torch.stack([w, -x, -y, -z], dim=-1),
        torch.stack([x, w, -z, y], dim=-1),
        torch.stack([y, z, w, -x], dim=-1),
        torch.stack([z, -y, x, w], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def _right_multiply_matrix(r: torch.Tensor) -> torch.Tensor:
    """`R(r)` such that `R(r) · v = v · r` (column-vector form, so
    v is the left operand of the Hamilton product). `r`: `(..., 4)`.
    Returns `(..., 4, 4)`."""
    w, x, y, z = r.unbind(-1)
    rows = [
        torch.stack([w, -x, -y, -z], dim=-1),
        torch.stack([x, w, z, -y], dim=-1),
        torch.stack([y, -z, w, x], dim=-1),
        torch.stack([z, y, -x, w], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def _build_iso_matrix(head_size: int, device, full: bool) -> torch.Tensor:
    """Build the (head_size, head_size) row-vector rotation matrix for
    IsoQuant. `full=True` uses both q_L and q_R; `False` is q_L only."""
    D = head_size
    D_padded = ((D + 3) // 4) * 4
    n_groups = D_padded // 4

    q_L = _random_unit_quaternion(n_groups, ISO_SEED_QL)
    L = _left_multiply_matrix(q_L)  # (n_groups, 4, 4)

    if full:
        q_R = _random_unit_quaternion(n_groups, ISO_SEED_QL + ISO_SEED_QR_OFFSET)
        # T(v) = q_L · v · conj(q_R). conj(q) = (w, -x, -y, -z).
        signs = torch.tensor([1.0, -1.0, -1.0, -1.0])
        q_R_conj = q_R * signs
        R = _right_multiply_matrix(q_R_conj)  # (n_groups, 4, 4)
        # Per-group rotation in column-vector form: L @ R.
        block_col = torch.matmul(L, R)
    else:
        block_col = L  # Fast: just L(q_L)

    # Place each 4×4 block at (4g, 4g). The blocks_col are in column-
    # vector form (`M_col · v_col`); we transpose at the end so that
    # `v_row @ M_row = (M_col · v_col)^T`.
    M = torch.zeros(D_padded, D_padded, dtype=torch.float32)
    g_idx = torch.arange(n_groups).repeat_interleave(16)
    r0_idx = torch.arange(4).repeat_interleave(4).repeat(n_groups)
    r1_idx = torch.arange(4).repeat(4 * n_groups)
    M[4 * g_idx + r0_idx, 4 * g_idx + r1_idx] = block_col.reshape(-1)

    M = M.T.contiguous()
    return M[:D, :D].to(torch.device(device)).contiguous()


class IsoFastStrategy(MatrixRotationStrategy):
    """IsoQuant-Fast: 4D block rotation via single quaternion left-multiply.
    3 DOF per block, ~half the FLOPs of IsoQuant-Full."""

    name = "iso_fast"

    def build_matrix(self, head_size, device):
        return _build_iso_matrix(head_size, device, full=False)


class IsoFullStrategy(MatrixRotationStrategy):
    """IsoQuant-Full: 4D block rotation via quaternion sandwich `q_L · v · q̄_R`.
    6 DOF per block — the full SO(4) per group. Best quality among the
    block-diagonal rotations (per the rotorquant Llama 3 PPL table)."""

    name = "iso_full"

    def build_matrix(self, head_size, device):
        return _build_iso_matrix(head_size, device, full=True)


register_rotation(IsoFastStrategy.name, IsoFastStrategy)
register_rotation(IsoFullStrategy.name, IsoFullStrategy)
