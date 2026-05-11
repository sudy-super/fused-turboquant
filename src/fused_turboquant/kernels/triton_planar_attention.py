"""
Fused Triton kernel for quantized attention scores with PlanarQuant rotation.

The attention math is identical to the RHT case once the query has been
rotated by the same orthogonal map used at encode time:

    <q, P_inv(centroids[idx])> = <P(q), centroids[idx]>

So if the caller pre-rotates the query with `PlanarRotation.forward(q)`, the
score kernel only needs the packed K indices, the centroid table, and the K
norms — which is exactly what `fused_qk_scores_rht` consumes. We therefore
share that kernel and expose `fused_qk_scores_planar` as a thin wrapper so
the calling code (e.g. `make_fused_attention_forward`) can pick a kernel by
quantizer kind without branching on internal details.

The wrapper exists as a separate function so future PlanarQuant-specific
optimisations (e.g. accepting `rot2` directly and rotating Q inline) can be
added without touching the RHT path.
"""

from __future__ import annotations

import torch

from fused_turboquant.kernels.triton_attention import (
    HAS_TRITON,
    fused_qk_scores_rht,
)


def fused_qk_scores_planar(
    q_rotated: torch.Tensor,
    key_indices: torch.Tensor,
    key_norms: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    bits: int = 3,
) -> torch.Tensor:
    """
    Compute Q · K^T from packed PlanarQuant-encoded keys.

    Args:
        q_rotated: queries pre-rotated by `PlanarRotation.forward()`.
            Shape [batch, n_q_heads, q_len, head_dim].
        key_indices: packed uint8 indices produced by `PlanarQuantMSE.encode`.
            Same packing layout as the RHT encoder, so the underlying kernel
            is shared with `fused_qk_scores_rht`.
        key_norms: float per-vector norms, shape [batch, n_kv_heads, kv_len].
        centroids: Lloyd-Max levels, shape [n_levels].
        scale: attention scaling factor (typically 1/sqrt(head_dim) or a
            model-specific value).
        bits: 2, 3, or 4.

    Returns: scores [batch, n_q_heads, q_len, kv_len].
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused PlanarQuant attention")
    return fused_qk_scores_rht(
        q_rotated,
        key_indices,
        key_norms,
        centroids,
        scale,
        bits=bits,
    )


def planar_rotate_query(q: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """Apply the PlanarQuant forward rotation to a query tensor.

    Convenience wrapper that handles arbitrary leading dims; the caller can
    drop in the result wherever the RHT path uses `randomized_hadamard(q, signs)`.
    """
    from fused_turboquant.core.planar import planar_rotate

    return planar_rotate(q.float(), rot2)
