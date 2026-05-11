"""
Fused Triton kernel for PlanarQuant_MSE encoding.

Performs the entire encode pipeline in a single GPU kernel launch:
    input -> 2D Givens rotate per pair -> normalize -> quantize -> pack -> output

The rotation needs only 2·(d/2) floats of cos/sin state, which lives in SRAM
alongside the input row. After rotation, norm computation, normalization,
quantization, and packing all run in registers — no HBM round-trip between
stages.

For 4-bit, the kernel emits final nibble-packed bytes directly (one byte per
pair, since one pair's two indices = exactly one byte). For 3-bit / 2-bit
the kernel writes the per-element indices (uint8 lane per index) and the
Python wrapper applies `pack_3bit` / `pack_2bit` — this still keeps the
rotate / normalize / quantize stages fused, which dominate runtime.

Compared to the RHT-based encode kernel (triton_encode.py), this variant
replaces the O(d log d) butterfly with O(d) Givens rotations: 4 FMAs per
pair. The bit-packing back end (4 / 3 / 2 bit) yields the same
`CompressedTensor` layout, so PlanarQuant is plug-compatible with the
existing packed-key attention kernel.

Reference: scrya-com/rotorquant — turboquant/triton_planarquant.py
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def is_triton_available() -> bool:
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _planar_fused_encode_kernel(
        x_ptr,
        rot2_ptr,           # (n_groups, 2) — cos, sin per pair
        boundaries_ptr,
        packed_out_ptr,
        norms_out_ptr,
        stride_x,
        stride_packed,
        N,
        D: tl.constexpr,
        N_GROUPS: tl.constexpr,
        BITS: tl.constexpr,
        N_LEVELS: tl.constexpr,
        N_BOUNDARIES: tl.constexpr,
        PACK_IN_KERNEL: tl.constexpr,  # 1 for 4-bit, 0 for 3-bit / 2-bit
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        g_idx = tl.arange(0, N_GROUPS)
        g_mask = g_idx < N_GROUPS

        # Gather the even (v0) and odd (v1) lanes of the input row separately.
        # This is cheaper than loading the whole d-dim row and then doing a
        # pairwise reshape inside the kernel.
        v0 = tl.load(x_ptr + pid * stride_x + g_idx * 2, mask=g_mask, other=0.0).to(tl.float32)
        v1 = tl.load(x_ptr + pid * stride_x + g_idx * 2 + 1, mask=g_mask, other=0.0).to(tl.float32)

        cos_t = tl.load(rot2_ptr + g_idx * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_idx * 2 + 1, mask=g_mask, other=0.0)

        # --- 2D Givens rotation per pair (4 FMAs) ---
        r0 = cos_t * v0 - sin_t * v1
        r1 = sin_t * v0 + cos_t * v1

        # --- Norm (Givens is orthogonal but we compute on r* so the regs we
        # already have stay live) ---
        norm_sq = tl.sum(r0 * r0, axis=0) + tl.sum(r1 * r1, axis=0)
        norm = tl.sqrt(norm_sq + 1e-16)
        tl.store(norms_out_ptr + pid, norm.to(tl.float16))

        inv_norm = 1.0 / (norm + 1e-8)
        n0 = r0 * inv_norm
        n1 = r1 * inv_norm

        # --- Quantize: torch.bucketize(x, boundaries) - 1 ---
        q0 = tl.zeros((N_GROUPS,), dtype=tl.int32)
        q1 = tl.zeros((N_GROUPS,), dtype=tl.int32)
        for i in tl.static_range(N_BOUNDARIES):
            b = tl.load(boundaries_ptr + i)
            q0 += (b < n0).to(tl.int32)
            q1 += (b < n1).to(tl.int32)
        q0 = tl.maximum(tl.minimum(q0 - 1, N_LEVELS - 1), 0)
        q1 = tl.maximum(tl.minimum(q1 - 1, N_LEVELS - 1), 0)

        if PACK_IN_KERNEL == 1:
            # 4-bit: nibble-pack each (q0, q1) pair into one byte.
            packed = ((q1 & 0xF) << 4) | (q0 & 0xF)
            tl.store(
                packed_out_ptr + pid * stride_packed + g_idx,
                packed.to(tl.uint8),
                mask=g_mask,
            )
        else:
            # Write one byte per index. Wrapper handles 3-bit / 2-bit packing.
            tl.store(
                packed_out_ptr + pid * stride_packed + g_idx * 2 + 0,
                q0.to(tl.uint8),
                mask=g_mask,
            )
            tl.store(
                packed_out_ptr + pid * stride_packed + g_idx * 2 + 1,
                q1.to(tl.uint8),
                mask=g_mask,
            )

    def triton_planar_fused_encode(
        x: torch.Tensor,
        rot2: torch.Tensor,
        boundaries: torch.Tensor,
        bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fused PlanarQuant_MSE encode: 2D rotate + norm + quantize + pack.

        Args:
            x: input tensor of shape (..., d), d even.
            rot2: rotation table of shape (d/2, 2) as (cos θ, sin θ).
            boundaries: Lloyd-Max bin edges of shape (num_levels + 1,).
            bits: quantization bits (2, 3, or 4).

        Returns:
            (packed_indices, norms) — packed uint8 indices (same layout as
            the RHT encoder, so attention/decoder kernels are interchangeable)
            and float16 norms.
        """
        from fused_turboquant.core.packing import pack_2bit, pack_3bit

        original_shape = x.shape
        d = original_shape[-1]
        if d % 2 != 0:
            raise ValueError(f"head_dim must be even for PlanarQuant, got {d}")

        x_flat = x.contiguous().view(-1, d).float()
        n = x_flat.shape[0]
        n_groups = d // 2
        n_levels = 1 << bits
        n_boundaries = boundaries.shape[0]

        # Kernel writes either nibble-packed bytes (4-bit) or one byte per
        # quantized index (3-bit / 2-bit, packed by the wrapper).
        pack_in_kernel = 1 if bits == 4 else 0
        if bits == 4:
            kernel_packed_dim = n_groups
        else:
            kernel_packed_dim = d

        packed_out = torch.empty((n, kernel_packed_dim), dtype=torch.uint8, device=x.device)
        norms_out = torch.empty(n, dtype=torch.float16, device=x.device)

        grid = (n,)
        _planar_fused_encode_kernel[grid](
            x_flat,
            rot2.contiguous().float(),
            boundaries.contiguous().float(),
            packed_out,
            norms_out,
            x_flat.stride(0),
            packed_out.stride(0),
            n,
            d,
            n_groups,
            bits,
            n_levels,
            n_boundaries,
            pack_in_kernel,
        )

        if bits == 4:
            final_packed_dim = d // 2
            packed = packed_out
        elif bits == 3:
            final_packed_dim = d * 3 // 8
            packed = pack_3bit(packed_out)
        elif bits == 2:
            final_packed_dim = d // 4
            packed = pack_2bit(packed_out)
        else:
            final_packed_dim = d
            packed = packed_out

        norms_shape = list(original_shape[:-1])
        packed_shape = list(original_shape[:-1]) + [final_packed_dim]
        return packed.view(packed_shape), norms_out.view(norms_shape)
