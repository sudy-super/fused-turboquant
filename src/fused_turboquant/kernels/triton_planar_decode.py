"""
Fused Triton kernel for PlanarQuant_MSE decoding.

Performs the entire decode pipeline in a single GPU kernel launch:
    packed input -> unpack -> gather centroids -> denormalize -> inverse 2D rotate -> output

The output buffer doubles as scratch for the inverse rotation (which is just
4 FMAs per pair), so no separate scratch allocation is required.

The bit-packing unpack logic mirrors `triton_decode.py` (RHT decode) so the
same packed layout is interchangeable between PlanarQuant and RHT.

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
    def _planar_fused_decode_kernel(
        packed_ptr,
        norms_ptr,
        levels_ptr,
        rot2_ptr,            # (n_groups, 2) — cos, sin per pair
        out_ptr,
        stride_packed,
        stride_out,
        N,
        D: tl.constexpr,
        N_GROUPS: tl.constexpr,
        BITS: tl.constexpr,
        N_LEVELS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        g_idx = tl.arange(0, N_GROUPS)
        g_mask = g_idx < N_GROUPS

        # --- Unpack the two indices of each pair (q0 at even lane, q1 at odd lane) ---
        if BITS == 4:
            # One packed byte holds (q1 << 4) | q0 per pair.
            packed_val = tl.load(
                packed_ptr + pid * stride_packed + g_idx,
                mask=g_mask,
                other=0,
            ).to(tl.int32)
            idx0 = packed_val & 0xF
            idx1 = (packed_val >> 4) & 0xF
        elif BITS == 3:
            # Bitstream: bit-offset for index i is i*3. q0 lives at even i = g*2,
            # q1 at odd i = g*2 + 1. We unpack each separately.
            packed_total = D * 3 // 8
            row_base = packed_ptr + pid * stride_packed

            bit_off0 = g_idx * 2 * 3
            byte_idx0 = bit_off0 >> 3
            bit_shift0 = bit_off0 & 7
            b0_lo = tl.load(row_base + byte_idx0, mask=g_mask, other=0).to(tl.int32)
            b0_hi = tl.load(
                row_base + byte_idx0 + 1,
                mask=g_mask & ((byte_idx0 + 1) < packed_total),
                other=0,
            ).to(tl.int32)
            idx0 = ((b0_lo | (b0_hi << 8)) >> bit_shift0) & 0x7

            bit_off1 = (g_idx * 2 + 1) * 3
            byte_idx1 = bit_off1 >> 3
            bit_shift1 = bit_off1 & 7
            b1_lo = tl.load(row_base + byte_idx1, mask=g_mask, other=0).to(tl.int32)
            b1_hi = tl.load(
                row_base + byte_idx1 + 1,
                mask=g_mask & ((byte_idx1 + 1) < packed_total),
                other=0,
            ).to(tl.int32)
            idx1 = ((b1_lo | (b1_hi << 8)) >> bit_shift1) & 0x7
        elif BITS == 2:
            # One packed byte holds (q3<<6)|(q2<<4)|(q1<<2)|q0 for two pairs.
            pack_idx = g_idx // 2
            within = (g_idx & 1)  # 0 → pair (q0,q1) at low bits, 1 → high bits
            packed_val = tl.load(
                packed_ptr + pid * stride_packed + pack_idx,
                mask=g_mask,
                other=0,
            ).to(tl.int32)
            shift0 = within * 4 + 0
            shift1 = within * 4 + 2
            idx0 = (packed_val >> shift0) & 0x3
            idx1 = (packed_val >> shift1) & 0x3
        else:
            idx0 = tl.load(
                packed_ptr + pid * stride_packed + g_idx * 2 + 0,
                mask=g_mask,
                other=0,
            ).to(tl.int32)
            idx1 = tl.load(
                packed_ptr + pid * stride_packed + g_idx * 2 + 1,
                mask=g_mask,
                other=0,
            ).to(tl.int32)

        # --- Gather centroids ---
        q0 = tl.load(levels_ptr + idx0, mask=g_mask, other=0.0).to(tl.float32)
        q1 = tl.load(levels_ptr + idx1, mask=g_mask, other=0.0).to(tl.float32)

        # --- Denormalize by row norm ---
        norm = tl.load(norms_ptr + pid).to(tl.float32)
        q0 = q0 * norm
        q1 = q1 * norm

        # --- Inverse 2D Givens rotation (transpose of forward: negate sin) ---
        cos_t = tl.load(rot2_ptr + g_idx * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_idx * 2 + 1, mask=g_mask, other=0.0)

        f0 = cos_t * q0 + sin_t * q1
        f1 = -sin_t * q0 + cos_t * q1

        # Store back in interleaved (even/odd) layout
        tl.store(
            out_ptr + pid * stride_out + g_idx * 2 + 0,
            f0,
            mask=g_mask,
        )
        tl.store(
            out_ptr + pid * stride_out + g_idx * 2 + 1,
            f1,
            mask=g_mask,
        )

    def triton_planar_fused_decode(
        packed_indices: torch.Tensor,
        norms: torch.Tensor,
        levels: torch.Tensor,
        rot2: torch.Tensor,
        bits: int,
        original_dim: int,
    ) -> torch.Tensor:
        """
        Fused PlanarQuant_MSE decode: unpack + dequant + denorm + inverse 2D rotate.

        Args:
            packed_indices: packed uint8 indices, layout matches the encoder.
            norms: float16 / float32 row norms.
            levels: Lloyd-Max centroids of shape (num_levels,).
            rot2: rotation table of shape (d/2, 2).
            bits: quantization bits (2, 3, or 4).
            original_dim: the original head_dim d.

        Returns:
            Decoded tensor of shape matching the original input (float32).
        """
        d = original_dim
        if d % 2 != 0:
            raise ValueError(f"head_dim must be even for PlanarQuant, got {d}")
        n_groups = d // 2
        n_levels = 1 << bits

        packed_flat = packed_indices.contiguous().view(-1, packed_indices.shape[-1])
        norms_flat = norms.contiguous().view(-1)
        n = norms_flat.shape[0]

        out = torch.empty((n, d), dtype=torch.float32, device=packed_indices.device)

        grid = (n,)
        _planar_fused_decode_kernel[grid](
            packed_flat,
            norms_flat,
            levels.contiguous().float(),
            rot2.contiguous().float(),
            out,
            packed_flat.stride(0),
            out.stride(0),
            n,
            d,
            n_groups,
            bits,
            n_levels,
        )

        original_shape = list(norms.shape) + [d]
        return out.view(original_shape)
