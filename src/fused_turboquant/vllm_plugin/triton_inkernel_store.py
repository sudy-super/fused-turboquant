"""Fused store kernels with in-kernel rotation.

These kernels replace the two-step "external cuBLAS GEMM rotation" + "stock
`_tq_fused_store_mse`" path with a single Triton kernel that does:

    raw K  ─►  norm  ─►  normalize  ─►  rotate (in-kernel)  ─►
            bucketize  ─►  pack MSE indices  ─►  store norm  ─►
            uniform V quant  ─►  pack V  ─►  slot scatter

For each rotation kind we ship a dedicated kernel that uses HBM scratch to
emulate cross-lane access (Triton has no efficient intra-program permute).
The rotation step itself differs per strategy:

  - rht:    `y = x @ H` where H is a Sylvester Hadamard. We use the FWHT
            butterfly (LOG2_D stages) — O(D log D) FMAs.
  - planar: per-pair Givens rotation — O(D) FMAs, 4 per pair.
  - rotor:  3×3 block-diagonal matmul — O(D) FMAs, 9 per block.
  - iso:    4×4 block-diagonal matmul — O(D) FMAs, 16 per block.

The downstream bucketize+pack+V-quant+slot-scatter logic is shared and
matches `vllm.v1.attention.ops.triton_turboquant_store._tq_fused_store_mse`
byte-for-byte so the on-disk slot layout is identical (and the existing
`_tq_decode_stage1` kernel reads back compatibly).
"""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton


# ─── Shared downstream: bucketize + pack MSE + store norm + V quant + slot scatter ─


@triton.jit
def _bucketize_pack_norm_v_store(
    y_vec,            # rotated normalized K vector (register)
    norm,             # K norm scalar (register)
    Value_ptr,        # raw V tensor
    Midpoints_ptr,
    KV_cache_ptr,
    base,             # pid * D
    slot_base,        # byte offset into KV_cache_ptr for (slot, head)
    d_offs,           # tl.arange(0, BLOCK_D)
    d_mask,           # d_offs < D
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Identical layout to vLLM's `_tq_fused_store_mse` from this point on."""
    # ── 1. BINARY SEARCH BUCKETIZE ───────────────────────────────────
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx_q = tl.minimum(lo, N_CENTROIDS - 1)

    # ── 2. PACK MSE INDICES ─────────────────────────────────────────
    if MSE_BITS == 4:
        idx_pairs = tl.reshape(idx_q, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)
    elif MSE_BITS == 3:
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx_q, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # ── 3. STORE NORM (fp16, 2 bytes) ──────────────────────────────
    norm_offset = MSE_BYTES
    vn_f16 = norm.to(tl.float16)
    vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
    tl.store(KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8))

    # ── 4. VALUE QUANTIZE + PACK ──────────────────────────────────
    val_cache_offset = KPS
    val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    val_min = tl.min(tl.where(d_mask, val_vec, float("inf")), axis=0)
    val_max = tl.max(tl.where(d_mask, val_vec, -float("inf")), axis=0)

    if VQB == 3:
        v_scale = (val_max - val_min) / 7.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
        q_vals = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 7
        )
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        q_grp = tl.reshape(q_vals, [BLOCK_GRP, 8])
        shifts_3bit = tl.arange(0, 8) * 3
        packed_24 = tl.sum(q_grp << shifts_3bit[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3, b0, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(KV_cache_ptr + slot_base + val_cache_offset + grp_offs * 3 + 2, b2, mask=grp_mask)
        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + sc_offset + 1, ((sc_u16 >> 8) & 0xFF).to(tl.uint8))
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + sc_offset + 3, ((zr_u16 >> 8) & 0xFF).to(tl.uint8))
    else:  # VQB == 4
        v_scale = (val_max - val_min) / 15.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 15
        )
        q_pairs = tl.reshape(q_all, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed_val = tl.sum((q_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        val_offs = tl.arange(0, BLOCK_D // 2)
        val_mask = val_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + val_offs,
            packed_val,
            mask=val_mask,
        )
        sc_offset = val_cache_offset + VAL_DATA_BYTES
        sc_f16 = v_scale.to(tl.float16)
        sc_u16 = sc_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (sc_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + sc_offset + 1, ((sc_u16 >> 8) & 0xFF).to(tl.uint8))
        zr_f16 = val_min.to(tl.float16)
        zr_u16 = zr_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (zr_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + sc_offset + 3, ((zr_u16 >> 8) & 0xFF).to(tl.uint8))


# ─── Helper: compute slot_base and the early bail check ─────────────────────


@triton.jit
def _slot_base(
    Slot_mapping_ptr,
    pid,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
):
    token_idx = pid // H
    head_idx = pid % H
    slot = tl.load(Slot_mapping_ptr + token_idx)
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    return slot, (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )


# ═══════════════════════════════════════════════════════════════════════
# RHT (Hadamard butterfly) — in-kernel FWHT via HBM scratch
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _fused_store_rht(
    Key_raw_ptr,      # [NH, D] float32 raw K
    Value_ptr,        # [NH, D] float32 raw V
    Scratch_ptr,      # [NH, D] float32 HBM scratch for butterfly
    Midpoints_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    LOG2_D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """RHT in-kernel: normalize → FWHT butterfly → fused store."""
    pid = tl.program_id(0)
    slot, slot_base = _slot_base(
        Slot_mapping_ptr, pid, H, BLOCK_SIZE,
        stride_cache_block, stride_cache_pos, stride_cache_head,
    )
    if slot < 0:
        return

    base = pid * D
    scratch_off = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 0. Load raw K, compute norm, normalize ─────────────────────
    k_raw = tl.load(Key_raw_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(k_raw * k_raw, axis=0)
    norm = tl.sqrt(norm_sq + 1e-16)
    x_hat = k_raw / (norm + 1e-8)

    # ── 1. FWHT butterfly (Sylvester, no random signs) ──────────────
    # Use HBM scratch since Triton can't do efficient cross-lane permute.
    tl.store(Scratch_ptr + scratch_off + d_offs, x_hat, mask=d_mask)
    tl.debug_barrier()
    for s in tl.static_range(LOG2_D):
        h = tl.constexpr(1 << s)
        xi = tl.load(Scratch_ptr + scratch_off + d_offs, mask=d_mask, other=0.0)
        xp = tl.load(
            Scratch_ptr + scratch_off + (d_offs ^ h),
            mask=((d_offs ^ h) < D),
            other=0.0,
        )
        is_even = (d_offs & h) == 0
        result = tl.where(is_even, xi + xp, xp - xi)
        tl.store(Scratch_ptr + scratch_off + d_offs, result, mask=d_mask)
        tl.debug_barrier()
    y_vec = tl.load(Scratch_ptr + scratch_off + d_offs, mask=d_mask, other=0.0)
    y_vec = y_vec * (1.0 / tl.sqrt(float(D)))

    # ── 2-5. Bucketize + pack + norm + V quant + scatter ───────────
    _bucketize_pack_norm_v_store(
        y_vec, norm, Value_ptr, Midpoints_ptr, KV_cache_ptr,
        base, slot_base, d_offs, d_mask,
        D=D, BLOCK_D=BLOCK_D, MSE_BYTES=MSE_BYTES, KPS=KPS,
        VQB=VQB, VAL_DATA_BYTES=VAL_DATA_BYTES, BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=MSE_BITS, N_CENTROIDS=N_CENTROIDS, BLOCK_GRP=BLOCK_GRP,
    )


# ═══════════════════════════════════════════════════════════════════════
# Planar / Rotor / Iso — block-diagonal rotation, materialized as (D, D)
# ═══════════════════════════════════════════════════════════════════════
#
# For block-diagonal rotations (Planar 2×2, Rotor 3×3, Iso 4×4), the
# matrix M is sparse with only `block_size * D` nonzero entries. But the
# easy way to implement the rotation in-kernel is to do a (1, D) × (D, D)
# matmul via HBM scratch.
#
# We treat all three (and any future MatrixRotationStrategy) with one
# generic kernel that loads the (D, D) matrix from a precomputed tensor.
# The matmul is row-by-row: for each output dim `d_out`, dot-product the
# `d_out`-th column of M with x_hat. To avoid `D**2` register pressure
# we materialize x_hat to HBM scratch and broadcast-multiply with M.


@triton.jit
def _fused_store_matrix(
    Key_raw_ptr,      # [NH, D] float32
    Value_ptr,        # [NH, D] float32
    Rotation_ptr,     # [D, D] float32 — precomputed by strategy
    Midpoints_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
):
    """Generic in-kernel (D,D) rotation. Used by Planar/Rotor/Iso.

    Per program (= one (token, head) pair):
      1. Load raw K vector into registers
      2. Compute norm + normalize in registers
      3. Load the rotation matrix M as a (BLOCK_D, BLOCK_D) tile
      4. Compute y[d_out] = sum_{d_in} x_hat[d_in] * M[d_in, d_out]
         via broadcast multiply + reduction
      5. Bucketize + pack + V quant + slot scatter

    For D ≤ 256 the BLOCK_D × BLOCK_D tile fits in SRAM (256KB float32),
    so this is a single in-program GEMM with no HBM round-trip on the
    rotation step.
    """
    pid = tl.program_id(0)
    slot, slot_base = _slot_base(
        Slot_mapping_ptr, pid, H, BLOCK_SIZE,
        stride_cache_block, stride_cache_pos, stride_cache_head,
    )
    if slot < 0:
        return

    base = pid * D
    d_in_offs = tl.arange(0, BLOCK_D)
    d_out_offs = tl.arange(0, BLOCK_D)
    d_in_mask = d_in_offs < D
    d_out_mask = d_out_offs < D

    # ── 0. Load raw K, compute norm, normalize ─────────────────────
    k_raw = tl.load(Key_raw_ptr + base + d_in_offs, mask=d_in_mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(k_raw * k_raw, axis=0)
    norm = tl.sqrt(norm_sq + 1e-16)
    x_hat = k_raw / (norm + 1e-8)

    # ── 1. Rotation: y = x_hat @ M ─────────────────────────────────
    # Load M as a (BLOCK_D, BLOCK_D) tile. Each lane holds one row of M.
    M_tile = tl.load(
        Rotation_ptr + d_in_offs[:, None] * D + d_out_offs[None, :],
        mask=d_in_mask[:, None] & d_out_mask[None, :],
        other=0.0,
    )
    # Broadcast-multiply x_hat against M_tile and reduce over d_in.
    # prods[d_in, d_out] = x_hat[d_in] * M[d_in, d_out]
    prods = x_hat[:, None] * M_tile
    y_vec = tl.sum(prods, axis=0)

    # ── 2-5. Downstream fused store ───────────────────────────────
    _bucketize_pack_norm_v_store(
        y_vec, norm, Value_ptr, Midpoints_ptr, KV_cache_ptr,
        base, slot_base, d_out_offs, d_out_mask,
        D=D, BLOCK_D=BLOCK_D, MSE_BYTES=MSE_BYTES, KPS=KPS,
        VQB=VQB, VAL_DATA_BYTES=VAL_DATA_BYTES, BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=MSE_BITS, N_CENTROIDS=N_CENTROIDS, BLOCK_GRP=BLOCK_GRP,
    )


# ─── Launcher helpers ────────────────────────────────────────────────────────


def _launch_rht(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    midpoints: torch.Tensor,
    tq_config,
    head_size: int,
):
    """RHT-specific launcher: in-kernel FWHT butterfly + fused store."""
    N, H_kv, D = key.shape
    NH = N * H_kv
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    log2_d = D.bit_length() - 1
    if (1 << log2_d) != D:
        raise ValueError(f"RHT requires power-of-2 head_dim, got {D}")

    mse_bytes = math.ceil(D * tq_config.key_mse_bits / 8)
    n_centroids = 2 ** tq_config.key_mse_bits
    val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1

    k_flat = key.float().reshape(NH, D).contiguous()
    v_flat = value.float().reshape(NH, D).contiguous()
    scratch = torch.empty((NH, D), dtype=torch.float32, device=key.device)

    grid = (NH,)
    _fused_store_rht[grid](
        k_flat, v_flat, scratch, midpoints,
        kv_cache.view(-1), slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=D,
        LOG2_D=log2_d,
        H=H_kv,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        KPS=tq_config.key_packed_size,
        VQB=tq_config.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=tq_config.key_mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=BLOCK_GRP,
        num_warps=4,
        num_stages=1,
    )


def _launch_matrix(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation_matrix: torch.Tensor,
    midpoints: torch.Tensor,
    tq_config,
    head_size: int,
):
    """Generic matrix-rotation launcher (Planar / Rotor / Iso)."""
    N, H_kv, D = key.shape
    NH = N * H_kv
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    mse_bytes = math.ceil(D * tq_config.key_mse_bits / 8)
    n_centroids = 2 ** tq_config.key_mse_bits
    val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1

    k_flat = key.float().reshape(NH, D).contiguous()
    v_flat = value.float().reshape(NH, D).contiguous()

    grid = (NH,)
    _fused_store_matrix[grid](
        k_flat, v_flat, rotation_matrix, midpoints,
        kv_cache.view(-1), slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=D,
        H=H_kv,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        MSE_BYTES=mse_bytes,
        KPS=tq_config.key_packed_size,
        VQB=tq_config.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=tq_config.key_mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=BLOCK_GRP,
        num_warps=4,
        num_stages=1,
    )
