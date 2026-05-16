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
    KEY_FP8: tl.constexpr = 0,         # 1 = store K as raw FP8 instead of MSE indices
    FP8_E4B15: tl.constexpr = 0,       # 1 = use e4b15 (SM<8.9), 0 = e4nv (Hopper+/Blackwell)
    V_LLOYD_MAX: tl.constexpr = 0,     # 1 = quantize V via Lloyd-Max (unit-norm + same midpoints as K)
                                       # 0 = existing per-vector uniform quant
):
    """Bucketize+pack the rotated K vector and quantize V into the
    paged-cache slot.

    Two storage modes for keys, selected at compile time:

    - `KEY_FP8 == 0` (default, MSE): bucketize `y_vec` against the
      Lloyd-Max midpoints, pack 2/3/4/8-bit indices, then store the
      per-vector FP16 norm. Slot layout: `[indices … | norm(2B)]`.
    - `KEY_FP8 == 1`: skip bucketize. Multiply `y_vec * norm` to undo
      the in-kernel normalization, cast to FP8 (e4b15 / e4nv per
      `FP8_E4B15`), store 1 byte per dim. Slot layout: `[fp8 K bytes]`,
      no norm slot.

    In both modes the value tensor is uniform-quantized and packed
    after the key region (offset `KPS = key_packed_size`)."""
    # ── 1. STORE KEYS ──────────────────────────────────────────────
    if KEY_FP8:
        # FP8 K path: undo normalization, cast, store raw bytes.
        y_unnorm = y_vec * norm
        if FP8_E4B15:
            k_fp8 = y_unnorm.to(tl.float8e4b15)
        else:
            k_fp8 = y_unnorm.to(tl.float8e4nv)
        k_fp8_u8 = k_fp8.to(tl.uint8, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + d_offs, k_fp8_u8, mask=d_mask)
        # No norm storage in FP8 mode.
    else:
        # MSE path: binary-search the centroid table.
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
        elif MSE_BITS == 2:
            # 4 two-bit indices per byte. BLOCK_D / 4 must be a positive power
            # of 2 — guaranteed since BLOCK_D = next_pow2(head_dim) and 4 | BLOCK_D
            # for the head_sizes we support (head_dim >= 64).
            idx_quads = tl.reshape(idx_q, [BLOCK_D // 4, 4])
            shifts_2 = tl.arange(0, 4) * 2
            packed = tl.sum((idx_quads & 0x3) << shifts_2[None, :], axis=1).to(tl.uint8)
            mse_offs = tl.arange(0, BLOCK_D // 4)
            mse_mask = mse_offs < MSE_BYTES
            tl.store(KV_cache_ptr + slot_base + mse_offs, packed, mask=mse_mask)
        elif MSE_BITS == 8:
            # 1 byte per index, no packing — used by boundary-protect
            # TQ at 8-bit (256 Lloyd-Max centroids).
            packed = idx_q.to(tl.uint8)
            mse_mask_8 = d_offs < MSE_BYTES
            tl.store(KV_cache_ptr + slot_base + d_offs, packed, mask=mse_mask_8)

        # ── 3. STORE NORM (fp16, 2 bytes) ──────────────────────────────
        norm_offset = MSE_BYTES
        vn_f16 = norm.to(tl.float16)
        vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + norm_offset, (vn_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + norm_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8))

    # ── 4. VALUE QUANTIZE + PACK ──────────────────────────────────
    val_cache_offset = KPS
    val_vec = tl.load(Value_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    if V_LLOYD_MAX and VQB == 4:
        # Lloyd-Max V path: unit-normalize, bucketize against the SAME midpoints
        # as the keys, pack 4-bit indices, store per-vector norm where the
        # uniform path would have stored scale. Zero-point byte is left as 0
        # so the slot layout (and `key_packed_size` / `value_data_bytes`) is
        # unchanged — the decode kernel reads `V_LLOYD_MAX` to interpret the
        # stored bytes correctly.
        v_norm_sq = tl.sum(val_vec * val_vec, axis=0)
        v_norm = tl.sqrt(v_norm_sq + 1e-16)
        v_unit = val_vec / (v_norm + 1e-8)

        v_lo = tl.zeros([BLOCK_D], dtype=tl.int32)
        v_hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
        for _ in range(MSE_BITS):
            v_mid = (v_lo + v_hi) >> 1
            v_safe_mid = tl.minimum(v_mid, N_CENTROIDS - 2)
            v_mid_val = tl.load(Midpoints_ptr + v_safe_mid, mask=d_mask, other=0.0)
            v_lo = tl.where(v_unit >= v_mid_val, v_mid + 1, v_lo)
            v_hi = tl.where(v_unit >= v_mid_val, v_hi, v_mid)
        v_idx = tl.minimum(v_lo, N_CENTROIDS - 1)

        v_pairs = tl.reshape(v_idx, [BLOCK_D // 2, 2])
        v_shifts = tl.arange(0, 2) * 4
        v_packed = tl.sum((v_pairs & 0xF) << v_shifts[None, :], axis=1).to(tl.uint8)
        v_offs = tl.arange(0, BLOCK_D // 2)
        v_mask = v_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + v_offs,
            v_packed,
            mask=v_mask,
        )
        sc_offset = val_cache_offset + VAL_DATA_BYTES
        vn_f16 = v_norm.to(tl.float16)
        vn_u16 = vn_f16.to(tl.uint16, bitcast=True)
        tl.store(KV_cache_ptr + slot_base + sc_offset, (vn_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + sc_offset + 1, ((vn_u16 >> 8) & 0xFF).to(tl.uint8))
        # Zero out unused 2 bytes (the uniform path's zero-point slot).
        # Use the same fp16 0.0 trick to avoid 0-d tensor edge cases.
        z_u16 = tl.full([], 0, tl.uint16)
        tl.store(KV_cache_ptr + slot_base + sc_offset + 2, (z_u16 & 0xFF).to(tl.uint8))
        tl.store(KV_cache_ptr + slot_base + sc_offset + 3, ((z_u16 >> 8) & 0xFF).to(tl.uint8))
        return

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
    elif VQB == 2:
        # 2-bit uniform: 4 levels, packed 4 entries per byte.
        v_scale = (val_max - val_min) / 3.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 3
        )
        q_quads = tl.reshape(q_all, [BLOCK_D // 4, 4])
        shifts_2 = tl.arange(0, 4) * 2
        packed_val = tl.sum((q_quads & 0x3) << shifts_2[None, :], axis=1).to(tl.uint8)
        val_offs = tl.arange(0, BLOCK_D // 4)
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
    elif VQB == 8:
        # 8-bit uniform: 256 levels, 1 byte per entry, no packing.
        v_scale = (val_max - val_min) / 255.0
        v_scale = tl.where(v_scale > 1e-8, v_scale, 1e-8)
        q_all = tl.minimum(
            tl.maximum(((val_vec - val_min) / v_scale + 0.5).to(tl.int32), 0), 255
        )
        packed_val = q_all.to(tl.uint8)
        val_mask_8 = d_offs < VAL_DATA_BYTES
        tl.store(
            KV_cache_ptr + slot_base + val_cache_offset + d_offs,
            packed_val,
            mask=val_mask_8,
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
    KEY_FP8: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    V_LLOYD_MAX: tl.constexpr = 0,
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
        KEY_FP8=KEY_FP8, FP8_E4B15=FP8_E4B15,
        V_LLOYD_MAX=V_LLOYD_MAX,
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
    KEY_FP8: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    V_LLOYD_MAX: tl.constexpr = 0,
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
        KEY_FP8=KEY_FP8, FP8_E4B15=FP8_E4B15,
        V_LLOYD_MAX=V_LLOYD_MAX,
    )


# ─── Launcher helpers ────────────────────────────────────────────────────────


def _use_fp8_e4b15(device: torch.device) -> bool:
    """Return True if FP8 e4b15 should be used (SM < 8.9, Ampere/early Ada).
    Hopper / Blackwell prefer e4nv which has no bias adjustment."""
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) < (8, 9)


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

    key_fp8 = bool(getattr(tq_config, "key_fp8", False))
    import os as _os
    v_lloyd_max = 1 if _os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1" else 0
    # MSE bits drops to 0 in FP8 mode (`BoundaryTurboQuantConfig.key_mse_bits`).
    # Force MSE_BITS to a small positive value so Triton's constexpr binary
    # search loop compiles, but the entire MSE branch is dead-code under
    # KEY_FP8=1 — the compiler eliminates it.
    eff_mse_bits = tq_config.key_mse_bits if not key_fp8 else 2
    mse_bytes = math.ceil(D * eff_mse_bits / 8) if not key_fp8 else 0
    n_centroids = 2 ** eff_mse_bits
    val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1
    fp8_e4b15 = _use_fp8_e4b15(key.device) if key_fp8 else False

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
        MSE_BITS=eff_mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=BLOCK_GRP,
        KEY_FP8=1 if key_fp8 else 0,
        FP8_E4B15=1 if fp8_e4b15 else 0,
        V_LLOYD_MAX=v_lloyd_max,
        num_warps=4,
        num_stages=1,
    )


# ═══════════════════════════════════════════════════════════════════════
# Block-diagonal in-kernel rotation (Planar B=2, Rotor B=3, Iso B=4)
# ═══════════════════════════════════════════════════════════════════════
#
# These rotations have block-diagonal structure: M is zero outside the
# B×B blocks along the diagonal. The generic (D, D) matmul above wastes
# O(D²) FLOPs and HBM bandwidth on the off-block zeros; here we exploit
# the structure to do only O(B·D) work per token.
#
# Storage per layer (vs (D, D) = D² floats):
#   B=2: 2·D floats   (~64x smaller for D=128)
#   B=3: 3·D floats   (3 × 128 = 384 vs 16384 for D=128)
#   B=4: 4·D floats
#
# Algorithm:
#   For each output position d:
#       group       = d // B
#       group_start = group * B
#       y[d] = Σ_{k=0..B-1}  coeffs[k, d] * x_hat[group_start + k]
#   where coeffs[k, d] = M[group_start + k, d] from the (D, D) matrix.
#
# Triton lacks efficient cross-lane permute, so the gather `x_hat[group_start + k]`
# goes through HBM scratch — same trick as the RHT butterfly, but with
# only B (not log2(D)) round-trips. For B=2/3/4 this is 2x–4x fewer
# round-trips than RHT (which needs log2(128)=7 or log2(256)=8 stages).
#
# Coeff layout: [B, D] row-major. Loading `coeffs[k, *]` for a fixed k
# is a contiguous read across `d_offs`, which coalesces perfectly.


@triton.jit
def _fused_store_block_diag(
    Key_raw_ptr,      # [NH, D] float32
    Value_ptr,        # [NH, D] float32
    Coeffs_ptr,       # [B, D] float32 — precomputed block-rotation coefficients
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
    B: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    MSE_BITS: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr = 16,
    KEY_FP8: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,
    V_LLOYD_MAX: tl.constexpr = 0,
):
    """Block-diagonal in-kernel rotation: normalize → in-register
    block rotation (no HBM scratch) → fused store.

    The previous implementation staged `x_hat` to a per-program HBM
    scratch and re-loaded `B` permuted views from it for the rotation.
    That path was needed for RHT's butterfly because every stage's
    partner indices are XOR-based across the whole tile, but for the
    block-diagonal case the partner index is just `(d // B) * B + k`,
    a small *strided* gather. The compiler can serve those B re-loads
    of `Key_raw_ptr + base + partner_pos` from L1/L2 (the whole `D`-byte
    row is already in cache from the norm load), so we skip the scratch
    round-trip entirely. Each call now does:

      1 raw-K load (norm pass) + B raw-K cache hits + B coeff loads
    instead of
      1 raw-K load + 1 scratch store + B scratch loads + B coeff loads.

    Net: `(B+1) * BLOCK_D * 4` fewer scratch bytes per call, no
    `tl.debug_barrier()`. Removes the scratch allocation entirely from
    the launcher."""
    pid = tl.program_id(0)
    slot, slot_base = _slot_base(
        Slot_mapping_ptr, pid, H, BLOCK_SIZE,
        stride_cache_block, stride_cache_pos, stride_cache_head,
    )
    if slot < 0:
        return

    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # ── 0. Load raw K, compute norm ─────────────────────────────────
    k_raw = tl.load(Key_raw_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(k_raw * k_raw, axis=0)
    norm = tl.sqrt(norm_sq + 1e-16)
    inv_norm = 1.0 / (norm + 1e-8)

    # ── 1. Block-wise rotation: y[d] = Σ_k coeffs[k, d] · x_hat[group_start + k]
    # Each iteration re-reads `Key_raw_ptr + base + partner_pos` (an L1
    # hit after the norm-pass load), normalizes per lane, and fuses with
    # the `coeffs[k, d]` row. No cross-lane HBM scratch.
    group_start = (d_offs // B) * B
    y_vec = tl.zeros([BLOCK_D], dtype=tl.float32)
    for k in tl.static_range(B):
        partner_pos = group_start + k
        partner_mask = (partner_pos < D) & d_mask
        k_at_partner = tl.load(
            Key_raw_ptr + base + partner_pos,
            mask=partner_mask, other=0.0,
        ).to(tl.float32)
        x_at_partner = k_at_partner * inv_norm
        coeff_k = tl.load(
            Coeffs_ptr + k * D + d_offs,
            mask=d_mask, other=0.0,
        )
        y_vec = y_vec + x_at_partner * coeff_k

    # ── 2-5. Bucketize + pack + norm + V quant + slot scatter ──────
    _bucketize_pack_norm_v_store(
        y_vec, norm, Value_ptr, Midpoints_ptr, KV_cache_ptr,
        base, slot_base, d_offs, d_mask,
        D=D, BLOCK_D=BLOCK_D, MSE_BYTES=MSE_BYTES, KPS=KPS,
        VQB=VQB, VAL_DATA_BYTES=VAL_DATA_BYTES, BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=MSE_BITS, N_CENTROIDS=N_CENTROIDS, BLOCK_GRP=BLOCK_GRP,
        KEY_FP8=KEY_FP8, FP8_E4B15=FP8_E4B15,
        V_LLOYD_MAX=V_LLOYD_MAX,
    )


def _launch_block_diag(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    coeffs: torch.Tensor,   # [B, D] precomputed
    midpoints: torch.Tensor,
    tq_config,
    head_size: int,
    block_size: int,
):
    """Block-diagonal launcher (Planar B=2 / Rotor B=3 / Iso B=4)."""
    N, H_kv, D = key.shape
    NH = N * H_kv
    cache_bs = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    key_fp8 = bool(getattr(tq_config, "key_fp8", False))
    import os as _os
    v_lloyd_max = 1 if _os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1" else 0
    eff_mse_bits = tq_config.key_mse_bits if not key_fp8 else 2
    mse_bytes = math.ceil(D * eff_mse_bits / 8) if not key_fp8 else 0
    n_centroids = 2 ** eff_mse_bits
    val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1
    fp8_e4b15 = _use_fp8_e4b15(key.device) if key_fp8 else False

    k_flat = key.float().reshape(NH, D).contiguous()
    v_flat = value.float().reshape(NH, D).contiguous()

    grid = (NH,)
    _fused_store_block_diag[grid](
        k_flat, v_flat, coeffs, midpoints,
        kv_cache.view(-1), slot_mapping,
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=D,
        H=H_kv,
        BLOCK_SIZE=cache_bs,
        BLOCK_D=BLOCK_D,
        B=block_size,
        MSE_BYTES=mse_bytes,
        KPS=tq_config.key_packed_size,
        VQB=tq_config.effective_value_quant_bits,
        VAL_DATA_BYTES=val_data_bytes,
        BLOCK_VAL=BLOCK_VAL,
        MSE_BITS=eff_mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=BLOCK_GRP,
        KEY_FP8=1 if key_fp8 else 0,
        FP8_E4B15=1 if fp8_e4b15 else 0,
        V_LLOYD_MAX=v_lloyd_max,
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
    """Generic matrix-rotation launcher (fallback / debugging)."""
    N, H_kv, D = key.shape
    NH = N * H_kv
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)

    key_fp8 = bool(getattr(tq_config, "key_fp8", False))
    import os as _os
    v_lloyd_max = 1 if _os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1" else 0
    eff_mse_bits = tq_config.key_mse_bits if not key_fp8 else 2
    mse_bytes = math.ceil(D * eff_mse_bits / 8) if not key_fp8 else 0
    n_centroids = 2 ** eff_mse_bits
    val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
    BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
    BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1
    fp8_e4b15 = _use_fp8_e4b15(key.device) if key_fp8 else False

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
        MSE_BITS=eff_mse_bits,
        N_CENTROIDS=n_centroids,
        BLOCK_GRP=BLOCK_GRP,
        KEY_FP8=1 if key_fp8 else 0,
        FP8_E4B15=1 if fp8_e4b15 else 0,
        V_LLOYD_MAX=v_lloyd_max,
        num_warps=4,
        num_stages=1,
    )
