"""Triton paged attention for boundary-protection FP16 layers.

vLLM's stock flash-attn (FA2 and FA4 alike) reads a paged K/V cache via
non-contiguous slices of `kv_fp16[..., :head_size]` / `[..., head_size:]`
when our boundary cache is laid out as interleaved `[K_fp16 | V_fp16]`
per slot. On Blackwell (RTX PRO 6000, SM 12.0) that path collapses to
mode failure once KV-cache block reuse begins (sample ~128 of GSM-8K),
regardless of FA version. See `v1_backend.py:_forward_raw` for the bug
description.

This module provides a custom Triton decode-attention kernel that
addresses the cache byte-by-byte (uint8 loads then `bitcast` to fp16),
so there is no implicit stride-2 view that FA could misinterpret. It is
CUDA-graph capturable (static shapes, no host sync, no `.item()`) and
intended as a drop-in replacement for `_forward_raw`'s decode branch.

Prefill still goes through `_sdpa_local` on the *current* K, V (no
cache read), matching the pre-d34258a python loop.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _boundary_fp16_decode_attn_kernel(
    # Inputs
    Q_ptr,                # [B, H_q, D] fp16 contiguous
    KV_cache_ptr,         # uint8 view of paged cache [num_blocks, block_size, H_k, slot_bytes]
    Block_table_ptr,      # [B, max_blocks] int32
    Seq_lens_ptr,         # [B] int32
    # Output
    Output_ptr,           # [B, H_q, D] fp16
    # KV cache strides (in BYTES since KV_cache_ptr is uint8)
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    # Q strides (in elements)
    stride_qb,
    stride_qh,
    # Output strides
    stride_outb,
    stride_outh,
    # Block table stride per batch
    stride_bt_b,
    # Compile-time constants
    NUM_KV_GROUPS: tl.constexpr,  # H_q // H_k
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,      # KV cache block size (e.g. 16)
    BLOCK_KV: tl.constexpr,        # K positions per inner-loop tile
    MAX_KV: tl.constexpr,          # block_size × max_blocks
    SCALE: tl.constexpr,           # softmax scale (1/sqrt(d))
):
    """One program per (batch, q_head). Loops over the full
    `[0, MAX_KV)` K range in `BLOCK_KV` tiles, masking out positions
    `>= seq_lens[batch]`. Online softmax + weighted-V accumulation."""
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)

    kv_head = pid_hq // NUM_KV_GROUPS

    d_offs = tl.arange(0, HEAD_DIM)

    # Q row for this (batch, q_head)
    q_base = pid_b * stride_qb + pid_hq * stride_qh
    q_raw = tl.load(Q_ptr + q_base + d_offs).to(tl.float32) * SCALE

    seq_len = tl.load(Seq_lens_ptr + pid_b).to(tl.int32)

    # Online softmax state
    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Iterate K positions in BLOCK_KV-sized chunks.
    for kv_start in tl.range(0, MAX_KV, BLOCK_KV):
        kv_pos = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_pos < seq_len

        # Resolve physical block index for each position via block_table.
        block_in_seq = kv_pos // BLOCK_SIZE
        block_phys = tl.load(
            Block_table_ptr + pid_b * stride_bt_b + block_in_seq,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)
        block_phys_safe = tl.maximum(block_phys, 0)
        pos_in_block = (kv_pos % BLOCK_SIZE).to(tl.int64)

        # Per-position byte base into the cache.
        slot_byte_base = (
            block_phys_safe * stride_cache_block
            + pos_in_block * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )  # [BLOCK_KV] int64

        # ── Load K ─────────────────────────────────────────────────────
        # K bytes are [0, 2*HEAD_DIM) of each slot. Each fp16 element is
        # 2 bytes. We load lo + hi uint8 separately to sidestep any
        # implicit stride-2 view that flash-attn misreads on Blackwell.
        k_lo_addr = slot_byte_base[:, None] + 2 * d_offs[None, :]
        k_hi_addr = k_lo_addr + 1
        k_lo = tl.load(
            KV_cache_ptr + k_lo_addr,
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.uint16)
        k_hi = tl.load(
            KV_cache_ptr + k_hi_addr,
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.uint16)
        k_u16 = k_lo | (k_hi << 8)
        k_fp16 = k_u16.to(tl.float16, bitcast=True)
        k_f32 = k_fp16.to(tl.float32)  # [BLOCK_KV, HEAD_DIM]

        # Scores = q · K^T (scaled). Already pre-scaled Q above.
        scores = tl.sum(q_raw[None, :] * k_f32, axis=1)  # [BLOCK_KV]
        scores = tl.where(kv_mask, scores, -float("inf"))

        # Online softmax update.
        m_new = tl.maximum(tl.max(scores, axis=0), m_prev)
        re_scale = tl.exp(m_prev - m_new)
        p = tl.exp(scores - m_new)

        # ── Load V ─────────────────────────────────────────────────────
        # V bytes are [2*HEAD_DIM, 4*HEAD_DIM) of each slot.
        v_lo_addr = slot_byte_base[:, None] + 2 * HEAD_DIM + 2 * d_offs[None, :]
        v_hi_addr = v_lo_addr + 1
        v_lo = tl.load(
            KV_cache_ptr + v_lo_addr,
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.uint16)
        v_hi = tl.load(
            KV_cache_ptr + v_hi_addr,
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.uint16)
        v_u16 = v_lo | (v_hi << 8)
        v_fp16 = v_u16.to(tl.float16, bitcast=True)
        v_f32 = v_fp16.to(tl.float32)  # [BLOCK_KV, HEAD_DIM]

        acc = acc * re_scale + tl.sum(p[:, None] * v_f32, axis=0)
        l_prev = l_prev * re_scale + tl.sum(p, axis=0)
        m_prev = m_new

    # Normalize and store.
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = acc / safe_l

    out_addr = Output_ptr + pid_b * stride_outb + pid_hq * stride_outh
    tl.store(out_addr + d_offs, out.to(tl.float16))


def boundary_fp16_decode_attention(
    query: torch.Tensor,        # [N, H_q, D] fp16 (N == B for pure decode)
    kv_cache: torch.Tensor,     # [num_blocks, block_size, H_k, slot_bytes_uint8]
    block_table: torch.Tensor,  # [B, max_blocks] int32 / int64
    seq_lens: torch.Tensor,     # [B] int32 / int64
    output: torch.Tensor,       # [N, H_q, D] fp16 (writes only first B rows)
    *,
    num_kv_groups: int,
    scale: float,
    block_kv: int = 16,
) -> torch.Tensor:
    """Launch the decode-only boundary attention. Assumes N == B (one
    query token per sequence). Caller must guarantee this (pure decode)
    or route prefill through `_sdpa_local`."""
    B = seq_lens.shape[0]
    H_q = query.shape[1]
    D = query.shape[2]
    max_blocks = block_table.shape[1]
    block_size_cache = kv_cache.shape[1]
    max_kv = max_blocks * block_size_cache

    kv_u8 = (
        kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
    )

    # Cast block_table and seq_lens to int32 for the kernel.
    bt_int32 = block_table.to(torch.int32) if block_table.dtype != torch.int32 else block_table
    seq_int32 = seq_lens.to(torch.int32) if seq_lens.dtype != torch.int32 else seq_lens

    # Strides for KV cache (in BYTES; the tensor is uint8 so PyTorch's
    # element strides equal byte strides).
    stride_cache_block = kv_u8.stride(0)
    stride_cache_pos = kv_u8.stride(1)
    stride_cache_head = kv_u8.stride(2)

    q_fp16 = query.to(torch.float16).contiguous() if query.dtype != torch.float16 else query.contiguous()

    grid = (B, H_q)
    _boundary_fp16_decode_attn_kernel[grid](
        q_fp16,
        kv_u8,
        bt_int32,
        seq_int32,
        output,
        stride_cache_block=stride_cache_block,
        stride_cache_pos=stride_cache_pos,
        stride_cache_head=stride_cache_head,
        stride_qb=q_fp16.stride(0),
        stride_qh=q_fp16.stride(1),
        stride_outb=output.stride(0),
        stride_outh=output.stride(1),
        stride_bt_b=bt_int32.stride(0),
        NUM_KV_GROUPS=num_kv_groups,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size_cache,
        BLOCK_KV=block_kv,
        MAX_KV=max_kv,
        SCALE=scale,
        num_warps=4,
        num_stages=2,
    )
    return output
