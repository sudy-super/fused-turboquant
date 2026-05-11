"""
fused-turboquant attention backend for vLLM v1 (vllm >= 0.20).

This backend overrides `AttentionBackendEnum.TURBOQUANT` to:
  - Add compatibility for layers vLLM's stock TurboQuant rejects
    (multimodal mm_prefix; head_size > 256 in flash-attn-2's varlen path).
  - Plug in a Planar (2D Givens) rotation alternative to the stock RHT
    quantizer via the `TURBOQUANT_KIND=planar` env var.
  - Disable vLLM's boundary-protection auto-skip (first/last 2 attention
    layers being forced to `kv_cache_dtype="auto"`), so every layer
    flows through the fast Triton path. See
    `plugin._patch_disable_boundary_protection`.

For RHT (`TURBOQUANT_KIND=rht`, the default) + a `turboquant_*` cache
dtype, the Impl **delegates to vLLM's stock `TurboQuantAttentionImpl`**
so we inherit its fused Triton store/decode kernels and matmul-based
WHT GEMM. That's what gets us speed parity with the upstream backend
on text-only models, and turns out to also preserve full GSM-8K
accuracy on Gemma 4 31B-it even without boundary protection (the RHT
rotation's strong inter-coordinate mixing is robust to all-layer
quantization).

Planar (`TURBOQUANT_KIND=planar`) is experimental on this plugin —
without boundary protection it collapses (0% on GSM-8K), because
2D Givens rotation only mixes adjacent pairs and the resulting
quantization error compounds badly through the boundary layers.
**Use `TURBOQUANT_KIND=rht` for any real workload.**

Selected via:
    LLM(model, kv_cache_dtype="turboquant_4bit_nc",
              attention_backend="TURBOQUANT")
    vllm serve <model> --kv-cache-dtype turboquant_4bit_nc \\
                       --attention-backend TURBOQUANT

KV cache layout (4-D, byte-indexed):
    (num_blocks, block_size, num_kv_heads, slot_size_padded)

`slot_size_padded` is `_slot_size_for(head_size, cache_dtype)` rounded up
to a power of 2 (see plugin.py for why: it lets the unify shim use the
integer-ratio path when sliding/full head_dims coexist).

Limitations / scope:
- ALiBi, encoder cross-attention, MLA are not supported.
- Planar is experimental and currently broken on GSM-8K without
  boundary protection — use RHT for production workloads.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, ClassVar, Optional

import torch
import torch.nn.functional as F

from fused_turboquant.core.quantizer import CompressedTensor, TurboQuantMSE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy vLLM imports
# ---------------------------------------------------------------------------

try:
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionType,
        MultipleOf,
    )

    HAS_V1 = True
except ImportError:
    HAS_V1 = False

    class AttentionBackend:  # type: ignore[no-redef]
        pass

    class AttentionImpl:  # type: ignore[no-redef]
        pass

    class AttentionType:  # type: ignore[no-redef]
        DECODER = "decoder"
        ENCODER = "encoder"

    class MultipleOf:  # type: ignore[no-redef]
        def __init__(self, base):
            self.base = base


# ---------------------------------------------------------------------------
# Env-var helpers — read inside __init__ so live env changes take effect.
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw == "1"


def _env_opt_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    return int(raw) if raw else None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class FusedTurboQuantV1Backend(AttentionBackend):
    """vLLM v1 attention backend backed by fused-turboquant kernels.

    Uses vLLM's built-in TurboQuant cache layout so we can drop into the
    existing TQFullAttentionSpec memory bookkeeping without monkey-patching
    the model executor.
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list] = [torch.float16, torch.bfloat16]
    # Only TurboQuant cache dtypes. We used to also list "auto"/"float16"/
    # "bfloat16" to support boundary-protection skip layers (which vLLM's
    # engine auto-forced to "auto"), but the plugin now disables boundary
    # protection (plugin._patch_disable_boundary_protection), so every
    # layer goes through our fast Triton path.
    supported_kv_cache_dtypes: ClassVar[list] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64, 128]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls():
        return FusedTurboQuantV1Impl

    @staticmethod
    def get_builder_cls():
        # vLLM's built-in TurboQuant metadata builder produces the same fields
        # vLLM's built-in TritonAttentionMetadataBuilder does (query_start_loc /
        # seq_lens / block_table / slot_mapping). We reuse it for both the
        # TurboQuant and raw fp16 forwards.
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantMetadataBuilder,
        )

        return TurboQuantMetadataBuilder

    @staticmethod
    def _slot_size_for(head_size: int, cache_dtype_str: str) -> int:
        """How many bytes each (token, head) slot occupies.

        Rounded up to the next power of 2 so that page sizes between layers
        with different head_sizes (e.g. Gemma 4 sliding head_dim=256 and full
        head_dim=512) are always in an integer ratio. That lets vLLM's
        `unify_kv_cache_spec_page_size` widen smaller layers by simply
        multiplying their block_size — the page_size_padded escape hatch is
        not needed. The actual per-slot data we write is bounded by the
        preset's `key_packed_size` and `value_packed_size`; everything past
        that point is unused zero-padding.
        """
        if cache_dtype_str is not None and cache_dtype_str.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            raw = TurboQuantConfig.from_cache_dtype(
                cache_dtype_str, head_size
            ).slot_size_aligned
        else:
            # fp16 / bf16 raw: 2 bytes per element × (K head_size + V head_size).
            raw = 4 * head_size
        if raw <= 1:
            return 1
        return 1 << (raw - 1).bit_length()

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple:
        """Per-layer cache shape.

        Returns `(num_blocks, block_size, num_kv_heads, slot_size)` where
        `slot_size = _slot_size_for(head_size, cache_dtype_str)`. The plugin
        rewrites every `Attention.get_kv_cache_spec` result into a
        `TQFullAttentionSpec(dtype=uint8, tq_slot_size=_slot_size_for(...))`,
        so the resulting `spec.page_size_bytes` and this shape's `.numel()`
        always agree.
        """
        slot = FusedTurboQuantV1Backend._slot_size_for(head_size, cache_dtype_str)
        return (num_blocks, block_size, num_kv_heads, slot)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Gemma 4 multimodal has prefix tokens that must full-attend.
        return True

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Accept any positive head_size — RHT only needs power-of-2 head_dims
        # but spec.head_size can be derived (effective_head_size). Validate
        # the real head_dim at Impl init time.
        return head_size > 0

    @classmethod
    def supports_attn_type(cls, attn_type) -> bool:
        # Decoder only. The fp16 fallback for encoder layers is gone now
        # that boundary protection is disabled.
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


# ---------------------------------------------------------------------------
# RHT fast path: subclass vLLM's stock TurboQuantAttentionImpl, override
# prefill to fall back to SDPA for head_size > 256 (flash-attn-2's limit,
# hit by Gemma 4's full attention layers). Everything else (store kernel,
# decode kernel, KV cache layout, NC) comes from the stock Impl, which
# is what gives us speed parity with the upstream backend.
# ---------------------------------------------------------------------------


def _make_gemma_compatible_tq_impl_cls():
    """Lazily build the subclass — vLLM imports may fail in environments
    without the stock TurboQuant backend; in that case the RHT fast path
    is unavailable and we'll fall back to the Planar slow path.
    """
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            triton_turboquant_decode_attention,
        )
    except ImportError:
        return None

    class _GemmaCompatibleTQImpl(TurboQuantAttentionImpl):
        """Stock TurboQuantAttentionImpl + SDPA prefill for head_size > 256.

        flash-attn-2's varlen path rejects `head_size > 256`. The stock
        Impl reaches for flash-attn unconditionally in `_prefill_attention`,
        so we override prefill to dispatch to an SDPA-based path for those
        layers (Gemma 4's full-attention layers have head_dim=512). The
        decode and store kernels are unchanged.
        """

        def _prefill_attention(
            self,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            Pi,
            centroids,
            PiT=None,
            layer=None,
        ):
            if self.head_size <= 256:
                return super()._prefill_attention(
                    query, key, value, kv_cache, attn_metadata, Pi, centroids, PiT, layer
                )
            # head_size > 256: SDPA per request; small chunked-prefill
            # continuations reuse the TQ decode kernel via a synthetic
            # block_table (same trick the stock Impl uses internally).
            N, Hq, D = query.shape
            Hk = key.shape[1]
            use_gqa = Hk < Hq
            num_reqs = attn_metadata.query_start_loc.shape[0] - 1
            out = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
            qsl = attn_metadata.query_start_loc.tolist()
            seq_lens_list = attn_metadata.seq_lens.tolist()
            cfg = self.tq_config
            for i in range(num_reqs):
                q_start, q_end = qsl[i], qsl[i + 1]
                q_len = q_end - q_start
                if q_len <= 0:
                    continue
                seq_len = seq_lens_list[i]
                q_seq = query[q_start:q_end]
                k_seq = key[q_start:q_end]
                v_seq = value[q_start:q_end]
                if q_len == seq_len:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    sub = F.scaled_dot_product_attention(
                        q_t, k_t, v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                else:
                    # Continuation: reuse decode kernel for each query token.
                    cached_len = seq_len - q_len
                    synth_seq_lens = torch.arange(
                        cached_len + 1,
                        seq_len + 1,
                        device=query.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    sub = triton_turboquant_decode_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        Pi=Pi,
                        centroids=centroids,
                        scale=self.scale,
                        mse_bits=cfg.key_mse_bits,
                        key_packed_size=cfg.key_packed_size,
                        value_quant_bits=cfg.effective_value_quant_bits,
                        key_fp8=cfg.key_fp8,
                        norm_correction=cfg.norm_correction,
                        PiT=PiT,
                    )
                out[q_start:q_end] = sub.to(query.dtype)
            return out

    return _GemmaCompatibleTQImpl


_GEMMA_COMPATIBLE_TQ_IMPL_CLS = _make_gemma_compatible_tq_impl_cls()


# ---------------------------------------------------------------------------
# Planar fast path: reuse the stock Triton kernels (`_tq_fused_store_mse`
# and `_tq_decode_stage1`) — they're rotation-agnostic. The store kernel
# accepts a pre-rotated `y` argument and bucketizes against Lloyd-Max
# midpoints; the decode kernel accepts a pre-rotated `q_rot` and does
# `score = vec_norms * (q_rot · c_vals) * scale`. So as long as we apply
# the SAME orthogonal rotation to both K (at store) and Q (at decode),
# the math is correct for any rotation choice — including Planar (2D
# Givens per pair) instead of Hadamard.
#
# We swap only the *external* rotation step in the launcher:
#   - RHT  : `y = (k / ||k||) @ PiT`  (cuBLAS GEMM)
#   - Planar: `y = planar_rotate(k / ||k||, rot2)`  (element-wise per pair)
# ---------------------------------------------------------------------------


def _make_gemma_compatible_planar_impl_cls():
    """Build the Planar fast-path Impl as a subclass of the stock TQ Impl.

    Overrides:
      - `_ensure_on_device`: build a Planar `rot2` table instead of a
        Hadamard matrix. Stored on the layer alongside the centroids /
        midpoints that the parent class sets up.
      - `_store_kv`: launch the stock store kernel after applying Planar
        rotation in PyTorch.
      - `_decode_attention`: launch the stock decode kernel after pre-
        rotating the query with Planar.
      - `_prefill_attention`: same SDPA-for-head>256 logic as the RHT
        compat class; continuation chunks reuse the Planar decode wrapper.
    """
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionImpl,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _tq_decode_stage1,
        )
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _use_fp8_e4b15,
        )
        from vllm.v1.attention.ops.triton_decode_attention import (
            _fwd_kernel_stage2,
        )
        from vllm.v1.attention.ops.triton_turboquant_store import (
            _tq_fused_store_mse,
        )
        from vllm.triton_utils import triton
        from fused_turboquant.core.planar import generate_planar_rot2
    except ImportError:
        return None

    PLANAR_SEED = 42  # deterministic across processes/TP ranks

    def _build_planar_matrix(rot2: torch.Tensor) -> torch.Tensor:
        """Materialize a (D, D) block-diagonal rotation matrix M such that
        `x @ M = planar_rotate(x, rot2)`. Lets us reuse cuBLAS GEMM for the
        rotation step, matching the speed of RHT's Hadamard matmul.

        For pair p with (cos θ, sin θ) = rot2[p], the 2×2 block of M placed
        at rows/cols [2p, 2p+1] is `[[c, s], [-s, c]]` — the transpose of
        the column-vector rotation `[[c, -s], [s, c]]`, so `x @ M` produces
        `[c*x0 - s*x1, s*x0 + c*x1]` (the same as planar_rotate).
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

    def planar_store(
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        planar_matrix: torch.Tensor,
        midpoints: torch.Tensor,
        mse_bits: int,
        key_packed_size: int,
        value_quant_bits: int,
    ) -> None:
        """Planar variant of `triton_turboquant_store` (MSE path only).
        Pre-rotates K via a precomputed (D, D) block-diagonal matrix;
        reuses the stock fused store kernel.
        """
        N, H, D = key.shape
        NH = N * H
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)
        mse_bytes = math.ceil(D * mse_bits / 8)
        n_centroids = 2**mse_bits
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
        stride_block = kv_cache.stride(0)
        stride_pos = kv_cache.stride(1)
        stride_head = kv_cache.stride(2)
        block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

        k_flat = key.float().reshape(NH, D)
        norms = k_flat.norm(dim=1, keepdim=True)
        x_hat = k_flat / (norms + 1e-8)
        y = (x_hat @ planar_matrix).contiguous()
        v_flat = value.float().reshape(NH, D)

        grid = (NH,)
        _tq_fused_store_mse[grid](
            y,
            norms.squeeze(1),
            v_flat,
            midpoints,
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            MSE_BITS=mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=block_grp,
            num_warps=4,
            num_stages=1,
        )

    def planar_decode_attention(
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        planar_matrix: torch.Tensor,
        centroids: torch.Tensor,
        scale: float,
        mse_bits: int,
        key_packed_size: int,
        value_quant_bits: int,
        norm_correction: bool = False,
        mid_o_buf=None,
        output_buf=None,
        lse_buf=None,
        buf_holder=None,
        max_num_kv_splits: int = 32,
    ) -> torch.Tensor:
        """Planar variant of `triton_turboquant_decode_attention`.
        Pre-rotates Q via the precomputed (D, D) Planar matrix; reuses the
        stock stage1/stage2 kernels.
        """
        B, Hq, D = query.shape
        Hk = kv_cache.shape[2]
        block_size = kv_cache.shape[1]
        kv_group_size = Hq // Hk
        device = query.device
        mse_bytes = math.ceil(D * mse_bits / 8)
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        BLOCK_D = triton.next_power_of_2(D)

        q_float = query.float()
        q_rot = (q_float @ planar_matrix).contiguous()

        NUM_KV_SPLITS = max_num_kv_splits
        if (
            mid_o_buf is not None
            and mid_o_buf.shape[0] >= B
            and mid_o_buf.shape[2] >= NUM_KV_SPLITS
        ):
            mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
        else:
            mid_o = torch.empty(
                B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device
            )
            if buf_holder is not None:
                buf_holder._planar_mid_o_buf = mid_o

        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        BLOCK_KV = 4
        grid = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            ATTN_SCALE=scale,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=1,
            num_stages=1,
        )

        if output_buf is not None and output_buf.shape[0] >= B:
            output = output_buf[:B, :Hq, :D]
        else:
            output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._planar_output_buf = output
        if lse_buf is not None and lse_buf.shape[0] >= B:
            lse = lse_buf[:B, :Hq]
        else:
            lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
            if buf_holder is not None:
                buf_holder._planar_lse_buf = lse

        grid2 = (B, Hq)
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            seq_lens,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            BLOCK_DV=BLOCK_D,
            Lv=D,
            num_warps=4,
            num_stages=2,
        )

        return output.to(query.dtype)

    class _GemmaCompatiblePlanarImpl(TurboQuantAttentionImpl):
        """Stock TurboQuantAttentionImpl with Planar rotation + head>256 SDPA."""

        def _ensure_on_device(self, layer, device):
            # Build deterministic Planar rotation table once per layer.
            if not hasattr(layer, "_planar_cached"):
                D = self.head_size
                rot2 = generate_planar_rot2(D, seed=PLANAR_SEED, device=device).to(
                    dtype=torch.float32
                )
                layer._planar_matrix = _build_planar_matrix(rot2)
                c = layer._tq_centroids.to(device=device, dtype=torch.float32)
                c_sorted, _ = c.sort()
                layer._planar_centroids = c_sorted
                layer._planar_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
                # Placeholders: parent `forward()` reads `_tq_Pi`, `_tq_PiT`
                # off the layer and forwards them as the `Pi` / `PiT` params
                # to `_decode_attention` and `_prefill_attention`. Our
                # Planar overrides ignore those params, so any tensor
                # works — we reuse the planar matrix to satisfy the access.
                layer._tq_Pi = layer._planar_matrix
                layer._tq_PiT = layer._planar_matrix
                layer._tq_midpoints = layer._planar_midpoints
                layer._planar_cached = True
                layer._tq_cached = True

        def _store_kv(self, key, value, kv_cache, slot_mapping, layer):
            planar_store(
                key,
                value,
                kv_cache,
                slot_mapping,
                layer._planar_matrix,
                layer._planar_midpoints,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
            )

        def _decode_attention(
            self,
            query,
            kv_cache,
            attn_metadata,
            Pi,
            centroids,
            PiT=None,
            layer=None,
        ):
            mid_o_buf = output_buf = lse_buf = None
            if layer is not None:
                mid_o_buf = getattr(layer, "_planar_mid_o_buf", None)
                output_buf = getattr(layer, "_planar_output_buf", None)
                lse_buf = getattr(layer, "_planar_lse_buf", None)
            return planar_decode_attention(
                query=query,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                planar_matrix=layer._planar_matrix,
                centroids=layer._planar_centroids,
                scale=self.scale,
                mse_bits=self.tq_config.key_mse_bits,
                key_packed_size=self.tq_config.key_packed_size,
                value_quant_bits=self.tq_config.effective_value_quant_bits,
                norm_correction=self.tq_config.norm_correction,
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                max_num_kv_splits=self.max_num_kv_splits,
            )

        def _prefill_attention(
            self,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            Pi,
            centroids,
            PiT=None,
            layer=None,
        ):
            # Fast path: identical to the parent's flash-attn varlen — it
            # operates on raw Q/K/V and doesn't touch the rotation. We just
            # gate on head_size <= 256 (flash-attn-2 limit). For head>256
            # or continuation chunks, fall through to per-sequence handling.
            from vllm.v1.attention.backends.turboquant_attn import _HAS_FLASH_ATTN

            if (
                self.head_size <= 256
                and _HAS_FLASH_ATTN
                and attn_metadata.max_query_len == attn_metadata.max_seq_len
            ):
                return self._flash_attn_varlen(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.max_query_len,
                )

            N, Hq, D = query.shape
            Hk = key.shape[1]
            use_gqa = Hk < Hq
            num_reqs = attn_metadata.query_start_loc.shape[0] - 1
            out = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
            qsl = attn_metadata.query_start_loc.tolist()
            seq_lens_list = attn_metadata.seq_lens.tolist()
            for i in range(num_reqs):
                q_start, q_end = qsl[i], qsl[i + 1]
                q_len = q_end - q_start
                if q_len <= 0:
                    continue
                seq_len = seq_lens_list[i]
                q_seq = query[q_start:q_end]
                k_seq = key[q_start:q_end]
                v_seq = value[q_start:q_end]
                if q_len == seq_len:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    sub = F.scaled_dot_product_attention(
                        q_t, k_t, v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                else:
                    cached_len = seq_len - q_len
                    synth_seq_lens = torch.arange(
                        cached_len + 1,
                        seq_len + 1,
                        device=query.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    sub = planar_decode_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        planar_matrix=layer._planar_matrix,
                        centroids=layer._planar_centroids,
                        scale=self.scale,
                        mse_bits=self.tq_config.key_mse_bits,
                        key_packed_size=self.tq_config.key_packed_size,
                        value_quant_bits=self.tq_config.effective_value_quant_bits,
                        norm_correction=self.tq_config.norm_correction,
                    )
                out[q_start:q_end] = sub.to(query.dtype)
            return out

    return _GemmaCompatiblePlanarImpl


_GEMMA_COMPATIBLE_PLANAR_IMPL_CLS = _make_gemma_compatible_planar_impl_cls()


# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------


class FusedTurboQuantV1Impl(AttentionImpl):
    """Two-way dispatcher: RHT delegate or Planar delegate.

    Picks one of:

    1. `kind=rht` (default): `_GemmaCompatibleTQImpl` — stock vLLM TQ Impl
       plus SDPA fallback when `head_size > 256` (Gemma 4 full attn).
    2. `kind=planar`: `_GemmaCompatiblePlanarImpl` — stock TQ Impl with the
       external rotation step replaced by a (D, D) block-diagonal Planar
       matmul.

    Both paths use the stock Triton store/decode kernels; the only
    difference is the rotation matrix.

    Boundary protection (vLLM normally pins first/last 2 layers to
    `kv_cache_dtype="auto"`) is disabled by the plugin so that every
    attention layer flows through one of these two fast paths — there is
    no raw fp16 SDPA fallback anymore.
    """

    accept_output_buffer: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type=None,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError(
                "FusedTurboQuantV1Impl does not support ALiBi attention"
            )
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "FusedTurboQuantV1Impl does not yet support KV sharing"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type or AttentionType.DECODER
        self._fast_impl = None  # set below for RHT+TQ path

        if not (
            isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
        ):
            raise ValueError(
                f"FusedTurboQuantV1Backend only supports turboquant_* cache "
                f"dtypes (got {kv_cache_dtype!r}). The plugin disables vLLM's "
                f"boundary-protection auto-skip so this should never fire — "
                f"if you see it, check `plugin._patch_disable_boundary_protection`."
            )

        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"FusedTurboQuantV1Impl only supports decoder attention, "
                f"got {self.attn_type}."
            )

        # Resolve preset → bit-widths via vLLM's TurboQuantConfig.
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        cfg = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.cfg = cfg

        # Allow env-var to override the quantizer kind ("rht" | "planar").
        self.kind = _env_str("TURBOQUANT_KIND", "rht")
        if self.kind not in ("rht", "planar"):
            raise ValueError(
                f"TURBOQUANT_KIND must be 'rht' or 'planar', got {self.kind!r}"
            )

        # Branch 1a: RHT + TQ → delegate to the stock Impl (fast path).
        if self.kind == "rht" and _GEMMA_COMPATIBLE_TQ_IMPL_CLS is not None:
            self._fast_impl = _GEMMA_COMPATIBLE_TQ_IMPL_CLS(
                num_heads,
                head_size,
                scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=alibi_slopes,
                sliding_window=sliding_window,
                kv_cache_dtype=kv_cache_dtype,
                logits_soft_cap=logits_soft_cap,
                attn_type=self.attn_type,
                kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            )
            logger.info(
                "FusedTurboQuantV1Impl init: RHT fast path (delegating to stock "
                "TurboQuantAttentionImpl). preset=%s head_size=%d num_heads=%d "
                "num_kv_heads=%d sliding_window=%s",
                kv_cache_dtype,
                head_size,
                num_heads,
                num_kv_heads,
                sliding_window,
            )
            return

        # Branch 1b: Planar + TQ → stock kernels with Planar rotation swap.
        if self.kind == "planar" and _GEMMA_COMPATIBLE_PLANAR_IMPL_CLS is not None:
            self._fast_impl = _GEMMA_COMPATIBLE_PLANAR_IMPL_CLS(
                num_heads,
                head_size,
                scale,
                num_kv_heads=num_kv_heads,
                alibi_slopes=alibi_slopes,
                sliding_window=sliding_window,
                kv_cache_dtype=kv_cache_dtype,
                logits_soft_cap=logits_soft_cap,
                attn_type=self.attn_type,
                kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            )
            logger.info(
                "FusedTurboQuantV1Impl init: Planar fast path (stock kernels + "
                "Planar rotation). preset=%s head_size=%d num_heads=%d "
                "num_kv_heads=%d sliding_window=%s",
                kv_cache_dtype,
                head_size,
                num_heads,
                num_kv_heads,
                sliding_window,
            )
            return

        # Branch 2 (legacy slow path): used only if the stock kernels are not
        # importable (unusual). Keeps the codebase functional in older vLLM
        # builds without `turboquant_attn`.
        if cfg.key_fp8:
            raise NotImplementedError(
                "fused-turboquant Planar / fallback kernels do not implement "
                "FP8 keys; use turboquant_4bit_nc / turboquant_3bit_nc / "
                "turboquant_k3v4_nc with TURBOQUANT_KIND=rht."
            )

        self.bits = cfg.key_quant_bits  # K bits
        self.v_bits = cfg.value_quant_bits  # V bits
        self.key_packed_size = cfg.key_packed_size  # K packed bytes + 2 byte fp16 norm
        self.value_packed_size = cfg.value_packed_size  # V packed bytes + 4 byte scale/zero
        self.slot_size = cfg.slot_size_aligned  # padded total

        if self.kind == "rht":
            QuantizerCls = TurboQuantMSE
        else:
            from fused_turboquant.core.planar import PlanarQuantMSE

            QuantizerCls = PlanarQuantMSE

        if self.bits not in (2, 3, 4):
            raise ValueError(
                f"Unsupported K bits={self.bits} from preset {kv_cache_dtype!r}. "
                f"fused-turboquant Planar path supports 2/3/4-bit only."
            )
        if self.v_bits not in (2, 3, 4):
            raise ValueError(
                f"Unsupported V bits={self.v_bits} from preset {kv_cache_dtype!r}. "
                f"fused-turboquant Planar path supports 2/3/4-bit only."
            )

        self.tq_k = QuantizerCls(head_dim=head_size, bits=self.bits, device="cuda")
        if self.v_bits == self.bits:
            self.tq_v = self.tq_k
        else:
            self.tq_v = QuantizerCls(head_dim=head_size, bits=self.v_bits, device="cuda")

        # Rotation state used to pre-rotate Q at decode time so the
        # compressed-K dot product is correct.
        if self.kind == "rht":
            self.rotation_state = self.tq_k.rotation.signs
        else:
            self.rotation_state = self.tq_k.rotation.rot2

        self.centroids_k = self.tq_k.quantizer.levels
        self.centroids_v = self.tq_v.quantizer.levels

        # Byte width of the packed indices ONLY (excluding norm).
        self.k_packed_bytes = math.ceil(head_size * self.bits / 8)
        self.v_packed_bytes = math.ceil(head_size * self.v_bits / 8)
        assert (
            self.k_packed_bytes + 2 <= self.key_packed_size
        ), f"K packed ({self.k_packed_bytes}+2) > preset K slot ({self.key_packed_size})"
        assert (
            self.v_packed_bytes + 2 <= self.value_packed_size
        ), f"V packed ({self.v_packed_bytes}+2) > preset V slot ({self.value_packed_size})"

        logger.info(
            "FusedTurboQuantV1Impl init: %s slow path. preset=%s K=%dbit V=%dbit "
            "head_size=%d num_heads=%d num_kv_heads=%d sliding_window=%s "
            "slot_size=%d",
            self.kind,
            kv_cache_dtype,
            self.bits,
            self.v_bits,
            head_size,
            num_heads,
            num_kv_heads,
            sliding_window,
            self.slot_size,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """GQA expand: [b, num_kv_heads, s, d] → [b, num_heads, s, d]."""
        if self.num_kv_groups == 1:
            return x
        b, h, s, d = x.shape
        x = x[:, :, None, :, :].expand(b, h, self.num_kv_groups, s, d)
        return x.reshape(b, h * self.num_kv_groups, s, d)

    def _store_kv(
        self,
        key: torch.Tensor,  # [n_tokens, num_kv_heads, head_size]
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, slot_size]
        slot_mapping: torch.Tensor,  # [n_tokens] flat slot index
    ) -> None:
        """Quantize K and V and write into the combined paged slot.

        Slot byte layout (per (block, position, head)):
            [0 ............................ key_packed_size)  K-half
                |- 0 .. k_packed_bytes)    K packed indices
                |- k_packed_bytes .. +2    K fp16 norm
                |- +2 .. key_packed_size   K padding
            [key_packed_size ............. slot_size)         V-half
                |- 0 .. v_packed_bytes     V packed indices
                |- v_packed_bytes .. +2    V fp16 norm
                |- +2 .. value_packed_size V padding (scale/zero in built-in TQ)
        """
        n_tokens = key.shape[0]
        if n_tokens == 0:
            return

        block_size = kv_cache.shape[1]

        k_comp = self.tq_k.encode(key.float())
        k_packed = k_comp.indices  # [n_tokens, num_kv_heads, k_packed_bytes]
        k_norms = k_comp.norms.to(torch.float16)  # [n_tokens, num_kv_heads] fp16

        v_comp = self.tq_v.encode(value.float())
        v_packed = v_comp.indices
        v_norms = v_comp.norms.to(torch.float16)

        slot_cpu = slot_mapping.tolist()
        for i, slot in enumerate(slot_cpu):
            if slot < 0:
                continue
            block_idx = slot // block_size
            offset = slot % block_size

            # Write K packed + K norm
            kv_cache[block_idx, offset, :, : self.k_packed_bytes] = k_packed[i]
            k_norm_b = (
                k_norms[i].contiguous().view(torch.uint8).reshape(self.num_kv_heads, 2)
            )
            kv_cache[
                block_idx,
                offset,
                :,
                self.k_packed_bytes : self.k_packed_bytes + 2,
            ] = k_norm_b

            # Write V packed + V norm into the V-half (starts at key_packed_size)
            v_off = self.key_packed_size
            kv_cache[
                block_idx, offset, :, v_off : v_off + self.v_packed_bytes
            ] = v_packed[i]
            v_norm_b = (
                v_norms[i].contiguous().view(torch.uint8).reshape(self.num_kv_heads, 2)
            )
            kv_cache[
                block_idx,
                offset,
                :,
                v_off + self.v_packed_bytes : v_off + self.v_packed_bytes + 2,
            ] = v_norm_b

    def _gather_half(
        self,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
        v_half: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read packed indices + fp16 norms for one sequence's K (v_half=False)
        or V (v_half=True). Returns (packed [1, num_kv_heads, seq_len, p_bytes],
        norms [1, num_kv_heads, seq_len] fp32)."""
        block_size = kv_cache.shape[1]
        n_blocks_used = (seq_len + block_size - 1) // block_size

        if v_half:
            slot_off = self.key_packed_size
            p_bytes = self.v_packed_bytes
        else:
            slot_off = 0
            p_bytes = self.k_packed_bytes

        packed_parts = []
        norm_parts = []
        for b in range(n_blocks_used):
            block_idx = int(block_table_row[b].item())
            tokens_here = min(block_size, seq_len - b * block_size)
            block = kv_cache[block_idx, :tokens_here]  # [tokens, num_kv_heads, slot]
            packed_parts.append(block[:, :, slot_off : slot_off + p_bytes])
            norm_bytes = block[
                :, :, slot_off + p_bytes : slot_off + p_bytes + 2
            ].contiguous()
            norm_parts.append(
                norm_bytes.view(torch.float16)
                .reshape(tokens_here, self.num_kv_heads)
                .float()
            )

        packed = torch.cat(packed_parts, dim=0)
        norms = torch.cat(norm_parts, dim=0)
        packed = packed.transpose(0, 1).unsqueeze(0).contiguous()
        norms = norms.transpose(0, 1).unsqueeze(0).contiguous()
        return packed, norms

    # ------------------------------------------------------------------
    # vLLM v1 hooks
    # ------------------------------------------------------------------

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Called by vLLM before forward to store new K/V into the cache."""
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        # Fast path: delegate the whole store call to the stock Impl, whose
        # internal `_store_kv` runs the fused store Triton kernel.
        if self._fast_impl is not None:
            self._fast_impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            return
        # Legacy slow path (stock TQ kernels not importable).
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._store_kv(k, v, kv_cache, slot_mapping)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Fast path: delegate to the stock Impl. The wrapping
        # `_GemmaCompatibleTQImpl` / `_GemmaCompatiblePlanarImpl` override
        # `_prefill_attention` to fall back to SDPA for head_size > 256
        # (Gemma 4 full-attention layers); everything else (decode kernel,
        # batched store/gather, NC) flows through unchanged.
        if self._fast_impl is not None:
            return self._fast_impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        num_tokens = query.shape[0]
        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)
        attn_out = torch.empty(
            N, self.num_heads, self.head_size, dtype=q.dtype, device=q.device
        )

        # Each sequence has its own (query_len, seq_len). Loop.
        query_start_loc = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        block_table = attn_metadata.block_table

        for i in range(len(seq_lens)):
            q_s = query_start_loc[i]
            q_e = query_start_loc[i + 1]
            q_len = q_e - q_s
            if q_len == 0:
                continue
            seq_len = seq_lens[i]
            context_len = seq_len - q_len

            q_i = q[q_s:q_e]
            k_i = key[q_s:q_e].view(q_len, self.num_kv_heads, self.head_size)
            v_i = value[q_s:q_e].view(q_len, self.num_kv_heads, self.head_size)

            if q_len > 1:
                if context_len > 0:
                    raise NotImplementedError(
                        "FusedTurboQuantV1Impl: chunked prefill (context_len>0 with "
                        "query_len>1) is not yet supported. Disable chunked prefill."
                    )
                attn_out[q_s:q_e] = self._prefill_one(q_i, k_i, v_i)
            else:
                attn_out[q_s] = self._decode_one(
                    q_i[0], kv_cache, block_table[i], seq_len
                )

        # Write back into vLLM's output buffer.
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output


    def _prefill_one(
        self,
        query: torch.Tensor,  # [q_len, num_heads, head_size]
        key: torch.Tensor,  # [q_len, num_kv_heads, head_size]
        value: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA on fp16 K/V for a single sequence with no past context.

        Returns [q_len, num_heads, head_size].
        """
        q = query.unsqueeze(0).transpose(1, 2)  # [1, h, s, d]
        k = key.unsqueeze(0).transpose(1, 2)
        v = value.unsqueeze(0).transpose(1, 2)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        attn = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )
        return attn.squeeze(0).transpose(0, 1).contiguous()

    def _decode_one(
        self,
        query: torch.Tensor,  # [num_heads, head_size]
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Decode: gather compressed K/V from cache, fused QK + softmax + matmul.

        Returns [num_heads, head_size].
        """
        # Gather K and pre-rotated dot-product
        k_packed, k_norms = self._gather_half(
            kv_cache, block_table_row, seq_len, v_half=False
        )

        if self.kind == "rht":
            from fused_turboquant.core.hadamard import randomized_hadamard
            from fused_turboquant.kernels.triton_attention import (
                fused_qk_scores_rht as qk_kernel,
            )

            q_rot = randomized_hadamard(query.float(), self.rotation_state)
        else:
            from fused_turboquant.core.planar import planar_rotate
            from fused_turboquant.kernels.triton_planar_attention import (
                fused_qk_scores_planar as qk_kernel,
            )

            q_rot = planar_rotate(query.float(), self.rotation_state)

        q_rot = q_rot.view(1, self.num_heads, 1, self.head_size)

        scores = qk_kernel(
            q_rot, k_packed, k_norms, self.centroids_k, self.scale, bits=self.bits
        )
        # scores: [1, num_heads, 1, seq_len]

        if self.sliding_window is not None and seq_len > self.sliding_window:
            window_start = seq_len - self.sliding_window
            scores[:, :, :, :window_start] = float("-inf")

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

        # Decode V
        v_packed, v_norms = self._gather_half(
            kv_cache, block_table_row, seq_len, v_half=True
        )
        ct = CompressedTensor(
            indices=v_packed,
            norms=v_norms,
            original_dim=self.head_size,
            bits=self.v_bits,
        )
        decoded_v = self.tq_v.decode(ct).to(query.dtype)
        decoded_v = self._repeat_kv(decoded_v)  # [1, num_heads, seq_len, head_size]

        attn = torch.matmul(weights, decoded_v)  # [1, num_heads, 1, head_size]
        return attn.squeeze(0).squeeze(1)
