"""
fused-turboquant attention backend for vLLM v1 (vllm >= 0.20).

This backend overrides `AttentionBackendEnum.TURBOQUANT` to:
  - Add compatibility for cases vLLM's stock TurboQuant rejects
    (multimodal mm_prefix; head_size > 256 in flash-attn-2's varlen path).
  - Disable vLLM's boundary-protection auto-skip
    (`plugin._patch_disable_boundary_protection`), so every layer
    flows through the fast Triton path.
  - Plug in a pluggable rotation kind via the `RotationStrategy`
    abstraction (`./rotation/`). RHT and Planar ship in-tree; future
    kinds (Rotorquant, etc.) drop in as new strategy subclasses
    without touching this file.

What we use from vLLM (the only remaining coupling):
  - `vllm.v1.attention.ops.triton_turboquant_store._tq_fused_store_mse`
    — fused bucketize + pack + value-quant store kernel
  - `vllm.v1.attention.ops.triton_turboquant_decode._tq_decode_stage1`
    — split-KV scoring + value accumulation kernel
  - `vllm.v1.attention.ops.triton_decode_attention._fwd_kernel_stage2`
    — log-sum-exp reduction across KV splits
  - `vllm.v1.attention.backends.fa_utils.flash_attn_varlen_func` and
    `get_flash_attn_version` for the prefill fast path
  - `TurboQuantMetadataBuilder` (passed unchanged to vLLM through
    `get_builder_cls`)

Everything else (rotation, layer attribute names, Impl class
structure) is owned by this package.

Selected via:
    LLM(model, kv_cache_dtype="turboquant_4bit_nc",
              attention_backend="TURBOQUANT")
    vllm serve <model> --kv-cache-dtype turboquant_4bit_nc \\
                       --attention-backend TURBOQUANT

KV cache layout (4-D, byte-indexed):
    (num_blocks, block_size, num_kv_heads, slot_size_padded)

`slot_size_padded` is `_slot_size_for(head_size, cache_dtype)` rounded
up to a power of 2 so cross-layer page sizes are in an integer ratio
(plugin.py handles the spec rewriting).

Limitations:
- ALiBi, encoder cross-attention, MLA are not supported.
- `turboquant_k8v4` (FP8 keys) is not yet supported in this refactor —
  use `turboquant_4bit_nc` / `turboquant_3bit_nc` / `turboquant_k3v4_nc`.
- Planar (`TURBOQUANT_KIND=planar`) is experimental; without boundary
  protection it collapses to 0% accuracy on GSM-8K. Use RHT in
  production.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, ClassVar, Optional

import torch
import torch.nn.functional as F

from fused_turboquant.vllm_plugin.rotation import (
    RotationStrategy,
    get_rotation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy vLLM imports (kept tight — only kernel ops + the AttentionImpl base).
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
# Env helpers
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


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
    # Only TurboQuant cache dtypes — the boundary-protection raw fp16
    # path is disabled at the engine level by the plugin.
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
        # Stock metadata builder — produces the (query_start_loc / seq_lens
        # / block_table / slot_mapping) fields that our forward consumes.
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantMetadataBuilder,
        )

        return TurboQuantMetadataBuilder

    @staticmethod
    def _slot_size_for(head_size: int, cache_dtype_str: str) -> int:
        """How many bytes each (token, head) slot occupies, rounded up
        to the next power of 2 so cross-layer page sizes stay in an
        integer ratio. See plugin.py's spec rewriting.
        """
        if cache_dtype_str is not None and cache_dtype_str.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            raw = TurboQuantConfig.from_cache_dtype(
                cache_dtype_str, head_size
            ).slot_size_aligned
        else:
            raw = 4 * head_size  # K bytes + V bytes (fp16)
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
        slot = FusedTurboQuantV1Backend._slot_size_for(head_size, cache_dtype_str)
        return (num_blocks, block_size, num_kv_heads, slot)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Gemma 4 multimodal has prefix tokens that need bidirectional
        # attention; vLLM marks the model as mm_prefix_lm regardless of
        # whether we pass images, so accept this flag.
        return True

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0

    @classmethod
    def supports_attn_type(cls, attn_type) -> bool:
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
# Impl — single class for all rotation kinds
# ---------------------------------------------------------------------------


# Threshold below which continuation prefill chunks reuse the decode
# kernel via synthetic per-query seq_lens (matches stock behavior).
_CONTINUATION_DECODE_THRESHOLD = 128


class FusedTurboQuantV1Impl(AttentionImpl):
    """Per-layer attention impl, parameterized by a `RotationStrategy`.

    The strategy provides:
      - `setup_layer(layer, head_size, centroids, device)` — cache
        rotation state on the layer (called once per layer)
      - `rotate_for_store(x_normalized, layer)` — applied to unit-norm K
        before the store kernel's bucketize step
      - `rotate_for_decode(q, layer)` — applied to Q before the decode
        kernel's score computation

    Adding a new rotation kind (e.g. Rotorquant) means subclassing
    `RotationStrategy` and registering it — no changes to this file.
    """

    accept_output_buffer: bool = True
    supports_quant_query_input: bool = False

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
            raise NotImplementedError("ALiBi attention is not supported")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not yet supported")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type or AttentionType.DECODER

        if not (
            isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
        ):
            raise ValueError(
                f"FusedTurboQuantV1Impl only supports turboquant_* cache dtypes "
                f"(got {kv_cache_dtype!r}). Boundary protection is disabled at "
                f"the plugin level so every layer should arrive here as a TQ "
                f"layer — if you see this, check "
                f"`plugin._patch_disable_boundary_protection`."
            )
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"FusedTurboQuantV1Impl only supports DECODER attention "
                f"(got {self.attn_type})."
            )

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        if self.tq_config.key_fp8:
            raise NotImplementedError(
                "FP8 keys (turboquant_k8v4) are not yet supported by this "
                "refactor — use turboquant_4bit_nc / turboquant_3bit_nc / "
                "turboquant_k3v4_nc."
            )

        kind = _env_str("TURBOQUANT_KIND", "rht")
        try:
            self.rotation: RotationStrategy = get_rotation(kind)
        except ValueError as e:
            raise ValueError(
                f"TURBOQUANT_KIND={kind!r} not registered. Pick one of the "
                f"built-ins (rht, planar) or register a custom strategy "
                f"via fused_turboquant.vllm_plugin.rotation.register_rotation."
            ) from e

        # Precomputed kernel constants
        cfg = self.tq_config
        self._mse_bytes = math.ceil(head_size * cfg.key_mse_bits / 8)
        self._val_data_bytes = math.ceil(
            head_size * cfg.effective_value_quant_bits / 8
        )
        self._n_centroids = 2**cfg.key_mse_bits

        # Flash-attn capability for prefill
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        self.fa_version = get_flash_attn_version(head_size=head_size)

        # NUM_KV_SPLITS for the decode kernel (stock uses this via config)
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

        logger.info(
            "FusedTurboQuantV1Impl init: rotation=%s preset=%s K=%dbit V=%dbit "
            "head_size=%d num_heads=%d num_kv_heads=%d sliding_window=%s",
            self.rotation.name,
            kv_cache_dtype,
            cfg.key_quant_bits,
            cfg.value_quant_bits,
            head_size,
            num_heads,
            num_kv_heads,
            sliding_window,
        )

    # ------------------------------------------------------------------
    # Per-layer setup
    # ------------------------------------------------------------------

    def _ensure_setup(self, layer, device) -> None:
        """Ask the rotation strategy to materialize its state on the
        layer the first time we see it. `layer._tq_centroids` is the
        Lloyd-Max level table that vLLM's `Attention._init_turboquant_buffers`
        already attached.
        """
        self.rotation.setup_layer(layer, self.head_size, layer._tq_centroids, device)

    # ------------------------------------------------------------------
    # Store: rotate K via strategy, launch stock fused store kernel
    # ------------------------------------------------------------------

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        self._ensure_setup(layer, key.device)
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        self._launch_store(k, v, kv_cache, slot_mapping, layer)

    def _launch_store(
        self,
        key: torch.Tensor,  # [N, H, D]
        value: torch.Tensor,  # [N, H, D]
        kv_cache: torch.Tensor,  # [num_blocks, block_size, H, slot] uint8
        slot_mapping: torch.Tensor,  # [N]
        layer,
    ) -> None:
        from vllm.triton_utils import triton
        from vllm.v1.attention.ops.triton_turboquant_store import (
            _tq_fused_store_mse,
        )

        N, H, D = key.shape
        NH = N * H
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_VAL = triton.next_power_of_2(self._val_data_bytes)
        block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1

        # Normalize K, then rotate via strategy.
        k_flat = key.float().reshape(NH, D)
        norms = k_flat.norm(dim=1, keepdim=True)
        x_hat = k_flat / (norms + 1e-8)
        y = self.rotation.rotate_for_store(x_hat, layer)
        v_flat = value.float().reshape(NH, D)

        grid = (NH,)
        _tq_fused_store_mse[grid](
            y,
            norms.squeeze(1),
            v_flat,
            self.rotation.get_midpoints(layer),
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=kv_cache.stride(0),
            stride_cache_pos=kv_cache.stride(1),
            stride_cache_head=kv_cache.stride(2),
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            MSE_BITS=self.tq_config.key_mse_bits,
            N_CENTROIDS=self._n_centroids,
            BLOCK_GRP=block_grp,
            num_warps=4,
            num_stages=1,
        )

    # ------------------------------------------------------------------
    # Forward: dispatch to prefill / decode
    # ------------------------------------------------------------------

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

        self._ensure_setup(layer, query.device)

        q = query[:N].view(N, self.num_heads, self.head_size)
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch
            attn_out = self._decode_attention(q, kv_cache, attn_metadata, layer)
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(q, k, v, kv_cache, attn_metadata, layer)
        else:
            # Mixed: decodes come first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=q.device, dtype=q.dtype
            )
            decode_meta = _split_meta(attn_metadata, 0, num_decodes, num_decode_tokens)
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, layer
            )
            prefill_meta = _split_meta(
                attn_metadata, num_decodes, None, num_decode_tokens, is_prefill=True
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                layer,
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------
    # Decode: rotate Q, run stock stage1 + stage2 kernels
    # ------------------------------------------------------------------

    def _decode_attention(
        self,
        query: torch.Tensor,  # [B, Hq, D]
        kv_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> torch.Tensor:
        from vllm.triton_utils import triton
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _tq_decode_stage1,
            _use_fp8_e4b15,
        )
        from vllm.v1.attention.ops.triton_decode_attention import (
            _fwd_kernel_stage2,
        )

        B, Hq, D = query.shape
        Hk = kv_cache.shape[2]
        block_size = kv_cache.shape[1]
        kv_group_size = Hq // Hk
        device = query.device

        q_rot = self.rotation.rotate_for_decode(query.float(), layer)

        BLOCK_D = triton.next_power_of_2(D)
        NUM_KV_SPLITS = self.max_num_kv_splits

        # Reuse per-layer scratch buffers (lazy alloc).
        mid_o = self._get_or_alloc_buf(
            layer, "_fused_mid_o", (B, Hq, NUM_KV_SPLITS, D + 1),
            dtype=torch.float32, device=device,
        )
        output = self._get_or_alloc_buf(
            layer, "_fused_output", (B, Hq, D),
            dtype=torch.float32, device=device,
        )
        lse = self._get_or_alloc_buf(
            layer, "_fused_lse", (B, Hq), dtype=torch.float32, device=device
        )

        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        BLOCK_KV = 4
        grid1 = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1[grid1](
            q_rot,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self.rotation.get_centroids(layer),
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            attn_metadata.block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=self.tq_config.key_mse_bits,
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            ATTN_SCALE=self.scale,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=0,  # FP8 not supported in this refactor
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=1,
            num_stages=1,
        )

        grid2 = (B, Hq)
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            attn_metadata.seq_lens,
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

    # ------------------------------------------------------------------
    # Prefill: flash-attn fast path or per-sequence loop
    # ------------------------------------------------------------------

    def _prefill_attention(
        self,
        query: torch.Tensor,  # [N, Hq, D]
        key: torch.Tensor,  # [N, Hk, D]
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> torch.Tensor:
        from vllm.v1.attention.backends.fa_utils import (
            is_flash_attn_varlen_func_available,
        )

        _has_fa = is_flash_attn_varlen_func_available()
        # Fast path: flash-attn varlen over the whole batch when every
        # request is first-chunk (max_query_len == max_seq_len) and
        # head_size is in flash-attn's supported range.
        if (
            _has_fa
            and self.head_size <= 256
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

        # Per-sequence loop: first-chunk → SDPA / flash-attn, continuation
        # → decode kernel (reading already-stored K from the cache).
        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        num_reqs = attn_metadata.query_start_loc.shape[0] - 1
        out = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        qsl = attn_metadata.query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()

        _cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]

            if q_len == seq_len:
                # First-chunk prefill — pure local attention.
                if _has_fa and self.head_size <= 256:
                    _cu_2[1] = q_len
                    sub = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=_cu_2,
                        cu_seqlens_k=_cu_2,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    sub = F.scaled_dot_product_attention(
                        q_t, k_t, v_t,
                        is_causal=True, scale=self.scale, enable_gqa=use_gqa,
                    ).transpose(0, 1)
            else:
                # Continuation chunk — past tokens already stored in the
                # TQ cache. Reuse the decode kernel via synthetic
                # per-query seq_lens for causal masking.
                cached_len = seq_len - q_len
                if q_len > _CONTINUATION_DECODE_THRESHOLD:
                    raise NotImplementedError(
                        f"Continuation prefill with q_len={q_len} > "
                        f"{_CONTINUATION_DECODE_THRESHOLD} not yet supported "
                        f"in the refactored backend."
                    )
                synth_seq_lens = torch.arange(
                    cached_len + 1,
                    seq_len + 1,
                    device=query.device,
                    dtype=attn_metadata.seq_lens.dtype,
                )
                synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                # Build a tiny ad-hoc metadata for _decode_attention.
                fake_meta = _FakeMetadata(
                    block_table=synth_bt,
                    seq_lens=synth_seq_lens,
                    max_seq_len=int(seq_len),
                    num_actual_tokens=q_len,
                )
                sub = self._decode_attention(q_seq, kv_cache, fake_meta, layer)
            out[q_start:q_end] = sub.to(query.dtype)
        return out

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        return flash_attn_varlen_func(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_or_alloc_buf(layer, attr, shape, dtype, device):
        """Lazily allocate a per-layer scratch buffer; grow if smaller."""
        buf = getattr(layer, attr, None)
        if (
            buf is None
            or buf.dtype != dtype
            or buf.device != torch.device(device)
            or any(buf.shape[i] < shape[i] for i in range(len(shape)))
        ):
            buf = torch.empty(shape, dtype=dtype, device=device)
            setattr(layer, attr, buf)
            return buf
        # Slice to the requested shape so callers see a fresh view each call.
        slices = tuple(slice(0, s) for s in shape)
        return buf[slices]


# ---------------------------------------------------------------------------
# Mixed-batch metadata helpers
# ---------------------------------------------------------------------------


class _FakeMetadata:
    """Minimal stand-in for `TurboQuantMetadata` when reusing the decode
    kernel inside the prefill continuation-chunk path. The kernel only
    reads `block_table`, `seq_lens`, and (via our forward()) doesn't
    need the rest."""

    def __init__(self, block_table, seq_lens, max_seq_len, num_actual_tokens):
        self.block_table = block_table
        self.seq_lens = seq_lens
        self.max_seq_len = max_seq_len
        self.num_actual_tokens = num_actual_tokens
        self.is_prefill = False
        self.num_decodes = num_actual_tokens
        self.num_decode_tokens = num_actual_tokens


def _split_meta(meta, start_req, end_req, decode_tokens_offset, is_prefill=False):
    """Slice a TurboQuantMetadata for the decode-or-prefill half of a
    mixed batch."""
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantMetadata

    if end_req is None:
        # prefill half: everything after `start_req`
        prefill_seq_lens = meta.seq_lens[start_req:]
        prefill_max_seq = max(prefill_seq_lens.tolist())
        return TurboQuantMetadata(
            seq_lens=prefill_seq_lens,
            slot_mapping=meta.slot_mapping[decode_tokens_offset:],
            block_table=meta.block_table[start_req:],
            query_start_loc=meta.query_start_loc[start_req:] - decode_tokens_offset,
            num_actual_tokens=int(meta.num_actual_tokens - decode_tokens_offset),
            max_query_len=meta.max_query_len,
            max_seq_len=prefill_max_seq,
            is_prefill=True,
        )
    # decode half
    return TurboQuantMetadata(
        seq_lens=meta.seq_lens[:end_req],
        slot_mapping=meta.slot_mapping[:decode_tokens_offset],
        block_table=meta.block_table[:end_req],
        query_start_loc=meta.query_start_loc[: end_req + 1],
        num_actual_tokens=decode_tokens_offset,
        max_query_len=1,
        max_seq_len=meta.max_seq_len,
        is_prefill=False,
    )
