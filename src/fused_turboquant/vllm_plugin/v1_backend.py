"""
fused-turboquant attention backend for vLLM v1 (vllm >= 0.20).

This backend overrides `AttentionBackendEnum.TURBOQUANT` and uses vLLM's
existing TurboQuant paged-cache layout end-to-end, so the framework's
`TQFullAttentionSpec` memory accounting "just works". Only the inner
encode / decode / attention-score computation is replaced with the
fused-turboquant Triton kernels (RHT or Planar rotation, mixed K/V
bit-widths, etc.).

Selected via:
    LLM(model, kv_cache_dtype="turboquant_4bit_nc",
              attention_backend="TURBOQUANT")
    vllm serve <model> --kv-cache-dtype turboquant_4bit_nc \\
                       --attention-backend TURBOQUANT

KV cache layout (matches vLLM's built-in TurboQuant exactly):
    (num_blocks, block_size, num_kv_heads, slot_size_aligned)

    slot bytes:
      [key_packed (key_packed_size B) | value_packed (value_packed_size B)]
      with key_packed_size and value_packed_size taken from the preset's
      TurboQuantConfig. We store the K norm in the trailing 2 B of the K
      half (fp16) and the V norm in the trailing 2 B of the V half (fp16);
      any remaining bytes in V (e.g. the built-in's scale+zero second pair)
      are zero padding for fused-turboquant since we use MSE-quantized V.

Limitations / scope:
- ALiBi, encoder, MLA, cross-attention paths are not supported.
- `compress_v=False` (raw fp16 V) is not yet supported on the v1 path —
  in v1 the slot bytes belong to the spec, so K-only mode would need a
  preset with `value_quant_bits` set to "raw fp16" which vLLM's built-in
  TurboQuantConfig does not currently offer.
- The fused-turboquant `TURBOQUANT_KIND=planar` and `TURBOQUANT_V_BITS`
  environment variables still apply, but the slot size budget is
  determined by the chosen preset (so V bits below the preset are wasted
  bytes; V bits above the preset will not fit and are clamped at init).
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
    # We accept the TurboQuant presets (text decoder layers) PLUS the standard
    # `auto` / fp16 / bf16 dtypes that vision-encoder layers ask for in
    # multimodal models like Gemma 4. Non-quantized layers fall through to a
    # plain SDPA path so the whole multimodal model can run on this single
    # backend.
    supported_kv_cache_dtypes: ClassVar[list] = [
        "auto",
        "float16",
        "bfloat16",
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
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple:
        """Per-layer cache shape.

        For TurboQuant presets we use vLLM's combined K+V slot layout (no
        leading 2 dim) so the framework's TQFullAttentionSpec memory accounting
        lines up. For `auto` / `float16` / `bfloat16` layers (e.g. vision
        encoder) we fall back to the canonical (num_blocks, 2, block_size,
        num_kv_heads, head_size) layout that every other backend produces.
        """
        if cache_dtype_str is not None and cache_dtype_str.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
            return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

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
        # Decoder (autoregressive) for text layers, encoder / encoder-only for
        # the vision tower's bidirectional attention. The encoder paths are
        # only exercised by the fp16 fallback.
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

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
# Impl
# ---------------------------------------------------------------------------


class FusedTurboQuantV1Impl(AttentionImpl):
    """Per-layer attention implementation using fused-turboquant kernels.

    Slot layout (matches vLLM's built-in TurboQuant byte map):

        [key_packed_size bytes][value_packed_size bytes (padded to slot_size_aligned)]

    Inside the K half, the last 2 bytes hold the fp16 norm for that vector
    (fused-turboquant's encode returns one fp16 norm per vector). Inside
    the V half, the trailing 2-4 bytes are reserved for the V norm /
    scale fields — we only use the first 2 bytes (fp16 norm).
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

        # Branch between the quantized TurboQuant path and the raw fp16 SDPA
        # fallback. The fallback is needed so that multimodal models like
        # Gemma 4 can run end-to-end on this single backend: text-decoder
        # layers stay quantized, vision-encoder layers (kv_cache_dtype="auto")
        # fall through to plain SDPA over the canonical (num_blocks, 2, ...)
        # cache layout.
        self._is_raw = not (
            isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
        )
        if self._is_raw:
            logger.info(
                "FusedTurboQuantV1Impl init: raw mode (kv_cache_dtype=%s) — "
                "no quantization, plain SDPA. head_size=%d num_heads=%d "
                "num_kv_heads=%d attn_type=%s sliding_window=%s",
                kv_cache_dtype,
                head_size,
                num_heads,
                num_kv_heads,
                self.attn_type,
                sliding_window,
            )
            return

        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"FusedTurboQuantV1Impl quantized path only supports decoder "
                f"attention, got {self.attn_type}. Use kv_cache_dtype='auto' "
                f"on this layer to fall through to the fp16 SDPA path."
            )

        # Resolve preset → bit-widths via vLLM's TurboQuantConfig.
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        cfg = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.cfg = cfg
        if cfg.key_fp8:
            raise NotImplementedError(
                "fused-turboquant kernels do not implement FP8 keys yet. "
                "Use turboquant_4bit_nc / turboquant_3bit_nc / turboquant_k3v4_nc."
            )

        self.bits = cfg.key_quant_bits  # K bits
        self.v_bits = cfg.value_quant_bits  # V bits
        self.key_packed_size = cfg.key_packed_size  # K packed bytes + 2 byte fp16 norm
        self.value_packed_size = cfg.value_packed_size  # V packed bytes + 4 byte scale/zero (we use 2)
        self.slot_size = cfg.slot_size_aligned  # padded total

        # Allow env-var to override the quantizer kind ("rht" | "planar").
        self.kind = _env_str("TURBOQUANT_KIND", "rht")
        if self.kind not in ("rht", "planar"):
            raise ValueError(
                f"TURBOQUANT_KIND must be 'rht' or 'planar', got {self.kind!r}"
            )

        if self.kind == "rht":
            QuantizerCls = TurboQuantMSE
        else:
            from fused_turboquant.core.planar import PlanarQuantMSE

            QuantizerCls = PlanarQuantMSE

        if self.bits not in (2, 3, 4):
            raise ValueError(
                f"Unsupported K bits={self.bits} from preset {kv_cache_dtype!r}. "
                f"fused-turboquant supports 2/3/4-bit only."
            )
        if self.v_bits not in (2, 3, 4):
            raise ValueError(
                f"Unsupported V bits={self.v_bits} from preset {kv_cache_dtype!r}. "
                f"fused-turboquant supports 2/3/4-bit only."
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
        # Sanity-check against the preset:
        assert (
            self.k_packed_bytes + 2 <= self.key_packed_size
        ), f"K packed ({self.k_packed_bytes}+2) > preset K slot ({self.key_packed_size})"
        assert (
            self.v_packed_bytes + 2 <= self.value_packed_size
        ), f"V packed ({self.v_packed_bytes}+2) > preset V slot ({self.value_packed_size})"

        logger.info(
            "FusedTurboQuantV1Impl init: preset=%s K=%dbit V=%dbit (kind=%s, "
            "head_size=%d, num_heads=%d, num_kv_heads=%d, sliding_window=%s, "
            "slot_size=%d)",
            kv_cache_dtype,
            self.bits,
            self.v_bits,
            self.kind,
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
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # Encoder attention has no persistent KV cache.
            return
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        if self._is_raw:
            self._store_raw_kv(k, v, kv_cache, slot_mapping)
        else:
            self._store_kv(k, v, kv_cache, slot_mapping)

    def _store_raw_kv(
        self,
        key: torch.Tensor,  # [n_tokens, num_kv_heads, head_size]
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # [num_blocks, 2, block_size, num_kv_heads, head_size]
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write raw fp16/bf16 K and V into the canonical paged layout."""
        n_tokens = key.shape[0]
        if n_tokens == 0:
            return
        block_size = kv_cache.shape[2]
        cache_dtype = kv_cache.dtype
        slot_cpu = slot_mapping.tolist()
        for i, slot in enumerate(slot_cpu):
            if slot < 0:
                continue
            block_idx = slot // block_size
            offset = slot % block_size
            kv_cache[block_idx, 0, offset] = key[i].to(cache_dtype)
            kv_cache[block_idx, 1, offset] = value[i].to(cache_dtype)

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

            if self._is_raw:
                attn_out[q_s:q_e] = self._forward_raw_seq(
                    q_i, k_i, v_i, kv_cache, block_table[i], seq_len, context_len
                )
            elif q_len > 1:
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

    # ------------------------------------------------------------------
    # raw fp16/bf16 SDPA fallback
    # ------------------------------------------------------------------

    def _forward_raw_seq(
        self,
        query: torch.Tensor,  # [q_len, num_heads, head_size]
        key: torch.Tensor,  # [q_len, num_kv_heads, head_size] (new tokens this step)
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # [num_blocks, 2, block_size, num_kv_heads, head_size]
        block_table_row: torch.Tensor,
        seq_len: int,
        context_len: int,
    ) -> torch.Tensor:
        """Plain SDPA on raw fp16/bf16 K/V for one sequence.

        Handles three sub-cases:
          - encoder / encoder-only: bidirectional attention on this step's
            K/V alone (no cache, no past context).
          - decoder, query_len > 1, no context: causal attention on new K/V.
          - decoder, query_len == 1: gather full cached K/V then attend.
        Chunked prefill (decoder, query_len > 1 with context_len > 0) raises
        NotImplementedError — same restriction as the TQ path.
        """
        q_len = query.shape[0]
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._sdpa_local(query, key, value, is_causal=False)

        if q_len > 1 and context_len == 0:
            return self._sdpa_local(query, key, value, is_causal=True)
        if q_len > 1 and context_len > 0:
            raise NotImplementedError(
                "FusedTurboQuantV1Impl raw fp16 fallback: chunked prefill is "
                "not supported. Disable chunked prefill."
            )

        # query_len == 1 (decoder decode step). Gather the full cached K/V
        # for this sequence and attend.
        cached_k, cached_v = self._gather_raw_kv(kv_cache, block_table_row, seq_len)
        return self._sdpa_with_cached(query, cached_k, cached_v, seq_len)

    def _sdpa_local(
        self,
        q: torch.Tensor,  # [q_len, num_heads, head_size]
        k: torch.Tensor,  # [q_len, num_kv_heads, head_size]
        v: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        """SDPA on the new K/V only (no cache). Returns [q_len, num_heads, head_size]."""
        qt = q.unsqueeze(0).transpose(1, 2)  # [1, h, s, d]
        kt = k.unsqueeze(0).transpose(1, 2)
        vt = v.unsqueeze(0).transpose(1, 2)
        kt = self._repeat_kv(kt)
        vt = self._repeat_kv(vt)
        out = F.scaled_dot_product_attention(
            qt, kt, vt, scale=self.scale, is_causal=is_causal
        )
        return out.squeeze(0).transpose(0, 1).contiguous()

    def _gather_raw_kv(
        self,
        kv_cache: torch.Tensor,  # [num_blocks, 2, block_size, num_kv_heads, head_size]
        block_table_row: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect the full cached K and V for one sequence.

        Returns:
            K, V each shaped [1, num_kv_heads, seq_len, head_size]
        """
        block_size = kv_cache.shape[2]
        n_blocks = (seq_len + block_size - 1) // block_size

        k_parts, v_parts = [], []
        for b in range(n_blocks):
            block_idx = int(block_table_row[b].item())
            tokens_here = min(block_size, seq_len - b * block_size)
            k_parts.append(kv_cache[block_idx, 0, :tokens_here])
            v_parts.append(kv_cache[block_idx, 1, :tokens_here])
        k = torch.cat(k_parts, dim=0)  # [seq_len, num_kv_heads, head_size]
        v = torch.cat(v_parts, dim=0)
        k = k.transpose(0, 1).unsqueeze(0).contiguous()
        v = v.transpose(0, 1).unsqueeze(0).contiguous()
        return k, v

    def _sdpa_with_cached(
        self,
        q: torch.Tensor,  # [1, num_heads, head_size]
        k: torch.Tensor,  # [1, num_kv_heads, seq_len, head_size]
        v: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Decode-step SDPA over the full cached K/V."""
        qt = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, 1, head_size]
        kt = self._repeat_kv(k)
        vt = self._repeat_kv(v)
        # We are at the most recent position; a causal mask of width 1 is a
        # no-op, but SDPA expects either a mask or is_causal=False here.
        out = F.scaled_dot_product_attention(
            qt, kt, vt, scale=self.scale, is_causal=False
        )
        return out.squeeze(0).squeeze(1)  # [num_heads, head_size]

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
