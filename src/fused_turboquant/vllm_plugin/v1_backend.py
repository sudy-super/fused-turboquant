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
    # TurboQuant presets PLUS raw fp16 / bf16 for boundary-protection
    # skip layers (default behavior: vLLM forces first/last 2 layers to
    # `kv_cache_dtype="auto"` for accuracy). Those layers run through
    # our raw fp16 SDPA fallback. Toggle with
    # `TURBOQUANT_BOUNDARY_PROTECT` env var.
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
        # Stock metadata builder — produces the (query_start_loc / seq_lens
        # / block_table / slot_mapping) fields that our forward consumes.
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantMetadataBuilder,
        )

        return TurboQuantMetadataBuilder

    @staticmethod
    def _slot_size_for(head_size: int, cache_dtype_str: str) -> int:
        """Last-dim element count for `get_kv_cache_shape`, rounded up
        to the next power of 2.

        The returned value is interpreted as the count of elements of
        the spec's dtype (uint8 for TQ layers, bf16 for boundary-skip
        layers). For both interpretations to fit in the same allocation
        when boundary protection is enabled:

          - TQ spec (dtype=uint8): need slot >= TQ slot bytes
            (`slot_size_aligned`).
          - Auto spec (dtype=bf16): need slot >= 2 * head_size
            (i.e. `4*head_size` bytes / 2 bytes-per-element).

        When boundary protection is disabled, only the TQ requirement
        applies and the slot can be smaller (saving memory).
        """
        from fused_turboquant.vllm_plugin.plugin import _boundary_protect_enabled

        raw_fp16_elems = 2 * head_size  # bf16 element count for K+V
        if cache_dtype_str is not None and cache_dtype_str.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )

            tq_raw = TurboQuantConfig.from_cache_dtype(
                cache_dtype_str, head_size
            ).slot_size_aligned
            raw = (
                max(tq_raw, raw_fp16_elems)
                if _boundary_protect_enabled()
                else tq_raw
            )
        else:
            raw = raw_fp16_elems
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
        # DECODER is the main case (all text decoder layers including
        # boundary-skip raw fp16 layers). ENCODER / ENCODER_ONLY are
        # accepted defensively for vision-tower attention if any model
        # routes it through vLLM's Attention class (Gemma 4 doesn't).
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

        # Deferred FP16 K-cache for prefix attention (rotorquant-style).
        # When enabled:
        #   - Prefill writes K/V to a per-layer FP16 side buffer (skips the
        #     rotation+quantize store kernel entirely).
        #   - Decode attention runs two phases:
        #       Phase 1: fp16 SDPA over the buffered prefix (fast, no dequant)
        #       Phase 2: stock decode kernel over the quantized decode region
        #     merged via log-sum-exp.
        #   - New decode tokens are quantized into the paged cache as usual.
        # Net: attention on the long prefix avoids centroid-lookup overhead.
        self.defer_prefill = (
            os.environ.get("TURBOQUANT_DEFER_PREFILL", "0") == "1"
        )

        # Boundary-protection-skip layers arrive here with kv_cache_dtype
        # == "auto" / "float16" / "bfloat16". Historically those went
        # through the raw FP16 SDPA / flash-attn path (`_is_raw=True`).
        # Now `TURBOQUANT_BOUNDARY_PROTECT` can also be a bit count
        # (2/3/4/8): in that case we still use the FP16-sized slot
        # vLLM allocated for the boundary layer, but we route through
        # the TQ pipeline at the requested bit width, leaving the
        # unused slot bytes inert.
        from fused_turboquant.vllm_plugin.plugin import _boundary_protect_mode
        bp_mode = _boundary_protect_mode()
        is_boundary_layer = not (
            isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_")
        )
        # bp_mode possible values:
        #   "off"  — no protection (boundary layers are regular TQ)
        #   "fp16" — boundary stored as raw FP16 (`_is_raw=True`)
        #   "fp8"  — boundary stored as FP8 keys (8-bit, no LUT)
        #   int N  — boundary stored as N-bit MSE keys (Lloyd-Max LUT)
        is_raw_mode = bp_mode == "fp16"
        is_fp8_mode = bp_mode == "fp8"
        is_bits_mode = isinstance(bp_mode, int)
        self._is_raw = is_boundary_layer and is_raw_mode
        if is_boundary_layer and is_fp8_mode:
            self._boundary_tq_bits = 8
            self._boundary_tq_use_fp8 = True
        elif is_boundary_layer and is_bits_mode:
            self._boundary_tq_bits = bp_mode
            self._boundary_tq_use_fp8 = False
        else:
            self._boundary_tq_bits = None
            self._boundary_tq_use_fp8 = False

        if self._is_raw:
            # We still need fa_version for the vectorized _forward_raw
            # path that calls flash_attn_varlen_func with a paged
            # block_table.
            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
            self.fa_version = get_flash_attn_version(head_size=head_size)
            # vLLM's plugin forces flash_attn_version=2 for the TURBOQUANT
            # backend, but `_forward_raw` (when re-enabled via
            # TQ_BOUNDARY_FA_PAGED=1) doesn't actually use any of the
            # FA-version-restricted features. On Blackwell (SM ≥ 10) we
            # can prefer FA4 if it's available.
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(0)[0] >= 10
            ):
                try:
                    from vllm.vllm_flash_attn.flash_attn_interface import (
                        is_fa_version_supported,
                    )
                    if is_fa_version_supported(4):
                        self.fa_version = 4
                except ImportError:
                    pass
            logger.info(
                "FusedTurboQuantV1Impl init: raw fp16 fallback (kv_cache_dtype=%s, "
                "boundary-protection layer). head_size=%d num_heads=%d "
                "num_kv_heads=%d attn_type=%s sliding_window=%s",
                kv_cache_dtype, head_size, num_heads, num_kv_heads,
                self.attn_type, sliding_window,
            )
            return

        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"FusedTurboQuantV1Impl TQ path only supports DECODER attention "
                f"(got {self.attn_type})."
            )

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        if self._boundary_tq_bits is not None:
            # Boundary layer with TQ override. We sit in a FP16-sized
            # slot but speak TQ at the override bit width. Use our own
            # config wrapper so the 8-bit case can be either FP8 OR
            # MSE@8bit (the vLLM stock config hard-codes 8 → FP8).
            from fused_turboquant.vllm_plugin.boundary_tq_config import (
                BoundaryTurboQuantConfig,
            )
            bits = self._boundary_tq_bits
            self.tq_config = BoundaryTurboQuantConfig(
                head_dim=head_size,
                key_quant_bits=bits,
                value_quant_bits=bits,
                norm_correction=True,
                key_use_fp8_at_8bit=self._boundary_tq_use_fp8,
            )
        else:
            self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        # FP8 keys: still no specialized store kernel in our refactor.
        # Allowed only as a boundary override (handled via the dispatch
        # path), and even then only when the user explicitly opted in
        # via TURBOQUANT_BOUNDARY_PROTECT=fp8.
        if self.tq_config.key_fp8 and self._boundary_tq_bits is None:
            raise NotImplementedError(
                "FP8 keys (turboquant_k8v4) as the *primary* preset are "
                "not yet supported by this refactor — use "
                "turboquant_4bit_nc / turboquant_3bit_nc / "
                "turboquant_k3v4_nc, or set "
                "TURBOQUANT_BOUNDARY_PROTECT=fp8 to enable FP8 only on "
                "boundary layers."
            )

        # Independent K / V bit-width override via env vars. vLLM's
        # CacheDType Literal only ships 3 quantized presets ({K=4,V=4},
        # {K=3,V=4}, {K=3,V=3}), so to expose all 9 of K, V ∈ {2, 3, 4}
        # without modifying vLLM we treat the preset as the *slot
        # allocation* and let TURBOQUANT_KEY_BITS / TURBOQUANT_VALUE_BITS
        # override the *effective* quantization width. Override bits
        # must be ≤ the preset's bits — pick the smallest preset whose
        # K, V slots fit the desired widths (`turboquant_4bit_nc` is a
        # safe maximal choice).
        key_override = os.environ.get("TURBOQUANT_KEY_BITS")
        val_override = os.environ.get("TURBOQUANT_VALUE_BITS")
        if key_override is not None or val_override is not None:
            preset_k = self.tq_config.key_quant_bits
            preset_v = self.tq_config.value_quant_bits
            new_k = int(key_override) if key_override is not None else preset_k
            new_v = int(val_override) if val_override is not None else preset_v
            if not (2 <= new_k <= 4) or not (2 <= new_v <= 4):
                raise ValueError(
                    f"TURBOQUANT_KEY_BITS / TURBOQUANT_VALUE_BITS must be in "
                    f"{{2, 3, 4}}; got K={new_k}, V={new_v}."
                )
            if new_k > preset_k or new_v > preset_v:
                raise ValueError(
                    f"Override (K={new_k}, V={new_v}) exceeds preset slot "
                    f"capacity (preset {kv_cache_dtype}: K={preset_k}, "
                    f"V={preset_v}). Pick a larger preset such as "
                    f"turboquant_4bit_nc and override down from there."
                )
            self.tq_config = TurboQuantConfig(
                head_dim=head_size,
                key_quant_bits=new_k,
                value_quant_bits=new_v,
                norm_correction=self.tq_config.norm_correction,
            )
            logger.info(
                "TURBOQUANT bit-width override: K=%d, V=%d (preset slot was K=%d, V=%d)",
                new_k, new_v, preset_k, preset_v,
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
        # Capacity of the FP16 prefix buffer used by the deferred-prefill
        # decode path. Fixed at impl-init time so the tensor shape never
        # changes across calls (required for CUDA-graph capture).
        # Use max(max_model_len, max_num_batched_tokens) so that vLLM's
        # multi-token dummy capture run also fits.
        sched = getattr(vllm_config, "scheduler_config", None)
        max_batch = getattr(sched, "max_num_batched_tokens", 0) if sched else 0
        self._fp16_prefix_max = max(
            vllm_config.model_config.max_model_len,
            int(max_batch),
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
        already attached — but:
          - boundary-protect TQ override layers were configured as
            non-TQ in vLLM (cache_dtype="auto") so vLLM may not have
            attached centroids at all; and
          - the per-K env-var override changes `key_mse_bits` away from
            the preset, so the centroid table vLLM produced doesn't have
            the right number of levels.
        Regenerate the table here whenever it's missing or wrong-sized.
        """
        expected_n = 2 ** self.tq_config.key_mse_bits
        needs_regen = (
            not hasattr(layer, "_tq_centroids")
            or layer._tq_centroids is None
            or layer._tq_centroids.shape[0] != expected_n
        )
        if needs_regen:
            from vllm.model_executor.layers.quantization.turboquant.centroids import (
                get_centroids,
            )
            layer._tq_centroids = get_centroids(
                d=self.head_size, bits=self.tq_config.key_mse_bits
            ).to(device=device, dtype=torch.float32)
            # Invalidate any rotation-strategy-side cache so midpoints
            # are regenerated against the new centroid table.
            if getattr(layer, "_fused_tq_cached", False):
                layer._fused_tq_cached = False
        self.rotation.setup_layer(layer, self.head_size, layer._tq_centroids, device)
        # Eager-allocate the deferred-prefill side buffers so vLLM's
        # CUDA-graph capture dummy run sees the hybrid path (and not the
        # non-hybrid one, which would later read stale paged-cache slots
        # for positions whose K only lives in this side buffer).
        if self.defer_prefill:
            self._ensure_fp16_prefix_buf(layer, device)

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
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return  # encoder attention has no persistent KV cache
        if self._is_raw:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            self._store_raw_kv(k, v, kv_cache, slot_mapping)
            return
        self._ensure_setup(layer, key.device)
        # Boundary-layer slot was allocated as bf16/fp16. The TQ store
        # kernels do byte-level addressing, so reinterpret as uint8 here.
        # No-op when kv_cache is already uint8 (regular TQ layer).
        if kv_cache.dtype != torch.uint8:
            kv_cache = kv_cache.view(torch.uint8)
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)

        if self.defer_prefill and N > 1 and N <= self._fp16_prefix_max:
            # Deferred FP16 K-cache: skip the rotation+quantize store kernel
            # entirely; just stash a FP16 copy on the layer for the decode
            # path to do fast SDPA against. NOTE: single-sequence assumption —
            # we overwrite the buffer per prefill, so batch > 1 prefills will
            # clobber each other. Skip when N exceeds the fixed buffer
            # capacity (e.g. vLLM's multi-token dummy capture batch).
            self._ensure_fp16_prefix_buf(layer, key.device)
            layer._fp16_prefix_k[:N].copy_(k.to(torch.float16))
            layer._fp16_prefix_v[:N].copy_(v.to(torch.float16))
            layer._fp16_prefix_len_t.fill_(N)
            return

        self._launch_store(k, v, kv_cache, slot_mapping, layer)

    def _ensure_fp16_prefix_buf(self, layer, device) -> None:
        """Lazily allocate FP16 prefix buffers at fixed max capacity.
        Tensor shapes never change after this, which is required for
        CUDA-graph capture (the decode path holds onto these tensors).

        Seed `_fp16_prefix_len_t = 1` (not 0) so vLLM's CUDA-graph dummy
        run sees a non-degenerate flash_attn call (cu_seqlens_k=[0, 1]
        rather than [0, 0]). Some flash-attn implementations have a
        short-circuit for the empty-k case that, once captured, doesn't
        generalize back to the non-empty case at replay time.
        """
        if hasattr(layer, "_fp16_prefix_k"):
            return
        cap = self._fp16_prefix_max
        layer._fp16_prefix_k = torch.zeros(
            cap, self.num_kv_heads, self.head_size,
            dtype=torch.float16, device=device,
        )
        layer._fp16_prefix_v = torch.zeros_like(layer._fp16_prefix_k)
        layer._fp16_prefix_capacity = cap
        layer._fp16_prefix_len_t = torch.ones(
            1, dtype=torch.int32, device=device,
        )

    def _launch_store(
        self,
        key: torch.Tensor,  # [N, H, D]
        value: torch.Tensor,  # [N, H, D]
        kv_cache: torch.Tensor,  # [num_blocks, block_size, H, slot] uint8
        slot_mapping: torch.Tensor,  # [N]
        layer,
    ) -> None:
        """Delegate to the rotation strategy's in-kernel fused store.

        Each strategy supplies its own Triton kernel that does
        normalization + rotation + bucketize/pack + V quant + slot
        scatter in a single launch. The base-class fallback
        (`RotationStrategy.launch_store`) handles strategies that don't
        yet ship an in-kernel variant.

        When `TURBOQUANT_V_LLOYD_MAX=1` or `TURBOQUANT_V_ROTATE=1`, V is
        pre-rotated by the same rotation matrix that K uses. The
        post-attention inverse rotation is applied in
        `_decode_attention_offset_only` after stage2. `_V_LLOYD_MAX=1`
        additionally swaps the V quant from per-vec uniform to shared
        Lloyd-Max centroids; `_V_ROTATE=1` keeps the existing uniform
        codec so we can isolate the contribution of rotation alone.
        """
        if (os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1"
                or os.environ.get("TURBOQUANT_V_ROTATE", "0") == "1"):
            M = getattr(layer, "_fused_tq_rotation", None)
            if M is not None:
                value = torch.einsum("nhd,de->nhe", value.float(), M).to(value.dtype)

        self.rotation.launch_store(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            layer=layer,
            tq_config=self.tq_config,
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

        # Boundary-layer slot was allocated as bf16/fp16. TQ kernels do
        # byte-level addressing, so reinterpret as uint8 once at the
        # entry. _forward_raw / _store_raw_kv each idempotently
        # re-view, so this is safe.
        if kv_cache.dtype != torch.uint8:
            kv_cache = kv_cache.view(torch.uint8)

        # Raw fp16 fallback for boundary-protection layers — bypasses
        # rotation strategy and stock TQ kernels entirely.
        if self._is_raw:
            return self._forward_raw(
                query, key, value, kv_cache, attn_metadata, output, N
            )

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
        # Hybrid path: when defer_prefill is on, go through the two-phase
        # attention. NOTE: this Python-int dispatch is incompatible with
        # vLLM's CUDA-graph capture model — the dummy capture run has no
        # FP16 prefix, so the captured graph corresponds to the non-
        # hybrid branch and switching at runtime corrupts output. Run
        # with `enforce_eager=True` when measuring the deferred-prefill
        # path.
        if self.defer_prefill and hasattr(layer, "_fp16_prefix_len_t"):
            return self._decode_attention_hybrid(
                query, kv_cache, attn_metadata, layer
            )

        # Bit widths outside vLLM's stock 3/4-bit support (2-bit and 8-bit)
        # go through our forked offset kernel where those branches live,
        # with kv_start_offset=0 so semantics match stock stage1. Also force
        # the offset path whenever the user enabled V Lloyd-Max — the stock
        # vLLM stage1 only knows the uniform-V dequant layout.
        _kbits = self.tq_config.key_mse_bits
        _vbits = self.tq_config.effective_value_quant_bits
        _v_lloyd = os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1"
        _v_rotate = os.environ.get("TURBOQUANT_V_ROTATE", "0") == "1"
        if _kbits not in (3, 4) or _vbits not in (3, 4) or _v_lloyd or _v_rotate:
            return self._decode_attention_offset_only(
                query, kv_cache, attn_metadata, layer
            )

        from vllm.triton_utils import triton
        from vllm.v1.attention.ops.triton_turboquant_decode import (
            _tq_decode_stage1,
            _use_fp8_e4b15,
        )
        # Used only when V Lloyd-Max is OFF (env-gated routing happens above
        # at line ~782; the stock kernel doesn't accept V_LLOYD_MAX).
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
        # Stock kernel: no V_LLOYD_MAX support. Reached only when V Lloyd-Max
        # is disabled (the dispatcher above routes the on case to our offset).
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
            MSE_BITS=max(self.tq_config.key_mse_bits, 2),
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            ATTN_SCALE=self.scale,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
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
    # Hybrid decode: FP16 SDPA on prefix + quantized kernel on the rest
    # ------------------------------------------------------------------

    def _decode_attention_offset_only(
        self,
        query: torch.Tensor,  # [B, Hq, D]
        kv_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> torch.Tensor:
        """Run only our forked offset stage1 + stage2 (no FA prefix phase).

        Used when there is no FP16 prefix to merge — e.g. for 2-bit K / V
        modes where we need our extended kernel's unpacking branches.
        Equivalent semantics to vLLM's stock `_tq_decode_stage1` but
        routed through our fork so the 2-bit branches are reachable.
        """
        from vllm.triton_utils import triton
        from vllm.v1.attention.ops.triton_decode_attention import (
            _fwd_kernel_stage2,
        )
        from fused_turboquant.vllm_plugin.triton_decode_offset import (
            _tq_decode_stage1_offset,
            _use_fp8_e4b15,
        )

        B, Hq, D = query.shape
        Hk = kv_cache.shape[2]
        block_size = kv_cache.shape[1]
        kv_group_size = Hq // Hk
        device = query.device

        q_rot = self.rotation.rotate_for_decode(query.float(), layer)
        BLOCK_D = triton.next_power_of_2(D)
        NUM_KV_SPLITS = self.max_num_kv_splits

        mid_o = self._get_or_alloc_buf(
            layer, "_fused_mid_o", (B, Hq, NUM_KV_SPLITS, D + 1),
            dtype=torch.float32, device=device,
        )
        output = self._get_or_alloc_buf(
            layer, "_fused_output", (B, Hq, D),
            dtype=torch.float32, device=device,
        )
        lse = self._get_or_alloc_buf(
            layer, "_fused_lse", (B, Hq), dtype=torch.float32, device=device,
        )

        # kv_start_offset = 0 (no FP16 prefix): pre-alloc once on `self`.
        kv_start_offset = self._get_or_alloc_buf(
            self, "_kv_start_zero", (B,),
            dtype=attn_metadata.seq_lens.dtype, device=device,
        )
        kv_start_offset.zero_()

        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        BLOCK_KV = 4
        grid1 = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1_offset[grid1](
            q_rot,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            kv_start_offset,
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
            MSE_BITS=max(self.tq_config.key_mse_bits, 2),
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            ATTN_SCALE=self.scale,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            V_LLOYD_MAX=1 if os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1" else 0,
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
        # V rotation: stage2 has accumulated in the rotated V space.
        # Map back to the original V space with a single per-head GEMM.
        if (os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1"
                or os.environ.get("TURBOQUANT_V_ROTATE", "0") == "1"):
            M = getattr(layer, "_fused_tq_rotation", None)
            if M is not None:
                output = torch.einsum("bhd,ed->bhe", output, M)
        return output.to(query.dtype)

    def _decode_attention_hybrid(
        self,
        query: torch.Tensor,  # [B, Hq, D]
        kv_cache: torch.Tensor,
        attn_metadata,
        layer,
    ) -> torch.Tensor:
        """Two-phase attention with deferred FP16 prefix.

        Phase 1: SDPA(Q, K_fp16_prefix, V_fp16_prefix) → out_p, lse_p
                 (the prefix never went through rotate/quantize, so we
                 use the *original* K — no Q rotation here).
        Phase 2: stock decode kernel over the quantized [prefix_len, seq_len)
                 region, using a rotated Q against rotated centroids.
        Merge:   out = softmax-merge by log-sum-exp.

        The two phases use different Qs (raw vs rotated), but because R
        is orthogonal `Q · K_orig = (R·Q) · (R·K_orig)`, the scores
        produced in each phase live on the same scale, so the LSE merge
        is mathematically exact (modulo quantization noise on phase 2).
        """
        from vllm.triton_utils import triton
        from vllm.v1.attention.ops.triton_decode_attention import (
            _fwd_kernel_stage2,
        )
        from fused_turboquant.vllm_plugin.triton_decode_offset import (
            _tq_decode_stage1_offset,
            _use_fp8_e4b15,
        )

        B, Hq, D = query.shape
        Hk = kv_cache.shape[2]
        block_size = kv_cache.shape[1]
        kv_group_size = Hq // Hk
        device = query.device

        self._ensure_fp16_prefix_buf(layer, device)

        # ── Phase 1: flash-attn SDPA over the FP16 prefix ──────────────
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        prefix_k_full = layer._fp16_prefix_k  # [capacity, Hk, D]
        prefix_v_full = layer._fp16_prefix_v
        prefix_capacity = layer._fp16_prefix_capacity
        prefix_len_t = layer._fp16_prefix_len_t  # GPU int32 [1]
        q_fa = query.to(prefix_k_full.dtype).contiguous()
        MAX_B = 64
        if not hasattr(self, "_cu_q_arange"):
            self._cu_q_arange = torch.arange(
                MAX_B + 1, dtype=torch.int32, device=device,
            )
        cu_q = self._cu_q_arange[: B + 1]
        cu_k = self._get_or_alloc_buf(
            self, "_cu_k_buf", (B + 1,), dtype=torch.int32, device=device,
        )
        cu_k.zero_()
        cu_k[1:].copy_(prefix_len_t.expand(B))
        fa_kwargs = dict(
            q=q_fa, k=prefix_k_full, v=prefix_v_full,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=B, max_seqlen_k=prefix_capacity,
            softmax_scale=self.scale, causal=False,
            return_softmax_lse=True,
        )
        if self.fa_version is not None:
            fa_kwargs["fa_version"] = self.fa_version
        out_prefix, lse_prefix = flash_attn_varlen_func(**fa_kwargs)
        if lse_prefix.dim() == 2 and lse_prefix.shape != (B, Hq):
            lse_prefix = lse_prefix.transpose(0, 1).contiguous()

        # No CPU-side early exit (would break CUDA-graph capture). We
        # always run phase 2 too; if eff_seq_lens turns out to be 0 the
        # offset kernel produces lse=-inf for every split and the merge
        # below drops phase 2 to zero weight.
        seq_lens = attn_metadata.seq_lens

        # ── Phase 2: quantized decode kernel on positions [prefix_len, seq_len) ─
        q_rot = self.rotation.rotate_for_decode(query.float(), layer)
        BLOCK_D = triton.next_power_of_2(D)
        NUM_KV_SPLITS = self.max_num_kv_splits

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

        # GPU-only construction: pre-alloc + GPU→GPU copy from prefix_len_t.
        kv_start_offset = self._get_or_alloc_buf(
            self, "_kv_start_offset_buf", (B,),
            dtype=seq_lens.dtype, device=device,
        )
        kv_start_offset.copy_(prefix_len_t.to(seq_lens.dtype).expand(B))

        fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
        BLOCK_KV = 4
        grid1 = (B, Hq, NUM_KV_SPLITS)
        _tq_decode_stage1_offset[grid1](
            q_rot,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            kv_start_offset,
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
            MSE_BITS=max(self.tq_config.key_mse_bits, 2),
            MSE_BYTES=self._mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=self._val_data_bytes,
            ATTN_SCALE=self.scale,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            V_LLOYD_MAX=1 if os.environ.get("TURBOQUANT_V_LLOYD_MAX", "0") == "1" else 0,
            num_warps=1,
            num_stages=1,
        )

        grid2 = (B, Hq)
        # Pass an "effective seq_lens" = (seq_lens - prefix_len) to stage2
        # so its empty-split bookkeeping matches what stage1 actually
        # processed.
        eff_seq_lens = (seq_lens - prefix_len_t.to(seq_lens.dtype)).clamp(min=0)
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            eff_seq_lens,
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

        # Stage2 produces NaN for splits where eff_seq_lens==0 (division
        # by zero in `acc / e_sum`). Replace with safe sentinels before
        # the merge so an empty phase 2 reduces to "use phase 1 only".
        # Use `torch.full` (kernel-arg scalar, no CPU→GPU copy) to stay
        # CUDA-graph-safe.
        lse = lse.nan_to_num(nan=-float("inf"))
        output = output.nan_to_num(nan=0.0)

        # ── LSE merge ──────────────────────────────────────────────────
        out_prefix_f = out_prefix.to(torch.float32)
        lse_prefix_f = lse_prefix.to(torch.float32)
        lse_max = torch.maximum(lse_prefix_f, lse)
        exp_p = torch.exp(lse_prefix_f - lse_max)
        exp_q = torch.exp(lse - lse_max)
        total = (exp_p + exp_q).clamp(min=1e-20)
        w_p = (exp_p / total).unsqueeze(-1)
        w_q = (exp_q / total).unsqueeze(-1)
        out_combined = out_prefix_f * w_p + output * w_q
        return out_combined.to(query.dtype)

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
    # Raw fp16 SDPA fallback (boundary-protection skip layers)
    # ------------------------------------------------------------------

    def _store_raw_kv(
        self,
        key: torch.Tensor,  # [N, num_kv_heads, head_size]
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write fp16 K and V into a byte-indexed slot.

        Vectorized GPU-side scatter (no `.tolist()` Python loop) so this
        path is safe under CUDA-graph capture. Negative slots are vLLM's
        "padding token" marker (dummy capture data, end-of-sequence,
        empty decode step); clamping them naïvely to slot 0 corrupts
        the real K/V at slot 0 — which over a long eval slowly poisons
        the boundary layers' first-token attention. We instead snapshot
        slot 0 before the scatter and restore it after iff no valid
        token actually mapped there.

        Slot byte layout:
            [0 .. 2*head_size)            K (fp16 bytes)
            [2*head_size .. 4*head_size)  V (fp16 bytes)
            [4*head_size .. slot)         padding
        """
        n_tokens = key.shape[0]
        if n_tokens == 0:
            return
        kv_cache_u8 = (
            kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
        )
        head_size = self.head_size
        k_bytes = v_bytes = 2 * head_size
        k_fp16 = key.to(torch.float16).contiguous()
        v_fp16 = value.to(torch.float16).contiguous()
        k_u8 = k_fp16.view(torch.uint8).reshape(n_tokens, self.num_kv_heads, k_bytes)
        v_u8 = v_fp16.view(torch.uint8).reshape(n_tokens, self.num_kv_heads, v_bytes)
        flat = kv_cache_u8.view(-1, self.num_kv_heads, kv_cache_u8.shape[-1])
        # Snapshot slot 0 before the scatter so we can restore it iff no
        # genuine token mapped there. (slot_mapping == -1) padding entries
        # clamp to 0 and would otherwise clobber the real K, V at slot 0
        # — over a long eval this slowly poisons the first-token attention
        # for the boundary layers. The `(slot_mapping == 0).any()` reduction
        # produces a 0-dim tensor and the conditional restore goes through
        # `torch.where`, so the whole block is cudagraph-capturable.
        slot0_k_backup = flat[0:1, :, :k_bytes].clone()
        slot0_v_backup = flat[0:1, :, k_bytes : k_bytes + v_bytes].clone()
        safe_slots = slot_mapping.clamp(min=0)
        flat[safe_slots, :, :k_bytes] = k_u8
        flat[safe_slots, :, k_bytes : k_bytes + v_bytes] = v_u8
        slot0_was_real_target = (slot_mapping == 0).any().view(1, 1, 1)
        flat[0:1, :, :k_bytes] = torch.where(
            slot0_was_real_target, flat[0:1, :, :k_bytes], slot0_k_backup,
        )
        flat[0:1, :, k_bytes : k_bytes + v_bytes] = torch.where(
            slot0_was_real_target,
            flat[0:1, :, k_bytes : k_bytes + v_bytes],
            slot0_v_backup,
        )

    def _gather_raw_kv(
        self,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read full cached K/V for one sequence. Returns each as
        `[1, num_kv_heads, seq_len, head_size]` fp16."""
        kv_cache_u8 = (
            kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
        )
        block_size = kv_cache_u8.shape[1]
        head_size = self.head_size
        k_bytes = v_bytes = 2 * head_size
        n_blocks = (seq_len + block_size - 1) // block_size
        k_parts, v_parts = [], []
        for b in range(n_blocks):
            block_idx = int(block_table_row[b].item())
            tokens_here = min(block_size, seq_len - b * block_size)
            block = kv_cache_u8[block_idx, :tokens_here]
            k_u8 = block[:, :, :k_bytes].contiguous()
            v_u8 = block[:, :, k_bytes : k_bytes + v_bytes].contiguous()
            k_parts.append(
                k_u8.view(torch.float16).reshape(tokens_here, self.num_kv_heads, head_size)
            )
            v_parts.append(
                v_u8.view(torch.float16).reshape(tokens_here, self.num_kv_heads, head_size)
            )
        k = torch.cat(k_parts, dim=0).transpose(0, 1).unsqueeze(0).contiguous()
        v = torch.cat(v_parts, dim=0).transpose(0, 1).unsqueeze(0).contiguous()
        return k, v

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """GQA expand: [b, num_kv_heads, s, d] → [b, num_heads, s, d]."""
        if self.num_kv_groups == 1:
            return x
        b, h, s, d = x.shape
        x = x[:, :, None, :, :].expand(b, h, self.num_kv_groups, s, d)
        return x.reshape(b, h * self.num_kv_groups, s, d)

    def _forward_raw(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Vectorized plain-FP16 attention for boundary-protection layers.

        Uses flash_attn_varlen_func with the paged block_table directly
        — no Python `.tolist()` loops, no per-sequence dispatching — so
        this path is CUDA-graph capturable. The trick: our slot byte
        layout for boundary slots is exactly `[K_fp16 | V_fp16]` (4 * D
        bytes, already a power of 2 for D ∈ {64, 128, 256, 512}), so we
        can `.view(torch.float16)` the cache and slice off K, V as the
        flash-attn paged-cache tensors.
        """
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        q = query[:N].view(N, self.num_heads, self.head_size)

        # View the byte-layout slots as FP16 [K | V]. Requires the slot
        # to be exactly 2*head_size FP16 elements (no padding).
        head_size = self.head_size
        kv_u8 = (
            kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
        )
        slot_bytes = kv_u8.shape[-1]
        expected_bytes = 4 * head_size  # K_fp16 + V_fp16
        if slot_bytes != expected_bytes:
            # Padded slot — fall back to the per-sequence Python loop.
            return self._forward_raw_python_loop(
                query, key, value, kv_cache, attn_metadata, output, N
            )

        # Workaround for the Blackwell + FA paged + non-contiguous K/V
        # cluster bug: route through a CUDA-graph-capturable path that
        # never lets flash-attn touch the interleaved `[K|V]`-packed
        # cache. Prefill is local SDPA on the current K, V (no cache
        # read, identical to the pre-d34258a python loop). Decode uses
        # the custom Triton kernel in `triton_boundary_attn.py`, which
        # reads K/V byte-by-byte (`uint8` loads → bitcast to fp16) to
        # avoid the stride-2 paged view that FA misreads on SM ≥ 10.
        #
        # Opt back into the original FA paged path with
        # TQ_BOUNDARY_FA_PAGED=1 (e.g. for benchmarking, or on hardware
        # unaffected by the cluster bug).
        if os.environ.get("TQ_BOUNDARY_FA_PAGED", "0") != "1":
            B = attn_metadata.seq_lens.shape[0]
            if N > B:
                # Pure prefill (B sequences, each with > 1 query token).
                # Run SDPA over current K, V — same numerics as
                # `_sdpa_local` in the python loop. is_causal handles
                # within-prompt masking.
                q = query[:N].view(N, self.num_heads, self.head_size)
                k = key[:N].view(N, self.num_kv_heads, self.head_size)
                v = value[:N].view(N, self.num_kv_heads, self.head_size)
                causal = self.attn_type not in (
                    AttentionType.ENCODER, AttentionType.ENCODER_ONLY,
                )
                attn_out = self._sdpa_local(q, k, v, is_causal=causal)
                if output.ndim == 3:
                    output[:N] = attn_out.to(output.dtype)
                else:
                    output[:N] = attn_out.reshape(N, -1).to(output.dtype)
                return output
            # Decode: custom Triton paged attention.
            from fused_turboquant.vllm_plugin.triton_boundary_attn import (
                boundary_fp16_decode_attention,
            )
            q = query[:N].view(N, self.num_heads, self.head_size).to(torch.float16)
            # Reshape output buffer to [N, H_q, D] for the kernel write.
            if output.ndim == 3:
                out_view = output[:N]
            else:
                out_view = output[:N].view(N, self.num_heads, self.head_size)
            out_view_fp16 = out_view if out_view.dtype == torch.float16 else \
                torch.empty(N, self.num_heads, self.head_size,
                            dtype=torch.float16, device=q.device)
            boundary_fp16_decode_attention(
                query=q,
                kv_cache=kv_cache,
                block_table=attn_metadata.block_table,
                seq_lens=attn_metadata.seq_lens,
                output=out_view_fp16,
                num_kv_groups=self.num_kv_groups,
                scale=self.scale,
            )
            if out_view.dtype != torch.float16:
                if output.ndim == 3:
                    output[:N] = out_view_fp16.to(output.dtype)
                else:
                    output[:N] = out_view_fp16.reshape(N, -1).to(output.dtype)
            return output

        # head_size > 256 exceeds flash-attn-2's limit. For pure-decode
        # batches we have a CUDA-graph-capturable alternative: gather a
        # padded K/V tensor and run PyTorch SDPA. For mixed / prefill
        # batches we fall back to the per-sequence Python loop.
        if head_size > 256:
            if (not attn_metadata.is_prefill
                    and self.attn_type == AttentionType.DECODER):
                return self._forward_raw_sdpa_paged(
                    query, kv_cache, attn_metadata, output, N
                )
            return self._forward_raw_python_loop(
                query, key, value, kv_cache, attn_metadata, output, N
            )

        kv_fp16 = kv_u8.view(torch.float16).view(
            kv_u8.shape[0], kv_u8.shape[1], kv_u8.shape[2], 2 * head_size,
        )
        k_cache = kv_fp16[..., :head_size]
        v_cache = kv_fp16[..., head_size:]

        causal = self.attn_type not in (
            AttentionType.ENCODER, AttentionType.ENCODER_ONLY
        )

        # Use flash_attn_varlen_func with paged block_table. The cache
        # already has the current K, V written via `_store_raw_kv`, so
        # `seqused_k = seq_lens` and we don't need to pass k_new / v_new.
        kwargs = dict(
            q=q.to(torch.float16),
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=attn_metadata.query_start_loc.to(torch.int32),
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens.to(torch.int32),
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=causal,
            block_table=attn_metadata.block_table.to(torch.int32),
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        attn_out = flash_attn_varlen_func(**kwargs)  # [N, num_heads, head_size]

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _forward_raw_sdpa_paged(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Pure-decode path that supports any head_size by using PyTorch
        SDPA over a padded paged-cache gather. CUDA-graph capturable:
        all shapes are static (max_blocks × block_size), with the
        attention mask zeroing out positions past `seq_lens[i]`.

        Memory cost: gathers `[B, max_blocks * block_size, num_kv_heads,
        head_size]` FP16 per call, which can be tens of MB for long
        contexts. Used only for boundary-protection layers with
        head_size > 256 (e.g. Gemma 4 31B global layers).
        """
        B = attn_metadata.seq_lens.shape[0]
        H_q = self.num_heads
        H_k = self.num_kv_heads
        D = self.head_size
        device = query.device

        q = query[:N].view(N, H_q, D).to(torch.float16)

        kv_u8 = (
            kv_cache.view(torch.uint8) if kv_cache.dtype != torch.uint8 else kv_cache
        )
        block_size = kv_u8.shape[1]
        block_table = attn_metadata.block_table  # [B, max_blocks]
        max_blocks = block_table.shape[1]
        max_kv = max_blocks * block_size

        # Gather all blocks per sequence: [B, max_blocks, block_size, H_k, slot_bytes]
        # Negative block ids are clamped to 0 — those positions are
        # masked out by `seq_lens` below, so the garbage content is
        # never used.
        bt_safe = block_table.clamp(min=0)
        selected = kv_u8[bt_safe]
        selected_fp16 = selected.view(torch.float16).view(B, max_blocks, block_size, H_k, 2 * D)
        k_all = selected_fp16[..., :D]
        v_all = selected_fp16[..., D:]
        k_all = k_all.reshape(B, max_kv, H_k, D)
        v_all = v_all.reshape(B, max_kv, H_k, D)

        # GQA expand on the head axis (Hq // Hk replicas of each KV head).
        if H_q != H_k:
            k_all = k_all.repeat_interleave(self.num_kv_groups, dim=2)
            v_all = v_all.repeat_interleave(self.num_kv_groups, dim=2)

        # SDPA wants [B, Hq, S, D]:
        k_sdpa = k_all.transpose(1, 2)
        v_sdpa = v_all.transpose(1, 2)
        # q is [N, Hq, D] with N == B for pure decode; add a length-1 q
        # axis.
        q_sdpa = q.view(B, 1, H_q, D).transpose(1, 2)  # [B, Hq, 1, D]

        # Build an attention mask that is True for positions < seq_lens[i].
        positions = torch.arange(max_kv, device=device, dtype=attn_metadata.seq_lens.dtype)
        mask = positions.unsqueeze(0) < attn_metadata.seq_lens.unsqueeze(1)  # [B, max_kv]
        attn_mask = mask.view(B, 1, 1, max_kv)

        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=attn_mask,
            scale=self.scale,
        )
        attn_out = out.squeeze(2)  # [B, Hq, D]

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _forward_raw_python_loop(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        N: int,
    ) -> torch.Tensor:
        """Original per-sequence implementation. Kept for the padded-slot
        case (where the FP16 view trick doesn't apply); not CUDA-graph
        capturable due to `.tolist()` calls."""
        q = query[:N].view(N, self.num_heads, self.head_size)
        attn_out = torch.empty(
            N, self.num_heads, self.head_size, dtype=q.dtype, device=q.device
        )
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

            if self.attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
                sub = self._sdpa_local(q_i, k_i, v_i, is_causal=False)
            elif q_len > 1 and context_len == 0:
                sub = self._sdpa_local(q_i, k_i, v_i, is_causal=True)
            elif q_len > 1 and context_len > 0:
                raise NotImplementedError(
                    "Raw fp16 fallback: chunked prefill (context_len>0, "
                    "q_len>1) is not supported."
                )
            else:
                cached_k, cached_v = self._gather_raw_kv(
                    kv_cache, block_table[i], seq_len
                )
                cached_k = cached_k.to(q_i.dtype)
                cached_v = cached_v.to(q_i.dtype)
                sub = self._sdpa_with_cached(q_i, cached_k, cached_v)
            attn_out[q_s:q_e] = sub

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _sdpa_local(self, q, k, v, is_causal: bool) -> torch.Tensor:
        """SDPA on local (no-cache) Q/K/V. Returns `[q_len, num_heads, head_size]`."""
        qt = q.unsqueeze(0).transpose(1, 2)  # [1, h, s, d]
        kt = self._repeat_kv(k.unsqueeze(0).transpose(1, 2))
        vt = self._repeat_kv(v.unsqueeze(0).transpose(1, 2))
        out = F.scaled_dot_product_attention(
            qt, kt, vt, scale=self.scale, is_causal=is_causal
        )
        return out.squeeze(0).transpose(0, 1).contiguous()

    def _sdpa_with_cached(self, q, cached_k, cached_v) -> torch.Tensor:
        """Decode SDPA over full cached K/V. q is `[1, num_heads, head_size]`."""
        qt = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, 1, head_size]
        kt = self._repeat_kv(cached_k)
        vt = self._repeat_kv(cached_v)
        out = F.scaled_dot_product_attention(
            qt, kt, vt, scale=self.scale, is_causal=False
        )
        return out.squeeze(0).squeeze(1)  # [num_heads, head_size]

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
