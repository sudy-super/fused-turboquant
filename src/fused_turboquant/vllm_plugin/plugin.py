"""
vLLM plugin registration for fused-turboquant attention backend.

Registers via vLLM's entry point system. Supports both:

- vLLM >= 0.20 (v1 only): registers FusedTurboQuantV1Backend against the
  `TURBOQUANT` slot in `vllm.v1.attention.backends.registry.AttentionBackendEnum`.
  Select with:
      LLM(model, attention_backend="TURBOQUANT")
      vllm serve <model> --attention-backend TURBOQUANT

- vLLM ≤ 0.18 (v0): registers the legacy FusedTurboQuantBackend through
  the v0 registry under the name `FUSED_TURBOQUANT`. Select with:
      vllm serve <model> --attention-backend FUSED_TURBOQUANT

Whichever API the installed vLLM exposes, we try both; failures are
logged at WARNING (not raised) so a single plugin entry point works
across both vLLM generations.
"""

from __future__ import annotations

import logging

import torch  # noqa: E402 — used by the get_kv_cache_spec monkey-patch

logger = logging.getLogger(__name__)


def register_backend() -> None:
    """vLLM plugin entry point — called automatically when fused-turboquant is installed.

    Tries to register on both v0 and v1 attention registries. Failures on
    one path are non-fatal: as long as one succeeds, the backend is usable
    on the installed vLLM.

    Also installs a small compatibility shim on
    `vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size` so that
    multimodal models with mixed TQ + raw fp16 page sizes that don't divide
    evenly can still come up — by widening the smaller specs via
    `page_size_padded` instead of raising NotImplementedError. This patch is
    a no-op when all page sizes already agree or are in an integer ratio.
    """
    registered_any = False
    registered_any |= _try_register_v1()
    registered_any |= _try_register_v0()

    if not registered_any:
        logger.warning(
            "vLLM attention backend registry not found (tried both v1 and v0 paths). "
            "fused-turboquant plugin requires vLLM >= 0.8. "
            "The TURBOQUANT / FUSED_TURBOQUANT backend will not be available."
        )

    _patch_attention_get_kv_cache_spec()
    _patch_unify_page_size()


def _patch_attention_get_kv_cache_spec() -> None:
    """Force every attention layer to expose a TQFullAttentionSpec.

    The default `Attention.get_kv_cache_spec` returns FullAttentionSpec for
    `kv_cache_dtype="auto"` (e.g. the vision encoder of multimodal Gemma 4),
    which has a leading-2 page layout and a different page size than
    TurboQuant text-decoder layers. By making every attention layer carry a
    TQFullAttentionSpec with `tq_slot_size = slot_size_for(head_size,
    cache_dtype)`, the framework can compare scalar page sizes and the unify
    shim can widen smaller layers via `page_size_padded`. Quantized layers
    behave exactly as before.
    """
    try:
        from vllm.model_executor.layers.attention import attention as _attn_mod
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
            TQFullAttentionSpec,
        )
        from fused_turboquant.vllm_plugin.v1_backend import FusedTurboQuantV1Backend
    except ImportError:
        return

    Attention = _attn_mod.Attention
    if getattr(Attention.get_kv_cache_spec, "_ft_patched", False):
        return

    original = Attention.get_kv_cache_spec

    def patched(self, *args, **kwargs):  # type: ignore[no-redef]
        spec = original(self, *args, **kwargs)
        cache_dtype = getattr(self, "kv_cache_dtype", "auto") or "auto"
        slot_size = FusedTurboQuantV1Backend._slot_size_for(self.head_size, cache_dtype)
        # All TurboQuant slots are byte arrays — vLLM views the raw allocation
        # as `spec.dtype` to compute element count, so we MUST advertise uint8
        # (1 byte per element) for the byte counting to line up with
        # `get_kv_cache_shape`. The model itself still runs in bf16 / fp16; we
        # just take responsibility for byte ↔ element packing inside our Impl.
        tq_kwargs = dict(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=torch.uint8,
            tq_slot_size=slot_size,
        )
        # SlidingWindowSpec on a TurboQuant layer: vLLM's stock code returns a
        # SlidingWindowSpec that pretends head_size is the slot size and
        # therefore ignores TurboQuant entirely. Re-emit it as a
        # TQFullAttentionSpec; the sliding-window mask is applied at runtime
        # by FusedTurboQuantV1Impl.forward().
        if isinstance(spec, SlidingWindowSpec) and cache_dtype.startswith("turboquant_"):
            return TQFullAttentionSpec(**tq_kwargs)
        # Plain `FullAttentionSpec` (auto / fp16 / bf16 layers like the
        # vision encoder of Gemma 4): redirect to TQFullAttentionSpec so we
        # use a uniform K+V slot layout that the unify shim can pad.
        if type(spec) is FullAttentionSpec:
            return TQFullAttentionSpec(**tq_kwargs)
        return spec

    patched._ft_patched = True  # type: ignore[attr-defined]
    Attention.get_kv_cache_spec = patched
    logger.info(
        "fused-turboquant: patched Attention.get_kv_cache_spec to emit "
        "TQFullAttentionSpec for raw fp16/bf16 layers (uniform 4-D slot layout)"
    )


def _patch_unify_page_size() -> None:
    """Replace `unify_kv_cache_spec_page_size` with a version that widens
    non-divisible specs via `page_size_padded` instead of erroring.

    The original vLLM impl (vllm.v1.core.kv_cache_utils) only knows how to
    inflate a layer's block_size by an integer ratio. For multimodal models
    that mix a TurboQuant-slot text decoder and a raw-fp16 vision encoder,
    those page sizes are typically not in an integer ratio (e.g. text
    slot=262 B × 16 heads vs vision raw=72 × 2 B = 144 B × 16 heads), so the
    upstream code raises NotImplementedError. AttentionSpec already supports
    `page_size_padded` as an override knob; we just teach the unifier to use
    it.
    """
    try:
        from dataclasses import replace
        from vllm.v1.core import kv_cache_utils
        from vllm.v1.kv_cache_interface import AttentionSpec
    except ImportError:
        return

    original = getattr(kv_cache_utils, "unify_kv_cache_spec_page_size", None)
    if original is None or getattr(original, "_ft_patched", False):
        return

    def patched(kv_cache_spec):
        page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
        if len(page_sizes) <= 1:
            return kv_cache_spec
        max_page_size = max(page_sizes)
        new_kv_cache_spec = {}
        for layer_name, layer_spec in kv_cache_spec.items():
            if layer_spec.page_size_bytes == max_page_size:
                new_kv_cache_spec[layer_name] = layer_spec
                continue
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                # Original strategy: bump block_size by the integer ratio.
                ratio = max_page_size // layer_page_size
                new_kv_cache_spec[layer_name] = replace(
                    layer_spec, block_size=layer_spec.block_size * ratio
                )
            else:
                # New: pad this layer's slot via page_size_padded so its
                # page_size_bytes lines up with max_page_size. Only works on
                # AttentionSpec subclasses (the only ones that expose the
                # padding knob).
                if not isinstance(layer_spec, AttentionSpec):
                    raise NotImplementedError(
                        "Non-AttentionSpec layer with non-divisible page size: "
                        f"{type(layer_spec).__name__} {layer_name}"
                    )
                logger.info(
                    "fused-turboquant: padding %s %s: real_page_size=%d → "
                    "padded=%d",
                    layer_name,
                    type(layer_spec).__name__,
                    layer_page_size,
                    max_page_size,
                )
                new_kv_cache_spec[layer_name] = replace(
                    layer_spec, page_size_padded=max_page_size
                )
        # Stamp the unified per-(block_size, num_kv_heads, head_size, dtype)
        # slot bytes into the backend so its get_kv_cache_shape can return a
        # tensor with the same numel as the padded page_size_bytes.
        #
        # IMPORTANT: this must run on every layer in the spec map, INCLUDING
        # the layer whose original page_size_bytes already equals
        # max_page_size — we still need to advertise its slot through the
        # override map so get_kv_cache_shape returns the same numel that vLLM
        # will allocate. (raw_tensor.size() is num_blocks * page_size_bytes
        # for the WHOLE spec map, so even the "winning" layer must match the
        # padded page_size.)
        try:
            from fused_turboquant.vllm_plugin.v1_backend import (
                FusedTurboQuantV1Backend,
            )
        except ImportError:
            return new_kv_cache_spec
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            TQFullAttentionSpec,
        )
        max_page_size_resolved = max_page_size
        for layer_name, spec in new_kv_cache_spec.items():
            if not isinstance(spec, (FullAttentionSpec, TQFullAttentionSpec)):
                continue
            # The final page size every layer needs to expose is the unified
            # max_page_size (after padding). Compute the per-token-per-head
            # slot from that and the (block_size, num_kv_heads) of THIS layer.
            slot = max_page_size_resolved // (spec.block_size * spec.num_kv_heads)
            for dt in (
                "auto",
                "float16",
                "bfloat16",
                "turboquant_4bit_nc",
                "turboquant_3bit_nc",
                "turboquant_k3v4_nc",
                "turboquant_k8v4",
            ):
                key = (spec.block_size, spec.num_kv_heads, spec.head_size, dt)
                FusedTurboQuantV1Backend._slot_size_overrides[key] = slot
        return new_kv_cache_spec

    patched._ft_patched = True  # type: ignore[attr-defined]
    kv_cache_utils.unify_kv_cache_spec_page_size = patched
    logger.info(
        "fused-turboquant: patched unify_kv_cache_spec_page_size to support "
        "non-divisible mixed page sizes via page_size_padded"
    )


def _try_register_v1() -> bool:
    """Register FusedTurboQuantV1Backend under AttentionBackendEnum.TURBOQUANT."""
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend as v1_register_backend,
        )
    except ImportError:
        return False

    try:
        v1_register_backend(
            AttentionBackendEnum.TURBOQUANT,
            "fused_turboquant.vllm_plugin.v1_backend.FusedTurboQuantV1Backend",
        )
        logger.info(
            "fused-turboquant: registered TURBOQUANT attention backend with vLLM v1"
        )
        return True
    except Exception as e:
        logger.warning("Failed to register TURBOQUANT backend on v1 registry: %s", e)
        return False


def _try_register_v0() -> bool:
    """Register FusedTurboQuantBackend on the legacy v0 registry."""
    try:
        from vllm.attention.backends.registry import AttentionBackendRegistry
    except ImportError:
        return False

    try:
        from fused_turboquant.vllm_plugin.backend import FusedTurboQuantBackend
    except Exception as e:
        logger.warning("Failed to import v0 FusedTurboQuantBackend: %s", e)
        return False

    try:
        AttentionBackendRegistry.register("FUSED_TURBOQUANT", FusedTurboQuantBackend)
        logger.info(
            "fused-turboquant: registered FUSED_TURBOQUANT attention backend with vLLM v0"
        )
        return True
    except Exception as e:
        logger.warning("Failed to register FUSED_TURBOQUANT backend on v0 registry: %s", e)
        return False
