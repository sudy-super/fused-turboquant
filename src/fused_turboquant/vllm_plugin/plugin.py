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
    _patch_disable_boundary_protection()


def _boundary_protect_mode() -> "str | int":
    """Parse `TURBOQUANT_BOUNDARY_PROTECT` env var.

    Returns:
      - `"off"`: no boundary protection — every layer goes through the
        configured TQ preset / `turboquant_*` cache dtype and the fast
        Triton path.
      - `"fp16"`: vLLM auto-adds the first/last 2 attention layers to
        `kv_cache_dtype_skip_layers` and forces them to
        `kv_cache_dtype="auto"` (raw fp16). The backend dispatches those
        to a paged flash-attn / SDPA path over raw fp16 K, V. This is
        the historical default.
      - An `int` in `{2, 3, 4, 8}`: same skip-layer mechanism as the
        fp16 mode (boundary slot is still FP16-sized at the spec level),
        but the boundary backend stores K / V as Lloyd-Max MSE keys and
        uniform-quantized values at the requested bit width instead of
        raw fp16. The fp16-sized slot has plenty of room — for d=128 we
        need at most 2·d + 6 = 262 bytes vs the 512-byte fp16 slot, so
        the extra capacity is just unused.

    Accepted strings (case-insensitive):
      `0` / `off` / `false` / `no`            → `"off"`
      `1` / `fp16` / `f16` / `bf16`           → `"fp16"`
      `2` / `3` / `4` / `8`                   → that int

    Backward-compat with the older boolean-style env var: `0` still
    means off, `1` still means fp16.
    """
    import os

    raw = os.environ.get("TURBOQUANT_BOUNDARY_PROTECT", "1").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return "off"
    if raw in ("1", "fp16", "f16", "bf16", "bfloat16"):
        return "fp16"
    try:
        bits = int(raw)
    except ValueError as e:
        raise ValueError(
            f"TURBOQUANT_BOUNDARY_PROTECT={raw!r} not understood. Use 0/off, "
            f"1/fp16, or a bit count from {{2, 3, 4, 8}}."
        ) from e
    if bits not in (2, 3, 4, 8):
        raise ValueError(
            f"TURBOQUANT_BOUNDARY_PROTECT={bits} not supported. Use one of "
            f"{{2, 3, 4, 8}} for a quantized boundary, or 1/fp16 for raw."
        )
    return bits


def _boundary_protect_enabled() -> bool:
    """Backward-compat boolean view of the boundary-protect mode."""
    return _boundary_protect_mode() != "off"


def _patch_disable_boundary_protection() -> None:
    """When `TURBOQUANT_BOUNDARY_PROTECT=0`, override
    `TurboQuantConfig.get_boundary_skip_layers` to return `[]`.
    Otherwise no-op (boundary protection stays at vLLM's default ON).
    Idempotent."""
    if _boundary_protect_enabled():
        logger.info(
            "fused-turboquant: TQ boundary protection ENABLED (default). "
            "Set TURBOQUANT_BOUNDARY_PROTECT=0 to disable and let every "
            "layer go through the fast Triton path."
        )
        return

    try:
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
    except ImportError:
        return

    current = TurboQuantConfig.__dict__.get("get_boundary_skip_layers")
    if current is not None and getattr(
        current.__func__ if hasattr(current, "__func__") else current,
        "_ft_patched",
        False,
    ):
        return

    def patched(num_layers, n=2):
        return []

    patched._ft_patched = True
    TurboQuantConfig.get_boundary_skip_layers = staticmethod(patched)
    logger.info(
        "fused-turboquant: TQ boundary protection DISABLED — every layer "
        "uses the configured turboquant_* cache dtype, all going through "
        "our fast Triton path."
    )


def _patch_attention_get_kv_cache_spec() -> None:
    """Force every attention layer to expose a TQFullAttentionSpec whose
    `tq_slot_size` matches what `FusedTurboQuantV1Backend.get_kv_cache_shape`
    returns.

    vLLM's stock `Attention.get_kv_cache_spec` can emit three relevant
    shapes for layers we'd like to back with TurboQuant kernels:

    - `SlidingWindowSpec` (e.g. Gemma 4 sliding text decoder layers)
    - `FullAttentionSpec` (e.g. raw fp16 / bf16 layers, or full text
      layers on vLLM builds without TQFullAttentionSpec)
    - `TQFullAttentionSpec` (full text layers with kv_cache_dtype set to a
      built-in `turboquant_*` preset)

    The third case is the subtle one: vLLM constructs the TQFullAttentionSpec
    with its OWN `tq_slot_size` derived from the preset's `slot_size_aligned`,
    which differs from the power-of-2 rounded value our backend uses in
    `get_kv_cache_shape`. That mismatch was the source of the
    `[120380, 16, 16, 1024]` vs half-size allocation bug — `spec.page_size_bytes`
    came from vLLM's number, the raw tensor allocator used `spec.page_size_bytes`
    too, and then our shape function returned a bigger numel that no longer fit.

    Fix: rebuild EVERY `TQFullAttentionSpec` (along with the other two
    cases) using our own `_slot_size_for(head_size, cache_dtype)`. Now
    `spec.page_size_bytes` and `get_kv_cache_shape().numel()` are derived
    from the same formula, so allocation always lines up.
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
        # CRITICAL: only rewrite for layers that will actually use the
        # TurboQuant backend (kv_cache_dtype="turboquant_*"). Layers with
        # kv_cache_dtype="auto" / "float16" / "bfloat16" are dispatched to
        # FLASH_ATTN / TRITON_ATTN / etc., whose `get_kv_cache_shape`
        # produces a 5-D `(2, num_blocks, block_size, num_kv_heads,
        # head_size)` layout in bf16 — entirely incompatible with our
        # 4-D uint8 byte slot. If we rewrote those specs to
        # `TQFullAttentionSpec(dtype=uint8, ...)`, the allocator would
        # carve a uint8 buffer that the non-TQ backend then tries to view
        # as bf16-typed 5-D, and the .view() call fails with a size
        # mismatch. Leave non-TQ layers' specs untouched.
        if not (
            isinstance(cache_dtype, str) and cache_dtype.startswith("turboquant_")
        ):
            return spec
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
            head_size_v=getattr(spec, "head_size_v", spec.head_size),
            dtype=torch.uint8,
            tq_slot_size=slot_size,
        )
        # Order matters: check TQFullAttentionSpec BEFORE FullAttentionSpec
        # because TQFullAttentionSpec is a subclass of FullAttentionSpec on
        # some vLLM versions. We rebuild it unconditionally to install our
        # own tq_slot_size (vLLM's internal default disagrees with our
        # get_kv_cache_shape's power-of-2 rounded value).
        if isinstance(spec, TQFullAttentionSpec):
            return TQFullAttentionSpec(**tq_kwargs)
        if isinstance(spec, SlidingWindowSpec):
            return TQFullAttentionSpec(**tq_kwargs)
        if isinstance(spec, FullAttentionSpec):
            return TQFullAttentionSpec(**tq_kwargs)
        return spec

    patched._ft_patched = True  # type: ignore[attr-defined]
    Attention.get_kv_cache_spec = patched
    logger.info(
        "fused-turboquant: patched Attention.get_kv_cache_spec to emit "
        "TQFullAttentionSpec with our own tq_slot_size on every layer"
    )


def _patch_unify_page_size() -> None:
    """Replace `unify_kv_cache_spec_page_size` with a version that widens
    smaller specs by an integer ratio of block_size.

    The stock vLLM implementation does exactly this when page sizes are in
    an integer ratio, and raises NotImplementedError otherwise. Our
    `_slot_size_for` rounds slot bytes up to a power of 2, so the cross-
    layer page_size ratios in Gemma 4 (sliding head_dim=256 vs full
    head_dim=512) are guaranteed to be a power-of-2 ratio. Therefore the
    integer-ratio path is always sufficient.

    Why patch at all if the stock impl can do this? Because some vLLM
    builds emit `NotImplementedError` for any mismatch without trying the
    ratio path. Stamping our own version makes the behavior deterministic
    across vLLM versions.
    """
    try:
        from dataclasses import replace
        from vllm.v1.core import kv_cache_utils
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
            layer_page_size = layer_spec.page_size_bytes
            if layer_page_size == max_page_size:
                new_kv_cache_spec[layer_name] = layer_spec
                continue
            if max_page_size % layer_page_size != 0:
                raise NotImplementedError(
                    "fused-turboquant unify shim expected integer-ratio "
                    f"page sizes (max={max_page_size}, layer "
                    f"{layer_name}={layer_page_size}). Power-of-2 slot "
                    "rounding in _slot_size_for should make this hold — "
                    "if you hit this, slot_size for some layer is not a "
                    "power of 2."
                )
            ratio = max_page_size // layer_page_size
            new_kv_cache_spec[layer_name] = replace(
                layer_spec, block_size=layer_spec.block_size * ratio
            )
        return new_kv_cache_spec

    patched._ft_patched = True  # type: ignore[attr-defined]
    kv_cache_utils.unify_kv_cache_spec_page_size = patched
    logger.info(
        "fused-turboquant: patched unify_kv_cache_spec_page_size to widen "
        "smaller layers by integer block_size ratio (power-of-2 only)"
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
