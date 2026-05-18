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
import os
import re

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
    _patch_extend_tq_presets()
    _patch_cache_dtype_validation()


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
    if raw in ("fp8", "f8"):
        # Boundary keys as FP8 (1 byte per element, no LUT). Same byte
        # count as 8-bit MSE but no codebook overhead — slightly worse
        # quality, slightly less compute.
        return "fp8"
    try:
        bits = int(raw)
    except ValueError as e:
        raise ValueError(
            f"TURBOQUANT_BOUNDARY_PROTECT={raw!r} not understood. Use 0/off, "
            f"1/fp16, fp8, or a bit count from {{2, 3, 4, 8}}."
        ) from e
    if bits not in (2, 3, 4, 8):
        raise ValueError(
            f"TURBOQUANT_BOUNDARY_PROTECT={bits} not supported. Use one of "
            f"{{2, 3, 4, 8}} for an MSE-quantized boundary, or 1/fp16 or "
            f"fp8 for raw / FP8 boundary."
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


_FT_KVDTYPE_PATTERN = re.compile(r"^turboquant_k([1-4])v([1-4])_nc$")


# Map between EngineArgs.additional_config["turboquant"] keys and the
# environment variables our v1_backend / plugin code reads internally.
# Order matches their reading sites for grep-ability.
_FT_TQ_CONFIG_ENV_MAP: dict[str, str] = {
    "kind": "TURBOQUANT_KIND",
    "boundary_protect": "TURBOQUANT_BOUNDARY_PROTECT",
    "v_rotate": "TURBOQUANT_V_ROTATE",
    "v_lloyd_max": "TURBOQUANT_V_LLOYD_MAX",
    "key_bits": "TURBOQUANT_KEY_BITS",
    "value_bits": "TURBOQUANT_VALUE_BITS",
    "defer_prefill": "TURBOQUANT_DEFER_PREFILL",
    "prefill_fa_version": "TQ_PREFILL_FA_VERSION",
    "cudagraph_mode": "TQ_CUDAGRAPH_MODE",
    "boundary_fa_paged": "TQ_BOUNDARY_FA_PAGED",
}


def _ft_coerce_env_value(value) -> str:
    """Format a single `additional_config["turboquant"][key]` value as a
    string suitable for an environment variable. Booleans become "0"/"1",
    everything else falls through to ``str(value)``."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _ft_apply_turboquant_additional_config(engine_args) -> None:
    """Read `engine_args.additional_config["turboquant"]` (if any) and
    project each entry into the corresponding TURBOQUANT_* / TQ_* env
    var so the rest of the plugin can consume it through the existing
    code paths. ``os.environ.setdefault`` is used throughout, so an
    explicit env-var override always wins over a config-supplied
    default."""
    extra = getattr(engine_args, "additional_config", None) or {}
    if not isinstance(extra, dict):
        return
    tq = extra.get("turboquant")
    if not isinstance(tq, dict):
        return
    unknown = [k for k in tq if k not in _FT_TQ_CONFIG_ENV_MAP]
    if unknown:
        logger.warning(
            "fused-turboquant: ignoring unknown additional_config['turboquant'] "
            "keys: %s. Accepted keys: %s",
            unknown, sorted(_FT_TQ_CONFIG_ENV_MAP.keys()),
        )
    for key, env_name in _FT_TQ_CONFIG_ENV_MAP.items():
        if key not in tq:
            continue
        value = tq[key]
        if value is None:
            continue
        os.environ.setdefault(env_name, _ft_coerce_env_value(value))


def _ft_remap_kv_cache_dtype(value):
    """If `value` is one of our extended `turboquant_k{K}v{V}_nc` aliases,
    pick the smallest stock TurboQuant preset whose K/V slots are wide
    enough to hold the requested K/V bits, and set
    `TURBOQUANT_KEY_BITS` / `TURBOQUANT_VALUE_BITS` so the downstream
    `v1_backend.py` override path realises the effective bit widths.
    Returns the remapped (or unchanged) string."""
    if not isinstance(value, str):
        return value
    m = _FT_KVDTYPE_PATTERN.match(value)
    if not m:
        return value
    k_bits, v_bits = int(m.group(1)), int(m.group(2))
    # Smallest stock preset whose K/V slots fit. Stock presets:
    #   turboquant_3bit_nc  → K slot = 3, V slot = 3
    #   turboquant_k3v4_nc  → K slot = 3, V slot = 4
    #   turboquant_4bit_nc  → K slot = 4, V slot = 4
    if k_bits <= 3 and v_bits <= 3:
        host = "turboquant_3bit_nc"
    elif k_bits <= 3 and v_bits <= 4:
        host = "turboquant_k3v4_nc"
    else:
        host = "turboquant_4bit_nc"
    # Set env vars so v1_backend.py picks them up via the existing
    # TURBOQUANT_KEY_BITS / _VALUE_BITS override path. Don't clobber a
    # value the user already set explicitly.
    os.environ.setdefault("TURBOQUANT_KEY_BITS", str(k_bits))
    os.environ.setdefault("TURBOQUANT_VALUE_BITS", str(v_bits))
    logger.info(
        "fused-turboquant: kv_cache_dtype %s → host %s (effective K=%d, V=%d)",
        value, host, k_bits, v_bits,
    )
    return host


def _patch_extend_tq_presets() -> None:
    """Allow `kv_cache_dtype="turboquant_k{K}v{V}_nc"` (K, V ∈ {1, 2, 3, 4})
    to be passed directly to vLLM as a preset name, instead of forcing the
    user to combine `TURBOQUANT_KEY_BITS` / `TURBOQUANT_VALUE_BITS` env vars
    with a stock `turboquant_*_nc` preset.

    vLLM types `cache_dtype` as a `Literal[...]` and Pydantic enforces it at
    runtime, so we can't just register a new preset name. Instead we hook
    `EngineArgs.__post_init__` and rewrite the value *before* `CacheConfig`
    is built: `turboquant_k3v2_nc` → host preset `turboquant_3bit_nc` plus
    `TURBOQUANT_KEY_BITS=3 TURBOQUANT_VALUE_BITS=2`. The bit-width override
    path inside `v1_backend.py` then activates and the user-visible
    behaviour matches what a real `turboquant_k3v2_nc` preset would do.

    Idempotent.
    """
    try:
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        return

    if getattr(EngineArgs, "_ft_kvdtype_remap_patched", False):
        return

    original = EngineArgs.__post_init__

    def patched(self):
        # Rewrite kv_cache_dtype on the EngineArgs object before its
        # CacheConfig is constructed. Both attribute names that have
        # been used historically are covered.
        for attr in ("kv_cache_dtype", "cache_dtype"):
            current = getattr(self, attr, None)
            if isinstance(current, str):
                remapped = _ft_remap_kv_cache_dtype(current)
                if remapped is not current:
                    setattr(self, attr, remapped)
        # Carry the user-requested `attention_config.flash_attn_version`
        # through to our TURBOQUANT prefill path. vLLM's TurboQuant pin
        # at `arg_utils.py:2003-2014` is about to overwrite anything
        # `>= 3` with `2` (because vLLM's own FA3+ backend asserts
        # FlashAttentionImpl, which TurboQuantAttentionImpl is not), so
        # we capture the user's preference *before* `original(self)`
        # runs and stash it in `TQ_PREFILL_FA_VERSION`. The plugin's
        # per-call `flash_attn_varlen_func(fa_version=...)` reads that
        # env var and routes around the vLLM pin. `setdefault` keeps an
        # explicit env-var override winning over the config field.
        ac = getattr(self, "attention_config", None)
        user_fa = getattr(ac, "flash_attn_version", None) if ac is not None else None
        if user_fa in (3, 4):
            os.environ.setdefault("TQ_PREFILL_FA_VERSION", str(user_fa))
        # Pull every TurboQuant knob out of `additional_config["turboquant"]`
        # so users don't have to set environment variables. Env vars still
        # win (setdefault) so existing scripts keep working.
        _ft_apply_turboquant_additional_config(self)
        original(self)

    patched.__wrapped__ = original  # type: ignore[attr-defined]
    EngineArgs.__post_init__ = patched  # type: ignore[assignment]
    EngineArgs._ft_kvdtype_remap_patched = True  # type: ignore[attr-defined]
    logger.info(
        "fused-turboquant: EngineArgs.__post_init__ now rewrites "
        "turboquant_k{K}v{V}_nc → host preset + bit-width env overrides "
        "(K, V ∈ {1, 2, 3, 4}), and forwards "
        "attention_config.flash_attn_version → TQ_PREFILL_FA_VERSION "
        "so users can pick FA3 / FA4 via the LLM(...) kwarg instead of "
        "an environment variable."
    )


def _patch_cache_dtype_validation() -> None:
    """No-op stub kept for backward-compat. Pydantic Literal validation on
    `CacheConfig.cache_dtype` can't be loosened at runtime without
    rebuilding the dataclass; we go through `_patch_extend_tq_presets`
    instead, which rewrites the alias before it reaches the Literal
    check. Left in place so older `register_backend()` call sites keep
    working."""
    return


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
