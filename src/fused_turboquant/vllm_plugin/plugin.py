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

logger = logging.getLogger(__name__)


def register_backend() -> None:
    """vLLM plugin entry point — called automatically when fused-turboquant is installed.

    Tries to register on both v0 and v1 attention registries. Failures on
    one path are non-fatal: as long as one succeeds, the backend is usable
    on the installed vLLM.
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
