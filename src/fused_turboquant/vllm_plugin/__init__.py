"""
vLLM plugin for fused-turboquant attention backend.

Provides the FUSED_TURBOQUANT attention backend that stores KV cache
in compressed packed uint8 format, achieving 3.8-7.1x memory reduction.

Usage:
    vllm serve Qwen/Qwen3-8B --attention-backend FUSED_TURBOQUANT

Configuration (environment variables):
    TURBOQUANT_BITS=4           Quantization bits for K (2, 3, or 4)
    TURBOQUANT_V_BITS=          Optional separate V-cache bit-width (defaults
                                to TURBOQUANT_BITS). Lets you run mixed
                                K/V precision, e.g. TURBOQUANT_BITS=4
                                TURBOQUANT_V_BITS=3.
    TURBOQUANT_COMPRESS_V=1     Compress values (1=yes, 0=K-only)
    TURBOQUANT_KIND=rht         Rotation kind: 'rht' (Randomized Hadamard
                                Transform, requires power-of-2 head_dim) or
                                'planar' (2D Givens rotation, requires only
                                even head_dim).
"""

from fused_turboquant.vllm_plugin.plugin import register_backend

__all__ = ["register_backend"]

# Auto-register when this subpackage is imported. vLLM child processes
# (EngineCore) inherit the parent's site-packages but each one constructs
# its own backend selector, so we need register_backend() to fire in every
# process where the plugin is touched. The general_plugins entry point
# fires the same function via importlib metadata, so calling it again here
# is idempotent — vLLM's register_backend just overwrites the
# AttentionBackendEnum.TURBOQUANT slot mapping.
try:
    register_backend()
except Exception:
    import logging
    logging.getLogger(__name__).debug(
        "auto-register failed (vLLM may not be installed)", exc_info=True
    )
