"""Abstract base for pluggable rotation strategies.

A `RotationStrategy` owns everything that's specific to a particular
orthogonal rotation kind (RHT, Planar, Rotorquant, ...): how to build
the rotation state on a layer, and how to apply the forward rotation
to K (at store time) and Q (at decode time).

The Triton store/decode kernels are rotation-agnostic — they just
bucketize against Lloyd-Max midpoints and compute a dot product
between the (pre-rotated) Q and the centroid-indexed K. So adding a
new rotation kind is a self-contained change: subclass
`RotationStrategy`, implement `setup_layer` / `rotate_for_store` /
`rotate_for_decode`, register the class via `register_rotation`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch


class RotationStrategy(ABC):
    """A pluggable orthogonal rotation for fused-turboquant.

    The contract:
      - `setup_layer` is called once per attention layer at first forward
        time, with the layer's centroid table and the target device. It
        should cache the rotation state on the layer (under attribute
        names of the implementer's choosing, but `_fused_tq_*` is the
        convention).
      - `rotate_for_store` is applied to unit-norm K vectors before the
        MSE bucketize step in the store kernel.
      - `rotate_for_decode` is applied to Q before computing
        `score = q_rot · c_vals` in the decode kernel.

    For most rotations both methods apply the same orthogonal matrix
    (since `q · K = (R·q) · (R·K)` for orthogonal R). Strategies that
    need different rotations for the two phases can override them
    independently — e.g. an asymmetric residual quantizer.
    """

    name: ClassVar[str]

    @abstractmethod
    def setup_layer(
        self,
        layer,
        head_size: int,
        centroids: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        """Build and cache rotation state on `layer`. Idempotent."""

    @abstractmethod
    def rotate_for_store(self, x_normalized: torch.Tensor, layer) -> torch.Tensor:
        """Rotate unit-norm K. Input/output shape `(..., D)`.

        Kept for back-compat and as a fallback path: used by
        `launch_store_external_rotation` when an in-kernel-rotation variant
        is not provided by the strategy.
        """

    @abstractmethod
    def rotate_for_decode(self, q: torch.Tensor, layer) -> torch.Tensor:
        """Rotate Q so the score `q_rot · centroid` recovers `q · K_original`."""

    def launch_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer,
        tq_config,
    ) -> None:
        """Launch the fused store kernel that scatters K and V into the cache.

        Strategies that have an in-kernel-rotation Triton kernel should
        override this method to dispatch to their bespoke kernel.

        The default implementation falls back to the external-rotation path
        (norm + GEMM via `rotate_for_store`, followed by vLLM's stock
        `_tq_fused_store_mse`). This preserves correctness for any
        `RotationStrategy` even if its in-kernel variant has not been
        ported yet.
        """
        from vllm.v1.attention.ops.triton_turboquant_store import (
            _tq_fused_store_mse,
        )
        from vllm.triton_utils import triton

        N, H_kv, D = key.shape
        NH = N * H_kv
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)
        import math

        mse_bytes = math.ceil(D * tq_config.key_mse_bits / 8)
        n_centroids = 2 ** tq_config.key_mse_bits
        val_data_bytes = math.ceil(D * tq_config.effective_value_quant_bits / 8)
        BLOCK_VAL = triton.next_power_of_2(val_data_bytes)
        BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1

        k_flat = key.float().reshape(NH, D)
        norms = k_flat.norm(dim=1, keepdim=True)
        x_hat = k_flat / (norms + 1e-8)
        y = self.rotate_for_store(x_hat, layer)
        v_flat = value.float().reshape(NH, D)

        grid = (NH,)
        _tq_fused_store_mse[grid](
            y,
            norms.squeeze(1),
            v_flat,
            self.get_midpoints(layer),
            kv_cache.view(-1),
            slot_mapping,
            stride_cache_block=kv_cache.stride(0),
            stride_cache_pos=kv_cache.stride(1),
            stride_cache_head=kv_cache.stride(2),
            D=D,
            H=H_kv,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes,
            KPS=tq_config.key_packed_size,
            VQB=tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            MSE_BITS=tq_config.key_mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=BLOCK_GRP,
            num_warps=4,
            num_stages=1,
        )

    def get_centroids(self, layer) -> torch.Tensor:
        """Sorted Lloyd-Max levels. Stored by `setup_layer`."""
        return layer._fused_tq_centroids

    def get_midpoints(self, layer) -> torch.Tensor:
        """Lloyd-Max midpoints (n_centroids-1,) for the store kernel's
        binary-search bucketize."""
        return layer._fused_tq_midpoints


_STRATEGIES: dict[str, type[RotationStrategy]] = {}


def register_rotation(name: str, cls: type[RotationStrategy]) -> None:
    """Register a strategy under `TURBOQUANT_KIND=<name>`. Idempotent
    re-registration overwrites (useful for testing / shadowing)."""
    _STRATEGIES[name] = cls


def get_rotation(name: str) -> RotationStrategy:
    """Instantiate the strategy registered under `name`. Raises with a
    helpful list if the name isn't registered."""
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown rotation kind: {name!r}. "
            f"Registered kinds: {sorted(_STRATEGIES.keys())}"
        )
    return _STRATEGIES[name]()


def available_rotations() -> list[str]:
    return sorted(_STRATEGIES.keys())
