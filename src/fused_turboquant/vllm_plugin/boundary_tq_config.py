"""Wrapper config for boundary-protection layers that supports both
FP8 keys and 8-bit MSE keys.

vLLM's stock `TurboQuantConfig` treats `key_quant_bits == 8` as FP8 by
hard-coded property. Our boundary-protection path wants finer control:
either FP8 (1 byte per element, no LUT) OR 8-bit MSE (Lloyd-Max with
256 centroids, still 1 byte per element). This duck-typed replacement
exposes the same property interface that the launchers / decode kernel
call sites read, with an extra `key_use_fp8_at_8bit` field.

Use only for boundary layers (`FusedTurboQuantV1Impl._boundary_tq_bits`
path). Other layers keep using vLLM's stock TurboQuantConfig.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoundaryTurboQuantConfig:
    """Minimal duck-typed TurboQuantConfig for boundary layers."""

    head_dim: int
    key_quant_bits: int  # 2..4 = MSE, 8 = MSE OR FP8 per key_use_fp8_at_8bit
    value_quant_bits: int  # 2..4, 8 for value uniform quantization
    norm_correction: bool = True
    key_use_fp8_at_8bit: bool = False  # default: 8-bit means MSE here
    seed: int = 42  # unused, but referenced by some legacy paths

    # ── Mode flags ─────────────────────────────────────────────────────
    @property
    def key_fp8(self) -> bool:
        return self.key_quant_bits == 8 and self.key_use_fp8_at_8bit

    # ── Derived bit-widths ─────────────────────────────────────────────
    @property
    def mse_bits(self) -> int:
        """Centroid table size = 2**mse_bits. For FP8 keys this is the V
        bit-width (centroids still needed for V dequant)."""
        if self.key_fp8:
            return self.value_quant_bits
        return self.key_quant_bits

    @property
    def key_mse_bits(self) -> int:
        """0 in FP8 mode, else equals key_quant_bits (2..8)."""
        if self.key_fp8:
            return 0
        return self.key_quant_bits

    @property
    def centroid_bits(self) -> int:
        return self.mse_bits

    @property
    def n_centroids(self) -> int:
        return 2 ** self.mse_bits

    @property
    def effective_value_quant_bits(self) -> int:
        return self.value_quant_bits

    # ── Cache slot layout ──────────────────────────────────────────────
    @property
    def key_packed_size(self) -> int:
        """Bytes for a single key vector inside one slot. FP8: head_dim
        bytes (1 byte per element). MSE: ceil(head_dim * mse_bits / 8)
        for indices + 2 bytes for the per-vector fp16 norm."""
        if self.key_fp8:
            return self.head_dim
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        return mse_bytes + 2

    @property
    def value_data_bytes(self) -> int:
        """Bytes for the per-vector value indices (no scale/zero)."""
        if self.value_quant_bits == 8:
            return self.head_dim
        return math.ceil(self.head_dim * self.value_quant_bits / 8)
