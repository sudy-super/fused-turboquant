"""RHT (Randomized Hadamard Transform) without the random sign flips.

The stock vLLM TurboQuant uses a pure Hadamard matrix (no signs). Per
its source comment, random sign flips don't improve Lloyd-Max
quantization quality because the quantizer is symmetric around zero
(sign-flipping a coordinate maps it to the mirror centroid with
identical distortion). We follow the same convention here so we can
share the stock Triton store/decode kernels — the kernels expect a
single rotation matrix `Pi` (and `Pi.T`, which equals `Pi` for
Hadamard).

The Hadamard matrix is built lazily once per (head_size, device) and
cached at module scope; multiple layers with the same head_size share
the same matrix tensor on the same device.
"""

from __future__ import annotations

import functools
import math

import torch

from .base import register_rotation
from .matrix import MatrixRotationStrategy


@functools.cache
def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Sylvester construction. 64 KB for d=128, 16 MB for d=2048."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class HadamardStrategy(MatrixRotationStrategy):
    name = "rht"

    def build_matrix(self, head_size, device):
        return _build_hadamard(head_size, str(torch.device(device)))

    def launch_store(self, key, value, kv_cache, slot_mapping, layer, tq_config):
        """In-kernel FWHT (Sylvester butterfly) — skips the (D, D) matmul
        entirely. The butterfly is O(D log D) FMAs with no random sign
        flips, which is exactly what the Sylvester construction produces.
        """
        from fused_turboquant.vllm_plugin.triton_inkernel_store import _launch_rht

        _launch_rht(
            key=key,
            value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            midpoints=self.get_midpoints(layer),
            tq_config=tq_config,
            head_size=key.shape[-1],
        )


register_rotation(HadamardStrategy.name, HadamardStrategy)
