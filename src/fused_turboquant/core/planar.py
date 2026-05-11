"""
PlanarQuant: 2D Givens-rotation-based quantizer.

Parallel to TurboQuant_MSE (RHT), but uses an SO(2) rotation per (d/2) pair
of coordinates instead of a O(d log d) Hadamard butterfly. Per-pair rotation
costs only 4 FMAs and the rotation state is 2 floats per pair — the lightest
member of the rotation-based quantizer family.

Pipeline (unfused fallback):
    Encode: x -> Planar rotate -> norm -> normalize -> Lloyd-Max quantize -> pack
    Decode: unpack -> dequantize -> denormalize -> inverse Planar rotate

Fused pipeline (Triton, automatic on CUDA):
    Encode: single kernel — rotate + norm + quantize + pack
    Decode: single kernel — unpack + dequant + denorm + inverse rotate

Reference: scrya-com/rotorquant — turboquant/planarquant.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from fused_turboquant.core.lloyd_max import LloydMaxQuantizer
from fused_turboquant.core.packing import (
    pack_2bit,
    pack_3bit,
    pack_nibbles,
    unpack_2bit,
    unpack_3bit,
    unpack_nibbles,
)
from fused_turboquant.core.quantizer import CompressedTensor


def _validate_even(d: int, name: str = "dim") -> None:
    if d < 2 or d % 2 != 0:
        raise ValueError(f"{name} must be a positive even integer, got {d}")


def _is_power_of_2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def generate_planar_rot2(
    dim: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate the (cos θ, sin θ) table for PlanarQuant 2D rotations.

    The d-dim vector is split into n_groups = d/2 disjoint pairs. Each pair
    gets its own random angle θ in [0, 2π) and contributes a single Givens
    rotation. The resulting rotation matrix is block-diagonal with 2x2 blocks.

    Returns: tensor of shape (n_groups, 2) holding [cos θ, sin θ] per group.
    """
    _validate_even(dim)
    n_groups = dim // 2
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # device="cpu" explicit — see generate_rht_signs for why.
    angles = torch.rand(n_groups, generator=gen, device="cpu") * (2.0 * math.pi)
    cos = angles.cos().to(torch.float32)
    sin = angles.sin().to(torch.float32)
    return torch.stack([cos, sin], dim=-1).to(device)


def planar_rotate(x: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """Apply per-pair 2D Givens rotation to vectors of shape (..., d).

    Each consecutive pair (v0, v1) is replaced by
        ( cos θ · v0 - sin θ · v1,  sin θ · v0 + cos θ · v1 ).
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"last dim must be even, got {d}")
    pairs = x.reshape(*x.shape[:-1], d // 2, 2)
    v0 = pairs[..., 0]
    v1 = pairs[..., 1]
    c = rot2[..., 0]
    s = rot2[..., 1]
    r0 = c * v0 - s * v1
    r1 = s * v0 + c * v1
    return torch.stack([r0, r1], dim=-1).reshape(*x.shape)


def planar_rotate_inverse(y: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """Inverse rotation (Givens matrices are orthogonal: R^{-1} = R^T, i.e. negate sin)."""
    d = y.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"last dim must be even, got {d}")
    pairs = y.reshape(*y.shape[:-1], d // 2, 2)
    v0 = pairs[..., 0]
    v1 = pairs[..., 1]
    c = rot2[..., 0]
    s = rot2[..., 1]
    r0 = c * v0 + s * v1
    r1 = -s * v0 + c * v1
    return torch.stack([r0, r1], dim=-1).reshape(*y.shape)


class PlanarRotation(nn.Module):
    """Persistent 2D Givens rotation layer.

    Mirrors the public API of `RHTRotation`: forward/inverse/to/extra_repr,
    so PlanarQuant can drop into the same encode/decode template as the RHT
    quantizer. Storage per layer: 2 floats per pair = 4·(d/2) bytes.
    """

    def __init__(self, dim: int, seed: int = 0, device: torch.device | str = "cpu"):
        super().__init__()
        _validate_even(dim, "dim")
        self.dim = dim
        self.n_groups = dim // 2
        self.register_buffer("rot2", generate_planar_rot2(dim, seed=seed, device=device))
        self._use_triton = False
        self._try_enable_triton()

    def _try_enable_triton(self) -> None:
        # The standalone Planar rotate kernel is not provided as a separate
        # primitive — the fused encode/decode kernels do rotation inline.
        # Inverse-rotation in the unfused decode path is fast in PyTorch.
        self._use_triton = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return planar_rotate(x, self.rot2)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return planar_rotate_inverse(y, self.rot2)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result._try_enable_triton()
        return result

    def extra_repr(self) -> str:
        bytes_per_layer = self.n_groups * 2 * 4
        return f"dim={self.dim}, n_groups={self.n_groups}, storage={bytes_per_layer} bytes"


class PlanarQuantMSE:
    """
    PlanarQuant_MSE: full encode/decode pipeline with 2D Givens rotation.

    Drop-in alternative to `TurboQuantMSE` (RHT). The public surface is
    intentionally identical so the same `CompressedTensor` flows through the
    rest of the stack (KV cache, packing, attention kernel).

    Usage:
        pq = PlanarQuantMSE(head_dim=256, bits=4, device="cuda")
        compressed = pq.encode(keys)
        decoded = pq.decode(compressed)
    """

    def __init__(
        self,
        head_dim: int,
        bits: int = 4,
        seed: int = 42,
        device: torch.device | str = "cpu",
        max_iterations: int = 300,
        num_grid_points: int = 50000,
    ):
        _validate_even(head_dim, "head_dim")
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")

        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        self.rotation = PlanarRotation(head_dim, seed=seed, device=device)
        self.quantizer = LloydMaxQuantizer(
            head_dim,
            bits=bits,
            device=device,
            max_iterations=max_iterations,
            num_grid_points=num_grid_points,
        )

        self._use_fused_triton = False
        self._try_enable_fused_triton()

    def _try_enable_fused_triton(self) -> None:
        try:
            from fused_turboquant.kernels.triton_planar_encode import is_triton_available
        except ImportError:
            return
        if is_triton_available() and str(self.device).startswith("cuda"):
            self._use_fused_triton = True

    # -- Encode ---------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> CompressedTensor:
        if self._use_fused_triton and x.is_cuda and _is_power_of_2(self.head_dim // 2):
            return self._encode_fused(x)
        return self._encode_unfused(x)

    def _encode_fused(self, x: torch.Tensor) -> CompressedTensor:
        from fused_turboquant.kernels.triton_planar_encode import (
            triton_planar_fused_encode,
        )

        packed, norms = triton_planar_fused_encode(
            x,
            self.rotation.rot2,
            self.quantizer.boundaries,
            self.bits,
        )
        return CompressedTensor(
            indices=packed,
            norms=norms,
            original_dim=self.head_dim,
            bits=self.bits,
        )

    def _encode_unfused(self, x: torch.Tensor) -> CompressedTensor:
        x = x.float()
        rotated = self.rotation(x)

        norms = torch.norm(rotated, dim=-1, keepdim=True)
        normalized = rotated / (norms + 1e-8)

        indices = self.quantizer.quantize(normalized)

        if self.bits == 4:
            packed = pack_nibbles(indices)
        elif self.bits == 3:
            packed = pack_3bit(indices)
        elif self.bits == 2:
            packed = pack_2bit(indices)
        else:
            packed = indices

        return CompressedTensor(
            indices=packed,
            norms=norms.squeeze(-1).half(),
            original_dim=self.head_dim,
            bits=self.bits,
        )

    # -- Decode ---------------------------------------------------------------

    def decode(self, compressed: CompressedTensor) -> torch.Tensor:
        if (
            self._use_fused_triton
            and compressed.indices.is_cuda
            and _is_power_of_2(self.head_dim // 2)
        ):
            return self._decode_fused(compressed)
        return self._decode_unfused(compressed)

    def _decode_fused(self, compressed: CompressedTensor) -> torch.Tensor:
        from fused_turboquant.kernels.triton_planar_decode import (
            triton_planar_fused_decode,
        )

        return triton_planar_fused_decode(
            compressed.indices,
            compressed.norms,
            self.quantizer.levels,
            self.rotation.rot2,
            compressed.bits,
            compressed.original_dim,
        )

    def _decode_unfused(self, compressed: CompressedTensor) -> torch.Tensor:
        if compressed.bits == 4:
            indices = unpack_nibbles(compressed.indices, compressed.original_dim)
        elif compressed.bits == 3:
            indices = unpack_3bit(compressed.indices, compressed.original_dim)
        elif compressed.bits == 2:
            indices = unpack_2bit(compressed.indices, compressed.original_dim)
        else:
            indices = compressed.indices

        reconstructed = self.quantizer.dequantize(indices)
        reconstructed = reconstructed * compressed.norms.float().unsqueeze(-1)
        decoded = self.rotation.inverse(reconstructed)

        return decoded

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def to(self, device: torch.device | str) -> "PlanarQuantMSE":
        self.device = device
        self.rotation = self.rotation.to(device)
        self.quantizer = self.quantizer.to(device)
        self._try_enable_fused_triton()
        return self
