"""Tests for the PlanarQuant_MSE quantizer pipeline."""

import pytest
import torch

from fused_turboquant.core.planar import (
    PlanarQuantMSE,
    PlanarRotation,
    planar_rotate,
    planar_rotate_inverse,
)


class TestPlanarRotation:
    def test_rejects_odd_dim(self):
        with pytest.raises(ValueError):
            PlanarRotation(dim=63)

    def test_preserves_norm(self, device):
        rot = PlanarRotation(dim=256, seed=7, device=device)
        x = torch.randn(64, 256, device=device)
        y = rot(x)
        assert y.shape == x.shape
        torch.testing.assert_close(
            torch.norm(x, dim=-1),
            torch.norm(y, dim=-1),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_inverse_roundtrip(self, device):
        rot = PlanarRotation(dim=128, seed=11, device=device)
        x = torch.randn(32, 128, device=device)
        y = rot(x)
        x_back = rot.inverse(y)
        torch.testing.assert_close(x, x_back, rtol=1e-5, atol=1e-5)

    def test_function_form_matches_module(self, device):
        rot = PlanarRotation(dim=128, seed=3, device=device)
        x = torch.randn(8, 128, device=device)
        y_mod = rot(x)
        y_fn = planar_rotate(x, rot.rot2)
        torch.testing.assert_close(y_mod, y_fn)
        x_back_mod = rot.inverse(y_mod)
        x_back_fn = planar_rotate_inverse(y_fn, rot.rot2)
        torch.testing.assert_close(x_back_mod, x_back_fn)


class TestPlanarQuantMSE:
    def test_roundtrip_quality_4bit(self, device):
        """4-bit PlanarQuant should achieve >0.99 cosine similarity."""
        pq = PlanarQuantMSE(head_dim=256, bits=4, device=device)
        x = torch.randn(64, 256, device=device)

        x_hat = pq.roundtrip(x)
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xhat_norm = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
        cosine = torch.mean(torch.sum(x_norm * xhat_norm, dim=-1)).item()
        assert cosine > 0.99, f"4-bit cosine similarity {cosine:.4f} < 0.99"

    def test_roundtrip_quality_3bit(self, device):
        pq = PlanarQuantMSE(head_dim=256, bits=3, device=device)
        x = torch.randn(64, 256, device=device)
        x_hat = pq.roundtrip(x)
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xhat_norm = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
        cosine = torch.mean(torch.sum(x_norm * xhat_norm, dim=-1)).item()
        assert cosine > 0.97, f"3-bit cosine similarity {cosine:.4f} < 0.97"

    def test_roundtrip_quality_2bit(self, device):
        pq = PlanarQuantMSE(head_dim=256, bits=2, device=device)
        x = torch.randn(64, 256, device=device)
        x_hat = pq.roundtrip(x)
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        xhat_norm = x_hat / (torch.norm(x_hat, dim=-1, keepdim=True) + 1e-8)
        cosine = torch.mean(torch.sum(x_norm * xhat_norm, dim=-1)).item()
        assert cosine > 0.90, f"2-bit cosine similarity {cosine:.4f} < 0.90"

    @pytest.mark.parametrize("bits", [4, 3, 2])
    @pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
    def test_compressed_tensor_shapes(self, device, head_dim, bits):
        pq = PlanarQuantMSE(head_dim=head_dim, bits=bits, device=device)
        x = torch.randn(16, head_dim, device=device)
        compressed = pq.encode(x)
        if bits == 4:
            expected_packed = head_dim // 2
        elif bits == 3:
            expected_packed = head_dim * 3 // 8
        elif bits == 2:
            expected_packed = head_dim // 4
        assert compressed.indices.shape == (16, expected_packed)
        assert compressed.norms.shape == (16,)
        assert compressed.original_dim == head_dim
        assert compressed.bits == bits

    def test_supports_non_power_of_2_even_dim(self, device):
        """PlanarQuant only requires an even head_dim, unlike RHT."""
        pq = PlanarQuantMSE(head_dim=192, bits=4, device=device)
        x = torch.randn(32, 192, device=device)
        x_hat = pq.roundtrip(x)
        assert x_hat.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
class TestPlanarQuantFusedVsUnfused:
    """The fused Triton encode/decode must agree with the PyTorch fallback."""

    @pytest.mark.parametrize("bits", [4, 3, 2])
    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_fused_matches_unfused(self, head_dim, bits):
        device = "cuda"
        torch.manual_seed(7)
        pq = PlanarQuantMSE(head_dim=head_dim, bits=bits, device=device)
        x = torch.randn(32, head_dim, device=device)

        pq._use_fused_triton = False
        c_unf = pq.encode(x)
        x_unf = pq.decode(c_unf)

        pq._try_enable_fused_triton()
        assert pq._use_fused_triton, "Fused Triton path should be enabled on CUDA"
        c_fused = pq.encode(x)
        x_fused = pq.decode(c_fused)

        # Same packed indices (the per-element quantization decision is
        # deterministic given the same rot2 / boundaries) and same norms.
        torch.testing.assert_close(c_fused.indices, c_unf.indices)
        torch.testing.assert_close(c_fused.norms, c_unf.norms)
        torch.testing.assert_close(x_fused, x_unf, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
class TestPlanarAttentionKernel:
    """fused_qk_scores_planar should match the explicit Q · K^T from decoded keys."""

    def test_dot_matches_explicit(self):
        from fused_turboquant.kernels.triton_planar_attention import (
            fused_qk_scores_planar,
            planar_rotate_query,
        )

        device = "cuda"
        torch.manual_seed(0)

        head_dim = 128
        bits = 4
        n_q_heads = 4
        n_kv_heads = 2
        kv_len = 16
        scale = 0.1

        pq = PlanarQuantMSE(head_dim=head_dim, bits=bits, device=device)

        # Build a fake K cache via per-token encode (mirrors what
        # CompressedKVCache.store_compressed_key does).
        k_raw = torch.randn(1, n_kv_heads, kv_len, head_dim, device=device)
        compressed = pq.encode(k_raw.float())
        packed_indices = compressed.indices
        norms = compressed.norms

        # Query: build the rotated version expected by the kernel
        q = torch.randn(1, n_q_heads, 1, head_dim, device=device)
        q_flat = q.float().reshape(-1, head_dim)
        q_rot = planar_rotate_query(q_flat, pq.rotation.rot2).view_as(q)

        scores = fused_qk_scores_planar(
            q_rot,
            packed_indices,
            norms,
            pq.quantizer.levels,
            scale,
            bits=bits,
        )

        # Explicit reference: decode K and compute Q · K^T
        decoded_k = pq.decode(compressed)  # [1, n_kv_heads, kv_len, head_dim]
        # GQA broadcast: each Q head shares a KV head
        gqa_ratio = n_q_heads // n_kv_heads
        decoded_k_expanded = decoded_k.repeat_interleave(gqa_ratio, dim=1)
        # Q is in original (non-rotated) basis here; the kernel computed
        # <P(q), centroids[idx]> * norm which equals <q, K_decoded>.
        ref = (q.float() * decoded_k_expanded.float()).sum(dim=-1, keepdim=True) * scale
        ref = ref.transpose(-1, -2).reshape(1, n_q_heads, 1, kv_len)

        torch.testing.assert_close(scores, ref, rtol=1e-3, atol=1e-3)
