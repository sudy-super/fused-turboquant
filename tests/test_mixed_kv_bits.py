"""Tests for the K/V independent bit-width feature in patch_model."""

import pytest
import torch

from fused_turboquant.core.quantizer import TurboQuantMSE
from fused_turboquant.hf.fused_cache import CompressedKVCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCompressedKVCacheMixedBits:
    """CompressedKVCache must route K and V to their respective quantizers."""

    def test_separate_quantizers_per_layer(self):
        device = "cuda"
        torch.manual_seed(0)
        head_dim = 128
        k_tq = TurboQuantMSE(head_dim=head_dim, bits=4, device=device)
        v_tq = TurboQuantMSE(head_dim=head_dim, bits=3, device=device)

        cache = CompressedKVCache(k_tq, compress_v=True)
        cache.set_layer_quantizer(0, k_tq)
        cache.set_layer_quantizer_v(0, v_tq)

        assert cache.get_layer_quantizer(0) is k_tq
        assert cache.get_layer_quantizer_v(0) is v_tq

    def test_v_falls_back_to_k_when_not_set(self):
        device = "cuda"
        head_dim = 128
        k_tq = TurboQuantMSE(head_dim=head_dim, bits=4, device=device)
        cache = CompressedKVCache(k_tq, compress_v=True)
        cache.set_layer_quantizer(0, k_tq)
        # No V-specific set: V should resolve to the K quantizer.
        assert cache.get_layer_quantizer_v(0) is k_tq

    def test_v_decode_uses_v_quantizer(self):
        """When K and V use different bit-widths, decode_values must use V's codebook."""
        device = "cuda"
        torch.manual_seed(0)
        head_dim = 128
        k_tq = TurboQuantMSE(head_dim=head_dim, bits=4, device=device)
        v_tq = TurboQuantMSE(head_dim=head_dim, bits=3, device=device)
        cache = CompressedKVCache(k_tq, compress_v=True)
        cache.set_layer_quantizer(0, k_tq)
        cache.set_layer_quantizer_v(0, v_tq)

        value_states = torch.randn(1, 2, 4, head_dim, device=device)
        cache.store_compressed_value(value_states, 0)
        decoded = cache.decode_values(0)
        assert decoded.shape == value_states.shape
        # 3-bit V should reconstruct the value direction reasonably.
        cos = torch.nn.functional.cosine_similarity(
            value_states.reshape(-1, head_dim).float(),
            decoded.reshape(-1, head_dim).float(),
        ).mean().item()
        assert cos > 0.95, f"V decode cos sim too low: {cos:.4f}"

    def test_k_path_is_independent_of_v_quantizer(self):
        """Setting a V-only quantizer must not affect K storage / lookup."""
        device = "cuda"
        head_dim = 128
        k_tq = TurboQuantMSE(head_dim=head_dim, bits=4, device=device)
        v_tq = TurboQuantMSE(head_dim=head_dim, bits=2, device=device)
        cache = CompressedKVCache(k_tq, compress_v=True)
        cache.set_layer_quantizer(0, k_tq)
        cache.set_layer_quantizer_v(0, v_tq)

        key_states = torch.randn(1, 2, 4, head_dim, device=device)
        cache.store_compressed_key(key_states, 0)
        entry = cache.get_compressed_key(0)
        # K should be packed with 4-bit nibbles: head_dim/2 bytes per vector.
        assert entry["packed_indices"].shape[-1] == head_dim // 2

    def test_packed_dim_differs_for_k_and_v(self):
        """K=4-bit (head_dim/2 bytes) and V=2-bit (head_dim/4 bytes) coexist."""
        device = "cuda"
        head_dim = 128
        k_tq = TurboQuantMSE(head_dim=head_dim, bits=4, device=device)
        v_tq = TurboQuantMSE(head_dim=head_dim, bits=2, device=device)
        cache = CompressedKVCache(k_tq, compress_v=True)
        cache.set_layer_quantizer(0, k_tq)
        cache.set_layer_quantizer_v(0, v_tq)

        ks = torch.randn(1, 2, 4, head_dim, device=device)
        vs = torch.randn(1, 2, 4, head_dim, device=device)
        cache.store_compressed_key(ks, 0)
        cache.store_compressed_value(vs, 0)

        k_entry = cache.get_compressed_key(0)
        v_entry = cache.get_compressed_value(0)
        assert k_entry["packed_indices"].shape[-1] == head_dim // 2  # 4-bit
        assert v_entry["packed_indices"].shape[-1] == head_dim // 4  # 2-bit


def test_patch_model_rejects_bad_v_bits():
    """Sanity: patch_model surfaces the same 2/3/4 constraint on v_bits."""
    from fused_turboquant.hf.fused_cache import patch_model

    class _Dummy:
        pass

    with pytest.raises(ValueError, match="v_bits"):
        patch_model(_Dummy(), bits=4, v_bits=5)
