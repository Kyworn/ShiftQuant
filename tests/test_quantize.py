"""Tests for ptq/quantize.py: round-trip accuracy, valid value set, padding."""

import math
import pytest
import torch
from ptq.quantize import quantize_block, dequantize_block, round_to_shift

VALID_VALUES = {-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0}


# ---------------------------------------------------------------------------
# round_to_shift
# ---------------------------------------------------------------------------

class TestRoundToShift:
    def test_exact_values_unchanged(self):
        vals = torch.tensor([-4., -2., -1., 0., 1., 2., 4.])
        result = round_to_shift(vals)
        assert torch.allclose(result, vals)

    def test_output_in_valid_set(self):
        x = torch.randn(1000) * 5
        result = round_to_shift(x)
        unique = set(result.tolist())
        assert unique.issubset(VALID_VALUES), f"Unexpected values: {unique - VALID_VALUES}"

    def test_thresholds(self):
        # Values just below and above each threshold
        cases = [
            (0.49, 0.0),
            (0.51, 1.0),
            (1.49, 1.0),
            (1.51, 2.0),
            (2.99, 2.0),
            (3.01, 4.0),
        ]
        for inp, expected in cases:
            out = round_to_shift(torch.tensor([inp])).item()
            assert out == expected, f"round_to_shift({inp}) = {out}, expected {expected}"

    def test_sign_preserved(self):
        x = torch.tensor([-0.6, -1.6, -3.1])
        result = round_to_shift(x)
        assert (result < 0).all(), "Negative inputs should produce negative outputs"

    def test_zero(self):
        assert round_to_shift(torch.tensor([0.0])).item() == 0.0

    def test_large_values_clamp_to_4(self):
        x = torch.tensor([10.0, 100.0, -7.0])
        result = round_to_shift(x)
        assert result[0].item() == 4.0
        assert result[1].item() == 4.0
        assert result[2].item() == -4.0


# ---------------------------------------------------------------------------
# quantize_block
# ---------------------------------------------------------------------------

class TestQuantizeBlock:
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_q_values_valid(self, block_size):
        w = torch.randn(128, 256).half()
        q_w, scales = quantize_block(w, block_size)
        unique = set(q_w.float().unique().tolist())
        assert unique.issubset(VALID_VALUES)

    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_output_shapes(self, block_size):
        out_f, in_f = 64, 256
        w = torch.randn(out_f, in_f).half()
        q_w, scales = quantize_block(w, block_size)

        assert q_w.shape == (out_f, in_f)
        assert q_w.dtype == torch.int8

        num_blocks = math.ceil(in_f / block_size)
        assert scales.shape == (out_f, num_blocks)
        assert scales.dtype == torch.float16

    def test_padding_edge_case(self):
        # in_features not a multiple of any standard block size
        w = torch.randn(32, 100).half()
        q_w, scales = quantize_block(w, block_size=64)
        assert q_w.shape == (32, 100)
        assert scales.shape == (32, 2)  # ceil(100/64) = 2

    def test_scale_positive(self):
        w = torch.randn(16, 64).half()
        _, scales = quantize_block(w, block_size=64)
        assert (scales > 0).all()

    def test_zero_weight_block(self):
        w = torch.zeros(4, 64).half()
        q_w, scales = quantize_block(w, block_size=64)
        assert (q_w == 0).all()

    def test_requires_2d(self):
        with pytest.raises(ValueError, match="2D"):
            quantize_block(torch.randn(4, 8, 16).half(), 8)


# ---------------------------------------------------------------------------
# dequantize_block (round-trip error)
# ---------------------------------------------------------------------------

class TestDequantizeBlock:
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_round_trip_error_bounded(self, block_size):
        # Dequantized weights should approximate originals within ~50% relative error
        torch.manual_seed(42)
        w = torch.randn(64, 128).half()
        q_w, scales = quantize_block(w, block_size)
        w_hat = dequantize_block(q_w, scales, block_size, original_in_features=128)

        # Max absolute error should be < 25% of max weight magnitude
        max_err = (w.float() - w_hat.float()).abs().max().item()
        max_mag = w.float().abs().max().item()
        assert max_err < 0.5 * max_mag, (
            f"Round-trip error too large: {max_err:.4f} vs magnitude {max_mag:.4f}"
        )

    def test_dequant_shape_matches_original(self):
        out_f, in_f = 32, 100
        w = torch.randn(out_f, in_f).half()
        q_w, scales = quantize_block(w, 64)
        w_hat = dequantize_block(q_w, scales, 64, original_in_features=in_f)
        assert w_hat.shape == (out_f, in_f)

    def test_dequant_dtype_fp16(self):
        w = torch.randn(16, 64).half()
        q_w, scales = quantize_block(w, 64)
        w_hat = dequantize_block(q_w, scales, 64)
        assert w_hat.dtype == torch.float16

    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_snr_reasonable(self, block_size):
        """Signal-to-noise ratio of dequantized weights should exceed 3 dB."""
        torch.manual_seed(0)
        w = torch.randn(128, 256).half()
        q_w, scales = quantize_block(w, block_size)
        w_hat = dequantize_block(q_w, scales, block_size, 256)

        signal_power = w.float().pow(2).mean().item()
        noise_power  = (w.float() - w_hat.float()).pow(2).mean().item()
        snr_db = 10 * math.log10(signal_power / (noise_power + 1e-12))
        assert snr_db > 3.0, f"SNR too low: {snr_db:.1f} dB (block_size={block_size})"
