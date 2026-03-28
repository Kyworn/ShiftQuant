"""
Tests for ptq/shift_matmul.py:
  - shift_matmul (einsum) matches F.linear(x, dequantized_weight)
  - shift_matmul_pure matches shift_matmul
  - Correct output shapes for various batch dims
"""

import pytest
import torch
import torch.nn.functional as F

from ptq.quantize import quantize_block, dequantize_block
from ptq.shift_matmul import shift_matmul, shift_matmul_pure


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def make_quantized(out_f: int, in_f: int, block_size: int, device: str):
    torch.manual_seed(7)
    w = torch.randn(out_f, in_f, device=device).half()
    q_w, scales = quantize_block(w, block_size)
    q_w = q_w.to(device)
    scales = scales.to(device)
    w_hat = dequantize_block(q_w, scales, block_size, in_f).to(device)
    return q_w, scales, w_hat


class TestShiftMatmulEinsum:
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_fp16_linear(self, block_size, device):
        out_f, in_f = 64, 128
        q_w, scales, w_hat = make_quantized(out_f, in_f, block_size, device)

        x = torch.randn(4, in_f, device=device)

        ref = F.linear(x.float(), w_hat.float())
        out = shift_matmul(x, q_w, scales, block_size)

        assert out.shape == ref.shape
        # Should match dequant reference to floating-point precision
        assert torch.allclose(out, ref, atol=1e-3), (
            f"Max diff: {(out - ref).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_shape_2d(self, device):
        q_w, scales, _ = make_quantized(32, 64, 64, device)
        x = torch.randn(8, 64, device=device)
        out = shift_matmul(x, q_w, scales, 64)
        assert out.shape == (8, 32)

    @pytest.mark.parametrize("device", DEVICES)
    def test_output_shape_3d(self, device):
        q_w, scales, _ = make_quantized(32, 64, 64, device)
        x = torch.randn(2, 8, 64, device=device)
        out = shift_matmul(x, q_w, scales, 64)
        assert out.shape == (2, 8, 32)

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_multiple_in_features(self, device):
        out_f, in_f = 32, 100
        q_w, scales, w_hat = make_quantized(out_f, in_f, 64, device)
        x = torch.randn(5, in_f, device=device)
        out = shift_matmul(x, q_w, scales, 64)
        ref = F.linear(x.float(), w_hat.float())
        assert torch.allclose(out, ref, atol=1e-3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_input(self, device):
        q_w, scales, _ = make_quantized(16, 32, 32, device)
        x = torch.zeros(4, 32, device=device)
        out = shift_matmul(x, q_w, scales, 32)
        assert torch.all(out == 0)


class TestShiftMatmulPure:
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_einsum_version(self, block_size, device):
        out_f, in_f = 48, 96
        q_w, scales, _ = make_quantized(out_f, in_f, block_size, device)
        x = torch.randn(6, in_f, device=device)

        out_einsum = shift_matmul(x, q_w, scales, block_size)
        out_pure   = shift_matmul_pure(x, q_w, scales, block_size)

        assert torch.allclose(out_einsum, out_pure, atol=1e-4), (
            f"Einsum vs pure max diff: {(out_einsum - out_pure).abs().max().item():.8f}"
        )

    @pytest.mark.parametrize("device", DEVICES)
    def test_matches_fp16_linear(self, device):
        out_f, in_f = 32, 64
        q_w, scales, w_hat = make_quantized(out_f, in_f, 64, device)
        x = torch.randn(4, in_f, device=device)

        ref  = F.linear(x.float(), w_hat.float())
        out  = shift_matmul_pure(x, q_w, scales, 64)
        assert torch.allclose(out, ref, atol=1e-3)

    @pytest.mark.parametrize("device", DEVICES)
    def test_3d_batch(self, device):
        q_w, scales, _ = make_quantized(32, 64, 64, device)
        x = torch.randn(2, 8, 64, device=device)
        out = shift_matmul_pure(x, q_w, scales, 64)
        assert out.shape == (2, 8, 32)
