"""
Integration tests: quantize a small transformer, verify forward pass works.
"""

import pytest
import torch
import torch.nn as nn

from ptq.quantized_linear import QuantizedLinear
from ptq.model_wrapper import quantize_model
from ptq.utils import compute_memory_footprint


class TinyMLP(nn.Module):
    """A minimal two-layer MLP for testing."""
    def __init__(self, d: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(torch.relu(self.fc1(x))))


class TinyTransformerBlock(nn.Module):
    """Minimal transformer block with attention-like projections."""
    def __init__(self, d: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate   = nn.Linear(d, d * 2, bias=True)
        self.down   = nn.Linear(d * 2, d, bias=True)
        self.norm   = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        mlp  = self.down(torch.relu(self.gate(x)))
        return self.norm(x + attn + mlp)


class TestQuantizedLinear:
    def test_from_linear_basic(self):
        linear = nn.Linear(128, 64)
        ql = QuantizedLinear.from_linear(linear, block_size=64)
        assert ql.q_weight.dtype == torch.int8
        assert ql.scales.dtype == torch.float16
        assert ql.q_weight.shape == (64, 128)
        assert ql.scales.shape == (64, 2)  # 128 / 64 = 2 blocks

    def test_forward_output_shape(self):
        linear = nn.Linear(128, 64)
        ql = QuantizedLinear.from_linear(linear, block_size=64)
        x = torch.randn(4, 128)
        out = ql(x)
        assert out.shape == (4, 64)

    def test_forward_no_nan(self):
        linear = nn.Linear(256, 128)
        ql = QuantizedLinear.from_linear(linear, block_size=64)
        x = torch.randn(8, 256)
        out = ql(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_dtype_preserved(self):
        linear = nn.Linear(64, 32)
        ql = QuantizedLinear.from_linear(linear, block_size=32)
        x = torch.randn(2, 64).half()
        out = ql(x)
        assert out.dtype == torch.float16

    def test_bias_none(self):
        linear = nn.Linear(64, 32, bias=False)
        ql = QuantizedLinear.from_linear(linear, block_size=32)
        assert ql.bias is None
        x = torch.randn(4, 64)
        out = ql(x)
        assert out.shape == (4, 32)

    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_output_close_to_fp16(self, block_size):
        torch.manual_seed(42)
        linear = nn.Linear(128, 64)
        ql = QuantizedLinear.from_linear(linear, block_size=block_size)

        x = torch.randn(8, 128)
        ref = linear(x)
        out = ql(x)

        # The quantized output should be in the right ballpark
        # (not exact, but not wildly off)
        rel_err = (ref - out).abs().mean() / (ref.abs().mean() + 1e-8)
        assert rel_err < 0.5, f"Relative error too large: {rel_err:.4f}"


class TestQuantizeModel:
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_linear_layers_replaced(self, block_size):
        model = TinyTransformerBlock(d=64)
        quantize_model(model, block_size=block_size)

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                pytest.fail(f"nn.Linear not replaced: {name}")

        assert isinstance(model.q_proj, QuantizedLinear)
        assert isinstance(model.gate, QuantizedLinear)

    def test_layernorm_not_replaced(self):
        model = TinyMLP(d=64)
        quantize_model(model)
        assert isinstance(model.norm, nn.LayerNorm)

    def test_skip_names_respected(self):
        model = TinyMLP(d=64)
        quantize_model(model, skip_names={"fc1"})
        assert isinstance(model.fc1, nn.Linear), "fc1 should be skipped"
        assert isinstance(model.fc2, QuantizedLinear), "fc2 should be quantized"

    def test_forward_pass_after_quantization(self):
        model = TinyTransformerBlock(d=64)
        quantize_model(model, block_size=64)
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()

    def test_forward_pass_mlp(self):
        model = TinyMLP(d=64)
        quantize_model(model, block_size=64)
        x = torch.randn(4, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 64)


class TestMemoryFootprint:
    def test_footprint_decreases_after_quantization(self):
        model = TinyTransformerBlock(d=64)
        fp_before = compute_memory_footprint(model)
        quantize_model(model, block_size=64)
        fp_after = compute_memory_footprint(model)

        # Quantized model should use less or equal memory
        assert fp_after["total_bytes"] <= fp_before["total_bytes"]

    def test_compression_ratio_above_one(self):
        model = TinyMLP(d=256)
        quantize_model(model, block_size=64)
        fp = compute_memory_footprint(model)
        assert fp["compression_ratio"] > 1.0, (
            f"Expected compression > 1, got {fp['compression_ratio']:.2f}"
        )

    def test_footprint_fields_present(self):
        model = TinyMLP(d=64)
        quantize_model(model, block_size=64)
        fp = compute_memory_footprint(model)
        for key in ("total_bytes", "total_mb", "fp16_equiv_bytes", "compression_ratio", "layers"):
            assert key in fp
