"""
Drop-in replacement for nn.Linear that stores weights in shift-quantized form.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .quantize import (
    quantize_block, quantize_block_mse,
    quantize_block_B, quantize_block_C, quantize_block_9val,
    dequantize_block,
)
from .shift_matmul import shift_matmul


class QuantizedLinear(nn.Module):
    """
    nn.Linear replacement with weights quantized to {-4,-2,-1,0,+1,+2,+4}.

    Buffers:
        q_weight: int8 [out_features, in_features]
        scales:   FP16 [out_features, num_blocks]
        bias:     FP16 [out_features] or absent

    Memory vs FP16 nn.Linear:
        block_size=32:  ~1.06 bytes/weight  (vs 2 bytes)
        block_size=64:  ~1.03 bytes/weight
        block_size=128: ~1.02 bytes/weight
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        num_blocks = (in_features + block_size - 1) // block_size

        self.register_buffer(
            "q_weight",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(out_features, num_blocks, dtype=torch.float16),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16),
            )
        else:
            self.bias = None  # type: ignore[assignment]

    # Supported grid names → quantize function (weight stays on device for gpu grids)
    _GRID_FNS = {
        "A":  (quantize_block,      "cpu"),
        "B":  (quantize_block_B,    "cpu"),
        "C":  (quantize_block_C,    "cpu"),
        "9v": (quantize_block_9val, "cpu"),
    }

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 64,
        grid: str = "A",
        calibrated: bool = False,
        n_candidates: int = 100,
    ) -> "QuantizedLinear":
        """
        Convert an nn.Linear to QuantizedLinear.

        Args:
            grid:          "A" (default 7v log), "B" (7v uniform), "C" (8v asymm), "9v" (9v uniform).
            calibrated:    If True, override grid with MSE-optimal scale search on GPU.
            n_candidates:  Grid resolution (only if calibrated=True).
        """
        has_bias = linear.bias is not None
        device = linear.weight.device
        ql = cls(linear.in_features, linear.out_features, bias=has_bias, block_size=block_size)

        if calibrated:
            q_w, scales = quantize_block_mse(
                linear.weight.data.float(), block_size, n_candidates=n_candidates
            )
            q_w, scales = q_w.cpu(), scales.cpu()
        else:
            fn, dev = cls._GRID_FNS[grid]
            w = linear.weight.data.float().to(dev)
            q_w, scales = fn(w, block_size)

        ql.q_weight.copy_(q_w.to(device))
        ql.scales.copy_(scales.to(device))
        if has_bias:
            ql.bias.copy_(linear.bias.data.half().to(device))  # type: ignore[union-attr]
        return ql.to(device)

    def forward(self, x: Tensor) -> Tensor:
        # shift_matmul returns float32; cast back to input dtype
        out = shift_matmul(x, self.q_weight, self.scales, self.block_size)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(x.dtype)

    def dequantized_weight(self) -> Tensor:
        """Return approximate FP16 weight for inspection or validation."""
        return dequantize_block(
            self.q_weight,
            self.scales,
            self.block_size,
            original_in_features=self.in_features,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"block_size={self.block_size}"
        )
