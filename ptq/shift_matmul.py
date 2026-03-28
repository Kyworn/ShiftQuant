"""
Shift-based matrix multiplication for quantized weights.

Two implementations:
  - shift_matmul:      Einsum-based reference (fast, used for PPL benchmarking).
  - shift_matmul_pure: True shift-and-accumulate (proof-of-concept, no multiply
                       on weights — uses torch.ldexp to implement x * 2^k).

Both are mathematically equivalent. The einsum version is faster in PyTorch
because it avoids Python-level loops and mask allocations. A future CUDA kernel
would make shift_matmul_pure the fast path.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def _pad_and_reshape(
    tensor: Tensor,
    in_f: int,
    block_size: int,
) -> tuple[Tensor, int]:
    """Pad last dim to block_size multiple, reshape to (..., num_blocks, block_size)."""
    pad_len = (block_size - in_f % block_size) % block_size
    if pad_len > 0:
        tensor = F.pad(tensor, (0, pad_len), value=0.0)
    num_blocks = tensor.shape[-1] // block_size
    new_shape = tensor.shape[:-1] + (num_blocks, block_size)
    return tensor.reshape(new_shape), num_blocks


def shift_matmul(
    x: Tensor,
    q_weight: Tensor,
    scales: Tensor,
    block_size: int,
) -> Tensor:
    """
    Memory-efficient shift matmul: dequantize weights on-the-fly, then F.linear.

    Mathematically equivalent to dequant(q_weight, scales) @ x.T but avoids
    the large [batch, seq, out_f, num_blocks] intermediate of a naive einsum.

    Args:
        x:          [..., in_features] float tensor (any batch dims).
        q_weight:   [out_features, in_features] int8, values in {-4,-2,-1,0,1,2,4}.
        scales:     [out_features, num_blocks] FP16 per-block scales.
        block_size: Block size used during quantization.

    Returns:
        [..., out_features] float32 tensor.
    """
    from .quantize import dequantize_block  # avoid circular at module level

    out_f, in_f = q_weight.shape
    w_hat = dequantize_block(q_weight, scales, block_size, original_in_features=in_f)
    return F.linear(x.float(), w_hat.float())


# Nonzero shift values: (q_value, shift_amount, negate)
# q_value == sign * 2^shift_amount
_SHIFT_TABLE = [
    (4,  2, False),
    (2,  1, False),
    (1,  0, False),
    (-1, 0, True),
    (-2, 1, True),
    (-4, 2, True),
]


def shift_matmul_pure(
    x: Tensor,
    q_weight: Tensor,
    scales: Tensor,
    block_size: int,
) -> Tensor:
    """
    Shift-and-accumulate matmul: no multiplications on weights.

    For each nonzero shift value k in {±1, ±2, ±4}:
        - Create a boolean mask where q_weight == k
        - For positions with shift 1: use x directly
          For positions with shift 2: use torch.ldexp(x, 1)  = x * 2^1
          For positions with shift 4: use torch.ldexp(x, 2)  = x * 2^2
        - Mask-and-sum over the block_size dimension, negate if k < 0
    Then multiply accumulated block results by per-block scales.

    torch.ldexp(mantissa, exponent) computes mantissa * 2^exponent, which is
    the floating-point equivalent of a left bit-shift on the significand.

    Args: same as shift_matmul.
    Returns: [..., out_features] float tensor.
    """
    out_f, in_f = q_weight.shape

    x_f = x.float()
    qw_f = q_weight.float()

    # Pad and reshape
    x_blocks, num_blocks = _pad_and_reshape(x_f, in_f, block_size)
    qw_blocks, _ = _pad_and_reshape(qw_f, in_f, block_size)

    # Accumulator: [..., out_f, num_blocks]
    batch_shape = x_blocks.shape[:-2]
    accum = torch.zeros(*batch_shape, out_f, num_blocks, device=x.device, dtype=torch.float32)

    # Pre-compute shifted versions of x_blocks (shared across out_f)
    # x_blocks: [..., num_blocks, block_size]
    x_shift = {
        0: x_blocks,
        1: torch.ldexp(x_blocks, torch.tensor(1, device=x.device)),
        2: torch.ldexp(x_blocks, torch.tensor(2, device=x.device)),
    }

    for q_val, shift_amt, negate in _SHIFT_TABLE:
        # mask: [out_f, num_blocks, block_size]
        mask = (qw_blocks == float(q_val))

        # shifted_x: [..., num_blocks, block_size]
        shifted_x = x_shift[shift_amt]

        # Broadcast and sum over block_size:
        # shifted_x: [..., 1, num_blocks, block_size]
        # mask:      [out_f, num_blocks, block_size]
        # product:   [..., out_f, num_blocks, block_size]
        # summed:    [..., out_f, num_blocks]
        contribution = (shifted_x.unsqueeze(-3) * mask.float()).sum(dim=-1)

        if negate:
            accum -= contribution
        else:
            accum += contribution

    # Scale: [..., out_f, num_blocks] * [out_f, num_blocks]
    scaled = accum * scales.float()

    # Sum over blocks
    return scaled.sum(dim=-1)
