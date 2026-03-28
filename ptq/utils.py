"""
Utilities: memory footprint reporting, 3-bit packing (for future CUDA kernels).
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .quantized_linear import QuantizedLinear


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------

def compute_memory_footprint(model: nn.Module) -> dict:
    """
    Walk model buffers and parameters, report memory per module type.

    Returns a dict with keys:
        'total_bytes':       total bytes consumed by all tensors
        'fp16_equiv_bytes':  bytes if every weight were stored as FP16
        'compression_ratio': fp16_equiv / total
        'layers':            list of per-layer dicts
    """
    layers = []
    total_bytes = 0
    fp16_equiv = 0

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            qw_bytes = module.q_weight.numel()            # int8 -> 1 byte each
            sc_bytes  = module.scales.numel() * 2         # FP16 -> 2 bytes each
            bias_bytes = (module.bias.numel() * 2
                          if module.bias is not None else 0)
            layer_bytes = qw_bytes + sc_bytes + bias_bytes
            layer_fp16  = module.q_weight.numel() * 2 + bias_bytes

            layers.append({
                "name": name,
                "type": "QuantizedLinear",
                "bytes": layer_bytes,
                "fp16_equiv": layer_fp16,
            })
            total_bytes += layer_bytes
            fp16_equiv  += layer_fp16

        elif isinstance(module, nn.Linear):
            w_bytes   = module.weight.numel() * module.weight.element_size()
            b_bytes   = (module.bias.numel() * module.bias.element_size()
                         if module.bias is not None else 0)
            layer_bytes = w_bytes + b_bytes
            layers.append({
                "name": name,
                "type": "Linear",
                "bytes": layer_bytes,
                "fp16_equiv": layer_bytes,
            })
            total_bytes += layer_bytes
            fp16_equiv  += layer_bytes

        elif isinstance(module, nn.Embedding):
            e_bytes = module.weight.numel() * module.weight.element_size()
            layers.append({
                "name": name,
                "type": "Embedding",
                "bytes": e_bytes,
                "fp16_equiv": e_bytes,
            })
            total_bytes += e_bytes
            fp16_equiv  += e_bytes

    compression = fp16_equiv / max(total_bytes, 1)
    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1024**2,
        "fp16_equiv_bytes": fp16_equiv,
        "fp16_equiv_mb": fp16_equiv / 1024**2,
        "compression_ratio": compression,
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# 3-bit packing (for future CUDA inference kernel)
# ---------------------------------------------------------------------------
# Encoding: value -> code
# -4 -> 0, -2 -> 1, -1 -> 2, 0 -> 3, +1 -> 4, +2 -> 5, +4 -> 6
# Stored as 3 bits; pack 8 values into 3 bytes (24 bits), 2 bits wasted.

_VAL_TO_CODE = {-4: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 4: 6}
_CODE_TO_VAL = {v: k for k, v in _VAL_TO_CODE.items()}


def pack_3bit(q_weight: Tensor) -> tuple[Tensor, tuple]:
    """
    Pack int8 quantized weights into 3-bit codes (8 values per 3 bytes).

    Args:
        q_weight: int8 tensor with values in {-4,-2,-1,0,1,2,4}.

    Returns:
        packed: uint8 tensor, shape roughly (numel * 3 // 8 + padding,)
        orig_shape: original shape for unpacking
    """
    orig_shape = q_weight.shape
    flat = q_weight.cpu().numpy().flatten().astype("int8")
    n = len(flat)

    # Pad to multiple of 8
    pad = (8 - n % 8) % 8
    if pad:
        flat = list(flat) + [0] * pad

    # Convert to 3-bit codes
    codes = [_VAL_TO_CODE[int(v)] for v in flat]

    # Pack 8 codes into 3 bytes: bits [0:3][3:6][6:9][9:12]...[21:24]
    packed_bytes = []
    for i in range(0, len(codes), 8):
        group = codes[i:i+8]
        # 24 bits total
        bits = 0
        for j, code in enumerate(group):
            bits |= (code & 0x7) << (j * 3)
        packed_bytes.append(bits & 0xFF)
        packed_bytes.append((bits >> 8) & 0xFF)
        packed_bytes.append((bits >> 16) & 0xFF)

    packed = torch.tensor(packed_bytes, dtype=torch.uint8)
    return packed, orig_shape


def unpack_3bit(packed: Tensor, orig_shape: tuple) -> Tensor:
    """
    Unpack 3-bit coded weights back to int8 values.

    Args:
        packed:     uint8 tensor from pack_3bit.
        orig_shape: original weight shape.

    Returns:
        int8 tensor with same shape as original.
    """
    raw = packed.cpu().tolist()
    n_total = math.prod(orig_shape)
    pad = (8 - n_total % 8) % 8
    n_padded = n_total + pad

    codes = []
    for i in range(0, len(raw), 3):
        bits = raw[i] | (raw[i+1] << 8) | (raw[i+2] << 16)
        for j in range(8):
            codes.append((bits >> (j * 3)) & 0x7)

    values = [_CODE_TO_VAL[c] for c in codes[:n_padded]]
    result = torch.tensor(values[:n_total], dtype=torch.int8)
    return result.reshape(orig_shape)
