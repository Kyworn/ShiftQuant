"""
Core quantization and dequantization for shift-based PTQ.

Quantizes FP16 weights to {-4, -2, -1, 0, +1, +2, +4} with per-block scales.
These values are all powers of 2 (or zero), so inference needs only bit-shifts
and additions — no multiplications.

Two scale strategies:
  - max/4 heuristic  (quantize_block):            scale = max(|block|) / 4
  - MSE-optimal calibration (quantize_block_mse): scale = argmin_s MSE(block, quant(block,s))

Encoding: values stored as int8 directly (-4, -2, -1, 0, 1, 2, 4).
Scale: FP16 per-block, shape [out_features, num_blocks].
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def round_to_shift(x: Tensor) -> Tensor:
    """
    Round each element to the nearest value in {-4, -2, -1, 0, +1, +2, +4}.

    Decision thresholds (on absolute value):
        [0, 0.5)  -> 0
        [0.5, 1.5) -> 1
        [1.5, 3.0) -> 2
        [3.0, inf) -> 4

    Args:
        x: Any shape float tensor, values typically in [-4, 4] after normalization.

    Returns:
        Tensor of same shape and dtype with values in {-4,-2,-1,0,1,2,4}.
    """
    sign = x.sign()
    a = x.abs()
    result = torch.where(a < 0.5, torch.zeros_like(a),
             torch.where(a < 1.5, torch.ones_like(a),
             torch.where(a < 3.0, torch.full_like(a, 2.0),
                                  torch.full_like(a, 4.0))))
    return sign * result


def quantize_block(
    weight: Tensor,
    block_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """
    Quantize a 2D weight matrix with per-block scaling.

    For each block of `block_size` consecutive elements along in_features:
        scale = max(abs(block)) / 4.0
        q     = round_to_shift(block / scale)

    Args:
        weight:     FP16 tensor, shape [out_features, in_features].
        block_size: Number of weights per scaling block (32, 64, or 128).

    Returns:
        q_weight:   int8 tensor, shape [out_features, in_features],
                    values in {-4, -2, -1, 0, 1, 2, 4}.
        scales:     FP16 tensor, shape [out_features, num_blocks],
                    where num_blocks = ceil(in_features / block_size).
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")

    out_f, in_f = weight.shape
    w = weight.float()

    # Pad in_features to a multiple of block_size
    pad_len = (block_size - in_f % block_size) % block_size
    if pad_len > 0:
        w = F.pad(w, (0, pad_len), value=0.0)

    in_f_padded = w.shape[1]
    num_blocks = in_f_padded // block_size

    # [out_f, num_blocks, block_size]
    blocks = w.reshape(out_f, num_blocks, block_size)

    # Per-block scale: max(|block|) / 4  ->  [out_f, num_blocks]
    scales = blocks.abs().amax(dim=-1) / 4.0
    scales = scales.clamp(min=1e-10)

    # Normalize and round
    normed = blocks / scales.unsqueeze(-1)
    q_blocks = round_to_shift(normed)

    # Reshape and trim padding  [out_f, in_f_padded] -> [out_f, in_f]
    q_weight = q_blocks.reshape(out_f, in_f_padded)[:, :in_f]

    return q_weight.to(torch.int8), scales.to(torch.float16)


def round_to_grid_B(x: Tensor) -> Tensor:
    """
    Grid B: {-3,-2,-1,0,+1,+2,+3} — 7 values, 3 bits, uniform spacing.
    No outlier coverage beyond ±3. Scale = max(|block|)/3.
    Thresholds: [0,0.5)→0, [0.5,1.5)→1, [1.5,2.5)→2, [2.5,∞)→3
    """
    sign = x.sign()
    a = x.abs()
    result = torch.where(a < 0.5, torch.zeros_like(a),
             torch.where(a < 1.5, torch.ones_like(a),
             torch.where(a < 2.5, torch.full_like(a, 2.0),
                                  torch.full_like(a, 3.0))))
    return sign * result


def round_to_grid_C(x: Tensor) -> Tensor:
    """
    Grid C: {-4,-3,-2,-1,0,+1,+2,+3} — 8 values, exactly 3 bits (2³).
    Asymmetric: negative side covers outliers to -4, positive side caps at +3.
    Scale = max(|block|)/4 — negative outliers protected, positive clips at 3/4×max.
    Thresholds: uniform step=1, but positive is capped at 3.
    """
    sign = x.sign()
    a = x.abs()
    mag = torch.where(a < 0.5, torch.zeros_like(a),
          torch.where(a < 1.5, torch.ones_like(a),
          torch.where(a < 2.5, torch.full_like(a, 2.0),
          torch.where(a < 3.5, torch.full_like(a, 3.0),
                               torch.full_like(a, 4.0)))))
    # Positive values: cap at 3 (+4 not in grid C)
    cap = torch.where(x > 0, torch.full_like(mag, 3.0), torch.full_like(mag, 4.0))
    return sign * torch.min(mag, cap)


def round_to_shift_9(x: Tensor) -> Tensor:
    """
    Round to the nearest value in {-4,-3,-2,-1,0,+1,+2,+3,+4}.

    9 values = 4 bits.  Uniform spacing (step=1) — fills the ±3 gap.
    Thresholds on |x|:
        [0,   0.5) → 0
        [0.5, 1.5) → 1
        [1.5, 2.5) → 2
        [2.5, 3.5) → 3
        [3.5, ∞)   → 4
    """
    sign = x.sign()
    a = x.abs()
    result = torch.where(a < 0.5, torch.zeros_like(a),
             torch.where(a < 1.5, torch.ones_like(a),
             torch.where(a < 2.5, torch.full_like(a, 2.0),
             torch.where(a < 3.5, torch.full_like(a, 3.0),
                                  torch.full_like(a, 4.0)))))
    return sign * result


def quantize_block_B(weight: Tensor, block_size: int = 64) -> tuple[Tensor, Tensor]:
    """Grid B: {-3..+3}, scale = max/3."""
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")
    out_f, in_f = weight.shape
    w = weight.float()
    pad_len = (block_size - in_f % block_size) % block_size
    if pad_len: w = F.pad(w, (0, pad_len))
    blocks = w.reshape(out_f, -1, block_size)
    scales = blocks.abs().amax(dim=-1) / 3.0
    scales = scales.clamp(min=1e-10)
    q = round_to_grid_B(blocks / scales.unsqueeze(-1))
    q_weight = q.reshape(out_f, -1)[:, :in_f]
    return q_weight.to(torch.int8), scales.to(torch.float16)


def quantize_block_C(weight: Tensor, block_size: int = 64) -> tuple[Tensor, Tensor]:
    """Grid C: {-4,-3,-2,-1,0,+1,+2,+3}, scale = max/4. Exactly 3 bits (2³)."""
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")
    out_f, in_f = weight.shape
    w = weight.float()
    pad_len = (block_size - in_f % block_size) % block_size
    if pad_len: w = F.pad(w, (0, pad_len))
    blocks = w.reshape(out_f, -1, block_size)
    scales = blocks.abs().amax(dim=-1) / 4.0
    scales = scales.clamp(min=1e-10)
    q = round_to_grid_C(blocks / scales.unsqueeze(-1))
    q_weight = q.reshape(out_f, -1)[:, :in_f]
    return q_weight.to(torch.int8), scales.to(torch.float16)


def quantize_block_9val(
    weight: Tensor,
    block_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """
    Quantize with the 9-value uniform grid {-4,-3,-2,-1,0,+1,+2,+3,+4}.

    Same block-scale structure as quantize_block, same max/4 scale.
    This is the scientific control: does filling the ±3 gap reduce PPL?
    4 bits per weight (vs 3 bits for the 7-value grid).
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")

    out_f, in_f = weight.shape
    w = weight.float()

    pad_len = (block_size - in_f % block_size) % block_size
    if pad_len > 0:
        w = F.pad(w, (0, pad_len), value=0.0)

    in_f_padded = w.shape[1]
    num_blocks   = in_f_padded // block_size
    blocks       = w.reshape(out_f, num_blocks, block_size)

    scales   = blocks.abs().amax(dim=-1) / 4.0
    scales   = scales.clamp(min=1e-10)
    normed   = blocks / scales.unsqueeze(-1)
    q_blocks = round_to_shift_9(normed)

    q_weight = q_blocks.reshape(out_f, in_f_padded)[:, :in_f]
    return q_weight.to(torch.int8), scales.to(torch.float16)


def find_optimal_scales(
    blocks: Tensor,
    n_candidates: int = 100,
) -> Tensor:
    """
    For each block find the scale s* that minimises MSE(block, round_to_shift(block/s)*s).

    The baseline is s0 = max(|block|)/4.  We search over s = s0 * alpha where
    alpha is sampled log-uniformly in [alpha_min, alpha_max].

    Intuition:  alpha < 1  →  finer grid, clips large values
                alpha > 1  →  coarser grid, avoids clipping
    For gaussian-like weights the optimal alpha is usually < 1 (clip outliers,
    improve coverage of the dense [0,2] region).

    Args:
        blocks:       [out_f, num_blocks, block_size] float32.
        n_candidates: Grid resolution for the 1-D scale search.

    Returns:
        optimal_scales: [out_f, num_blocks] float32 — the MSE-minimising scale
                        for every block.
    """
    # Baseline: max/4
    s0 = blocks.abs().amax(dim=-1).clamp(min=1e-10) / 4.0   # [out_f, num_blocks]

    # Log-uniform candidates in [0.1 × s0, 1.5 × s0]
    # (clipping below baseline almost always helps for gaussian weights)
    log_alphas = torch.linspace(
        math.log(0.1), math.log(1.5), n_candidates,
        device=blocks.device, dtype=blocks.dtype,
    )                                                          # [n_cand]
    alphas = log_alphas.exp()

    best_mse    = torch.full_like(s0, float("inf"))
    best_scales = s0.clone()

    for alpha in alphas:
        s = s0 * alpha                                         # [out_f, num_blocks]
        normed  = blocks / s.unsqueeze(-1)                    # [out_f, num_blocks, bs]
        q       = round_to_shift(normed)
        sq_err  = (blocks - q * s.unsqueeze(-1)).pow(2).mean(dim=-1)  # [out_f, num_blocks]

        improved = sq_err < best_mse
        best_mse    = torch.where(improved, sq_err, best_mse)
        best_scales = torch.where(improved, s, best_scales)

    return best_scales


def quantize_block_mse(
    weight: Tensor,
    block_size: int = 64,
    n_candidates: int = 100,
) -> tuple[Tensor, Tensor]:
    """
    Like quantize_block but the per-block scale minimises reconstruction MSE
    instead of using the max/4 heuristic.

    This is purely a weight-space optimisation — no calibration data required.
    The search over scales compensates for the non-uniform spacing of
    {0,1,2,4}: an optimal alpha < 1 clips the largest weights and maps the
    dense mid-range into the finer part of the shift grid.

    Args:
        weight:       FP16 tensor [out_features, in_features].
        block_size:   Elements per block.
        n_candidates: Grid resolution for scale search (default 100 is fast enough).

    Returns:
        q_weight: int8 [out_features, in_features], values in {-4,-2,-1,0,1,2,4}.
        scales:   FP16 [out_features, num_blocks] — MSE-optimal per-block scales.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")

    out_f, in_f = weight.shape
    w = weight.float()

    pad_len = (block_size - in_f % block_size) % block_size
    if pad_len > 0:
        w = F.pad(w, (0, pad_len), value=0.0)

    in_f_padded = w.shape[1]
    num_blocks   = in_f_padded // block_size
    blocks       = w.reshape(out_f, num_blocks, block_size)

    scales = find_optimal_scales(blocks, n_candidates=n_candidates)   # [out_f, num_blocks]
    scales = scales.clamp(min=1e-10)

    normed   = blocks / scales.unsqueeze(-1)
    q_blocks = round_to_shift(normed)

    q_weight = q_blocks.reshape(out_f, in_f_padded)[:, :in_f]
    return q_weight.to(torch.int8), scales.to(torch.float16)


def dequantize_block(
    q_weight: Tensor,
    scales: Tensor,
    block_size: int,
    original_in_features: int | None = None,
) -> Tensor:
    """
    Reconstruct approximate FP16 weights from quantized representation.

    Args:
        q_weight:             int8 tensor, shape [out_features, in_features].
        scales:               FP16 tensor, shape [out_features, num_blocks].
        block_size:           Block size used during quantization.
        original_in_features: If provided, trims padded columns at the end.

    Returns:
        Approximate FP16 weight tensor, shape [out_features, original_in_features].
    """
    out_f = q_weight.shape[0]
    in_f = q_weight.shape[1]
    num_blocks = scales.shape[1]
    in_f_padded = num_blocks * block_size

    w = q_weight.float()

    # Re-pad if in_f < in_f_padded (quantize_block trimmed it)
    if in_f < in_f_padded:
        w = F.pad(w, (0, in_f_padded - in_f), value=0.0)

    # [out_f, num_blocks, block_size]
    blocks = w.reshape(out_f, num_blocks, block_size)

    # Multiply by block scales
    dequant = blocks * scales.float().unsqueeze(-1)

    # [out_f, in_f_padded] -> trim to original_in_features
    result = dequant.reshape(out_f, in_f_padded)
    trim = original_in_features if original_in_features is not None else in_f
    return result[:, :trim].to(torch.float16)
