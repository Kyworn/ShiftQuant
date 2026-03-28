"""
AWQ-style activation-aware quantization for shift-based grids.

Core idea (Lin et al., 2023 — adapted for shift grid):
  Not all input channels are equal. A channel j with large mean activation
  s_x[j] contributes more to the output: error on W[:, j] gets amplified by s_x[j].

  Solution: before quantizing, scale up important channels:
      W_new[:, j] = W[:, j] * α[j]          (amplify important channels)
  At inference, divide the input accordingly:
      y = Q(W_new) * (x / α)  ≈  W * x      (undo the scale)

  The per-channel scale α = s_x^δ where δ ∈ [0,1] is searched per-layer.
  δ=0 → no AWQ (identity), δ=1 → full activation-scale weighting.

  Error objective (activation-weighted weight error — diagonal Hessian approx):
      E(δ) = || (Q(W_new) - W_new) * (s_x / α) ||²_F
           = || quant_err[:, j] * (s_x[j] / α[j]) ||²_F

  This weights quantization error by s_x/α: channels with large activation
  relative to their scale get penalised more, so the search finds α that
  equalises the error contribution across channels.

Inference wrapper:
  AWQLinear stores α and divides the input before calling QuantizedLinear.
  Zero architecture-specific assumptions — works with any model.
  (Optional future: absorb 1/α into the preceding LayerNorm weight for free.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .quantize import quantize_block, quantize_block_9val, dequantize_block
from .quantized_linear import QuantizedLinear

_GRID_QUANT_FNS = {
    "A":  quantize_block,
    "9v": quantize_block_9val,
}


# ---------------------------------------------------------------------------
# Per-layer AWQ scale search
# ---------------------------------------------------------------------------

def find_awq_scale(
    weight:     Tensor,
    act_scales: Tensor,
    block_size: int = 32,
    n_delta:    int = 20,
    grid:       str = "A",
) -> Tensor:
    """
    Search for the per-channel smooth scale α = act_scales^δ that minimises
    the activation-weighted quantization error.

    Args:
        weight:     [out_f, in_f] float32 — original FP16 weights.
        act_scales: [in_f] float32 — mean |activation| per input channel.
        block_size: Block size for quantization.
        n_delta:    Grid points for δ search in [0, 1].

    Returns:
        alpha: [in_f] float32 — per-channel smooth scale.
    """
    device = weight.device
    s = act_scales.to(device).clamp(min=1e-6)          # [in_f]
    w = weight.float()                                  # [out_f, in_f]
    in_f = w.shape[1]

    best_err   = float("inf")
    best_alpha = torch.ones(in_f, device=device)

    for delta in torch.linspace(0.0, 1.0, n_delta, device=device):
        alpha = s.pow(delta)                            # [in_f]

        # Scale weights by alpha (amplify important channels)
        w_scaled = w * alpha.unsqueeze(0)               # [out_f, in_f]

        # Quantize the scaled weights (using the chosen grid)
        quant_fn = _GRID_QUANT_FNS[grid]
        q, scales = quant_fn(w_scaled, block_size)
        w_hat = dequantize_block(q, scales, block_size, in_f).float()

        # Activation-weighted error: weight quant error by (s / alpha)
        # Channels with large s/alpha are important and penalised more
        weights_j = (s / alpha).unsqueeze(0)            # [1, in_f]
        err = ((w_hat - w_scaled) * weights_j).pow(2).mean().item()

        if err < best_err:
            best_err   = err
            best_alpha = alpha.clone()

    return best_alpha


# ---------------------------------------------------------------------------
# AWQ inference wrapper
# ---------------------------------------------------------------------------

class AWQLinear(nn.Module):
    """
    QuantizedLinear wrapped with a per-channel input scale.

    At inference:  y = Q(W * diag(α)) * (x / α)  ≈  W * x

    The division x / α is cheap (broadcast over in_features).
    """

    def __init__(
        self,
        quantized_linear: QuantizedLinear,
        input_scale: Tensor,               # [in_features]
    ) -> None:
        super().__init__()
        self.ql = quantized_linear
        self.register_buffer("input_scale", input_scale.float())

    @classmethod
    def from_linear(
        cls,
        linear:     nn.Linear,
        act_scales: Tensor,
        block_size: int = 32,
        n_delta:    int = 20,
        grid:       str = "A",
    ) -> "AWQLinear":
        """
        Convert nn.Linear → AWQLinear in one shot.

        1. Find per-channel smooth scale α via grid search.
        2. Scale weights: W_new = W * α.
        3. Quantize W_new with shift-quantization.
        4. Store α as input_scale (applied at inference).
        """
        device = linear.weight.device
        w = linear.weight.data.float()

        # Find optimal per-channel scale
        alpha = find_awq_scale(w, act_scales.to(device), block_size, n_delta, grid=grid)

        # Scale weights by alpha, quantize on device
        w_scaled = w * alpha.unsqueeze(0)
        quant_fn = _GRID_QUANT_FNS[grid]
        q_w, scales = quant_fn(w_scaled, block_size)
        q_w, scales = q_w.cpu(), scales.cpu()

        # Build QuantizedLinear from the scaled weight
        has_bias = linear.bias is not None
        ql = QuantizedLinear(linear.in_features, linear.out_features,
                             bias=has_bias, block_size=block_size)
        ql.q_weight.copy_(q_w.to(device))
        ql.scales.copy_(scales.to(device))
        if has_bias:
            ql.bias.copy_(linear.bias.data.half().to(device))  # type: ignore[union-attr]
        ql = ql.to(device)

        return cls(ql, alpha.cpu())

    def forward(self, x: Tensor) -> Tensor:
        # Divide input by alpha to undo the weight scaling
        x_scaled = x / self.input_scale.to(x.device, x.dtype)
        return self.ql(x_scaled)

    def extra_repr(self) -> str:
        return (f"in={self.ql.in_features}, out={self.ql.out_features}, "
                f"block_size={self.ql.block_size}")


# ---------------------------------------------------------------------------
# Model-level AWQ quantization
# ---------------------------------------------------------------------------

DEFAULT_SKIP: frozenset[str] = frozenset({"lm_head", "embed_tokens"})


def quantize_model_awq(
    model:       nn.Module,
    act_scales:  dict[str, Tensor],
    block_size:  int = 32,
    n_delta:     int = 20,
    grid:        str = "A",
    skip_names:  frozenset[str] = DEFAULT_SKIP,
    verbose:     bool = False,
) -> nn.Module:
    """
    Replace all eligible nn.Linear layers with AWQLinear in-place.

    Args:
        model:      FP16 model.
        act_scales: Output of collect_activation_scales() — name → [in_features].
        block_size: Quantization block size.
        n_delta:    Delta grid resolution for scale search.
        skip_names: Layer name substrings to skip.
        verbose:    Print progress.

    Returns:
        Same model with Linear layers replaced by AWQLinear.
    """
    targets: list[tuple[nn.Module, str, nn.Linear, str]] = []

    for mod_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{mod_name}.{child_name}" if mod_name else child_name
            if any(s in full_name for s in skip_names):
                continue
            targets.append((module, child_name, child, full_name))

    replaced = 0
    for parent, child_name, linear, full_name in targets:
        if full_name not in act_scales:
            if verbose:
                print(f"  [skip — no act stats] {full_name}")
            continue

        awq = AWQLinear.from_linear(
            linear,
            act_scales[full_name],
            block_size=block_size,
            n_delta=n_delta,
            grid=grid,
        )
        setattr(parent, child_name, awq)
        if verbose:
            print(f"  [AWQ] {full_name}  [{linear.out_features}×{linear.in_features}]")
        replaced += 1
        del linear

    torch.cuda.empty_cache()
    if verbose:
        print(f"\nAWQ layers: {replaced}")
    return model
