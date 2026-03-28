"""
Diagnostic: why is PPL degradation so strong?

Hypotheses:
  H1 — Outlier-driven scale inflation: one large outlier per block forces
        a large scale, squashing all other weights to ~0.
  H2 — Non-uniform shift grid: {0,1,2,4} has a gap at 3 (and between 0 and 1),
        so mid-range values incur large absolute error compared to uniform 4-bit.
  H3 — Specific layers (attention vs MLP, early vs late) dominate the error.

Outputs:
  - Per-layer MSE and relative error
  - Distribution of normalized weights (do they fall in the "bad" regions?)
  - Fraction of blocks that have a kurtosis outlier
  - Comparison: shift-quantization MSE vs hypothetical uniform 4-bit MSE
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ptq.quantize import quantize_block, dequantize_block, round_to_shift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).pow(2).mean().item()

def relative_error(orig: torch.Tensor, recon: torch.Tensor) -> float:
    """||W - W_hat|| / ||W||"""
    return ((orig.float() - recon.float()).norm() / (orig.float().norm() + 1e-10)).item()

def uniform_4bit_mse(weight: torch.Tensor, block_size: int) -> float:
    """MSE of a hypothetical uniform symmetric 4-bit quantizer (16 levels in [-max, +max])."""
    w = weight.float()
    pad = (block_size - w.shape[1] % block_size) % block_size
    if pad:
        w = F.pad(w, (0, pad))
    blocks = w.reshape(w.shape[0], -1, block_size)
    scales = blocks.abs().amax(dim=-1)          # max of block
    normed = blocks / scales.unsqueeze(-1).clamp(min=1e-10)  # in [-1, 1]
    # 4-bit symmetric: 8 positive levels -> step = 1/7.5
    step = 1.0 / 7.5
    q = torch.round(normed / step) * step
    q = q.clamp(-1, 1)
    return ((blocks - q * scales.unsqueeze(-1)).pow(2).mean()).item()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(model_name: str = "Qwen/Qwen2-1.5B", block_sizes=(32, 64, 128)):
    from transformers import AutoModelForCausalLM
    print(f"Loading {model_name} (FP16, CPU for analysis)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cpu",
        local_files_only=True, trust_remote_code=True,
    )
    model.eval()

    # -------------------------------------------------------------------------
    # 1. Hypothesis H1: outlier-driven scale inflation
    #    Metric: "outlier ratio" = fraction of blocks where max(|w|) > 5 * median(|w|)
    #    Also: "scale waste" = how much of the dynamic range is used by the
    #    non-outlier weights if we remove the top-1 outlier per block.
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("H1 — OUTLIER-DRIVEN SCALE INFLATION")
    print("="*70)

    global_outlier_blocks = {bs: 0 for bs in block_sizes}
    global_total_blocks   = {bs: 0 for bs in block_sizes}
    global_scale_waste    = {bs: [] for bs in block_sizes}   # range used without top-1

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        w = module.weight.data.float()
        out_f, in_f = w.shape

        for bs in block_sizes:
            pad = (bs - in_f % bs) % bs
            wp = F.pad(w, (0, pad)) if pad else w
            blocks = wp.reshape(out_f, -1, bs)  # [out_f, n_blocks, bs]

            block_max  = blocks.abs().amax(dim=-1)      # [out_f, n_blocks]
            block_med  = blocks.abs().median(dim=-1).values

            # Outlier block: max > 5 * median
            outlier_mask = (block_max > 5 * block_med)
            n_out = outlier_mask.sum().item()
            n_tot = outlier_mask.numel()
            global_outlier_blocks[bs] += n_out
            global_total_blocks[bs]   += n_tot

            # Scale waste: for each block, range covered by non-max weights
            # i.e. second-largest abs value / max abs value
            top2 = blocks.abs().topk(2, dim=-1).values  # [out_f, n_blocks, 2]
            second_max = top2[..., 1]   # [out_f, n_blocks]
            waste = (second_max / block_max.clamp(min=1e-10)).mean().item()
            global_scale_waste[bs].append(waste)

    print(f"\n{'Block size':>12} | {'Outlier blocks':>16} | {'% outlier':>10} | {'Avg scale utilization':>22}")
    print("-" * 68)
    for bs in block_sizes:
        pct = 100.0 * global_outlier_blocks[bs] / max(global_total_blocks[bs], 1)
        avg_util = 100.0 * sum(global_scale_waste[bs]) / len(global_scale_waste[bs])
        print(f"{bs:>12} | {global_outlier_blocks[bs]:>16,} | {pct:>9.2f}% | {avg_util:>21.1f}%")

    print("\n  'Outlier block' = max(|w|) > 5 × median(|w|) in that block.")
    print("  'Scale utilization' = second-largest |w| / max(|w|) per block.")
    print("  Low utilization → scale dominated by single outlier.")

    # -------------------------------------------------------------------------
    # 2. Hypothesis H2: non-uniform shift grid leaves large gaps
    #    Metric: distribution of normalized weights before rounding
    #    How many weights fall in the "bad" zone [1.5, 3.0) (rounds to 2, but
    #    max error = 1) vs [0.5, 1.5) (rounds to 1, max error = 0.5)?
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("H2 — NON-UNIFORM SHIFT GRID")
    print("="*70)

    # Collect all normalized weight values for bs=64
    bs = 64
    buckets = {
        "[0, 0.5)":   0,
        "[0.5, 1.5)": 0,
        "[1.5, 3.0)": 0,
        "[3.0, 4.0]": 0,
    }
    total_weights = 0
    shift_mse_total = 0.0
    uniform4_mse_total = 0.0
    n_layers = 0

    print(f"\nUsing block_size={bs} for grid analysis.")
    print(f"\n{'Layer':>40} | {'Shift MSE':>12} | {'Unif-4b MSE':>12} | {'Rel err':>8}")
    print("-" * 82)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        w = module.weight.data.float()
        out_f, in_f = w.shape

        # Normalized weights (absolute value)
        pad = (bs - in_f % bs) % bs
        wp = F.pad(w, (0, pad)) if pad else w
        blocks = wp.reshape(out_f, -1, bs)
        scales = blocks.abs().amax(dim=-1).clamp(min=1e-10) / 4.0
        normed_abs = (blocks / scales.unsqueeze(-1)).abs().flatten()

        b0 = (normed_abs < 0.5).sum().item()
        b1 = ((normed_abs >= 0.5) & (normed_abs < 1.5)).sum().item()
        b2 = ((normed_abs >= 1.5) & (normed_abs < 3.0)).sum().item()
        b3 = (normed_abs >= 3.0).sum().item()
        buckets["[0, 0.5)"]   += b0
        buckets["[0.5, 1.5)"] += b1
        buckets["[1.5, 3.0)"] += b2
        buckets["[3.0, 4.0]"] += b3
        total_weights += normed_abs.numel()

        # Per-layer MSE comparison
        q_w, s = quantize_block(module.weight.data, bs)
        w_hat = dequantize_block(q_w, s, bs, in_f)
        s_mse = mse(module.weight.data, w_hat)
        u_mse = uniform_4bit_mse(module.weight.data, bs)
        r_err = relative_error(module.weight.data, w_hat)

        shift_mse_total   += s_mse
        uniform4_mse_total += u_mse
        n_layers += 1

        short = name[-38:] if len(name) > 38 else name
        print(f"{short:>40} | {s_mse:>12.6f} | {u_mse:>12.6f} | {r_err:>7.3f}")

    print("\n  Normalized weight distribution (|w_norm|, all layers, bs=64):")
    print(f"  {'Range':>12} | {'Count':>12} | {'%':>8} | {'Max quant error':>16}")
    print("  " + "-" * 58)
    max_errors = {
        "[0, 0.5)":   "0.5  (rounds to 0)",
        "[0.5, 1.5)": "0.5  (rounds to 1)",
        "[1.5, 3.0)": "1.0  (rounds to 2)",
        "[3.0, 4.0]": "1.0  (rounds to 4)",
    }
    for bucket, count in buckets.items():
        pct = 100.0 * count / max(total_weights, 1)
        print(f"  {bucket:>12} | {count:>12,} | {pct:>7.2f}% | {max_errors[bucket]:>16}")

    print(f"\n  Average shift MSE    : {shift_mse_total/max(n_layers,1):.6f}")
    print(f"  Average uniform-4bit : {uniform4_mse_total/max(n_layers,1):.6f}")
    print(f"  Ratio (shift/unif4)  : {shift_mse_total/max(uniform4_mse_total,1e-10):.2f}×")
    print("\n  → Ratio > 1 means shift quantization is worse than uniform 4-bit.")

    # -------------------------------------------------------------------------
    # 3. Per-layer relative error ranking (worst 10)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("H3 — WORST LAYERS (by relative error, bs=64)")
    print("="*70)

    layer_errors = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        w = module.weight.data
        q_w, s = quantize_block(w, 64)
        w_hat = dequantize_block(q_w, s, 64, w.shape[1])
        r = relative_error(w, w_hat)
        layer_errors.append((name, r, w.shape))

    layer_errors.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Layer':>45} | {'Rel error':>10} | {'Shape':>16}")
    print("-" * 78)
    for name, err, shape in layer_errors[:15]:
        short = name[-43:] if len(name) > 43 else name
        print(f"{short:>45} | {err:>9.4f}  | {str(tuple(shape)):>16}")

    # -------------------------------------------------------------------------
    # 4. Actionable insight: what block size minimizes error for outlier layers?
    # -------------------------------------------------------------------------
    worst_name = layer_errors[0][0]
    print(f"\n  Zoom on worst layer: {worst_name}")
    for name, module in model.named_modules():
        if name == worst_name:
            w = module.weight.data
            print(f"  Shape: {tuple(w.shape)}   |w|_max={w.abs().max():.4f}  |w|_mean={w.abs().mean():.4f}")
            print(f"  Kurtosis: {w.float().kurtosis():.2f}  (>3 = heavy tails)")
            print()
            print(f"  {'BS':>6} | {'Rel error':>10} | {'MSE':>12} | {'vs Unif-4b':>12}")
            print("  " + "-" * 48)
            for bs in [16, 32, 64, 128, 256]:
                q_w, s = quantize_block(w, bs)
                w_hat = dequantize_block(q_w, s, bs, w.shape[1])
                r = relative_error(w, w_hat)
                s_mse = mse(w, w_hat)
                u_mse = uniform_4bit_mse(w, bs)
                print(f"  {bs:>6} | {r:>9.4f}  | {s_mse:>12.6f} | {u_mse:>12.6f}")
            break


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2-1.5B")
    args = p.parse_args()
    analyze(args.model)
