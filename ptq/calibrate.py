"""
Calibration: collect per-channel mean absolute activation values
for every linear layer in the model.

Used by AWQ-style quantization to find which input channels are
"important" (large activations → large contribution to output error).
"""

import torch
import torch.nn as nn
from torch import Tensor


def collect_activation_scales(
    model: nn.Module,
    tokenizer,
    text: str,
    n_samples: int = 128,
    seq_len: int = 512,
    device: str | torch.device = "cuda",
    skip_names: frozenset[str] = frozenset({"lm_head", "embed_tokens"}),
) -> dict[str, Tensor]:
    """
    Run the model on calibration text and collect per-channel activation stats.

    For each eligible nn.Linear, records:
        mean(|x_j|) over all (batch, sequence) positions and calibration samples.

    Args:
        model:      FP16 model (unquantized).
        tokenizer:  Matching tokenizer.
        text:       Raw calibration text (WikiText-103 or similar).
        n_samples:  Number of non-overlapping seq_len windows to process.
        seq_len:    Tokens per window.
        device:     Compute device.
        skip_names: Layer name substrings to ignore.

    Returns:
        dict mapping full module name → Tensor[in_features] (mean |activation|).
    """
    running_sum: dict[str, Tensor] = {}
    counts:      dict[str, int]    = {}
    hooks = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            x = inputs[0].detach().float()              # [B, T, C] or [B, C]
            x_flat = x.reshape(-1, x.shape[-1])         # [B*T, C]
            abs_mean = x_flat.abs().mean(dim=0).cpu()   # [C]
            if name not in running_sum:
                running_sum[name] = abs_mean
                counts[name] = 1
            else:
                n = counts[name]
                running_sum[name] = running_sum[name] * (n / (n + 1)) + abs_mean / (n + 1)
                counts[name] += 1
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(s in name for s in skip_names):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = enc.input_ids[0]                        # [total_tokens]

    processed = 0
    with torch.no_grad():
        for start in range(0, len(token_ids) - seq_len, seq_len):
            chunk = token_ids[start : start + seq_len].unsqueeze(0).to(device)
            model(chunk)
            processed += 1
            if processed >= n_samples:
                break

    for h in hooks:
        h.remove()

    print(f"  Collected activation scales from {processed} windows "
          f"({processed * seq_len:,} tokens), {len(running_sum)} layers.")
    return running_sum
