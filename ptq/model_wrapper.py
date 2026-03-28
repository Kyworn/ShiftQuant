"""
Model-level quantization: walk all nn.Linear layers and replace with
QuantizedLinear in-place, skipping embeddings and lm_head.
"""

import torch
import torch.nn as nn
from typing import Iterable

from .quantized_linear import QuantizedLinear

# Layers to leave in FP16 (embeddings are critical; lm_head is often weight-tied)
DEFAULT_SKIP_NAMES: frozenset[str] = frozenset({"lm_head", "embed_tokens"})


def quantize_model(
    model: nn.Module,
    block_size: int = 64,
    grid: str = "A",
    calibrated: bool = False,
    n_candidates: int = 100,
    skip_names: Iterable[str] | None = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Replace all eligible nn.Linear modules with QuantizedLinear in-place.

    Args:
        model:        HuggingFace or custom nn.Module.
        block_size:   Shift-quantization block size (32, 64, or 128).
        calibrated:   If True, use MSE-optimal scale search (Option 4).
                      If False, use max/4 heuristic (baseline).
        n_candidates: Grid resolution for MSE scale search (ignored if not calibrated).
        skip_names:   Module name substrings to exclude.
        verbose:      Print each replaced layer.

    Returns:
        The same model object with Linear layers swapped.
    """
    skip = frozenset(skip_names) if skip_names is not None else DEFAULT_SKIP_NAMES

    replaced = 0
    targets: list[tuple[nn.Module, str, nn.Linear]] = []

    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if any(s in full_name for s in skip):
                continue
            targets.append((module, child_name, child))

    for parent, child_name, linear in targets:
        quantized = QuantizedLinear.from_linear(
            linear, block_size,
            grid=grid,
            calibrated=calibrated,
            n_candidates=n_candidates,
        )
        setattr(parent, child_name, quantized)
        if verbose:
            tag = "[cal]" if calibrated else f"[{grid}]"
            print(f"  {tag} {child_name}  [{linear.out_features}x{linear.in_features}]")
        replaced += 1
        del linear

    torch.cuda.empty_cache()

    if verbose:
        print(f"\nTotal layers quantized: {replaced}")

    return model
