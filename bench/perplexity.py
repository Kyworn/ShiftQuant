"""
Perplexity computation on WikiText-103 (or any text dataset).

Uses non-overlapping windows of `max_length` tokens to avoid O(n^2) cost.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    max_length: int = 2048,
    device: str | torch.device = "cuda",
    batch_size: int = 1,
) -> float:
    """
    Compute perplexity of a model on raw text using non-overlapping windows.

    Args:
        model:      Language model (HuggingFace CausalLM).
        tokenizer:  Matching tokenizer.
        text:       Raw text string to evaluate on.
        max_length: Tokens per window (default 2048).
        device:     Computation device.
        batch_size: Number of windows processed simultaneously.

    Returns:
        Perplexity (lower is better).
    """
    model.eval()
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids: Tensor = encodings.input_ids[0]  # [seq_len]

    total_nll = 0.0
    total_tokens = 0

    # Non-overlapping windows of max_length tokens
    n = input_ids.size(0)
    windows = []
    for begin in range(0, n - 1, max_length):
        end = min(begin + max_length + 1, n)  # +1 for targets
        window = input_ids[begin:end]
        if window.size(0) < 2:
            break
        windows.append(window)

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i : i + batch_size]

        # Pad to same length within batch
        max_w = max(w.size(0) for w in batch_windows)
        padded = torch.stack([
            F.pad(w, (0, max_w - w.size(0)), value=tokenizer.pad_token_id or 0)
            for w in batch_windows
        ])  # [batch, max_w]

        inp = padded[:, :-1].to(device)
        tgt = padded[:, 1:].to(device)

        outputs = model(inp)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # logits: [batch, seq-1, vocab]

        # Compute NLL for each token, ignoring padding
        for j, window in enumerate(batch_windows):
            w_len = window.size(0) - 1  # number of prediction targets
            log_probs = F.log_softmax(logits[j, :w_len].float(), dim=-1)
            targets_j = tgt[j, :w_len]
            nll = F.nll_loss(log_probs, targets_j, reduction="sum")
            total_nll += nll.item()
            total_tokens += w_len

    if total_tokens == 0:
        return float("inf")

    return math.exp(total_nll / total_tokens)


def load_wikitext103_test() -> str:
    """Download and return WikiText-103 test set as a single string."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    return "\n\n".join(ds["text"])  # type: ignore
