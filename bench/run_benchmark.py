"""
PTQ Shift-Quantization Benchmark
=================================
Measures perplexity on WikiText-103 for:
  1. FP16 baseline
  2. Shift-quantized with block sizes 32, 64, 128

Usage:
    python -m bench.run_benchmark --model Qwen/Qwen2-1.5B
    python -m bench.run_benchmark --model meta-llama/Llama-3.2-3B --block-sizes 64 128
    python -m bench.run_benchmark --model Qwen/Qwen2-1.5B --device cpu  # for testing
"""

import argparse
import copy
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.perplexity import compute_perplexity, load_wikitext103_test
from ptq.model_wrapper import quantize_model
from ptq.awq import quantize_model_awq
from ptq.calibrate import collect_activation_scales
from ptq.utils import compute_memory_footprint


def format_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n/1024**3:.2f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024:.1f} KB"


def print_table(rows: list[dict]) -> None:
    headers = ["Configuration", "PPL", "Δ PPL", "Memory", "Compression", "Time (s)"]
    widths   = [22,              8,     8,       10,       13,            10]

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"

    print(sep)
    print(fmt.format(*headers))
    print(sep)

    baseline_ppl = rows[0]["ppl"] if rows else None
    for row in rows:
        delta = ""
        if baseline_ppl and row["ppl"] != baseline_ppl:
            delta = f"+{row['ppl'] - baseline_ppl:.2f}"
        print(fmt.format(
            row["name"],
            f"{row['ppl']:.2f}",
            delta,
            row["memory"],
            row["compression"],
            f"{row['time']:.1f}",
        ))

    print(sep)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    print("Loading WikiText-103 test set...")
    test_text = load_wikitext103_test()
    # Use a reasonable subset for speed (first ~1M characters covers ~200k tokens)
    if args.max_chars:
        test_text = test_text[: args.max_chars]

    rows = []

    # -------------------------------------------------------------------------
    # FP16 baseline
    # -------------------------------------------------------------------------
    print("\nEvaluating FP16 baseline...")
    t0 = time.time()
    ppl_fp16 = compute_perplexity(
        model, tokenizer, test_text,
        max_length=args.max_length, device=device,
    )
    t_fp16 = time.time() - t0

    fp16_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    rows.append({
        "name": "FP16 baseline",
        "ppl": ppl_fp16,
        "memory": format_bytes(fp16_bytes),
        "compression": "1.00×",
        "time": t_fp16,
    })
    print(f"  PPL: {ppl_fp16:.3f}   Memory: {format_bytes(fp16_bytes)}")

    # -------------------------------------------------------------------------
    # Collect activation scales once (reused by all AWQ runs)
    # -------------------------------------------------------------------------
    act_scales = None
    if args.awq:
        print("\nCollecting activation scales for AWQ calibration...")
        t0 = time.time()
        # Use first 64k chars of test text as calibration data
        # ~4 chars/token → 128 samples × 512 tokens × 4 = 262k chars minimum
        calib_chars = max(args.calib_samples * 512 * 4, 64_000)
        calib_text = test_text[:calib_chars]
        act_scales = collect_activation_scales(
            model, tokenizer, calib_text,
            n_samples=args.calib_samples, seq_len=512, device=device,
        )
        print(f"  Done in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # Shift-quantized runs — reload FP16 model fresh for each run
    # -------------------------------------------------------------------------
    # Build run configs: (block_size, grid, calibrated, awq, label)
    configs = []
    for bs in args.block_sizes:
        for g in args.grids:
            configs.append((bs, g, False, False, f"{g}  bs={bs}"))
        if args.calibrated:
            configs.append((bs, "A", True, False, f"A-cal bs={bs}"))
        if args.awq:
            for ag in args.awq_grids:
                configs.append((bs, ag, False, True, f"AWQ-{ag} bs={bs}"))

    for block_size, grid, calibrated, use_awq, label in configs:
        print(f"\nQuantizing [{label}]...")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()

        t0 = time.time()
        if use_awq:
            quantize_model_awq(model, act_scales, block_size=block_size,
                               n_delta=args.n_delta, grid=grid, verbose=False)
        else:
            quantize_model(model, block_size=block_size, grid=grid, calibrated=calibrated,
                           n_candidates=args.n_candidates, verbose=False)
        q_time = time.time() - t0
        print(f"  Quantization done in {q_time:.1f}s")

        footprint = compute_memory_footprint(model)
        mem_str   = format_bytes(footprint["total_bytes"])
        ratio_str = f"{footprint['compression_ratio']:.2f}×"

        print(f"  Evaluating PPL...")
        t0 = time.time()
        ppl_q = compute_perplexity(
            model, tokenizer, test_text,
            max_length=args.max_length, device=device,
        )
        t_q = time.time() - t0

        rows.append({
            "name": label,
            "ppl": ppl_q,
            "memory": mem_str,
            "compression": ratio_str,
            "time": t_q,
        })
        print(f"  PPL: {ppl_q:.3f}   Memory: {mem_str}   Compression: {ratio_str}")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print_table(rows)
    print()
    print("PPL: lower is better.  Δ PPL: degradation vs FP16 baseline.")
    print(f"Compression: theoretical (int8 weights + FP16 scales), "
          f"not 3-bit packed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PTQ shift-quantization benchmark on WikiText-103"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2-1.5B",
        help="HuggingFace model ID (default: Qwen/Qwen2-1.5B)",
    )
    parser.add_argument(
        "--block-sizes", nargs="+", type=int, default=[32, 64, 128],
        metavar="N",
        help="Block sizes to benchmark (default: 32 64 128)",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Token window length for PPL evaluation (default: 2048)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=1_000_000,
        help="Limit test text to first N chars for speed (0=no limit, default: 1M)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Compute device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--grids", nargs="+", default=["A"],
        choices=["A", "B", "C", "9v"],
        help="Grids: A=7v-log, B=7v-unif, C=8v-asymm, 9v=9v-unif (default: A)",
    )
    parser.add_argument(
        "--calibrated", action="store_true",
        help="Also run MSE-calibrated scale (grid A)",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=100,
        help="Grid resolution for MSE scale search (default: 100)",
    )
    parser.add_argument(
        "--awq", action="store_true",
        help="Run AWQ-style activation-aware quantization",
    )
    parser.add_argument(
        "--awq-grids", nargs="+", default=["A"],
        choices=["A", "9v"],
        help="Grids to use with AWQ (default: A). Use 'A 9v' to test both.",
    )
    parser.add_argument(
        "--calib-samples", type=int, default=128,
        help="Number of calibration windows for AWQ (default: 128)",
    )
    parser.add_argument(
        "--n-delta", type=int, default=20,
        help="Delta grid resolution for AWQ scale search (default: 20)",
    )
    args = parser.parse_args()
    if args.max_chars == 0:
        args.max_chars = None  # type: ignore[assignment]

    run(args)


if __name__ == "__main__":
    main()
