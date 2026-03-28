# ShiftQuant

**Analyzing the Limits of Shift-Based Post-Training Quantization for LLMs**

Shift-based quantization restricts weights to exact powers of two `{−4, −2, −1, 0, +1, +2, +4}`,
enabling multiply-free inference via bit-shifts.  This repo contains the full research pipeline:
quantization code, AWQ adaptation, WikiText-103 benchmark, diagnostic analysis, and paper draft.

Model: **Qwen2-1.5B** · Dataset: **WikiText-103** · Hardware: RTX 5080 16 GB

---

## Key results

| Method | PPL | Δ vs FP16 | Recovery |
|:---|:---:|:---:|:---:|
| FP16 baseline | 9.58 | — | — |
| Shift PTQ (Grid A, bs=32) | 12.53 | +2.95 | 0% |
| + 9-value uniform grid | 11.61 | +2.03 | 31% |
| + AWQ (isocompression) | 11.88 | +2.30 | 22% |
| **+ AWQ × 9v grid** | **11.06** | **+1.48** | **50%** |

The grid improvement (−0.92 PPL) and AWQ improvement (−0.65 PPL) are **93.6% orthogonal** —
they can be optimised independently without interaction loss.

---

## Findings

1. **+30.8% PPL baseline cost** at block size 32.  Dominant cause: a structural gap at ±3 in
   the log-uniform grid leaves 25.6% of normalized weights in a region with 2× higher max
   quantization error than a uniform 4-bit quantizer (4.48× MSE ratio).

2. **No 7-value grid escapes the gap.**  Uniform Grid B `{−3..+3}` and asymmetric Grid C
   `{−4..+3}` are both *worse* than the log-uniform baseline.  Outlier coverage and gap
   coverage are incompatible objectives at 3-bit precision.

3. **Weight-MSE scale optimisation backfires** (+13.6 PPL at bs=128).  Minimising
   `‖W − Q(W)‖²` clips high-magnitude weights that dominate model output — an empirical
   rediscovery of the GPTQ/AWQ motivation.

4. **AWQ recovers 22% at isocompression**, and calibration saturates at 30 windows (15k
   tokens, ~1 second).  No benefit from additional calibration data.

---

## Repository structure

```
PTQ/
├── ptq/
│   ├── quantize.py          # Grid A/B/C/9v quantization + MSE calibration
│   ├── quantized_linear.py  # Drop-in nn.Linear replacement
│   ├── shift_matmul.py      # Dequantize-and-multiply + pure shift reference
│   ├── awq.py               # AWQ diagonal-Hessian adaptation
│   ├── calibrate.py         # Activation scale collection (forward hooks)
│   ├── model_wrapper.py     # Model-level layer replacement
│   └── utils.py             # Memory footprint accounting
├── bench/
│   ├── perplexity.py        # WikiText-103 PPL (non-overlapping 2048-token windows)
│   └── run_benchmark.py     # CLI: all grids, block sizes, AWQ, calibration
├── analysis/
│   └── diagnose.py          # H1/H2/H3 diagnostic scripts
├── paper/
│   ├── abstract.md
│   ├── introduction.md
│   ├── related_work.md
│   ├── method.md
│   ├── results.md
│   ├── conclusion.md
│   ├── build_pdf.py         # Assembles sections → PDF via WeasyPrint
│   └── shiftquant.pdf       # Compiled paper
└── tests/                   # 66 unit tests
```

---

## Usage

```bash
# Install dependencies
uv sync   # or: pip install -r requirements.txt

# FP16 baseline + Grid A across block sizes
python -m bench.run_benchmark --model Qwen/Qwen2-1.5B

# Full experiment: all grids + AWQ
python -m bench.run_benchmark \
    --model Qwen/Qwen2-1.5B \
    --block-sizes 32 \
    --grids A B C 9v \
    --awq --awq-grids A 9v \
    --calib-samples 30

# MSE calibration ablation
python -m bench.run_benchmark --model Qwen/Qwen2-1.5B --calibrated

# Run tests
pytest tests/
```

---

## Paper

The full paper draft is in `paper/`.  To rebuild the PDF:

```bash
uv tool install weasyprint markdown
python paper/build_pdf.py
```

---

## Citation

```bibtex
@misc{shiftquant2026,
  title   = {ShiftQuant: Analyzing the Limits of Shift-Based Post-Training
             Quantization for LLMs},
  author  = {Anonymous},
  year    = {2026},
  url     = {https://github.com/Kyworn/ShiftQuant}
}
```
