# Abstract

We present a systematic empirical analysis of shift-based post-training quantization
(PTQ) for large language models.  Shift-based quantization restricts weights to
exact powers of two — enabling multiply-free inference via bit-shifts — but has been
studied almost exclusively in quantization-aware training settings.  We apply a
7-value log-uniform shift grid {−4, −2, −1, 0, +1, +2, +4} post-training to
Qwen2-1.5B and evaluate perplexity on WikiText-103, reporting four main findings.

(1) **Baseline cost.**  The shift grid degrades perplexity by +30.8% at block size 32
(+2.95 PPL, 9.58 → 12.53).  Systematic diagnosis identifies the dominant cause as
a structural gap at ±3: the power-of-two constraint places no level between 2 and 4,
leaving 25.6% of normalized weights in a region with twice the maximum quantization
error of a uniform 4-bit quantizer (4.48× higher MSE).

(2) **Negative result: grid incompatibility.**  No 7-value grid can simultaneously
cover outliers at ±4 and fill the gap at ±3.  Alternative grids with uniform spacing
(Grid B) or 8-value asymmetric coverage (Grid C) both perform worse than the
log-uniform baseline, establishing an intrinsic trade-off at 3-bit precision.  Nor can
per-block MSE-optimal scale search close the gap: it consistently worsens PPL (up to
+13.6 PPL at block size 128) by clipping high-magnitude weights that dominate output
error — an empirical rediscovery of the motivation behind GPTQ and AWQ.

(3) **AWQ recovery.**  Adapting AWQ's diagonal-Hessian approximation to the shift grid
recovers 22% of the PPL gap at isocompression (AWQ-A: +2.30 PPL).  Calibration
requires only 30 windows (15,360 tokens), with no measurable benefit from additional data.

(4) **Orthogonality.**  A 9-value uniform grid (filling the ±3 gap, costs 1 extra bit)
and AWQ activation weighting (isocompression) improve PPL by −0.92 and −0.65 respectively,
with 93.6% orthogonality.  Their combination (AWQ-9v) recovers 50% of the FP16 gap
(+1.48 PPL).  The near-independence of the two axes means grid design and scale
calibration can be optimised separately without significant interaction loss.
