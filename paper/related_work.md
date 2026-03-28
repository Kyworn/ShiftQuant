# Related Work

## Shift-based and power-of-two quantization

**DeepShift** (Elhoushi et al., 2019) replaces all multiplications in a neural
network with bit-shifts and sign flips, representing weights as `2^p × s` where
`p ∈ Z` and `s ∈ {−1, +1}`.  Training uses a straight-through estimator for the
discrete shift operation.  DeepShift targets full retraining rather than
post-training quantization and focuses on CNNs rather than transformer LLMs.

**APoT** (Li et al., 2019 — Additive Powers-of-Two) extends the shift idea to sums
of two power-of-two terms, enabling richer grids (e.g. encoding {0, 1, 2, 3, 4} as
combinations 1+2, 1+1, 1, 0) while keeping inference multiply-free.  APoT demonstrates
that non-uniform power-of-two grids outperform uniform quantization in the QAT setting.
Our work tests the same structural hypothesis in the one-shot PTQ setting and finds
the opposite: the non-uniform structure of Grid A creates a structural gap that
*worsens* PTQ quality relative to a uniform grid at the same bit width.

**PoT-QAT** (Elgenedy et al., 2026) is a recent power-of-two quantization-aware
training method showing that fine-tuned shift grids can match INT4 quality on
transformer models.  The key distinction relative to our work is the gradient-based
weight update: PoT-QAT spends training compute to adapt the model to the grid, while
we apply the same grid post-training with zero weight update.  Our results quantify
the PTQ penalty for this choice: +30.8% PPL at 3 bits vs. FP16.

**BitNet** (Wang et al., 2023; Ma et al., 2024) pushes shift quantization to the extreme:
ternary weights {−1, 0, +1} trained from scratch.  BitNet-b1.58 achieves competitive
perplexity at 1.58 bits per weight with full retraining.  Our work is complementary in
scope: we study what is achievable at 3 bits without any training.

The shared limitation of DeepShift, APoT, PoT-QAT, and BitNet relative to our setting
is the requirement for gradient-based optimization.  To our knowledge, systematic
characterisation of one-shot shift PTQ for LLMs — including failure modes and the
effectiveness of calibration-based corrections — has not been previously reported.

## Post-training quantization for LLMs

**GPTQ** (Frantar et al., 2022) is a layer-wise second-order PTQ method that uses the
inverse Hessian of the quantization error (computed from calibration data) to update
remaining unquantized weights, compensating for already-quantized ones.  GPTQ achieves
near-lossless 4-bit quantization for 7B+ parameter models.  Our §4.4 result —
that per-block MSE-optimal scaling *worsens* PPL — experimentally validates the core
motivation of GPTQ: the unweighted ‖W − Q(W)‖² objective is inappropriate because it
treats all weights equally regardless of their activation magnitude.

**AWQ** (Lin et al., 2023 — Activation-aware Weight Quantization) identifies "salient"
input channels (large mean activation magnitude) and scales them before quantization,
dividing the corresponding inputs at inference.  The per-channel scale α = s_x^δ is
searched to minimise an activation-weighted reconstruction error.  AWQ is
isocompression: α is absorbed into the weight matrix, adding no additional memory.
We adapt AWQ to the shift grid in §3.3 and show it recovers 22% of the PPL gap
at isocompression, with effects that are 94% orthogonal to grid choice.

**LLM.int8()** (Dettmers et al., 2022) characterised the outlier problem in transformer
attention and proposed mixed-precision decomposition (FP16 for outlier channels, INT8
for the rest) as a fix.  We adopt a similar diagnostic methodology: formulate discrete
hypotheses (H1/H2/H3), test each independently, and attribute improvement to specific
mechanisms rather than treating the system as a black box.

**SqueezeLLM** (Kim et al., 2023) uses mixed-precision quantization: most weights at
3 bits, a small fraction of sensitive weights at full precision.  This is complementary
to our work — mixed precision addresses H3 (layer-specific sensitivity) while our
grid and AWQ improvements address H2 (structural gap).

## Practical deployment formats

**GGUF / llama.cpp IQ4_NL** (Gerganov et al.) implements importance-matrix-weighted
quantization with non-linear codebooks optimised empirically for LLM weight
distributions.  IQ4_NL achieves strong PPL at 4 bits on consumer hardware and is
among the dominant deployment formats.  Our work targets the 3-bit regime with an
additional multiply-free constraint that GGUF formats do not impose.  The comparison
is instructive: the inability to use a 3-bit shift grid at the same quality as IQ4_NL
motivates understanding exactly *why* the gap exists and whether calibration can close it.

## Non-uniform quantization

**NF4** (Dettmers et al., 2023 — NormalFloat) spaces quantization levels according to
expected quantiles of a standard normal distribution.  NF4 outperforms uniform 4-bit
quantization because levels are concentrated where weight density is highest.  Grid A
shares the non-uniform philosophy but imposes the additional constraint that all levels
be powers of two.  §4.2 (H2) documents the cost: the power-of-two constraint creates
a gap at ±3 where 25.6% of weights reside, with 2× higher maximum quantization error
than a uniform quantizer in the same region.
