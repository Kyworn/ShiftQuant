# Results

All experiments use Qwen2-1.5B (FP16 baseline PPL = 9.577) evaluated on the
WikiText-103 test set with non-overlapping 2048-token windows. Quantization
targets linear layers only; embeddings and `lm_head` remain in FP16.
Memory figures count quantized weights (int8) + per-block FP16 scales;
theoretical 3-bit packing would yield ≈5× compression.

---

## 4.1  Baseline: block size sweep (Grid A)

**Grid A** is the natural shift-quantization grid {−4, −2, −1, 0, +1, +2, +4}.
All seven values are exact powers of two (or zero), enabling a multiply-free
inference kernel.  The per-block scale is `s = max(|block|) / 4`.

**Table 1.** PPL and memory for Grid A across block sizes.

| Block size | PPL    | Δ PPL  | Memory | Compression |
|:----------:|:------:|:------:|:------:|:-----------:|
| FP16       |  9.577 |  —     | 2.88 GB | 1.00×      |
| 32         | 12.526 | +2.949 | 2.17 GB | 1.53×      |
| 64         | 13.744 | +4.167 | 2.13 GB | 1.56×      |
| 128        | 15.338 | +5.761 | 2.11 GB | 1.57×      |

Smaller blocks recover more PPL at negligible memory cost (6 MB difference
between bs=32 and bs=128 on a 1.5B model). All subsequent experiments use
**bs=32** as the reference configuration.

The +30.8% degradation at bs=32 is the central question driving the rest of
this section.

---

## 4.2  Diagnostic: identifying the failure mode

We formulate three hypotheses for the degradation and test each independently.

### H1 — Outlier-driven scale inflation

An outlier in a block forces a large scale, compressing all other weights into
a narrow normalized range.  We define an *outlier block* as one where
`max(|w|) > 5 × median(|w|)`.

**Table 2.** Outlier block statistics.

| Block size | Outlier blocks | % outlier | Avg scale utilization |
|:----------:|:--------------:|:---------:|:---------------------:|
| 32         | 6,288,571      | 15.4%     | 83.3%                 |
| 64         | 3,895,374      | 19.0%     | 85.7%                 |
| 128        | 2,706,750      | 26.4%     | 87.3%                 |

Scale utilization (second-largest `|w|` / largest `|w|` per block) averages
83–87%, indicating that the scale is not wholly dominated by a single outlier.
**H1 is real but not catastrophic.** It explains why smaller blocks help
(fewer co-residents per outlier) but does not fully explain the +30.8% gap.

### H2 — Non-uniform grid leaves a structural gap at ±3

Grid A's levels {0, 1, 2, 4} have thresholds at 0.5, 1.5, and **3.0**.
The interval [1.5, 3.0) maps to the level 2, with a maximum absolute
quantization error of **1.0** — twice the error of a uniform 4-bit quantizer
in the same region.

**Table 3.** Distribution of normalized weights (`|w/s|`) and their maximum
quantization error, at block size 64 (Grid A, all layers of Qwen2-1.5B).

| Normalized range | Weight fraction | Max quant error | Grid level |
|:----------------:|:---------------:|:---------------:|:----------:|
| [0, 0.5)         | 27.2%           | 0.50            | 0          |
| [0.5, 1.5)       | 41.4%           | 0.50            | ±1         |
| **[1.5, 3.0)**   | **25.6%**       | **1.00**        | **±2**     |
| [3.0, 4.0]       |  5.9%           | 1.00            | ±4         |

**31.4% of weights** fall in zones with maximum error 1.0 (the [1.5, 3.0) and
[3.0, 4.0] intervals combined). By comparison, a symmetric uniform 4-bit
quantizer has maximum error ≤ 0.5 everywhere.  Averaging across all 196
quantized layers:

```
Mean shift-grid MSE  :  2.7 × 10⁻⁵
Mean uniform-4bit MSE:  0.6 × 10⁻⁵
Ratio                :  4.48×
```

**H2 is the dominant failure mode.**  The log-uniform spacing of Grid A
concentrates levels near zero (where weight density is low) and leaves a large
gap at ±3 (where 25.6% of weights reside).

### H3 — Layer-specific error concentration

Sorting layers by relative reconstruction error (`‖W − Q(W)‖ / ‖W‖`) reveals
that the 15 worst layers are all `v_proj` and `k_proj` projections with shape
(256 × 1536) — the grouped-query attention (GQA) projections.  Their small
output dimension (256 rows vs. 1536 for `q_proj`) offers less statistical
redundancy, making each weight error more consequential.  This pattern is
consistent with prior work on attention projection sensitivity
(Dettmers et al., 2022).

---

## 4.3  Grid ablation: can 3 bits fix the ±3 gap?

The H2 result motivates a direct test: does filling the ±3 gap with a
different grid improve PPL?

We evaluate three alternative 3-bit grids against Grid A and the 9-value
4-bit reference:

- **Grid A** (baseline): {−4, −2, −1, 0, +1, +2, +4} — log-uniform, 7 values, 3 bits
- **Grid B**: {−3, −2, −1, 0, +1, +2, +3} — uniform, 7 values, 3 bits, scale = max/3
- **Grid C**: {−4, −3, −2, −1, 0, +1, +2, +3} — asymmetric, 8 values, **3 bits exact** (2³)
- **Grid 9v**: {−4, −3, −2, −1, 0, +1, +2, +3, +4} — uniform, 9 values, 4 bits (control)

**Table 4.** Grid ablation at block size 32.

| Grid | Values | Bits | PPL    | Δ PPL  | vs. baseline |
|:----:|:------:|:----:|:------:|:------:|:------------:|
| A    | 7      | 3    | 12.526 | +2.949 | —            |
| B    | 7      | 3    | 14.214 | +4.637 | **−1.69** worse |
| C    | 8      | 3    | 13.596 | +4.019 | **−1.07** worse |
| 9v   | 9      | 4    | 11.610 | +2.033 | **+0.92** better |

**Grid B is worse than A.**  Although B has uniform spacing, it clips outliers
at ±3×scale (versus ±4×scale for A).  The outlier coverage of Grid A outweighs
the benefit of filling the ±3 gap.

**Grid C is also worse than A.**  The 8-value asymmetric grid protects negative
outliers (reaching −4) but systematically clips positive weights above +3×scale
(75% of the maximum).  Without a structural asymmetry in the weight distribution
of Qwen2-1.5B, the clipping cost dominates.

**This establishes a negative result:** *no 7-value grid can simultaneously
provide outlier coverage (levels at ±4) and fill the ±3 gap (a level at ±3).*
The two objectives are incompatible at 3-bit precision.  The 9v control (+0.92
PPL improvement over A) confirms that the ±3 gap accounts for approximately
**31% of the total PPL degradation**, matching the weight fraction in Table 3.

---

## 4.4  Negative result: weight-MSE scale optimization

Motivated by the H2 diagnosis, we attempted to compensate for the non-uniform
grid by finding the per-block scale `s*` that minimises reconstruction MSE
(instead of using `max(|block|)/4`).  We performed a grid search over 100
candidates in `[0.1 × s₀, 1.5 × s₀]` (log-uniform), where `s₀ = max/4`.

The optimal scale shifts to `α < 1` (mean α = 0.825, 96% of blocks), clipping
large weights to improve coverage of the dense [0, 2] region.

**PPL results (Grid A, block size 32/64/128):**

| Block size | max/4  | MSE-optimal | Direction |
|:----------:|:------:|:-----------:|:---------:|
| 32         | 12.526 | 12.896      | **worse** |
| 64         | 13.744 | 18.627      | **worse** |
| 128        | 15.338 | 28.909      | **worse** |

MSE-optimal scales consistently worsen PPL, with degradation growing with
block size.  The reason is fundamental: minimising `‖W − Q(W)‖²` penalises
all weights equally, yet large-magnitude weights contribute
disproportionately to the model output `Wx`.  By clipping them to improve
average MSE, the scale optimisation increases the output error on precisely
the weights that matter most.

**This result experimentally validates the motivation for GPTQ-style
(Frantar et al., 2022) and AWQ-style (Lin et al., 2023) methods**, which
replace the unweighted MSE with a Hessian-weighted or activation-weighted
objective.  Our shift-quantized setting reproduces this finding in a novel
and particularly clean form: the effect is amplified by the non-uniform grid,
making it observable even at small scale.

---

## 4.5  AWQ-style activation-aware quantization

We implement a diagonal-Hessian approximation of AWQ for the shift grid.
For each linear layer with weight matrix `W ∈ R^{out × in}` and input
activation statistics `s_x ∈ R^{in}` (mean `|x_j|` over 120 calibration
windows, 61,440 tokens):

1. Search over `δ ∈ [0, 1]` (20 grid points):
   `α(δ) = s_x^δ`
2. Scale weights:  `W_new = W · diag(α)`
3. Quantize `W_new` with Grid A or 9v
4. At inference: `y = Q(W_new) · (x / α)`

The objective minimises the activation-weighted reconstruction error:

```
E(δ) = ‖ (Q(W_new) − W_new) · diag(s_x / α) ‖²_F
```

which down-weights channels where α absorbs the activation magnitude,
protecting high-activation channels from large quantization error.

**Table 5.** Full comparison at block size 32 (all methods, same compression).

| Method         | Grid | Calibration | PPL    | Δ PPL  | Recovery |
|:--------------:|:----:|:-----------:|:------:|:------:|:--------:|
| FP16           | —    | —           |  9.577 |  —     |   —      |
| Shift PTQ      | A    | none        | 12.526 | +2.949 |   0%     |
| Shift PTQ      | 9v   | none        | 11.610 | +2.033 |  31.0%   |
| AWQ            | A    | 120 windows | 11.877 | +2.300 |  22.0%   |
| **AWQ**        | **9v** | **120 windows** | **11.059** | **+1.482** | **49.8%** |

**AWQ alone recovers 22% of the PPL gap** (−0.65 PPL, from +2.95 to +2.30),
isocompression, using 61k tokens of calibration text.

---

## 4.6  Orthogonality of grid and AWQ effects

The key structural result of this work is the near-independence of the two
improvement axes:

```
Δ_grid (9v vs A)      = −0.916  [fills ±3 gap, costs 1 bit]
Δ_AWQ  (AWQ-A vs A)   = −0.649  [activation weighting, isocompression]
─────────────────────────────────────────────────────────────────────
Sum (if 100% orthogonal)         = −1.565
Observed (AWQ-9v vs A)           = −1.467
Orthogonality                    =  93.6%
```

**Figure 1 (described).** Waterfall chart: FP16 (9.58) → Grid A (12.53) →
arrow −0.92 labeled "Fill ±3 gap (9v, +1 bit)" → 11.61 → arrow −0.65
labeled "AWQ activation weighting (isocompression)" → 10.96 ≈ 11.06 observed.
Residual gap to FP16 shaded.

The 6.4% non-orthogonal fraction arises because both methods partly address
the same underlying issue in outlier-heavy blocks: a large activation on a
channel that also happens to produce a high-error ±3 normalized weight.
This shared variance is small but measurable.

**The practical implication** is that grid design and scale calibration can be
optimised independently without significant interaction loss — a useful
property for systematic quantization research.

---

## 4.7  Calibration cost: how many windows does AWQ need?

A practical concern for AWQ deployment is the amount of calibration data required.
We compare AWQ-A and AWQ-9v at 30 windows (15,360 tokens, 1 second of collection)
versus 120 windows (61,440 tokens, reported in §4.5).

**Table 6.** AWQ sensitivity to calibration data size (block size 32).

| Method  | 30 windows | 120 windows | Δ PPL |
|:-------:|:----------:|:-----------:|:-----:|
| AWQ-A   | 11.835     | 11.877      | −0.042 |
| AWQ-9v  | 11.021     | 11.059      | −0.038 |

The difference is ≤ 0.04 PPL in both cases — below the noise floor of our
evaluation setup.  The slight advantage of the smaller sample (marginally lower
PPL) is not statistically meaningful.

**Interpretation:** per-channel mean activation magnitudes converge within
~15k tokens.  Channel importance is a structural property of the model
architecture (e.g., which channels are systematically amplified by the
attention mechanism), not a function of the specific text content.  This is
consistent with the observation in Lin et al. (2023) that AWQ calibration
requires only 128 samples at 512 tokens = ~65k tokens and is insensitive to
domain mismatch.

**Practical implication:** 30 calibration windows (≈1 second on an RTX 5080)
are sufficient.  The full 120-window budget is unnecessary; even a small
held-out subset of the target domain suffices.

---

## Summary of findings

| Finding | Evidence | Implication |
|:--------|:---------|:------------|
| Shift-grid PPL gap is +30.8% at bs=32 | Table 1 | Baseline cost of multiply-free inference |
| H1 (outliers) is real but secondary | Table 2, 15% outlier blocks | Smaller blocks help, not sufficient |
| H2 (±3 gap) explains ~31% of degradation | Tables 3–4, 4.48× MSE ratio | Structural limit of 3-bit log-uniform grid |
| No 7-value grid escapes the coverage/gap tradeoff | Table 4, grids B and C | Negative result: problem is intrinsic to 3 bits |
| Weight-MSE optimisation backfires | §4.4 | Validates GPTQ/AWQ motivation empirically |
| AWQ recovers 22% isocompression | Table 5 | Activation weighting works for shift grids |
| Grid + AWQ effects are 94% orthogonal | §4.6 | Both can be optimised independently |
| Best 3-bit result: AWQ-A at +2.30 PPL | Table 5 | Practical ceiling without extra bits |
| Best combined: AWQ-9v at +1.48 PPL | Table 5 | 50% recovery, costs 4 bits |
| AWQ calibration converges at 15k tokens | Table 6 | 30 windows = 120 windows (±0.04) |
