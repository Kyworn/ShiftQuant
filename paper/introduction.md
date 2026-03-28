# Introduction

The computational cost of large language model inference has motivated a rich line
of research in post-training quantization (PTQ): reducing weight precision from FP16
to 8, 4, or fewer bits without retraining the model.  Most production deployments
now use 4-bit or 8-bit integer formats (Dettmers et al., 2022; Frantar et al., 2022;
Lin et al., 2023), achieving 2–4× memory compression with modest perplexity loss.

A more aggressive alternative is *shift-based quantization*, which restricts weights
to exact powers of two:

```
G_A = {−4, −2, −1, 0, +1, +2, +4}
```

Every nonzero value in this set is a power of two (±2^k, k ∈ {0, 1, 2}).  The
arithmetic consequence is that weight–input multiplication reduces to a bit-shift and
a conditional negation — no multiplier circuit is required.  This property is
attractive for edge inference on microcontrollers, FPGAs, and future multiply-free
accelerators (Elhoushi et al., 2019; Ma et al., 2024).

**The gap in existing work.**  Prior shift-based methods (DeepShift, PoT-QAT, APoT)
are almost exclusively *quantization-aware training* (QAT) approaches: the model is
retrained or fine-tuned with simulated shift quantization.  Post-training shift
quantization — applying the constraint to a pretrained LLM with no weight update —
has received comparatively little systematic study.  Crucially, the *failure modes*
of one-shot shift PTQ have not been characterised, which makes it difficult to know
whether the gap to FP16 is fundamental or merely an artefact of suboptimal calibration.

**This paper.**  We provide a systematic empirical analysis of shift-based PTQ applied
to Qwen2-1.5B (Qwen Team, 2023) evaluated on WikiText-103.  Our contributions are:

1. **Baseline and diagnostic** (§4.1–4.2).  We establish that a 7-value log-uniform
   shift grid at block size 32 degrades perplexity by +30.8% relative to FP16
   (+2.95 PPL, 9.58 → 12.53).  We decompose this degradation into three hypotheses
   — outlier-driven scale inflation (H1), structural grid gap at ±3 (H2), and
   layer-specific error concentration (H3) — and show that H2 is the dominant factor,
   explaining ~31% of the degradation through a 4.48× MSE ratio versus a uniform
   4-bit quantizer.

2. **Negative result: 3-bit grid incompatibility** (§4.3).  We prove empirically that
   no 7-value grid can simultaneously cover outliers (levels at ±4) and fill the ±3
   gap (a level at ±3).  Alternative grids B (uniform {−3..+3}) and C (asymmetric
   8-value) both perform *worse* than the baseline.  The trade-off is intrinsic to
   3-bit precision.

3. **Negative result: weight-MSE scale optimisation** (§4.4).  A per-block scale
   search minimising ‖W − Q(W)‖² consistently *worsens* PPL by up to +13.6 PPL at
   block size 128.  We show this empirically validates the motivation behind
   GPTQ (Frantar et al., 2022) and AWQ (Lin et al., 2023): unweighted MSE is the
   wrong objective because high-magnitude weights contribute disproportionately
   to model output.

4. **AWQ adaptation** (§4.5).  We adapt the AWQ diagonal-Hessian approximation to
   the shift grid.  AWQ-A (isocompression) recovers 22% of the PPL gap (−0.65 PPL).
   Combining AWQ with a 4-bit uniform 9-value grid (AWQ-9v) recovers 50% (−1.48 PPL,
   reaching 11.06).

5. **Orthogonality of grid and calibration effects** (§4.6).  The grid improvement
   (9v vs A: −0.92 PPL) and the AWQ improvement (−0.65 PPL) combine with 93.6%
   orthogonality: the observed improvement (−1.47) is 94% of the theoretical maximum
   (−1.57) under full independence.  This means the two axes can be optimised
   independently — a practically useful property for systematic quantization research.

6. **Calibration efficiency** (§4.7).  AWQ results are insensitive to calibration
   data volume: 30 windows (15k tokens, ~1 second) gives the same PPL as 120 windows
   (61k tokens), with ≤ 0.04 PPL difference.

**Scope and limitations.**  This work is confined to PTQ (no weight updates), a
single 1.5B-parameter model, and weight quantization of linear layers only.  We do
not study activation quantization, KV-cache compression, or models above 3B
parameters.  The findings on grid structure and the MSE calibration failure mode are
likely to generalise (they follow from mathematical properties of the grid), while
the specific PPL numbers are model-dependent.

**Paper structure.**  Section 2 reviews related work.  Section 3 formalises the
shift quantization framework and our AWQ adaptation.  Section 4 presents all
experimental results.  Section 5 concludes.
