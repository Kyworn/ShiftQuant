# Conclusion

We presented a systematic analysis of shift-based post-training quantization applied
to Qwen2-1.5B on WikiText-103.  The central question was: *what are the fundamental
limits of one-shot shift PTQ, and where do the losses come from?*

**What we found.**  The 7-value log-uniform shift grid incurs a +30.8% PPL penalty at
block size 32 (+2.95 PPL, 9.58 → 12.53).  Diagnosis identifies the dominant cause as
a structural property of the grid: the power-of-two constraint places consecutive
levels at 2 and 4, leaving a gap at ±3 where 25.6% of normalized weights reside with
twice the maximum quantization error of a uniform 4-bit quantizer (H2).  This is not
a failure of the specific implementation — it is a consequence of the requirement that
all levels be exact powers of two.

Attempts to fix the gap directly fail:

- *No 7-value grid* can simultaneously cover outliers at ±4 and fill the gap at ±3.
  Uniform Grid B and asymmetric Grid C are both *worse* than the log-uniform baseline.
- *Weight-MSE scale optimisation* consistently *worsens* PPL (up to +13.6 PPL at
  block size 128), because minimising ‖W − Q(W)‖² penalises all weights equally while
  high-magnitude weights dominate model output.

These two negative results are the most technically informative findings: they locate
the problem precisely and rule out the most natural solutions.

**What works.**  The AWQ diagonal-Hessian approximation adapted to the shift grid
recovers 22% of the PPL gap at isocompression (AWQ-A: +2.30 PPL, −0.65 vs baseline).
Combining AWQ with a 9-value uniform grid that fills the ±3 gap (AWQ-9v) recovers 50%
(+1.48 PPL).  The grid improvement and the AWQ improvement are 93.6% orthogonal —
they address different parts of the error distribution and can be optimised independently
without significant interaction loss.  AWQ calibration is robust: 30 windows (15k tokens,
~1 second) gives the same result as 120 windows (Δ ≤ 0.04 PPL).

**Practical recommendation.**  For a practitioner deploying shift-based PTQ today:

- Use block size 32 (negligible memory cost over bs=128, meaningful PPL gain).
- Apply AWQ with 30–60 calibration windows; additional data provides no benefit.
- If multiply-free inference is the strict constraint: AWQ-A (+2.30 PPL) is the ceiling.
- If a single additional bit is acceptable: AWQ-9v (+1.48 PPL) with a uniform 9-value
  grid achieves 50% PPL recovery at the same block-scale memory cost.

**Limitations.**  This study is confined to a single 1.5B-parameter model.  PPL
numbers are model-dependent; the structural findings (grid gap theorem, MSE calibration
failure) follow from mathematical properties and are expected to generalise.  We do not
evaluate activation quantization, KV-cache compression, or the actual runtime benefit
of shift inference (no fused CUDA kernel is provided).

**Future work.**  The most important open questions are:

1. *Larger models.*  The PPL penalty of shift PTQ likely decreases with scale, as
   larger models are known to be more robust to weight perturbation.  Testing AWQ-9v
   on Qwen2-7B or Llama-3-8B would clarify whether the 50% recovery ceiling holds.

2. *Layer-wise sensitivity (H3).*  The 15 worst layers are `v_proj` and `k_proj`
   projections with shape (256 × 1536) — GQA projections with limited statistical
   redundancy.  Mixed precision (keeping these layers in FP16 or 4-bit while shifting
   the rest) is the natural next step.

3. *Fused shift kernel.*  A CUDA kernel implementing `y = Σ_b s_b × (x_b << k_b)`
   directly would demonstrate the real throughput benefit of multiply-free inference.
   On current GPU hardware the benefit is limited (GPUs have fast multipliers), but
   the approach is relevant for edge ASICs and FPGAs where multiplier area dominates.

4. *3-bit packing.*  Int8 storage wastes 5 bits per weight.  True 3-bit packing would
   reduce weight memory by 62.5%, from 2.17 GB to ~0.97 GB — a 3× compression over FP16.
   Combined with AWQ-9v, this is a practically compelling point on the PPL–memory curve.
