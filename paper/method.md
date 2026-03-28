# Method

## 3.1  Shift-based weight quantization

Let `W ∈ R^{out × in}` be a weight matrix of a linear layer.  We partition the `in`
dimension into non-overlapping blocks of size `B`, padding to a multiple of `B` with
zeros if necessary.  For each block `b` of `B` consecutive weights along the input
dimension, we compute a per-block scale:

```
s_b = max(|w_i|  for i ∈ block b) / 4
```

and normalize: `w̃_i = w_i / s_b`.  We then round each normalized weight to the nearest
value in the **shift grid** G:

```
G_A = {−4, −2, −1, 0, +1, +2, +4}
```

The rounding function `round_G(x)` is threshold-based (branchless):

| `|x|` range     | Nearest level | Max absolute error |
|:---------------:|:-------------:|:-----------------:|
| [0, 0.5)        | 0             | 0.50              |
| [0.5, 1.5)      | ±1            | 0.50              |
| [1.5, 3.0)      | ±2            | 1.00              |
| [3.0, ∞)        | ±4            | 1.00              |

The quantized weight is `q_i = round_G(w̃_i) ∈ G_A`, stored as int8.  The per-block
scale `s_b` is stored as float16.  Memory cost: 1 byte per weight (int8) + 1 float16
scale per `B` weights = `in × out + (in × out / B) × 2` bytes, vs. `in × out × 2`
bytes for FP16.  Compression ratio: `2 / (1 + 2/B)` → approaches 2× as B grows.

**Inference.**  Since every nonzero value in G_A is ±2^k for k ∈ {0, 1, 2}, the
product `q_i × x_j` reduces to a conditional bit-shift and negation:
`x_j << k` if q_i = ±2^k, negated if q_i < 0.  No multiplier circuit is required.
In practice, we implement inference via on-the-fly dequantization + `F.linear` for
benchmarking, deferring a fused shift kernel to future work.

## 3.2  Alternative grids

We evaluate three alternative grids to isolate the effect of specific design choices:

```
G_B = {−3, −2, −1, 0, +1, +2, +3}   (uniform 7-value, scale = max/3)
G_C = {−4, −3, −2, −1, 0, +1, +2, +3}  (asymmetric 8-value = 2³, scale = max/4)
G_9v = {−4, −3, −2, −1, 0, +1, +2, +3, +4}  (uniform 9-value, scale = max/4)
```

Note that G_B does not require multiply-free inference (±3 is not a power of two).
G_C and G_9v likewise include ±3.  The purpose of grids B, C, 9v is diagnostic: they
let us measure the PPL cost of the gap at ±3 in G_A, not to propose practical
alternatives to the multiply-free constraint.

## 3.3  AWQ adaptation for shift grids

We adapt the Activation-aware Weight Quantization method (Lin et al., 2023) to
arbitrary shift grids.  The key insight of AWQ is that input channels with large
mean activation magnitude s_x[j] = E[|x_j|] contribute disproportionately to the
output error: a quantization error δW[:, j] is amplified by s_x[j] in the output.

**Smooth scaling.**  For each linear layer with weight matrix `W` and per-channel
activation statistics `s_x ∈ R^{in}` (collected on calibration data), we introduce
a per-channel scale `α ∈ R^{in}` and transform:

```
W_new[:, j] = W[:, j] × α[j]           (scale weights up)
y = Q(W_new) × (x / α)  ≈  W × x      (divide input at inference)
```

Since `α` multiplies a column of `W` and divides the corresponding scalar input,
the transformation is mathematically transparent: the model output is unchanged if
`Q(W_new) = W_new` (i.e., if quantization error is zero).

**Scale search.**  We parametrize `α = s_x^δ` for `δ ∈ [0, 1]`, so `δ = 0`
is a no-op and `δ = 1` gives full activation-scale weighting.  The optimal `δ` is
found by grid search over `n_δ = 20` evenly spaced values, minimising:

```
E(δ) = ‖ (Q(W_new) − W_new) · diag(s_x / α) ‖²_F
```

This is a diagonal-Hessian approximation: the factor `s_x[j] / α[j]` weights the
column-wise reconstruction error by the effective sensitivity of channel `j` after
scaling.  Channels where `α[j]` absorbs most of the activation magnitude contribute
less to the objective, incentivising the search to protect high-activation channels.

**Calibration data.**  We collect `s_x` by running the unquantized model on `n_cal`
non-overlapping windows of 512 tokens from the calibration corpus and recording
the mean absolute activation magnitude per channel via forward hooks.  We use
`n_cal = 30` unless otherwise noted (§4.7).

**Inference.**  `AWQLinear` stores the quantized weight matrix `Q(W_new)` (int8)
and per-block scales (float16) via `QuantizedLinear`, plus the input scale vector
`α` (float32, one scalar per input channel).  At inference:

```python
y = QuantizedLinear(x / α)
```

The division `x / α` is a broadcast elementwise division over the `in_features`
dimension, adding negligible overhead.  Memory: `α` costs `in_features × 4` bytes
per layer ≈ 0.3% of total weight memory for Qwen2-1.5B.

**Skipped layers.**  `embed_tokens` and `lm_head` remain in FP16 in all experiments.
These layers account for ~18% of Qwen2-1.5B parameters but are critical for token
embedding stability and, in the case of `lm_head`, are typically weight-tied to the
embedding matrix.

## 3.4  Memory accounting

We report theoretical memory as:

```
total = quantized_weight_bytes + scale_bytes + fp16_layer_bytes
```

where `quantized_weight_bytes = in × out × 1` (int8), `scale_bytes = (in × out / B) × 2`
(float16), and `fp16_layer_bytes` covers embeddings, `lm_head`, biases, and
LayerNorm parameters.  We do not pack int8 weights to 3 bits; the reported compression
ratios reflect int8 storage.  Theoretical 3-bit packing would yield ≈2.7× additional
compression of the weight bytes, reducing total model size by approximately 40% further.
All experiments store and compute in int8/FP16; no 3-bit kernel is evaluated.
