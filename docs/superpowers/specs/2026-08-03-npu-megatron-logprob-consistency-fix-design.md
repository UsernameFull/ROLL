# NPU Megatron Logprob Consistency Fix

## Goal

Restore numerical consistency between Megatron actor/reference forward passes and vLLM rollout inference on Ascend NPU, and make the associated PPO diagnostics mathematically valid under masking and data parallelism.

## Scope

This change fixes three confirmed defects:

1. The NPU Megatron local path converts a token-validity mask directly to a four-dimensional boolean attention mask. Megatron interprets `True` as masked, so valid tokens are hidden and no causal mask is constructed.
2. `agg_loss(..., loss_agg_mode="token-mean")` does not multiply by `loss_mask`, allowing excluded tokens to contribute to losses and metrics.
3. RLVR logprob diagnostics retain only rank 0 metadata while some values use global denominators, producing misleading values such as an initial PPO ratio mean of `0.5` with DP size 2.

No FP8/MXFP8 quantization algorithm or model-update format will be changed.

## Design

### NPU local attention mask

Add a small pure helper in the Megatron strategy module that converts a two-dimensional token-validity mask (`1/True = valid`) into Megatron's four-dimensional mask (`True = blocked`). The result is the union of:

- an upper-triangular causal mask; and
- a key-padding mask derived from invalid token positions.

Use this helper only for the existing NPU local-transformer branch. The Transformer Engine branch continues to pass `None`, preserving its current fused causal-attention behavior.

### Token-mean aggregation

Change the token-mean numerator to multiply `loss_mat`, per-sample weights, and `loss_mask`. Keep the existing denominator behavior so callers that supply a global token count continue to reduce correctly across ranks.

### Data-parallel diagnostics

Mark the RLVR diagnostics metadata key as global during actor train result concatenation. Flatten the per-rank diagnostic lists before logging. Diagnostics will therefore show one record per participating rank rather than silently presenting rank 0 as a global value. Existing regular `metrics` aggregation remains unchanged.

The diagnostic payload will include rank identity so records are unambiguous. Ratio mean/min/max/quantiles will come from the already-computed local masked-token statistics instead of a metric normalized by a global sample count. Values that require a true global quantile remain explicitly per-rank; no statistically invalid averaging of quantiles will be introduced.

## Error handling

The attention-mask helper will reject non-two-dimensional input rather than silently broadcasting an unexpected layout. Empty diagnostic lists remain valid. Existing diagnostic failures continue to be non-fatal.

## Tests

Add focused CPU tests for:

- causal and padding semantics of the generated Megatron mask;
- all-valid and partially padded batches;
- token-mean aggregation excluding masked values and respecting sample weights;
- diagnostic metadata aggregation/flattening across two simulated ranks;
- preservation of the first optimizer-batch invariant: current/old ratio statistics are exactly one before an update.

## Acceptance criteria

After deployment, a one-step NPU run should show:

- first optimizer-batch PPO ratio min/median/max equal to 1;
- FP16 actor old logprobs close to FP16 reference and vLLM inference logprobs;
- step-0 reference KL near zero, with only expected backend/quantization drift;
- clip fractions bounded to `[0, 1]` when reported as fractions;
- diagnostics from both DP ranks clearly identified.
