# Megatron FP8 Layerwise Diagnostics Design

## Goal

Locate the first Transformer layer where an FP8 forward diverges materially from a BF16 shadow forward while holding the model parameters and probe batch fixed.

## Scope

- Run only for the existing first optimizer-batch diagnostic.
- Reuse the existing post-update BF16-shadow and native-precision probe forwards; do not add an optimizer step or another full forward.
- Inspect Transformer layer outputs on the local pipeline/model shard.
- Emit bounded, JSON-serializable statistics in the existing `[RLVR_LOGPROB_DIAG]` payload.

This diagnostic does not implement W8A16/W16A8 execution modes and does not modify training results.

## Layer Discovery

Traverse each unwrapped model chunk and select modules whose class name is `TransformerLayer`. Record:

- model chunk index;
- module path;
- `layer_number` when available;
- module class and module package.

Keys include the model chunk index and module path so virtual-pipeline chunks cannot collide.

## Activation Capture

Register temporary forward hooks only around the two existing post-update probes. Extract the first tensor from a tensor or tuple/list output. Unsupported outputs are recorded as capture errors instead of failing training.

For each hook invocation:

- detach the tensor;
- flatten it without changing the model output;
- select evenly spaced values with a deterministic stride;
- transfer at most 65,536 sampled values per layer and precision to CPU FP32;
- record invocation count, original element count, tensor shapes, and sampled count.

Hooks are removed in `finally`, including when a probe fails.

## Statistics

Match BF16 and native captures by layer key. For each matched layer, report:

- BF16 and native mean, RMS, absolute maximum, and nonfinite fraction;
- delta mean, RMS, absolute mean, P50, P95, P99, and maximum;
- relative RMS, using BF16 RMS as the denominator;
- cosine similarity over finite paired samples;
- original and sampled element counts and hook invocation counts.

Also report unmatched layers and per-layer capture errors. A layer with no finite paired values is reported as an error and does not abort training.

The result is stored at:

```text
logprob_probe.layerwise_bf16_vs_native_after
```

## Error Handling and Safety

- The diagnostic is best-effort; capture/statistics failures are recorded in `layerwise_error`.
- Probe errors retain the existing behavior.
- Hook removal is mandatory and happens before returning or propagating a probe failure.
- Captured tensors are detached CPU samples, so the diagnostic does not retain autograd graphs or full activations.
- Model parameters, training modes, RNG state, and optimizer state are not modified by layer capture.

## Tests

Pure unit tests cover:

- deterministic bounded sampling;
- tensor extraction from supported and unsupported outputs;
- finite and nonfinite layer-delta statistics;
- zero-reference RMS behavior;
- layer matching, ordering, and unmatched/error reporting;
- hook cleanup after success and exceptions using small fake Transformer layers.

Existing parameter-interpolation diagnostics must remain unchanged.

## Acceptance Criteria

- Existing diagnostics and tests continue to pass.
- The first optimizer-batch payload contains ordered local-layer results for both BF16 and FP8 training runs.
- FP16 native runs produce zero BF16-vs-native layer deltas within exact repeat behavior.
- FP8 native runs identify the first layer where relative RMS or cosine similarity degrades.
- No hooks remain registered after the diagnostic, including error paths.
