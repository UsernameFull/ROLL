# MXFP8 Weight Update Lifecycle Design

## Goal

Make Ascend MXFP8 rollout weight refreshes safe and efficient for both online-quantized and pre-quantized vLLM models.

This change addresses two problems:

1. A worker created from an already-transformed MXFP8 model may incorrectly record that no runtime transformation has been applied, causing the first RL weight refresh to skip restoring Hugging Face weight layouts.
2. A bucketed model refresh currently restores and transforms the full MXFP8 model for every bucket instead of once per complete model update.

Training-side FP8 export semantics and quantization-description coverage are out of scope.

## Design

### Detect the initial runtime layout

When the vLLM worker extension initializes, it will inspect the loaded model for MXFP8 runtime markers. A module counts as transformed when it has an active `_mxfp8_transformed` marker. The result initializes `_mxfp8_transformation_applied` instead of always setting it to `False`.

This makes both initialization paths explicit:

- `load_format=auto` with a pre-quantized checkpoint: detect the transformation performed during vLLM model loading.
- `load_format=dummy` with online quantization: detect the actual state rather than assuming whether vLLM processed the dummy weights.

### Add an explicit model-update transaction

The update lifecycle will have three phases:

```text
begin_model_update
    restore transformed MXFP8 modules to HF layouts once
    mark the worker as updating

load bucket 1..N
    quantize incoming BF16/FP16 tensors
    call model.load_weights for only the current bucket
    do not transform the full model

process_weights_after_loading (commit)
    transform MXFP8 modules once
    restore graph-safe runtime tensor references
    invalidate graph caches only when in-place reference preservation is unavailable
    clear the updating state
```

The pipeline will call `begin_model_update` immediately before each model update group transfers weights. The existing `process_weights_after_loading` call remains the commit boundary.

The call is added through the existing delegation layers:

```text
BasePipeline
  -> target Cluster
  -> pipeline/executor Worker
  -> InferenceStrategy
  -> VllmStrategy
  -> CustomAsyncLLM RPC
  -> vLLM WorkerBase
```

Strategies that do not need a transaction use the base no-op implementation.

### Preserve standalone loading behavior

`WorkerBase.load_weights()` and bucket receivers must distinguish transaction and non-transaction calls:

- During a model-update transaction, loading a bucket skips full-model finalization.
- Outside a transaction, loading remains atomic and performs restore, load, and finalize before returning.

This preserves existing direct vLLM test helpers and independent weight-loading callers.

### State and error handling

The worker owns `_model_update_in_progress`.

- Beginning an update while one is already active raises `RuntimeError`.
- Committing without an active MXFP8 update keeps existing post-processing behavior, because initialization and non-transaction callers may legitimately request it.
- The transaction flag is cleared only after successful finalization. If loading or finalization fails, the worker stays in an explicit failed/in-progress state and must not serve generation with a partially updated model.
- Non-MXFP8 workers accept the begin call as a no-op and retain their existing processing path.

## Compatibility

- No YAML changes are required.
- Existing `online_quantization=ascend_mxfp8` and `quantization=ascend` configurations retain their public behavior.
- GPU `vllm_fp8` behavior is unchanged except for passing through the new no-op begin hook.
- LoRA and SGLang behavior are unchanged.

## Tests

Unit tests will verify:

1. Initial transformed-state detection returns true only for models carrying an active MXFP8 transformation marker.
2. Beginning a transaction restores a transformed model once.
3. Loading multiple buckets during a transaction does not finalize between buckets.
4. Commit finalizes once and clears transaction state.
5. A direct non-transaction load still finalizes immediately.
6. A repeated begin call raises a clear error.

Existing online-quantization configuration and runtime-reference tests remain unchanged and must continue to pass.

Hardware acceptance should additionally run two consecutive policy updates on Ascend in eager and ACL Graph modes, then verify successful generation and changed model outputs after each refresh.
