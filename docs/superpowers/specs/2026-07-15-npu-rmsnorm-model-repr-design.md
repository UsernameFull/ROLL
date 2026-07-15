# NPU RMSNorm Model Repr Compatibility Design

## Goal

Restore the original full Megatron model representation in initialization logs while avoiding the TransformerEngineNPU `RMSNorm.extra_repr()` failure caused by a missing `normalized_shape` attribute.

## Scope

- Apply the compatibility behavior only when `current_platform.is_npu()` is true.
- Target only `transformer_engine.pytorch.module.rmsnorm.RMSNorm` instances that lack `normalized_shape` and have a `weight` parameter.
- Preserve the existing representation behavior on CUDA, ROCm, and CPU.
- Restore the full model representation in both Megatron inference and training initialization logs.

## Design

Add a formatting helper in `roll/utils/logging.py`. On non-NPU platforms, it returns the normal `repr(value)` directly. On NPU, it walks all `torch.nn.Module` objects contained in the value and identifies affected TransformerEngineNPU RMSNorm modules without importing TransformerEngineNPU.

For each affected module, the helper temporarily assigns:

```python
normalized_shape = tuple(module.weight.shape)
```

It then calls the original `repr(value)`. A `try/finally` block removes every temporary attribute even when representation generation raises another exception. Existing `normalized_shape` attributes are never overwritten or removed.

The Megatron strategy logs pass `self.model.get_models()` to this helper, restoring the complete list and nested module representation that was emitted before the defensive model-type-only logging change.

## Output Compatibility

For healthy modules, the generated text is exactly their normal `repr()` output. For the affected TransformerEngineNPU RMSNorm, the temporary value matches the shape metadata that its `extra_repr()` intends to display, so its output matches a correctly initialized implementation.

No prefixes, fallback messages, or replacement summaries are added to the log message.

## State Safety

The compatibility attribute exists only while `repr()` executes. It is not retained in the model, included in later serialization, or visible to training and inference code after logging completes.

## Tests

Add unit coverage for:

1. An affected TransformerEngineNPU-style RMSNorm can be represented successfully on NPU.
2. Its output matches the output from an equivalent instance with a valid `normalized_shape`.
3. The temporary attribute is removed after successful formatting.
4. The temporary attribute is removed if nested representation raises an exception.
5. Non-NPU formatting calls ordinary `repr()` without compatibility mutation.
6. Existing `normalized_shape` values are preserved.
