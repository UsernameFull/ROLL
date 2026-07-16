# MegatronAdaptor FP8 Import Migration Design

## Goal

Migrate the Ascend Megatron FP8 runtime from the legacy MindSpeed package layout to the standalone
`MegatronAdaptor core_r0.17.0` and `TransformerEngineNPU` stack used by `npu_ci_all`.

The migration must preserve the current FP8 argument derivation and Transformer Engine integration while removing all
runtime imports from `mindspeed.*`. Missing standalone dependencies must fail during NPU Megatron initialization with a
clear error instead of silently disabling NPU patches.

## Scope

The change covers:

- NPU Megatron runtime bootstrap and adaptor module tracking.
- MegatronAdaptor feature-default and FP8 argument synchronization.
- TransformerEngineNPU symbol discovery through canonical Megatron interfaces.
- Megatron NPU tests and the NPU CI dependency stack.
- Internal names and messages that still describe the runtime integration as MindSpeed-specific.

The existing MXFP8 model-update transaction and vLLM weight-loading changes are preserved without modification.

## Non-goals

- Supporting both `mindspeed.megatron_adaptor` and standalone `megatron_adaptor` in the same release.
- Falling back to the legacy MindSpeed package when standalone dependencies are unavailable.
- Changing FP8 formats, recipes, checkpoint conversion, or model-update semantics.
- Refactoring unrelated NPU operator patches or general Megatron initialization.

## Runtime Design

### Strict standalone adaptor bootstrap

On NPU, Megatron initialization imports `megatron_adaptor` before model-parallel initialization and FP8 model
construction. The imported module is retained as the single source of truth for adaptor-loaded checks and repatching.

The bootstrap remains a no-op on non-NPU platforms. On NPU, an import failure raises `RuntimeError` that names the
required `MegatronAdaptor core_r0.17.0` dependency. The runtime does not attempt `mindspeed.megatron_adaptor`.

```text
initialize_megatron(args)
  -> bootstrap_npu_runtime()
       -> import torch_npu
       -> import megatron_adaptor
       -> validate canonical Transformer Engine integration
       -> install existing NPU RNG compatibility hooks
  -> sync_megatron_adaptor_args(args)
  -> initialize distributed/model-parallel groups
```

### FP8 argument synchronization

The existing derived-argument behavior remains intact. Runtime arguments are obtained from
`megatron_adaptor.utils.args_utils.get_mindspeed_args`; the upstream helper keeps its existing name even though its
package moved.

ROLL continues to synchronize shared fields and derive the following FP8 and attention values:

- `fp8` and `fp8_format` remain aliases.
- Enabling FP8 implies `transformer_impl="transformer_engine"` when not explicitly set.
- `use_flash_attn_npu_batch_invariant=True` disables `use_flash_attn`.
- Transformer Engine on NPU enables `use_flash_attn` by default when batch-invariant attention is not selected.

When synchronized values change after initial patching, the runtime invokes `megatron_adaptor.repatch(updates)`. A
repatch failure is logged as a warning because the existing behavior treats it as a compatibility refresh rather than
the primary bootstrap. Import and required-symbol failures remain fatal.

Internal ROLL APIs are renamed from `mindspeed` terminology to `megatron_adaptor` terminology. The upstream
`get_mindspeed_args` function name is not wrapped or renamed outside the runtime boundary.

### TransformerEngineNPU integration

After `megatron_adaptor` is loaded, ROLL discovers Transformer Engine classes and checkpoint helpers only through
`megatron.core.extensions.transformer_engine`. It no longer imports `TENorm` or patched model modules from
`mindspeed.core.*`.

ROLL keeps the existing `_npu_te_checkpoint` call-signature adapter for Qwen3-VL activation checkpointing. After
MegatronAdaptor bootstrap, it publishes that adapter only to the canonical
`megatron.core.extensions.transformer_engine` module when `te_checkpoint` is absent. It no longer synthesizes or
patches `transformer_engine.*` or `mindspeed.core.*` modules. FP8 initialization fails with a dependency-oriented error
when the standalone adaptor or TransformerEngineNPU did not expose the required Transformer Engine module or `TENorm`.

This keeps model code independent of the vendor package's internal directory layout and matches the integration
boundary used by MegatronAdaptor 0.17.

## Code Changes

### `mcore_adapter/src/mcore_adapter/npu_runtime.py`

- Replace the legacy adaptor importer and loaded-module checks with standalone `megatron_adaptor` equivalents.
- Replace `mindspeed.args_utils` with `megatron_adaptor.utils.args_utils`.
- Repatch through the loaded `megatron_adaptor` module.
- Remove `mindspeed.core.*` module patch targets and imports.
- Rename internal constants and functions that describe the integration as MindSpeed-specific.
- Raise clear runtime errors for missing required standalone packages or FP8 Transformer Engine symbols.

### `mcore_adapter/src/mcore_adapter/initialize.py`

- Export and call the renamed MegatronAdaptor feature-default and argument-sync helpers.
- Preserve bootstrap-before-distributed-initialization ordering.

### `mcore_adapter/src/mcore_adapter/models/model_config.py`

- Apply feature defaults through the renamed MegatronAdaptor helper.
- Preserve the current FP8/Transformer Engine validation and attention derivation.

### Tests and CI

- Update Megatron NPU tests to import `megatron_adaptor`.
- Add focused runtime tests for non-NPU no-op behavior, strict NPU import failure, standalone argument discovery,
  FP8-derived argument synchronization, and repatching.
- Update `.github/workflows/ci-npu-mindspeed.yml` in place to install and verify Megatron-Core 0.17,
  MegatronAdaptor `core_r0.17.0`, and TransformerEngineNPU, following `npu_ci_all`.
- Keep the existing workflow filename to avoid unrelated repository wiring changes; update its display names, inputs,
  cache key, and test step labels to MegatronAdaptor terminology.

## Error Handling

- Non-NPU execution never requires or imports MegatronAdaptor.
- Missing `megatron_adaptor` on NPU raises a `RuntimeError` before distributed initialization.
- Missing `megatron_adaptor.utils.args_utils` is treated as an incompatible MegatronAdaptor installation and raises a
  dependency-oriented `RuntimeError`.
- Missing required TransformerEngineNPU symbols in FP8 mode raises a `RuntimeError` that identifies
  TransformerEngineNPU as the expected provider.
- Repatch exceptions remain warnings and include the attempted updates, preserving diagnosability without hiding the
  original initialization result.

## Compatibility

- Supported NPU stack: Megatron-Core 0.17.x, MegatronAdaptor `core_r0.17.0`, and TransformerEngineNPU.
- Legacy `mindspeed.megatron_adaptor` environments are intentionally unsupported.
- CUDA and CPU initialization behavior is unchanged.
- Current Megatron FP8 configuration keys and YAML files remain valid.
- Current vLLM MXFP8 model-update transaction changes remain untouched.

## Verification

Automated verification will include:

1. Static search proving runtime and tests no longer import `mindspeed.*`.
2. Unit tests for strict standalone bootstrap, argument synchronization, repatching, and non-NPU isolation.
3. Existing configuration tests for Megatron FP8 aliases and Transformer Engine selection.
4. Python compilation of modified modules.
5. NPU CI execution with the standalone 0.17 stack.

Hardware acceptance requires the Megatron FP8 model to initialize on Ascend, run one training step, and complete a
weight refresh into the existing MXFP8 rollout path without a legacy MindSpeed package installed.
