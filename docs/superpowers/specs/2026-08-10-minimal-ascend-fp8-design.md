# Minimal Ascend FP8 Support Design

## Objective

Reduce the current branch's difference from `alibaba/main` while preserving
three end-to-end Ascend FP8 capabilities:

1. Megatron FP8 computation with BF16/FP16 model parameters.
2. vLLM online conversion of synchronized BF16/FP16 weights to Ascend MXFP8.
3. Megatron training from a ModelSlim pre-quantized MXFP8 checkpoint.

The implementation targets one dependency matrix only:

- vLLM `0.23.0`
- vLLM-Ascend `0.23.0rc1`
- MegatronAdaptor `core_r0.17.0`
- TransformerEngineNPU paired with MegatronAdaptor

Supporting these fixed versions is more important than preserving compatibility
with earlier vLLM or vLLM-Ascend releases.

## Current-State Finding

At the start of this audit, the branch was 48 commits ahead of `alibaba/main`,
with 6,305 insertions and 406 deletions. The difference combines FP8
functionality with NPU enablement, old vLLM compatibility, SGLang and FSDP2
experiments, diagnostics, CI, container setup, examples, and design history.
The largest avoidable cost is the version-dispatch and
parameter-compatibility code in the vLLM FP8 and worker patches.

## Architecture

### Megatron FP8 computation

`TrainingArguments` remains the public configuration boundary. It accepts the
Megatron FP8 format and recipe, validates supported combinations, and passes
only the required values through `megatron_strategy` to `McaModelConfig`.

On NPU, initialization imports MegatronAdaptor before constructing the model,
applies its defaults, and synchronizes ROLL arguments into the adaptor. Native
Megatron and TransformerEngineNPU then own FP8 autocast, recipes, scaling, and
trainable state. ROLL must not duplicate those algorithms.

### vLLM online MXFP8

The vLLM strategy accepts `online_quantization: ascend_mxfp8`. During engine
configuration ROLL builds the smallest ModelSlim-compatible quantization
description required by supported dense and MoE Qwen models and passes it as
an Ascend quantization configuration.

During a weight update, vLLM first applies its normal TP/EP sharding. The ROLL
worker hook then quantizes each eligible floating-point shard with
`torch_npu.npu_dynamic_mx_quant`, supplies its scale tensor, and delegates
packing and runtime preparation to the vLLM-Ascend 0.23 quantization method.
Repeated updates must replace the existing quantized values without changing
parameter identity required by graph execution.

No vLLM 0.10-0.20 processing implementation or version selector is retained.

### ModelSlim pre-quantized training

An explicit `quantized_checkpoint_format: ascend_mxfp8` switch selects this
path. A small checkpoint adapter reads ModelSlim quantization metadata,
distinguishes quantized and floating-point tensors, and pairs each quantized
weight with its scale sidecar.

The adapter does not implement QKV merging, MoE merging, or distributed
sharding. It calls one fixed MindSpeed-TE loader contract for those operations.
If the loader is unavailable or a weight/scale pair is incomplete, loading
fails with a specific error; it never silently dequantizes into BF16 training.

## Minimality Rules

The implementation will retain a changed file only when it is on one of the
three runtime paths above or is required by their focused tests.

Remove:

- vLLM branches and implementations for releases before 0.23;
- SGLang, FSDP2, CUDA/H100, and unrelated model extensions;
- diagnostic logging and log-probability investigation code;
- duplicate parameter-subclass and runtime compatibility abstractions already
  provided by the fixed upstream versions;
- redundant examples, historical design documents, and FP8-specific CI
  scaffolding not required at runtime.

Retain:

- one four-NPU example exercising Megatron FP8 and vLLM online MXFP8;
- focused configuration, metadata parsing, quantization, and repeat-update
  regression tests;
- the previously requested MoE group-count, NPU RNG, short-response whitening,
  and GDN worker fixes;
- the user's existing Dockerfile and untracked BF16 example changes.

General NPU changes are retained only if the four-NPU FP8 path directly
requires them. Merely being useful to another NPU model or framework is not
sufficient.

## Error Handling

- Reject unsupported dependency versions at startup with the expected version
  matrix in the error message.
- Reject conflicting `quantization` and `online_quantization` settings.
- Reject `fp8_param=True` unless NPU, TransformerEngineNPU, MXFP8 recipe, and
  the explicit ModelSlim checkpoint format are all active.
- Reject missing or malformed quantization metadata and missing scale tensors.
- Reject a missing MindSpeed-TE trainable loader instead of falling back to
  dequantized parameters.

## Verification

CPU-capable tests cover configuration validation, quantization-description
generation, checkpoint metadata and weight/scale pairing, and update lifecycle
logic with mocked quantization operations.

The NPU smoke test uses the retained Qwen3 0.6B four-NPU example and verifies:

1. Megatron creates FP8-enabled TransformerEngineNPU layers.
2. One forward/backward and optimizer step completes.
3. BF16/master weights synchronize to the vLLM worker.
4. The worker produces MXFP8 weights and scales after sharding.
5. Two consecutive weight updates and one generation complete.

The final review compares both changed files and line count against
`alibaba/main`, checks that every remaining difference maps to an approved
runtime path, and reports any verification that cannot run locally.
