# vLLM Ascend Platform Detection Fix

## Problem

`roll.third_party.vllm.fp8` imports `current_platform` from `vllm.platforms` and calls
`current_platform.is_npu()`. The vLLM-Ascend 0.23 out-of-tree platform does not expose
that method, so module import fails with `TypeError: 'NoneType' object is not callable`.

## Design

Keep the two platform responsibilities explicit:

- Import the vLLM platform as `vllm_platform` for vLLM-specific capabilities such as
  `is_rocm()`, `is_fp8_fnuz()`, and `fp8_dtype()`.
- Import the ROLL platform as `roll_platform` and use `roll_platform.is_npu()` for the
  four decisions that control installation of legacy FP8 Linear and MoE patches.

This preserves existing CUDA and ROCm behavior while preventing legacy vLLM FP8 patches
from being installed on Ascend.

## Validation

- Compile the changed module.
- Run the local FP8 utility and online-quantization tests.
- Verify no `vllm_platform.is_npu()` calls remain.
- Verify the diff has no whitespace errors.

