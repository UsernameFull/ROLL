# MegatronAdaptor NPU CI Migration Design

## Goal

Migrate the `npu_ci_all` branch from the legacy MindSpeed integration to the
Ascend-supported Megatron-Core 0.17 stack, using these source repositories:

- `https://gitcode.com/Ascend/MegatronAdaptor.git` at `core_r0.17.0`
- `https://gitcode.com/Ascend/TransformerEngineNPU.git` at `main`

The work is complete only when the corresponding workflow on
`UsernameFull/ROLL` succeeds for the pushed commit.

## Scope

1. Replace runtime and test imports of `mindspeed.megatron_adaptor` with
   `megatron_adaptor`.
2. Read MegatronAdaptor defaults from
   `megatron_adaptor.utils.args_utils.get_mindspeed_args` and rename ROLL's
   helper to describe the new integration.
3. Update `mcore_adapter` dependency bounds from Megatron-Core 0.16 to the
   compatible 0.17 series.
4. Replace the MindSpeed repository/ref workflow inputs and install step with
   MegatronAdaptor and TransformerEngineNPU inputs and source installs.
5. Add an import/version smoke check before the existing distributed offload
   test so the workflow proves that the requested packages are active.

## CI Dependency Flow

The workflow installs ROLL requirements first, then installs
TransformerEngineNPU and MegatronAdaptor from their configured GitCode refs.
Installing TransformerEngineNPU replaces any preinstalled NVIDIA
`transformer_engine` distribution because both expose the same Python package.
MegatronAdaptor is imported before Megatron-backed ROLL code so its NPU patches
are registered.

The workflow keeps the existing A3 runner, Ascend container, runtime setup,
model preparation, and two-process offload test. Cache keys include the new
dependency refs so incompatible cached environments are not reused.

## Validation and Iteration

Local validation covers Python syntax, workflow YAML parsing, dependency-bound
consistency, and removal of executable MindSpeed imports. After commit and
push, the GitHub Actions run for the exact head SHA is monitored. On failure,
the failing job logs are inspected, a focused correction is committed and
pushed, and monitoring repeats until the workflow succeeds.

## Non-goals

- Supporting MindSpeed and MegatronAdaptor simultaneously.
- Refactoring unrelated NPU, Megatron, or test code.
- Changing the runner image or hardware unless CI evidence proves it is
  required for the requested dependency stack.
