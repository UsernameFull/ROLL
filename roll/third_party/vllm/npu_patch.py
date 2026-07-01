# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM-Ascend hardware compatibility patches.

Handles known incompatibilities on specific Ascend SoC generations:

- **A2 (SoC 220-225)**: MC2 (MatMul+AllReduce fusion) unsupported in
  single-card multi-process scenarios → fall back to AllGather.
- **w4a8_dynamic MoE**: AllGatherEP unsupported → fall back to AllToAll.
- **vLLM 0.13 MoE weight_loader**: transpose fix for w1/w3.
- **vLLM 0.13 RotaryEmbedding**: disable flash_attn usage on NPU.

Environment variable ``VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2=1``
(default) controls whether A2 MC2 patches are applied.
"""

import os
from functools import wraps

from roll.utils.logging import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_soc_version() -> int:
    """Return the Ascend SoC version number."""
    try:
        import torch_npu

        return torch_npu.npu.get_soc_version()
    except Exception:
        return 0


def _is_a2() -> bool:
    """Check whether the current device is Ascend A2 (SoC 220-225)."""
    return 220 <= _get_soc_version() <= 225


def _is_a3() -> bool:
    """Check whether the current device is Ascend A3 (SoC 250-255)."""
    return 250 <= _get_soc_version() <= 255


# ---------------------------------------------------------------------------
# Pre-launch safety checks
# ---------------------------------------------------------------------------


def check_vllm_ascend_before_server_launch():
    """Validate vLLM-Ascend configuration before the server starts.

    On A2 devices, ``VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`` must be disabled
    in single-card multi-process scenarios.  This function raises an
    ``AssertionError`` if the env-var is set on A2 hardware.
    """
    if not _is_a2():
        return

    enable_matmul_allreduce = bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0")))
    if enable_matmul_allreduce:
        raise AssertionError(
            "Ascend A2 does not support VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE "
            "in single-card multi-process scenarios.  Set the environment variable to 0."
        )


# ---------------------------------------------------------------------------
# A2 MC2 fallback – vLLM 0.11
# ---------------------------------------------------------------------------


def _patch_moe_comm_method_v011(fn):
    """vLLM 0.11: replace MC2 with AllGather on A2."""

    @wraps(fn)
    def wrapper(self, num_tokens, with_prefill):
        moe_comm_method = fn(self, num_tokens, with_prefill)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import AscendSocVersion, enable_sp, get_ascend_soc_version

        soc_version = get_ascend_soc_version()
        if soc_version in {AscendSocVersion.A2} and moe_comm_method == MoECommType.MC2:
            quant_type = getattr(self.vllm_config.model_config.hf_config, "moe_quantize", None)
            if quant_type == "w4a8_dynamic":
                moe_comm_method = MoECommType.ALLTOALL
            else:
                moe_comm_method = MoECommType.ALLGATHER

        if with_prefill:
            if enable_sp():
                moe_comm_method = MoECommType.ALLGATHER
            else:
                moe_comm_method = MoECommType.NAIVE_MULTICAST

        return moe_comm_method

    return wrapper


def _patch_matmul_and_reduce_v011(fn):
    """vLLM 0.11: disable mmrs_fusion on A2."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version

        soc_version = get_ascend_soc_version()
        if soc_version in {AscendSocVersion.A2}:
            from vllm.forward_context import get_forward_context

            try:
                forward_context = get_forward_context()
                forward_context.mmrs_fusion = False
            except AssertionError:
                pass
        return fn(self, *args, **kwargs)

    return wrapper


def apply_vllm_ascend_v011_patches():
    """Apply A2 MC2 fallback patches for vLLM 0.11.x."""
    if not _is_a2():
        return

    try:
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

        NPUModelRunner._select_moe_comm_method = _patch_moe_comm_method_v011(
            NPUModelRunner._select_moe_comm_method
        )
        SequenceRowParallelOp.matmul_and_reduce = _patch_matmul_and_reduce_v011(
            SequenceRowParallelOp.matmul_and_reduce
        )
        logger.info("Applied vLLM-Ascend 0.11 A2 MC2 patches")
    except Exception as e:
        logger.warning("Failed to apply vLLM-Ascend 0.11 A2 patches: %s", e)


# ---------------------------------------------------------------------------
# A2 MC2 fallback – vLLM 0.13
# ---------------------------------------------------------------------------


def _patch_select_moe_comm_method_v013(fn):
    """vLLM 0.13: replace MC2 with AllGather on A2."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        moe_comm_method = fn(*args, **kwargs)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        ascend_device_type = get_ascend_device_type()
        if ascend_device_type in {AscendDeviceType.A2} and moe_comm_method == MoECommType.MC2:
            moe_comm_method = MoECommType.ALLGATHER
        return moe_comm_method

    return wrapper


def _patch_matmul_and_reduce_v013(fn):
    """vLLM 0.13: disable mmrs_fusion on A2."""

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        ascend_device_type = get_ascend_device_type()
        if ascend_device_type in {AscendDeviceType.A2}:
            from vllm.forward_context import get_forward_context

            try:
                forward_context = get_forward_context()
                forward_context.mmrs_fusion = False
            except AssertionError:
                pass
        return fn(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# vLLM 0.13 rotary embedding init patch
# ---------------------------------------------------------------------------


def _patch_vllm013_rotary_emb():
    """On NPU, disable flash_attn in RotaryEmbedding for vLLM >= 0.13."""
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    def vllm013_npu_rotary_embedding_init_impl(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super(ApplyRotaryEmb, self).__init__()
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        self.apply_rotary_emb_flash_attn = None

    ApplyRotaryEmb.__init__ = vllm013_npu_rotary_embedding_init_impl


# ---------------------------------------------------------------------------
# vLLM 0.13 MoE weight loader transpose fix
# ---------------------------------------------------------------------------


def _patch_vllm013_moe_weight_loader():
    """Fix MoE weight loader for NPU: transpose w1/w3 weight dims."""

    def patched_weight_loader(fn):
        @wraps(fn)
        def wrapper(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            if (shard_id in ("w1", "w3") and param.shape[1] == self.hidden_size) or (
                shard_id == "w2" and param.shape[2] == self.hidden_size
            ):
                param.data = param.data.transpose(1, 2)
            return fn(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success)

        return wrapper

    from vllm.model_executor.layers.fused_moe import FusedMoE

    FusedMoE.weight_loader = patched_weight_loader(FusedMoE.weight_loader)


def apply_vllm_ascend_v013_patches():
    """Apply A2 MC2 + RoPE + MoE patches for vLLM 0.13.x."""
    try:
        from vllm_ascend import ascend_forward_context
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp

        ascend_forward_context.select_moe_comm_method = _patch_select_moe_comm_method_v013(
            ascend_forward_context.select_moe_comm_method
        )
        SequenceRowParallelOp.matmul_and_reduce = _patch_matmul_and_reduce_v013(
            SequenceRowParallelOp.matmul_and_reduce
        )
        _patch_vllm013_rotary_emb()
        _patch_vllm013_moe_weight_loader()
        logger.info("Applied vLLM-Ascend 0.13 patches (MC2 / RoPE / MoE-wl)")
    except Exception as e:
        logger.warning("Failed to apply vLLM-Ascend 0.13 patches: %s", e)


# ---------------------------------------------------------------------------
# Main entry point – called during vLLM server / worker init
# ---------------------------------------------------------------------------


def apply_npu_vllm_patches():
    """Apply all Ascend NPU compatibility patches for vLLM.

    Called during vLLM server startup (before the AsyncLLM is created) to
    ensure the NPU hardware quirks are handled correctly.
    """
    import vllm
    from packaging.version import Version

    vllm_ver = Version(vllm.__version__)

    if Version("0.13.0") <= vllm_ver <= Version("0.14.0"):
        apply_vllm_ascend_v013_patches()
    elif Version("0.11.0") <= vllm_ver < Version("0.13.0"):
        apply_vllm_ascend_v011_patches()
    else:
        logger.debug(
            "vLLM %s – NPU hardware patches are only applied for v0.11–v0.13; "
            "newer versions may handle these natively.",
            vllm.__version__,
        )
