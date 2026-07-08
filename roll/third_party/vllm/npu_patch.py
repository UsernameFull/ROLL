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

"""vLLM-Ascend runtime checks for the NPU path.

ROLL's NPU vLLM integration assumes vLLM/vLLM-Ascend >= 0.18. Older
vLLM-Ascend 0.11/0.13 monkey patches have been removed; hardware behavior
should come from the installed vLLM-Ascend runtime itself.
"""

import os

from packaging.version import Version

from roll.utils.logging import get_logger

logger = get_logger()

MIN_NPU_VLLM_VERSION = "0.18.0"


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


def check_vllm_ascend_before_server_launch():
    """Validate vLLM-Ascend configuration before the server starts."""
    if not _is_a2():
        return

    enable_matmul_allreduce = bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0")))
    if enable_matmul_allreduce:
        raise AssertionError(
            "Ascend A2 does not support VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE "
            "in single-card multi-process scenarios. Set the environment variable to 0."
        )


def apply_npu_vllm_patches():
    """Validate the NPU vLLM version and rely on native vLLM-Ascend behavior."""
    import vllm

    if Version(vllm.__version__) < Version(MIN_NPU_VLLM_VERSION):
        raise RuntimeError(
            f"ROLL NPU vLLM integration requires vLLM>={MIN_NPU_VLLM_VERSION}; "
            f"detected vLLM {vllm.__version__}. Legacy vLLM-Ascend compatibility patches "
            "for older releases have been removed."
        )

    logger.debug(
        "vLLM %s on NPU uses native vLLM-Ascend >=%s behavior; no legacy patches are applied.",
        vllm.__version__,
        MIN_NPU_VLLM_VERSION,
    )
