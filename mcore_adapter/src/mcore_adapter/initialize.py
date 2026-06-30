import os
import random
import sys
from typing import TYPE_CHECKING
import importlib.util

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel

from .platforms import current_platform
from .utils import get_logger

if TYPE_CHECKING:
    from .training_args import TrainingArguments


logger = get_logger(__name__)


_NPU_RUNTIME_BOOTSTRAPPED = False


def _has_megatron_training():
    return importlib.util.find_spec("megatron.training") is not None


def ensure_npu_transformer_engine_symbols():
    if not current_platform.is_npu():
        return

    try:
        import megatron.core.extensions.transformer_engine as te_ext
        from mindspeed.core.transformer.custom_layers.transformer_engine import TENorm
    except ImportError:
        return

    if getattr(te_ext, "TENorm", None) is not TENorm:
        te_ext.TENorm = TENorm

    for module_name in (
        "megatron.core.models.gpt.gpt_layer_specs",
        "mindspeed.core.models.gpt.gpt_layer_specs",
        "mindspeed.core.transformer.transformer_block",
    ):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "TENorm", None) is not TENorm:
            setattr(module, "TENorm", TENorm)


def bootstrap_npu_runtime():
    global _NPU_RUNTIME_BOOTSTRAPPED

    if _NPU_RUNTIME_BOOTSTRAPPED or not current_platform.is_npu():
        return

    import torch_npu  # noqa: F401

    try:
        import mindspeed.megatron_adaptor  # noqa: F401
    except ImportError:
        pass

    import megatron.core.tensor_parallel.random as meg_random

    if not hasattr(meg_random, "_npu_patched"):
        meg_random.initialize_rng_tracker()

        def patched_set(new_state, device=-1, graph_safe=False):
            torch.npu.set_rng_state(new_state)
            return

        def patched_get(device="npu", clone=False, graph_safe=False):
            return torch.npu.get_rng_state()

        meg_random._set_cuda_rng_state = patched_set
        meg_random._get_cuda_rng_state = patched_get

        rng_state = torch.npu.get_rng_state()
        meg_random._CUDA_RNG_STATE_TRACKER.states_["model-parallel-rng"] = rng_state
        meg_random._CUDA_RNG_STATE_TRACKER.states_["data-parallel-rng"] = rng_state

        meg_random._npu_patched = True

    if not hasattr(torch.cuda, "_npu_patched"):
        torch.cuda.current_device = lambda: torch.npu.current_device()
        torch.cuda._npu_patched = True

    _NPU_RUNTIME_BOOTSTRAPPED = True


def apply_mindspeed_feature_defaults(config):
    if "mindspeed.megatron_adaptor" not in sys.modules:
        return

    try:
        from mindspeed.args_utils import get_mindspeed_args
    except ImportError:
        return

    for name, value in vars(get_mindspeed_args(get_defaults=True)).items():
        if not hasattr(config, name):
            setattr(config, name, value)


def sync_mindspeed_args(args: "TrainingArguments"):
    if "mindspeed.megatron_adaptor" not in sys.modules:
        return

    try:
        from mindspeed.args_utils import get_mindspeed_args
    except ImportError:
        return

    mindspeed_args = get_mindspeed_args()
    updates = {}
    for name in (
        "transformer_impl",
        "fp8",
        "fp8_format",
        "fp8_recipe",
        "fp8_param",
        "micro_batch_size",
        "context_parallel_size",
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "use_ascend_mc2",
        "use_ascend_coc",
        "use_gmm_fp8",
        "use_flash_attn",
        "use_flash_attn_npu_batch_invariant",
        "te_comparison_with_cpu",
        "te_comparison_with_bf16",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                updates[name] = value

    if updates.get("fp8") and "fp8_format" not in updates:
        updates["fp8_format"] = updates["fp8"]
    if updates.get("fp8_format") and "fp8" not in updates:
        updates["fp8"] = updates["fp8_format"]
    if updates.get("fp8") and "transformer_impl" not in updates:
        updates["transformer_impl"] = "transformer_engine"
    if current_platform.is_npu():
        if updates.get("use_flash_attn_npu_batch_invariant"):
            updates["use_flash_attn"] = False
        elif updates.get("transformer_impl") == "transformer_engine":
            updates.setdefault("use_flash_attn", True)

    changed = False
    for name, value in updates.items():
        if getattr(mindspeed_args, name, None) != value:
            setattr(mindspeed_args, name, value)
            changed = True

    if changed and current_platform.is_npu() and _has_megatron_training():
        try:
            import mindspeed.megatron_adaptor as megatron_adaptor

            if hasattr(megatron_adaptor, "repatch"):
                megatron_adaptor.repatch(updates)
        except Exception as e:
            logger.warning("Failed to repatch MindSpeed args: %s", e)
    if updates.get("fp8") or updates.get("transformer_impl") == "transformer_engine":
        ensure_npu_transformer_engine_symbols()


def is_distribute_initialized():
    return mpu.model_parallel_is_initialized()


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        seed = seed_  # TuningFactory dataloader requires seed be the same for all ranks
        # # Ensure that different pipeline MP stages get different seeds.
        # seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # # Ensure different data parallel ranks get different seeds
        # if data_parallel_random_init:
        #     seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if current_platform.is_cuda() and current_platform.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def initialize_megatron(args: "TrainingArguments"):
    bootstrap_npu_runtime()
    sync_mindspeed_args(args)
    if not is_distribute_initialized():
        _initialize_distributed(args)
    _set_random_seed(args.seed)


def _initialize_distributed(args: "TrainingArguments"):
    """Initialize torch.distributed and core model parallel."""
    logger.info(f"Initializing mpu on device {args.device}")
    if not torch.distributed.is_initialized():
        # Manually set the device ids.
        current_platform.set_device(args.device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.ddp_backend or current_platform.communication_backend,
            rank=int(os.getenv("RANK", "0")),
            world_size=int(os.getenv("WORLD_SIZE", "1")),
            timeout=args.ddp_timeout_delta,
        )
    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if mpu.model_parallel_is_initialized():
        logger.info("model parallel is already initialized")
    else:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size if args.context_parallel_size is not None else 1,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
            distributed_timeout_minutes=args.ddp_timeout_delta.total_seconds() // 60,
        )
        logger.info(f"initialized tensor model parallel with size {mpu.get_tensor_model_parallel_world_size()}")
        logger.info(f"initialized pipeline model parallel with size {mpu.get_pipeline_model_parallel_world_size()}")
