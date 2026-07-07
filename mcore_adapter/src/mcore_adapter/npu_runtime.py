import importlib.util
import sys
import types
from typing import TYPE_CHECKING

import torch
from megatron.core import tensor_parallel

from .platforms import current_platform
from .utils import get_logger

if TYPE_CHECKING:
    from .training_args import TrainingArguments


logger = get_logger(__name__)

# Core NPU Megatron adaptor args that ROLL always overrides when deriving fp8/attention settings.
_MINDSPEED_DERIVED_ARGS = frozenset({
    "transformer_impl",
    "fp8",
    "fp8_format",
    "use_flash_attn",
    "use_flash_attn_npu_batch_invariant",
    "te_comparison_with_cpu",
    "te_comparison_with_bf16",
})

_NPU_RUNTIME_BOOTSTRAPPED = False


def _has_megatron_training():
    return importlib.util.find_spec("megatron.training") is not None


def _import_npu_megatron_adaptor():
    try:
        return __import__("mindspeed.megatron_adaptor")
    except ImportError:
        return None


def _npu_megatron_adaptor_loaded():
    return "mindspeed.megatron_adaptor" in sys.modules


def _npu_te_checkpoint(function, distribute_saved_activations, get_rng_state_tracker, tp_group, *args):
    return tensor_parallel.checkpoint(function, distribute_saved_activations, *args)


def _ensure_npu_te_checkpoint_symbols(te_ext):
    if not hasattr(te_ext, "te_checkpoint"):
        te_ext.te_checkpoint = _npu_te_checkpoint

    te_module = sys.modules.setdefault("transformer_engine", types.ModuleType("transformer_engine"))
    pytorch_module = sys.modules.setdefault(
        "transformer_engine.pytorch", types.ModuleType("transformer_engine.pytorch")
    )
    distributed_module = sys.modules.setdefault(
        "transformer_engine.pytorch.distributed",
        types.ModuleType("transformer_engine.pytorch.distributed"),
    )
    if not hasattr(distributed_module, "checkpoint"):
        distributed_module.checkpoint = _npu_te_checkpoint
    if not hasattr(te_module, "pytorch"):
        te_module.pytorch = pytorch_module
    if not hasattr(pytorch_module, "distributed"):
        pytorch_module.distributed = distributed_module


def _patch_loaded_npu_transformer_modules(te_norm):
    for module_name in (
        "megatron.core.models.gpt.gpt_layer_specs",
        "mindspeed.core.models.gpt.gpt_layer_specs",
        "megatron.core.transformer.transformer_block",
        "mindspeed.core.transformer.transformer_block",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if getattr(module, "TENorm", None) is not te_norm:
            setattr(module, "TENorm", te_norm)
        if getattr(module, "te_checkpoint", None) is None:
            setattr(module, "te_checkpoint", _npu_te_checkpoint)


def get_te_checkpoint_or_none():
    try:
        from megatron.core.extensions.transformer_engine import te_checkpoint
    except ImportError:
        return None
    return te_checkpoint


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
    _ensure_npu_te_checkpoint_symbols(te_ext)
    _patch_loaded_npu_transformer_modules(TENorm)


def bootstrap_npu_runtime():
    global _NPU_RUNTIME_BOOTSTRAPPED

    if _NPU_RUNTIME_BOOTSTRAPPED or not current_platform.is_npu():
        return

    import torch_npu  # noqa: F401

    _import_npu_megatron_adaptor()
    ensure_npu_transformer_engine_symbols()

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
    if not _npu_megatron_adaptor_loaded():
        return

    try:
        from mindspeed.args_utils import get_mindspeed_args
    except ImportError:
        return

    for name, value in vars(get_mindspeed_args(get_defaults=True)).items():
        if not hasattr(config, name):
            setattr(config, name, value)


def sync_mindspeed_args(args: "TrainingArguments"):
    if not _npu_megatron_adaptor_loaded():
        return

    try:
        from mindspeed.args_utils import get_mindspeed_args
    except ImportError:
        return

    mindspeed_args = get_mindspeed_args()

    # Auto-discover syncable args: any field present on both ROLL TrainingArguments
    # and MindSpeed is eligible.  The derived-args set ensures fp8/attention rules
    # are applied even when the source arg was not directly set on args.
    mindspeed_arg_names = set(vars(get_mindspeed_args(get_defaults=True)))
    updates = {}
    for name in mindspeed_arg_names | _MINDSPEED_DERIVED_ARGS:
        value = getattr(args, name, None)
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
            megatron_adaptor = sys.modules.get("mindspeed.megatron_adaptor")

            if megatron_adaptor is not None and hasattr(megatron_adaptor, "repatch"):
                megatron_adaptor.repatch(updates)
        except Exception as e:
            logger.warning("Failed to repatch NPU Megatron adaptor args: %s", e)
    if updates.get("fp8") or updates.get("transformer_impl") == "transformer_engine":
        ensure_npu_transformer_engine_symbols()
