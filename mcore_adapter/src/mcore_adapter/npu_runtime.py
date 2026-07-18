import importlib.util
import sys
from typing import TYPE_CHECKING

import torch
from megatron.core import tensor_parallel

from .platforms import current_platform
from .utils import get_logger

if TYPE_CHECKING:
    from .training_args import TrainingArguments


logger = get_logger(__name__)

# Core NPU Megatron adaptor args that ROLL always overrides when deriving fp8/attention settings.
_MEGATRON_ADAPTOR_DERIVED_ARGS = frozenset({
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


def _import_megatron_adaptor():
    try:
        return __import__("megatron_adaptor")
    except ImportError as exc:
        raise RuntimeError(
            "MegatronAdaptor core_r0.17.0 is required to initialize Megatron on Ascend NPU."
        ) from exc


def _megatron_adaptor_loaded():
    return "megatron_adaptor" in sys.modules


def _npu_te_checkpoint(function, distribute_saved_activations, get_rng_state_tracker, tp_group, *args):
    return tensor_parallel.checkpoint(function, distribute_saved_activations, *args)


def _ensure_npu_te_checkpoint_symbol(te_ext):
    if not hasattr(te_ext, "te_checkpoint"):
        te_ext.te_checkpoint = _npu_te_checkpoint


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
    except ImportError as exc:
        raise RuntimeError(
            "TransformerEngineNPU is required for Megatron FP8 on Ascend NPU."
        ) from exc

    if not hasattr(te_ext, "TENorm"):
        raise RuntimeError(
            "TransformerEngineNPU did not provide megatron.core.extensions.transformer_engine.TENorm."
        )
    _ensure_npu_te_checkpoint_symbol(te_ext)


def bootstrap_npu_runtime():
    global _NPU_RUNTIME_BOOTSTRAPPED

    if _NPU_RUNTIME_BOOTSTRAPPED or not current_platform.is_npu():
        return

    import torch_npu  # noqa: F401

    _import_megatron_adaptor()
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


def apply_megatron_adaptor_feature_defaults(config):
    if not _megatron_adaptor_loaded():
        return

    try:
        from megatron_adaptor.utils.args_utils import get_mindspeed_args
    except ImportError as exc:
        raise RuntimeError(
            "The installed MegatronAdaptor is incompatible: megatron_adaptor.utils.args_utils is missing."
        ) from exc

    for name, value in vars(get_mindspeed_args(get_defaults=True)).items():
        if not hasattr(config, name):
            setattr(config, name, value)


def sync_megatron_adaptor_args(args: "TrainingArguments"):
    if not _megatron_adaptor_loaded():
        return

    try:
        from megatron_adaptor.utils.args_utils import get_mindspeed_args
    except ImportError as exc:
        raise RuntimeError(
            "The installed MegatronAdaptor is incompatible: megatron_adaptor.utils.args_utils is missing."
        ) from exc

    adaptor_args = get_mindspeed_args()

    # Auto-discover syncable args: any field present on both ROLL TrainingArguments
    # and MegatronAdaptor is eligible. The derived-args set ensures fp8/attention rules
    # are applied even when the source arg was not directly set on args.
    adaptor_arg_names = set(vars(get_mindspeed_args(get_defaults=True)))
    updates = {}
    for name in adaptor_arg_names | _MEGATRON_ADAPTOR_DERIVED_ARGS:
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
        if getattr(adaptor_args, name, None) != value:
            setattr(adaptor_args, name, value)
            changed = True

    if changed and current_platform.is_npu() and _has_megatron_training():
        try:
            megatron_adaptor = sys.modules.get("megatron_adaptor")

            if megatron_adaptor is not None and hasattr(megatron_adaptor, "repatch"):
                megatron_adaptor.repatch(updates)
        except Exception as e:
            logger.warning("Failed to repatch NPU Megatron adaptor args: %s", e)
    if updates.get("fp8") or updates.get("transformer_impl") == "transformer_engine":
        ensure_npu_transformer_engine_symbols()
