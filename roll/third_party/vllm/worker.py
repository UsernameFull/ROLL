import gc
import hashlib
import json
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Iterable, Optional, Tuple

import torch
import vllm
from packaging.version import Version

from roll.platforms import current_platform
from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.third_party.vllm.parameter_utils import (
    TemporaryParameterSubclassTypes,
    copy_parameter_metadata,
    parameter_from_runtime_ref,
    runtime_value_from_ref,
)
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.fp8 import is_mxfp8_ascend, per_block_fp8_quant_ascend
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


def _is_legacy_non_npu_vllm_before(version: str) -> bool:
    return not current_platform.is_npu() and Version(vllm.__version__) < Version(version)


# ---------------------------------------------------------------------------
# MXFP8 weight lifecycle helpers (Ascend NPU)
# ---------------------------------------------------------------------------


def restore_mxfp8_weights_for_loading(model: torch.nn.Module) -> None:
    """Restore MXFP8-transformed weights to their original HuggingFace shapes.

    On Ascend NPUs, vLLM-Ascend's ``process_weights_after_loading`` transposes
    and reshapes weights for optimal NPU inference.  Before calling
    ``model.load_weights()`` we must revert those transformations so the
    weight loaders receive weights in the original HF layout.

    This function iterates through all modules that have been marked as
    MXFP8-transformed and calls the Ascend-provided restore method.
    """
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    restored_count = 0
    for _name, module in model.named_modules():
        if not isinstance(module, (LinearBase, FusedMoE)):
            continue
        if not hasattr(module, "_mxfp8_transformed"):
            continue
        if not hasattr(module, "quant_method"):
            continue

        quant_method = module.quant_method
        # The Ascend quant method may be wrapped in an outer object.
        inner = getattr(quant_method, "quant_method", quant_method)
        restore_fn = getattr(inner, "restore_weights_for_rl_loading", None)
        if restore_fn is not None:
            restore_fn(module)
            restored_count += 1

    if restored_count > 0:
        logger.info(
            "MXFP8: restored %d modules to HF shapes before weight loading",
            restored_count,
        )


def apply_mxfp8_transformation_after_loading(model: torch.nn.Module) -> None:
    """Re-apply MXFP8 transformations after ``model.load_weights()``.

    After weights have been loaded in HF format, the Ascend-required
    transpose/reshape transformations must be re-applied so the model is
    ready for NPU inference.  This calls ``process_weights_after_loading``
    on every module that tracks its original shapes via
    ``_mxfp8_original_shapes``.
    """
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    transformed_count = 0
    for _name, module in model.named_modules():
        if not isinstance(module, (LinearBase, FusedMoE)):
            continue
        if not hasattr(module, "_mxfp8_original_shapes"):
            continue
        if hasattr(module, "quant_method") and hasattr(module.quant_method, "process_weights_after_loading"):
            module.quant_method.process_weights_after_loading(module)
            transformed_count += 1

    if transformed_count > 0:
        logger.info(
            "MXFP8: re-applied transformation on %d modules after weight loading",
            transformed_count,
        )


def _vllm_config_context(vllm_config):
    try:
        from vllm.config import set_current_vllm_config

        return set_current_vllm_config(vllm_config)
    except ImportError:
        return nullcontext()


def _temporary_parameter_subclass_types(model: torch.nn.Module):
    return TemporaryParameterSubclassTypes(model)


_MXFP8_RUNTIME_TENSOR_ATTRS = (
    "weight",
    "weight_scale",
    "weight_scale_inv",
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_scale_inv",
)


class Mxfp8WeightLifecycle:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def iter_runtime_modules(self):
        from vllm.model_executor.layers.linear import LinearBase

        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        except ImportError:
            from vllm.model_executor.layers.fused_moe import FusedMoE

        for name, module in self.model.named_modules():
            if not isinstance(module, (LinearBase, FusedMoE)):
                continue
            if not (hasattr(module, "_mxfp8_original_shapes") or hasattr(module, "_mxfp8_transformed")):
                continue
            yield name, module

    def has_runtime_refs(self) -> bool:
        return any(getattr(module, "_roll_mxfp8_runtime_refs", None) for _name, module in self.iter_runtime_modules())

    def record_runtime_refs(self) -> int:
        recorded = 0
        for _name, module in self.iter_runtime_modules():
            refs = {}
            for attr in _MXFP8_RUNTIME_TENSOR_ATTRS:
                tensor = _tensor_data(getattr(module, attr, None))
                if tensor is not None:
                    refs[attr] = tensor

            if refs:
                module._roll_mxfp8_runtime_refs = refs
                recorded += len(refs)

        if recorded > 0:
            logger.info("MXFP8: recorded %d runtime tensor refs for graph-safe reload", recorded)
        return recorded

    def restore_runtime_refs(self) -> int:
        restored = 0
        errors = []

        for module_name, module in self.iter_runtime_modules():
            refs = getattr(module, "_roll_mxfp8_runtime_refs", None)
            if not refs:
                continue

            for attr, ref in refs.items():
                current = getattr(module, attr, None)
                tensor = _tensor_data(current)
                if tensor is None:
                    errors.append(f"{module_name}.{attr}: missing transformed tensor")
                    continue

                if _tensor_signature(tensor) != _tensor_signature(ref):
                    errors.append(
                        f"{module_name}.{attr}: expected {_tensor_signature(ref)}, got {_tensor_signature(tensor)}"
                    )
                    continue

                with torch.no_grad():
                    ref.copy_(tensor, non_blocking=False)

                setattr(module, attr, runtime_value_from_ref(ref, current))
                restored += 1

        if errors:
            preview = "; ".join(errors[:5])
            raise RuntimeError(
                "MXFP8 graph-safe reload failed because transformed tensor metadata changed. "
                "Use enforce_eager=True or recapture graphs through a coordinated graph lifecycle. "
                f"First mismatches: {preview}"
            )

        if restored > 0:
            logger.info("MXFP8: restored %d transformed tensors in-place for graph-safe reload", restored)
        return restored

    def apply_transformation_after_loading(self, preserve_runtime_refs: bool) -> bool:
        has_runtime_refs = preserve_runtime_refs and self.has_runtime_refs()

        apply_mxfp8_transformation_after_loading(self.model)

        if has_runtime_refs:
            self.restore_runtime_refs()
            self.record_runtime_refs()
            return True

        self.record_runtime_refs()
        return False


def _tensor_data(value) -> Optional[torch.Tensor]:
    if isinstance(value, torch.nn.Parameter):
        return value.data
    if isinstance(value, torch.Tensor):
        return value
    return None


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _iter_mxfp8_runtime_modules(model: torch.nn.Module):
    yield from Mxfp8WeightLifecycle(model).iter_runtime_modules()


def _record_mxfp8_runtime_refs(model: torch.nn.Module) -> int:
    """Remember transformed runtime tensors so graph replay can keep addresses."""
    return Mxfp8WeightLifecycle(model).record_runtime_refs()


def _restore_mxfp8_runtime_refs(model: torch.nn.Module) -> int:
    """Copy newly transformed tensors back into the refs captured by ACL graph."""
    return Mxfp8WeightLifecycle(model).restore_runtime_refs()


def _has_mxfp8_runtime_refs(model: torch.nn.Module) -> bool:
    return Mxfp8WeightLifecycle(model).has_runtime_refs()


def _apply_mxfp8_transformation_after_loading(
    model: torch.nn.Module,
    preserve_runtime_refs: bool,
) -> bool:
    return Mxfp8WeightLifecycle(model).apply_transformation_after_loading(preserve_runtime_refs)


def _clear_aclgraph_cache(obj, seen: set[int] | None = None) -> int:
    """Best-effort invalidation for vLLM-Ascend ACL graph caches.

    ACL graphs capture parameter storage addresses and should not be replayed
    after RL weight refits that replace FP8 weight/scale storage.  Clearing the
    entries makes the next request re-capture graphs with the updated weights.
    """
    if obj is None:
        return 0
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    cleared = 0

    # Clear ACL graph entries directly on this object.
    entries = getattr(obj, "concrete_aclgraph_entries", None)
    if isinstance(entries, dict):
        cleared += len(entries)
        entries.clear()

    # Recurse into child attributes: nn.Module children, plus any dict/list/tuple
    # attributes that may hold graph managers / dispatchers / runners.
    if isinstance(obj, torch.nn.Module):
        for child in obj.children():
            cleared += _clear_aclgraph_cache(child, seen)

    try:
        obj_vars = vars(obj)
    except TypeError:
        obj_vars = {}
    for attr_name, child in obj_vars.items():
        if attr_name.startswith("_"):
            continue
        if isinstance(child, (dict, list, tuple)):
            items = child.values() if isinstance(child, dict) else child
            for item in items:
                cleared += _clear_aclgraph_cache(item, seen)
        elif isinstance(child, torch.nn.Module):
            cleared += _clear_aclgraph_cache(child, seen)
    return cleared


def invalidate_aclgraph_cache_after_weight_update(model_runner) -> None:
    cleared = _clear_aclgraph_cache(model_runner)
    if cleared > 0:
        logger.info("ACLGraph: cleared %d cached graph entries after weight update", cleared)


def _quant_config_from_model(model) -> Optional[object]:
    """Best-effort extraction of the vLLM quant_config from a model runner/model."""
    if model is None:
        return None

    # model may be a ModelRunner/Worker that directly carries vllm_config.
    if hasattr(model, "vllm_config"):
        return getattr(model.vllm_config, "quant_config", None)

    # model may be the inner nn.Module; try walking up via model_runner ref.
    inner = getattr(model, "model", model)
    if hasattr(inner, "vllm_config"):
        return getattr(inner.vllm_config, "quant_config", None)

    runner = getattr(inner, "model_runner", None)
    if runner is not None and hasattr(runner, "vllm_config"):
        return getattr(runner.vllm_config, "quant_config", None)
    return None


def _get_module_from_param_name(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    """Resolve the vLLM module that owns a checkpoint parameter name."""
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except ImportError:
        from vllm.model_executor.layers.fused_moe import FusedMoE

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names in packed_modules_mapping.items()
        for original_name in original_names
    }

    module_path = name.split(".")[:-1]
    if module_path and module_path[-1] in reversed_mapping:
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        for part in module_path:
            if isinstance(current_module, FusedMoE):
                return current_module
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError):
        return None
    return current_module


# Router/gate weights in MoE models should stay in full precision (FLOAT).
# These names match vLLM's packed-module naming for Qwen-MoE, Mixtral, etc.
_MOE_ROUTER_WEIGHT_NAME_SUFFIXES = (
    ".gate.weight",
    ".router.weight",
    ".shared_expert_gate.weight",
)


def _is_moe_router_weight(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _MOE_ROUTER_WEIGHT_NAME_SUFFIXES)


def _is_mxfp8_weight_name(
    name: str,
    model: torch.nn.Module,
    seen_params: set[str],
    mxfp8_param_names: set[str],
) -> bool:
    if name not in seen_params:
        seen_params.add(name)
        if not name.endswith("weight"):
            return False
        if _is_moe_router_weight(name):
            return False

        from vllm.model_executor.layers.linear import LinearBase

        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        except ImportError:
            from vllm.model_executor.layers.fused_moe import FusedMoE

        module = _get_module_from_param_name(model, name)
        if isinstance(module, LinearBase) and module.weight.dtype == torch.float8_e4m3fn:
            mxfp8_param_names.add(name)
        elif (
            isinstance(module, FusedMoE)
            and module.w13_weight.dtype == torch.float8_e4m3fn
            and module.w2_weight.dtype == torch.float8_e4m3fn
        ):
            mxfp8_param_names.add(name)

    return name in mxfp8_param_names


def _quantize_mxfp8_weights_for_loading(
    weights: Iterable[Tuple[str, torch.Tensor]],
    model: torch.nn.Module,
    dtype: torch.dtype,
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Quantize bf16/fp16 synced weights into Ascend MXFP8 weight+scale pairs."""
    seen_params: set[str] = set()
    mxfp8_param_names: set[str] = set()

    for name, weight in weights:
        if not _is_mxfp8_weight_name(name, model, seen_params, mxfp8_param_names):
            yield name, weight
            continue

        if weight.dtype == torch.float8_e4m3fn:
            yield name, weight
            continue

        quantized_weight, weight_scale = per_block_fp8_quant_ascend(weight, dtype=dtype)
        weight_scale = weight_scale.squeeze(-1)
        yield name, quantized_weight
        yield f"{name}_scale", weight_scale


class TensorLoraManager:
    def __init__(self):
        self.lora_params = OrderedDict()
        self.add_lora_count = 0

    def add_weight(self, name: str, weight: torch.Tensor):
        self.lora_params[name] = weight

    def build_request(self, peft_config: dict) -> TensorLoRARequest:
        """
        Generate a unique LoRA ID based on the PEFT configuration rather than
        using a timestamp to assert all tp-ranks get the same LoRA ID.
        """
        self.add_lora_count += 1
        peft_config["add_lora_count"] = self.add_lora_count
        peft_config_str = json.dumps(peft_config, sort_keys=True)
        hash_obj = hashlib.sha256(peft_config_str.encode("utf-8"))
        hex_dig = hash_obj.hexdigest()
        lora_int_id = int(hex_dig, 16) % 0x7FFFFFFF

        lora_request = TensorLoRARequest(
            lora_name=f"{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path="dummy_lora_path",
            peft_config=peft_config,
            lora_tensors=self.lora_params,
        )
        del self.lora_params
        self.lora_params = OrderedDict()
        return lora_request


class WorkerBase:
    def custom_init_worker(self, *args, **kwargs):
        self.weight_loaded: bool = True
        self.kv_cache_loaded: bool = True
        self.buffers = None
        self.buffer_cache = None
        self.tensor_lora_manager = TensorLoraManager()
        # Detect MXFP8 once at init time.
        quant_config = _quant_config_from_model(self.model_runner)
        self._is_mxfp8_model: bool = is_mxfp8_ascend(quant_config)
        # Tracks whether NPU layout transformations have been applied to the
        # current model weights.  False until the first process_weights_after_loading
        # or _finalize_weight_loading call; used to skip the restore step on the
        # very first RL weight sync (nothing to restore yet).
        self._mxfp8_transformation_applied: bool = False
        logger.info(
            "MXFP8 worker detection: enabled=%s quant_config=%s",
            self._is_mxfp8_model,
            type(quant_config).__name__ if quant_config is not None else None,
        )

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    # ------------------------------------------------------------------
    # Shared MXFP8 weight-loading helpers
    # ------------------------------------------------------------------

    def _prepare_weights_for_loading(self, model, named_params):
        """Restore HF shapes and quantize weights for Ascend MXFP8 models.

        The restore step is skipped on the very first weight sync because
        process_weights_after_loading has not run yet (no NPU layout
        transformations applied), so there is nothing to undo.
        """
        if not self._is_mxfp8_model:
            return named_params
        if self._mxfp8_transformation_applied:
            restore_mxfp8_weights_for_loading(model)
        return _quantize_mxfp8_weights_for_loading(
            named_params, model,
            dtype=getattr(self.model_config, "dtype", torch.bfloat16),
        )

    def _finalize_weight_loading(self, model):
        """Apply MXFP8 transforms and invalidate ACL graphs after load_weights."""
        if self._is_mxfp8_model:
            self._mxfp8_transformation_applied = True
            with _vllm_config_context(self.vllm_config):
                if _apply_mxfp8_transformation_after_loading(model, preserve_runtime_refs=True):
                    return  # graph-safe reload – no invalidation needed
        invalidate_aclgraph_cache_after_weight_update(self.model_runner)

    def _load_weights_internal(self, model, named_params):
        """Load weights through the MXFP8 lifecycle: restore → quantize → load → finalize.

        Returns the materialized weight list (after possible quantization) so
        it can be re-used for MTP/EAGLE drafter models.
        """
        named_params = list(self._prepare_weights_for_loading(model, named_params))
        with _temporary_parameter_subclass_types(model):
            model.load_weights(weights=named_params)
        self._finalize_weight_loading(model)
        return named_params

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # before updating the parameters, we need to reinitialize the previously released model
        self.reload_model()
        model = self.model_runner.model

        if _is_legacy_non_npu_vllm_before("0.8.5"):
            from roll.third_party.vllm.vllm_utils import patch_vllm_moe_model_weight_loader
            patch_vllm_moe_model_weight_loader(model)

        weights_list = self._load_weights_internal(model, weights)

        # MTP/EAGLE drafter models are separate modules and need the same live
        # weight refresh as the target model.
        if hasattr(self.model_runner, "drafter") and hasattr(self.model_runner.drafter, "model"):
            logger.info("Updating drafter (MTP/EAGLE) model weights...")
            drafter_model = self.model_runner.drafter.model
            with _temporary_parameter_subclass_types(drafter_model):
                drafter_model.load_weights(weights=weights_list)

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True
        if _is_legacy_non_npu_vllm_before("0.8.5") and self.buffers is not None:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self.buffers:
                    buffer.data.copy_(self.buffers[name].data)
            self.buffers = None

    def offload_states(self, level):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            return
        if _is_legacy_non_npu_vllm_before("0.8.5") and level == 2:
            # https://github.com/vllm-project/vllm/issues/16564
            model = self.model_runner.model
            self.buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
        self.sleep(level)
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, "recv_manager"):
            self.recv_manager.clear()
        gc.collect()
        current_platform.empty_cache()

    def setup_collective_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        group_rank = self.rank + rank_offset
        collective.init_collective_group(
            world_size,
            rank=group_rank,
            backend=backend,
            group_name=group_name,
            master_addr=master_address,
            master_port=master_port,
        )
        logger.info(f"setup_collective_group: {group_name} rank: {group_rank} world_size: {world_size}")

    def broadcast_parameter(self, names, dtypes, shapes, group_name, is_lora=False):
        weights_and_handles = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            weight = torch.empty(shape, dtype=target_dtype, device=self.device)
            handle = collective.broadcast(tensor=weight, src_rank=0, group_name=group_name, async_op=True)
            weights_and_handles.append((name, weight, handle))

        def weights_iter():
            for name, weight, handle in weights_and_handles:
                handle.wait()
                yield name, weight

        if is_lora:
            for name, weight in weights_iter():
                self.tensor_lora_manager.add_weight(name, weight)
            return
        self.load_weights(weights=weights_iter())

    def update_parameter_in_bucket(self, serialized_named_tensors, is_lora=False):
        monkey_patch_torch_reductions()
        bucket_with_meta = MultiprocessingSerializer.deserialize(serialized_named_tensors[self.rank])
        named_params = named_tensors_from_bucket(**bucket_with_meta)
        if is_lora:
            for name, weight in named_params:
                self.tensor_lora_manager.add_weight(name, weight)
            return

        self.reload_model()
        self._load_weights_internal(self.model_runner.model, named_params)

    def process_weights_after_loading(self):
        if not current_platform.is_npu():
            vllm_ver = Version(vllm.__version__)
            is_supported = (
                vllm_ver >= Version("0.11.1")
                or vllm_ver
                in (
                    Version("0.11.0"),
                    Version("0.11.1rc1"),
                    Version("0.11.1rc2.dev0+gc3a722fcb.d20251021"),
                )
            )
            if not is_supported:
                return

        from vllm.model_executor.model_loader.utils import process_weights_after_loading
        from vllm.utils.torch_utils import set_default_torch_dtype

        load_device = self.device_config.device
        if self.vllm_config.load_config.device is not None:
            load_device = self.vllm_config.load_config.device
        with set_default_torch_dtype(self.model_config.dtype), _vllm_config_context(self.vllm_config):
            process_weights_after_loading(self.model_runner.model, self.model_config, torch.device(load_device))

        if self._is_mxfp8_model:
            self._mxfp8_transformation_applied = True
            _record_mxfp8_runtime_refs(self.model_runner.model)
        invalidate_aclgraph_cache_after_weight_update(self.model_runner)


class WorkerV1(WorkerBase):
    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        patch_vllm_lora_manager()
        # Re-detect MXFP8 in case model_runner wasn't fully initialised during
        # the base-class init (V1 lazy-init edge case).
        if not self._is_mxfp8_model:
            quant_config = _quant_config_from_model(self.model_runner)
            self._is_mxfp8_model = is_mxfp8_ascend(quant_config)
            logger.info(
                "MXFP8 worker re-detection: enabled=%s quant_config=%s",
                self._is_mxfp8_model,
                type(quant_config).__name__ if quant_config is not None else None,
            )

    # Use custom prefix because worker_extension_cls can not has
    # conflicting method name with vllm worker.
    def custom_add_lora(self, peft_config) -> bool:
        lora_request = self.tensor_lora_manager.build_request(peft_config)
        super().reload_model()
        return self.model_runner.add_lora(lora_request)
