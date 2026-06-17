import gc
import hashlib
import json
import time
from collections import OrderedDict
from typing import Iterable, Optional, Tuple

import torch
import vllm
from packaging.version import Version

from roll.platforms import current_platform
from roll.third_party.vllm.vllm_utils import TensorLoRARequest, patch_vllm_lora_manager
from roll.utils.collective import collective
from roll.utils.cuda_ipc_utils import MultiprocessingSerializer
from roll.utils.fp8 import is_mxfp8_ascend, per_block_fp8_quant_ascend
from roll.utils.logging import get_logger
from roll.utils.send_recv_utils import monkey_patch_torch_reductions, named_tensors_from_bucket

logger = get_logger()


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


def _quant_config_from_model(model) -> Optional[object]:
    """Best-effort extraction of the vLLM quant_config from a model runner/model."""
    if model is None:
        return None
    # model may be a ModelRunner – look for vllm_config.
    runner = model if not hasattr(model, "model") else None
    if runner is not None and hasattr(runner, "vllm_config"):
        return getattr(runner.vllm_config, "quant_config", None)
    # model may be the inner nn.Module; try walking up via model_runner ref.
    inner = getattr(model, "model", model)
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

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # before updating the parameters, we need to reinitialize the previously released model
        self.reload_model()
        model = self.model_runner.model

        if vllm.__version__ < "0.8.5":
            from roll.third_party.vllm.vllm_utils import patch_vllm_moe_model_weight_loader

            patch_vllm_moe_model_weight_loader(model)

        # Ascend MXFP8 three-phase weight loading lifecycle:
        #   ① restore HF shapes → ② load_weights → ③ re-apply NPU transforms
        if self._is_mxfp8_model:
            restore_mxfp8_weights_for_loading(model)
            weights = _quantize_mxfp8_weights_for_loading(
                weights,
                model,
                dtype=getattr(self.model_config, "dtype", torch.bfloat16),
            )

        model.load_weights(weights=weights)

        if self._is_mxfp8_model:
            apply_mxfp8_transformation_after_loading(model)

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True
        if vllm.__version__ < "0.8.5" and self.buffers is not None:
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
        if vllm.__version__ < "0.8.5" and level == 2:
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
        model = self.model_runner.model

        # Ascend MXFP8 three-phase lifecycle (same as load_weights).
        if self._is_mxfp8_model:
            restore_mxfp8_weights_for_loading(model)
            named_params = _quantize_mxfp8_weights_for_loading(
                named_params,
                model,
                dtype=getattr(self.model_config, "dtype", torch.bfloat16),
            )

        model.load_weights(named_params)

        if self._is_mxfp8_model:
            apply_mxfp8_transformation_after_loading(model)

    def process_weights_after_loading(self):
        if Version(vllm.__version__) >= Version("0.11.1"):
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            from vllm.utils.torch_utils import set_default_torch_dtype
            device_config = self.device_config
            load_config = self.vllm_config.load_config
            load_device = (device_config.device if load_config.device is None else load_config.device)
            target_device = torch.device(load_device)
            with set_default_torch_dtype(self.model_config.dtype):
                process_weights_after_loading(self.model_runner.model,self.model_config,target_device)
        if (Version("0.11.0") == Version(vllm.__version__) or
                Version("0.11.1rc1") == Version(vllm.__version__) or
                Version("0.11.1rc2.dev0+gc3a722fcb.d20251021") == Version(vllm.__version__)):
            from vllm.model_executor.model_loader.utils import process_weights_after_loading,set_default_torch_dtype
            device_config = self.device_config
            load_config = self.vllm_config.load_config
            load_device = (device_config.device if load_config.device is None else load_config.device)
            target_device = torch.device(load_device)
            with set_default_torch_dtype(self.model_config.dtype):
                process_weights_after_loading(self.model_runner.model,self.model_config,target_device)


class WorkerV1(WorkerBase):
    def custom_init_worker(self, *args, **kwargs):
        super().custom_init_worker(*args, **kwargs)
        patch_vllm_lora_manager()
        # Re-detect MXFP8 in case model_runner wasn't fully initialised during
        # the base-class init (V1 lazy-init edge case).
        if not self._is_mxfp8_model:
            quant_config = _quant_config_from_model(self.model_runner)
            self._is_mxfp8_model = is_mxfp8_ascend(quant_config)

    # Use custom prefix because worker_extension_cls can not has
    # conflicting method name with vllm worker.
    def custom_add_lora(self, peft_config) -> bool:
        lora_request = self.tensor_lora_manager.build_request(peft_config)
        super().reload_model()
        return self.model_runner.add_lora(lora_request)
