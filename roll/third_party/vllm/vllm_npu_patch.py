"""NPU IPC and vLLM-Ascend MoE compatibility patches used by ROLL.

The patch functions load torch_npu and vllm_ascend on demand. Install the
corresponding patch only from an NPU path, before serialization or model load.
The MoE wrappers cover unquantized Parameter replacements observed with
vllm_ascend 0.23.0rc1, without changing backend weight layout or copy logic.
"""

import os
from functools import wraps


_STORAGE_DEVICE_INDEX = 6
_PHYSICAL_DEVICE_TAG = "roll_npu_physical"
_MOE_PARAMETER_NAMES = ("w13_weight", "w2_weight")


def _visible_npu_devices():
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    return [device.strip() for device in visible.split(",")] if visible else None


def _npu_device_to_physical_id(device):
    devices = _visible_npu_devices()
    index = int(device)
    if index < 0 or (devices is not None and index >= len(devices)):
        raise RuntimeError(f"Invalid NPU IPC device index {index} for visible devices {devices}")
    physical = devices[index] if devices is not None else str(index)
    return _PHYSICAL_DEVICE_TAG, physical


def _npu_device_from_physical_id(device):
    # Continue accepting tensors serialized without the ROLL patch.
    if not isinstance(device, tuple) or device[0] != _PHYSICAL_DEVICE_TAG:
        return device
    physical = device[1]
    devices = _visible_npu_devices()
    if devices is None:
        return int(physical)
    if physical not in devices:
        raise RuntimeError(f"IPC source NPU {physical} is not visible in {devices}")
    return devices.index(physical)


def _replace_storage_device(args, modifier):
    if len(args) <= _STORAGE_DEVICE_INDEX:
        raise RuntimeError("Unsupported torch_npu IPC rebuild arguments: missing storage device")
    return (*args[:_STORAGE_DEVICE_INDEX], modifier(args[_STORAGE_DEVICE_INDEX]), *args[_STORAGE_DEVICE_INDEX + 1 :])


def _reduce_npu_tensor(tensor):
    from torch.multiprocessing import reductions
    from torch_npu.multiprocessing import reductions as npu_reductions

    # Keep CPU tensors on PyTorch's normal path, including when torch_npu's
    # reducer delegates back to reductions.reduce_tensor for non-NPU tensors.
    if tensor.device.type != "npu":
        return reductions._roll_npu_fallback_reduce_tensor_original(tensor)
    rebuild, args = reductions._roll_npu_reduce_tensor_original(tensor)
    if rebuild is npu_reductions.rebuild_npu_tensor:
        return _rebuild_npu_tensor, _replace_storage_device(args, _npu_device_to_physical_id)
    return rebuild, args


def _rebuild_npu_tensor(*args):
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    # A trainer can see physical NPU 13 as logical 0 while a TP2 worker sees
    # physical NPUs 12,13 as logical 0,1. Reopen IPC on the same physical NPU.
    return rebuild_npu_tensor(*_replace_storage_device(args, _npu_device_from_physical_id))


def patch_npu_ipc_reductions():
    """Register an importable NPU reducer with physical device identifiers."""
    from torch.multiprocessing import reductions
    from torch_npu.multiprocessing import reductions as npu_reductions

    if hasattr(reductions, "_roll_npu_reduce_tensor_original"):
        return
    reductions._roll_npu_reduce_tensor_original = npu_reductions._npu_reduce_tensor
    reductions._roll_npu_fallback_reduce_tensor_original = reductions.reduce_tensor
    reductions.reduce_tensor = _reduce_npu_tensor
    reductions.init_reductions()


def validate_npu_ipc_bucket(bucket, device):
    """Reject a cross-device bucket before a weight copy can fault on NPU."""
    if bucket.device != device:
        raise RuntimeError(f"NPU IPC bucket device {bucket.device} does not match worker {device}")


def _capture_moe_parameter_attrs(modules):
    saved = []
    for module in modules:
        for name in _MOE_PARAMETER_NAMES:
            param = module._parameters.get(name)
            if param is not None and param.__dict__:
                saved.append((module, name, param.__dict__.copy()))
    return saved


def _restore_moe_parameter_attrs(saved):
    for module, name, attrs in saved:
        param = module._parameters.get(name)
        if param is not None:
            for key, value in attrs.items():
                # Preserve any metadata deliberately supplied by the backend.
                param.__dict__.setdefault(key, value)


def patch_vllm_ascend_moe_metadata():
    """Preserve MoE loading metadata across backend Parameter replacements.

    Install before the first model load: initial weight postprocessing already
    replaces parameters, before ROLL's custom_init_worker RPC is invoked.
    """
    from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
    from vllm_ascend.worker.worker import NPUWorker

    process = getattr(AscendUnquantizedFusedMoEMethod, "process_weights_after_loading", None)
    wake_up = getattr(NPUWorker, "wake_up", None)
    if not callable(process) or not callable(wake_up):
        raise RuntimeError("Unsupported vllm_ascend MoE interface: expected weight postprocessing and NPUWorker.wake_up")

    if not getattr(process, "_roll_preserves_moe_metadata", False):

        @wraps(process)
        def process_with_metadata(self, layer, *args, **kwargs):
            saved = _capture_moe_parameter_attrs((layer,))
            result = process(self, layer, *args, **kwargs)
            _restore_moe_parameter_attrs(saved)
            return result

        process_with_metadata._roll_preserves_moe_metadata = True
        AscendUnquantizedFusedMoEMethod.process_weights_after_loading = process_with_metadata

    if not getattr(wake_up, "_roll_preserves_moe_metadata", False):

        @wraps(wake_up)
        def wake_up_with_metadata(self, *args, **kwargs):
            if self.vllm_config.quant_config is not None:
                return wake_up(self, *args, **kwargs)
            saved = _capture_moe_parameter_attrs(self.model_runner.model.modules())
            result = wake_up(self, *args, **kwargs)
            _restore_moe_parameter_attrs(saved)
            return result

        wake_up_with_metadata._roll_preserves_moe_metadata = True
        NPUWorker.wake_up = wake_up_with_metadata
