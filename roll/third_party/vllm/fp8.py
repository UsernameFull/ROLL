import inspect
import weakref
from functools import partial
from typing import List
from unittest.mock import patch

import torch
import vllm
from packaging.version import Version
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.parameter import BlockQuantScaleParameter, ModelWeightParameter
from vllm.platforms import current_platform
from vllm.model_executor.utils import set_weight_attrs
from vllm._custom_ops import scaled_fp8_quant as per_tensor_fp8_quant
from vllm.model_executor.layers.quantization.utils.w8a8_utils import requantize_with_max_scale

from roll.utils.fp8 import (
    per_block_fp8_quant,
    per_block_fp8_quant_ascend,
    is_mxfp8_ascend,
    load_mxfp8_weight,
)
from roll.utils.logging import get_logger
from roll.third_party.vllm.parameter_utils import (
    parameter_from_data_and_source,
    parameter_from_subclass_attributes,
    replace_parameter_preserve_metadata,
    restore_layer_parameter_metadata,
)

logger = get_logger()

# ---------------------------------------------------------------------------
# Quant config setup (vLLM engine creation time)
# ---------------------------------------------------------------------------


def update_quant_config(config, vllm_config):
    """Enable ROLL's serialized FP8 path for explicit HF override configs.

    Standard FP8 block quantization is selected through
    ``hf_overrides.quantization_config``. MXFP8 uses a separate Ascend config
    path and is handled by ``update_mxfp8_quant_config``.
    """
    if not vllm_config.quant_config:
        return
    if not isinstance(vllm_config.quant_config, Fp8Config):
        return
    quantization_config = (config or {}).get("hf_overrides", {}).get("quantization_config")
    if not quantization_config:
        return

    assert quantization_config["quant_method"] == "fp8"
    assert vllm_config.quant_config.activation_scheme == "dynamic"
    vllm_config.quant_config.is_checkpoint_fp8_serialized = True
    vllm_config.quant_config.skip_process_weights_after_loading = True
    logger.info(
        f"Using custom vLLM quantization, block size {vllm_config.quant_config.weight_block_size}"
    )


def update_mxfp8_quant_config(vllm_config):
    """Configure vLLM config for Ascend MXFP8 quantization.

    The Ascend MXFP8 path uses ``vllm_ascend``'s ``AscendModelSlimConfig``
    rather than vLLM's native ``Fp8Config``.  We record that MXFP8 is active
    and force the checkpoint-serialized flag so ROLL's custom loaders are used.
    """
    if not vllm_config.quant_config:
        return
    if not is_mxfp8_ascend(vllm_config.quant_config):
        return

    # MXFP8 configs are always pre-serialized from ROLL's perspective.
    for name, value in (
        ("is_checkpoint_fp8_serialized", True),
        ("skip_process_weights_after_loading", False),
    ):
        try:
            setattr(vllm_config.quant_config, name, value)
        except Exception:
            logger.debug("Ascend MXFP8 quant_config does not allow setting %s", name)

    logger.info("Ascend MXFP8 quantization detected – using NPU dynamic MX quantization")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_quant_fn(layer: Module, quant_config):
    """Return the appropriate quantisation function for *layer*.

    Returns a callable ``(weight, block_size_or_dtype) -> (qweight, scale)``
    and the second argument to pass to it.
    """
    if is_mxfp8_ascend(quant_config):
        return per_block_fp8_quant_ascend, getattr(layer, "params_dtype", torch.bfloat16)
    return per_block_fp8_quant, layer.weight_block_size


def _linear_scale_param(layer: Module):
    if hasattr(layer, "weight_scale_inv"):
        return layer.weight_scale_inv
    return layer.weight_scale


def _replace_parameter_preserve_subclass(layer: Module, param_name: str, new_data: torch.Tensor | None) -> None:
    replace_parameter_preserve_metadata(layer, param_name, new_data)


def _restore_layer_param_subclass_attrs(layer: Module, old_params: dict[str, Parameter]) -> None:
    restore_layer_parameter_metadata(layer, old_params)


def _make_process_weights_after_loading_for_vllm20(original_fn):
    def _patched_process_weights_after_loading(self, layer: Module) -> None:
        old_params = dict(layer.named_parameters(recurse=False))
        with patch(
            "vllm.model_executor.layers.quantization.fp8.replace_parameter",
            _replace_parameter_preserve_subclass,
        ):
            original_fn(self, layer)
        _restore_layer_param_subclass_attrs(layer, old_params)

    return _patched_process_weights_after_loading


# ---------------------------------------------------------------------------
# Dense (Linear) layer patches
# ---------------------------------------------------------------------------


def _fp8_linear_weight_loader(
    layer: weakref.ReferenceType,
    original_weight_loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    layer = layer()
    assert param is layer.weight
    target_device = layer.weight.device
    is_mxfp8 = getattr(layer, "_roll_mxfp8", False)

    with target_device:
        weight = ModelWeightParameter(
            data=layer.weight.data if layer.weight_block_size else layer.weight.data.t(),
            input_dim=1,
            output_dim=0,
            weight_loader=original_weight_loader,
        )
        if loaded_weight.dtype == torch.float8_e4m3fn:
            # Already quantized – pass through unchanged.
            loaded_weight = loaded_weight.to(target_device)
            original_weight_loader(weight, loaded_weight, *args, **kwargs)
        elif is_mxfp8:
            # Ascend MXFP8 path: dynamic MX quant with torch_npu.
            loaded_weight = loaded_weight.to(target_device)
            # MXFP8 always uses "_scale" suffix on NPU.
            scale_param = _linear_scale_param(layer)
            weight_scale_inv = BlockQuantScaleParameter(
                data=scale_param.data,
                input_dim=1,
                output_dim=0,
                weight_loader=original_weight_loader,
            )
            load_mxfp8_weight(
                loaded_weight, layer, weight, weight_scale_inv, original_weight_loader, *args, **kwargs
            )
        elif layer.weight_block_size:
            # GPU blockwise FP8 path.
            loaded_weight = loaded_weight.to(target_device)
            scale_param = _linear_scale_param(layer)
            weight_scale_inv = BlockQuantScaleParameter(
                data=scale_param.data,
                input_dim=1,
                output_dim=0,
                weight_loader=original_weight_loader,
            )
            qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
            original_weight_loader(weight, qweight, *args, **kwargs)
            original_weight_loader(weight_scale_inv, scale, *args, **kwargs)
        else:
            # Per-tensor FP8 path.
            loaded_weight = loaded_weight.to(target_device)
            qweight, scale = per_tensor_fp8_quant(loaded_weight, scale=None)
            original_weight_loader(weight, qweight, *args, **kwargs)
            original_weight_loader(layer.per_shard_scale, scale, *args, **kwargs)
            layer.shard_loaded += 1
            if layer.shard_loaded == layer.shard_num:
                weight_scale, weight = requantize_with_max_scale(
                    weight=layer.weight.t(),
                    weight_scale=layer.per_shard_scale,
                    logical_widths=layer.logical_widths,
                )
                layer.weight.copy_(weight.t())
                layer.weight_scale.copy_(weight_scale)
                layer.shard_loaded = 0


def _fp8_linear_weight_scale_loader(
    layer: weakref.ReferenceType,
    original_weight_loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    layer = layer()
    scale_param = _linear_scale_param(layer)
    assert param is scale_param
    target_device = scale_param.device
    with target_device:
        weight_scale_inv = BlockQuantScaleParameter(
            data=scale_param.data,
            input_dim=1,
            output_dim=0,
            weight_loader=original_weight_loader,
        )
        original_weight_loader(weight_scale_inv, loaded_weight, *args, **kwargs)


def _fp8_linear_create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: List[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    _original_fp8_linear_create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, **extra_weight_attrs,
    )

    if not getattr(self.quant_config, "is_checkpoint_fp8_serialized", False):
        return

    assert self.quant_config.activation_scheme == "dynamic"
    assert not self.use_marlin  # not implemented yet, because lack weight loader for channelwise weight_scale

    # TODO support ROCM
    assert not current_platform.is_rocm()
    assert not current_platform.is_fp8_fnuz()

    # Check whether this is an Ascend MXFP8 model.
    layer._roll_mxfp8 = is_mxfp8_ascend(self.quant_config)

    # store essential config in layer for custom weight loader
    layer.weight_block_size = self.quant_config.weight_block_size

    weight_loader = layer.weight.weight_loader
    weight_loader = partial(_fp8_linear_weight_loader, weakref.ref(layer), weight_loader)  # patch weight loader
    layer.weight = (
        Parameter(layer.weight.data, requires_grad=False)
        if layer.weight_block_size
        else Parameter(layer.weight.data.t(), requires_grad=False)
    )
    layer.weight.weight_loader = weight_loader

    if layer.weight_block_size or layer._roll_mxfp8:
        weight_scale_inv_loader = layer.weight_scale_inv.weight_loader
        weight_scale_inv_loader = partial(_fp8_linear_weight_scale_loader, weakref.ref(layer), weight_scale_inv_loader)
        layer.weight_scale_inv = Parameter(layer.weight_scale_inv.data, requires_grad=False)
        layer.weight_scale_inv.weight_loader = weight_scale_inv_loader
    else:
        # does not support is_checkpoint_fp8_serialized now
        layer.per_shard_scale = layer.weight_scale
        layer.weight_scale = Parameter(
            torch.zeros(1, device=layer.weight.device, dtype=torch.float32), requires_grad=False
        )
        layer.shard_num = len(output_partition_sizes)
        layer.shard_loaded = 0

    if not hasattr(layer, "input_scale"):
        layer.register_parameter("input_scale", None)


_original_fp8_linear_create_weights = Fp8LinearMethod.create_weights
_original_fp8_linear_process_weights_after_loading = Fp8LinearMethod.process_weights_after_loading
Fp8LinearMethod.create_weights = _fp8_linear_create_weights


def _fp8_linear_process_weights_after_loading(self, layer: Module) -> None:
    """Process FP8 linear weights after loading (vLLM 0.10–0.19).

    Unified implementation that handles API differences across vLLM versions:
    - 0.10.x: ``_maybe_pad_weight`` + manual subclass param creation
    - 0.11.x: ``process_fp8_weight_block_strategy`` → ``weight_scale``,
      ``maybe_post_process_fp8_weight_block`` with optional cutlass arg,
      delete legacy ``weight_scale_inv``
    - 0.14.x: ``process_fp8_weight_block_strategy`` → ``weight_scale_inv``,
      ensure ``input_scale`` attribute exists
    """
    if not getattr(self, "block_quant", bool(getattr(self.quant_config, "weight_block_size", None))):
        return

    assert self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    scale_param = _linear_scale_param(layer)
    vllm_ver = Version(vllm.__version__)

    if vllm_ver >= Version("0.11.0"):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            maybe_post_process_fp8_weight_block,
            process_fp8_weight_block_strategy,
        )

        weight_data, scale_data = process_fp8_weight_block_strategy(layer.weight, scale_param)

        layer.weight = parameter_from_subclass_attributes(
            ModelWeightParameter(
                data=weight_data.data,
                output_dim=0, input_dim=1,
                weight_loader=layer.weight.weight_loader,
            )
        )

        if vllm_ver >= Version("0.14.0"):
            layer.weight_scale_inv = parameter_from_subclass_attributes(
                BlockQuantScaleParameter(
                    data=scale_data.data,
                    output_dim=0, input_dim=1,
                    weight_loader=scale_param.weight_loader,
                )
            )
            if not hasattr(layer, "input_scale"):
                layer.input_scale = None
        else:
            layer.weight_scale = parameter_from_subclass_attributes(
                BlockQuantScaleParameter(
                    data=scale_data.data,
                    output_dim=0, input_dim=1,
                    weight_loader=scale_param.weight_loader,
                )
            )
            if hasattr(layer, "weight_scale_inv"):
                del layer.weight_scale_inv

        if vllm_ver == Version("0.11.0"):
            maybe_post_process_fp8_weight_block(layer, self.cutlass_block_fp8_supported)
        else:
            maybe_post_process_fp8_weight_block(layer)
    else:
        # vLLM 0.10.x: pre-process_fp8_weight_block_strategy API.
        weight_data = self._maybe_pad_weight(layer.weight.data)
        layer.weight = parameter_from_subclass_attributes(
            ModelWeightParameter(
                data=weight_data,
                output_dim=0, input_dim=1,
                weight_loader=layer.weight.weight_loader,
            )
        )
        layer.weight_scale_inv = parameter_from_subclass_attributes(
            BlockQuantScaleParameter(
                data=scale_param.data,
                output_dim=0, input_dim=1,
                weight_loader=scale_param.weight_loader,
            )
        )


def _select_fp8_linear_process_weights_after_loading():
    """v0.20+ delegates to upstream; earlier versions use ROLL's unified impl."""
    if Version(vllm.__version__) >= Version("0.20.0"):
        return _make_process_weights_after_loading_for_vllm20(
            _original_fp8_linear_process_weights_after_loading
        )
    return _fp8_linear_process_weights_after_loading


Fp8LinearMethod.process_weights_after_loading = _select_fp8_linear_process_weights_after_loading()


# ---------------------------------------------------------------------------
# MoE layer patches
# ---------------------------------------------------------------------------


def _fp8_moe_w13_weight_loader(
    layer: weakref.ReferenceType,
    original_weight_loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    layer = layer()
    assert param is layer.w13_weight
    target_device = layer.w13_weight.device
    is_mxfp8 = getattr(layer, "_roll_mxfp8", False)

    with target_device:
        loaded_weight = loaded_weight.to(target_device)
        if loaded_weight.dtype == torch.float8_e4m3fn:
            original_weight_loader(layer.w13_weight, loaded_weight, *args, **kwargs)
        elif is_mxfp8:
            load_mxfp8_weight(
                loaded_weight, layer, layer.w13_weight, layer.w13_weight_scale_inv, original_weight_loader, *args, **kwargs
            )
        else:
            qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
            original_weight_loader(layer.w13_weight, qweight, *args, **kwargs)
            original_weight_loader(layer.w13_weight_scale_inv, scale, *args, **kwargs)


def _fp8_moe_w2_weight_loader(
    layer: weakref.ReferenceType,
    original_weight_loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
) -> None:
    layer = layer()
    assert param is layer.w2_weight
    target_device = layer.w2_weight.device
    is_mxfp8 = getattr(layer, "_roll_mxfp8", False)

    with target_device:
        loaded_weight = loaded_weight.to(target_device)
        if loaded_weight.dtype == torch.float8_e4m3fn:
            original_weight_loader(layer.w2_weight, loaded_weight, *args, **kwargs)
        elif is_mxfp8:
            load_mxfp8_weight(
                loaded_weight, layer, layer.w2_weight, layer.w2_weight_scale_inv, original_weight_loader, *args, **kwargs
            )
        else:
            qweight, scale = per_block_fp8_quant(loaded_weight, layer.weight_block_size)
            original_weight_loader(layer.w2_weight, qweight, *args, **kwargs)
            original_weight_loader(layer.w2_weight_scale_inv, scale, *args, **kwargs)


def _fp8_moe_create_weights(
    self,
    layer: Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    _original_fp8_moe_create_weights(
        self, layer, num_experts, hidden_size, intermediate_size_per_partition,
        params_dtype, **extra_weight_attrs,
    )

    if not getattr(self.quant_config, "is_checkpoint_fp8_serialized", False):
        return

    assert self.quant_config.activation_scheme == "dynamic"
    assert self.quant_config.weight_block_size is not None

    # TODO support ROCM
    # https://github.com/vllm-project/vllm/blob/v0.8.4/vllm/model_executor/layers/quantization/fp8.py#L655
    assert not current_platform.is_rocm()
    assert not current_platform.is_fp8_fnuz()
    assert current_platform.fp8_dtype() == torch.float8_e4m3fn

    self.rocm_aiter_moe_enabled = False  # set in original process_weights_after_loading

    # TODO: support ep
    assert layer.local_num_experts == num_experts

    if getattr(self, "_setup_kernel", None):
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

        unsupported_backends = [
            Fp8MoeBackend.AITER,
            Fp8MoeBackend.MARLIN,
            Fp8MoeBackend.FLASHINFER_CUTLASS,
            Fp8MoeBackend.FLASHINFER_TRTLLM,
            # TODO: support inflight fp8 quantization for DEEPGEMM and BATCHED_DEEPGEMM
            Fp8MoeBackend.DEEPGEMM,
            Fp8MoeBackend.BATCHED_DEEPGEMM,
        ]
        assert self.fp8_backend not in unsupported_backends

    # Check whether this is an Ascend MXFP8 model.
    layer._roll_mxfp8 = is_mxfp8_ascend(self.quant_config)

    # store essential config in layer for custom weight loader
    layer.weight_block_size = self.quant_config.weight_block_size

    w13_weight_loader = layer.w13_weight.weight_loader
    w13_weight_loader = partial(_fp8_moe_w13_weight_loader, weakref.ref(layer), w13_weight_loader)
    layer.w13_weight.weight_loader = w13_weight_loader
    set_weight_attrs(layer.w13_weight, {"roll_skip_patch_moe": True})  # TODO: remove once vllm 0.8.4 is deprecated

    w2_weight_loader = layer.w2_weight.weight_loader
    w2_weight_loader = partial(_fp8_moe_w2_weight_loader, weakref.ref(layer), w2_weight_loader)
    layer.w2_weight.weight_loader = w2_weight_loader
    set_weight_attrs(layer.w2_weight, {"roll_skip_patch_moe": True})  # TODO: remove once vllm 0.8.4 is deprecated

    # do not need patch weight loader of scale
    assert type(layer.w13_weight_scale_inv) == Parameter
    assert type(layer.w2_weight_scale_inv) == Parameter


_original_fp8_moe_create_weights = Fp8MoEMethod.create_weights
_original_fp8_moe_process_weights_after_loading = Fp8MoEMethod.process_weights_after_loading
Fp8MoEMethod.create_weights = _fp8_moe_create_weights


def _fp8_moe_process_weights_after_loading_vllm10(self, layer: Module) -> None:
    if hasattr(self, "block_quant") and not self.block_quant:
        return

    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled
    from vllm.model_executor.layers.quantization.fp8 import _is_col_major, _swap_w13_to_w31
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_col_major_tma_aligned_tensor,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import is_blackwell_deep_gemm_used

    self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_enabled:
        w13_weight = _swap_w13_to_w31(layer.w13_weight.data)
        w13_weight_scale_inv = _swap_w13_to_w31(layer.w13_weight_scale_inv.data)
        w2_weight = layer.w2_weight.data
        w2_weight_scale_inv = layer.w2_weight_scale_inv.data
    else:
        w13_weight = layer.w13_weight.data
        w13_weight_scale_inv = layer.w13_weight_scale_inv.data
        w2_weight = layer.w2_weight.data
        w2_weight_scale_inv = layer.w2_weight_scale_inv.data

    layer.w13_weight = parameter_from_data_and_source(w13_weight, layer.w13_weight)
    layer.w13_weight_scale_inv = parameter_from_data_and_source(
        w13_weight_scale_inv, layer.w13_weight_scale_inv
    )
    layer.w2_weight = parameter_from_data_and_source(w2_weight, layer.w2_weight)
    layer.w2_weight_scale_inv = parameter_from_data_and_source(
        w2_weight_scale_inv, layer.w2_weight_scale_inv
    )

    if self.allow_deep_gemm and not is_blackwell_deep_gemm_used():
        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous(),
                layer.w13_weight_scale_inv,
            )
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous(),
                layer.w2_weight_scale_inv,
            )

    if is_blackwell_deep_gemm_used():
        assert layer.weight_block_size is not None
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(layer.w13_weight.data, layer.w13_weight_scale_inv.data, block_sz)
        requant_weight_ue8m0_inplace(layer.w2_weight.data, layer.w2_weight_scale_inv.data, block_sz)

        if _is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous(),
                layer.w13_weight_scale_inv,
            )
        if _is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous(),
                layer.w2_weight_scale_inv,
            )


def _fp8_moe_process_weights_after_loading_vllm11(self, layer: Module) -> None:
    if hasattr(self, "block_quant") and not self.block_quant:
        return

    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        expert_weight_is_col_major,
        requant_weight_ue8m0_inplace,
    )
    from vllm.utils.deep_gemm import (
        get_col_major_tma_aligned_tensor,
        is_deep_gemm_e8m0_used,
    )

    try:
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import is_rocm_aiter_moe_enabled

        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
    except ImportError:
        from vllm._aiter_ops import rocm_aiter_ops

        self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    assert self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_backend is not None:
        layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        layer.w13_weight_scale_inv.data = swap_w13_to_w31(layer.w13_weight_scale_inv.data)

    if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv),
                layer.w13_weight_scale_inv,
            )
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv),
                layer.w2_weight_scale_inv,
            )

    if is_deep_gemm_e8m0_used():
        assert layer.weight_block_size is not None
        block_sz = tuple(layer.weight_block_size)
        requant_weight_ue8m0_inplace(layer.w13_weight.data, layer.w13_weight_scale_inv.data, block_sz)
        requant_weight_ue8m0_inplace(layer.w2_weight.data, layer.w2_weight_scale_inv.data, block_sz)

        if expert_weight_is_col_major(layer.w13_weight_scale_inv):
            layer.w13_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv),
                layer.w13_weight_scale_inv,
            )
        if expert_weight_is_col_major(layer.w2_weight_scale_inv):
            layer.w2_weight_scale_inv = parameter_from_data_and_source(
                get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv),
                layer.w2_weight_scale_inv,
            )


def _fp8_moe_process_weights_after_loading_vllm14(self, layer: Module) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format,
        make_fp8_moe_kernel,
    )

    w13 = layer.w13_weight
    w2 = layer.w2_weight
    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    w13_input_scale = layer.w13_input_scale
    w2_input_scale = layer.w2_input_scale

    w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=self.fp8_backend,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
    )

    layer.w13_weight = parameter_from_data_and_source(w13, layer.w13_weight)
    layer.w2_weight = parameter_from_data_and_source(w2, layer.w2_weight)
    layer.w13_weight_scale_inv = parameter_from_data_and_source(w13_scale, layer.w13_weight_scale_inv)
    layer.w2_weight_scale_inv = parameter_from_data_and_source(w2_scale, layer.w2_weight_scale_inv)

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config:
        assert self.experts_cls is not None
        sig = inspect.signature(make_fp8_moe_kernel)
        if "routing_tables" in sig.parameters:
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )
        else:
            self.kernel, self.use_inplace = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
            )


def _select_fp8_moe_process_weights_after_loading():
    vllm_ver = Version(vllm.__version__)
    if vllm_ver >= Version("0.20.0"):
        return _make_process_weights_after_loading_for_vllm20(
            _original_fp8_moe_process_weights_after_loading
        )
    if vllm_ver >= Version("0.14.0"):
        return _fp8_moe_process_weights_after_loading_vllm14
    if vllm_ver >= Version("0.11.0"):
        return _fp8_moe_process_weights_after_loading_vllm11
    return _fp8_moe_process_weights_after_loading_vllm10


Fp8MoEMethod.process_weights_after_loading = _select_fp8_moe_process_weights_after_loading()
