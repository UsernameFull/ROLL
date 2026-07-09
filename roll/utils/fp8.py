from collections.abc import Callable, Mapping
from typing import Any, List, Optional

import torch

from roll.utils.logging import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Ascend NPU availability
# ---------------------------------------------------------------------------

_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Ascend MXFP8 detection
# ---------------------------------------------------------------------------


def _get_quant_config_value(quant_config: Any, key: str, default: Any = None) -> Any:
    if isinstance(quant_config, Mapping):
        return quant_config.get(key, default)
    return getattr(quant_config, key, default)


def _is_ascend_quant_description(quant_description: Any) -> bool:
    if isinstance(quant_description, Mapping):
        return str(quant_description.get("quant_method", "")).lower() == "ascend"
    return False


def is_mxfp8_ascend(quant_config) -> bool:
    """Detect whether the quant_config represents an Ascend MXFP8 configuration.

    Ascend NPUs use ``vllm_ascend.quantization.modelslim_config.AscendModelSlimConfig``
    with ``quant_method == "ascend"`` instead of vLLM's native ``Fp8Config``.

    Args:
        quant_config: The vLLM or SGLang quantization config object.

    Returns:
        True if this is an Ascend MXFP8 config.
    """
    if quant_config is None:
        return False

    if str(_get_quant_config_value(quant_config, "quant_method", "")).lower() == "ascend":
        return True

    if _is_ascend_quant_description(_get_quant_config_value(quant_config, "quant_description")):
        return True

    quantization_config = _get_quant_config_value(quant_config, "quantization_config")
    if _is_ascend_quant_description(quantization_config):
        return True

    try:
        from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig
    except ImportError:
        return False

    if not isinstance(quant_config, AscendModelSlimConfig):
        return False

    return _is_ascend_quant_description(getattr(quant_config, "quant_description", None))


# ---------------------------------------------------------------------------
# FP8 constants
# ---------------------------------------------------------------------------

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min


# ---------------------------------------------------------------------------
# Ascend MXFP8 dynamic quantization
# ---------------------------------------------------------------------------


def per_block_fp8_quant_ascend(
    param_value: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to MXFP8 format using Ascend NPU dynamic quantization.

    On Ascend NPUs, MXFP8 (Microscaling FP8) uses a different block-sharing
    scheme for scales compared to NVIDIA's blockwise FP8.  This function wraps
    ``torch_npu.npu_dynamic_mx_quant`` which performs the quantization with
    the correct MX format.

    Args:
        param_value: Input high-precision tensor (bf16/fp16).
        dtype: Cast target before quantization (default: ``torch.bfloat16``).

    Returns:
        Tuple of ``(quantized_weight, scale)``:
            - quantized_weight: FP8 quantized tensor.
            - scale: Per-block scale factors (inverse scale).
    """
    if not _NPU_AVAILABLE:
        raise RuntimeError(
            "Ascend NPU is required for MXFP8 quantization but torch_npu "
            "is not available.  Install torch_npu and ensure NPU devices "
            "are visible."
        )

    if dtype is None:
        dtype = torch.bfloat16

    param_lp, param_scale = torch_npu.npu_dynamic_mx_quant(
        param_value.to(dtype),
        axis=-1,
        dst_type=torch_npu.float8_e4m3fn,
    )
    param_scale = param_scale.flatten(-2, -1)

    return param_lp, param_scale


def load_mxfp8_weight(
    loaded_weight: torch.Tensor,
    layer: torch.nn.Module,
    weight_param: torch.Tensor,
    scale_param: torch.Tensor,
    weight_loader: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Quantize a BF16/FP16 weight to Ascend MXFP8 and load both qweight and scale.

    Shared helper for vLLM and SGLang FP8 monkey-patches.  Eliminates the
    duplication of the MXFP8 quantize-then-load pattern across linear, w13,
    and w2 weight loaders.

    Args:
        loaded_weight: High-precision weight loaded from checkpoint (bf16/fp16).
        layer: The target module (used to resolve ``params_dtype``).
        weight_param: Weight parameter to load the quantized tensor into.
        scale_param: Scale parameter to load the per-block scale into.
        weight_loader: The original framework weight loader callable.
        *args: Passed through to *weight_loader*.
        **kwargs: Passed through to *weight_loader*.
    """
    qweight, scale = per_block_fp8_quant_ascend(
        loaded_weight,
        dtype=getattr(layer, "params_dtype", torch.bfloat16),
    )
    weight_loader(weight_param, qweight, *args, **kwargs)
    weight_loader(scale_param, scale, *args, **kwargs)


# ---------------------------------------------------------------------------
# GPU blockwise FP8 quantization (existing)
# ---------------------------------------------------------------------------

# Block quant operator
#
# Borrow from transformers
#   https://huggingface.co/docs/transformers/en/quantization/finegrained_fp8
#   https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/quantizers/quantizer_finegrained_fp8.py#L83
#
# May use op from torchao:
#   https://github.com/pytorch/ao/pull/1668
#   https://github.com/volcengine/verl/pull/3084
def per_block_fp8_quant(param_value: torch.Tensor, weight_block_size: List[int]):
    """
    Quantizes weights to FP8 format using Block-wise quantization
    """
    block_size_m, block_size_n = weight_block_size

    rows, cols = param_value.shape[-2:]

    if rows % block_size_m != 0 or cols % block_size_n != 0:
        raise ValueError(
            f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
        )
    param_value_orig_shape = param_value.shape

    param_value = param_value.reshape(
        -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
    ).permute(0, 1, 3, 2, 4)

    # Calculate scaling factor for each block
    max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
    scale = FP8_MAX / max_abs
    scale_orig_shape = scale.shape
    scale = scale.unsqueeze(-1).unsqueeze(-1)

    # Quantize the weights
    quantized_param = torch.clamp(param_value * scale, min=FP8_MIN, max=FP8_MAX).to(FP8_DTYPE)

    quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
    # Reshape back to matrix shape
    quantized_param = quantized_param.reshape(param_value_orig_shape)

    # Construct the final, correct shape for the scales
    num_row_blocks = rows // block_size_m
    num_col_blocks = cols // block_size_n
    # This preserves original batch dimensions, if any
    final_scale_shape = (*param_value_orig_shape[:-2], num_row_blocks, num_col_blocks)
    # Reshape directly to the correct shape and take the reciprocal
    scale = scale.reshape(final_scale_shape).reciprocal()

    # TODO: DeepGemm scales need to be transposed and aligned (said in vLLM fp8.py)?

    # TODO: On B200, DeepGemm only support E8M0 scale

    return quantized_param, scale
