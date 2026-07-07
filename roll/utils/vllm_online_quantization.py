from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


ASCEND_MXFP8_ONLINE_QUANTIZATION = "ascend_mxfp8"
ASCEND_MXFP8_QUANT_TYPE = "W8A8_MXFP8"
FLOAT_QUANT_TYPE = "FLOAT"
DEFAULT_MXFP8_GROUP_SIZE = 32


_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")
_DENSE_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
_MOE_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_ROUTER_PROJECTIONS = ("gate", "router")
_TEXT_CONFIG_ATTRS = ("text_config", "llm_config", "language_config")


def default_load_format_for_quantization(kwargs: Mapping[str, Any]) -> str:
    """Return ROLL's default vLLM load_format for quantized rollout configs."""
    online_quantization = kwargs.get("online_quantization")
    if online_quantization not in (None, False, ""):
        return "dummy"
    if kwargs.get("quantization") == "ascend":
        return "auto"
    return "dummy"


def _get_config_value(config: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _get_text_config(config: Any) -> Any:
    for attr in _TEXT_CONFIG_ATTRS:
        child_config = getattr(config, attr, None)
        if child_config is not None:
            return child_config
    return config


def _get_num_hidden_layers(config: Any, options: dict[str, Any]) -> int:
    if "num_hidden_layers" in options:
        return int(options["num_hidden_layers"])

    text_config = _get_text_config(config)
    num_hidden_layers = _get_config_value(
        text_config,
        ("num_hidden_layers", "num_layers", "n_layer", "num_decoder_layers"),
    )
    if num_hidden_layers is None:
        raise ValueError(
            "online_quantization=ascend_mxfp8 requires num_hidden_layers in the HF config. "
            "Set strategy_config.online_quantization_config.num_hidden_layers for custom models."
        )
    return int(num_hidden_layers)


def _is_moe_config(config: Any, options: dict[str, Any]) -> bool:
    if "is_moe" in options:
        return bool(options["is_moe"])

    text_config = _get_text_config(config)
    model_type = str(getattr(text_config, "model_type", getattr(config, "model_type", "")) or "")
    if "moe" in model_type.lower():
        return True

    for attr in ("num_experts", "num_local_experts", "n_routed_experts", "routed_scaling_factor"):
        value = getattr(text_config, attr, None)
        if isinstance(value, int) and value > 1:
            return True
        if value is not None and attr == "routed_scaling_factor":
            return True

    return getattr(text_config, "moe_intermediate_size", None) is not None


def _is_moe_layer(config: Any, layer_idx: int, is_moe_model: bool) -> bool:
    if not is_moe_model:
        return False

    text_config = _get_text_config(config)
    first_k_dense_replace = getattr(text_config, "first_k_dense_replace", None)
    if first_k_dense_replace is not None and layer_idx < int(first_k_dense_replace):
        return False

    moe_layer_freq = getattr(text_config, "moe_layer_freq", None)
    if isinstance(moe_layer_freq, int) and moe_layer_freq > 1:
        return layer_idx % moe_layer_freq == 0
    if isinstance(moe_layer_freq, (list, tuple)):
        return bool(moe_layer_freq[layer_idx])

    return True


def _set_projection(description: dict[str, Any], prefix: str, projections: tuple[str, ...], quant_type: str) -> None:
    for projection in projections:
        description[f"{prefix}.{projection}.weight"] = quant_type


def build_ascend_mxfp8_quant_description(
    hf_config: Any,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an in-memory ModelSlim-style description for online MXFP8 rollout.

    vLLM-Ascend normally expects a ``quant_model_description.json`` generated
    by ModelSlim.  For RL rollout we start from bf16 training weights, so ROLL
    synthesizes the structural description and quantizes weights as they are
    synchronized into the vLLM worker.
    """
    options = dict(options or {})
    group_size = int(options.get("group_size", DEFAULT_MXFP8_GROUP_SIZE))
    quant_type = str(options.get("quant_type", ASCEND_MXFP8_QUANT_TYPE))
    layer_prefix = str(options.get("layer_prefix", "model.layers"))
    num_hidden_layers = _get_num_hidden_layers(hf_config, options)
    is_moe_model = _is_moe_config(hf_config, options)

    description: dict[str, Any] = {
        "quant_method": "ascend",
        "group_size": group_size,
        "version": str(options.get("version", "1.0.0")),
        "model_quant_type": quant_type,
    }

    embedding_keys = options.get("embedding_keys", ("model.embed_tokens.weight", "lm_head.weight"))
    for key in embedding_keys:
        description[str(key)] = FLOAT_QUANT_TYPE

    for layer_idx in range(num_hidden_layers):
        base = f"{layer_prefix}.{layer_idx}"
        _set_projection(description, f"{base}.self_attn", _ATTENTION_PROJECTIONS, quant_type)

        mlp_prefix = f"{base}.mlp"
        if _is_moe_layer(hf_config, layer_idx, is_moe_model):
            _set_projection(description, mlp_prefix, _ROUTER_PROJECTIONS, FLOAT_QUANT_TYPE)
            for projection in _MOE_EXPERT_PROJECTIONS:
                description[f"{mlp_prefix}.experts.0.{projection}.weight"] = quant_type
            description[f"{mlp_prefix}.experts.weight"] = quant_type
            _set_projection(description, f"{mlp_prefix}.shared_expert", _DENSE_MLP_PROJECTIONS, quant_type)
        else:
            _set_projection(description, mlp_prefix, _DENSE_MLP_PROJECTIONS, quant_type)

    description.update(deepcopy(options.get("quant_description", {})))
    description.update(deepcopy(options.get("extra_quant_description", {})))
    return description


def _load_hf_config(model_name_or_path: str, kwargs: dict[str, Any]) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
        revision=kwargs.get("revision"),
    )


def apply_online_quantization_config(kwargs: dict[str, Any], hf_config: Any | None = None) -> dict[str, Any] | None:
    """Apply ROLL-only online quantization options to vLLM kwargs.

    Supported config:

    .. code-block:: yaml

        strategy_config:
          online_quantization: ascend_mxfp8
          online_quantization_config:
            group_size: 32
    """
    online_quantization = kwargs.pop("online_quantization", None)
    online_quantization_config = kwargs.pop("online_quantization_config", None) or {}
    if online_quantization in (None, False, ""):
        return None
    if online_quantization != ASCEND_MXFP8_ONLINE_QUANTIZATION:
        raise ValueError(
            f"Unsupported online_quantization={online_quantization!r}. "
            f"Only {ASCEND_MXFP8_ONLINE_QUANTIZATION!r} is currently supported."
        )
    if not isinstance(online_quantization_config, Mapping):
        raise TypeError("online_quantization_config must be a mapping when online_quantization is enabled.")
    online_quantization_config = dict(online_quantization_config)

    quantization = kwargs.get("quantization")
    if quantization not in (None, "ascend"):
        raise ValueError(
            "online_quantization=ascend_mxfp8 requires strategy_config.quantization to be omitted or set to 'ascend'."
        )
    kwargs["quantization"] = "ascend"
    kwargs["load_format"] = "dummy"

    model_name_or_path = kwargs.get("model")
    if hf_config is None:
        if not model_name_or_path:
            raise ValueError("online_quantization=ascend_mxfp8 requires the vLLM model path to be set.")
        hf_config = _load_hf_config(str(model_name_or_path), kwargs)

    generated_quant_config = build_ascend_mxfp8_quant_description(hf_config, online_quantization_config)

    hf_overrides = kwargs.get("hf_overrides") or {}
    if not isinstance(hf_overrides, Mapping):
        raise TypeError("online_quantization=ascend_mxfp8 requires hf_overrides to be a mapping when provided.")
    hf_overrides = deepcopy(hf_overrides)

    user_quant_config = hf_overrides.get("quantization_config") or {}
    if not isinstance(user_quant_config, Mapping):
        raise TypeError("hf_overrides.quantization_config must be a mapping for online_quantization=ascend_mxfp8.")
    user_quant_config = dict(user_quant_config)
    if user_quant_config.get("quant_method", "ascend") != "ascend":
        raise ValueError("hf_overrides.quantization_config.quant_method must be 'ascend' for Ascend online MXFP8.")

    generated_quant_config.update(deepcopy(user_quant_config))
    generated_quant_config["quant_method"] = "ascend"
    hf_overrides["quantization_config"] = generated_quant_config
    kwargs["hf_overrides"] = hf_overrides
    return generated_quant_config
