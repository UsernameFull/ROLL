from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ASCEND_MXFP8_ONLINE_QUANTIZATION = "ascend_mxfp8"
ASCEND_MXFP8_QUANT_TYPE = "W8A8_MXFP8"
FLOAT_QUANT_TYPE = "FLOAT"

_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")
_DENSE_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj", "gate_up_proj")


def default_load_format_for_quantization(kwargs: Mapping[str, Any]) -> str:
    if kwargs.get("online_quantization"):
        return "dummy"
    return "auto" if kwargs.get("quantization") == "ascend" else "dummy"


def _set_projections(description: dict[str, Any], prefix: str, names: tuple[str, ...], value: str) -> None:
    description.update({f"{prefix}.{name}.weight": value for name in names})


def _is_moe_model(config: Any, options: Mapping[str, Any]) -> bool:
    if "is_moe" in options:
        return bool(options["is_moe"])
    model_type = str(getattr(config, "model_type", "")).lower()
    return "moe" in model_type or any(
        getattr(config, name, 0) not in (None, 0, 1)
        for name in ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size")
    )


def _is_moe_layer(config: Any, layer_idx: int) -> bool:
    if layer_idx < int(getattr(config, "first_k_dense_replace", 0) or 0):
        return False
    frequency = getattr(config, "moe_layer_freq", 1)
    if isinstance(frequency, (list, tuple)):
        return bool(frequency[layer_idx])
    return layer_idx % int(frequency or 1) == 0


def build_ascend_mxfp8_quant_description(
    hf_config: Any,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ModelSlim description needed by vLLM-Ascend for Qwen rollout."""
    options = dict(options or {})
    config = getattr(hf_config, "text_config", hf_config)
    num_layers = options.get("num_hidden_layers", getattr(config, "num_hidden_layers", None))
    if num_layers is None:
        raise ValueError("online_quantization=ascend_mxfp8 requires num_hidden_layers in the HF config.")

    description: dict[str, Any] = {
        "quant_method": "ascend",
        "group_size": int(options.get("group_size", 32)),
        "version": "1.0.0",
        "model_quant_type": ASCEND_MXFP8_QUANT_TYPE,
        "model.embed_tokens.weight": FLOAT_QUANT_TYPE,
        "lm_head.weight": FLOAT_QUANT_TYPE,
    }
    is_moe = _is_moe_model(config, options)
    for layer_idx in range(int(num_layers)):
        layer = f"model.layers.{layer_idx}"
        _set_projections(description, f"{layer}.self_attn", _ATTENTION_PROJECTIONS, ASCEND_MXFP8_QUANT_TYPE)
        mlp = f"{layer}.mlp"
        if is_moe and _is_moe_layer(config, layer_idx):
            _set_projections(description, mlp, ("gate", "router"), FLOAT_QUANT_TYPE)
            _set_projections(
                description,
                f"{mlp}.experts.0",
                ("gate_proj", "up_proj", "down_proj"),
                ASCEND_MXFP8_QUANT_TYPE,
            )
            description[f"{mlp}.experts.weight"] = ASCEND_MXFP8_QUANT_TYPE
            _set_projections(
                description,
                f"{mlp}.shared_expert",
                _DENSE_MLP_PROJECTIONS,
                ASCEND_MXFP8_QUANT_TYPE,
            )
        else:
            _set_projections(description, mlp, _DENSE_MLP_PROJECTIONS, ASCEND_MXFP8_QUANT_TYPE)

    extra = options.get("extra_quant_description", {})
    if not isinstance(extra, Mapping):
        raise TypeError("online_quantization_config.extra_quant_description must be a mapping.")
    description.update(extra)
    return description


def apply_online_quantization_config(kwargs: dict[str, Any], hf_config: Any | None = None) -> dict[str, Any] | None:
    online_quantization = kwargs.pop("online_quantization", None)
    options = kwargs.pop("online_quantization_config", None) or {}
    if not online_quantization:
        return None
    if online_quantization != ASCEND_MXFP8_ONLINE_QUANTIZATION:
        raise ValueError(f"Unsupported online_quantization={online_quantization!r}.")
    if not isinstance(options, Mapping):
        raise TypeError("online_quantization_config must be a mapping.")
    if kwargs.get("quantization") not in (None, "ascend"):
        raise ValueError(
            "online_quantization=ascend_mxfp8 requires strategy_config.quantization "
            "to be omitted or set to 'ascend'."
        )

    kwargs["quantization"] = "ascend"
    kwargs["load_format"] = "dummy"
    if hf_config is None:
        from transformers import AutoConfig

        if not kwargs.get("model"):
            raise ValueError("online_quantization=ascend_mxfp8 requires the vLLM model path.")
        hf_config = AutoConfig.from_pretrained(
            kwargs["model"],
            trust_remote_code=bool(kwargs.get("trust_remote_code", True)),
            revision=kwargs.get("revision"),
        )

    description = build_ascend_mxfp8_quant_description(hf_config, dict(options))
    hf_overrides = kwargs.get("hf_overrides") or {}
    if not isinstance(hf_overrides, Mapping):
        raise TypeError("hf_overrides must be a mapping for Ascend online MXFP8.")
    hf_overrides = dict(hf_overrides)
    user_description = hf_overrides.get("quantization_config", {})
    if not isinstance(user_description, Mapping):
        raise TypeError("hf_overrides.quantization_config must be a mapping.")
    description.update(user_description)
    if description.get("quant_method") != "ascend":
        raise ValueError("hf_overrides.quantization_config.quant_method must be 'ascend'.")
    hf_overrides["quantization_config"] = description
    kwargs["hf_overrides"] = hf_overrides
    return description
