from types import SimpleNamespace

import pytest
import torch

from mcore_adapter import TrainingArguments
from roll.third_party.megatron.model_update import estimate_weight_size_for_model_update, normalize_weight_for_model_update


def test_training_args_syncs_fp8_aliases():
    args = TrainingArguments(output_dir="tmp", fp8="e4m3", fp8_recipe="mxfp8")

    assert args.fp8 == "e4m3"
    assert args.fp8_format == "e4m3"


def test_training_args_rejects_mismatched_fp8_aliases():
    with pytest.raises(ValueError, match="must match"):
        TrainingArguments(output_dir="tmp", fp8="e4m3", fp8_format="hybrid", fp8_recipe="mxfp8")


def test_training_args_rejects_fp8_param():
    with pytest.raises(ValueError, match="fp8_param is not supported"):
        TrainingArguments(output_dir="tmp", fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True)


def test_training_args_rejects_fp8_recipe_without_fp8_format():
    with pytest.raises(ValueError, match="fp8_recipe requires fp8"):
        TrainingArguments(output_dir="tmp", fp8_recipe="mxfp8")


def test_normalize_weight_for_model_update_unwraps_fp8_like_weight():
    data = torch.ones(2, 2, dtype=torch.float32)
    weight = SimpleNamespace(data=data, dtype=torch.bfloat16)

    normalized = normalize_weight_for_model_update(weight)

    assert torch.is_tensor(normalized)
    assert normalized.dtype == torch.bfloat16
    assert torch.equal(normalized, data.to(torch.bfloat16))


def test_normalize_weight_for_model_update_prefers_dequantize_hook():
    class Fp8LikeWeight:
        data = torch.ones(2, 2, dtype=torch.uint8)
        dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.uint8

        def dequantize(self):
            return torch.ones(2, 2, dtype=torch.bfloat16)

    normalized = normalize_weight_for_model_update(Fp8LikeWeight())

    assert normalized.dtype == torch.bfloat16
    assert torch.equal(normalized, torch.ones(2, 2, dtype=torch.bfloat16))


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch does not expose native float8 dtype")
def test_normalize_weight_for_model_update_casts_native_float8():
    weight = torch.ones(2, 2).to(torch.float8_e4m3fn)

    normalized = normalize_weight_for_model_update(weight)

    assert normalized.dtype == torch.bfloat16


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="torch does not expose native float8 dtype")
def test_estimate_weight_size_for_model_update_uses_normalized_dtype():
    weight = torch.ones(2, 2).to(torch.float8_e4m3fn)

    assert estimate_weight_size_for_model_update(weight) == 4 * 2
