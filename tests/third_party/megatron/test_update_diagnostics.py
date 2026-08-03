import math

import pytest
import torch
from torch import nn

from roll.third_party.megatron.update_diagnostics import (
    build_masked_logprob_delta_statistics,
    build_parameter_update_statistics,
    snapshot_named_parameters,
)


def test_snapshot_named_parameters_clones_multiple_models_to_cpu():
    first = nn.Linear(2, 1, bias=False)
    second = nn.Linear(1, 1, bias=False)
    second.weight.requires_grad_(False)

    snapshot = snapshot_named_parameters([first, second])
    original_first = snapshot["model0.weight"].clone()
    with torch.no_grad():
        first.weight.add_(1.0)

    assert set(snapshot) == {"model0.weight"}
    assert all(tensor.device.type == "cpu" for tensor in snapshot.values())
    assert torch.equal(snapshot["model0.weight"], original_first)


def test_build_parameter_update_statistics_returns_reducible_raw_values():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))
    snapshot = snapshot_named_parameters(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 6.0]]))

    stats = build_parameter_update_statistics(snapshot, model, learning_rate=0.5)

    assert stats["param_sq_sum"] == pytest.approx(25.0)
    assert stats["update_sq_sum"] == pytest.approx(4.0)
    assert stats["numel"] == 2
    assert stats["changed_numel"] == 1
    assert stats["update_abs_max"] == pytest.approx(2.0)
    assert stats["param_norm"] == pytest.approx(5.0)
    assert stats["update_norm"] == pytest.approx(2.0)
    assert stats["relative_update_norm"] == pytest.approx(0.4)
    assert stats["relative_update_per_lr"] == pytest.approx(0.8)
    assert stats["changed_fraction"] == pytest.approx(0.5)
    assert stats["tensor_count"] == 1
    assert stats["top_parameters"][0]["name"] == "model0.weight"


def test_build_parameter_update_statistics_handles_empty_and_nonfinite_parameters():
    empty_stats = build_parameter_update_statistics({}, nn.Identity())
    assert empty_stats["numel"] == 0
    assert empty_stats["relative_update_norm"] == 0.0
    assert empty_stats["top_parameters"] == []

    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, math.inf, math.nan]]))
    snapshot = snapshot_named_parameters(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, math.inf, math.nan]]))

    stats = build_parameter_update_statistics(snapshot, model)

    assert stats["param_sq_sum"] == pytest.approx(1.0)
    assert stats["update_sq_sum"] == pytest.approx(1.0)
    assert stats["changed_numel"] == 1
    assert stats["param_nonfinite_numel"] == 2
    assert stats["update_nonfinite_numel"] == 2


def test_build_parameter_update_statistics_limits_top_parameters_to_ten():
    model = nn.Module()
    for index in range(12):
        model.register_parameter(f"parameter_{index}", nn.Parameter(torch.tensor(float(index + 1))))
    snapshot = snapshot_named_parameters(model)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_(float(index + 1))

    stats = build_parameter_update_statistics(snapshot, model, top_k=12)

    assert len(stats["top_parameters"]) == 10
    assert stats["top_parameters"][0]["name"] == "model0.parameter_11"
    assert stats["update_abs_max"] == pytest.approx(12.0)


def test_build_parameter_update_statistics_allows_snapshot_dtype_difference():
    model = nn.Linear(1, 1, bias=False, dtype=torch.float32)
    snapshot = {name: value.double() for name, value in snapshot_named_parameters(model).items()}

    stats = build_parameter_update_statistics(snapshot, model)

    assert stats["update_sq_sum"] == 0.0


def test_build_masked_logprob_delta_statistics_returns_expected_distribution():
    deltas = torch.tensor([[math.log(0.7), 0.0, math.log(1.3), 1000.0]], dtype=torch.float64)
    before = torch.zeros_like(deltas)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    stats = build_masked_logprob_delta_statistics(deltas, before, mask)

    expected_rms = deltas[0, :3].square().mean().sqrt().item()
    assert stats["token_count"] == 3
    assert stats["finite_token_count"] == 3
    assert stats["delta_rms"] == pytest.approx(expected_rms)
    assert stats["ratio_outside_0_8_1_2_fraction"] == pytest.approx(2 / 3)
    assert stats["ratio_nonfinite_fraction"] == 0.0


def test_build_masked_logprob_delta_statistics_handles_empty_mask_and_nonfinite_values():
    after = torch.tensor([[0.1, math.inf, math.nan]])
    before = torch.tensor([[0.0, 0.0, math.nan]])

    empty_stats = build_masked_logprob_delta_statistics(after, before, torch.zeros_like(after, dtype=torch.bool))
    assert empty_stats["token_count"] == 0
    assert empty_stats["delta_rms"] is None

    stats = build_masked_logprob_delta_statistics(after, before, torch.ones_like(after, dtype=torch.bool))
    assert stats["token_count"] == 3
    assert stats["finite_token_count"] == 1
    assert stats["nonfinite_token_count"] == 2
    assert stats["nonfinite_fraction"] == pytest.approx(2 / 3)
    assert stats["delta_rms"] == pytest.approx(0.1)
    assert stats["ratio_nonfinite_fraction"] == pytest.approx(2 / 3)


def test_build_masked_logprob_delta_statistics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Mask shape"):
        build_masked_logprob_delta_statistics(torch.ones(2), torch.ones(2), torch.ones(1))
