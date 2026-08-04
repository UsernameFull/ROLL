import math

import pytest
import torch
from torch import nn

from roll.third_party.megatron.update_diagnostics import (
    LayerActivationCapture,
    build_activation_delta_statistics,
    build_layerwise_activation_statistics,
    build_logprob_interpolation_statistics,
    build_masked_logprob_delta_statistics,
    build_parameter_update_statistics,
    copy_parameter_snapshot_to_models,
    extract_first_tensor,
    interpolate_parameter_snapshots_to_models,
    sample_tensor_values,
    snapshot_named_parameters,
)


class TransformerLayer(nn.Module):
    def __init__(self, layer_number: int, scale: float = 1.0):
        super().__init__()
        self.layer_number = layer_number
        self.scale = scale

    def forward(self, inputs):
        return inputs * self.scale, None


class TinyTransformer(nn.Module):
    def __init__(self, scales=(1.0,)):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerLayer(layer_number=index + 1, scale=scale) for index, scale in enumerate(scales)
        )

    def forward(self, inputs):
        hidden_states = inputs
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
        return hidden_states


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


def test_extract_first_tensor_supports_transformer_layer_outputs():
    expected = torch.ones(2)

    assert extract_first_tensor(expected) is expected
    assert extract_first_tensor((None, [expected])) is expected
    assert extract_first_tensor({"hidden_states": expected}) is None


def test_sample_tensor_values_is_deterministic_and_bounded():
    tensor = torch.arange(10, dtype=torch.bfloat16).reshape(2, 5)

    sampled = sample_tensor_values(tensor, max_samples=4)

    assert sampled.device.type == "cpu"
    assert sampled.dtype == torch.float32
    assert torch.equal(sampled, torch.tensor([0.0, 3.0, 6.0, 9.0]))
    with pytest.raises(ValueError, match="positive"):
        sample_tensor_values(tensor, max_samples=0)


def test_build_activation_delta_statistics_returns_relative_error_and_cosine():
    bf16 = torch.tensor([1.0, 2.0, 3.0])
    native = torch.tensor([2.0, 2.0, 4.0])

    stats = build_activation_delta_statistics(bf16, native)

    assert stats["sample_count"] == 3
    assert stats["finite_pair_count"] == 3
    assert stats["delta_rms"] == pytest.approx(math.sqrt(2 / 3))
    assert stats["relative_rms"] == pytest.approx(math.sqrt(2 / 14))
    assert stats["cosine_similarity"] == pytest.approx(18 / math.sqrt(14 * 24))
    assert stats["delta_abs_p95"] == pytest.approx(1.0)


def test_build_activation_delta_statistics_handles_zero_and_nonfinite_values():
    identical_zero = build_activation_delta_statistics(torch.zeros(2), torch.zeros(2))
    assert identical_zero["relative_rms"] == 0.0
    assert identical_zero["cosine_similarity"] == 1.0

    stats = build_activation_delta_statistics(
        torch.tensor([0.0, math.inf, math.nan]),
        torch.tensor([1.0, 0.0, math.nan]),
    )
    assert stats["sample_count"] == 3
    assert stats["finite_pair_count"] == 1
    assert stats["nonfinite_pair_count"] == 2
    assert stats["relative_rms"] is None
    assert stats["cosine_similarity"] is None


def test_layer_activation_capture_discovers_layers_samples_outputs_and_removes_hooks():
    model = TinyTransformer(scales=(2.0, 3.0))
    capture = LayerActivationCapture(model, max_samples_per_layer=4)

    with capture:
        model(torch.arange(6, dtype=torch.float32))

    records = capture.snapshot()
    assert list(records) == ["model0.layers.0", "model0.layers.1"]
    assert records["model0.layers.0"]["layer_number"] == 1
    assert records["model0.layers.0"]["call_count"] == 1
    assert records["model0.layers.0"]["original_numel"] == 6
    assert records["model0.layers.0"]["sample"].numel() <= 4
    assert not model.layers[0]._forward_hooks
    assert not model.layers[1]._forward_hooks


def test_layer_activation_capture_removes_hooks_after_exception():
    model = TinyTransformer()
    capture = LayerActivationCapture(model)

    with pytest.raises(RuntimeError, match="probe failed"):
        with capture:
            raise RuntimeError("probe failed")

    assert not model.layers[0]._forward_hooks


def test_build_layerwise_activation_statistics_orders_and_reports_unmatched_layers():
    bf16_model = TinyTransformer(scales=(1.0, 1.0))
    native_model = TinyTransformer(scales=(1.0, 2.0))
    bf16_capture = LayerActivationCapture(bf16_model, max_samples_per_layer=8)
    native_capture = LayerActivationCapture(native_model, max_samples_per_layer=8)
    inputs = torch.arange(4, dtype=torch.float32)
    with bf16_capture:
        bf16_model(inputs)
    with native_capture:
        native_model(inputs)

    native_records = native_capture.snapshot()
    unmatched = native_records.pop("model0.layers.1")
    native_records["model0.extra_layer"] = unmatched
    result = build_layerwise_activation_statistics(
        bf16_capture.snapshot(),
        native_records,
        max_samples_per_layer=8,
    )

    assert result["matched_layer_count"] == 1
    assert result["layers"][0]["key"] == "model0.layers.0"
    assert result["layers"][0]["delta_rms"] == 0.0
    assert result["unmatched_bf16_layers"] == ["model0.layers.1"]
    assert result["unmatched_native_layers"] == ["model0.extra_layer"]


def test_interpolate_and_restore_parameter_snapshots():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0, 2.0]]))
    before = snapshot_named_parameters(model)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[4.0, 6.0]]))
    after = snapshot_named_parameters(model)

    progress = interpolate_parameter_snapshots_to_models(
        before=before,
        after=after,
        models=model,
        alpha=0.25,
    )

    assert torch.equal(model.weight, torch.tensor([[1.0, 3.0]]))
    assert progress["requested_alpha"] == 0.25
    assert progress["realized_update_norm_ratio"] == pytest.approx(0.25)
    assert progress["realized_alpha_projection"] == pytest.approx(0.25)
    copy_parameter_snapshot_to_models(after, model)
    assert torch.equal(model.weight, torch.tensor([[4.0, 6.0]]))


def test_interpolate_parameter_snapshots_validates_before_mutating():
    model = nn.Linear(1, 1, bias=False)
    original = model.weight.detach().clone()
    snapshot = snapshot_named_parameters(model)

    with pytest.raises(ValueError, match="alpha"):
        interpolate_parameter_snapshots_to_models(
            before=snapshot,
            after=snapshot,
            models=model,
            alpha=math.nan,
        )
    assert torch.equal(model.weight, original)

    with pytest.raises(ValueError, match="names differ"):
        interpolate_parameter_snapshots_to_models(
            before={},
            after=snapshot,
            models=model,
            alpha=0.5,
        )
    assert torch.equal(model.weight, original)


def test_interpolate_parameter_snapshots_reports_bf16_rounding_progress():
    model = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1.0)
    before = snapshot_named_parameters(model)
    with torch.no_grad():
        model.weight.fill_(1.0078125)
    after = snapshot_named_parameters(model)

    progress = interpolate_parameter_snapshots_to_models(
        before=before,
        after=after,
        models=model,
        alpha=0.25,
    )

    assert model.weight.item() == 1.0
    assert progress["requested_alpha"] == 0.25
    assert progress["realized_update_norm_ratio"] == 0.0
    assert progress["realized_alpha_projection"] == 0.0


def test_build_logprob_interpolation_statistics_returns_path_deltas():
    mask = torch.tensor([[True, True]])
    alphas = [0.0, 0.5, 1.0]
    bf16_runs = [torch.zeros(1, 2), torch.full((1, 2), 0.1), torch.full((1, 2), 0.2)]
    native_runs = [torch.full((1, 2), 0.2), torch.full((1, 2), 0.4), torch.full((1, 2), 0.8)]

    points = build_logprob_interpolation_statistics(
        alphas=alphas,
        bf16_runs=bf16_runs,
        native_runs=native_runs,
        mask=mask,
    )

    assert [point["alpha"] for point in points] == alphas
    assert points[0]["bf16_from_alpha_0"]["delta_rms"] == 0.0
    assert points[1]["bf16_from_alpha_0"]["delta_rms"] == pytest.approx(0.1)
    assert points[2]["native_from_alpha_0"]["delta_rms"] == pytest.approx(0.6)
    assert points[2]["native_from_previous"]["delta_rms"] == pytest.approx(0.4)
    assert points[2]["bf16_vs_native"]["delta_mean"] == pytest.approx(-0.6)


def test_build_logprob_interpolation_statistics_skips_unavailable_pipeline_outputs():
    points = build_logprob_interpolation_statistics(
        alphas=[0.0, 1.0],
        bf16_runs=[None, None],
        native_runs=[torch.zeros(1, 1), torch.ones(1, 1)],
        mask=torch.ones(1, 1, dtype=torch.bool),
    )

    assert set(points[1]) == {"alpha", "native_from_alpha_0", "native_from_previous"}


@pytest.mark.parametrize(
    ("alphas", "bf16_runs", "native_runs", "message"),
    [
        ([], [], [], "At least one"),
        ([0.0, 1.0], [torch.zeros(1)], [torch.zeros(1)] * 2, "run counts"),
        ([0.25, 1.0], [torch.zeros(1)] * 2, [torch.zeros(1)] * 2, "first"),
        ([0.0, 0.0], [torch.zeros(1)] * 2, [torch.zeros(1)] * 2, "increasing"),
    ],
)
def test_build_logprob_interpolation_statistics_validates_path(alphas, bf16_runs, native_runs, message):
    with pytest.raises(ValueError, match=message):
        build_logprob_interpolation_statistics(
            alphas=alphas,
            bf16_runs=bf16_runs,
            native_runs=native_runs,
            mask=torch.ones(1, dtype=torch.bool),
        )
