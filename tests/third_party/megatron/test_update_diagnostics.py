import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from roll.third_party.megatron.update_diagnostics import (
    build_masked_logprob_delta_statistics,
    build_tensor_update_statistics,
    flatten_fp8_update_diagnostics,
    iter_named_model_parameters,
    iter_optimizer_master_parameters,
    should_run_fp8_first_update_diagnostics,
    snapshot_named_tensors,
)


class FakeDistributedOptimizer:
    def __init__(self, bf16_master: torch.Tensor, fp32_parameter: torch.Tensor) -> None:
        self.shard_fp32_from_float16_groups = [[bf16_master]]
        self.shard_fp32_groups = [[fp32_parameter]]


class FakeFloat16Optimizer:
    def __init__(self, bf16_master: torch.Tensor, fp32_parameter: torch.Tensor) -> None:
        self.fp32_from_float16_groups = [[bf16_master]]
        self.fp32_from_fp32_groups = [[fp32_parameter]]


class FakeChainedOptimizer:
    def __init__(self, *optimizers: object) -> None:
        self.chained_optimizers = list(optimizers)


def test_iter_optimizer_master_parameters_supports_distributed_and_deduplicates() -> None:
    master = torch.tensor([1.0], dtype=torch.float32)
    fp32_parameter = torch.tensor([2.0], dtype=torch.float32)
    optimizer = FakeDistributedOptimizer(master, fp32_parameter)
    optimizer.shard_fp32_groups.append([master])

    named = list(iter_optimizer_master_parameters(optimizer))

    assert [id(tensor) for _name, tensor in named] == [id(master), id(fp32_parameter)]
    assert named[0][0].endswith("shard_fp32_from_float16_groups.0.0")
    assert named[1][0].endswith("shard_fp32_groups.0.0")


def test_iter_optimizer_master_parameters_supports_float16_chained_and_fp32() -> None:
    first = torch.tensor([1.0], dtype=torch.float32)
    second = torch.tensor([2.0], dtype=torch.float32)
    third = nn.Parameter(torch.tensor([3.0], dtype=torch.float32))
    mixed_precision = FakeFloat16Optimizer(first, second)
    fp32_optimizer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": [third]}]))

    named = list(iter_optimizer_master_parameters(FakeChainedOptimizer(mixed_precision, fp32_optimizer)))

    assert [id(tensor) for _name, tensor in named] == [id(first), id(second), id(third)]
    assert all(name.startswith("optimizer.optimizer") for name, _tensor in named)


def test_iter_optimizer_master_parameters_rejects_non_fp32_main_weight() -> None:
    optimizer = FakeDistributedOptimizer(
        torch.tensor([1.0], dtype=torch.bfloat16),
        torch.tensor([2.0], dtype=torch.float32),
    )

    with pytest.raises(TypeError, match="must be FP32"):
        list(iter_optimizer_master_parameters(optimizer))


def test_build_tensor_update_statistics_is_full_and_exact() -> None:
    current = torch.tensor([1.0, 2.5, 3.0, 8.0], dtype=torch.bfloat16)
    before = snapshot_named_tensors([("weight", current)])
    current.copy_(torch.tensor([1.0, 3.0, 3.0, 10.0], dtype=torch.bfloat16))

    statistics = build_tensor_update_statistics(before, [("weight", current)], learning_rate=0.5)

    assert statistics["statistics_mode"] == "full_no_sampling"
    assert statistics["numel"] == 4
    assert statistics["changed_numel"] == 2
    assert statistics["changed_fraction"] == pytest.approx(0.5)
    assert statistics["update_abs_max"] == pytest.approx(2.0)
    assert statistics["update_norm"] == pytest.approx(math.sqrt(4.25))
    assert statistics["top_tensors_local"][0]["name"] == "weight"


def test_build_tensor_update_statistics_preserves_small_fp32_update() -> None:
    current = torch.tensor([1.0e-4, 2.0e-4], dtype=torch.float32)
    before = snapshot_named_tensors([("master", current)])
    current[0] += 1.0e-8
    actual_delta = float(current[0].double() - before["master"][0].double())

    statistics = build_tensor_update_statistics(before, [("master", current)], learning_rate=1.0e-8)

    assert statistics["changed_numel"] == 1
    assert statistics["update_norm"] == pytest.approx(abs(actual_delta), rel=0, abs=1.0e-15)
    assert statistics["update_abs_max"] == pytest.approx(abs(actual_delta), rel=0, abs=1.0e-15)


def test_named_model_parameter_snapshot_is_independent_and_cpu_backed() -> None:
    model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    named_parameters = list(iter_named_model_parameters(model))
    snapshot = snapshot_named_tensors(named_parameters)
    original = snapshot["model0.weight"].clone()

    with torch.no_grad():
        model.weight.add_(1.0)

    assert snapshot["model0.weight"].device.type == "cpu"
    assert torch.equal(snapshot["model0.weight"], original)


def test_build_masked_logprob_delta_statistics_reports_ppo_movement() -> None:
    before = torch.zeros((1, 4), dtype=torch.float32)
    after = torch.tensor([[math.log(0.7), 0.0, math.log(1.3), 100.0]])
    mask = torch.tensor([[True, True, True, False]])

    statistics = build_masked_logprob_delta_statistics(after, before, mask)

    expected_rms = after[0, :3].double().square().mean().sqrt().item()
    assert statistics["token_count"] == 3
    assert statistics["before_logprob_mean"] == 0.0
    assert statistics["after_logprob_mean"] == pytest.approx(after[0, :3].mean().item())
    assert statistics["delta_rms"] == pytest.approx(expected_rms)
    assert statistics["half_delta_sq_mean"] == pytest.approx(0.5 * expected_rms**2)
    assert statistics["ratio_outside_0_8_1_2_fraction"] == pytest.approx(2 / 3)


def test_flatten_fp8_update_diagnostics_ignores_missing_rank_payloads() -> None:
    payloads = [[[{"rank": 0}]], None, [], {"rank": 2}]

    assert flatten_fp8_update_diagnostics(payloads) == [{"rank": 0}, {"rank": 2}]


@pytest.mark.parametrize(
    ("fp8", "global_step", "optimizer_batch_idx", "expected"),
    [
        ("e4m3", 0, 0, True),
        (None, 0, 0, False),
        ("e4m3", 1, 0, False),
        ("e4m3", 0, 1, False),
    ],
)
def test_should_run_fp8_first_update_diagnostics(
    fp8: object,
    global_step: int,
    optimizer_batch_idx: int,
    expected: bool,
) -> None:
    assert should_run_fp8_first_update_diagnostics(fp8, global_step, optimizer_batch_idx) is expected
