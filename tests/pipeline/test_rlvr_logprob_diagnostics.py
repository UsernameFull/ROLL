import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.actor_worker import ActorWorker
from roll.pipeline.rlvr.logprob_diagnostics import (
    DIAGNOSTICS_META_KEY,
    PRIVATE_METRIC_PREFIX,
    build_inference_logprob_diagnostics,
    build_ratio_statistics,
    build_token_logprob_records,
    diagnostics_enabled,
    flatten_diagnostic_payloads,
)


def test_diagnostics_meta_info_aggregates_and_flattens_all_ranks():
    rank_zero = DataProto(
        meta_info={
            "metrics": {"loss": 1.0},
            DIAGNOSTICS_META_KEY: [{"event": "rank_zero"}],
        }
    )
    rank_one = DataProto(
        meta_info={
            "metrics": {"loss": 2.0},
            DIAGNOSTICS_META_KEY: [{"event": "rank_one"}],
        }
    )

    merged = DataProto.concat(
        [rank_zero, rank_one],
        global_keys={"metrics", DIAGNOSTICS_META_KEY},
    )

    diagnostics = flatten_diagnostic_payloads(merged.meta_info[DIAGNOSTICS_META_KEY])

    assert diagnostics == [{"event": "rank_zero"}, {"event": "rank_one"}]
    assert merged.meta_info["metrics"]["loss"] == [1.0, 2.0]


def test_diagnostics_enabled_for_all_ranks_only_on_first_step():
    assert diagnostics_enabled(global_step=0, rank=0)
    assert diagnostics_enabled(global_step=0, rank=1)
    assert not diagnostics_enabled(global_step=1, rank=0)
    assert not diagnostics_enabled(global_step=1, rank=1)


def test_build_ratio_statistics_masks_and_reports_quantiles():
    ratio = torch.tensor([[1.0, 2.0, 99.0], [3.0, float("inf"), 4.0]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    stats = build_ratio_statistics(ratio, response_mask)

    assert stats["ratio_token_count"] == 4
    assert stats["ratio_nonfinite_fraction"] == 0.25
    assert stats["ratio_mean"] == 2.0
    assert stats["ratio_median"] == 2.0
    assert stats["ratio_min"] == 1.0
    assert stats["ratio_max"] == 3.0


def test_first_optimizer_batch_ratio_statistics_are_one():
    ratio = torch.ones((2, 4))
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    stats = build_ratio_statistics(ratio, response_mask)

    assert stats["ratio_mean"] == 1.0
    assert stats["ratio_min"] == 1.0
    assert stats["ratio_median"] == 1.0
    assert stats["ratio_max"] == 1.0


def test_build_token_logprob_records_is_bounded_and_aligned():
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
        ]
    )
    response_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
    )
    current = torch.zeros((3, 4))
    old = torch.zeros((3, 4))
    infer = torch.full((3, 4), -1.0)
    reference = torch.full((3, 4), -2.0)

    records = build_token_logprob_records(
        input_ids=input_ids,
        response_mask=response_mask,
        current_log_probs=current,
        old_log_probs=old,
        infer_log_probs=infer,
        ref_log_probs=reference,
        max_samples=2,
        max_tokens_per_sample=2,
    )

    assert len(records) == 4
    assert [record["sample_idx"] for record in records] == [0, 0, 1, 1]
    assert [record["token_id"] for record in records] == [12, 13, 21, 22]
    assert records[0]["sequence_position"] == 2
    assert records[0]["current_old_ratio"] == 1.0
    assert math.isclose(records[0]["old_infer_ratio"], math.e)


def test_build_inference_logprob_diagnostics_uses_only_finite_scored_tokens():
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    response_mask = torch.tensor([[0, 1, 1, 1]])
    old = torch.tensor([[0.0, -1.0, -2.0, -3.0]])
    infer = torch.tensor([[0.0, -1.2, -2.1, -3.4]])
    teacher = torch.tensor([[float("nan"), -1.1, float("nan"), -3.2]])

    payload = build_inference_logprob_diagnostics(
        input_ids=input_ids,
        response_mask=response_mask,
        old_log_probs=old,
        infer_log_probs=infer,
        teacher_log_probs=teacher,
    )

    assert payload["token_count"] == 2
    assert payload["scored_response_fraction"] == pytest.approx(2 / 3)
    assert payload["decode_minus_teacher"]["delta_rms"] == pytest.approx(math.sqrt(0.025))
    assert payload["old_minus_teacher"]["delta_mean"] == pytest.approx(0.15)
    assert payload["old_minus_decode"]["delta_mean"] == pytest.approx(0.3)
    assert [record["token_id"] for record in payload["records"]] == [12, 14]
    assert [record["sequence_position"] for record in payload["records"]] == [2, 4]


def test_build_inference_logprob_diagnostics_returns_empty_without_scored_tokens():
    payload = build_inference_logprob_diagnostics(
        input_ids=torch.tensor([[10, 11, 12]]),
        response_mask=torch.tensor([[1, 1]]),
        old_log_probs=torch.zeros((1, 2)),
        infer_log_probs=torch.zeros((1, 2)),
        teacher_log_probs=torch.full((1, 2), float("nan")),
    )

    assert payload == {}


def test_actor_worker_consumes_first_update_probe_before_once():
    worker = ActorWorker.__new__(ActorWorker)
    probe = {
        "log_probs": torch.tensor([[-1.0, -2.0]]),
        "response_mask": torch.tensor([[True, False]]),
    }
    worker._first_update_probe_before = probe

    assert worker.consume_first_update_probe_before() is probe
    assert worker.consume_first_update_probe_before() is None


def test_actor_worker_merges_update_probe_into_optimizer_payload_and_removes_private_metrics():
    update_probe = {
        "parameter_update": {"update_norm": 0.25},
        "logprob_delta": {"delta_rms": 0.5},
    }
    pop_update_probe = MagicMock(return_value=update_probe)
    worker = ActorWorker.__new__(ActorWorker)
    worker.rank = 0
    worker.worker_config = SimpleNamespace(name="actor")
    worker.strategy = SimpleNamespace(
        scheduler=SimpleNamespace(get_last_lr=MagicMock(return_value=[1e-6])),
        pop_update_probe_diagnostics=pop_update_probe,
    )
    worker._driver_logprob_diagnostics = []
    worker.logger = MagicMock()
    private_ratio_key = f"{PRIVATE_METRIC_PREFIX}ratio_mean"
    metrics = {
        private_ratio_key: 1.25,
        "actor/ratio_min@min": 0.75,
        "actor/ratio_max@max": 1.5,
        "actor/approxkl@sum": 0.1,
        "actor/grad_norm": 2.0,
    }

    public_metrics = worker.log_optimizer_batch_diagnostics(global_step=0, batch_idx=0, metrics=metrics)

    pop_update_probe.assert_called_once_with()
    assert private_ratio_key not in public_metrics
    assert public_metrics["actor/approxkl@sum"] == 0.1
    assert len(worker._driver_logprob_diagnostics) == 1
    payload = worker._driver_logprob_diagnostics[0]
    assert payload["event"] == "optimizer_batch_complete"
    assert payload["ratio_mean"] == 1.25
    assert payload["lr_after_update"] == 1e-6
    assert payload["parameter_update"] == update_probe["parameter_update"]
    assert payload["logprob_delta"] == update_probe["logprob_delta"]


def test_actor_worker_skips_update_probe_outside_first_global_step():
    pop_update_probe = MagicMock(return_value={"parameter_update": {"update_norm": 0.25}})
    worker = ActorWorker.__new__(ActorWorker)
    worker.rank = 0
    worker.strategy = SimpleNamespace(pop_update_probe_diagnostics=pop_update_probe)
    worker._driver_logprob_diagnostics = []
    metrics = {f"{PRIVATE_METRIC_PREFIX}ratio_mean": 1.0, "actor/approxkl@sum": 0.0}

    returned = worker.log_optimizer_batch_diagnostics(global_step=1, batch_idx=0, metrics=metrics)

    assert returned is metrics
    pop_update_probe.assert_not_called()
    assert worker._driver_logprob_diagnostics == []
