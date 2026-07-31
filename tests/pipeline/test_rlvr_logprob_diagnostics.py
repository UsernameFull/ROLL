import math

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.logprob_diagnostics import (
    DIAGNOSTICS_META_KEY,
    build_ratio_statistics,
    build_token_logprob_records,
    diagnostics_enabled,
)


def test_diagnostics_meta_info_keeps_rank_zero_payload():
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

    merged = DataProto.concat([rank_zero, rank_one])

    assert merged.meta_info[DIAGNOSTICS_META_KEY] == [{"event": "rank_zero"}]
    assert merged.meta_info["metrics"]["loss"] == [1.0, 2.0]


def test_diagnostics_enabled_only_for_first_step_rank_zero():
    assert diagnostics_enabled(global_step=0, rank=0)
    assert not diagnostics_enabled(global_step=1, rank=0)
    assert not diagnostics_enabled(global_step=0, rank=1)


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
