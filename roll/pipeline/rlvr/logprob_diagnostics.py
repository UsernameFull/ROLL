import math
from typing import Dict, List

import torch


LOG_PREFIX = "[RLVR_LOGPROB_DIAG]"
INFERENCE_LOG_PREFIX = "[RLVR_INFER_DIAG]"
DIAGNOSTICS_META_KEY = "rlvr_logprob_diagnostics"
PRIVATE_METRIC_PREFIX = "_rlvr_logprob_diag/"


def diagnostics_enabled(global_step: int, rank: int) -> bool:
    """Return whether the fixed first-step diagnostic should run on this rank."""
    return global_step == 0


def flatten_diagnostic_payloads(payloads: List[object]) -> List[Dict[str, object]]:
    """Flatten per-rank diagnostic lists produced by ``DataProto.concat``."""
    flattened: List[Dict[str, object]] = []
    for payload in payloads:
        if isinstance(payload, list):
            flattened.extend(flatten_diagnostic_payloads(payload))
        elif isinstance(payload, dict):
            flattened.append(payload)
    return flattened


def build_ratio_statistics(ratio: torch.Tensor, response_mask: torch.Tensor) -> Dict[str, float]:
    """Build robust statistics over finite, response-token PPO ratios."""
    selected = ratio.detach()[response_mask.bool()].float()
    if selected.numel() == 0:
        return {}

    finite_mask = torch.isfinite(selected)
    finite = selected[finite_mask]
    stats = {
        "ratio_token_count": int(selected.numel()),
        "ratio_nonfinite_fraction": float((~finite_mask).float().mean().item()),
    }
    if finite.numel() == 0:
        return stats

    quantiles = torch.quantile(
        finite,
        torch.tensor([0.01, 0.5, 0.99], device=finite.device, dtype=finite.dtype),
    )
    stats.update(
        {
            "ratio_mean": float(finite.mean().item()),
            "ratio_p01": float(quantiles[0].item()),
            "ratio_median": float(quantiles[1].item()),
            "ratio_p99": float(quantiles[2].item()),
            "ratio_min": float(finite.min().item()),
            "ratio_max": float(finite.max().item()),
        }
    )
    return stats


def build_token_logprob_records(
    *,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    infer_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    max_samples: int = 2,
    max_tokens_per_sample: int = 32,
) -> List[Dict[str, float]]:
    """Return a bounded, JSON-serializable selected-token logprob sample."""
    records: List[Dict[str, float]] = []
    labels = input_ids[:, 1:]
    sample_count = min(max_samples, response_mask.shape[0])

    for sample_idx in range(sample_count):
        positions = torch.nonzero(response_mask[sample_idx].bool(), as_tuple=False).flatten()
        positions = positions[:max_tokens_per_sample]
        for response_offset, position_tensor in enumerate(positions):
            position = int(position_tensor.item())
            current = float(current_log_probs[sample_idx, position].detach().float().item())
            old = float(old_log_probs[sample_idx, position].detach().float().item())
            infer = float(infer_log_probs[sample_idx, position].detach().float().item())
            reference = float(ref_log_probs[sample_idx, position].detach().float().item())
            current_old_log_ratio = current - old
            old_infer_log_ratio = old - infer
            records.append(
                {
                    "sample_idx": sample_idx,
                    "response_offset": response_offset,
                    "sequence_position": position + 1,
                    "token_id": int(labels[sample_idx, position].detach().item()),
                    "current_logp": current,
                    "old_logp": old,
                    "infer_logp": infer,
                    "reference_logp": reference,
                    "current_old_log_ratio": current_old_log_ratio,
                    "current_old_ratio": _safe_exp(current_old_log_ratio),
                    "old_infer_log_ratio": old_infer_log_ratio,
                    "old_infer_ratio": _safe_exp(old_infer_log_ratio),
                }
            )
    return records


def build_inference_logprob_diagnostics(
    *,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    infer_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    max_records: int = 64,
) -> Dict[str, object]:
    """Compare vLLM decode, vLLM teacher forcing, and Megatron logprobs."""
    response = response_mask.detach().bool()
    old = old_log_probs.detach().float()
    infer = infer_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    labels = input_ids.detach()[:, 1:]

    if not (response.shape == old.shape == infer.shape == teacher.shape == labels.shape):
        raise ValueError(
            "inference diagnostic tensors must share [batch, sequence - 1] shape: "
            f"response={tuple(response.shape)}, old={tuple(old.shape)}, infer={tuple(infer.shape)}, "
            f"teacher={tuple(teacher.shape)}, labels={tuple(labels.shape)}"
        )

    response_count = int(response.sum().item())
    if response_count == 0:
        return {}

    teacher_scored = response & torch.isfinite(teacher)
    valid = teacher_scored & torch.isfinite(old) & torch.isfinite(infer)
    token_count = int(valid.sum().item())
    if token_count == 0:
        return {}

    records: List[Dict[str, object]] = []
    for sample_tensor, position_tensor in torch.nonzero(valid, as_tuple=False)[:max_records]:
        sample_idx = int(sample_tensor.item())
        position = int(position_tensor.item())
        old_value = float(old[sample_idx, position].item())
        infer_value = float(infer[sample_idx, position].item())
        teacher_value = float(teacher[sample_idx, position].item())
        records.append(
            {
                "sample_idx": sample_idx,
                "sequence_position": position + 1,
                "token_id": int(labels[sample_idx, position].item()),
                "old_logp": old_value,
                "decode_logp": infer_value,
                "teacher_logp": teacher_value,
                "decode_minus_teacher": infer_value - teacher_value,
                "old_minus_teacher": old_value - teacher_value,
                "old_minus_decode": old_value - infer_value,
            }
        )

    return {
        "token_count": token_count,
        "response_token_count": response_count,
        "scored_response_fraction": float(teacher_scored.sum().item() / response_count),
        "decode_minus_teacher": _build_logprob_delta_statistics(infer, teacher, valid),
        "old_minus_teacher": _build_logprob_delta_statistics(old, teacher, valid),
        "old_minus_decode": _build_logprob_delta_statistics(old, infer, valid),
        "records": records,
    }


def _build_logprob_delta_statistics(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Return robust statistics for ``left - right`` over ``mask``."""
    delta = (left - right)[mask]
    abs_delta = delta.abs()
    quantiles = torch.quantile(
        abs_delta,
        torch.tensor([0.5, 0.95, 0.99], device=delta.device, dtype=delta.dtype),
    )
    ratio = torch.exp(delta)
    ratio_outside = (ratio < 0.8) | (ratio > 1.2) | ~torch.isfinite(ratio)
    return {
        "delta_mean": float(delta.mean().item()),
        "delta_rms": float(delta.square().mean().sqrt().item()),
        "delta_abs_mean": float(abs_delta.mean().item()),
        "delta_abs_p50": float(quantiles[0].item()),
        "delta_abs_p95": float(quantiles[1].item()),
        "delta_abs_p99": float(quantiles[2].item()),
        "delta_abs_max": float(abs_delta.max().item()),
        "ratio_outside_0_8_1_2_fraction": float(ratio_outside.float().mean().item()),
    }


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf
