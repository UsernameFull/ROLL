import math
from typing import Dict, List

import torch


LOG_PREFIX = "[RLVR_LOGPROB_DIAG]"
PRIVATE_METRIC_PREFIX = "_rlvr_logprob_diag/"


def diagnostics_enabled(global_step: int, rank: int) -> bool:
    """Return whether the fixed first-step diagnostic should run."""
    return global_step == 0 and rank == 0


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


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf
