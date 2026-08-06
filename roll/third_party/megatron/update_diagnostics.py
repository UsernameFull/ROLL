"""Exact first-update diagnostics for Megatron FP8 training."""

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import Optional, TypeAlias

import torch
from torch import nn


FP8_UPDATE_DIAGNOSTICS_META_KEY = "fp8_first_update_diagnostics"
FP8_UPDATE_LOG_PREFIX = "[RLVR_FP8_UPDATE_DIAG]"
FP8_UPDATE_SWEEP_ALPHAS = (
    0.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0,
)

TensorSnapshot: TypeAlias = dict[str, torch.Tensor]
NamedTensors: TypeAlias = Iterable[tuple[str, torch.Tensor]]

_MASTER_GROUP_ATTRIBUTES = (
    "shard_fp32_from_float16_groups",
    "shard_fp32_groups",
    "fp32_from_float16_groups",
    "fp32_from_fp32_groups",
)
_STATISTICS_CHUNK_SIZE = 1_048_576


def should_run_fp8_first_update_diagnostics(fp8: object, global_step: int, optimizer_batch_idx: int) -> bool:
    """Return whether unconditional FP8 first-update diagnostics should run."""
    return bool(fp8) and global_step == 0 and optimizer_batch_idx == 0


def flatten_fp8_update_diagnostics(payloads: Iterable[object]) -> list[dict[str, object]]:
    """Flatten nested diagnostic payloads created by distributed concatenation."""
    flattened: list[dict[str, object]] = []
    for payload in payloads:
        if isinstance(payload, list):
            flattened.extend(flatten_fp8_update_diagnostics(payload))
        elif isinstance(payload, dict):
            flattened.append(payload)
    return flattened


def iter_named_model_parameters(models: nn.Module | Iterable[nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield trainable forward parameters with names stable across snapshots."""
    model_iterable = [models] if isinstance(models, nn.Module) else models
    for model_index, model in enumerate(model_iterable):
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)!r}")
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                yield f"model{model_index}.{name}", parameter


def iter_optimizer_master_parameters(optimizer: object) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield FP32 main parameters from supported Megatron optimizer wrappers."""
    seen_optimizers: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(current: object, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
        optimizer_id = id(current)
        if optimizer_id in seen_optimizers:
            return
        seen_optimizers.add(optimizer_id)

        chained = getattr(current, "chained_optimizers", None)
        if chained is not None:
            for optimizer_index, sub_optimizer in enumerate(chained):
                yield from visit(sub_optimizer, f"{prefix}.optimizer{optimizer_index}")
            return

        found_explicit_groups = False
        for attribute in _MASTER_GROUP_ATTRIBUTES:
            groups = getattr(current, attribute, None)
            if groups is None:
                continue
            found_explicit_groups = True
            for group_index, group in enumerate(groups):
                for parameter_index, parameter in enumerate(group):
                    if not isinstance(parameter, torch.Tensor) or id(parameter) in seen_tensors:
                        continue
                    if parameter.dtype != torch.float32:
                        raise TypeError(
                            f"Megatron master parameter must be FP32: {prefix}.{attribute}."
                            f"{group_index}.{parameter_index} has dtype={parameter.dtype}"
                        )
                    seen_tensors.add(id(parameter))
                    yield f"{prefix}.{attribute}.{group_index}.{parameter_index}", parameter
        if found_explicit_groups:
            return

        base_optimizer = getattr(current, "optimizer", None)
        param_groups = getattr(base_optimizer, "param_groups", None)
        if param_groups is None:
            param_groups = getattr(current, "param_groups", None)
        if param_groups is None:
            raise TypeError(f"Unsupported Megatron optimizer wrapper: {type(current)!r}")

        for group_index, group in enumerate(param_groups):
            for parameter_index, parameter in enumerate(group.get("params", ())):
                if not isinstance(parameter, torch.Tensor) or id(parameter) in seen_tensors:
                    continue
                if parameter.dtype != torch.float32:
                    raise TypeError(
                        f"Megatron master parameter must be FP32: {prefix}.param_groups."
                        f"{group_index}.{parameter_index} has dtype={parameter.dtype}"
                    )
                seen_tensors.add(id(parameter))
                yield f"{prefix}.param_groups.{group_index}.{parameter_index}", parameter

    yield from visit(optimizer, "optimizer")


def snapshot_named_tensors(named_tensors: NamedTensors) -> TensorSnapshot:
    """Clone every named tensor to CPU without changing its dtype."""
    snapshot: TensorSnapshot = {}
    for name, tensor in named_tensors:
        if name in snapshot:
            raise ValueError(f"Duplicate tensor name in diagnostic snapshot: {name}")
        snapshot[name] = tensor.detach().to(device="cpu").clone()
    return snapshot


@torch.no_grad()
def apply_scaled_tensor_update(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    current_tensors: NamedTensors,
    *,
    alpha: float,
) -> None:
    """Set tensors to ``before + alpha * (after - before)`` from CPU snapshots."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if set(before) != set(after):
        missing_after = sorted(set(before) - set(after))
        missing_before = sorted(set(after) - set(before))
        raise ValueError(
            "Snapshot names differ: "
            f"missing_after={missing_after[:3]}, missing_before={missing_before[:3]}"
        )

    current_items = list(current_tensors)
    current_names: set[str] = set()
    for name, current in current_items:
        if name in current_names:
            raise ValueError(f"Duplicate current tensor name: {name}")
        if name not in before:
            raise ValueError(f"Current tensor is missing from snapshots: {name}")
        current_names.add(name)

        before_tensor = before[name]
        after_tensor = after[name]
        if before_tensor.device.type != "cpu" or after_tensor.device.type != "cpu":
            raise ValueError(f"Diagnostic snapshots must reside on CPU: {name}")
        if before_tensor.shape != after_tensor.shape or before_tensor.shape != current.shape:
            raise ValueError(
                f"Tensor shape changed for {name}: before={tuple(before_tensor.shape)}, "
                f"after={tuple(after_tensor.shape)}, current={tuple(current.shape)}"
            )
        if before_tensor.dtype != after_tensor.dtype or before_tensor.dtype != current.dtype:
            raise TypeError(
                f"Tensor dtype changed for {name}: before={before_tensor.dtype}, "
                f"after={after_tensor.dtype}, current={current.dtype}"
            )
    missing_names = set(before) - current_names
    if missing_names:
        preview = ", ".join(sorted(missing_names)[:3])
        raise ValueError(f"Snapshot tensors are missing from current values: {preview}")

    for name, current in current_items:
        _apply_scaled_tensor_update(before[name], after[name], current, alpha=alpha)


def iter_leaf_optimizers(optimizer: object) -> Iterator[object]:
    """Yield non-chained Megatron optimizers in stable order."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is None:
        yield optimizer
        return
    for sub_optimizer in chained:
        yield from iter_leaf_optimizers(sub_optimizer)


def build_tensor_update_statistics(
    before: Mapping[str, torch.Tensor],
    current_tensors: NamedTensors,
    *,
    learning_rate: Optional[float] = None,
    top_k: int = 5,
) -> dict[str, object]:
    """Compute exact, non-sampled update statistics against a CPU snapshot."""
    if learning_rate is not None and learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")

    totals = _empty_raw_update_statistics()
    current_names: set[str] = set()
    per_tensor: list[dict[str, object]] = []
    for name, current in current_tensors:
        if name not in before:
            raise ValueError(f"Current tensor is missing from snapshot: {name}")
        if name in current_names:
            raise ValueError(f"Duplicate current tensor name: {name}")
        current_names.add(name)
        raw = _build_raw_update_statistics(before[name], current)
        for key in totals:
            if key == "update_abs_max":
                totals[key] = max(float(totals[key]), float(raw[key]))
            else:
                totals[key] += raw[key]
        per_tensor.append({"name": name, **raw, **_derive_update_statistics(raw, learning_rate)})

    missing_names = set(before) - current_names
    if missing_names:
        preview = ", ".join(sorted(missing_names)[:3])
        raise ValueError(f"Snapshot tensors are missing from current values: {preview}")

    top_tensors = sorted(
        per_tensor,
        key=lambda item: (float(item["update_abs_max"]), float(item["update_sq_sum"])),
        reverse=True,
    )[:top_k]
    return {
        **totals,
        **_derive_update_statistics(totals, learning_rate),
        "tensor_count": len(per_tensor),
        "top_tensors_local": top_tensors,
        "statistics_mode": "full_no_sampling",
        "local_accumulation_dtype": "float64",
    }


def build_masked_logprob_delta_statistics(
    after_log_probs: torch.Tensor,
    before_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, object]:
    """Return full masked statistics for ``after_log_probs - before_log_probs``."""
    if after_log_probs.shape != before_log_probs.shape:
        raise ValueError(
            "Logprob shapes differ: "
            f"after={tuple(after_log_probs.shape)}, before={tuple(before_log_probs.shape)}"
        )
    if mask.shape != after_log_probs.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match logprobs {tuple(after_log_probs.shape)}")

    selected_after = after_log_probs.detach()[mask.bool()].to(device="cpu", dtype=torch.float64)
    selected_before = before_log_probs.detach()[mask.bool()].to(device="cpu", dtype=torch.float64)
    token_count = selected_after.numel()
    if token_count == 0:
        return _empty_logprob_statistics()

    finite_pair = torch.isfinite(selected_after) & torch.isfinite(selected_before)
    paired_after = selected_after[finite_pair]
    paired_before = selected_before[finite_pair]
    delta = paired_after - paired_before
    finite_delta = torch.isfinite(delta)
    paired_after = paired_after[finite_delta]
    paired_before = paired_before[finite_delta]
    delta = delta[finite_delta]
    finite_token_count = delta.numel()
    nonfinite_token_count = token_count - finite_token_count
    base = {
        "token_count": token_count,
        "finite_token_count": finite_token_count,
        "nonfinite_token_count": nonfinite_token_count,
        "nonfinite_fraction": nonfinite_token_count / token_count,
    }
    if finite_token_count == 0:
        return {**base, **_unavailable_logprob_statistics()}

    absolute_delta = delta.abs()
    quantiles = torch.quantile(absolute_delta, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    ratio_is_finite = torch.isfinite(torch.exp(delta))
    ratio_outside = (delta < math.log(0.8)) | (delta > math.log(1.2))
    return {
        **base,
        "before_logprob_mean": float(paired_before.mean().item()),
        "before_logprob_rms": float(paired_before.square().mean().sqrt().item()),
        "before_logprob_min": float(paired_before.min().item()),
        "before_logprob_max": float(paired_before.max().item()),
        "after_logprob_mean": float(paired_after.mean().item()),
        "after_logprob_rms": float(paired_after.square().mean().sqrt().item()),
        "after_logprob_min": float(paired_after.min().item()),
        "after_logprob_max": float(paired_after.max().item()),
        "delta_mean": float(delta.mean().item()),
        "delta_rms": float(delta.square().mean().sqrt().item()),
        "delta_abs_mean": float(absolute_delta.mean().item()),
        "delta_abs_p50": float(quantiles[0].item()),
        "delta_abs_p95": float(quantiles[1].item()),
        "delta_abs_p99": float(quantiles[2].item()),
        "delta_abs_max": float(absolute_delta.max().item()),
        "half_delta_sq_mean": float((0.5 * delta.square().mean()).item()),
        "ratio_outside_0_8_1_2_fraction": float(ratio_outside.double().mean().item()),
        "ratio_nonfinite_fraction": float(
            (nonfinite_token_count + (~ratio_is_finite).double().sum().item()) / token_count
        ),
    }


@torch.no_grad()
def _apply_scaled_tensor_update(
    before: torch.Tensor,
    after: torch.Tensor,
    current: torch.Tensor,
    *,
    alpha: float,
) -> None:
    before_flat = before.reshape(-1)
    after_flat = after.reshape(-1)
    current_flat = current.detach().reshape(-1)
    source = before_flat if alpha == 0.0 else after_flat if alpha == 1.0 else None
    for start in range(0, current.numel(), _STATISTICS_CHUNK_SIZE):
        end = min(start + _STATISTICS_CHUNK_SIZE, current.numel())
        if source is not None:
            target = source[start:end].to(device=current.device)
        else:
            target = before_flat[start:end].clone()
            target.lerp_(after_flat[start:end], alpha)
            target = target.to(device=current.device)
        current_flat[start:end].copy_(target)


def _build_raw_update_statistics(before: torch.Tensor, current: torch.Tensor) -> dict[str, int | float]:
    if before.device.type != "cpu":
        raise ValueError("Diagnostic snapshots must reside on CPU")
    if before.shape != current.shape:
        raise ValueError(f"Tensor shape changed: before={tuple(before.shape)}, current={tuple(current.shape)}")

    totals = _empty_raw_update_statistics()
    before_flat = before.reshape(-1)
    current_flat = current.detach().reshape(-1)
    for start in range(0, before.numel(), _STATISTICS_CHUNK_SIZE):
        end = min(start + _STATISTICS_CHUNK_SIZE, before.numel())
        before_chunk = before_flat[start:end]
        current_chunk = current_flat[start:end].to(device="cpu")
        finite_before = torch.isfinite(before_chunk)
        finite_pair = finite_before & torch.isfinite(current_chunk)
        before_fp64 = before_chunk[finite_before].to(dtype=torch.float64)
        delta = (
            current_chunk[finite_pair].to(dtype=torch.float64)
            - before_chunk[finite_pair].to(dtype=torch.float64)
        )
        finite_delta = delta[torch.isfinite(delta)]

        totals["param_sq_sum"] += float(before_fp64.square().sum().item())
        totals["update_sq_sum"] += float(finite_delta.square().sum().item())
        totals["numel"] += before_chunk.numel()
        totals["changed_numel"] += int(((current_chunk != before_chunk) & finite_pair).sum().item())
        if finite_delta.numel():
            totals["update_abs_max"] = max(
                float(totals["update_abs_max"]), float(finite_delta.abs().max().item())
            )
        totals["param_nonfinite_numel"] += int((~finite_before).sum().item())
        totals["update_nonfinite_numel"] += before_chunk.numel() - finite_delta.numel()
    return totals


def _empty_raw_update_statistics() -> dict[str, int | float]:
    return {
        "param_sq_sum": 0.0,
        "update_sq_sum": 0.0,
        "numel": 0,
        "changed_numel": 0,
        "update_abs_max": 0.0,
        "param_nonfinite_numel": 0,
        "update_nonfinite_numel": 0,
    }


def _derive_update_statistics(
    raw: Mapping[str, int | float], learning_rate: Optional[float]
) -> dict[str, Optional[float]]:
    param_norm = math.sqrt(max(float(raw["param_sq_sum"]), 0.0))
    update_norm = math.sqrt(max(float(raw["update_sq_sum"]), 0.0))
    relative_update_norm = _safe_divide(update_norm, param_norm)
    return {
        "param_norm": param_norm,
        "update_norm": update_norm,
        "relative_update_norm": relative_update_norm,
        "changed_fraction": _safe_divide(float(raw["changed_numel"]), float(raw["numel"])),
        "relative_update_per_lr": (
            _safe_divide(relative_update_norm, learning_rate)
            if relative_update_norm is not None and learning_rate is not None
            else None
        ),
        "learning_rate_applied": learning_rate,
    }


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator != 0:
        return numerator / denominator
    if numerator == 0:
        return 0.0
    return None


def _empty_logprob_statistics() -> dict[str, object]:
    return {
        "token_count": 0,
        "finite_token_count": 0,
        "nonfinite_token_count": 0,
        "nonfinite_fraction": 0.0,
        **_unavailable_logprob_statistics(),
    }


def _unavailable_logprob_statistics() -> dict[str, None]:
    return {
        "before_logprob_mean": None,
        "before_logprob_rms": None,
        "before_logprob_min": None,
        "before_logprob_max": None,
        "after_logprob_mean": None,
        "after_logprob_rms": None,
        "after_logprob_min": None,
        "after_logprob_max": None,
        "delta_mean": None,
        "delta_rms": None,
        "delta_abs_mean": None,
        "delta_abs_p50": None,
        "delta_abs_p95": None,
        "delta_abs_p99": None,
        "delta_abs_max": None,
        "half_delta_sq_mean": None,
        "ratio_outside_0_8_1_2_fraction": None,
        "ratio_nonfinite_fraction": None,
    }
