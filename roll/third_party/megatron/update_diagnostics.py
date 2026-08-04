"""CPU-only helpers for diagnosing optimizer updates and log-probability drift."""

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Optional, TypeAlias

import torch
from torch import nn


ParameterSnapshot: TypeAlias = dict[str, torch.Tensor]
DiagnosticValue: TypeAlias = float | int | None | str | list[dict[str, object]]
DiagnosticStatistics: TypeAlias = dict[str, DiagnosticValue]


def snapshot_named_parameters(models: nn.Module | Iterable[nn.Module]) -> ParameterSnapshot:
    """Clone all named model parameters to CPU for a later optimizer-update comparison."""
    return {name: _cpu_clone(parameter) for name, parameter in _iter_named_parameters(models)}


def build_parameter_update_statistics(
    before: Mapping[str, torch.Tensor],
    models: nn.Module | Iterable[nn.Module],
    *,
    learning_rate: Optional[float] = None,
    top_k: int = 10,
) -> DiagnosticStatistics:
    """Compare current parameters with a CPU snapshot and return update statistics.

    The raw ``*_sq_sum`` and ``*_numel`` fields can be reduced across ranks by
    summation, while ``update_abs_max`` can be reduced with a maximum. Non-finite
    parameter pairs are counted and excluded from norms and maxima.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    if learning_rate is not None and learning_rate < 0:
        raise ValueError(f"learning_rate must be non-negative, got {learning_rate}")

    current_names: set[str] = set()
    per_parameter: list[dict[str, object]] = []
    totals = _empty_raw_update_statistics()

    for name, parameter in _iter_named_parameters(models):
        if name not in before:
            raise ValueError(f"Current model parameter is missing from snapshot: {name}")
        current_names.add(name)
        previous = before[name]
        if previous.device.type != "cpu":
            raise ValueError(f"Snapshot parameter must be on CPU: {name}")
        if previous.shape != parameter.shape:
            raise ValueError(
                f"Parameter shape changed for {name}: snapshot={tuple(previous.shape)}, current={tuple(parameter.shape)}"
            )

        current = _cpu_clone(parameter)
        raw = _parameter_raw_update_statistics(previous, current)
        for key in totals:
            if key == "update_abs_max":
                totals[key] = max(totals[key], raw[key])
            else:
                totals[key] += raw[key]
        per_parameter.append({"name": name, **raw, **_derive_update_statistics(raw, learning_rate)})

    missing_names = set(before) - current_names
    if missing_names:
        missing_preview = ", ".join(sorted(missing_names)[:3])
        raise ValueError(f"Snapshot parameters are missing from current model: {missing_preview}")

    top_parameters = sorted(
        per_parameter,
        key=lambda stats: (float(stats["update_sq_sum"]), float(stats["update_abs_max"])),
        reverse=True,
    )[: min(top_k, 10)]
    return {
        **totals,
        **_derive_update_statistics(totals, learning_rate),
        "tensor_count": len(per_parameter),
        "top_parameters": top_parameters,
    }


def build_masked_logprob_delta_statistics(
    after_log_probs: torch.Tensor,
    before_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> DiagnosticStatistics:
    """Return robust statistics for masked ``after - before`` log-probability deltas.

    Empty masks and masks containing only non-finite values produce counts plus
    ``None`` for unavailable distribution statistics instead of NaN values.
    """
    if after_log_probs.shape != before_log_probs.shape:
        raise ValueError(
            "Log-probability tensors must have identical shapes: "
            f"after={tuple(after_log_probs.shape)}, before={tuple(before_log_probs.shape)}"
        )
    if mask.shape != after_log_probs.shape:
        raise ValueError(
            f"Mask shape must match log probabilities: mask={tuple(mask.shape)}, log_probs={tuple(after_log_probs.shape)}"
        )

    selected_after = after_log_probs.detach()[mask.bool()].to(device="cpu", dtype=torch.float64)
    selected_before = before_log_probs.detach()[mask.bool()].to(device="cpu", dtype=torch.float64)
    token_count = selected_after.numel()
    if token_count == 0:
        return _empty_logprob_delta_statistics()

    finite_pair = torch.isfinite(selected_after) & torch.isfinite(selected_before)
    pair_delta = selected_after[finite_pair] - selected_before[finite_pair]
    finite_delta = pair_delta[torch.isfinite(pair_delta)]
    finite_token_count = finite_delta.numel()
    nonfinite_token_count = token_count - finite_token_count
    base: DiagnosticStatistics = {
        "token_count": token_count,
        "finite_token_count": finite_token_count,
        "nonfinite_token_count": nonfinite_token_count,
        "nonfinite_fraction": nonfinite_token_count / token_count,
    }
    if finite_token_count == 0:
        return {**base, **_unavailable_logprob_delta_statistics()}

    absolute_delta = finite_delta.abs()
    quantiles = torch.quantile(absolute_delta, torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64))
    ratio_finite = torch.isfinite(torch.exp(finite_delta))
    ratio_outside = (finite_delta < math.log(0.8)) | (finite_delta > math.log(1.2))
    return {
        **base,
        "delta_mean": float(finite_delta.mean().item()),
        "delta_rms": float(finite_delta.square().mean().sqrt().item()),
        "delta_abs_mean": float(absolute_delta.mean().item()),
        "delta_abs_p50": float(quantiles[0].item()),
        "delta_abs_p95": float(quantiles[1].item()),
        "delta_abs_p99": float(quantiles[2].item()),
        "delta_abs_max": float(absolute_delta.max().item()),
        "half_delta_sq_mean": float((0.5 * finite_delta.square().mean()).item()),
        "ratio_outside_0_8_1_2_fraction": float(ratio_outside.double().mean().item()),
        "ratio_nonfinite_fraction": float(
            (nonfinite_token_count + (~ratio_finite).double().sum().item()) / token_count
        ),
    }


def interpolate_parameter_snapshots_to_models(
    *,
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    models: nn.Module | Iterable[nn.Module],
    alpha: float,
) -> dict[str, float | int | None]:
    """Load ``before + alpha * (after - before)`` into model forward parameters.

    Interpolation is computed in FP32 on CPU and then rounded to each model
    parameter's dtype. Returned progress statistics expose any loss of requested
    interpolation resolution caused by BF16 parameter rounding.
    """
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha}")

    named_parameters = list(_iter_named_parameters(models))
    _validate_parameter_snapshot(before, named_parameters, label="before")
    _validate_parameter_snapshot(after, named_parameters, label="after")

    full_delta_sq_sum = 0.0
    realized_delta_sq_sum = 0.0
    realized_full_dot_sum = 0.0
    changed_numel = 0
    numel = 0
    with torch.no_grad():
        for name, parameter in named_parameters:
            before_float = before[name].to(dtype=torch.float32)
            after_float = after[name].to(dtype=torch.float32)
            full_delta = after_float - before_float
            if alpha == 0.0:
                interpolated = before[name]
            elif alpha == 1.0:
                interpolated = after[name]
            else:
                interpolated = torch.lerp(before_float, after_float, alpha).to(dtype=parameter.dtype)

            realized_float = interpolated.to(dtype=torch.float32)
            realized_delta = realized_float - before_float
            full_delta_sq_sum += float(full_delta.square().sum().item())
            realized_delta_sq_sum += float(realized_delta.square().sum().item())
            realized_full_dot_sum += float((realized_delta * full_delta).sum().item())
            changed_numel += int((realized_delta != 0).sum().item())
            numel += parameter.numel()
            parameter.copy_(interpolated.to(device=parameter.device, dtype=parameter.dtype))

    full_update_norm = math.sqrt(full_delta_sq_sum)
    realized_update_norm = math.sqrt(realized_delta_sq_sum)
    return {
        "requested_alpha": float(alpha),
        "full_update_norm": full_update_norm,
        "realized_update_norm": realized_update_norm,
        "realized_update_norm_ratio": _safe_divide(realized_update_norm, full_update_norm),
        "realized_alpha_projection": _safe_divide(realized_full_dot_sum, full_delta_sq_sum),
        "realized_changed_fraction": _safe_divide(float(changed_numel), float(numel)),
    }


def copy_parameter_snapshot_to_models(
    snapshot: Mapping[str, torch.Tensor], models: nn.Module | Iterable[nn.Module]
) -> None:
    """Restore a validated CPU parameter snapshot to model forward parameters."""
    named_parameters = list(_iter_named_parameters(models))
    _validate_parameter_snapshot(snapshot, named_parameters, label="restore")
    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(snapshot[name].to(device=parameter.device, dtype=parameter.dtype))


def build_logprob_interpolation_statistics(
    *,
    alphas: Sequence[float],
    bf16_runs: Sequence[torch.Tensor | None],
    native_runs: Sequence[torch.Tensor | None],
    mask: torch.Tensor,
) -> list[dict[str, object]]:
    """Compare BF16/native logprobs along one parameter-update interpolation path."""
    if not alphas:
        raise ValueError("At least one interpolation alpha is required")
    if len(bf16_runs) != len(alphas) or len(native_runs) != len(alphas):
        raise ValueError(
            "Interpolation run counts must match alphas: "
            f"alphas={len(alphas)}, bf16={len(bf16_runs)}, native={len(native_runs)}"
        )
    if alphas[0] != 0.0:
        raise ValueError(f"The first interpolation alpha must be 0.0, got {alphas[0]}")
    if any(not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError(f"Interpolation alphas must be finite and in [0, 1], got {list(alphas)}")
    if any(current <= previous for previous, current in zip(alphas, alphas[1:])):
        raise ValueError(f"Interpolation alphas must be strictly increasing, got {list(alphas)}")

    points: list[dict[str, object]] = []
    bf16_baseline = bf16_runs[0]
    native_baseline = native_runs[0]
    for index, alpha in enumerate(alphas):
        bf16_current = bf16_runs[index]
        native_current = native_runs[index]
        point: dict[str, object] = {"alpha": float(alpha)}
        if bf16_current is not None and bf16_baseline is not None:
            point["bf16_from_alpha_0"] = build_masked_logprob_delta_statistics(
                bf16_current, bf16_baseline, mask
            )
        if native_current is not None and native_baseline is not None:
            point["native_from_alpha_0"] = build_masked_logprob_delta_statistics(
                native_current, native_baseline, mask
            )
        if bf16_current is not None and native_current is not None:
            point["bf16_vs_native"] = build_masked_logprob_delta_statistics(
                bf16_current, native_current, mask
            )
        if index > 0:
            bf16_previous = bf16_runs[index - 1]
            native_previous = native_runs[index - 1]
            if bf16_current is not None and bf16_previous is not None:
                point["bf16_from_previous"] = build_masked_logprob_delta_statistics(
                    bf16_current, bf16_previous, mask
                )
            if native_current is not None and native_previous is not None:
                point["native_from_previous"] = build_masked_logprob_delta_statistics(
                    native_current, native_previous, mask
                )
        points.append(point)
    return points


def _iter_named_parameters(models: nn.Module | Iterable[nn.Module]) -> Iterator[tuple[str, nn.Parameter]]:
    model_iterable = [models] if isinstance(models, nn.Module) else models
    for model_index, model in enumerate(model_iterable):
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected torch.nn.Module, got {type(model)!r}")
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            yield f"model{model_index}.{name}", parameter


def _validate_parameter_snapshot(
    snapshot: Mapping[str, torch.Tensor],
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    label: str,
) -> None:
    current_names = {name for name, _ in named_parameters}
    snapshot_names = set(snapshot)
    if current_names != snapshot_names:
        missing = sorted(current_names - snapshot_names)[:3]
        extra = sorted(snapshot_names - current_names)[:3]
        raise ValueError(f"{label} snapshot names differ from model: missing={missing}, extra={extra}")
    for name, parameter in named_parameters:
        value = snapshot[name]
        if value.device.type != "cpu":
            raise ValueError(f"{label} snapshot parameter must be on CPU: {name}")
        if value.shape != parameter.shape:
            raise ValueError(
                f"{label} snapshot shape differs for {name}: "
                f"snapshot={tuple(value.shape)}, model={tuple(parameter.shape)}"
            )


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").clone()


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


def _parameter_raw_update_statistics(before: torch.Tensor, after: torch.Tensor) -> dict[str, int | float]:
    before_float = before.to(dtype=torch.float64)
    after_float = after.to(dtype=torch.float64)
    finite_pair = torch.isfinite(before_float) & torch.isfinite(after_float)
    finite_before = before_float[torch.isfinite(before_float)]
    pair_delta = after_float[finite_pair] - before_float[finite_pair]
    finite_delta = pair_delta[torch.isfinite(pair_delta)]
    update_abs_max = float(finite_delta.abs().max().item()) if finite_delta.numel() else 0.0
    return {
        "param_sq_sum": float(finite_before.square().sum().item()),
        "update_sq_sum": float(finite_delta.square().sum().item()),
        "numel": before.numel(),
        "changed_numel": int((finite_delta != 0).sum().item()),
        "update_abs_max": update_abs_max,
        "param_nonfinite_numel": int((~torch.isfinite(before_float)).sum().item()),
        "update_nonfinite_numel": before.numel() - finite_delta.numel(),
    }


def _derive_update_statistics(
    raw: Mapping[str, int | float], learning_rate: Optional[float]
) -> dict[str, Optional[float]]:
    param_norm = math.sqrt(float(raw["param_sq_sum"]))
    update_norm = math.sqrt(float(raw["update_sq_sum"]))
    relative_update_norm = _safe_divide(update_norm, param_norm)
    changed_fraction = _safe_divide(float(raw["changed_numel"]), float(raw["numel"]))
    relative_update_per_lr = None
    if learning_rate is not None and relative_update_norm is not None:
        relative_update_per_lr = _safe_divide(relative_update_norm, learning_rate)
    return {
        "param_norm": param_norm,
        "update_norm": update_norm,
        "relative_update_norm": relative_update_norm,
        "changed_fraction": changed_fraction,
        "relative_update_per_lr": relative_update_per_lr,
    }


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator != 0:
        return numerator / denominator
    if numerator == 0:
        return 0.0
    return None


def _empty_logprob_delta_statistics() -> DiagnosticStatistics:
    return {
        "token_count": 0,
        "finite_token_count": 0,
        "nonfinite_token_count": 0,
        "nonfinite_fraction": 0.0,
        **_unavailable_logprob_delta_statistics(),
    }


def _unavailable_logprob_delta_statistics() -> DiagnosticStatistics:
    return {
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
