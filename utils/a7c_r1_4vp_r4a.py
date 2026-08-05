from collections.abc import Mapping

import torch
import torch.nn.functional as F


_COMPONENT_NAMES = ("outer", "boundary", "target", "gate_aux")


def _require_positive_finite(value, name):
    scalar = float(value)
    if not torch.isfinite(torch.tensor(scalar)) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _require_finite_aligned(base, point, gates):
    if base.ndim != 1 or point.ndim != 2 or gates.ndim != 2:
        raise ValueError("base, point, and gates must align as [T], [T,N], [T,N]")
    if point.shape != gates.shape or point.shape[0] != base.shape[0]:
        raise ValueError("base, point, and gates must align on frames and carriers")
    if base.numel() < 2 or point.shape[1] < 1:
        raise ValueError("aligned renderer inputs require at least two frames and one carrier")
    if not all(torch.isfinite(value).all() for value in (base, point, gates)):
        raise ValueError("renderer inputs must be finite")


def reconstruct_renderer_sequence(base, point, gates, *, epsilon):
    epsilon = _require_positive_finite(epsilon, "epsilon")
    _require_finite_aligned(base, point, gates)
    result = base - point.sum(dim=1) + (point * gates).sum(dim=1)
    if not torch.isfinite(result).all() or torch.abs(result.mean()) <= epsilon:
        raise ValueError("renderer sequence mean must be finite and nonzero")
    return result


def mean_normalized_trajectory(values, *, epsilon):
    epsilon = _require_positive_finite(epsilon, "epsilon")
    if values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("trajectory must be a finite vector with at least two frames")
    mean = values.mean()
    if torch.abs(mean) <= epsilon:
        raise ValueError("trajectory mean must be nonzero")
    return values / torch.clamp(torch.abs(mean), min=epsilon)


def signed_trajectory_component(
    base, point, candidate_gates, teacher_gates, *, delta, epsilon
):
    delta = _require_positive_finite(delta, "delta")
    candidate = reconstruct_renderer_sequence(
        base, point, candidate_gates, epsilon=epsilon
    )
    teacher = reconstruct_renderer_sequence(
        base, point, teacher_gates, epsilon=epsilon
    )
    candidate_delta = torch.diff(
        mean_normalized_trajectory(candidate, epsilon=epsilon)
    )
    teacher_delta = torch.diff(
        mean_normalized_trajectory(teacher, epsilon=epsilon)
    )
    return F.huber_loss(
        candidate_delta, teacher_delta, reduction="mean", delta=delta
    )


def _target_response_component(
    base, point, candidate_gates, teacher_gates, *, delta, epsilon
):
    delta = _require_positive_finite(delta, "delta")
    candidate = reconstruct_renderer_sequence(
        base, point, candidate_gates, epsilon=epsilon
    )
    teacher = reconstruct_renderer_sequence(
        base, point, teacher_gates, epsilon=epsilon
    )
    denominator = torch.clamp(torch.abs(base), min=float(epsilon))
    return F.huber_loss(
        candidate / denominator,
        teacher / denominator,
        reduction="mean",
        delta=delta,
    )


def signed_renderer_trajectory_components(
    candidate_gates,
    teacher_gates,
    streams,
    *,
    renderer_delta,
    target_delta,
    gate_delta,
    gate_temporal_weight,
    epsilon,
):
    _require_finite_aligned(
        torch.ones(
            candidate_gates.shape[0],
            dtype=candidate_gates.dtype,
            device=candidate_gates.device,
        ),
        candidate_gates,
        teacher_gates,
    )
    gate_delta = _require_positive_finite(gate_delta, "gate_delta")
    gate_temporal_weight = _require_positive_finite(
        gate_temporal_weight, "gate_temporal_weight"
    )
    if not isinstance(streams, Mapping) or set(streams) != {
        "target",
        "outer",
        "boundary",
    }:
        raise ValueError("renderer streams must contain target, outer, and boundary")

    components = {}
    for signal in ("outer", "boundary"):
        stream = streams[signal]
        if not isinstance(stream, Mapping) or set(stream) != {"base", "point"}:
            raise ValueError(f"{signal} renderer stream must contain base and point")
        components[signal] = signed_trajectory_component(
            stream["base"],
            stream["point"],
            candidate_gates,
            teacher_gates,
            delta=renderer_delta,
            epsilon=epsilon,
        )

    target = streams["target"]
    if not isinstance(target, Mapping) or set(target) != {"base", "point"}:
        raise ValueError("target renderer stream must contain base and point")
    components["target"] = _target_response_component(
        target["base"],
        target["point"],
        candidate_gates,
        teacher_gates,
        delta=target_delta,
        epsilon=epsilon,
    )
    gate = F.huber_loss(
        candidate_gates,
        teacher_gates,
        reduction="mean",
        delta=gate_delta,
    )
    temporal = F.huber_loss(
        torch.diff(candidate_gates, dim=0),
        torch.diff(teacher_gates, dim=0),
        reduction="mean",
        delta=renderer_delta,
    )
    components["gate_aux"] = gate + gate_temporal_weight * temporal
    return components


def freeze_initial_scales(components, *, minimum):
    minimum = _require_positive_finite(minimum, "minimum scale")
    if not isinstance(components, Mapping) or set(components) != set(_COMPONENT_NAMES):
        raise ValueError("initial scale components must match the registered components")
    scales = {}
    for name in _COMPONENT_NAMES:
        value = components[name]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"initial scale {name} must be scalar")
            value = value.detach().cpu().item()
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or value < minimum:
            raise ValueError(f"initial scale {name} must be finite and positive")
        scales[name] = value
    return scales


def signed_renderer_trajectory_loss(
    components,
    scales,
    residual,
    *,
    outer_weight,
    boundary_weight,
    target_weight,
    gate_aux_weight,
    residual_weight,
):
    if not isinstance(components, Mapping) or set(components) != set(_COMPONENT_NAMES):
        raise ValueError("loss component keys must match the registered components")
    if not isinstance(scales, Mapping) or set(scales) != set(_COMPONENT_NAMES):
        raise ValueError("loss scale keys must match the registered components")
    registered = {
        "outer_weight": (outer_weight, 1.0),
        "boundary_weight": (boundary_weight, 1.0),
        "target_weight": (target_weight, 0.1),
        "gate_aux_weight": (gate_aux_weight, 0.1),
        "residual_weight": (residual_weight, 0.00001),
    }
    for name, (actual, expected) in registered.items():
        if float(actual) != expected:
            raise ValueError(f"{name} must equal the frozen R4-A value {expected}")
    if not isinstance(residual, torch.Tensor) or not torch.isfinite(residual).all():
        raise ValueError("residual must be a finite tensor")

    normalized = {}
    for name in _COMPONENT_NAMES:
        component = components[name]
        if not isinstance(component, torch.Tensor) or component.numel() != 1:
            raise ValueError(f"component {name} must be a scalar tensor")
        if not torch.isfinite(component).all():
            raise ValueError(f"component {name} must be finite")
        scale = _require_positive_finite(scales[name], f"scale {name}")
        normalized[name] = component / scale

    total = (
        outer_weight * normalized["outer"]
        + boundary_weight * normalized["boundary"]
        + target_weight * normalized["target"]
        + gate_aux_weight * normalized["gate_aux"]
        + residual_weight * torch.mean(torch.abs(residual))
    )
    return {"loss": total, **{f"normalized_{k}": v for k, v in normalized.items()}}
