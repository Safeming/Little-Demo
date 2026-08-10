from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.a7c_r1_4vp_r4a import (
    reconstruct_renderer_sequence,
    signed_trajectory_component,
)
from utils.a7c_temporal_joint_projection import solve_temporal_joint_projection


_SCALE_COMPONENTS = (
    "trajectory_outer",
    "trajectory_boundary",
    "gain_outer",
    "gain_boundary",
    "target",
    "gate",
    "action",
)
_ALL_COMPONENTS = (*_SCALE_COMPONENTS, "projection")


def _positive_finite(value, name):
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _finite_tensor(value, name, *, ndim=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and nonempty")
    return value


def exact_projected_straight_through(
    raw_gates,
    runtime_mass,
    a5_weight,
    contract,
):
    raw = _finite_tensor(raw_gates, "raw_gates", ndim=2)
    if not isinstance(contract, Mapping):
        raise ValueError("projection contract must be a mapping")
    solved = solve_temporal_joint_projection(
        raw_gates=raw.detach().cpu().numpy().astype(np.float64),
        runtime_mass=np.asarray(runtime_mass, dtype=np.float64),
        a5_weight=np.asarray(a5_weight, dtype=np.float64),
        minimum_gate=float(contract["minimum_gate"]),
        maximum_gate=float(contract["maximum_gate"]),
        selection_threshold=float(contract["selection_threshold"]),
        proxy_target_response=float(contract["proxy_target_response"]),
        maximum_gate_jump=float(contract["maximum_projection_gate_jump"]),
        rho_tolerance=float(contract["lexicographic_tolerance"]),
        primal_tolerance=float(contract["solver_primal_tolerance"]),
        residual_tolerance=float(contract["solver_residual_tolerance"]),
    )
    exact = torch.as_tensor(solved["gates"], dtype=raw.dtype, device=raw.device)
    deployed = raw + (exact - raw).detach()
    if not torch.equal(deployed.detach(), exact):
        raise RuntimeError("straight-through forward must equal exact projected gates")
    return deployed, solved["certificate"]


def normalized_flicker(values, *, epsilon):
    vector = _finite_tensor(values, "flicker values", ndim=1)
    if vector.numel() < 2:
        raise ValueError("normalized flicker requires at least two values")
    epsilon = _positive_finite(epsilon, "epsilon")
    return torch.mean(torch.abs(torch.diff(vector))) / torch.clamp(
        torch.abs(vector.mean()), min=epsilon
    )


def renderer_gain(base, edited, *, epsilon):
    base = _finite_tensor(base, "base renderer sequence", ndim=1)
    edited = _finite_tensor(edited, "edited renderer sequence", ndim=1)
    if base.shape != edited.shape:
        raise ValueError("renderer gain sequences must align")
    denominator = torch.clamp(normalized_flicker(base, epsilon=epsilon), min=epsilon)
    return 1.0 - normalized_flicker(edited, epsilon=epsilon) / denominator


def _renderer_stream(streams, name):
    if not isinstance(streams, Mapping) or set(streams) != {
        "target",
        "outer",
        "boundary",
    }:
        raise ValueError("renderer streams must contain target, outer, and boundary")
    stream = streams[name]
    if not isinstance(stream, Mapping) or set(stream) != {"base", "point"}:
        raise ValueError(f"{name} renderer stream must contain base and point")
    return stream


def projection_aware_components(
    deployed_gates,
    raw_gates,
    teacher_gates,
    base_gates,
    streams,
    *,
    trajectory_delta,
    gain_delta,
    target_delta,
    gate_delta,
    temporal_delta,
    temporal_weight,
    projection_scale,
    epsilon,
):
    deployed = _finite_tensor(deployed_gates, "deployed gates", ndim=2)
    raw = _finite_tensor(raw_gates, "raw gates", ndim=2)
    teacher = _finite_tensor(teacher_gates, "teacher gates", ndim=2)
    base_gates = _finite_tensor(base_gates, "base gates", ndim=2)
    if not (raw.shape == deployed.shape == teacher.shape == base_gates.shape):
        raise ValueError("raw, deployed, teacher, and base gates must align")
    trajectory_delta = _positive_finite(trajectory_delta, "trajectory delta")
    gain_delta = _positive_finite(gain_delta, "gain delta")
    target_delta = _positive_finite(target_delta, "target delta")
    gate_delta = _positive_finite(gate_delta, "gate delta")
    temporal_delta = _positive_finite(temporal_delta, "temporal delta")
    temporal_weight = _positive_finite(temporal_weight, "temporal weight")
    projection_scale = _positive_finite(projection_scale, "projection scale")
    epsilon = _positive_finite(epsilon, "epsilon")

    components = {}
    for signal in ("outer", "boundary"):
        stream = _renderer_stream(streams, signal)
        components[f"trajectory_{signal}"] = signed_trajectory_component(
            stream["base"],
            stream["point"],
            deployed,
            teacher,
            delta=trajectory_delta,
            epsilon=epsilon,
        )
        candidate = reconstruct_renderer_sequence(
            stream["base"], stream["point"], deployed, epsilon=epsilon
        )
        reference = reconstruct_renderer_sequence(
            stream["base"], stream["point"], teacher, epsilon=epsilon
        )
        candidate_gain = renderer_gain(stream["base"], candidate, epsilon=epsilon)
        reference_gain = renderer_gain(stream["base"], reference, epsilon=epsilon)
        components[f"gain_{signal}"] = F.huber_loss(
            candidate_gain, reference_gain, reduction="mean", delta=gain_delta
        )

    target = _renderer_stream(streams, "target")
    candidate_target = reconstruct_renderer_sequence(
        target["base"], target["point"], deployed, epsilon=epsilon
    )
    reference_target = reconstruct_renderer_sequence(
        target["base"], target["point"], teacher, epsilon=epsilon
    )
    denominator = torch.clamp(torch.abs(target["base"]), min=epsilon)
    components["target"] = F.huber_loss(
        candidate_target / denominator,
        reference_target / denominator,
        reduction="mean",
        delta=target_delta,
    )
    components["gate"] = F.huber_loss(
        deployed, teacher, reduction="mean", delta=gate_delta
    ) + temporal_weight * F.huber_loss(
        torch.diff(deployed, dim=0),
        torch.diff(teacher, dim=0),
        reduction="mean",
        delta=temporal_delta,
    )

    candidate_action = (base_gates - deployed).reshape(-1)
    teacher_action = (base_gates - teacher).reshape(-1)
    teacher_norm = torch.linalg.vector_norm(teacher_action)
    if not torch.isfinite(teacher_norm) or float(teacher_norm.detach()) <= epsilon:
        raise ValueError("teacher action norm must be finite and greater than epsilon")
    candidate_norm = torch.linalg.vector_norm(candidate_action)
    cosine = torch.dot(candidate_action, teacher_action) / torch.clamp(
        candidate_norm * teacher_norm, min=epsilon
    )
    components["action"] = 1.0 - torch.clamp(cosine, min=-1.0, max=1.0)
    components["projection"] = torch.mean(
        torch.abs(raw - deployed.detach())
    ) / projection_scale

    if set(components) != set(_ALL_COMPONENTS):
        raise RuntimeError("projection-aware component registration is incomplete")
    if not all(value.numel() == 1 and torch.isfinite(value) for value in components.values()):
        raise ValueError("projection-aware components must be finite scalars")
    return components


def freeze_global_median_scales(segment_components, *, minimum):
    minimum = _positive_finite(minimum, "minimum scale")
    if not isinstance(segment_components, Sequence) or not segment_components:
        raise ValueError("scale components require at least one segment")
    rows = []
    for components in segment_components:
        if not isinstance(components, Mapping) or set(components) != set(_SCALE_COMPONENTS):
            raise ValueError("scale component keys must match the registered components")
        row = []
        for name in _SCALE_COMPONENTS:
            value = components[name]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(f"scale component {name} must be scalar")
                value = value.detach().cpu().item()
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"scale component {name} must be finite")
            row.append(value)
        rows.append(row)
    medians = np.median(np.asarray(rows, dtype=np.float64), axis=0)
    if not np.isfinite(medians).all() or np.any(medians <= minimum):
        raise ValueError("global median scales must be finite and positive")
    return {name: float(value) for name, value in zip(_SCALE_COMPONENTS, medians)}


def projection_aware_loss(components, scales, residual, *, residual_weight):
    if not isinstance(components, Mapping) or set(components) != set(_ALL_COMPONENTS):
        raise ValueError("loss component keys must match the registered components")
    if not isinstance(scales, Mapping) or set(scales) != set(_SCALE_COMPONENTS):
        raise ValueError("loss scale keys must match the registered components")
    if float(residual_weight) != 0.00001:
        raise ValueError("residual_weight must equal the frozen R4-B0 value 1e-5")
    residual = _finite_tensor(residual, "residual")
    normalized = {}
    for name in _SCALE_COMPONENTS:
        component = components[name]
        if not isinstance(component, torch.Tensor) or component.numel() != 1:
            raise ValueError(f"component {name} must be a scalar tensor")
        if not torch.isfinite(component):
            raise ValueError(f"component {name} must be finite")
        normalized[name] = component / _positive_finite(scales[name], f"scale {name}")
    projection = components["projection"]
    if not isinstance(projection, torch.Tensor) or projection.numel() != 1:
        raise ValueError("projection component must be a scalar tensor")
    if not torch.isfinite(projection):
        raise ValueError("projection component must be finite")

    renderer = torch.stack(
        [normalized[name] for name in _SCALE_COMPONENTS[:4]]
    ).mean()
    preservation = torch.stack(
        [normalized[name] for name in _SCALE_COMPONENTS[4:]] + [projection]
    ).mean()
    total = renderer + preservation + residual_weight * torch.mean(torch.abs(residual))
    return {
        "loss": total,
        "renderer_loss": renderer,
        "preservation_loss": preservation,
        **{f"normalized_{name}": value for name, value in normalized.items()},
        "normalized_projection": projection,
    }


def projection_diagnostics(raw, exact, *, changed_threshold):
    raw = np.asarray(raw, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    threshold = _positive_finite(changed_threshold, "changed threshold")
    if raw.shape != exact.shape or raw.size == 0 or raw.ndim != 2:
        raise ValueError("raw and exact projection arrays must align")
    if not np.isfinite(raw).all() or not np.isfinite(exact).all():
        raise ValueError("projection diagnostic arrays must be finite")
    displacement = np.abs(raw - exact)
    return {
        "raw_to_exact_mean_absolute_displacement": float(displacement.mean()),
        "raw_to_exact_maximum_absolute_displacement": float(displacement.max()),
        "raw_to_exact_changed_fraction": float(np.mean(displacement > threshold)),
    }


def evaluate_fit_projected_entry(summary, contract):
    if not isinstance(summary, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("fit entry requires summary and contract mappings")
    action = summary.get("projected_action_diagnostics", {})
    checks = {
        "fit_loss_improved": bool(summary.get("fit_loss_improved", False)),
        "fit_projected_teacher_mae": float(
            summary.get("fit_projected_teacher_mae", np.inf)
        ) <= float(contract["maximum_fit_projected_teacher_mae"]),
        "fit_outer_recovery": float(summary.get("fit_outer_recovery", -np.inf))
        >= float(contract["minimum_fit_outer_recovery"]),
        "fit_boundary_recovery": float(
            summary.get("fit_boundary_recovery", -np.inf)
        ) >= float(contract["minimum_fit_boundary_recovery"]),
        "fit_outer_positive_segment_fraction": float(
            summary.get("fit_outer_positive_segment_fraction", -np.inf)
        ) >= float(contract["minimum_fit_positive_segment_fraction"]),
        "fit_boundary_positive_segment_fraction": float(
            summary.get("fit_boundary_positive_segment_fraction", -np.inf)
        ) >= float(contract["minimum_fit_positive_segment_fraction"]),
        "fit_action_cosine": float(action.get("action_cosine", -np.inf))
        >= float(contract["minimum_fit_action_cosine"]),
        "fit_top_k_overlap": float(
            action.get("top_k_suppression_overlap", -np.inf)
        ) >= float(contract["minimum_fit_top_k_overlap"]),
        "fit_missed_suppression_fraction": float(
            action.get("missed_teacher_suppression_fraction", np.inf)
        ) <= float(contract["maximum_fit_missed_suppression_fraction"]),
        "raw_to_exact_mean_absolute_displacement": float(
            summary.get("raw_to_exact_mean_absolute_displacement", np.inf)
        ) <= float(contract["maximum_fit_raw_to_exact_mae"]),
        "raw_to_exact_changed_fraction": float(
            summary.get("raw_to_exact_changed_fraction", np.inf)
        ) <= float(contract["maximum_fit_projection_changed_fraction"]),
        "projection_certificates_passed": bool(
            summary.get("projection_certificates_passed", False)
        ),
        "held_teacher_values_accessed": not bool(
            summary.get("held_teacher_values_accessed", True)
        ),
        "held_renderer_values_accessed": not bool(
            summary.get("held_renderer_values_accessed", True)
        ),
    }
    failure_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failure_reasons,
        "status": (
            "FIT_PROJECTED_ENTRY_POSITIVE"
            if not failure_reasons
            else str(contract["fit_negative_status"])
        ),
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def _held_rows_inaccessible(teacher_values, renderer_streams, fit_mask):
    mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
    teacher = np.asarray(teacher_values)
    if teacher.ndim < 1 or teacher.shape[0] != mask.size or not np.any(mask):
        raise ValueError("teacher values and fit mask must align")
    if not np.isfinite(teacher[mask]).all() or np.isfinite(teacher[~mask]).any():
        return False
    if not isinstance(renderer_streams, Mapping) or set(renderer_streams) != {
        "target",
        "outer",
        "boundary",
    }:
        raise ValueError("observability renderer streams are incomplete")
    for signal in ("target", "outer", "boundary"):
        stream = renderer_streams[signal]
        if not isinstance(stream, Mapping) or set(stream) != {"base", "point"}:
            raise ValueError(f"observability {signal} stream is incomplete")
        for values in stream.values():
            array = np.asarray(values)
            if array.ndim < 1 or array.shape[0] != mask.size:
                raise ValueError("observability renderer rows must align with fit mask")
            if not np.isfinite(array[mask]).all() or np.isfinite(array[~mask]).any():
                return False
    return True


def _observability_payload_valid(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("observability closure must return a mapping")
    if set(payload) != {
        "loss",
        "components",
        "projection_certificates_passed",
    }:
        raise ValueError("observability closure returned unexpected keys")
    loss = payload["loss"]
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError("observability loss must be a scalar tensor")
    components = payload["components"]
    if not isinstance(components, Mapping) or not components:
        raise ValueError("observability components must be a nonempty mapping")
    for value in components.values():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise ValueError("observability components must be scalar tensors")
    return loss, components


def run_gradient_observability_preflight(
    model,
    deployed_loss_closure,
    *,
    teacher_values,
    renderer_streams,
    fit_mask,
    learning_rate,
    weight_decay,
    minimum_gradient_norm,
    step_count,
):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("observability model must be a torch module")
    if not callable(deployed_loss_closure):
        raise ValueError("observability loss closure must be callable")
    learning_rate = _positive_finite(learning_rate, "learning rate")
    weight_decay = float(weight_decay)
    if not np.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight decay must be finite and nonnegative")
    minimum_gradient_norm = _positive_finite(
        minimum_gradient_norm, "minimum gradient norm"
    )
    if int(step_count) != 1:
        raise ValueError("observability must use exactly one ephemeral step")

    source_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    held_safe = _held_rows_inaccessible(
        teacher_values, renderer_streams, fit_mask
    )
    clone = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(
        clone.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    optimizer.zero_grad(set_to_none=True)
    initial_payload = deployed_loss_closure(clone)
    initial_loss, initial_components = _observability_payload_valid(initial_payload)
    loss_finite = bool(torch.isfinite(initial_loss).all())
    components_finite = all(
        bool(torch.isfinite(value).all()) for value in initial_components.values()
    )
    certificates_passed = bool(
        initial_payload["projection_certificates_passed"]
    )

    gradients_finite = False
    gradient_norm = 0.0
    if loss_finite:
        initial_loss.backward()
        gradients = [
            parameter.grad
            for parameter in clone.parameters()
            if parameter.requires_grad
        ]
        gradients_finite = bool(gradients) and all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        )
        if gradients_finite:
            gradient_norm = float(
                torch.sqrt(
                    torch.stack(
                        [torch.sum(gradient.detach().square()) for gradient in gradients]
                    ).sum()
                ).cpu()
            )

    gradient_observable = gradients_finite and gradient_norm > minimum_gradient_norm
    final_loss_value = None
    final_certificates_passed = False
    final_components_finite = False
    step_decreased = False
    if (
        held_safe
        and loss_finite
        and components_finite
        and certificates_passed
        and gradient_observable
    ):
        optimizer.step()
        final_payload = deployed_loss_closure(clone)
        final_loss, final_components = _observability_payload_valid(final_payload)
        final_certificates_passed = bool(
            final_payload["projection_certificates_passed"]
        )
        final_components_finite = all(
            bool(torch.isfinite(value).all()) for value in final_components.values()
        )
        if bool(torch.isfinite(final_loss).all()):
            final_loss_value = float(final_loss.detach().cpu())
            step_decreased = final_loss_value < float(initial_loss.detach().cpu())

    source_unchanged = all(
        torch.equal(value, source_state[name])
        for name, value in model.state_dict().items()
    )
    checks = {
        "held_rows_inaccessible": held_safe,
        "initial_loss_finite": loss_finite,
        "components_finite": components_finite and (
            final_components_finite if final_loss_value is not None else True
        ),
        "projection_certificates_passed": certificates_passed and (
            final_certificates_passed if final_loss_value is not None else True
        ),
        "gradients_finite": gradients_finite,
        "gradient_observable": gradient_observable,
        "ephemeral_step_decreased_loss": step_decreased,
        "source_model_unchanged": source_unchanged,
    }
    failures = [name for name, passed in checks.items() if not passed]
    passed = not failures
    return {
        "verdict": (
            "FEATURE_OBSERVABILITY_POSITIVE"
            if passed
            else "FEATURE_OBSERVABILITY_NEGATIVE"
        ),
        "passed": passed,
        "initial_loss": (
            float(initial_loss.detach().cpu()) if loss_finite else None
        ),
        "final_loss": final_loss_value,
        "gradient_norm": gradient_norm,
        "checks": checks,
        "failure_reasons": failures,
        "held_teacher_values_accessed": not held_safe,
        "held_renderer_values_accessed": not held_safe,
    }
