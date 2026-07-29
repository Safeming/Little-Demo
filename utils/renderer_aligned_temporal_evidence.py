from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np
import torch


SIGNALS = ("target", "outer", "boundary")


def extract_renderer_region_contributions(
    *,
    rendered: torch.Tensor,
    attribution_colors: torch.Tensor,
    target_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    edit_sensitivity: torch.Tensor,
    retain_graph: bool = False,
) -> dict[str, torch.Tensor]:
    if rendered.ndim != 3 or rendered.shape[0] != 3:
        raise ValueError("rendered must have shape [3, H, W]")
    if attribution_colors.ndim != 2 or attribution_colors.shape[1] != 3:
        raise ValueError("attribution_colors must have shape [N, 3]")
    expected_mask_shape = tuple(rendered.shape[1:])
    masks = {
        "target": target_mask,
        "valid": valid_mask,
        "boundary": boundary_mask,
    }
    for name, mask in masks.items():
        if tuple(mask.shape) != expected_mask_shape:
            raise ValueError(f"{name}_mask must match rendered image dimensions")
    sensitivity = edit_sensitivity.reshape(-1)
    if sensitivity.shape[0] != attribution_colors.shape[0]:
        raise ValueError("edit_sensitivity must match Gaussian count")

    target = target_mask.to(device=rendered.device, dtype=rendered.dtype).clamp(0.0, 1.0)
    valid = valid_mask.to(device=rendered.device, dtype=rendered.dtype).clamp(0.0, 1.0)
    boundary = boundary_mask.to(device=rendered.device, dtype=rendered.dtype).clamp(0.0, 1.0)
    outer = (valid * (1.0 - target)).clamp(0.0, 1.0)
    grad_output = torch.stack((target, outer, boundary), dim=0)
    gradients = torch.autograd.grad(
        rendered,
        attribution_colors,
        grad_outputs=grad_output,
        retain_graph=retain_graph,
        create_graph=False,
    )[0]
    weighted = gradients.clamp_min(0.0) * sensitivity.to(
        device=gradients.device, dtype=gradients.dtype
    )[:, None].clamp_min(0.0)
    return {name: weighted[:, index] for index, name in enumerate(SIGNALS)}


def _initialize_state(state: MutableMapping, shape: tuple[int, int]) -> None:
    state["shape"] = shape
    state["last_frame_index"] = None
    state["visible_count"] = np.zeros(shape, dtype=np.int64)
    state["consecutive_visible_count"] = np.zeros(shape, dtype=np.int64)
    state["visibility_transition_count"] = np.zeros(shape, dtype=np.int64)
    state["visibility_pair_count"] = np.zeros(shape, dtype=np.int64)
    state["boundary_crossing_count"] = np.zeros(shape, dtype=np.int64)
    state["previous_visible"] = None
    state["previous_boundary_state"] = None
    for signal in SIGNALS:
        state[f"{signal}_mean"] = np.zeros(shape, dtype=np.float64)
        state[f"{signal}_m2"] = np.zeros(shape, dtype=np.float64)
        state[f"{signal}_flicker_sum"] = np.zeros(shape, dtype=np.float64)
        state[f"previous_{signal}"] = None


def accumulate_renderer_contribution_frame(
    state,
    *,
    frame_index: int,
    target_contribution,
    outer_contribution,
    boundary_contribution,
    visibility_epsilon: float = 1.0e-8,
) -> None:
    if not isinstance(state, MutableMapping):
        raise ValueError("state must be a mutable mapping")
    arrays = {
        "target": np.asarray(target_contribution, dtype=np.float64),
        "outer": np.asarray(outer_contribution, dtype=np.float64),
        "boundary": np.asarray(boundary_contribution, dtype=np.float64),
    }
    shape = arrays["target"].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays.values()):
        raise ValueError("renderer contributions must have matching shape [N, C]")
    if any(not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in arrays.values()):
        raise ValueError("renderer contributions must be finite and non-negative")
    if not state:
        _initialize_state(state, shape)
    if tuple(state.get("shape", ())) != shape:
        raise ValueError("frame shape does not match accumulator state shape")
    last_frame_index = state["last_frame_index"]
    if last_frame_index is not None and int(frame_index) <= int(last_frame_index):
        raise ValueError("frame_index must be strictly increasing")

    visible = (arrays["target"] + arrays["outer"]) > float(visibility_epsilon)
    boundary_state = np.zeros(shape, dtype=np.int8)
    boundary_state[visible & (arrays["target"] > arrays["outer"])] = 1
    boundary_state[visible & (arrays["target"] == arrays["outer"])] = 2
    boundary_state[visible & (arrays["outer"] > arrays["target"])] = 3
    previous_visible = state["previous_visible"]
    if previous_visible is not None:
        both_visible = previous_visible & visible
        state["consecutive_visible_count"] += both_visible
        state["visibility_transition_count"] += previous_visible != visible
        state["visibility_pair_count"] += 1
        state["boundary_crossing_count"][both_visible] += (
            boundary_state[both_visible] != state["previous_boundary_state"][both_visible]
        )
        for signal, values in arrays.items():
            previous = state[f"previous_{signal}"]
            denominator = 0.5 * (values + previous) + float(visibility_epsilon)
            relative = np.abs(values - previous) / denominator
            state[f"{signal}_flicker_sum"][both_visible] += relative[both_visible]

    count = state["visible_count"]
    new_count = count + visible
    for signal, values in arrays.items():
        delta = values - state[f"{signal}_mean"]
        state[f"{signal}_mean"][visible] += delta[visible] / new_count[visible]
        delta2 = values - state[f"{signal}_mean"]
        state[f"{signal}_m2"][visible] += delta[visible] * delta2[visible]
        state[f"previous_{signal}"] = values.copy()
    state["visible_count"] = new_count
    state["last_frame_index"] = int(frame_index)
    state["previous_visible"] = visible.copy()
    state["previous_boundary_state"] = boundary_state


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros(numerator.shape, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def finalize_renderer_contribution_evidence(state) -> dict[str, np.ndarray]:
    if not isinstance(state, MutableMapping) or "shape" not in state:
        raise ValueError("cannot finalize an empty renderer contribution state")
    visible = np.asarray(state["visible_count"], dtype=np.int64)
    pairs = np.asarray(state["consecutive_visible_count"], dtype=np.int64)
    means = {signal: np.asarray(state[f"{signal}_mean"], dtype=np.float64) for signal in SIGNALS}
    stds = {
        signal: np.sqrt(np.maximum(_safe_ratio(state[f"{signal}_m2"], visible), 0.0))
        for signal in SIGNALS
    }
    flicker = {
        signal: _safe_ratio(state[f"{signal}_flicker_sum"], pairs)
        for signal in SIGNALS
    }
    total_mean = means["target"] + means["outer"]
    scale = np.max(total_mean, axis=0, keepdims=True) if total_mean.size else np.ones((1, 0))
    scale = np.maximum(scale, 1.0e-8)
    target_weight = np.clip(means["target"] / scale, 0.0, 1.0)
    outer_weight = np.clip(means["outer"] / scale, 0.0, 1.0)
    boundary_weight = np.clip(means["boundary"] / scale, 0.0, 1.0)
    return {
        "temporal_visible_count": visible.astype(np.int32),
        "temporal_consecutive_visible_count": pairs.astype(np.int32),
        "temporal_target_ratio_mean": target_weight.astype(np.float32),
        "temporal_target_ratio_std": np.clip(stds["target"] / scale, 0.0, 1.0).astype(np.float32),
        "temporal_target_flicker": np.clip(flicker["target"], 0.0, 1.0).astype(np.float32),
        "temporal_outer_ratio_mean": outer_weight.astype(np.float32),
        "temporal_outer_ratio_std": np.clip(stds["outer"] / scale, 0.0, 1.0).astype(np.float32),
        "temporal_outer_flicker": np.clip(flicker["outer"], 0.0, 1.0).astype(np.float32),
        "temporal_boundary_crossing_rate": np.clip(
            _safe_ratio(state["boundary_crossing_count"], pairs), 0.0, 1.0
        ).astype(np.float32),
        "temporal_visibility_transition_rate": np.clip(
            _safe_ratio(state["visibility_transition_count"], state["visibility_pair_count"]),
            0.0,
            1.0,
        ).astype(np.float32),
        "renderer_target_contribution_mean_raw": means["target"].astype(np.float32),
        "renderer_outer_contribution_mean_raw": means["outer"].astype(np.float32),
        "renderer_boundary_contribution_mean_raw": means["boundary"].astype(np.float32),
        "renderer_target_contribution_weight": target_weight.astype(np.float32),
        "renderer_outer_contribution_weight": outer_weight.astype(np.float32),
        "renderer_boundary_contribution_weight": boundary_weight.astype(np.float32),
        "renderer_target_contribution_flicker": flicker["target"].astype(np.float32),
        "renderer_outer_contribution_flicker": flicker["outer"].astype(np.float32),
        "renderer_boundary_contribution_flicker": flicker["boundary"].astype(np.float32),
    }
