from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # Pure NumPy evidence tools do not require torch.
    torch = None
    nn = None


FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "camera_id",
        "camera_index",
        "frame_id",
        "frame_index",
        "subject_id",
        "gaussian_id",
        "image_name",
        "previous_frame",
    }
)


def validate_feature_schema(names: list[str]) -> tuple[str, ...]:
    schema = tuple(str(name) for name in names)
    if not schema or len(set(schema)) != len(schema):
        raise ValueError("feature schema must be non-empty and unique")
    forbidden = sorted(set(schema) & FORBIDDEN_FEATURE_NAMES)
    if forbidden:
        raise ValueError(f"forbidden feature fields: {forbidden}")
    return schema


def contiguous_block_ids(frame_indices, block_count: int) -> np.ndarray:
    frames = np.asarray(frame_indices).reshape(-1)
    blocks = int(block_count)
    if frames.size < blocks or blocks < 1:
        raise ValueError("block_count must be in [1, frame_count]")
    if np.any(np.diff(frames) < 0):
        raise ValueError("frame indices must be sorted")
    result = np.empty(frames.size, dtype=np.int16)
    for block, indices in enumerate(np.array_split(np.arange(frames.size), blocks)):
        result[indices] = block
    return result


def target_preserving_gate(point_gate, lower_support):
    gate = np.asarray(point_gate, dtype=np.float64)
    support = np.asarray(lower_support, dtype=np.float64)
    if np.any(~np.isfinite(gate)) or np.any(~np.isfinite(support)):
        raise ValueError("gate and support must be finite")
    if np.any(gate < 0.0) or np.any(gate > 1.0):
        raise ValueError("point gate must be in [0, 1]")
    if np.any(support < 0.0) or np.any(support > 1.0):
        raise ValueError("lower support must be in [0, 1]")
    return 1.0 - (1.0 - gate) * (1.0 - support)


def normalized_flicker(values) -> float:
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    if signal.size <= 1:
        return 0.0
    return float(
        np.mean(np.abs(np.diff(signal))) / max(abs(float(np.mean(signal))), 1.0e-12)
    )


def extract_runtime_probe_features(
    *,
    means3d,
    world_view_transform,
    camera_center,
    visibility,
    radii,
    opacity,
    a5_lower_weight,
    selected_lower,
) -> np.ndarray:
    points = np.asarray(means3d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("means3d must have shape [N, 3]")
    count = points.shape[0]
    transform = np.asarray(world_view_transform, dtype=np.float64)
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    if transform.shape != (4, 4):
        raise ValueError("world_view_transform must have shape [4, 4]")
    homogeneous = np.concatenate([points, np.ones((count, 1))], axis=1)
    camera = homogeneous @ transform
    denominator = np.where(np.abs(camera[:, 3]) > 1.0e-8, camera[:, 3], 1.0)
    camera_xyz = camera[:, :3] / denominator[:, None]
    depth = np.maximum(np.abs(camera_xyz[:, 2]), 1.0e-6)
    direction = center[None, :] - points
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-8)
    visible = np.asarray(visibility, dtype=np.float64).reshape(-1)
    radius = np.asarray(radii, dtype=np.float64).reshape(-1)
    alpha = np.asarray(opacity, dtype=np.float64).reshape(-1)
    lower = np.asarray(a5_lower_weight, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected_lower, dtype=np.float64).reshape(-1)
    if any(value.shape != (count,) for value in (visible, radius, alpha, lower, selected)):
        raise ValueError("per-carrier probe values must match means3d")
    features = np.stack(
        [
            visible,
            np.log1p(np.maximum(radius, 0.0)),
            camera_xyz[:, 0] / depth,
            camera_xyz[:, 1] / depth,
            np.log(depth),
            direction[:, 0],
            direction[:, 1],
            direction[:, 2],
            alpha,
            visible * np.maximum(alpha, 0.0) * np.square(np.maximum(radius, 0.0)),
            lower,
            selected,
        ],
        axis=1,
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("runtime probe features must be finite")
    return features


def fit_feature_normalization(features, *, sample_mask) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    selected = np.asarray(sample_mask, dtype=bool).reshape(-1)
    if values.ndim != 3 or selected.shape != (values.shape[0],) or not np.any(selected):
        raise ValueError("features and non-empty sample_mask must share sample dimension")
    fitting = values[selected].reshape(-1, values.shape[-1])
    mean = np.mean(fitting, axis=0)
    scale = np.std(fitting, axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


class BoundedCarrierMLP(nn.Module if nn is not None else object):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions,
        *,
        minimum_gate: float,
        maximum_gate: float = 1.0,
        initial_gate: float = 0.999,
    ):
        if nn is None:
            raise ImportError("torch is required for BoundedCarrierMLP")
        super().__init__()
        dimensions = [int(input_dimension), *map(int, hidden_dimensions), 1]
        layers = []
        for source, destination in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(source, destination), nn.SiLU()))
        final = nn.Linear(dimensions[-2], 1)
        nn.init.zeros_(final.weight)
        ratio = (float(initial_gate) - minimum_gate) / (maximum_gate - minimum_gate)
        if not 0.0 < ratio < 1.0:
            raise ValueError("initial_gate must be strictly inside gate bounds")
        nn.init.constant_(final.bias, float(np.log(ratio / (1.0 - ratio))))
        layers.append(final)
        self.network = nn.Sequential(*layers)
        self.minimum_gate = float(minimum_gate)
        self.maximum_gate = float(maximum_gate)

    def forward(self, features):
        unit = torch.sigmoid(self.network(features)).squeeze(-1)
        return self.minimum_gate + (self.maximum_gate - self.minimum_gate) * unit


def build_canary_splits(
    *,
    camera_index,
    frame_index,
    fit_camera_indices,
    audit_camera_indices,
    block_count: int,
) -> dict:
    cameras = np.asarray(camera_index).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    if cameras.shape != frames.shape:
        raise ValueError("camera and frame indices must match")
    fit_mask = np.isin(cameras, np.asarray(fit_camera_indices))
    audit_mask = np.isin(cameras, np.asarray(audit_camera_indices))
    if np.any(fit_mask & audit_mask) or not np.all(fit_mask | audit_mask):
        raise ValueError("fit and audit cameras must form a disjoint partition")
    held = [np.zeros(cameras.size, dtype=bool) for _ in range(int(block_count))]
    for camera in fit_camera_indices:
        indices = np.flatnonzero(cameras == camera)
        order = np.argsort(frames[indices], kind="stable")
        sorted_indices = indices[order]
        block_ids = contiguous_block_ids(frames[sorted_indices], int(block_count))
        for block in range(int(block_count)):
            held[block][sorted_indices[block_ids == block]] = True
    return {"fit_mask": fit_mask, "audit_mask": audit_mask, "held_block_masks": held}


def evaluate_contribution_predictions(
    *, target, outer, boundary, point_target, point_outer, point_boundary, gates
) -> dict:
    base = {
        "target": np.asarray(target, dtype=np.float64).reshape(-1),
        "outer": np.asarray(outer, dtype=np.float64).reshape(-1),
        "boundary": np.asarray(boundary, dtype=np.float64).reshape(-1),
    }
    points = {
        "target": np.asarray(point_target, dtype=np.float64),
        "outer": np.asarray(point_outer, dtype=np.float64),
        "boundary": np.asarray(point_boundary, dtype=np.float64),
    }
    gate = np.asarray(gates, dtype=np.float64)
    if any(value.shape != gate.shape for value in points.values()):
        raise ValueError("point contributions and gates must match")
    candidate = {
        name: base[name] - np.sum(points[name], axis=1) + np.sum(points[name] * gate, axis=1)
        for name in base
    }
    base_iou = base["target"] / np.maximum(base["target"] + base["outer"], 1.0e-12)
    candidate_iou = candidate["target"] / np.maximum(
        candidate["target"] + candidate["outer"], 1.0e-12
    )
    return {
        "target": candidate["target"],
        "outer": candidate["outer"],
        "boundary": candidate["boundary"],
        "outer_gain": 1.0 - normalized_flicker(candidate["outer"]) / max(normalized_flicker(base["outer"]), 1.0e-12),
        "boundary_gain": 1.0 - normalized_flicker(candidate["boundary"]) / max(normalized_flicker(base["boundary"]), 1.0e-12),
        "minimum_target_response": float(np.min(candidate["target"] / np.maximum(base["target"], 1.0e-12))),
        "maximum_soft_iou_drop": float(np.max(base_iou - candidate_iou)),
    }
