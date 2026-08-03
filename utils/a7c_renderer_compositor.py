from __future__ import annotations

import numpy as np


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
