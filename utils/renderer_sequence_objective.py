from __future__ import annotations

import numpy as np


SIGNALS = ("target", "outer", "boundary")


def _safe_scalar_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) > 1.0e-12:
        return float(numerator / denominator)
    return 1.0 if abs(numerator) <= 1.0e-12 else float("inf")


def _summarize_by_camera(values: np.ndarray, cameras: np.ndarray) -> dict[str, float]:
    camera_metrics = []
    for camera in np.unique(cameras):
        signal = values[cameras == camera]
        mean = float(np.mean(signal, dtype=np.float64))
        adjacent = float(np.mean(np.abs(np.diff(signal)), dtype=np.float64))
        if signal.size <= 1:
            adjacent = 0.0
        camera_metrics.append(
            (mean, adjacent, _safe_scalar_ratio(adjacent, abs(mean)))
        )
    return {
        "mean_response": float(np.mean([row[0] for row in camera_metrics])),
        "adjacent_absolute_change": float(
            np.mean([row[1] for row in camera_metrics])
        ),
        "normalized_flicker": float(np.mean([row[2] for row in camera_metrics])),
    }


def summarize_renderer_sequence_objective(
    *,
    a5_weights,
    candidate_weights,
    target_sequence,
    outer_sequence,
    boundary_sequence,
    camera_index,
    frame_index,
    processed_part_indices,
) -> dict:
    a5 = np.asarray(a5_weights, dtype=np.float64)
    candidate = np.asarray(candidate_weights, dtype=np.float64)
    if a5.ndim != 2 or candidate.shape != a5.shape:
        raise ValueError("a5_weights and candidate_weights must have matching shape [N, C]")
    sequences = {
        "target": np.asarray(target_sequence, dtype=np.float64),
        "outer": np.asarray(outer_sequence, dtype=np.float64),
        "boundary": np.asarray(boundary_sequence, dtype=np.float64),
    }
    sample_count = sequences["target"].shape[0]
    if sequences["target"].ndim != 3 or sequences["target"].shape[1:] != a5.shape:
        raise ValueError("renderer sequences must have shape [S, N, C]")
    if any(value.shape != sequences["target"].shape for value in sequences.values()):
        raise ValueError("renderer sequences must have matching shapes")
    if any(not np.all(np.isfinite(value)) or np.any(value < 0.0) for value in sequences.values()):
        raise ValueError("renderer sequences must be finite and non-negative")
    cameras = np.asarray(camera_index)
    frames = np.asarray(frame_index)
    if cameras.shape != (sample_count,) or frames.shape != (sample_count,):
        raise ValueError("camera_index and frame_index must match sequence sample count")
    if sample_count == 0:
        raise ValueError("renderer sequences must not be empty")
    if np.any(np.diff(cameras.astype(np.int64)) < 0):
        raise ValueError("camera_index must be nondecreasing")
    for camera in np.unique(cameras):
        camera_frames = frames[cameras == camera].astype(np.int64)
        if camera_frames.size > 1 and np.any(np.diff(camera_frames) <= 0):
            raise ValueError("frame_index must be strictly increasing within each camera")

    parts = [int(index) for index in processed_part_indices]
    if not parts or any(index < 0 or index >= a5.shape[1] for index in parts):
        raise ValueError("processed_part_indices must contain valid part indices")
    methods = {"a5": {}, "a7": {}}
    weights_by_method = {"a5": a5, "a7": candidate}
    for method, weights in weights_by_method.items():
        for part_index in parts:
            metrics = {}
            for signal, sequence in sequences.items():
                values = sequence[:, :, part_index] @ weights[:, part_index]
                summary = _summarize_by_camera(values, cameras)
                metrics.update(
                    {
                        f"{signal}_mean_response": summary["mean_response"],
                        f"{signal}_adjacent_absolute_change": summary[
                            "adjacent_absolute_change"
                        ],
                        f"{signal}_normalized_flicker": summary[
                            "normalized_flicker"
                        ],
                    }
                )
            methods[method][str(part_index)] = metrics

    per_part_ratios = {}
    metric_names = list(methods["a5"][str(parts[0])])
    for part_index in parts:
        key = str(part_index)
        per_part_ratios[key] = {
            metric: _safe_scalar_ratio(
                methods["a7"][key][metric], methods["a5"][key][metric]
            )
            for metric in metric_names
        }
    aggregate_ratios = {
        metric: float(
            np.mean([per_part_ratios[str(index)][metric] for index in parts])
        )
        for metric in metric_names
    }
    return {
        "processed_part_indices": parts,
        "methods": methods,
        "per_part_ratios": per_part_ratios,
        "aggregate_ratios": aggregate_ratios,
    }
