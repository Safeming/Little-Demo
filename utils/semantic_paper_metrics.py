from __future__ import annotations

from collections.abc import Iterable, Mapping

import cv2
import numpy as np


def _valid_bool_mask(shape, valid_mask=None) -> np.ndarray:
    if valid_mask is None:
        return np.ones(shape, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"valid_mask shape {valid.shape} does not match {shape}")
    return valid


def binary_segmentation_metrics(prediction, target, valid_mask=None) -> dict[str, float | int | bool]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have matching shapes")
    valid = _valid_bool_mask(pred.shape, valid_mask)
    pred &= valid
    truth &= valid
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    predicted = int(pred.sum())
    target_count = int(truth.sum())
    return {
        "intersection": intersection,
        "union": union,
        "predicted": predicted,
        "target": target_count,
        "target_empty": target_count == 0,
        "iou": float(intersection / union) if union > 0 else 1.0,
        "precision": float(intersection / predicted) if predicted > 0 else (1.0 if target_count == 0 else 0.0),
        "recall": float(intersection / target_count) if target_count > 0 else 1.0,
    }


def soft_iou(prediction, target, valid_mask=None) -> float:
    pred = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 1.0)
    truth = np.clip(np.asarray(target, dtype=np.float64), 0.0, 1.0)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have matching shapes")
    valid = _valid_bool_mask(pred.shape, valid_mask)
    intersection = float(np.minimum(pred, truth)[valid].sum())
    union = float(np.maximum(pred, truth)[valid].sum())
    return intersection / union if union > 0.0 else 1.0


def _boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("boundary metrics require 2D masks")
    if not np.any(binary):
        return binary.astype(bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (binary.astype(bool) & ~eroded.astype(bool)).astype(bool)


def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    tolerance = max(0, int(tolerance))
    if tolerance == 0:
        return np.asarray(mask, dtype=bool)
    size = tolerance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1).astype(bool)


def boundary_metrics(prediction, target, *, tolerance: int = 2, valid_mask=None) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have matching shapes")
    valid = _valid_bool_mask(pred.shape, valid_mask)
    pred_boundary = _boundary(pred & valid) & valid
    target_boundary = _boundary(truth & valid) & valid
    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())
    pred_match = int((pred_boundary & _dilate(target_boundary, tolerance) & valid).sum())
    target_match = int((target_boundary & _dilate(pred_boundary, tolerance) & valid).sum())
    precision = float(pred_match / pred_count) if pred_count > 0 else (1.0 if target_count == 0 else 0.0)
    recall = float(target_match / target_count) if target_count > 0 else 1.0
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    matched = min(pred_match, target_match)
    union = pred_count + target_count - matched
    return {
        "predicted_boundary": pred_count,
        "target_boundary": target_count,
        "matched_predicted_boundary": pred_match,
        "matched_target_boundary": target_match,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_iou": float(matched / union) if union > 0 else 1.0,
    }


def aggregate_part_metrics(rows: Iterable[Mapping]) -> dict[str, float | int]:
    rows = list(rows)
    evaluated = [row for row in rows if int(row.get("target", 0)) > 0]
    macro = float(np.mean([float(row["iou"]) for row in evaluated])) if evaluated else 0.0
    intersection = sum(int(row.get("intersection", 0)) for row in rows)
    union = sum(int(row.get("union", 0)) for row in rows)
    return {
        "macro_miou": macro,
        "micro_iou": float(intersection / union) if union > 0 else 0.0,
        "evaluated_part_count": len(evaluated),
        "empty_target_part_count": len(rows) - len(evaluated),
        "intersection": intersection,
        "union": union,
    }


def _sorted_curve(rows: Iterable[Mapping], retention_key: str) -> list[dict]:
    curve = [dict(row) for row in rows]
    if not curve:
        raise ValueError("retention curve is empty")
    curve.sort(key=lambda row: float(row[retention_key]))
    return curve


def interpolate_curve_at_retention(
    rows: Iterable[Mapping],
    retention: float,
    *,
    retention_key: str = "retention",
) -> dict:
    curve = _sorted_curve(rows, retention_key)
    target = float(retention)
    minimum = float(curve[0][retention_key])
    maximum = float(curve[-1][retention_key])
    epsilon = 1.0e-9
    if target < minimum - epsilon or target > maximum + epsilon:
        raise ValueError(
            f"retention {target} is outside observed retention range [{minimum}, {maximum}]"
        )
    for row in curve:
        if abs(float(row[retention_key]) - target) <= epsilon:
            result = dict(row)
            result[retention_key] = target
            return result
    for left, right in zip(curve[:-1], curve[1:]):
        left_x = float(left[retention_key])
        right_x = float(right[retention_key])
        if left_x <= target <= right_x:
            fraction = (target - left_x) / max(right_x - left_x, epsilon)
            result = {retention_key: target}
            shared_keys = set(left) & set(right)
            for key in shared_keys:
                if key == retention_key:
                    continue
                left_value = left[key]
                right_value = right[key]
                if isinstance(left_value, (int, float, np.number)) and isinstance(
                    right_value, (int, float, np.number)
                ):
                    result[key] = float(left_value) + fraction * (float(right_value) - float(left_value))
                elif left_value == right_value:
                    result[key] = left_value
            return result
    raise ValueError(f"cannot interpolate retention {target}")


def shared_retention_targets(curves: Mapping[str, Iterable[Mapping]], targets: Iterable[float]) -> list[float]:
    if not curves:
        return []
    ranges = []
    for rows in curves.values():
        curve = _sorted_curve(rows, "retention")
        ranges.append((float(curve[0]["retention"]), float(curve[-1]["retention"])))
    lower = max(item[0] for item in ranges)
    upper = min(item[1] for item in ranges)
    return [float(target) for target in targets if lower <= float(target) <= upper]
