#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


SCALING_IDX = 4
OPACITY_IDX = 6
BOUNDARY_TAG_IDX = 7
BOUNDARY_OPACITY_RESIDUAL_IDX = 8
BOUNDARY_SCALING_RESIDUAL_IDX = 9
OPTIMIZER_STATE_IDX = 16


def _load_ids(path: Path, key: str, limit: int | None) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(x) for x in data.get(key, [])]
    if limit is not None and limit >= 0:
        ids = ids[:limit]
    return ids


def _valid_ids(ids: list[int], point_count: int) -> torch.Tensor:
    seen = set()
    valid = []
    for idx in ids:
        if idx in seen or idx < 0 or idx >= point_count:
            continue
        seen.add(idx)
        valid.append(idx)
    return torch.tensor(valid, dtype=torch.long)


def _sigmoid_logit(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.log(value / (1.0 - value))


def _ensure_tensor(model_args: list, idx: int, shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
    value = model_args[idx]
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        value = torch.zeros(shape, dtype=like.dtype, device=like.device)
        model_args[idx] = value
    return value.clone()


def _boundary_support(model_args: list, point_count: int, device: torch.device) -> torch.Tensor:
    support = torch.zeros((point_count,), dtype=torch.float32, device=device)
    tag = model_args[BOUNDARY_TAG_IDX]
    if torch.is_tensor(tag) and tag.shape[0] == point_count:
        support = torch.maximum(support, tag.detach().to(device=device, dtype=torch.float32).reshape(-1).clamp(0.0, 1.0))
    opacity_residual = model_args[BOUNDARY_OPACITY_RESIDUAL_IDX]
    if torch.is_tensor(opacity_residual) and opacity_residual.shape[0] == point_count and opacity_residual.numel() > 0:
        support = torch.maximum(
            support,
            (opacity_residual.detach().abs().amax(dim=-1) > 1.0e-8).to(device=device, dtype=torch.float32),
        )
    scaling_residual = model_args[BOUNDARY_SCALING_RESIDUAL_IDX]
    if torch.is_tensor(scaling_residual) and scaling_residual.shape[0] == point_count and scaling_residual.numel() > 0:
        support = torch.maximum(
            support,
            (torch.norm(scaling_residual.detach(), dim=-1) > 1.0e-8).to(device=device, dtype=torch.float32),
        )
    return support.clamp(0.0, 1.0)


def _effective_scaling(model_args: list) -> torch.Tensor:
    raw = model_args[SCALING_IDX].detach().float()
    point_count = int(raw.shape[0])
    support = _boundary_support(model_args, point_count, raw.device)
    residual = model_args[BOUNDARY_SCALING_RESIDUAL_IDX]
    if torch.is_tensor(residual) and residual.shape == raw.shape:
        raw = raw + residual.detach().float() * support.unsqueeze(-1)
    return torch.exp(raw)


def _effective_opacity(model_args: list) -> torch.Tensor:
    raw = model_args[OPACITY_IDX].detach().float()
    point_count = int(raw.shape[0])
    support = _boundary_support(model_args, point_count, raw.device)
    residual = model_args[BOUNDARY_OPACITY_RESIDUAL_IDX]
    if torch.is_tensor(residual) and residual.shape == raw.shape:
        raw = raw + residual.detach().float() * support.unsqueeze(-1)
    return torch.sigmoid(raw)


def _stats(model_args: list, ids: torch.Tensor) -> dict:
    if ids.numel() == 0:
        return {
            "count": 0,
            "scale_mean": 0.0,
            "scale_max_mean": 0.0,
            "scale_max_p90": 0.0,
            "opacity_mean": 0.0,
            "opacity_min": 0.0,
            "opacity_max": 0.0,
        }
    scaling = _effective_scaling(model_args)[ids]
    opacity = _effective_opacity(model_args)[ids]
    scale_max = scaling.amax(dim=-1)
    return {
        "count": int(ids.numel()),
        "scale_mean": float(scaling.mean().item()),
        "scale_max_mean": float(scale_max.mean().item()),
        "scale_max_p90": float(torch.quantile(scale_max, 0.90).item()),
        "opacity_mean": float(opacity.mean().item()),
        "opacity_min": float(opacity.min().item()),
        "opacity_max": float(opacity.max().item()),
    }


def _apply_scale_edit(model_args: list, ids: torch.Tensor, factor: float, mode: str, clear_existing_residual: bool) -> None:
    if ids.numel() == 0 or math.isclose(float(factor), 1.0):
        return
    if factor <= 0:
        raise ValueError(f"scale factor must be positive, got {factor}")
    delta = float(math.log(factor))
    scaling = model_args[SCALING_IDX].clone()
    if mode == "direct":
        scaling[ids] = scaling[ids] + delta
        model_args[SCALING_IDX] = scaling
        return

    residual = _ensure_tensor(model_args, BOUNDARY_SCALING_RESIDUAL_IDX, scaling.shape, scaling)
    if clear_existing_residual:
        residual[ids] = 0.0
    residual[ids] = residual[ids] + delta
    model_args[BOUNDARY_SCALING_RESIDUAL_IDX] = residual


def _apply_opacity_edit(model_args: list, ids: torch.Tensor, factor: float, mode: str, clear_existing_residual: bool) -> None:
    if ids.numel() == 0 or math.isclose(float(factor), 1.0):
        return
    if factor <= 0:
        raise ValueError(f"opacity factor must be positive, got {factor}")
    opacity_raw = model_args[OPACITY_IDX].clone()
    current_opacity = _effective_opacity(model_args)[ids].to(device=opacity_raw.device, dtype=opacity_raw.dtype)
    target = (current_opacity * float(factor)).clamp(1.0e-4, 1.0 - 1.0e-4)
    target_raw = _sigmoid_logit(target)
    if mode == "direct":
        opacity_raw[ids] = target_raw
        model_args[OPACITY_IDX] = opacity_raw
        return

    residual = _ensure_tensor(model_args, BOUNDARY_OPACITY_RESIDUAL_IDX, opacity_raw.shape, opacity_raw)
    if clear_existing_residual:
        residual[ids] = 0.0
    # Add the smallest residual delta that makes the edited points hit the requested
    # actual opacity under the current raw opacity.
    residual[ids] = residual[ids] + (target_raw - opacity_raw[ids])
    model_args[BOUNDARY_OPACITY_RESIDUAL_IDX] = residual


def _mark_boundary_tags(model_args: list, ids: torch.Tensor) -> None:
    if ids.numel() == 0:
        return
    point_count = int(model_args[SCALING_IDX].shape[0])
    tag = model_args[BOUNDARY_TAG_IDX]
    if not torch.is_tensor(tag) or tag.shape[0] != point_count:
        tag = torch.zeros((point_count,), dtype=model_args[SCALING_IDX].dtype, device=model_args[SCALING_IDX].device)
    else:
        tag = tag.clone()
    tag[ids] = torch.maximum(tag[ids], torch.ones_like(tag[ids]))
    model_args[BOUNDARY_TAG_IDX] = tag


def _reset_gaussian_optimizer_state(model_args: list) -> None:
    if len(model_args) <= OPTIMIZER_STATE_IDX:
        return
    opt_state = model_args[OPTIMIZER_STATE_IDX]
    if not isinstance(opt_state, dict):
        return
    model_args[OPTIMIZER_STATE_IDX] = {
        "state": {},
        "param_groups": opt_state.get("param_groups", []),
    }


def _move_tensors(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Make v275 local signed calibration checkpoint variants.")
    parser.add_argument("--input-ckpt", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--edit-mode", choices=("residual", "direct"), default="residual")
    parser.add_argument("--max-shrink-points", type=int, default=-1)
    parser.add_argument("--max-grow-points", type=int, default=-1)
    parser.add_argument("--shrink-scale-factor", type=float, default=1.0)
    parser.add_argument("--grow-scale-factor", type=float, default=1.0)
    parser.add_argument("--shrink-opacity-factor", type=float, default=1.0)
    parser.add_argument("--grow-opacity-factor", type=float, default=1.0)
    parser.add_argument("--clear-existing-residual-on-edited", action="store_true")
    parser.add_argument("--mark-boundary-tags", action="store_true")
    parser.add_argument("--reset-gaussian-optimizer-state", action="store_true")
    parser.add_argument("--save-gaussian-device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    ckpt = torch.load(args.input_ckpt, map_location="cpu")
    if not isinstance(ckpt, (tuple, list)) or len(ckpt) < 1:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")
    ckpt_list = list(ckpt)
    model_args = list(ckpt_list[0])
    if len(model_args) < 18:
        raise ValueError(f"Expected GaussianModel capture tuple with 18 entries, got {len(model_args)}")

    scaling = model_args[SCALING_IDX]
    if not torch.is_tensor(scaling) or scaling.ndim != 2:
        raise ValueError("Checkpoint has no valid scaling tensor")
    point_count = int(scaling.shape[0])
    max_shrink = None if args.max_shrink_points is None or args.max_shrink_points < 0 else args.max_shrink_points
    max_grow = None if args.max_grow_points is None or args.max_grow_points < 0 else args.max_grow_points
    shrink_ids = _valid_ids(_load_ids(args.candidate_json, "shrink_point_ids", max_shrink), point_count)
    grow_ids = _valid_ids(_load_ids(args.candidate_json, "grow_point_ids", max_grow), point_count)

    before = {
        "shrink": _stats(model_args, shrink_ids),
        "grow": _stats(model_args, grow_ids),
    }

    _apply_scale_edit(
        model_args,
        shrink_ids,
        args.shrink_scale_factor,
        args.edit_mode,
        args.clear_existing_residual_on_edited,
    )
    _apply_scale_edit(
        model_args,
        grow_ids,
        args.grow_scale_factor,
        args.edit_mode,
        args.clear_existing_residual_on_edited,
    )
    _apply_opacity_edit(
        model_args,
        shrink_ids,
        args.shrink_opacity_factor,
        args.edit_mode,
        args.clear_existing_residual_on_edited,
    )
    _apply_opacity_edit(
        model_args,
        grow_ids,
        args.grow_opacity_factor,
        args.edit_mode,
        args.clear_existing_residual_on_edited,
    )

    if args.mark_boundary_tags:
        _mark_boundary_tags(model_args, shrink_ids)
        _mark_boundary_tags(model_args, grow_ids)
    if args.reset_gaussian_optimizer_state:
        _reset_gaussian_optimizer_state(model_args)

    after = {
        "shrink": _stats(model_args, shrink_ids),
        "grow": _stats(model_args, grow_ids),
    }
    save_device = torch.device("cuda" if args.save_gaussian_device == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt_list[0] = tuple(_move_tensors(model_args, save_device))
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(ckpt_list), args.output_ckpt)

    report = {
        "input_ckpt": str(args.input_ckpt),
        "output_ckpt": str(args.output_ckpt),
        "candidate_json": str(args.candidate_json),
        "edit_mode": args.edit_mode,
        "point_count": point_count,
        "shrink_count": int(shrink_ids.numel()),
        "grow_count": int(grow_ids.numel()),
        "shrink_point_ids": [int(x) for x in shrink_ids.tolist()],
        "grow_point_ids": [int(x) for x in grow_ids.tolist()],
        "factors": {
            "shrink_scale_factor": float(args.shrink_scale_factor),
            "grow_scale_factor": float(args.grow_scale_factor),
            "shrink_opacity_factor": float(args.shrink_opacity_factor),
            "grow_opacity_factor": float(args.grow_opacity_factor),
        },
        "before": before,
        "after": after,
        "clear_existing_residual_on_edited": bool(args.clear_existing_residual_on_edited),
        "mark_boundary_tags": bool(args.mark_boundary_tags),
        "reset_gaussian_optimizer_state": bool(args.reset_gaussian_optimizer_state),
        "save_gaussian_device": str(save_device),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
