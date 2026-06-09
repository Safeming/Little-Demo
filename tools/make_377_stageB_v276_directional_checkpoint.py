#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


XYZ_IDX = 1
BOUNDARY_TAG_IDX = 7
BINDING_STATE_IDX = 12
OPTIMIZER_STATE_IDX = 16


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


def _cap_norm(delta: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0 or delta.numel() == 0:
        return delta
    norm = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return delta * torch.clamp(float(max_norm) / norm, max=1.0)


def _select_points(plan: dict, args: argparse.Namespace, point_count: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    rows = []
    for row in plan.get("points", []):
        pid = int(row["point_idx"])
        if pid < 0 or pid >= point_count:
            continue
        if args.direction != "all" and str(row.get("dominant_direction", "")) != args.direction:
            continue
        if float(row.get("direction_consistency", 0.0)) < float(args.min_direction_consistency):
            continue
        if float(row.get("conflict_ratio", 0.0)) > float(args.max_conflict_ratio):
            continue
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("weight_sum", 0.0)) * float(item.get("direction_consistency", 0.0)), reverse=True)
    if args.max_points >= 0:
        rows = rows[: int(args.max_points)]
    ids = torch.tensor([int(row["point_idx"]) for row in rows], dtype=torch.long)
    deltas = torch.tensor(
        [[float(row["delta_x"]), float(row["delta_y"]), float(row["delta_z"])] for row in rows],
        dtype=torch.float32,
    )
    return ids, deltas, rows


def _sync_binding_state(model_args: list, ids: torch.Tensor, delta: torch.Tensor) -> None:
    if ids.numel() == 0 or len(model_args) <= BINDING_STATE_IDX:
        return
    binding_state = model_args[BINDING_STATE_IDX]
    if not isinstance(binding_state, dict):
        return
    binding_state = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in binding_state.items()
    }
    for key in ("bound_xyz", "local_offset"):
        value = binding_state.get(key, None)
        if torch.is_tensor(value) and value.shape[0] > int(ids.max().item()) and value.shape[-1] == 3:
            value = value.clone()
            value[ids] = value[ids] + delta.to(device=value.device, dtype=value.dtype)
            binding_state[key] = value

    anchor_normal = binding_state.get("anchor_normal", None)
    if torch.is_tensor(anchor_normal) and anchor_normal.shape[0] > int(ids.max().item()) and anchor_normal.shape[-1] == 3:
        normals = anchor_normal[ids].to(device=delta.device, dtype=delta.dtype)
        delta_normal_mag = torch.sum(delta * normals, dim=-1, keepdim=True)
        delta_normal = delta_normal_mag * normals
        delta_tangent = delta - delta_normal
        for key, add in (("normal_offset", delta_normal), ("tangent_offset", delta_tangent)):
            value = binding_state.get(key, None)
            if torch.is_tensor(value) and value.shape[0] > int(ids.max().item()) and value.shape[-1] == 3:
                value = value.clone()
                value[ids] = value[ids] + add.to(device=value.device, dtype=value.dtype)
                binding_state[key] = value
        normal_offset = binding_state.get("normal_offset", None)
        if torch.is_tensor(normal_offset) and normal_offset.shape[0] > int(ids.max().item()):
            surface = binding_state.get("surface_distance", None)
            if torch.is_tensor(surface) and surface.shape[0] > int(ids.max().item()):
                surface = surface.clone()
                surface[ids] = torch.norm(normal_offset[ids].to(device=surface.device, dtype=surface.dtype), dim=-1)
                binding_state["surface_distance"] = surface
    model_args[BINDING_STATE_IDX] = binding_state


def _mark_boundary_tags(model_args: list, ids: torch.Tensor) -> None:
    if ids.numel() == 0:
        return
    xyz = model_args[XYZ_IDX]
    point_count = int(xyz.shape[0])
    tag = model_args[BOUNDARY_TAG_IDX]
    if not torch.is_tensor(tag) or tag.shape[0] != point_count:
        tag = torch.zeros((point_count,), dtype=xyz.dtype, device=xyz.device)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply v276 component directional canonical xyz checkpoint edit.")
    parser.add_argument("--input-ckpt", required=True, type=Path)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--direction", choices=("all", "outer", "inner"), default="all")
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=-1)
    parser.add_argument("--max-point-step", type=float, default=0.006)
    parser.add_argument("--min-direction-consistency", type=float, default=0.25)
    parser.add_argument("--max-conflict-ratio", type=float, default=0.70)
    parser.add_argument("--sync-binding-state", action=argparse.BooleanOptionalAction, default=True)
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

    xyz = model_args[XYZ_IDX].clone()
    point_count = int(xyz.shape[0])
    plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
    ids, deltas, selected_rows = _select_points(plan, args, point_count)
    ids = ids.to(device=xyz.device)
    deltas = deltas.to(device=xyz.device, dtype=xyz.dtype) * float(args.delta_scale)
    deltas = _cap_norm(deltas, float(args.max_point_step))

    before_xyz = xyz[ids].clone() if ids.numel() > 0 else xyz.new_zeros((0, 3))
    if ids.numel() > 0:
        xyz[ids] = xyz[ids] + deltas
    model_args[XYZ_IDX] = xyz
    if args.sync_binding_state:
        _sync_binding_state(model_args, ids, deltas)
    if args.mark_boundary_tags:
        _mark_boundary_tags(model_args, ids)
    if args.reset_gaussian_optimizer_state:
        _reset_gaussian_optimizer_state(model_args)

    save_device = torch.device("cuda" if args.save_gaussian_device == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt_list[0] = tuple(_move_tensors(model_args, save_device))
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(ckpt_list), args.output_ckpt)

    after_xyz = xyz[ids].clone() if ids.numel() > 0 else xyz.new_zeros((0, 3))
    norms = torch.norm(deltas, dim=-1) if deltas.numel() > 0 else torch.zeros((0,), dtype=xyz.dtype)
    report = {
        "input_ckpt": str(args.input_ckpt),
        "output_ckpt": str(args.output_ckpt),
        "plan_json": str(args.plan_json),
        "direction": args.direction,
        "delta_scale": float(args.delta_scale),
        "max_points": int(args.max_points),
        "max_point_step": float(args.max_point_step),
        "selected_count": int(ids.numel()),
        "selected_point_ids": [int(x) for x in ids.detach().cpu().tolist()],
        "delta_norm_mean": float(norms.mean().item()) if norms.numel() > 0 else 0.0,
        "delta_norm_max": float(norms.max().item()) if norms.numel() > 0 else 0.0,
        "before_xyz_mean": [float(x) for x in before_xyz.float().mean(dim=0).detach().cpu().tolist()] if before_xyz.numel() > 0 else [0.0, 0.0, 0.0],
        "after_xyz_mean": [float(x) for x in after_xyz.float().mean(dim=0).detach().cpu().tolist()] if after_xyz.numel() > 0 else [0.0, 0.0, 0.0],
        "sync_binding_state": bool(args.sync_binding_state),
        "mark_boundary_tags": bool(args.mark_boundary_tags),
        "reset_gaussian_optimizer_state": bool(args.reset_gaussian_optimizer_state),
        "save_gaussian_device": str(save_device),
        "selected_rows": selected_rows[:40],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
