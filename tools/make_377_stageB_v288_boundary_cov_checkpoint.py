#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


BOUNDARY_TAG_IDX = 7
BOUNDARY_OPACITY_RESIDUAL_IDX = 8
BOUNDARY_SCALING_RESIDUAL_IDX = 9
SEMANTIC_REGION_IDX_OLD = 10
OPTIMIZER_STATE_IDX_NEW = 17


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _select_points(
    point_csv: Path,
    direction: str,
    limit: int,
    min_visible_hits: int,
    min_signed_margin: float,
) -> list[int]:
    rows = list(csv.DictReader(point_csv.open(encoding="utf-8")))
    selected: list[tuple[float, int]] = []
    for row in rows:
        point_idx = int(row["point_idx"])
        visible = int(float(row.get("visible_frame_hits", 0) or 0))
        if visible < min_visible_hits:
            continue
        over = _as_float(row, "over_score_sum")
        under = _as_float(row, "under_score_sum")
        if direction == "over":
            score = over - under
        else:
            score = under - over
        if score < min_signed_margin:
            continue
        selected.append((score, point_idx))
    selected.sort(reverse=True)
    seen: set[int] = set()
    ids: list[int] = []
    for _, idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        ids.append(idx)
        if limit >= 0 and len(ids) >= limit:
            break
    return ids


def _ensure_len19(model_args: list) -> list:
    if len(model_args) == 19:
        return model_args
    if len(model_args) != 18:
        raise ValueError(f"Expected 18 or 19 gaussian args, got {len(model_args)}")
    point_count = int(model_args[0 + 1].shape[0]) if torch.is_tensor(model_args[1]) else int(model_args[4].shape[0])
    like = model_args[6] if torch.is_tensor(model_args[6]) else model_args[4]
    cov = torch.zeros((point_count, 1), dtype=like.dtype, device=like.device)
    return model_args[:10] + [cov] + model_args[10:]


def _resize_cov_residual(cov: torch.Tensor, point_count: int, channels: int) -> torch.Tensor:
    channels = max(int(channels), 1)
    if not torch.is_tensor(cov):
        like = torch.zeros((point_count, channels), dtype=torch.float32)
        return like
    source = cov.detach().clone()
    if source.ndim == 1:
        source = source.reshape(-1, 1)
    resized = torch.zeros((point_count, channels), dtype=source.dtype, device=source.device)
    copy_rows = min(int(source.shape[0]), int(point_count))
    copy_cols = min(int(source.shape[1]), int(channels))
    if copy_rows > 0 and copy_cols > 0:
        resized[:copy_rows, :copy_cols] = source[:copy_rows, :copy_cols]
    return resized


def _reset_optimizer_state(model_args: list) -> None:
    opt_state = model_args[OPTIMIZER_STATE_IDX_NEW]
    if not isinstance(opt_state, dict):
        return
    model_args[OPTIMIZER_STATE_IDX_NEW] = {
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
    parser = argparse.ArgumentParser(description="Create v288 checkpoint with checkpoint-internal boundary covariance residual.")
    parser.add_argument("--input-ckpt", required=True, type=Path)
    parser.add_argument("--point-csv", required=True, type=Path)
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--max-over-points", type=int, default=96)
    parser.add_argument("--max-under-points", type=int, default=96)
    parser.add_argument("--min-visible-hits", type=int, default=12)
    parser.add_argument("--min-signed-margin", type=float, default=0.5)
    parser.add_argument("--over-residual", type=float, default=-0.015)
    parser.add_argument("--under-residual", type=float, default=0.015)
    parser.add_argument("--cov-residual-channels", type=int, default=1)
    parser.add_argument("--signed-two-channel", action="store_true")
    parser.add_argument("--mark-boundary-tags", action="store_true")
    parser.add_argument("--reset-gaussian-optimizer-state", action="store_true")
    parser.add_argument("--save-gaussian-device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    ckpt = torch.load(args.input_ckpt, map_location="cpu")
    if not isinstance(ckpt, (tuple, list)) or len(ckpt) < 1:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")
    ckpt_list = list(ckpt)
    model_args = _ensure_len19(list(ckpt_list[0]))
    point_count = int(model_args[1].shape[0])

    over_ids = [idx for idx in _select_points(
        args.point_csv,
        "over",
        args.max_over_points,
        args.min_visible_hits,
        args.min_signed_margin,
    ) if 0 <= idx < point_count]
    under_ids = [idx for idx in _select_points(
        args.point_csv,
        "under",
        args.max_under_points,
        args.min_visible_hits,
        args.min_signed_margin,
    ) if 0 <= idx < point_count]
    over_set = set(over_ids)
    under_ids = [idx for idx in under_ids if idx not in over_set]

    cov_channels = max(int(args.cov_residual_channels), 2 if args.signed_two_channel else 1)
    cov = _resize_cov_residual(model_args[10], point_count, cov_channels)
    cov.zero_()
    if over_ids:
        cov[torch.tensor(over_ids, dtype=torch.long), 0] = float(args.over_residual)
    if under_ids:
        under_channel = 1 if args.signed_two_channel and cov.shape[1] > 1 else 0
        cov[torch.tensor(under_ids, dtype=torch.long), under_channel] = float(args.under_residual)
    model_args[10] = cov

    if args.mark_boundary_tags:
        tag = model_args[BOUNDARY_TAG_IDX]
        if not torch.is_tensor(tag) or tag.shape[0] != point_count:
            tag = torch.zeros((point_count,), dtype=cov.dtype)
        else:
            tag = tag.clone().reshape(-1)
        edited = torch.tensor(over_ids + under_ids, dtype=torch.long)
        if edited.numel() > 0:
            tag[edited] = 1.0
        model_args[BOUNDARY_TAG_IDX] = tag

    if args.reset_gaussian_optimizer_state:
        _reset_optimizer_state(model_args)

    if args.save_gaussian_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt_list[0] = tuple(_move_tensors(model_args, device))
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuple(ckpt_list), args.output_ckpt)

    report = {
        "input_ckpt": str(args.input_ckpt),
        "output_ckpt": str(args.output_ckpt),
        "point_csv": str(args.point_csv),
        "point_count": point_count,
        "over_count": len(over_ids),
        "under_count": len(under_ids),
        "over_residual": float(args.over_residual),
        "under_residual": float(args.under_residual),
        "cov_residual_channels": int(cov.shape[1]),
        "signed_two_channel": bool(args.signed_two_channel),
        "min_visible_hits": int(args.min_visible_hits),
        "min_signed_margin": float(args.min_signed_margin),
        "over_ids": over_ids,
        "under_ids": under_ids,
        "cov_abs_mean": float(cov.abs().mean().item()),
        "cov_nonzero": int((cov.abs() > 0).sum().item()),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("output_ckpt", "over_count", "under_count", "cov_nonzero")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
