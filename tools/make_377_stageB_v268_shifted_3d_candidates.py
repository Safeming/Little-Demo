#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene import GaussianModel, Scene


def _range_spec_from_values(values: list[int]) -> list[int]:
    values = sorted(set(int(v) for v in values))
    if len(values) <= 1:
        return [values[0], values[0] + 1, 1] if values else [0, 1, 1]
    step = values[1] - values[0]
    if step > 0 and all(values[i + 1] - values[i] == step for i in range(len(values) - 1)):
        return [values[0], values[-1] + step, step]
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1) if values[i + 1] > values[i]]
    gcd_step = diffs[0]
    for diff in diffs[1:]:
        gcd_step = math.gcd(gcd_step, diff)
    return [values[0], values[-1] + max(1, int(gcd_step)), max(1, int(gcd_step))]


def _read_candidates(path: Path, max_candidates: int) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            row["frame"] = int(float(row["frame"]))
            row["birth_cam"] = int(float(row["birth_cam"]))
            row["birth_x"] = float(row["birth_x"])
            row["birth_y"] = float(row["birth_y"])
            row["score"] = float(row.get("score", 0.0))
            row["xyz"] = [float(v) for v in ast.literal_eval(row["xyz"])]
            for key in ("footprint_score", "actual_inner_pixels", "actual_outer_pixels", "heldout_inner_views"):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    rows.sort(
        key=lambda item: (
            item.get("footprint_score", 0.0),
            item.get("actual_inner_pixels", 0.0),
            item.get("heldout_inner_views", 0.0),
            -item.get("actual_outer_pixels", 0.0),
            item.get("score", 0.0),
        ),
        reverse=True,
    )
    if max_candidates > 0:
        rows = rows[:max_candidates]
    return rows, fieldnames


def _view_ids(name: str) -> tuple[int, int]:
    text = str(name)
    return int(text.split("_f")[0].replace("c", "")), int(text.split("_f")[-1])


def _camera_axes(view, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    right = torch.as_tensor(view.R[:, 0], device=device, dtype=dtype)
    up = torch.as_tensor(view.R[:, 1], device=device, dtype=dtype)
    right = torch.nn.functional.normalize(right.reshape(1, 3), dim=-1).reshape(3)
    up = torch.nn.functional.normalize(up.reshape(1, 3), dim=-1).reshape(3)
    return right, up


def main() -> int:
    parser = argparse.ArgumentParser(description="Create v268 shifted 3D support candidate CSVs for render-in-loop set search.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-summary", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--right-offset-world", type=float, default=0.0)
    parser.add_argument("--up-offset-world", type=float, default=0.0)
    parser.add_argument("--ray-offset-world", type=float, default=0.0)
    parser.add_argument("--pixel-x-offset", type=float, default=0.0)
    parser.add_argument("--pixel-y-offset", type=float, default=0.0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    rows, fieldnames = _read_candidates(args.source_csv, args.max_candidates)
    if not rows:
        raise RuntimeError(f"no candidates in {args.source_csv}")

    cams = sorted({int(r["birth_cam"]) for r in rows})
    frames = sorted({int(r["frame"]) for r in rows})
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_csv.parent / "scene_for_axes")
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.test_views.view = cams
    config.dataset.test_frames.view = _range_spec_from_values(frames)
    if args.parser_root:
        config.dataset.parsing_prior.parser_root = str(args.parser_root)
    if args.compact_mapping:
        config.dataset.parsing_prior.compact_mapping_file = str(args.compact_mapping)
    if "resume" not in config:
        config.resume = {}
    config.resume.allow_partial_converter_load = False
    config.resume.restore_gaussian_optimizer_state = False
    config.resume.restore_converter_optimizer_state = False
    config.resume.restore_converter_scheduler_state = False

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    loaded_iteration = int(scene.load_checkpoint(str(args.load_ckpt)))

    view_lookup = {}
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        cam, frame = _view_ids(view.image_name)
        if cam in cams and frame in frames:
            view_lookup[(cam, frame)] = view

    shifted = []
    shift_norms = []
    device = torch.device("cuda")
    dtype = torch.float32
    for row in rows:
        view = view_lookup.get((int(row["birth_cam"]), int(row["frame"])))
        if view is None:
            raise RuntimeError(f"missing birth view c{row['birth_cam']:02d}_f{row['frame']}")
        xyz = torch.tensor(row["xyz"], device=device, dtype=dtype)
        right, up = _camera_axes(view, device=device, dtype=dtype)
        cam_center = torch.as_tensor(view.camera_center, device=device, dtype=dtype)
        ray = torch.nn.functional.normalize((xyz - cam_center).reshape(1, 3), dim=-1).reshape(3)

        depth = float(torch.norm(xyz - cam_center).item())
        px_world = 2.0 * depth * math.tan(float(view.FoVx) * 0.5) / max(float(view.image_width), 1.0)
        py_world = 2.0 * depth * math.tan(float(view.FoVy) * 0.5) / max(float(view.image_height), 1.0)
        delta = (
            right * (float(args.right_offset_world) + float(args.pixel_x_offset) * px_world)
            + up * (float(args.up_offset_world) - float(args.pixel_y_offset) * py_world)
            + ray * float(args.ray_offset_world)
        )
        new_row = dict(row)
        new_xyz = xyz + delta
        new_row["xyz"] = [float(v) for v in new_xyz.detach().cpu().tolist()]
        new_row["v268_shift_tag"] = str(args.tag)
        new_row["v268_shift_right_world"] = float(args.right_offset_world)
        new_row["v268_shift_up_world"] = float(args.up_offset_world)
        new_row["v268_shift_ray_world"] = float(args.ray_offset_world)
        new_row["v268_shift_px"] = float(args.pixel_x_offset)
        new_row["v268_shift_py"] = float(args.pixel_y_offset)
        shifted.append(new_row)
        shift_norms.append(float(torch.norm(delta).item()))

    out_fields = list(fieldnames)
    for key in (
        "v268_shift_tag",
        "v268_shift_right_world",
        "v268_shift_up_world",
        "v268_shift_ray_world",
        "v268_shift_px",
        "v268_shift_py",
    ):
        if key not in out_fields:
            out_fields.append(key)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in shifted:
            public = dict(row)
            public["xyz"] = json.dumps(public["xyz"])
            writer.writerow(public)

    summary = {
        "status": "ok",
        "tag": str(args.tag),
        "source_csv": str(args.source_csv),
        "out_csv": str(args.out_csv),
        "loaded_iteration": loaded_iteration,
        "candidate_count": len(shifted),
        "right_offset_world": float(args.right_offset_world),
        "up_offset_world": float(args.up_offset_world),
        "ray_offset_world": float(args.ray_offset_world),
        "pixel_x_offset": float(args.pixel_x_offset),
        "pixel_y_offset": float(args.pixel_y_offset),
        "mean_shift_norm": float(sum(shift_norms) / max(len(shift_norms), 1)),
        "max_shift_norm": float(max(shift_norms) if shift_norms else 0.0),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
