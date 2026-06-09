#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.graphics_utils import geom_transform_points


def _parse_csv_ints(value: str) -> list[int]:
    out = []
    for part in str(value).replace("[", "").replace("]", "").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


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
    gcd_step = max(1, int(gcd_step))
    return [values[0], values[-1] + gcd_step, gcd_step]


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor]:
    ndc = geom_transform_points(points.detach(), view.full_proj_transform)
    width = int(view.image_width)
    height = int(view.image_height)
    px = (ndc[:, 0] + 1.0) * 0.5 * float(max(width - 1, 1))
    py = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(max(height - 1, 1))
    valid = torch.isfinite(ndc).all(dim=-1)
    valid &= ndc[:, 2] > 0.0
    valid &= px >= 0.0
    valid &= px <= float(max(width - 1, 0))
    valid &= py >= 0.0
    valid &= py <= float(max(height - 1, 0))
    return torch.stack((px, py), dim=-1), valid


def _read_candidates(path: Path, max_candidates: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["frame"] = int(float(row["frame"]))
            row["birth_cam"] = int(float(row["birth_cam"]))
            row["birth_x"] = float(row["birth_x"])
            row["birth_y"] = float(row["birth_y"])
            row["score"] = float(row.get("score", 0.0))
            row["xyz"] = [float(v) for v in ast.literal_eval(row["xyz"])]
            for key in ("inner_views", "outer_views", "heldout_inner_views", "heldout_outer_views"):
                if key in row:
                    row[key] = float(row[key])
            for key in ("footprint_score", "footprint_inner_pixels", "footprint_outer_pixels"):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
    if rows and "footprint_score" in rows[0]:
        rows.sort(
            key=lambda item: (
                item.get("footprint_score", 0.0),
                item.get("footprint_inner_pixels", 0.0),
                -item.get("footprint_outer_pixels", 0.0),
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda item: (item.get("heldout_inner_views", 0.0), item.get("inner_views", 0.0), item.get("score", 0.0)), reverse=True)
    if max_candidates > 0:
        rows = rows[:max_candidates]
    return rows


def _copy_hydra_config(base_hydra: Path, out_dir: Path, config) -> None:
    hydra_dir = out_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hydra.yaml", "overrides.yaml"):
        src = base_hydra / name
        if src.exists():
            shutil.copy2(src, hydra_dir / name)
    OmegaConf.save(config, hydra_dir / "config.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Append v262 ray-carve support candidates to a v233d checkpoint without training.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--load-ckpt", required=True, type=Path)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parser-root", default="")
    parser.add_argument("--compact-mapping", default="")
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--checkpoint-iteration", type=int, default=135711)
    parser.add_argument("--parent-screen-radius", type=float, default=42.0)
    parser.add_argument("--child-opacity-factor", type=float, default=0.70)
    parser.add_argument("--child-opacity-floor", type=float, default=0.025)
    parser.add_argument("--child-opacity-ceiling", type=float, default=0.30)
    parser.add_argument("--child-scale-factor", type=float, default=0.45)
    args = parser.parse_args()

    candidates = _read_candidates(args.candidates_csv, args.max_candidates)
    if not candidates:
        raise RuntimeError(f"no candidates in {args.candidates_csv}")

    views = sorted({int(c["birth_cam"]) for c in candidates})
    frames = sorted({int(c["frame"]) for c in candidates})
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_dir)
    config.load_ckpt = str(args.load_ckpt)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.preload = False
    config.dataset.test_views.view = views
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
    config.resume.disable_densify_on_resume = True
    config.resume.disable_opacity_reset_on_resume = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _copy_hydra_config(args.config_path.parent, args.out_dir, config)

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, str(args.out_dir))
    scene.eval()
    loaded_iteration = int(scene.load_checkpoint(str(args.load_ckpt)))
    background = torch.tensor([1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0], dtype=torch.float32, device="cuda")

    view_lookup = {}
    for idx in range(len(scene.test_dataset)):
        view = scene.test_dataset[idx]
        image_name = str(view.image_name)
        for cam in views:
            token = f"c{cam:02d}_f"
            if image_name.startswith(token):
                frame = int(image_name.split("_f")[-1])
                if frame in frames:
                    view_lookup[(cam, frame)] = view

    base_count = int(scene.gaussians.get_xyz.shape[0])
    candidates_by_key = defaultdict(list)
    for cand_idx, cand in enumerate(candidates):
        candidates_by_key[(int(cand["birth_cam"]), int(cand["frame"]))].append((cand_idx, cand))

    parent_indices = []
    canonical_xyz = []
    parent_pick_rows = []
    for key, key_candidates in candidates_by_key.items():
        view = view_lookup.get(key)
        if view is None:
            continue
        with torch.no_grad():
            pkg = render(view, loaded_iteration, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
        deformed = pkg["deformed_gaussian"]
        xy, proj_valid = _project_points(deformed.get_xyz[:base_count], view)
        visible = proj_valid & pkg["visibility_filter"][:base_count].detach().bool() & (pkg["radii"][:base_count].detach() > 0)
        visible_idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
        visible_xy = xy[visible].detach()
        visible_deformed = deformed.get_xyz[visible_idx].detach().float() if visible_idx.numel() > 0 else None
        fwd_transform = getattr(deformed, "fwd_transform", None)
        for cand_idx, cand in key_candidates:
            target_xy = torch.tensor([cand["birth_x"], cand["birth_y"]], dtype=torch.float32, device="cuda")
            candidate_deformed = torch.tensor(cand["xyz"], dtype=torch.float32, device="cuda")
            if visible_idx.numel() > 0:
                screen_dist = torch.norm(visible_xy - target_xy.reshape(1, 2), dim=-1)
                within = screen_dist <= float(args.parent_screen_radius)
                if bool(within.any().item()):
                    local_idx = visible_idx[within]
                    local_deformed = deformed.get_xyz[local_idx].detach().float()
                    local_screen = screen_dist[within]
                    local_3d = torch.norm(local_deformed - candidate_deformed.reshape(1, 3), dim=-1)
                    pick_local = int(torch.argmin(local_screen / max(args.parent_screen_radius, 1.0) + 8.0 * local_3d).item())
                    parent_idx = int(local_idx[pick_local].item())
                else:
                    parent_idx = int(visible_idx[int(torch.argmin(screen_dist).item())].item())
            else:
                dist3d = torch.norm(deformed.get_xyz[:base_count].detach().float() - candidate_deformed.reshape(1, 3), dim=-1)
                parent_idx = int(torch.argmin(dist3d).item())

            parent_deformed = deformed.get_xyz[parent_idx].detach().float()
            parent_canonical = scene.gaussians.get_xyz[parent_idx].detach().float()
            if torch.is_tensor(fwd_transform) and fwd_transform.ndim == 3 and fwd_transform.shape[0] > parent_idx:
                rot = fwd_transform[parent_idx, :3, :3].detach().float()
                delta_deformed = candidate_deformed - parent_deformed
                try:
                    delta_canonical = torch.linalg.solve(rot, delta_deformed.reshape(3, 1)).reshape(3)
                except RuntimeError:
                    delta_canonical = torch.matmul(torch.linalg.pinv(rot), delta_deformed.reshape(3, 1)).reshape(3)
                new_xyz = parent_canonical + delta_canonical
            else:
                new_xyz = parent_canonical + (candidate_deformed - parent_deformed)

            parent_indices.append(parent_idx)
            canonical_xyz.append(new_xyz.detach())
            parent_pick_rows.append({
                "candidate_index": cand_idx,
                "frame": int(cand["frame"]),
                "birth_cam": int(cand["birth_cam"]),
                "parent_idx": parent_idx,
                "birth_x": float(cand["birth_x"]),
                "birth_y": float(cand["birth_y"]),
                "score": float(cand.get("score", 0.0)),
                "candidate_deformed_xyz": cand["xyz"],
                "canonical_xyz": [float(v) for v in new_xyz.detach().cpu().tolist()],
            })
        del pkg, deformed, xy, proj_valid, visible, visible_idx, visible_xy, visible_deformed
        torch.cuda.empty_cache()

    if not parent_indices:
        raise RuntimeError("no candidates could be assigned a parent")

    parent_idx = torch.tensor(parent_indices, dtype=torch.long, device="cuda")
    new_xyz = torch.stack(canonical_xyz, dim=0).to(device="cuda", dtype=scene.gaussians._xyz.dtype)
    new_features_dc = scene.gaussians._features_dc[parent_idx].detach().clone()
    new_features_rest = scene.gaussians._features_rest[parent_idx].detach().clone()
    parent_opacity = scene.gaussians.get_opacity[parent_idx].detach()
    child_opacity = (parent_opacity * float(args.child_opacity_factor)).clamp(
        min=max(float(args.child_opacity_floor), 1.0e-4),
        max=min(float(args.child_opacity_ceiling), 1.0 - 1.0e-4),
    )
    new_opacity = scene.gaussians.inverse_opacity_activation(child_opacity)
    parent_scaling = scene.gaussians.get_scaling[parent_idx].detach()
    new_scaling = scene.gaussians.scaling_inverse_activation((parent_scaling * float(args.child_scale_factor)).clamp_min(1.0e-6))
    new_rotation = scene.gaussians._rotation[parent_idx].detach().clone()
    new_boundary_opacity_residual = scene.gaussians._boundary_opacity_residual[parent_idx].detach().clone()
    new_boundary_scaling_residual = scene.gaussians._boundary_scaling_residual[parent_idx].detach().clone()

    if scene.gaussians.has_binding_state():
        new_binding_state = {}
        for key, value in scene.gaussians.binding_state.items():
            if torch.is_tensor(value) and value.shape[0] == base_count:
                new_binding_state[key] = value[parent_idx].detach().clone()
        new_binding_state = scene.gaussians._clear_newborn_binding_flags(new_binding_state)
        new_binding_state = scene.gaussians._annotate_densified_binding_lineage(
            new_binding_state,
            parent_idx,
            iteration=int(args.checkpoint_iteration),
        )
        new_binding_state = scene.gaussians._update_binding_offsets(new_binding_state, new_xyz - scene.gaussians.get_xyz[parent_idx].detach())
        count = int(parent_idx.shape[0])
        new_binding_state["boundary_support_role"] = torch.ones((count,), dtype=torch.long, device="cuda")
        new_binding_state["boundary_support_anchor_index"] = parent_idx.clone()
        new_binding_state["boundary_support_birth_iter"] = torch.full((count,), int(args.checkpoint_iteration), dtype=torch.long, device="cuda")
        new_binding_state["boundary_support_confidence"] = torch.tensor([float(c.get("score", 0.0)) for c in candidates[:count]], dtype=torch.float32, device="cuda").clamp(0.0, 1.0)
        new_binding_state["boundary_support_view_id"] = torch.tensor([int(c["birth_cam"]) for c in candidates[:count]], dtype=torch.long, device="cuda")
        new_binding_state["boundary_support_frame_id"] = torch.tensor([int(c["frame"]) for c in candidates[:count]], dtype=torch.long, device="cuda")
    else:
        new_binding_state = None

    scene.gaussians.densification_postfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_binding_state=new_binding_state,
        new_boundary_tags=torch.ones((new_xyz.shape[0],), dtype=torch.float32, device="cuda"),
        new_boundary_opacity_residual=new_boundary_opacity_residual,
        new_boundary_scaling_residual=new_boundary_scaling_residual,
        new_live_boundary_score=torch.ones((new_xyz.shape[0],), dtype=torch.float32, device="cuda"),
    )

    ckpt_path = scene.save_checkpoint(int(args.checkpoint_iteration), filename=f"ckpt{int(args.checkpoint_iteration)}.pth", verbose=True)
    summary = {
        "status": "ok",
        "base_ckpt": str(args.load_ckpt),
        "loaded_iteration": loaded_iteration,
        "checkpoint": ckpt_path,
        "checkpoint_iteration": int(args.checkpoint_iteration),
        "base_point_count": base_count,
        "appended_count": int(new_xyz.shape[0]),
        "final_point_count": int(scene.gaussians.get_xyz.shape[0]),
        "candidates_csv": str(args.candidates_csv),
        "parent_screen_radius": float(args.parent_screen_radius),
        "mean_child_opacity": float(child_opacity.mean().item()),
        "mean_child_scale": float((parent_scaling * float(args.child_scale_factor)).amax(dim=-1).mean().item()),
    }
    (args.out_dir / "v262_append_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "v262_append_parent_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parent_pick_rows[0].keys()))
        writer.writeheader()
        writer.writerows(parent_pick_rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
