#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import geom_transform_points


DEFAULT_SELECT = (
    "render_c21_f000240.png",
    "render_c21_f000300.png",
    "render_c22_f000240.png",
    "render_c23_f000300.png",
    "render_c23_f000420.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v226 Gaussian cleanup checkpoints for subject 377 StageB.")
    parser.add_argument("--base-exp", required=True, type=Path)
    parser.add_argument("--base-ckpt", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/ZJUMoCap"))
    parser.add_argument("--parser-root", type=Path, default=Path("data/parsers_from_hulk_multiview"))
    parser.add_argument("--compact-mapping", type=Path, default=Path("configs/semantic/hulk_cihp_compact_6.json"))
    parser.add_argument("--base-iter", type=int, default=111710)
    parser.add_argument("--select", nargs="*", default=list(DEFAULT_SELECT))
    parser.add_argument("--head-bottom-ratio", type=float, default=0.36)
    parser.add_argument("--head-pad-ratio-x", type=float, default=0.18)
    parser.add_argument("--head-pad-ratio-y", type=float, default=0.05)
    parser.add_argument("--outside-dilate-kernel", type=int, default=7)
    parser.add_argument("--render-fg-threshold", type=float, default=6.0 / 255.0)
    parser.add_argument("--opacity-sample-threshold", type=float, default=0.035)
    parser.add_argument("--min-point-opacity", type=float, default=0.01)
    parser.add_argument("--write-report-only", action="store_true")
    return parser.parse_args()


def update_cfg(cfg, key: str, value) -> None:
    OmegaConf.update(cfg, key, value, merge=False, force_add=True)


def build_cfg(args: argparse.Namespace, exp_dir: Path):
    cfg_path = args.base_exp / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    update_cfg(cfg, "mode", "test")
    update_cfg(cfg, "exp_dir", str(exp_dir))
    update_cfg(cfg, "dataset.root_dir", str(args.data_root))
    update_cfg(cfg, "dataset.preload", False)
    update_cfg(cfg, "dataset.test_views.view", [21, 22, 23])
    update_cfg(cfg, "dataset.test_frames.view", [0, 570, 60])
    update_cfg(cfg, "dataset.parsing_prior.enable", True)
    update_cfg(cfg, "dataset.parsing_prior.roi_enable", True)
    update_cfg(cfg, "dataset.parsing_prior.parser_root", str(args.parser_root))
    update_cfg(cfg, "dataset.parsing_prior.parser_layout", "cihp_subject")
    update_cfg(cfg, "dataset.parsing_prior.use_direct_parser_labels", True)
    update_cfg(cfg, "dataset.parsing_prior.compact_mapping_file", str(args.compact_mapping))
    update_cfg(cfg, "wandb_disable", True)
    return cfg


def prepare_exp_dir(base_exp: Path, exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    hydra_src = base_exp / ".hydra"
    hydra_dst = exp_dir / ".hydra"
    if hydra_src.exists() and not hydra_dst.exists():
        shutil.copytree(str(hydra_src), str(hydra_dst))
    elif hydra_src.exists():
        for src in hydra_src.iterdir():
            if src.is_file():
                shutil.copy2(str(src), str(hydra_dst / src.name))


def load_scene(args: argparse.Namespace, exp_dir: Path):
    cfg = build_cfg(args, exp_dir)
    gaussians = GaussianModel(cfg.model.gaussian)
    scene = Scene(cfg, gaussians, str(exp_dir))
    scene.eval()
    loaded_iteration = scene.load_checkpoint(str(args.base_ckpt))
    return cfg, scene, loaded_iteration


def tensor_mask_2d(view) -> torch.Tensor:
    mask = getattr(view, "hard_mask", None)
    if not torch.is_tensor(mask):
        mask = getattr(view, "original_mask", None)
    if not torch.is_tensor(mask):
        raise RuntimeError(f"view {getattr(view, 'image_name', '<unknown>')} has no mask tensor")
    if mask.dim() == 3:
        mask = mask[0]
    return (mask > 0.5)


def head_crop(mask: torch.Tensor, bottom_ratio: float, pad_ratio_x: float, pad_ratio_y: float):
    coords = torch.nonzero(mask, as_tuple=False)
    h, w = mask.shape
    if coords.numel() == 0:
        return 0, 0, w, max(1, int(round(0.4 * h)))
    ys = coords[:, 0]
    xs = coords[:, 1]
    y_top = int(ys.min().item())
    y_bottom = int(ys.max().item())
    x_left = int(xs.min().item())
    x_right = int(xs.max().item())
    body_h = max(y_bottom - y_top + 1, 1)
    body_w = max(x_right - x_left + 1, 1)
    pad_x = int(round(body_w * max(pad_ratio_x, 0.0)))
    pad_y = int(round(body_h * max(pad_ratio_y, 0.0)))
    x1 = max(x_left - pad_x, 0)
    x2 = min(x_right + 1 + pad_x, w)
    y1 = max(y_top - pad_y, 0)
    y2 = min(y_top + int(round(body_h * min(max(bottom_ratio, 0.1), 0.75))) + pad_y, h)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def project_points(points: torch.Tensor, camera):
    ndc = geom_transform_points(points.detach(), camera.full_proj_transform)
    width = int(camera.image_width)
    height = int(camera.image_height)
    px = ((ndc[:, 0] + 1.0) * 0.5 * float(max(width - 1, 1))).round().long()
    py = ((1.0 - (ndc[:, 1] + 1.0) * 0.5) * float(max(height - 1, 1))).round().long()
    valid = torch.isfinite(ndc).all(dim=1)
    valid = valid & (ndc[:, 2] > 0.0)
    valid = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px.clamp(0, max(width - 1, 0)), py.clamp(0, max(height - 1, 0)), valid


def dilate_bool_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask
    src = mask.detach().cpu().numpy().astype(np.uint8)
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    dilated = cv2.dilate(src, kernel, iterations=1).astype(bool)
    return torch.from_numpy(dilated).to(device=mask.device)


def compute_candidate_scores(args: argparse.Namespace, cfg, scene, loaded_iteration: int, report_dir: Path):
    selected = set(args.select)
    point_count = int(scene.gaussians.get_xyz.shape[0])
    counts = torch.zeros(point_count, dtype=torch.int32, device="cuda")
    score = torch.zeros(point_count, dtype=torch.float32, device="cuda")
    view_rows = []

    bg_color = [1, 1, 1] if cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for view in scene.test_dataset:
            render_name = f"render_{view.image_name}.png"
            if render_name not in selected:
                continue
            pkg = render(view, loaded_iteration, scene, cfg.pipeline, background, return_opacity=True, compute_loss=False)
            pc = pkg["deformed_gaussian"]
            if pc.get_xyz.shape[0] != point_count:
                raise RuntimeError(f"deformed point count changed for {view.image_name}: {pc.get_xyz.shape[0]} vs {point_count}")
            px, py, valid = project_points(pc.get_xyz, view)
            visible = pkg["visibility_filter"].detach().bool()
            radii = pkg["radii"].detach().float().clamp_min(0.0)
            opacity = pc.get_opacity.detach().reshape(-1).float()

            gt = tensor_mask_2d(view)
            x1, y1, x2, y2 = head_crop(gt, args.head_bottom_ratio, args.head_pad_ratio_x, args.head_pad_ratio_y)
            head_region = torch.zeros_like(gt, dtype=torch.bool)
            head_region[y1:y2, x1:x2] = True
            render_fg = pkg["render"].detach().clamp(0.0, 1.0).amax(dim=0) > float(args.render_fg_threshold)
            opacity_fg = pkg["opacity_render"].detach()[0] > float(args.opacity_sample_threshold)
            outside_pixels = head_region & (~gt) & render_fg & opacity_fg
            outside_pixels = dilate_bool_mask(outside_pixels, args.outside_dilate_kernel)

            sampled = outside_pixels[py, px]
            candidate = valid & visible & sampled & (opacity >= float(args.min_point_opacity))
            counts += candidate.to(dtype=counts.dtype)
            radii_scale = radii / radii.max().clamp_min(1.0)
            score += candidate.float() * (opacity + 0.25 * radii_scale)
            view_rows.append({
                "image_name": view.image_name,
                "render_name": render_name,
                "outside_pixels": int(outside_pixels.sum().item()),
                "candidate_points": int(candidate.sum().item()),
                "head_crop": [int(x1), int(y1), int(x2), int(y2)],
            })

    if not view_rows:
        raise RuntimeError(f"No selected test views found. Selected={sorted(selected)}")

    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "candidate_views.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "render_name", "outside_pixels", "candidate_points", "head_crop"], delimiter="\t")
        writer.writeheader()
        writer.writerows(view_rows)
    return counts, score, view_rows


def select_mask(counts: torch.Tensor, score: torch.Tensor, min_views: int, topk: int) -> torch.Tensor:
    eligible = counts >= int(min_views)
    eligible_count = int(eligible.sum().item())
    if eligible_count <= 0:
        return torch.zeros_like(eligible)
    if topk <= 0 or eligible_count <= topk:
        return eligible
    eligible_indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    eligible_scores = score[eligible_indices]
    top_indices = eligible_indices[torch.topk(eligible_scores, k=int(topk), largest=True).indices]
    mask = torch.zeros_like(eligible)
    mask[top_indices] = True
    return mask


def save_variant(args, variant, selected_mask: torch.Tensor, report_dir: Path, global_report: dict):
    exp_dir = args.out_root / f"377_hulk_light_{variant['name']}_{args.run_id}"
    prepare_exp_dir(args.base_exp, exp_dir)
    cfg, scene, loaded_iteration = load_scene(args, exp_dir)
    selected_mask = selected_mask.to(device=scene.gaussians.get_xyz.device)
    before_points = int(scene.gaussians.get_xyz.shape[0])
    selected_count = int(selected_mask.sum().item())
    if variant["mode"] == "cap":
        changed = scene.gaussians.reset_offender_subset(
            selected_mask,
            opacity_factor=variant["opacity_factor"],
            scaling_factor=variant["scaling_factor"],
            max_opacity=variant["max_opacity"],
            boundary_tag_value=1.0,
        )
    elif variant["mode"] == "prune":
        scene.gaussians.prune_points(selected_mask)
        changed = selected_count
    else:
        raise ValueError(variant["mode"])
    after_points = int(scene.gaussians.get_xyz.shape[0])
    ckpt_path = scene.save_checkpoint(args.base_iter, filename="best_ckpt.pth", verbose=True)
    scene.save_checkpoint(args.base_iter, filename=f"ckpt{args.base_iter}.pth", verbose=False)

    info = {
        "variant": variant,
        "exp_dir": str(exp_dir),
        "checkpoint": str(ckpt_path),
        "loaded_iteration": int(loaded_iteration or 0),
        "base_iter": int(args.base_iter),
        "selected_count": selected_count,
        "changed_count": int(changed),
        "before_points": before_points,
        "after_points": after_points,
        "global_report": global_report,
    }
    with (exp_dir / "v226_cleanup_info.json").open("w") as f:
        json.dump(info, f, indent=2)
    torch.save(
        {
            "selected_mask": selected_mask.detach().cpu(),
            "variant": variant,
        },
        exp_dir / "v226_selected_mask.pt",
    )
    return info


def main() -> int:
    args = parse_args()
    report_dir = args.out_root / f"377_hulk_light_v226a_candidate_report_{args.run_id}"
    prepare_exp_dir(args.base_exp, report_dir)
    cfg, scene, loaded_iteration = load_scene(args, report_dir)
    counts, score, view_rows = compute_candidate_scores(args, cfg, scene, int(loaded_iteration or args.base_iter), report_dir)
    opacity = scene.gaussians.get_opacity.detach().reshape(-1).float()
    scaling = scene.gaussians.get_scaling.detach().amax(dim=1).float()
    global_report = {
        "run_id": args.run_id,
        "base_exp": str(args.base_exp),
        "base_ckpt": str(args.base_ckpt),
        "loaded_iteration": int(loaded_iteration or 0),
        "point_count": int(counts.numel()),
        "candidate_views": view_rows,
        "counts_ge_1": int((counts >= 1).sum().item()),
        "counts_ge_2": int((counts >= 2).sum().item()),
        "counts_ge_3": int((counts >= 3).sum().item()),
        "max_count": int(counts.max().item()),
        "score_sum": float(score.sum().item()),
        "opacity_mean": float(opacity.mean().item()),
        "scaling_max_mean": float(scaling.mean().item()),
    }
    with (report_dir / "candidate_report.json").open("w") as f:
        json.dump(global_report, f, indent=2)
    torch.save(
        {
            "counts": counts.detach().cpu(),
            "score": score.detach().cpu(),
            "opacity": opacity.detach().cpu(),
            "scaling_max": scaling.detach().cpu(),
        },
        report_dir / "candidate_scores.pt",
    )

    variants = [
        {
            "name": "v226b_opacity_cap_mild",
            "mode": "cap",
            "min_views": 1,
            "topk": 1600,
            "opacity_factor": 0.45,
            "max_opacity": 0.18,
            "scaling_factor": 0.85,
        },
        {
            "name": "v226c_opacity_cap_strong",
            "mode": "cap",
            "min_views": 1,
            "topk": 2400,
            "opacity_factor": 0.20,
            "max_opacity": 0.10,
            "scaling_factor": 0.70,
        },
        {
            "name": "v226d_prune_highconf",
            "mode": "prune",
            "min_views": 2,
            "topk": 900,
        },
    ]

    summary_rows = []
    if not args.write_report_only:
        for variant in variants:
            mask = select_mask(counts, score, variant["min_views"], variant["topk"])
            info = save_variant(args, variant, mask, report_dir, global_report)
            summary_rows.append(info)

    summary_path = args.out_root / f"v226_cleanup_summary_{args.run_id}.tsv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "mode", "exp_dir", "checkpoint", "selected_count", "before_points", "after_points"],
            delimiter="\t",
        )
        writer.writeheader()
        for info in summary_rows:
            writer.writerow({
                "variant": info["variant"]["name"],
                "mode": info["variant"]["mode"],
                "exp_dir": info["exp_dir"],
                "checkpoint": info["checkpoint"],
                "selected_count": info["selected_count"],
                "before_points": info["before_points"],
                "after_points": info["after_points"],
            })

    print(json.dumps(global_report, indent=2))
    print(f"REPORT_DIR={report_dir}")
    print(f"SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
