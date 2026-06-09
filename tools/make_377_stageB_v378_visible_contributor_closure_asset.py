#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.adopted_geometry import apply_explicit_binding_render_preset
from utils.general_utils import fix_random
from utils.graphics_utils import geom_transform_points

IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _parse_ints(text: str) -> list[int]:
    out = []
    for token in str(text or "").replace("[", "").replace("]", "").split(","):
        token = token.strip()
        if token:
            out.append(int(float(token)))
    return out


def _range_for_values(values: list[int]) -> list[int]:
    values = sorted(set(int(v) for v in values))
    if not values:
        return [0, 1, 1]
    if len(values) == 1:
        return [values[0], values[0] + 1, 1]
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1) if values[i + 1] > values[i]]
    step = diffs[0]
    for diff in diffs[1:]:
        step = math.gcd(step, diff)
    return [values[0], values[-1] + max(1, step), max(1, step)]


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def _project_points(points: torch.Tensor, view) -> tuple[torch.Tensor, torch.Tensor]:
    ndc = geom_transform_points(points.detach(), view.full_proj_transform.to(device=points.device, dtype=points.dtype))
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


def _view_mask(view) -> torch.Tensor:
    mask = getattr(view, "hard_mask", None)
    if not torch.is_tensor(mask):
        mask = getattr(view, "original_mask", None)
    if not torch.is_tensor(mask):
        raise RuntimeError(f"view {getattr(view, 'image_name', '<unknown>')} has no mask")
    mask = mask.detach().float().to("cuda")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask[:1]
    else:
        mask = mask.reshape(1, *mask.shape[-2:])
    return mask > 0.5


def _render_support(render_rgb: torch.Tensor, threshold: float, close_kernel: int) -> torch.Tensor:
    luma = render_rgb[0] * 0.299 + render_rgb[1] * 0.587 + render_rgb[2] * 0.114
    chroma = render_rgb.max(dim=0).values - render_rgb.min(dim=0).values
    support = ((luma > float(threshold)) | (chroma > float(threshold) * 0.75)).float().unsqueeze(0)
    close_kernel = int(close_kernel)
    if close_kernel > 1:
        if close_kernel % 2 == 0:
            close_kernel += 1
        pad = close_kernel // 2
        support = F.max_pool2d(support.unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
        support = 1.0 - F.max_pool2d((1.0 - support).unsqueeze(0), close_kernel, stride=1, padding=pad)[0]
    return support > 0.5


def _dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    width = max(0, int(width))
    if width <= 0:
        return mask.float()
    kernel = width * 2 + 1
    return F.max_pool2d(mask.float().unsqueeze(0), kernel, stride=1, padding=width)[0]


def _load_validation_rows(path: Path, max_rows: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                target_gain = float(row.get("target_gain", 0.0) or 0.0)
            except ValueError:
                target_gain = 0.0
            if row.get("status") == "keep" or abs(target_gain) > 1.0e-9:
                continue
            pair_id = str(row.get("pair_id", "") or "")
            image_name = str(row.get("image_name", "") or "")
            if not pair_id or IMAGE_RE.match(image_name) is None:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r.get("image_name", ""), r.get("source_component_key", ""), r.get("pair_id", "")))
    return rows[:max_rows] if max_rows > 0 else rows


def _image_specs(images: list[str]) -> tuple[list[int], list[int]]:
    views, frames = [], []
    for image in images:
        match = IMAGE_RE.match(image)
        if match:
            views.append(int(match.group("cam")))
            frames.append(int(match.group("frame")))
    return sorted(set(views)), _range_for_values(frames)


def _build_scene(args, images: list[str]):
    views, frames = _image_specs(images)
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_struct(config, False)
    config.mode = "test"
    config.exp_dir = str(args.out_json.parent / "v378_visible_contributor_scene")
    config.load_ckpt = str(args.checkpoint)
    config.dataset.root_dir = str(args.dataset_root)
    config.dataset.subject = args.subject
    config.dataset.preload = False
    config.dataset.train_views = _parse_ints(args.train_views)
    config.dataset.train_frames = _parse_ints(args.train_frames)
    config.dataset.test_views.view = views
    config.dataset.test_frames.view = frames
    config.dataset.parsing_prior.enable = False
    config.dataset.parsing_prior.roi_enable = False
    config.explicit_binding_render_preset = args.explicit_binding_render_preset
    config.wandb_disable = True
    config.render_export_refine = False
    if "resume" not in config:
        config.resume = {}
    config.resume.allow_partial_converter_load = True
    config.resume.restore_gaussian_optimizer_state = False
    config.resume.restore_converter_optimizer_state = False
    config.resume.restore_converter_scheduler_state = False
    apply_explicit_binding_render_preset(config, repo_root=ROOT)
    fix_random(int(config.get("seed", 0)))
    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.eval()
    iteration = int(scene.load_checkpoint(str(args.checkpoint)))
    background = torch.tensor([1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0], dtype=torch.float32, device="cuda")
    return scene, config, iteration, background


def _find_dataset_index(dataset, image_name: str) -> int | None:
    for index, row in enumerate(getattr(dataset, "data", [])):
        cam_name = row.get("cam_name", "")
        frame_idx = int(row.get("frame_idx", -1))
        candidate = f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}"
        if candidate == image_name:
            return index
    return None


def _target_xy(group: dict, children: list[dict]) -> tuple[float, float] | None:
    for item in children + [group]:
        for x_key, y_key in (("target_screen_x", "target_screen_y"), ("activation_screen_x", "activation_screen_y")):
            if item.get(x_key) is not None and item.get(y_key) is not None:
                try:
                    return float(item[x_key]), float(item[y_key])
                except Exception:
                    pass
    for x_key, y_key in (("residual_mask_cx", "residual_mask_cy"), ("source_centroid_x", "source_centroid_y")):
        if group.get(x_key) is not None and group.get(y_key) is not None:
            try:
                return float(group[x_key]), float(group[y_key])
            except Exception:
                pass
    return None


def _render_records(args, images: list[str]) -> dict[str, dict]:
    scene, _config, iteration, background = _build_scene(args, images)
    records = {}
    with torch.no_grad():
        for image_name in images:
            dataset_index = _find_dataset_index(scene.test_dataset, image_name)
            if dataset_index is None:
                continue
            view = scene.test_dataset[dataset_index]
            pkg = render(view, iteration, scene, scene.cfg.pipeline, background, compute_loss=False, return_opacity=False)
            pc = pkg["deformed_gaussian"]
            base_count = int(scene.gaussians.get_xyz.shape[0])
            xyz = pc.get_xyz[:base_count].detach()
            canonical = getattr(pc, "canonical_xyz", scene.gaussians.get_xyz)[:base_count].detach()
            xy, proj_valid = _project_points(xyz, view)
            radii = pkg["radii"][:base_count].detach().float().to(device=xyz.device).reshape(-1)
            visible = proj_valid & pkg["visibility_filter"][:base_count].detach().bool().to(device=xyz.device) & (radii > 0)
            gt = _view_mask(view)
            support = _render_support(pkg["render"].detach().float().clamp(0.0, 1.0), args.render_support_threshold, args.close_kernel)
            near_gt = _dilate(gt, int(args.search_band_width)) > 0.5
            inner = (gt & (~support))[0].detach().cpu().numpy().astype(bool)
            outer = (support & (~gt) & near_gt)[0].detach().cpu().numpy().astype(bool)
            records[image_name] = {
                "xy": xy.detach().cpu(),
                "visible": visible.detach().cpu(),
                "radii": radii.detach().cpu(),
                "canonical": canonical.detach().cpu(),
                "inner": inner,
                "outer": outer,
            }
            del pkg
            torch.cuda.empty_cache()
    return records


def _disk_counts(mask: np.ndarray, center_xy: tuple[float, float], radius: float) -> int:
    height, width = mask.shape
    x0 = max(0, int(math.floor(float(center_xy[0]) - float(radius))))
    x1 = min(width - 1, int(math.ceil(float(center_xy[0]) + float(radius))))
    y0 = max(0, int(math.floor(float(center_xy[1]) - float(radius))))
    y1 = min(height - 1, int(math.ceil(float(center_xy[1]) + float(radius))))
    if x1 < x0 or y1 < y0:
        return 0
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    disk = (xx - float(center_xy[0])) ** 2 + (yy - float(center_xy[1])) ** 2 <= float(radius) ** 2
    return int((mask[y0 : y1 + 1, x0 : x1 + 1] & disk).sum())


def _pick_visible_ids(
    record: dict,
    target_xy: tuple[float, float],
    radius_px: float,
    max_ids: int,
    footprint_radius_px: float,
    min_inner_pixels: int,
    max_outer_pixels: int,
) -> list[int]:
    visible_idx = torch.nonzero(record["visible"], as_tuple=False).reshape(-1)
    if visible_idx.numel() <= 0:
        return []
    xy = record["xy"][visible_idx]
    radii = record["radii"][visible_idx].float()
    target = torch.tensor([float(target_xy[0]), float(target_xy[1])], dtype=torch.float32)
    dist = torch.linalg.norm(xy - target.reshape(1, 2), dim=-1)
    within = dist <= float(radius_px)
    if not bool(within.any().item()):
        order = torch.argsort(dist)[: max(1, int(max_ids))]
        return [int(visible_idx[int(i)].item()) for i in order]
    local_idx = visible_idx[within]
    local_dist = dist[within]
    local_radii = radii[within]
    keep_rows = []
    for local_i, dist_i, radius_i in zip(local_idx.detach().cpu().tolist(), local_dist.detach().cpu().tolist(), local_radii.detach().cpu().tolist()):
        xy_i = record["xy"][int(local_i)]
        footprint_radius = max(float(footprint_radius_px), float(radius_i) * 1.35, 2.0)
        inner_pixels = _disk_counts(record["inner"], (float(xy_i[0]), float(xy_i[1])), footprint_radius)
        outer_pixels = _disk_counts(record["outer"], (float(xy_i[0]), float(xy_i[1])), footprint_radius)
        if inner_pixels >= int(min_inner_pixels) and outer_pixels <= int(max_outer_pixels):
            keep_rows.append((int(local_i), float(inner_pixels), float(outer_pixels), float(dist_i), float(radius_i)))
    if not keep_rows:
        return []
    keep_rows.sort(key=lambda item: (item[1], -item[2], -item[3], item[4]), reverse=True)
    return [int(row[0]) for row in keep_rows[: max(1, int(max_ids))]]


def _retarget_items(items: list[dict], pair_id: str, new_pair_id: str, ids: list[int], canonical: torch.Tensor) -> list[dict]:
    out = []
    if not ids:
        return out
    ids = [int(i) for i in ids]
    centers = canonical[torch.tensor(ids, dtype=torch.long)].numpy()
    for index, item in enumerate(items):
        copy = dict(item)
        local_ids = ids[index % len(ids):] + ids[: index % len(ids)]
        local_ids = local_ids[: min(len(local_ids), max(4, min(8, len(ids))))]
        if not local_ids:
            local_ids = ids[:]
        center = centers[index % len(centers)]
        copy["pair_id"] = new_pair_id
        copy["component_key"] = f"{copy.get('component_key', 'vc')}__v378vc{index}"
        copy["source_pair_id"] = pair_id
        copy["anchor_point_ids"] = local_ids
        copy["top_point_ids"] = local_ids
        copy["source_top_point_ids"] = local_ids
        copy["anchor_explicit_ids_required"] = True
        copy["anchor_owner_gate"] = False
        copy["owner_gate"] = False
        copy["activation_owner_gate"] = False
        copy["canonical_center"] = [float(center[0]), float(center[1]), float(center[2])]
        copy["anchor_mode"] = "semantic_local_frame"
        copy["anchor_local_frame"] = True
        copy["rotate_covariance_with_anchor"] = True
        out.append(copy)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build generic no-gain visible-contributor closure split-child asset.")
    parser.add_argument("--config-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--seed-json", required=True, type=Path)
    parser.add_argument("--base-json", default="", type=Path)
    parser.add_argument("--validation-tsv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--train-views", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--train-frames", default="0,570,60")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    parser.add_argument("--max-actions", type=int, default=96)
    parser.add_argument("--visible-radius-px", type=float, default=36.0)
    parser.add_argument("--anchor-ids", type=int, default=8)
    parser.add_argument("--render-support-threshold", type=float, default=0.025)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--search-band-width", type=int, default=24)
    parser.add_argument("--footprint-radius-px", type=float, default=5.0)
    parser.add_argument("--min-footprint-inner-pixels", type=int, default=1)
    parser.add_argument("--max-footprint-outer-pixels", type=int, default=0)
    args = parser.parse_args()

    seed = json.loads(args.seed_json.read_text(encoding="utf-8"))
    rows = _load_validation_rows(args.validation_tsv, int(args.max_actions))
    pairs = {row["pair_id"]: row for row in rows}
    groups_by_pair = {str(g.get("pair_id", "")): g for g in seed.get("action_groups", [])}
    children_by_pair: dict[str, list[dict]] = {}
    actions_by_pair: dict[str, list[dict]] = {}
    for child in seed.get("children", []):
        children_by_pair.setdefault(str(child.get("pair_id", "")), []).append(child)
    for action in seed.get("actions", []):
        actions_by_pair.setdefault(str(action.get("pair_id", "")), []).append(action)

    images = sorted({row["image_name"] for row in rows})
    records = _render_records(args, images)
    out_groups, out_children, out_actions, summary_rows = [], [], [], []
    for pair_id, row in pairs.items():
        group = groups_by_pair.get(pair_id)
        if not group:
            continue
        children = children_by_pair.get(pair_id, [])
        target = _target_xy(group, children)
        record = records.get(row["image_name"])
        if target is None or record is None:
            continue
        ids = _pick_visible_ids(
            record,
            target,
            float(args.visible_radius_px),
            int(args.anchor_ids),
            float(args.footprint_radius_px),
            int(args.min_footprint_inner_pixels),
            int(args.max_footprint_outer_pixels),
        )
        if not ids:
            continue
        new_pair_id = f"{pair_id}:v378_vc"
        new_group = dict(group)
        new_group["pair_id"] = new_pair_id
        new_group["source_pair_id"] = pair_id
        new_group["visible_contributor_anchor_ids"] = ids
        new_group["strength_variant"] = f"{new_group.get('strength_variant', 'base')}_v378_vc"
        out_groups.append(new_group)
        out_children.extend(_retarget_items(children, pair_id, new_pair_id, ids, record["canonical"]))
        out_actions.extend(_retarget_items(actions_by_pair.get(pair_id, []), pair_id, new_pair_id, ids, record["canonical"]))
        summary_rows.append({
            "pair_id": new_pair_id,
            "source_pair_id": pair_id,
            "image_name": row["image_name"],
            "source_component_key": row.get("source_component_key", ""),
            "target_x": f"{target[0]:.3f}",
            "target_y": f"{target[1]:.3f}",
            "anchor_ids": ";".join(str(i) for i in ids),
            "anchor_count": len(ids),
        })

    base_groups: list[dict] = []
    base_children: list[dict] = []
    base_actions: list[dict] = []
    if str(args.base_json):
        base = json.loads(args.base_json.read_text(encoding="utf-8"))
        base_groups = list(base.get("action_groups", []))
        base_children = list(base.get("children", []))
        base_actions = list(base.get("actions", []))

    payload = {
        "version": "v378_generic_visible_contributor_closure_asset",
        "source_seed_json": str(args.seed_json),
        "source_validation_tsv": str(args.validation_tsv),
        "policy": "Retarget no-gain split-child actions to renderer-visible Gaussian contributors near the residual target.",
        "base_json": str(args.base_json) if str(args.base_json) else "",
        "group_count": len(base_groups) + len(out_groups),
        "child_count": len(base_children) + len(out_children),
        "action_count": len(base_actions) + len(out_actions),
        "closure_group_count": len(out_groups),
        "action_groups": base_groups + out_groups,
        "children": base_children + out_children,
        "actions": base_actions + out_actions,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["pair_id", "source_pair_id", "image_name", "source_component_key", "target_x", "target_y", "anchor_ids", "anchor_count"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"out_json": str(args.out_json), "groups": len(out_groups), "children": len(out_children), "actions": len(out_actions)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
