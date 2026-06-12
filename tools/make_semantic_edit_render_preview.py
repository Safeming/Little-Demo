#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.part_label_bank import PART_NAMES, load_part_label_bank


EDIT_COLORS = {
    "hair": (220, 45, 45),
    "face": (245, 170, 95),
    "upper": (35, 125, 245),
    "lower": (45, 210, 95),
    "shoes": (190, 85, 225),
    "skin": (255, 215, 125),
}


def resolve_part_weights(*, labels, soft_weights, part_index: int, mode: str, threshold: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int16).reshape(-1)
    mode = str(mode).lower()
    if mode == "hard":
        return (labels == int(part_index)).astype(np.float32)
    if mode == "soft":
        weights = np.asarray(soft_weights, dtype=np.float32)
        if weights.ndim != 2 or weights.shape[0] != labels.shape[0]:
            raise ValueError("soft_weights must have shape [N, C] and match labels")
        values = weights[:, int(part_index)].astype(np.float32, copy=False)
        return np.where(values >= float(threshold), values, 0.0).astype(np.float32, copy=False)
    raise ValueError(f"unsupported edit mode: {mode}")


def blend_edit_colors(base_colors, weights, target_rgb, *, alpha: float) -> np.ndarray:
    base = np.asarray(base_colors, dtype=np.float32)
    weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    target = np.asarray(target_rgb, dtype=np.float32).reshape(1, 3)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_colors must have shape [N, 3]")
    if weights_arr.shape[0] != base.shape[0]:
        raise ValueError("weights must match base color count")
    amount = np.clip(weights_arr, 0.0, 1.0)[:, None] * float(alpha)
    edited = base * (1.0 - amount) + target * amount
    return np.clip(edited, 0.0, 1.0).astype(np.float32, copy=False)


def _tensor_to_rgb_uint8(image) -> np.ndarray:
    import torch

    if torch.is_tensor(image):
        arr = image.detach().float().cpu().numpy()
    else:
        arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)


def compose_preview_sheet(panels: list[dict], output: Path | str, *, thumb_size: int = 192) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not panels:
        raise ValueError("panels must not be empty")
    columns = max(len(panel["images"]) for panel in panels)
    rows = len(panels)
    pad = 10
    header_h = 28
    row_label_h = 24
    width = pad + columns * (thumb_size + pad)
    height = pad + header_h + rows * (thumb_size + row_label_h + pad)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, (label, _) in enumerate(panels[0]["images"]):
        x = pad + col * (thumb_size + pad)
        draw.text((x + 4, pad + 8), str(label), fill=(0, 0, 0), font=font)
    for row, panel in enumerate(panels):
        y = pad + header_h + row * (thumb_size + row_label_h + pad)
        draw.text((pad, y + 4), f"{panel['view']} | {panel['part']}", fill=(0, 0, 0), font=font)
        for col, (_, image) in enumerate(panel["images"]):
            x = pad + col * (thumb_size + pad)
            img = image.convert("RGB").copy()
            img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
            tile = Image.new("RGB", (thumb_size, thumb_size), (245, 245, 245))
            tile.paste(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
            canvas.paste(tile, (x, y + row_label_h))
            draw.rectangle((x, y + row_label_h, x + thumb_size - 1, y + row_label_h + thumb_size - 1), outline=(170, 170, 170))
    canvas.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render color-edit previews from semantic part label banks.")
    parser.add_argument("--part-label-bank", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parts", nargs="+", default=("lower", "shoes"), choices=list(PART_NAMES))
    parser.add_argument("--views", nargs="*", default=None)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--soft-threshold", type=float, default=0.20)
    parser.add_argument("--edit-alpha", type=float, default=0.75)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    return parser.parse_args()


def _select_named_records(records: list[dict], names: list[str] | None, max_views: int) -> list[dict]:
    if names:
        wanted = set(str(name) for name in names)
        selected = [record for record in records if str(record["image_name"]) in wanted]
        missing = sorted(wanted - {str(record["image_name"]) for record in selected})
        if missing:
            raise ValueError(f"views not found in view_records: {missing}")
        return selected
    return records[: max(1, int(max_views))]


def make_render_preview(args: argparse.Namespace) -> dict:
    import torch
    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import _find_dataset_index, _load_config, _load_view_records

    bank = load_part_label_bank(args.part_label_bank)
    labels = np.asarray(bank.get("editable_label", bank["part_label"]), dtype=np.int16).reshape(-1)
    soft_weights = np.asarray(bank.get("soft_edit_weights", np.zeros((labels.shape[0], len(PART_NAMES)), dtype=np.float32)), dtype=np.float32)
    if soft_weights.shape != (labels.shape[0], len(PART_NAMES)):
        raise ValueError(f"soft_edit_weights must have shape ({labels.shape[0]}, {len(PART_NAMES)})")

    asset_root = args.asset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    records = _select_named_records(_load_view_records(asset_root), args.views, int(args.max_views))
    config_path = args.config.resolve() if args.config else asset_root.parent.parent / ".hydra" / "config.yaml"
    config = _load_config(config_path, checkpoint, asset_root, records, args)
    bg_color = [1, 1, 1] if bool(config.dataset.white_background) else [0, 0, 0]
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    stats = []
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, str(asset_root.parent))
        scene.eval()
        iteration = int(scene.load_checkpoint(str(checkpoint)))
        point_count = int(scene.gaussians.get_xyz.shape[0])
        if labels.shape[0] != point_count:
            raise ValueError(f"label bank point count {labels.shape[0]} does not match scene {point_count}")
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        target_cache = {
            part: torch.tensor(np.asarray(EDIT_COLORS[part], dtype=np.float32) / 255.0, dtype=torch.float32, device="cuda")
            for part in args.parts
        }
        labels_np = labels
        for record in records:
            dataset_index = _find_dataset_index(scene.test_dataset, record["image_name"])
            if dataset_index is None:
                raise RuntimeError(f"image {record['image_name']} not present in dataset")
            view = scene.test_dataset[dataset_index]
            deformed, _, colors_precomp = scene.convert_gaussians(view, iteration, compute_loss=False)
            base_pkg = rasterize_gaussians(view, deformed, config.pipeline, background, colors_precomp=colors_precomp, return_opacity=False)
            base_rgb = _tensor_to_rgb_uint8(base_pkg["render"].clamp(0.0, 1.0))
            base_colors_np = colors_precomp.detach().float().cpu().numpy()
            for part in args.parts:
                part_index = PART_NAMES.index(part)
                hard_weights = resolve_part_weights(
                    labels=labels_np,
                    soft_weights=soft_weights,
                    part_index=part_index,
                    mode="hard",
                    threshold=float(args.soft_threshold),
                )
                soft_values = resolve_part_weights(
                    labels=labels_np,
                    soft_weights=soft_weights,
                    part_index=part_index,
                    mode="soft",
                    threshold=float(args.soft_threshold),
                )
                target_np = np.asarray(EDIT_COLORS[part], dtype=np.float32) / 255.0
                hard_colors = torch.from_numpy(
                    blend_edit_colors(base_colors_np, hard_weights, target_np, alpha=float(args.edit_alpha))
                ).to(device=colors_precomp.device, dtype=colors_precomp.dtype)
                soft_colors = torch.from_numpy(
                    blend_edit_colors(base_colors_np, soft_values, target_np, alpha=float(args.edit_alpha))
                ).to(device=colors_precomp.device, dtype=colors_precomp.dtype)
                hard_pkg = rasterize_gaussians(view, deformed, config.pipeline, background, colors_precomp=hard_colors, return_opacity=False)
                soft_pkg = rasterize_gaussians(view, deformed, config.pipeline, background, colors_precomp=soft_colors, return_opacity=False)
                hard_rgb = _tensor_to_rgb_uint8(hard_pkg["render"].clamp(0.0, 1.0))
                soft_rgb = _tensor_to_rgb_uint8(soft_pkg["render"].clamp(0.0, 1.0))
                stem = f"{record['image_name']}_{part}"
                imageio.imwrite(frames_dir / f"{stem}_rgb.png", base_rgb)
                imageio.imwrite(frames_dir / f"{stem}_hard.png", hard_rgb)
                imageio.imwrite(frames_dir / f"{stem}_soft.png", soft_rgb)
                panels.append(
                    {
                        "view": str(record["image_name"]),
                        "part": part,
                        "images": [
                            ("RGB", Image.fromarray(base_rgb)),
                            ("Hard edit", Image.fromarray(hard_rgb)),
                            ("Soft edit", Image.fromarray(soft_rgb)),
                        ],
                    }
                )
                stats.append(
                    {
                        "view": str(record["image_name"]),
                        "part": part,
                        "hard_selected_count": int(np.sum(hard_weights > 0.0)),
                        "soft_selected_count": int(np.sum(soft_values > 0.0)),
                        "soft_weight_sum": float(np.sum(soft_values)),
                        "soft_weight_mean_selected": float(np.mean(soft_values[soft_values > 0.0])) if np.any(soft_values > 0.0) else 0.0,
                    }
                )
                del hard_pkg
                del soft_pkg
            del base_pkg
            del deformed
            del colors_precomp
            torch.cuda.empty_cache()
    sheet_path = output_dir / "semantic_edit_render_preview_sheet.png"
    compose_preview_sheet(panels, sheet_path, thumb_size=192)
    summary = {
        "preview_sheet": str(sheet_path),
        "part_label_bank": str(args.part_label_bank),
        "checkpoint": str(checkpoint),
        "asset_root": str(asset_root),
        "view_count": int(len(records)),
        "part_count": int(len(args.parts)),
        "soft_threshold": float(args.soft_threshold),
        "edit_alpha": float(args.edit_alpha),
        "stats": stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = make_render_preview(args)
    print(f"wrote {summary['preview_sheet']}")
    print(f"wrote {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
