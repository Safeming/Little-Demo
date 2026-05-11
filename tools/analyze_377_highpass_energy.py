#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


RENDER_RE = re.compile(r"render_c(?P<cam>\d+)_f(?P<frame>\d+)\.png$")


def _read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _dog(gray: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return gray.astype(np.float32) - blur


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _crop_arrays(
    render_hf: np.ndarray,
    gt_hf: np.ndarray,
    mask: np.ndarray,
    crop: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if crop is None:
        return render_hf, gt_hf, mask
    x1, y1, x2, y2 = crop
    height, width = mask.shape
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return render_hf[y1:y2, x1:x2], gt_hf[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def analyze_sample(
    render_path: Path,
    dataset_root: Path,
    subject: str,
    sigma: float,
    crop: tuple[int, int, int, int] | None,
) -> dict:
    match = RENDER_RE.match(render_path.name)
    if match is None:
        raise ValueError(f"Unexpected render filename: {render_path.name}")

    cam = match.group("cam")
    frame = int(match.group("frame"))
    frame_name = f"{frame:06d}"
    gt_path = dataset_root / subject / cam / f"{frame_name}.jpg"
    mask_path = dataset_root / subject / cam / f"{frame_name}.png"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    render_rgb = _read_rgb(render_path)
    size = (render_rgb.shape[1], render_rgb.shape[0])
    gt_rgb = _read_rgb(gt_path, size=size)
    mask = _read_mask(mask_path, size=size)

    render_hf = np.abs(_dog(_luma(render_rgb), sigma=sigma))
    gt_hf = np.abs(_dog(_luma(gt_rgb), sigma=sigma))
    fg_render_hp = _safe_mean(render_hf[mask])
    fg_gt_hp = _safe_mean(gt_hf[mask])
    fg_hp_l1 = _safe_mean(np.abs(render_hf[mask] - gt_hf[mask]))

    crop_render_hf, crop_gt_hf, crop_mask = _crop_arrays(render_hf, gt_hf, mask, crop)
    crop_render_hp = _safe_mean(crop_render_hf[crop_mask])
    crop_gt_hp = _safe_mean(crop_gt_hf[crop_mask])
    crop_hp_l1 = _safe_mean(np.abs(crop_render_hf[crop_mask] - crop_gt_hf[crop_mask]))

    eps = 1.0e-8
    return {
        "render": str(render_path),
        "cam": cam,
        "frame": frame,
        "fg_pixels": int(mask.sum()),
        "crop_fg_pixels": int(crop_mask.sum()),
        "fg_render_hp": fg_render_hp,
        "fg_gt_hp": fg_gt_hp,
        "fg_hp_ratio": fg_render_hp / max(fg_gt_hp, eps),
        "fg_hp_l1": fg_hp_l1,
        "crop_render_hp": crop_render_hp,
        "crop_gt_hp": crop_gt_hp,
        "crop_hp_ratio": crop_render_hp / max(crop_gt_hp, eps),
        "crop_hp_l1": crop_hp_l1,
    }


def _mean(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([float(record[key]) for record in records]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure render/GT high-frequency energy for CoreView_377 renders.")
    parser.add_argument("--render-exp", required=True, type=Path)
    parser.add_argument("--dataset-root", default=Path("data/ZJUMoCap"), type=Path)
    parser.add_argument("--subject", default="CoreView_377")
    parser.add_argument("--split-dir", default="test-view")
    parser.add_argument("--select", nargs="*", default=None)
    parser.add_argument("--sigma", type=float, default=1.15)
    parser.add_argument("--crop", nargs=4, type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    render_dir = args.render_exp / args.split_dir / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    out_dir = args.out_dir or (args.render_exp / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.select:
        render_paths = [render_dir / name for name in args.select]
    else:
        render_paths = sorted(render_dir.glob("render_c*_f*.png"))
    crop = tuple(args.crop) if args.crop else None

    records = []
    for path in render_paths:
        if not path.exists():
            continue
        records.append(analyze_sample(path, args.dataset_root, args.subject, args.sigma, crop))

    summary = {
        "render_exp": str(args.render_exp),
        "n_samples": len(records),
        "sigma": args.sigma,
        "crop": list(crop) if crop is not None else None,
        "fg_render_hp_mean": _mean(records, "fg_render_hp"),
        "fg_gt_hp_mean": _mean(records, "fg_gt_hp"),
        "fg_hp_ratio_mean": _mean(records, "fg_hp_ratio"),
        "fg_hp_l1_mean": _mean(records, "fg_hp_l1"),
        "crop_render_hp_mean": _mean(records, "crop_render_hp"),
        "crop_gt_hp_mean": _mean(records, "crop_gt_hp"),
        "crop_hp_ratio_mean": _mean(records, "crop_hp_ratio"),
        "crop_hp_l1_mean": _mean(records, "crop_hp_l1"),
        "samples": records,
    }
    (out_dir / "highpass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "highpass_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
