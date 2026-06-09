#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene import GaussianModel, Scene
from utils.adopted_geometry import apply_explicit_binding_render_preset
from utils.general_utils import fix_random


IMAGE_RE = re.compile(r"c(?P<cam>\d+)_f(?P<frame>\d+)$")


def _parse_image_names(path: Path) -> list[str]:
    image_names: list[str] = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            image_name = str(row.get("image_name", "") or "").strip()
            if not image_name or image_name in seen:
                continue
            if IMAGE_RE.match(image_name) is None:
                continue
            seen.add(image_name)
            image_names.append(image_name)
    return image_names


def _list_to_omegaconf_repr(values: list[int | str]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _dataset_specs_for_images(image_names: list[str]) -> tuple[str, str]:
    pairs = []
    for image_name in image_names:
        match = IMAGE_RE.match(image_name)
        if match is None:
            continue
        pairs.append((int(match.group("cam")), int(match.group("frame"))))
    if not pairs:
        raise ValueError("no valid cXX_fYYYYYY image names found")
    views = sorted({cam for cam, _ in pairs})
    frames = sorted({frame for _, frame in pairs})
    if len(frames) == 1:
        frame_spec = [frames[0], frames[0] + 1, 1]
    else:
        frame_set = set(frames)
        step = 1
        for candidate in range(1, max(frames) - min(frames) + 1):
            generated = set(range(min(frames), max(frames) + 1, candidate))
            if frame_set.issubset(generated):
                step = candidate
                break
        frame_spec = [min(frames), max(frames) + step, step]
    return _list_to_omegaconf_repr(views), _list_to_omegaconf_repr(frame_spec)


def _find_dataset_index(dataset, image_name: str) -> int | None:
    for index, row in enumerate(getattr(dataset, "data", [])):
        cam_name = row.get("cam_name", "")
        frame_idx = int(row.get("frame_idx", -1))
        candidate = f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}"
        if candidate == image_name:
            return index
    return None


def _load_config(config_path: Path, overrides: list[str]):
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.set_struct(config, False)
    if "suffix" not in config:
        config.suffix = "test-view"
    config.dataset.preload = False
    apply_explicit_binding_render_preset(config, repo_root=REPO_ROOT)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Export renderer-time posed/canonical Gaussian coordinates.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--action-list-tsv", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--out-tsv", required=True, type=Path)
    parser.add_argument("--exp-dir", default="", type=Path)
    parser.add_argument("--explicit-binding-render-preset", default="v338_temporal_selector_grow_only_guard")
    parser.add_argument("--dataset-root", default="", type=Path)
    parser.add_argument("--subject", default="")
    parser.add_argument("--train-views", default="")
    parser.add_argument("--train-frames", default="")
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args()

    image_names = _parse_image_names(args.action_list_tsv)
    views_spec, frames_spec = _dataset_specs_for_images(image_names)

    exp_dir = str(args.exp_dir) if str(args.exp_dir) else str(args.out_npz.parent / "renderer_space_export")
    overrides = [
        "mode=test",
        f"load_ckpt={args.checkpoint}",
        f"exp_dir={exp_dir}",
        "dataset.preload=false",
        f"dataset.test_views.view={views_spec}",
        f"dataset.test_frames.view={frames_spec}",
        "dataset.parsing_prior.enable=false",
        "dataset.parsing_prior.roi_enable=false",
        f"explicit_binding_render_preset={args.explicit_binding_render_preset}",
        "wandb_disable=true",
    ]
    if str(args.dataset_root):
        overrides.append(f"dataset.root_dir={args.dataset_root}")
    if args.subject:
        overrides.append(f"dataset.subject={args.subject}")
    if args.train_views:
        overrides.append(f"dataset.train_views={args.train_views}")
    if args.train_frames:
        overrides.append(f"dataset.train_frames={args.train_frames}")

    config = _load_config(args.config, overrides)
    fix_random(int(config.get("seed", 0)))

    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        loaded_iteration = int(scene.load_checkpoint(str(args.checkpoint)))
        render_iteration = int(args.iteration) if int(args.iteration) >= 0 else loaded_iteration

        arrays: dict[str, object] = {}
        rows = []
        stored_names = []
        for output_index, image_name in enumerate(image_names):
            dataset_index = _find_dataset_index(scene.test_dataset, image_name)
            if dataset_index is None:
                raise RuntimeError(f"image {image_name} not present in exported test dataset")
            view = scene.test_dataset[dataset_index]
            pc, _, _ = scene.convert_gaussians(view, render_iteration, compute_loss=False)
            posed = pc.get_xyz.detach().cpu().numpy().astype("float32")
            canonical = getattr(pc, "canonical_xyz", pc.get_xyz).detach().cpu().numpy().astype("float32")
            arrays[f"posed_xyz_{output_index}"] = posed
            arrays[f"canonical_xyz_{output_index}"] = canonical
            stored_names.append(image_name)
            rows.append({
                "index": output_index,
                "image_name": image_name,
                "point_count": int(posed.shape[0]),
                "loaded_iteration": loaded_iteration,
                "render_iteration": render_iteration,
            })

    arrays["image_names"] = np.asarray(stored_names, dtype="U64")
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **arrays)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["index", "image_name", "point_count", "loaded_iteration", "render_iteration"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_npz} images={len(stored_names)} render_iteration={render_iteration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
