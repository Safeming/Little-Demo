#!/usr/bin/env python3
"""Prepare a raw ZJU-MoCap subject for this project.

The upstream converter in this repo requires `human_body_prior` and
`preprocess_datasets`. The local ZJU dump already contains `new_params` and
`new_vertices`, so this lightweight converter builds the ARAH-style layout
directly from those files and the repo's SMPL misc arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-subject-dir",
        type=Path,
        required=True,
        help="Raw subject directory, e.g. /remote-home/ming/dataSet/CoreView_377.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Processed ZJUMoCap root. The subject folder is created under it.",
    )
    parser.add_argument(
        "--subject-name",
        default=None,
        help="Output subject name. Defaults to the raw directory name.",
    )
    parser.add_argument(
        "--mask-source",
        choices=("mask", "mask_cihp"),
        default="mask",
        help="Raw mask directory used for the training foreground mask.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy RGB/mask files instead of creating symlinks.",
    )
    parser.add_argument(
        "--overwrite-models",
        action="store_true",
        help="Regenerate model npz files even if they already exist.",
    )
    parser.add_argument(
        "--camera-translation-scale",
        type=float,
        default=1.0e-3,
        help="Scale applied to ZJU camera translations when exporting cam_params.json.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_lbs():
    lbs_path = repo_root() / "models" / "pose_correction" / "lbs.py"
    spec = importlib.util.spec_from_file_location("lbs_mod", lbs_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load LBS module from {lbs_path}")
    spec.loader.exec_module(module)
    return module.lbs


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve() and not copy_files:
            return
        if dst.is_dir():
            raise IsADirectoryError(dst)
        dst.unlink()
    if copy_files:
        import shutil

        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_cam_params(annots: dict, subject_out: Path, camera_translation_scale: float) -> None:
    cams = annots["cams"]
    num_cams = len(cams["K"])
    camera_dict = {
        "all_cam_names": [str(i + 1) for i in range(num_cams)],
        "camera_translation_scale": camera_translation_scale,
    }
    for cam_idx in range(num_cams):
        cam_name = str(cam_idx + 1)
        camera_dict[cam_name] = {
            "K": np.asarray(cams["K"][cam_idx]).tolist(),
            "D": np.asarray(cams["D"][cam_idx]).tolist(),
            "R": np.asarray(cams["R"][cam_idx]).tolist(),
            "T": (np.asarray(cams["T"][cam_idx]) * camera_translation_scale).tolist(),
        }

    with open(subject_out / "cam_params.json", "w", encoding="utf-8") as handle:
        json.dump(camera_dict, handle)


def link_images_and_masks(raw_subject_dir: Path, subject_out: Path, annots: dict, mask_source: str, copy_files: bool) -> None:
    for frame_meta in annots["ims"]:
        for cam_idx, rel_img_path in enumerate(frame_meta["ims"]):
            cam_name = str(cam_idx + 1)
            rel_img_path = Path(rel_img_path)
            img_src = raw_subject_dir / rel_img_path
            mask_src = raw_subject_dir / mask_source / rel_img_path.with_suffix(".png")
            if not img_src.exists():
                raise FileNotFoundError(img_src)
            if not mask_src.exists():
                raise FileNotFoundError(mask_src)
            cam_dir = subject_out / cam_name
            link_or_copy(img_src, cam_dir / rel_img_path.name, copy_files)
            link_or_copy(mask_src, cam_dir / f"{rel_img_path.stem}.png", copy_files)


def load_body_model_arrays(body_models_dir: Path) -> dict[str, torch.Tensor]:
    misc_dir = body_models_dir / "misc"
    v_template = np.load(misc_dir / "v_templates.npz")["neutral"]
    lbs_weights = np.load(misc_dir / "skinning_weights_all.npz")["neutral"]
    posedirs = np.load(misc_dir / "posedirs_all.npz")["neutral"]
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    shapedirs = np.load(misc_dir / "shapedirs_all.npz")["neutral"]
    j_regressor = np.load(misc_dir / "J_regressors.npz")["neutral"]
    kintree_table = np.load(misc_dir / "kintree_table.npy")
    return {
        "v_template": torch.tensor(v_template, dtype=torch.float32).unsqueeze(0),
        "lbs_weights": torch.tensor(lbs_weights, dtype=torch.float32),
        "posedirs": torch.tensor(posedirs, dtype=torch.float32),
        "shapedirs": torch.tensor(shapedirs, dtype=torch.float32),
        "j_regressor": torch.tensor(j_regressor, dtype=torch.float32),
        "parents": torch.tensor(kintree_table[0], dtype=torch.long),
    }


def convert_models(raw_subject_dir: Path, subject_out: Path, overwrite_models: bool) -> None:
    body_models_dir = repo_root() / "body_models"
    if not body_models_dir.exists():
        raise FileNotFoundError(f"Missing body model directory: {body_models_dir}")

    lbs = load_lbs()
    arrays = load_body_model_arrays(body_models_dir)
    models_out = subject_out / "models"
    models_out.mkdir(parents=True, exist_ok=True)

    param_files = sorted((raw_subject_dir / "new_params").glob("*.npy"), key=lambda p: int(p.stem))
    vert_files = sorted((raw_subject_dir / "new_vertices").glob("*.npy"), key=lambda p: int(p.stem))
    if not param_files:
        raise FileNotFoundError(f"No parameter files under {raw_subject_dir / 'new_params'}")
    if len(param_files) != len(vert_files):
        raise RuntimeError("new_params and new_vertices do not have the same number of frames")

    with torch.no_grad():
        for frame_idx, (param_path, vert_path) in enumerate(zip(param_files, vert_files)):
            out_path = models_out / f"{frame_idx:06d}.npz"
            if out_path.exists() and not overwrite_models:
                continue

            params = np.load(param_path, allow_pickle=True).item()
            verts_gt = np.load(vert_path).astype(np.float32)

            pose_full = params["poses"].astype(np.float32).copy()
            pose_full[:, :3] = params["Rh"].astype(np.float32)
            betas = params["shapes"][:, :10].astype(np.float32)
            trans = params["Th"].astype(np.float32)

            verts_posed, _, _, bone_transforms, _, _, v_shaped, _ = lbs(
                betas=torch.tensor(betas, dtype=torch.float32),
                pose=torch.tensor(pose_full, dtype=torch.float32),
                v_template=arrays["v_template"].clone(),
                clothed_v_template=None,
                shapedirs=arrays["shapedirs"].clone(),
                posedirs=arrays["posedirs"].clone(),
                J_regressor=arrays["j_regressor"].clone(),
                parents=arrays["parents"],
                lbs_weights=arrays["lbs_weights"].clone(),
                dtype=torch.float32,
            )

            verts_pred = verts_posed[0].cpu().numpy() + trans
            trans = trans + (verts_gt - verts_pred).mean(axis=0, keepdims=True)

            np.savez(
                out_path,
                minimal_shape=v_shaped[0].cpu().numpy().astype(np.float32),
                betas=betas.astype(np.float32),
                bone_transforms=bone_transforms[0].cpu().numpy().astype(np.float32),
                trans=trans[0].astype(np.float32),
                root_orient=params["Rh"][0].astype(np.float32),
                pose_body=params["poses"][0, 3:66].astype(np.float32),
                pose_hand=params["poses"][0, 66:].astype(np.float32),
            )

            if frame_idx % 100 == 0 or frame_idx == len(param_files) - 1:
                print(f"Converted frame {frame_idx + 1}/{len(param_files)}", flush=True)


def main() -> int:
    args = parse_args()
    raw_subject_dir = args.raw_subject_dir.resolve()
    subject_name = args.subject_name or raw_subject_dir.name
    subject_out = args.output_root.resolve() / subject_name

    annots_path = raw_subject_dir / "annots.npy"
    if not annots_path.exists():
        raise FileNotFoundError(annots_path)

    subject_out.mkdir(parents=True, exist_ok=True)
    annots = np.load(annots_path, allow_pickle=True).item()
    write_cam_params(annots, subject_out, args.camera_translation_scale)
    link_images_and_masks(raw_subject_dir, subject_out, annots, args.mask_source, args.copy_files)
    convert_models(raw_subject_dir, subject_out, args.overwrite_models)
    print(f"Prepared subject at {subject_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
