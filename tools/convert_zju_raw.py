#!/usr/bin/env python3
"""Convert raw ZJU-MoCap data using the original preprocessing logic.

This script rewrites the earlier lightweight converter around the same logic as
the provided ``preprocess_ZJU-MoCap.py``:

- read camera parameters from ``annots.npy``
- read SMPL parameters from ``new_params`` (or a chosen source)
- compute ``minimal_shape``, ``bone_transforms``, and adjusted ``trans`` with
  ``human_body_prior`` + EasyMocap SMPL utilities
- write the layout expected by ``dataset/zjumocap.py`` under ``data/ZJUMoCap``

Compared with the standalone preprocessing snippet, this version:

- supports one or more subjects under ``zju_raw``
- keeps the repo-local default paths
- can symlink or copy images/masks
- optionally writes ``body_models/misc`` from a neutral SMPL pickle
- fails with a clear message if the required Python packages are unavailable
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation


DEFAULT_SOURCE_ROOT = Path("zju_raw")
DEFAULT_TARGET_ROOT = Path("data/ZJUMoCap")
DEFAULT_BODY_MODELS_ROOT = Path("body_models")
NEUTRAL_MODEL_CANDIDATES = [
    Path("body_models/smpl/neutral/model.pkl"),
    Path("body_models/smpl/neutral/SMPL_NEUTRAL.pkl"),
    Path("/remote-home/ming/AnimGuass/data/smpl/SMPL_NEUTRAL.pkl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw ZJU-MoCap data into the format expected by this repo."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Folder containing raw CoreView_* subjects, or one specific subject folder.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Where the converted subject folders are written.",
    )
    parser.add_argument(
        "--body-models-root",
        type=Path,
        default=DEFAULT_BODY_MODELS_ROOT,
        help="Local body-model root used by this repo.",
    )
    parser.add_argument(
        "--neutral-model",
        type=Path,
        default=None,
        help="Neutral SMPL pickle for BodyModel and body_models/misc generation.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject subset, e.g. CoreView_377 CoreView_386.",
    )
    parser.add_argument(
        "--params-source",
        choices=["params", "new_params"],
        default="new_params",
        help="Raw SMPL parameter directory used to create `models/*.npz`.",
    )
    parser.add_argument(
        "--opt-params-source",
        choices=["none", "params", "new_params"],
        default="none",
        help="Optional second parameter directory written to `opt_models/*.npz`.",
    )
    parser.add_argument(
        "--mask-source",
        choices=["mask", "mask_cihp"],
        default="mask_cihp",
        help="Raw mask directory used for output `.png` files.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy images and masks instead of symlinking them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing converted subject directory.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=0,
        help="Optional positive limit for converted frames per subject.",
    )
    parser.add_argument(
        "--skip-body-models",
        action="store_true",
        help="Skip writing `body_models/misc` even when a neutral SMPL pickle is available.",
    )
    parser.add_argument(
        "--extra-pythonpath",
        nargs="*",
        default=[],
        help="Extra paths prepended to `sys.path` before importing preprocessing dependencies.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_or_link(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def resolve_neutral_model(explicit_path: Optional[Path]) -> Optional[Path]:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    for candidate in NEUTRAL_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def discover_subject_roots(source_root: Path, requested_subjects: Optional[list[str]]) -> list[Path]:
    if (source_root / "annots.npy").exists():
        subject_roots = [source_root]
    else:
        subject_roots = sorted(
            path for path in source_root.iterdir() if path.is_dir() and (path / "annots.npy").exists()
        )
    if requested_subjects:
        wanted = set(requested_subjects)
        subject_roots = [path for path in subject_roots if path.name in wanted]
    return subject_roots


def load_annots(subject_root: Path) -> dict:
    annots_path = subject_root / "annots.npy"
    if not annots_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annots_path}")
    annots = np.load(annots_path, allow_pickle=True).item()
    if "cams" not in annots or "ims" not in annots:
        raise KeyError(f"Unexpected annots.npy structure in {annots_path}")
    return annots


def write_body_models_misc(neutral_model_path: Path, body_models_root: Path) -> None:
    ensure_dir(body_models_root)
    with open(neutral_model_path, "rb") as handle:
        neutral_data = pickle.load(handle, encoding="latin1")

    misc_root = body_models_root / "misc"
    ensure_dir(misc_root)

    faces = neutral_data["f"].astype(np.int64)
    j_regressor = neutral_data["J_regressor"].toarray().astype(np.float32)
    posedirs = neutral_data["posedirs"].astype(np.float32)
    shapedirs = neutral_data["shapedirs"].astype(np.float32)
    weights = neutral_data["weights"].astype(np.float32)
    v_template = neutral_data["v_template"].astype(np.float32)
    kintree_table = neutral_data["kintree_table"].astype(np.int32)

    np.savez(misc_root / "faces.npz", faces=faces)
    np.savez(misc_root / "J_regressors.npz", male=j_regressor, female=j_regressor, neutral=j_regressor)
    np.savez(misc_root / "posedirs_all.npz", male=posedirs, female=posedirs, neutral=posedirs)
    np.savez(misc_root / "shapedirs_all.npz", male=shapedirs, female=shapedirs, neutral=shapedirs)
    np.savez(misc_root / "skinning_weights_all.npz", male=weights, female=weights, neutral=weights)
    np.savez(misc_root / "v_templates.npz", male=v_template, female=v_template, neutral=v_template)
    np.save(misc_root / "kintree_table.npy", kintree_table)


def prepare_runtime_imports(extra_pythonpaths: Iterable[str]) -> tuple[type, object]:
    for path in reversed(list(extra_pythonpaths)):
        if path:
            sys.path.insert(0, path)

    missing = []
    try:
        body_model_mod = importlib.import_module("human_body_prior.body_model.body_model")
        BodyModel = body_model_mod.BodyModel
    except ModuleNotFoundError as exc:
        BodyModel = None
        missing.append(str(exc))

    try:
        smplmodel_mod = importlib.import_module("preprocess_datasets.easymocap.smplmodel")
        load_model = smplmodel_mod.load_model
    except ModuleNotFoundError as exc:
        load_model = None
        missing.append(str(exc))

    if BodyModel is None or load_model is None:
        joined = "; ".join(missing)
        raise RuntimeError(
            "Missing preprocessing dependencies. Install/import `human_body_prior` and "
            f"`preprocess_datasets.easymocap` first. Details: {joined}"
        )

    return BodyModel, load_model


def get_subject_camera_names(seq_name: str) -> list[str]:
    if seq_name in ["CoreView_313", "CoreView_315"]:
        cam_names = list(range(1, 20)) + [22, 23]
    else:
        cam_names = list(range(1, 24))
    return [str(cam_name) for cam_name in cam_names]


def get_image_and_mask_dirs(subject_root: Path, seq_name: str, cam_name: str, mask_source: str) -> tuple[Path, Path]:
    if seq_name in ["CoreView_313", "CoreView_315"]:
        image_dir = subject_root / f"Camera ({cam_name})"
        mask_dir = subject_root / mask_source / f"Camera ({cam_name})"
    else:
        image_dir = subject_root / f"Camera_B{cam_name}"
        mask_dir = subject_root / mask_source / f"Camera_B{cam_name}"
    return image_dir, mask_dir


def get_frame_index(img_file: Path, seq_name: str) -> tuple[int, int]:
    if seq_name in ["CoreView_313", "CoreView_315"]:
        idx = int(img_file.stem.split("_")[4])
        frame_index = idx - 1
    else:
        idx = int(img_file.stem)
        frame_index = idx
    return idx, frame_index


def compute_model_npz(
    smpl_file: Path,
    body_model,
    body_model_em,
    device: torch.device,
) -> dict:
    params = np.load(smpl_file, allow_pickle=True).item()

    root_orient = Rotation.from_rotvec(np.array(params["Rh"]).reshape([-1])).as_matrix()
    trans = np.array(params["Th"]).reshape([3, 1])

    betas = np.array(params["shapes"], dtype=np.float32)
    poses = np.array(params["poses"], dtype=np.float32)
    pose_body = poses[:, 3:66].copy()
    pose_hand = poses[:, 66:].copy()

    poses_torch = torch.from_numpy(poses).to(device)
    pose_body_torch = torch.from_numpy(pose_body).to(device)
    pose_hand_torch = torch.from_numpy(pose_hand).to(device)
    betas_torch = torch.from_numpy(betas).to(device)

    new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32)
    new_trans = trans.reshape([1, 3]).astype(np.float32)

    new_root_orient_torch = torch.from_numpy(new_root_orient).to(device)
    new_trans_torch = torch.from_numpy(new_trans).to(device)

    with torch.no_grad():
        body = body_model(betas=betas_torch)
        minimal_shape = body.v.detach().cpu().numpy()[0].astype(np.float32)

        body = body_model(
            root_orient=new_root_orient_torch,
            pose_body=pose_body_torch,
            pose_hand=pose_hand_torch,
            betas=betas_torch,
            trans=new_trans_torch,
        )

        verts = body_model_em(
            poses=poses_torch,
            shapes=betas_torch,
            Rh=new_root_orient_torch,
            Th=new_trans_torch,
            return_verts=True,
        )[0].detach().cpu().numpy()

        vertices = body.v.detach().cpu().numpy()[0]
        new_trans = new_trans + (verts - vertices).mean(0, keepdims=True).astype(np.float32)
        new_trans_torch = torch.from_numpy(new_trans).to(device)

        body = body_model(
            root_orient=new_root_orient_torch,
            pose_body=pose_body_torch,
            pose_hand=pose_hand_torch,
            betas=betas_torch,
            trans=new_trans_torch,
        )

        bone_transforms = body.bone_transforms.detach().cpu().numpy()[0].astype(np.float32)
        jtr_posed = body.Jtr.detach().cpu().numpy()[0].astype(np.float32)

    return {
        "minimal_shape": minimal_shape,
        "betas": betas,
        "Jtr_posed": jtr_posed,
        "bone_transforms": bone_transforms,
        "trans": new_trans[0].astype(np.float32),
        "root_orient": new_root_orient[0].astype(np.float32),
        "pose_body": pose_body[0].astype(np.float32),
        "pose_hand": pose_hand[0].astype(np.float32),
    }


def write_model_sequence(
    seq_name: str,
    subject_root: Path,
    model_out_dir: Path,
    params_source: str,
    body_model,
    body_model_em,
    device: torch.device,
    frame_limit: int,
) -> None:
    ensure_dir(model_out_dir)
    params_root = subject_root / params_source
    if not params_root.exists():
        raise FileNotFoundError(f"Missing parameter directory: {params_root}")

    param_files = sorted(params_root.glob("*.npy"))
    if frame_limit > 0:
        param_files = param_files[:frame_limit]

    for smpl_file in param_files:
        idx = int(smpl_file.stem)
        if seq_name in ["CoreView_313", "CoreView_315"]:
            out_name = f"{idx:06d}.npz"
        else:
            out_name = f"{idx:06d}.npz"

        model_data = compute_model_npz(smpl_file, body_model, body_model_em, device)
        np.savez(model_out_dir / out_name, **model_data)


def convert_subject(
    seq_name: str,
    source_subject_root: Path,
    target_subject_root: Path,
    mask_source: str,
    copy_files: bool,
    frame_limit: int,
    body_model,
    load_model,
    params_source: str,
    opt_params_source: str,
) -> None:
    annots = load_annots(source_subject_root)
    cameras = annots["cams"]
    cam_names = get_subject_camera_names(seq_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model_em = load_model(gender="neutral", model_type="smpl")

    ensure_dir(target_subject_root)
    all_cam_params = {"all_cam_names": cam_names}

    models_dir = target_subject_root / "models"
    write_model_sequence(
        seq_name=seq_name,
        subject_root=source_subject_root,
        model_out_dir=models_dir,
        params_source=params_source,
        body_model=body_model,
        body_model_em=body_model_em,
        device=device,
        frame_limit=frame_limit,
    )

    if opt_params_source != "none" and opt_params_source != params_source:
        write_model_sequence(
            seq_name=seq_name,
            subject_root=source_subject_root,
            model_out_dir=target_subject_root / "opt_models",
            params_source=opt_params_source,
            body_model=body_model,
            body_model_em=body_model_em,
            device=device,
            frame_limit=frame_limit,
        )

    for cam_idx, cam_name in enumerate(cam_names):
        if seq_name in ["CoreView_313", "CoreView_315"]:
            k_mat = cameras["K"][cam_idx]
            d_vec = cameras["D"][cam_idx]
            r_mat = cameras["R"][cam_idx]
        else:
            k_mat = cameras["K"][cam_idx].tolist()
            d_vec = cameras["D"][cam_idx].tolist()
            r_mat = cameras["R"][cam_idx].tolist()

        t_vec = (np.array(cameras["T"][cam_idx], dtype=np.float32).reshape(3, 1) / 1000.0).tolist()
        all_cam_params[cam_name] = {
            "K": k_mat,
            "D": d_vec,
            "R": r_mat,
            "T": t_vec,
        }

        cam_out_dir = target_subject_root / cam_name
        ensure_dir(cam_out_dir)

        image_dir, mask_dir = get_image_and_mask_dirs(source_subject_root, seq_name, cam_name, mask_source)
        img_files = sorted(Path(p) for p in glob.glob(os.path.join(image_dir, "*.jpg")))
        if frame_limit > 0:
            img_files = img_files[:frame_limit]

        for img_file in img_files:
            idx, _ = get_frame_index(img_file, seq_name)
            mask_file = mask_dir / f"{img_file.stem}.png"
            if not mask_file.exists():
                raise FileNotFoundError(f"Missing mask file: {mask_file}")
            copy_or_link(img_file, cam_out_dir / f"{idx:06d}.jpg", copy_files)
            copy_or_link(mask_file, cam_out_dir / f"{idx:06d}.png", copy_files)

    with open(target_subject_root / "cam_params.json", "w") as handle:
        json.dump(all_cam_params, handle)


def main() -> None:
    args = parse_args()
    if not args.source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {args.source_root}")

    subject_roots = discover_subject_roots(args.source_root, args.subjects)
    if not subject_roots:
        raise FileNotFoundError(f"No raw ZJU subjects found under: {args.source_root}")

    neutral_model = resolve_neutral_model(args.neutral_model)
    if neutral_model is None:
        raise FileNotFoundError(
            "Could not find a neutral SMPL model. Pass `--neutral-model` or place one at "
            "`body_models/smpl/neutral/model.pkl` or `body_models/smpl/neutral/SMPL_NEUTRAL.pkl`."
        )

    if not args.skip_body_models:
        print(f"[info] writing body_models metadata from {neutral_model}")
        write_body_models_misc(neutral_model, args.body_models_root)

    BodyModel, load_model = prepare_runtime_imports(args.extra_pythonpath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = BodyModel(bm_path=str(neutral_model), num_betas=10, batch_size=1).to(device)

    ensure_dir(args.target_root)

    for source_subject_root in subject_roots:
        seq_name = source_subject_root.name
        target_subject_root = args.target_root / seq_name
        if target_subject_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"Target subject already exists: {target_subject_root}. Use --overwrite to replace it.")
            remove_path(target_subject_root)

        print(f"[info] converting {source_subject_root}")
        convert_subject(
            seq_name=seq_name,
            source_subject_root=source_subject_root,
            target_subject_root=target_subject_root,
            mask_source=args.mask_source,
            copy_files=args.copy_files,
            frame_limit=args.frame_limit,
            body_model=body_model,
            load_model=load_model,
            params_source=args.params_source,
            opt_params_source=args.opt_params_source,
        )
        print(
            f"[done] {seq_name}: mask_source={args.mask_source}, models={args.params_source}, "
            f"opt_models={args.opt_params_source} -> {target_subject_root}"
        )


if __name__ == "__main__":
    main()
