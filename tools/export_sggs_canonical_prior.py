#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sggs_released_code_canonical import (
    build_topology_geometric_features,
    build_topology_knn,
    interpolate_smpl_prior,
)


EXPECTED_SGGS_HEAD = "27b9ed9c9e4c5663deb169247c2339ccafe1c254"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bone_transforms_02v(joints: np.ndarray) -> np.ndarray:
    transforms = np.tile(np.eye(4), (24, 1, 1))
    for chain, angle in (([1, 4, 7, 10], 45), ([2, 5, 8, 11], -45)):
        rotation = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        for index, joint in enumerate(chain):
            transforms[joint, :3, :3] = rotation
            translation = joints[joint].copy()
            if index > 0:
                parent = chain[index - 1]
                translation = rotation @ (translation - joints[parent]) + transforms[parent, :3, 3]
            transforms[joint, :3, 3] = translation
        transforms[chain, :3, 3] -= joints[chain] @ rotation.T
    return transforms


def _canonical_smpl(avatar_root: Path, body_models: Path, subject: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    model_paths = sorted((Path(avatar_root) / subject / "models").glob("*.npz"))
    if not model_paths:
        raise FileNotFoundError(f"missing avatar models for {subject}")
    model = np.load(model_paths[0])
    minimal_shape = np.asarray(model["minimal_shape"], dtype=np.float32)
    skinning_path = Path(body_models) / "misc/skinning_weights_all.npz"
    regressor_path = Path(body_models) / "misc/J_regressors.npz"
    skinning = np.asarray(np.load(skinning_path)["neutral"], dtype=np.float32)
    regressor = np.asarray(np.load(regressor_path)["neutral"], dtype=np.float32)
    if minimal_shape.shape[0] != skinning.shape[0] or regressor.shape[1] != minimal_shape.shape[0]:
        raise ValueError("SMPL model and misc arrays have incompatible vertex counts")
    joints = regressor @ minimal_shape
    transforms = _bone_transforms_02v(joints)
    blended = (skinning @ transforms.reshape(24, 16)).reshape(-1, 4, 4)
    vertices = (blended[:, :3, :3] @ minimal_shape[..., None]).squeeze(-1) + blended[:, :3, 3]
    provenance = {
        "source_model": str(model_paths[0].resolve()),
        "source_model_sha256": _sha256(model_paths[0]),
        "skinning_weights": str(skinning_path.resolve()),
        "skinning_weights_sha256": _sha256(skinning_path),
        "joint_regressor": str(regressor_path.resolve()),
        "joint_regressor_sha256": _sha256(regressor_path),
    }
    return torch.from_numpy(vertices).float(), torch.from_numpy(skinning).float(), provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the SG-GS released-code topology prior.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--avatar-data-root", type=Path, default=Path("/remote-home/ming/3dgs-avatar-release-main/data/ZJUMoCap"))
    parser.add_argument("--body-models", type=Path, default=Path("/remote-home/ming/3dgs-avatar-release-main/body_models"))
    parser.add_argument("--sggs-repo", type=Path, default=Path("/remote-home/ming/SGGS"))
    parser.add_argument("--smpl-knn-k", type=int, default=4)
    parser.add_argument("--knn-k", type=int, default=4)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    canonical_xyz = torch.load(args.input / "canonical_xyz.pt", map_location="cpu").float()
    if int(source_manifest.get("view_count", -1)) != 80:
        raise ValueError("expected exactly 80 frozen training views")
    if canonical_xyz.shape != (int(source_manifest.get("point_count", -1)), 3):
        raise ValueError("canonical point count does not match source manifest")
    sggs_head = subprocess.run(
        ["git", "-C", str(args.sggs_repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if sggs_head != EXPECTED_SGGS_HEAD:
        raise ValueError(f"unexpected SG-GS HEAD: {sggs_head}")
    smpl_xyz, smpl_skinning, smpl_provenance = _canonical_smpl(
        args.avatar_data_root, args.body_models, source_manifest["subject"]
    )
    interpolated = interpolate_smpl_prior(
        canonical_xyz, smpl_xyz, smpl_skinning, k=args.smpl_knn_k
    )
    features = build_topology_geometric_features(
        canonical_xyz,
        smpl_xyz,
        interpolated["skinning_weights"],
        interpolated["native_semantic_probs"],
    )
    topology = build_topology_knn(features[:, :29], k=args.knn_k)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.output / "topology_features.pt")
    torch.save(interpolated["native_labels"], args.output / "native_labels.pt")
    torch.save(interpolated["native_semantic_probs"], args.output / "native_semantic_probs.pt")
    torch.save(topology, args.output / "topology_knn.pt")
    manifest = {
        "schema_version": 1,
        "method": "SG-GS-Released-Code-Canonical (controlled-input adaptation)",
        "subject": source_manifest["subject"],
        "point_count": int(canonical_xyz.shape[0]),
        "view_count": 80,
        "source_frozen_views": str(args.input.resolve()),
        "source_manifest_sha256": _sha256(args.input / "manifest.json"),
        "source_canonical_xyz_sha256": _sha256(args.input / "canonical_xyz.pt"),
        "source_checkpoint": source_manifest["source_checkpoint"],
        "source_checkpoint_sha256": source_manifest["source_checkpoint_sha256"],
        "sggs_repo": str(args.sggs_repo.resolve()),
        "sggs_head": sggs_head,
        "smpl_knn_k": args.smpl_knn_k,
        "topology_knn_k": args.knn_k,
        "feature_dim": 32,
        "feature_layout": {"skinning": [0, 24], "native_semantics": [24, 29], "xyz": [29, 32]},
        "native_part_names": ["spine", "leg", "arm_hand", "head_neck", "hips"],
        "frozen": ["gaussians", "appearance", "deformation", "pose", "SMPL", "topology_features"],
        "trainable": ["compact6_readout_mlp"],
        "smpl_provenance": smpl_provenance,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "point_count": manifest["point_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
