from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


REQUIRED_RELEASE_FILES = ("README.md", "environment.yml", ".gitmodules")
LOCAL_IMPORT_MODULES = ("diff_gaussian_rasterization_obj", "sparseconvnet")
JOINT_TO_PART = (4, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_release_tree(repo: Path) -> dict:
    repo = Path(repo)
    present = {name: (repo / name).is_file() for name in REQUIRED_RELEASE_FILES}
    present["license"] = any(repo.glob("LICENSE*")) or any(repo.glob("license*"))
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in repo.rglob("*.py")
        if path.is_file()
    )
    declared = [module for module in LOCAL_IMPORT_MODULES if module in source_text]
    missing = [
        module
        for module in declared
        if not (repo / module).exists() and not any(repo.rglob(f"{module}*.so"))
    ]
    return {
        "repo": str(repo.resolve()),
        "present": present,
        "declared_missing_local_modules": missing,
    }


def _matching_lines(lines: list[str], needle: str, *, commented: bool) -> list[int]:
    result = []
    for number, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if (stripped.startswith("#") is commented) and needle in stripped:
            result.append(number)
    return result


def scan_semantic_code(train_path: Path) -> dict:
    train_path = Path(train_path)
    lines = train_path.read_text(encoding="utf-8", errors="replace").splitlines()
    init_lines = _matching_lines(lines, "gaussians.frozen_labels = labels.cuda()", commented=False)
    active_semantic = _matching_lines(lines, "loss += semantic_loss", commented=False)
    commented_semantic = _matching_lines(lines, "loss += semantic_loss", commented=True)
    active_neighborhood = _matching_lines(lines, "loss_consistency = neighborhood_consistency_loss", commented=False)
    commented_neighborhood = _matching_lines(lines, "loss_consistency = neighborhood_consistency_loss", commented=True)
    return {
        "train_path": str(train_path.resolve()),
        "train_sha256": _sha256(train_path),
        "active_smpl_label_initialization": bool(init_lines),
        "active_semantic_loss": bool(active_semantic),
        "commented_semantic_loss": bool(commented_semantic),
        "active_neighborhood_consistency": bool(active_neighborhood),
        "commented_neighborhood_consistency": bool(commented_neighborhood),
        "evidence": {
            "smpl_label_initialization": init_lines,
            "active_semantic_loss": active_semantic,
            "commented_semantic_loss": commented_semantic,
            "active_neighborhood_consistency": active_neighborhood,
            "commented_neighborhood_consistency": commented_neighborhood,
        },
    }


def probe_modules(python: Path, modules: Iterable[str]) -> dict:
    results = {}
    for module in modules:
        completed = subprocess.run(
            [str(python), "-c", f"import {module}"],
            text=True,
            capture_output=True,
            check=False,
        )
        error = (completed.stderr or completed.stdout).strip()
        results[str(module)] = {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "error": error,
        }
    return results


def build_identity_record(repo: Path) -> dict:
    repo = Path(repo)
    remote = subprocess.run(
        ["git", "-C", str(repo), "remote", "get-url", "origin"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {
        "repository": "Maxwell-Zhao/SGGS",
        "remote": remote,
        "head": head,
        "project_page": "https://sggs-projectpage.github.io/",
        "arxiv": "2408.09665",
        "code_url": "https://github.com/Maxwell-Zhao/SGGS",
    }


def fingerprint_record(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def native_smpl_labels(skinning_weights: torch.Tensor) -> torch.Tensor:
    if skinning_weights.ndim != 2 or skinning_weights.shape[1] != len(JOINT_TO_PART):
        raise ValueError("skinning_weights must have shape [N, 24]")
    if not torch.isfinite(skinning_weights).all():
        raise ValueError("skinning_weights must be finite")
    mapping = torch.tensor(JOINT_TO_PART, dtype=torch.long, device=skinning_weights.device)
    return mapping[torch.argmax(skinning_weights, dim=1)]


def interpolate_smpl_prior(
    gaussian_xyz: torch.Tensor,
    smpl_xyz: torch.Tensor,
    skinning_weights: torch.Tensor,
    *,
    k: int = 4,
) -> dict[str, torch.Tensor]:
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz must have shape [N, 3]")
    if smpl_xyz.ndim != 2 or smpl_xyz.shape[1] != 3:
        raise ValueError("smpl_xyz must have shape [M, 3]")
    if skinning_weights.shape != (smpl_xyz.shape[0], len(JOINT_TO_PART)):
        raise ValueError("skinning_weights must match SMPL vertices and have 24 channels")
    if not 1 <= int(k) <= int(smpl_xyz.shape[0]):
        raise ValueError("k must be between 1 and the SMPL vertex count")
    if not all(torch.isfinite(value).all() for value in (gaussian_xyz, smpl_xyz, skinning_weights)):
        raise ValueError("prior inputs must be finite")
    distances = torch.cdist(gaussian_xyz.float(), smpl_xyz.float())
    knn_distances, knn_indices = distances.topk(k=int(k), dim=1, largest=False)
    inverse = 1.0 / (knn_distances + 1.0e-6)
    knn_weights = inverse / inverse.sum(dim=1, keepdim=True)
    interpolated_skinning = (skinning_weights[knn_indices] * knn_weights[..., None]).sum(dim=1)
    interpolated_skinning = interpolated_skinning / interpolated_skinning.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    vertex_labels = native_smpl_labels(skinning_weights)
    vertex_probs = F.one_hot(vertex_labels, num_classes=5).to(dtype=skinning_weights.dtype)
    native_probs = (vertex_probs[knn_indices] * knn_weights[..., None]).sum(dim=1)
    native_probs = native_probs / native_probs.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return {
        "skinning_weights": interpolated_skinning,
        "native_semantic_probs": native_probs,
        "native_labels": torch.argmax(native_probs, dim=1),
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "knn_weights": knn_weights,
    }


def build_topology_geometric_features(
    gaussian_xyz: torch.Tensor,
    smpl_xyz: torch.Tensor,
    skinning_weights: torch.Tensor,
    native_semantic_probs: torch.Tensor,
) -> torch.Tensor:
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz must have shape [N, 3]")
    if skinning_weights.shape != (gaussian_xyz.shape[0], 24):
        raise ValueError("interpolated skinning_weights must have shape [N, 24]")
    if native_semantic_probs.shape != (gaussian_xyz.shape[0], 5):
        raise ValueError("native_semantic_probs must have shape [N, 5]")
    if smpl_xyz.ndim != 2 or smpl_xyz.shape[1] != 3:
        raise ValueError("smpl_xyz must have shape [M, 3]")
    center = 0.5 * (smpl_xyz.min(dim=0).values + smpl_xyz.max(dim=0).values)
    radius = torch.linalg.vector_norm(smpl_xyz - center, dim=1).max().clamp_min(1.0e-8)
    normalized_xyz = ((gaussian_xyz - center) / radius).clamp(-1.0, 1.0)
    features = torch.cat((skinning_weights, native_semantic_probs, normalized_xyz), dim=1)
    if features.shape[1] != 32 or not torch.isfinite(features).all():
        raise ValueError("topology-geometric features must be finite with 32 channels")
    return features


def topology_consistency_loss(
    probabilities: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_weights: torch.Tensor,
) -> torch.Tensor:
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must have shape [N, C]")
    if knn_indices.shape != knn_weights.shape or knn_indices.shape[0] != probabilities.shape[0]:
        raise ValueError("KNN tensors must have matching shape [N, K]")
    neighbors = probabilities[knn_indices.long()]
    center = probabilities[:, None, :]
    divergence = center * (torch.log(center.clamp_min(1.0e-10)) - torch.log(neighbors.clamp_min(1.0e-10)))
    per_edge = divergence.sum(dim=-1)
    normalized_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    return (per_edge * normalized_weights).sum(dim=1).mean()


def build_topology_knn(
    topology_features: torch.Tensor,
    *,
    k: int = 4,
    chunk_size: int = 4096,
) -> dict[str, torch.Tensor]:
    if topology_features.ndim != 2 or topology_features.shape[0] < 2:
        raise ValueError("topology_features must have shape [N, D] with N >= 2")
    if not 1 <= int(k) < int(topology_features.shape[0]):
        raise ValueError("k must be between 1 and point_count - 1")
    if chunk_size <= 0 or not torch.isfinite(topology_features).all():
        raise ValueError("chunk_size must be positive and features finite")
    all_distances = []
    all_indices = []
    point_count = int(topology_features.shape[0])
    for start in range(0, point_count, int(chunk_size)):
        stop = min(point_count, start + int(chunk_size))
        distances = torch.cdist(topology_features[start:stop].float(), topology_features.float())
        row_ids = torch.arange(stop - start, device=distances.device)
        col_ids = torch.arange(start, stop, device=distances.device)
        distances[row_ids, col_ids] = float("inf")
        values, indices = distances.topk(k=int(k), dim=1, largest=False)
        all_distances.append(values)
        all_indices.append(indices)
    knn_distances = torch.cat(all_distances, dim=0)
    knn_indices = torch.cat(all_indices, dim=0)
    inverse = 1.0 / (knn_distances + 1.0e-6)
    weights = inverse / inverse.sum(dim=1, keepdim=True)
    return {"indices": knn_indices, "distances": knn_distances, "weights": weights}
