from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


REQUIRED_RELEASE_FILES = ("README.md", "environment.yml", ".gitmodules")
LOCAL_IMPORT_MODULES = ("diff_gaussian_rasterization_obj", "sparseconvnet")


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
