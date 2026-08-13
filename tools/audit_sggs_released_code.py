#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.sggs_released_code_canonical import (
    build_identity_record,
    fingerprint_record,
    probe_modules,
    scan_release_tree,
    scan_semantic_code,
)


PROBE_MODULES = (
    "torch",
    "cv2",
    "hydra",
    "omegaconf",
    "wandb",
    "lpips",
    "pytorch3d",
    "tinycudann",
    "diff_gaussian_rasterization",
    "diff_gaussian_rasterization_obj",
    "simple_knn",
    "sparseconvnet",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the released SG-GS repository without modifying it.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--body-models", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/opt/miniconda3/envs/ictrl/bin/python"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-network", action="store_true")
    return parser


def _native_launch_record(args, release, dependencies) -> dict:
    missing_inputs = []
    for name, path in (("dataset", args.dataset), ("body_models", args.body_models)):
        if not Path(path).exists():
            missing_inputs.append(name)
    unavailable = [name for name, value in dependencies.items() if not value["available"]]
    missing_local = release["declared_missing_local_modules"]
    blockers = list(dict.fromkeys([*missing_inputs, *missing_local, *unavailable]))
    first = blockers[0] if blockers else "native launch not attempted by read-only audit"
    return {
        "status": "blocked",
        "first_blocker": first,
        "blockers": blockers,
        "command": [
            str(args.python),
            str(Path(args.repo) / "train.py"),
            "dataset=zjumocap_377_mono",
            "opt.iterations=1",
            "wandb_disable=true",
        ],
        "mutated_official_repository": False,
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    release = scan_release_tree(args.repo)
    semantic = scan_semantic_code(Path(args.repo) / "train.py")
    dependencies = probe_modules(args.python, PROBE_MODULES)
    report = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_identity": build_identity_record(args.repo),
        "release_completeness": release,
        "semantic_code_state": semantic,
        "dependency_probe": dependencies,
        "native_launch": _native_launch_record(args, release, dependencies),
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "body_models": str(args.body_models.resolve()),
            "python": str(args.python),
        },
        "network_verification_skipped": bool(args.skip_network),
    }
    report["record_sha256"] = fingerprint_record(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
