#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.audit_a7c_r1_4vp_r4a_signed_renderer as r4a_audit


def verify_frozen_artifacts(root: Path) -> dict[str, str]:
    freeze_path = Path(root) / "models_frozen.json"
    if not freeze_path.is_file():
        raise ValueError("R4-B0 models_frozen manifest is missing")
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 36:
        raise ValueError("R4-B0 models_frozen must contain exactly 36 learned hashes")
    names = (
        "model.pt",
        "predictions.npz",
        "projection_certificates.json",
        "observability.json",
        "summary.json",
        "fit_projected_entry.json",
    )
    expected_names = {
        f"training/fold_{fold}/{name}"
        for fold in range(6)
        for name in names
    }
    if set(artifacts) != expected_names:
        raise ValueError("R4-B0 frozen artifact paths differ from contract")
    observed = {}
    for relative, fingerprint in artifacts.items():
        path = Path(root) / relative
        if not path.is_file() or r4a_audit.r3_audit._sha256(path) != str(fingerprint):
            raise ValueError(f"R4-B0 frozen artifact mismatch: {relative}")
        observed[relative] = str(fingerprint)
    return observed


def _run(args) -> tuple[dict, int]:
    original_verifier = r4a_audit.r3_audit.verify_frozen_artifacts
    r4a_audit.r3_audit.verify_frozen_artifacts = verify_frozen_artifacts
    try:
        payload, status = r4a_audit._run(args)
    finally:
        r4a_audit.r3_audit.verify_frozen_artifacts = original_verifier
    payload["stage"] = "r1_4vp_r4b0_projection_aware_held_canary"
    return payload, status


def parse_args(argv=None):
    return r4a_audit.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        payload, status = _run(args)
    except Exception as error:
        payload = {
            "stage": "r1_4vp_r4b0_projection_aware_held_canary",
            "verdict": "TRAINING_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "deployment_eligible": False,
            "teacher_eligible": False,
            "paper_test_eligible": False,
        }
        r4a_audit.r3_audit.r2_audit._write_json(
            args.output_dir / "held_block_summary.json", payload
        )
        r4a_audit.r3_audit.r2_audit._write_json(
            args.output_dir.parent / "summary.json", payload
        )
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    r4a_audit.r3_audit.r2_audit._write_json(
        args.output_dir / "held_block_summary.json", payload
    )
    r4a_audit.r3_audit.r2_audit._write_json(
        args.output_dir.parent / "summary.json", payload
    )
    print(json.dumps(payload["learned_summary"], indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
