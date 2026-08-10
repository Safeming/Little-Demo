#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.audit_a7c_r1_4vp_r4a_signed_renderer as r4a_audit


def _run(args) -> tuple[dict, int]:
    payload, status = r4a_audit._run(args)
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
