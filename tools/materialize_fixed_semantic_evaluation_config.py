#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.semantic_eval_protocol import load_protocol, protocol_fingerprint


def materialize_fixed_config(
    *,
    protocol: dict,
    template: dict,
    checkpoint_fingerprint: str,
    bank_fingerprint: str,
) -> dict:
    selected = dict(template.get("selected", {}))
    required = ("soft_threshold", "support_threshold", "boundary_radius")
    missing = [key for key in required if key not in selected]
    if missing:
        raise ValueError(f"template missing frozen keys: {missing}")
    return {
        "protocol_name": str(protocol["protocol_name"]),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "bank_fingerprint": str(bank_fingerprint),
        "selection_mode": "cross_subject_fixed_from_template",
        "template_protocol_name": str(template.get("protocol_name", "")),
        "selected": {
            "soft_threshold": float(selected["soft_threshold"]),
            "support_threshold": float(selected["support_threshold"]),
            "boundary_radius": int(selected["boundary_radius"]),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a fixed cross-subject semantic config.")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--checkpoint-fingerprint", required=True)
    parser.add_argument("--bank-fingerprint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = materialize_fixed_config(
        protocol=load_protocol(args.protocol),
        template=json.loads(args.template.read_text(encoding="utf-8")),
        checkpoint_fingerprint=args.checkpoint_fingerprint,
        bank_fingerprint=args.bank_fingerprint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
