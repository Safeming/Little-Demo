#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.frozen_semantic_method import (
    load_a7_temporal_contract,
    load_frozen_semantic_method,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the frozen paper method contract.")
    parser.add_argument("--method-freeze", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-config", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config is not None:
        if args.base_config is None:
            raise SystemExit("--base-config is required with --config")
        contract = load_a7_temporal_contract(args.config, args.base_config)
        print(
            json.dumps(
                {
                    "freeze_id": contract["freeze_id"],
                    "base_method": contract["base_method"],
                    "runtime_state": contract["runtime_state"],
                    "base_method_freeze_fingerprint": contract[
                        "base_method_freeze_fingerprint"
                    ],
                    "a7_contract_fingerprint": contract["_fingerprint"],
                    "source": contract["_source"],
                    "base_source": contract["_base_method_source"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.method_freeze is None:
        raise SystemExit("either --method-freeze or --config is required")
    frozen = load_frozen_semantic_method(args.method_freeze)
    print(
        json.dumps(
            {
                "freeze_id": frozen["freeze_id"],
                "status": frozen["status"],
                "primary_method": frozen["primary_method"],
                "extension_methods": frozen["extension_methods"],
                "fingerprint": frozen["_fingerprint"],
                "source": frozen["_source"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
