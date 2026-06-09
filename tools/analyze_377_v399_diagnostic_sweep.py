from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = [
    "log_dir",
    "selected",
    "iteration",
    "hard_delta",
    "hard_floor_pass",
    "fg_boundary_edge_safe",
    "bad_frame_max_outer",
    "bad_frame_max_hard",
    "bad_frame_not_worse_than_v398",
    "support_cap_saturated",
    "over_cap_saturation",
]

V398_REFERENCE_MAX_OUTER = 7.0
V398_REFERENCE_MAX_HARD = 0.00052784


def _float_value(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_selected_json(log_dir: Path) -> Path:
    preferred = log_dir / "v399_selected_checkpoint.json"
    if preferred.exists():
        return preferred
    matches = sorted(log_dir.glob("*_selected_checkpoint.json"))
    if matches:
        return matches[0]
    return preferred


def summarize_run(log_dir: Path) -> dict[str, str]:
    log_dir = Path(log_dir)
    payload_path = _find_selected_json(log_dir)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    selected = payload.get("selected") or {}
    bad = payload.get("bad_frame_diagnostics", {}) or {}
    cap = payload.get("cap_diagnostics", {}) or {}
    floor = payload.get("v392_floor", {}) or {}

    hard_delta = _float_value(selected.get("hard_delta"))
    hard_floor = _float_value(floor.get("hard_delta"), default=-0.00023074)
    fg_safe = _float_value(selected.get("fg_delta"), default=1.0) <= 0.0
    boundary_safe = _float_value(selected.get("boundary_delta"), default=1.0) <= 0.0
    edge_safe = _float_value(selected.get("edge_delta"), default=1.0) <= 0.0
    max_outer = _float_value(bad.get("selected_bad_frame_max_outer_delta"))
    max_hard = _float_value(bad.get("selected_bad_frame_max_hard_delta"))
    over_saturation = cap.get("selected_support_over_cap_saturation")

    return {
        "log_dir": str(log_dir),
        "selected": "1" if selected else "0",
        "iteration": str(selected.get("iteration", "")),
        "hard_delta": f"{hard_delta:.8f}",
        "hard_floor_pass": "1" if selected and hard_delta <= hard_floor else "0",
        "fg_boundary_edge_safe": "1" if selected and fg_safe and boundary_safe and edge_safe else "0",
        "bad_frame_max_outer": f"{max_outer:.4f}",
        "bad_frame_max_hard": f"{max_hard:.8f}",
        "bad_frame_not_worse_than_v398": "1" if (
            max_outer <= V398_REFERENCE_MAX_OUTER
            and max_hard <= V398_REFERENCE_MAX_HARD
        ) else "0",
        "support_cap_saturated": "1" if bool(cap.get("selected_support_cap_saturated")) else "0",
        "over_cap_saturation": "" if over_saturation is None else f"{_float_value(over_saturation):.4f}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize v399 diagnostic sweep outputs.")
    parser.add_argument("log_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)

    writer = csv.DictWriter(sys.stdout, delimiter="\t", fieldnames=FIELDS)
    writer.writeheader()
    for log_dir in args.log_dirs:
        if not log_dir.exists():
            continue
        writer.writerow(summarize_run(log_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
