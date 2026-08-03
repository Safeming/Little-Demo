#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.a7c_oracle_capacity import (
    assemble_teacher_gate_matrix,
    evaluate_camera_oracles,
    load_teacher_artifact,
    normalized_flicker,
    save_teacher_artifact,
)
from utils.part_label_bank import PART_NAMES


SUBJECTS = ("377", "386", "387", "393", "394")


def _inputs(subject: str) -> tuple[Path, Path, Path]:
    if subject == "377":
        return (
            REPO_ROOT / "exp/acceptdata/a7_dual_evidence_v5_3_canary_377/evidence/377/evidence.npz",
            REPO_ROOT / "exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_377/banks/footprint_evidence_target/part_label_bank.npz",
            REPO_ROOT / "exp/acceptdata/a7_dual_evidence_v5_4_canary_377/candidates/377/dual_evidence_camera_time_v5_4/candidate_summary.json",
        )
    root = REPO_ROOT / "exp/acceptdata/a7_pose_camera_multisubject_v1" / f"CoreView_{subject}"
    return (
        root / "evidence/evidence.npz",
        REPO_ROOT / f"exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_{subject}/banks/footprint_evidence_target/part_label_bank.npz",
        root / "capacity/capacity_summary.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_gains(base, candidate, block_count: int) -> list[float]:
    output = []
    for indices in np.array_split(np.arange(len(base)), int(block_count)):
        output.append(
            1.0
            - normalized_flicker(np.asarray(candidate)[indices])
            / max(normalized_flicker(np.asarray(base)[indices]), 1.0e-12)
        )
    return output


def evaluate_subject(
    subject: str,
    contract: dict,
    *,
    teacher_output: Path | None = None,
    contract_sha256: str = "",
) -> dict:
    evidence_path, bank_path, capacity_path = _inputs(subject)
    for path in (evidence_path, bank_path, capacity_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(evidence_path, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    with np.load(bank_path, allow_pickle=False) as bank:
        weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)
    capacity_payload = json.loads(capacity_path.read_text(encoding="utf-8"))
    capacity = capacity_payload["capacity_summary"]
    frequency = np.asarray(capacity["consensus"]["selection_frequency"])
    carrier_ids = np.flatnonzero(
        frequency >= int(contract["minimum_fold_selection_count"])
    )
    part = PART_NAMES.index(str(contract["part"]))
    part_weights = weights[:, part]
    camera_index = np.asarray(evidence["renderer_sequence_camera_index"])
    fields = {
        "target": np.asarray(evidence["renderer_target_contribution_sequence"], dtype=np.float64)[:, :, part],
        "outer": np.asarray(evidence["renderer_outer_contribution_sequence"], dtype=np.float64)[:, :, part],
        "boundary": np.asarray(evidence["renderer_boundary_contribution_sequence"], dtype=np.float64)[:, :, part],
    }
    totals = {key: value @ part_weights for key, value in fields.items()}
    point = {
        key: value[:, carrier_ids] * part_weights[carrier_ids][None, :]
        for key, value in fields.items()
    }
    method_rows = {method: [] for method in ("global", "point", "ray")}
    block_rows = {method: {"outer": [], "boundary": []} for method in method_rows}
    point_gates_by_camera = {}
    for camera in np.unique(camera_index):
        selected = camera_index == camera
        result = evaluate_camera_oracles(
            target=totals["target"][selected],
            outer=totals["outer"][selected],
            boundary=totals["boundary"][selected],
            point_target=point["target"][selected],
            point_outer=point["outer"][selected],
            point_boundary=point["boundary"][selected],
            minimum_gate=float(contract["minimum_gate"]),
            minimum_target_response=float(contract["minimum_target_response"]),
            knot_count=int(contract["gate_knot_count"]),
            temporal_block_count=int(contract["temporal_block_count"]),
        )
        for method, values in result.items():
            gates = np.asarray(values["gates"])
            method_rows[method].append(
                {
                    "camera": int(camera),
                    "outer_gain": float(values["outer_gain"]),
                    "boundary_gain": float(values["boundary_gain"]),
                    "minimum_target_response": float(values["minimum_target_response"]),
                    "maximum_adjacent_gate_change": float(np.max(np.abs(np.diff(gates, axis=0)))),
                }
            )
            for signal in ("outer", "boundary"):
                block_rows[method][signal].extend(
                    _block_gains(
                        totals[signal][selected],
                        values[signal],
                        int(contract["temporal_block_count"]),
                    )
                )
        point_gates_by_camera[int(camera)] = np.asarray(result["point"]["gates"])
    methods = {}
    for method, rows in method_rows.items():
        summary = {
            "outer_gain": float(np.mean([row["outer_gain"] for row in rows])),
            "boundary_gain": float(np.mean([row["boundary_gain"] for row in rows])),
            "minimum_target_response": float(min(row["minimum_target_response"] for row in rows)),
            "maximum_adjacent_gate_change": float(max(row["maximum_adjacent_gate_change"] for row in rows)),
            "cameras": rows,
        }
        for signal in ("outer", "boundary"):
            gains = np.asarray(block_rows[method][signal], dtype=np.float64)
            summary[f"{signal}_positive_block_fraction"] = float(np.mean(gains > 0.0))
            summary[f"{signal}_block_gain_quantile"] = float(
                np.quantile(gains, float(contract["block_gain_quantile"]))
            )
            summary[f"{signal}_worst_block_gain"] = float(np.min(gains))
        summary["promotion_passed"] = bool(
            summary["outer_gain"] >= float(contract["minimum_oracle_outer_gain"])
            and summary["boundary_gain"] >= float(contract["minimum_oracle_boundary_gain"])
            and summary["minimum_target_response"] >= float(contract["minimum_target_response"]) - 1e-7
            and summary["maximum_adjacent_gate_change"] <= float(contract["maximum_adjacent_gate_change"]) + 1e-7
            and all(
                summary[f"{signal}_positive_block_fraction"] >= float(contract["minimum_positive_block_fraction"])
                and summary[f"{signal}_block_gain_quantile"] >= float(contract["minimum_block_gain_quantile"])
                and summary[f"{signal}_worst_block_gain"] >= -float(contract["maximum_worst_block_regression"])
                for signal in ("outer", "boundary")
            )
        )
        methods[method] = summary
    fingerprints = {
        "evidence_sha256": _sha256(evidence_path),
        "a5_bank_sha256": _sha256(bank_path),
        "capacity_sha256": _sha256(capacity_path),
    }
    if teacher_output is not None:
        teacher_gates = assemble_teacher_gate_matrix(
            camera_index, point_gates_by_camera
        )
        sources = {
            "evidence": fingerprints["evidence_sha256"],
            "a5_bank": fingerprints["a5_bank_sha256"],
            "capacity": fingerprints["capacity_sha256"],
            "contract": contract_sha256,
        }
        if teacher_output.exists():
            existing = load_teacher_artifact(teacher_output)
            if not np.array_equal(existing["carrier_ids"], carrier_ids):
                raise ValueError("existing teacher carrier IDs differ")
            for name, value in sources.items():
                if str(existing[f"source_{name}_fingerprint"]) != value:
                    raise ValueError(f"existing teacher {name} fingerprint differs")
        save_teacher_artifact(
            teacher_output,
            gates=teacher_gates,
            carrier_ids=carrier_ids,
            camera_index=camera_index,
            frame_index=np.asarray(evidence["renderer_sequence_frame_index"]),
            minimum_gate=float(contract["minimum_gate"]),
            maximum_gate=float(contract["maximum_gate"]),
            source_fingerprints=sources,
        )
    return {
        "subject": subject,
        "carrier_count": int(len(carrier_ids)),
        "methods": methods,
        "input_fingerprints": fingerprints,
        "paper_test_eligible": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate A7c oracle capacity.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS)
    parser.add_argument("--teacher-output", type=Path)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    requested_subjects = tuple(args.subjects or contract["subjects"])
    if args.teacher_output is not None and len(requested_subjects) != 1:
        parser.error("--teacher-output requires exactly one subject")
    args.output_root.mkdir(parents=True, exist_ok=True)
    subjects = []
    for subject in requested_subjects:
        result = evaluate_subject(
            str(subject),
            contract,
            teacher_output=args.teacher_output,
            contract_sha256=_sha256(args.contract),
        )
        subjects.append(result)
        path = args.output_root / f"CoreView_{subject}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[A7c oracle] subject={subject} complete", flush=True)
    summary = {
        "schema_version": 1,
        "oracle_id": contract["oracle_id"],
        "subject_count": len(subjects),
        "methods": {},
        "subjects": subjects,
        "paper_test_eligible": False,
    }
    for method in ("global", "point", "ray"):
        rows = [row["methods"][method] for row in subjects]
        summary["methods"][method] = {
            "promotion_subject_count": sum(row["promotion_passed"] for row in rows),
            "mean_outer_gain": float(np.mean([row["outer_gain"] for row in rows])),
            "mean_boundary_gain": float(np.mean([row["boundary_gain"] for row in rows])),
            "route_supported": bool(sum(row["promotion_passed"] for row in rows) >= 4),
        }
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    summary["output_fingerprint"] = hashlib.sha256(encoded.encode()).hexdigest()
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "method", "outer_gain", "boundary_gain", "promotion_passed"])
        writer.writeheader()
        for subject in subjects:
            for method, values in subject["methods"].items():
                writer.writerow({"subject": subject["subject"], "method": method, "outer_gain": values["outer_gain"], "boundary_gain": values["boundary_gain"], "promotion_passed": values["promotion_passed"]})
    print(json.dumps(summary["methods"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
