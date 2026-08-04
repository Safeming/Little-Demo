#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_a7c_r1_2a_quotient_compositor import summarize_records
from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams,
    _load_probe,
    _load_teacher_manifest,
    verify_source_file,
)
from utils.a7c_feasibility_oracle import (
    bisect_feasible_gain,
    classify_oracle,
    solve_fixed_gain_oracle,
)
from utils.a7c_quotient_compositor import runtime_target_mass
from utils.a7c_renderer_compositor import build_canary_splits
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def _infeasible(error: RuntimeError) -> bool:
    return "infeasible" in str(error).lower()


def _solve_or_none(kwargs):
    try:
        return solve_fixed_gain_oracle(**kwargs)
    except RuntimeError as error:
        if _infeasible(error):
            return None
        raise


def _capacity_search(base_kwargs, mode: str, tolerance: float) -> dict:
    feasible_results = {}

    def solve_gain(gain: float):
        if mode == "balanced":
            outer_gain, boundary_gain = gain, gain
        elif mode == "boundary_conditioned":
            outer_gain, boundary_gain = 0.005, gain
        elif mode == "independent_outer":
            outer_gain, boundary_gain = gain, None
        elif mode == "independent_boundary":
            outer_gain, boundary_gain = None, gain
        else:
            raise ValueError(f"unknown capacity mode {mode}")
        result = _solve_or_none(
            {
                **base_kwargs,
                "minimum_outer_gain": outer_gain,
                "minimum_boundary_gain": boundary_gain,
            }
        )
        if result is not None:
            feasible_results[float(gain)] = result
        return result

    lower_probe = solve_gain(-0.01)
    if lower_probe is None:
        return {
            "status": "conditioning_infeasible",
            "feasible_lower": None,
            "infeasible_upper": -0.01,
            "interval_width": None,
            "iterations": 0,
            "feasible_endpoint_metrics": None,
            "feasible_endpoint_certificate": None,
        }
    interval = bisect_feasible_gain(
        lambda gain: solve_gain(gain) is not None,
        lower=-0.01,
        upper=1.00001,
        tolerance=float(tolerance),
    )
    endpoint = feasible_results.get(float(interval["feasible_lower"]))
    if endpoint is None:
        endpoint = solve_gain(float(interval["feasible_lower"]))
    return {
        "status": "bracketed",
        **interval,
        "feasible_endpoint_metrics": endpoint["metrics"],
        "feasible_endpoint_certificate": endpoint["certificate"],
    }


def evaluate_record_oracle(*, base_kwargs, contract, fold: int, camera: int) -> dict:
    tolerance = float(contract["oracle_bisection_tolerance"])
    searches = {
        "balanced": _capacity_search(base_kwargs, "balanced", tolerance),
        "boundary_conditioned": _capacity_search(
            base_kwargs, "boundary_conditioned", tolerance
        ),
        "independent_outer": _capacity_search(
            base_kwargs, "independent_outer", tolerance
        ),
        "independent_boundary": _capacity_search(
            base_kwargs, "independent_boundary", tolerance
        ),
    }
    witness = _solve_or_none(
        {
            **base_kwargs,
            "minimum_outer_gain": float(contract["minimum_outer_gain"]),
            "minimum_boundary_gain": float(
                contract["oracle_boundary_witness_gain"]
            ),
        }
    )
    return {
        "fold": int(fold),
        "camera_index": int(camera),
        **searches,
        "sufficient_witness": None
        if witness is None
        else {
            "metrics": witness["metrics"],
            "certificate": witness["certificate"],
        },
    }


def _gain_summary(records, contract, source: str) -> dict:
    rows = []
    for record in records:
        if source == "optimistic":
            outer_gain = float(record["independent_outer"]["infeasible_upper"])
            boundary_gain = float(
                record["independent_boundary"]["infeasible_upper"]
            )
            row = {
                "outer_gain": outer_gain,
                "boundary_gain": boundary_gain,
                "minimum_target_response": float(
                    contract["minimum_target_response"]
                ),
                "maximum_soft_iou_drop": float(
                    contract["maximum_selection_soft_iou_drop"]
                ),
                "maximum_adjacent_gate_change": float(
                    contract["maximum_projection_gate_jump"]
                ),
            }
        elif source == "sufficient_witness":
            witness = record["sufficient_witness"]
            if witness is None:
                return None
            row = dict(witness["metrics"])
        else:
            raise ValueError(f"unknown gain summary source {source}")
        rows.append(row)
    return summarize_records(rows, contract)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate R1.3-P renderer-constrained gate capacity."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _run(args) -> dict:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_specs = (
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.evidence, contract["source_evidence_sha256"], "evidence"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
        (
            REPO_ROOT / contract["source_r1_3p_design"],
            contract["source_r1_3p_design_sha256"],
            "R1.3-P design",
        ),
    )
    source_fingerprints = {
        name: verify_source_file(path, expected, name)
        for path, expected, name in source_specs
    }
    probe = _load_probe(args.probe)
    teacher = _load_teacher_manifest(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"probe and teacher {key} differ")
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    camera_index = np.asarray(teacher["camera_index"])
    frame_index = np.asarray(teacher["frame_index"])
    if not np.array_equal(
        evidence["renderer_sequence_camera_index"], camera_index
    ) or not np.array_equal(
        evidence["renderer_sequence_frame_index"], frame_index
    ):
        raise ValueError("source manifests differ")

    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    all_weights = np.asarray(bank["soft_edit_weights"], dtype=np.float64)[
        :, part_index
    ]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    a5_weight = all_weights[carrier_ids]
    streams = _build_streams(evidence, all_weights, carrier_ids, part_index)
    features = np.asarray(probe["features"], dtype=np.float32)
    feature_names = list(map(str, probe["feature_names"]))
    field = {
        name: features[:, :, feature_names.index(name)]
        for name in (
            "alpha_transmittance_mass",
            "semantic_support_mean",
            "alpha_mean",
        )
    }
    runtime_mass = runtime_target_mass(
        alpha_transmittance_mass=torch.from_numpy(
            field["alpha_transmittance_mass"]
        ),
        a5_weight=torch.from_numpy(a5_weight.astype(np.float32)),
        semantic_support_mean=torch.from_numpy(field["semantic_support_mean"]),
        alpha_mean=torch.from_numpy(field["alpha_mean"]),
    ).numpy()
    split = build_canary_splits(
        camera_index=camera_index,
        frame_index=frame_index,
        fit_camera_indices=(0, 1, 2, 3),
        audit_camera_indices=(4, 5, 6, 7),
        block_count=int(contract["temporal_block_count"]),
    )

    records = []
    for fold, held in enumerate(split["held_block_masks"]):
        for camera in range(4):
            selected = np.asarray(
                held & split["fit_mask"] & (camera_index == camera), dtype=bool
            )
            indices = np.flatnonzero(selected)
            if indices.size < 2 or not np.all(
                np.diff(frame_index[indices]) == int(contract["frame_stride"])
            ):
                raise ValueError("oracle held segment is not contiguous")
            record_streams = {
                "objective": {
                    signal: {
                        key: np.asarray(value)[selected]
                        for key, value in streams["objective"][signal].items()
                    }
                    for signal in ("outer", "boundary")
                },
                "guard": {
                    signal: {
                        key: np.asarray(value)[selected]
                        for key, value in streams["guard"][signal].items()
                    }
                    for signal in ("target", "outer")
                },
            }
            common = {
                "runtime_mass": runtime_mass[selected],
                "a5_weight": a5_weight,
                "streams": record_streams,
                "minimum_gate": float(contract["minimum_gate"]),
                "maximum_gate": float(contract["maximum_gate"]),
                "selection_threshold": float(contract["selection_threshold"]),
                "proxy_target_response": float(contract["proxy_target_response"]),
                "maximum_gate_jump": float(
                    contract["maximum_projection_gate_jump"]
                ),
                "minimum_target_response": float(
                    contract["minimum_target_response"]
                ),
                "maximum_soft_iou_drop": float(
                    contract["maximum_selection_soft_iou_drop"]
                ),
                "primal_tolerance": float(contract["solver_primal_tolerance"]),
                "residual_tolerance": float(
                    contract["solver_residual_tolerance"]
                ),
            }
            records.append(
                evaluate_record_oracle(
                    base_kwargs=common,
                    contract=contract,
                    fold=fold,
                    camera=camera,
                )
            )
            print(
                json.dumps(
                    {
                        "oracle_record_complete": len(records),
                        "fold": fold,
                        "camera_index": camera,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(records) != 24:
        raise RuntimeError("oracle requires exactly 24 held records")
    optimistic_summary = _gain_summary(records, contract, "optimistic")
    sufficient_witness_summary = _gain_summary(
        records, contract, "sufficient_witness"
    )
    sufficient_passed = bool(
        sufficient_witness_summary is not None
        and sufficient_witness_summary["passed"]
    )
    verdict = classify_oracle(
        sufficient_audit_passed=sufficient_passed,
        optimistic_summary=optimistic_summary,
        contract=contract,
    )
    summary = {
        "experiment_id": contract["experiment_id"],
        "execution_status": "COMPLETED",
        "verdict": verdict,
        "record_count": len(records),
        "optimistic_summary": optimistic_summary,
        "sufficient_witness_summary": sufficient_witness_summary,
        "source_fingerprints": source_fingerprints,
        "paper_test_eligible": False,
    }
    return {"records": records, "summary": summary}


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run(args)
    except Exception as error:
        summary = {
            "execution_status": "ORACLE_ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
            "paper_test_eligible": False,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    (args.output_dir / "records.json").write_text(
        json.dumps(
            {
                "records": result["records"],
                "paper_test_eligible": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
