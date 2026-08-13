#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.semantic_eval_protocol import load_protocol, protocol_fingerprint
from utils.sggs_released_code_canonical import select_loso_threshold

SUBJECTS = ("377", "386", "394")
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50)
TARGET_RETENTION = 0.6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _exact_retention(rows: list[dict], baseline: str, target: float) -> dict | None:
    for row in rows:
        if row.get("baseline") == baseline and abs(float(row["retention"]) - target) < 1.0e-8:
            return row
    return None


def build_method_row(method: str, subject: str, report_dir: Path, *, target_retention: float) -> dict:
    report_dir = Path(report_dir)
    matched = [row for row in _csv_rows(report_dir / "matched_retention.csv") if row["baseline"] == "B4"]
    summaries = _csv_rows(report_dir / "baseline_summary.csv")
    summary = next(row for row in summaries if row["baseline"] == "B4")
    target = _exact_retention(matched, "B4", float(target_retention))
    row_04 = _exact_retention(matched, "B4", 0.4)
    max_retention = max((float(row["retention"]) for row in matched), default=0.0)
    return {
        "method": str(method),
        "subject": str(subject),
        "baseline_id": "B4",
        "retention_0p6_feasible": target is not None,
        "actionable_leakage_at_0p6": float(target["actionable_leakage"]) if target else "",
        "raw_leakage_at_0p6": float(target["raw_leakage"]) if target else "",
        "actionable_leakage_ratio_at_0p6": float(target["actionable_leakage_ratio"]) if target else "",
        "edit_strength_at_0p6": float(target["edit_strength"]) if target else "",
        "max_reachable_retention": float(max_retention),
        "actionable_leakage_at_0p4": float(row_04["actionable_leakage"]) if row_04 else "",
        "raw_leakage_at_0p4": float(row_04["raw_leakage"]) if row_04 else "",
        "macro_miou": float(summary["macro_miou"]),
        "mean_boundary_f1": float(summary["mean_boundary_f1"]),
        "source_dir": str(report_dir.resolve()),
    }


def _subject_paths(paper_root: Path, experiment_root: Path, subject: str) -> dict[str, Path]:
    frozen_manifest = json.loads(
        (
            paper_root
            / "exp/external/saga_canonical_five_subject_20260812_120625_bjt"
            / f"CoreView_{subject}/frozen_views/manifest.json"
        ).read_text(encoding="utf-8")
    )
    checkpoint = Path(frozen_manifest["source_checkpoint"])
    strict_root = checkpoint.parent.parent
    return {
        "protocol": paper_root / f"configs/semantic/coreview{subject}_strict_paper_protocol.json",
        "checkpoint": checkpoint,
        "config": Path(frozen_manifest["source_config"]),
        "external_bank": experiment_root / f"CoreView_{subject}/train_30k/part_label_bank.npz",
        "evidence_bank": strict_root / "banks/voting_evidence_target_support/part_label_bank.npz",
        "voting_bank": strict_root / "banks/multiview_voting/part_label_bank.npz",
        "validation_assets": strict_root / "assets/validation/test-view/semantic_editable_assets",
        "test_assets": strict_root / "assets/test/test-view/semantic_editable_assets",
    }


def _run_evaluator(
    *,
    python: Path,
    paper_root: Path,
    paths: dict[str, Path],
    subject: str,
    split: str,
    output: Path,
    threshold: float | None = None,
    frozen_config: Path | None = None,
) -> None:
    if (output / "summary.json").is_file():
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(paper_root / "tools/evaluate_semantic_editing_paper_protocol.py"),
        "--protocol", str(paths["protocol"]),
        "--protocol-split", split,
        "--raw-trained-bank", str(paths["external_bank"]),
        "--trained-bank", str(paths["evidence_bank"]),
        "--voting-bank", str(paths["voting_bank"]),
        "--checkpoint", str(paths["checkpoint"]),
        "--asset-root", str(paths[f"{split}_assets"]),
        "--config", str(paths["config"]),
        "--dataset-root", str(paper_root / "data/ZJUMoCap"),
        "--subject", f"CoreView_{subject}",
        "--output-dir", str(output),
        "--baselines", "B1", "B4",
        "--retention-reference-baseline", "B1",
    ]
    if split == "validation":
        command.extend(
            ["--soft-threshold", str(threshold), "--support-threshold", "0.1", "--boundary-radius", "6"]
        )
    else:
        command.extend(["--frozen-config", str(frozen_config)])
    with (output / "evaluation.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=paper_root, stdout=log, stderr=subprocess.STDOUT, check=True)


def _load_validation_report(report_dir: Path) -> dict:
    matched = _csv_rows(report_dir / "matched_retention.csv")
    summary = next(row for row in _csv_rows(report_dir / "baseline_summary.csv") if row["baseline"] == "B4")
    return {
        "matched_retention": matched,
        "macro_miou": float(summary["macro_miou"]),
        "mean_boundary_f1": float(summary["mean_boundary_f1"]),
        "report_dir": str(report_dir.resolve()),
    }


def _write_frozen_config(
    *,
    path: Path,
    paths: dict[str, Path],
    subject: str,
    selected: dict,
) -> None:
    protocol = load_protocol(paths["protocol"])
    payload = {
        "external_method": "SG-GS-Released-Code-Canonical (controlled-input adaptation)",
        "selection_mode": "leave_one_subject_out_sggs_released_code_canonical",
        "held_out_subject": subject,
        "donor_subjects": selected["donor_subjects"],
        "protocol_name": protocol["protocol_name"],
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "checkpoint_fingerprint": _sha256(paths["checkpoint"]),
        "bank_fingerprint": _sha256(paths["evidence_bank"]),
        "external_bank_fingerprint": _sha256(paths["external_bank"]),
        "sggs_head": "27b9ed9c9e4c5663deb169247c2339ccafe1c254",
        "selection_objective": [
            "require_every_donor_target_retention_if_feasible",
            "otherwise_max_every_donor_common_reachable_retention",
            "min_donor_mean_actionable_leakage_at_selection_retention",
            "max_donor_mean_macro_miou",
            "max_donor_mean_boundary_f1",
            "min_soft_threshold",
        ],
        "selected": {
            "soft_threshold": selected["soft_threshold"],
            "support_threshold": 0.1,
            "boundary_radius": 6,
            "validation_target_retention": TARGET_RETENTION,
            "validation_target_feasible": selected["validation_target_feasible"],
            "validation_selection_retention": selected["validation_selection_retention"],
            "validation_common_max_retention": selected["common_max_retention"],
            "mean_donor_actionable_leakage": selected["mean_actionable_leakage"],
            "mean_donor_raw_leakage": selected["mean_raw_leakage"],
            "mean_donor_macro_miou": selected["mean_macro_miou"],
            "mean_donor_boundary_f1": selected["mean_boundary_f1"],
            "donor_metrics": selected["donor_metrics"],
        },
        "candidate_trace": selected["candidate_trace"],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_aggregate(paper_root: Path, experiment_root: Path, sggs_rows: list[dict]) -> None:
    prior_path = (
        paper_root
        / "exp/external/gaussian_grouping_canonical_three_subject_20260813_0958_bjt"
        / "aggregate/gg_a5_saga_test_comparison.json"
    )
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    rows = [*prior["comparison_rows"], *sggs_rows]
    output = experiment_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "sggs_a5_saga_gaussian_grouping_test_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    a5 = {str(row["subject"]): row for row in rows if row["method"] == "A5"}
    feasible = [row for row in sggs_rows if row["retention_0p6_feasible"]]
    improvements = []
    for row in feasible:
        reference = a5[str(row["subject"])]
        external = float(row["actionable_leakage_at_0p6"])
        ours = float(reference["actionable_leakage_at_0p6"])
        improvements.append(
            {
                "subject": str(row["subject"]),
                "a5_actionable_leakage": ours,
                "sggs_actionable_leakage": external,
                "a5_relative_leakage_reduction_vs_sggs": (external - ours) / external,
            }
        )
    payload = {
        "protocol": "strict_paper_protocol_test",
        "target_retention": TARGET_RETENTION,
        "subjects": list(SUBJECTS),
        "comparison_rows": rows,
        "sggs_summary": {
            "mean_macro_miou": sum(row["macro_miou"] for row in sggs_rows) / len(sggs_rows),
            "mean_boundary_f1": sum(row["mean_boundary_f1"] for row in sggs_rows) / len(sggs_rows),
            "subjects_reaching_0p6": len(feasible),
        },
        "a5_vs_sggs": {
            "per_subject": improvements,
            "mean_relative_leakage_reduction": (
                sum(row["a5_relative_leakage_reduction_vs_sggs"] for row in improvements) / len(improvements)
                if improvements
                else None
            ),
        },
    }
    (output / "sggs_a5_saga_gaussian_grouping_test_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict LOSO evaluation for SG-GS canonical adaptation.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--paper-root", type=Path, default=Path("/remote-home/ming/3dgs-avatar-release-main"))
    parser.add_argument("--python", type=Path, default=Path("/opt/miniconda3/envs/ictrl/bin/python"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validation: dict[str, dict[float, dict]] = {}
    all_paths = {}
    for subject in SUBJECTS:
        paths = _subject_paths(args.paper_root, args.experiment_root, subject)
        all_paths[subject] = paths
        if not (args.experiment_root / f"CoreView_{subject}/train_30k/COMPLETE").is_file():
            raise ValueError(f"training is not complete for CoreView_{subject}")
        validation[subject] = {}
        for threshold in THRESHOLDS:
            token = str(threshold).replace(".", "p")
            output = args.experiment_root / f"CoreView_{subject}/evaluation/validation_threshold_{token}"
            _run_evaluator(
                python=args.python,
                paper_root=args.paper_root,
                paths=paths,
                subject=subject,
                split="validation",
                output=output,
                threshold=threshold,
            )
            validation[subject][threshold] = _load_validation_report(output)

    sggs_rows = []
    for subject in SUBJECTS:
        selected = select_loso_threshold(validation, held_out_subject=subject, target_retention=TARGET_RETENTION)
        evaluation_root = args.experiment_root / f"CoreView_{subject}/evaluation"
        frozen = evaluation_root / "frozen_sggs_loso_config.json"
        _write_frozen_config(path=frozen, paths=all_paths[subject], subject=subject, selected=selected)
        test_output = evaluation_root / "test_sggs_loso"
        _run_evaluator(
            python=args.python,
            paper_root=args.paper_root,
            paths=all_paths[subject],
            subject=subject,
            split="test",
            output=test_output,
            frozen_config=frozen,
        )
        test_summary = json.loads((test_output / "summary.json").read_text(encoding="utf-8"))
        if int(test_summary.get("processed_views", -1)) != 9:
            raise ValueError(f"strict test must process 9 views for CoreView_{subject}")
        sggs_rows.append(build_method_row("SG-GS", subject, test_output, target_retention=TARGET_RETENTION))
    _write_aggregate(args.paper_root, args.experiment_root, sggs_rows)
    print(args.experiment_root / "aggregate/sggs_a5_saga_gaussian_grouping_test_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
