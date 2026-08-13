#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUBJECTS = ("377", "386", "394")
PARTS = ("hair", "shoes")
METHODS = ("saga", "a5")
FIXED_VIEW = "c22_f000420"
TEST_VIEWS = (
    "c21_f000180",
    "c21_f000420",
    "c21_f000540",
    "c22_f000180",
    "c22_f000420",
    "c22_f000540",
    "c23_f000180",
    "c23_f000420",
    "c23_f000540",
)
COLUMNS = (
    ("Input", "input", "input"),
    ("Hair / SAGA", "saga", "hair"),
    ("Hair / Ours", "a5", "hair"),
    ("Shoes / SAGA", "saga", "shoes"),
    ("Shoes / Ours", "a5", "shoes"),
)
COARSE_STRENGTHS = tuple(round(index / 10.0, 2) for index in range(1, 11))
TARGET_RETENTION = 0.6
RETENTION_TOLERANCE = 0.03
OPERATING_POINT_DEFINITION = "main_table_point_activation_retention"


def build_render_command(
    spec: Mapping,
    *,
    output_dir: Path | str,
    python_bin: Path | str,
    methods: Sequence[str],
    strengths: Sequence[float] | None = None,
    method_part_strengths: Path | str | None = None,
    a5_threshold: float | None = None,
    metrics_only: bool = False,
) -> list[str]:
    command = [
        str(python_bin),
        "tools/render_semantic_real_editing_paper_suite.py",
        "--subject",
        str(spec["subject"]),
        "--raw-bank",
        str(spec["raw_bank"]),
        "--voting-bank",
        str(spec["voting_bank"]),
        "--a5-bank",
        str(spec["a5_bank"]),
        "--saga-bank",
        str(spec["saga_bank"]),
        "--saga-threshold",
        "0.5",
        "--loso-config",
        str(spec["loso_config"]),
        "--method-freeze",
        str(spec["method_freeze"]),
        "--checkpoint",
        str(spec["checkpoint"]),
        "--asset-root",
        str(spec["asset_root"]),
        "--output-dir",
        str(output_dir),
        "--views",
        *TEST_VIEWS,
        "--methods",
        *[str(value) for value in methods],
        "--tasks",
        "recolor",
        "--parts",
        *PARTS,
        "--explicit-binding-render-preset",
        "none",
    ]
    if strengths is not None:
        command.extend(["--edit-strengths", *[str(float(value)) for value in strengths]])
    if method_part_strengths is not None:
        command.extend(["--method-part-strengths", str(method_part_strengths)])
    if a5_threshold is not None:
        command.extend(["--a5-threshold", str(float(a5_threshold))])
    if metrics_only:
        command.append("--metrics-only")
    return command


def build_subject_specs(repo_root: Path | str, output_root: Path | str) -> list[dict]:
    repo_root = Path(repo_root).resolve()
    output_root = Path(output_root).resolve()
    source_names = {
        "377": "coreview377_multisubject_strict_20260721",
        "386": "coreview386_multisubject_strict_20260719",
        "394": "coreview394_multisubject_strict_20260722",
    }
    saga_root = repo_root / "exp/external/saga_canonical_five_subject_20260812_120625_bjt"
    specs = []
    for subject in SUBJECTS:
        source = repo_root / "exp/acceptdata" / source_names[subject]
        specs.append(
            {
                "subject": subject,
                "source_root": source,
                "checkpoint": source / "base_train_40k/ckpt40000.pth",
                "asset_root": source / "assets/test/test-view/semantic_editable_assets",
                "raw_bank": source / "banks/raw_trained/part_label_bank.npz",
                "voting_bank": source / "banks/multiview_voting/part_label_bank.npz",
                "a5_bank": repo_root
                / f"exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_{subject}/banks/footprint_evidence_target/part_label_bank.npz",
                "saga_bank": saga_root / f"CoreView_{subject}/train_30k/part_label_bank.npz",
                "saga_operating_points": saga_root
                / f"CoreView_{subject}/evaluation/test_saga_readouts/matched_retention.csv",
                "a5_operating_points": repo_root
                / f"exp/acceptdata/frozen_a5_five_subject_main_20260723/CoreView_{subject}/main/matched_retention.csv",
                "loso_config": repo_root
                / f"exp/acceptdata/frozen_a5_five_subject_loso_stats_20260723/CoreView_{subject}/loso_frozen_config.json",
                "method_freeze": repo_root / "configs/semantic/frozen_a5_main_method_v1.json",
                "output_dir": output_root / "subjects" / f"CoreView_{subject}",
            }
        )
    return specs


def build_dry_run_manifest(specs: Sequence[Mapping], *, python_bin: Path | str) -> list[dict]:
    manifest = []
    for spec in specs:
        strength_path = Path(spec["output_dir"]) / "method_part_strengths.json"
        manifest.append(
            {
                "subject": str(spec["subject"]),
                "stages": ["frozen_operating_point", "final_render"],
                "fixed_view": FIXED_VIEW,
                "test_views": list(TEST_VIEWS),
                "saga_operating_points": str(spec["saga_operating_points"]),
                "a5_operating_points": str(spec["a5_operating_points"]),
                "final_command_template": {
                    "output_dir": str(Path(spec["output_dir"]) / "final_render"),
                    "method_part_strengths": str(strength_path),
                    "a5_threshold_source": str(spec["a5_operating_points"]),
                },
            }
        )
    return manifest


def _read_csv(path: Path | str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path | str, rows: Sequence[Mapping]) -> None:
    path = Path(path)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(command: Sequence[str], *, log_path: Path, gpu: str, repo_root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [str(value) for value in command],
            cwd=repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _validate_inputs(specs: Sequence[Mapping]) -> None:
    for spec in specs:
        for key in (
            "checkpoint",
            "asset_root",
            "raw_bank",
            "voting_bank",
            "a5_bank",
            "saga_bank",
            "saga_operating_points",
            "a5_operating_points",
            "loso_config",
            "method_freeze",
        ):
            path = Path(spec[key])
            if not path.exists():
                raise FileNotFoundError(f"missing CoreView_{spec['subject']} input {key}: {path}")


def _fine_strengths(selections: Sequence[Mapping]) -> tuple[float, ...]:
    values = set()
    for selection in selections:
        center = float(selection["selected_strength"])
        lower = max(0.02, center - 0.1)
        upper = min(1.0, center + 0.1)
        count = int(round((upper - lower) / 0.02))
        values.update(round(lower + index * 0.02, 2) for index in range(count + 1))
        values.add(round(upper, 2))
    return tuple(sorted(value for value in values if 0.0 < value <= 1.0))


def _selection_rows(rows: Sequence[Mapping], *, retention: float) -> list[dict]:
    return [
        select_matched_strength(
            rows,
            method=method,
            part=part,
            reference_method="voting",
            retention=retention,
        )
        for method in METHODS
        for part in PARTS
    ]


def _frame_manifest(subject: str, final_dir: Path, metric_rows: Sequence[Mapping]) -> list[dict]:
    rows = []
    for view in TEST_VIEWS:
        input_path = final_dir / "frames" / f"{view}_rgb.png"
        rows.append(
            {
                "subject": str(subject),
                "view": view,
                "method": "input",
                "part": "input",
                "frame": str(input_path.resolve()),
            }
        )
    for row in metric_rows:
        rows.append(
            {
                "subject": str(subject),
                "view": str(row["view"]),
                "method": str(row["method"]),
                "part": str(row["part"]),
                "frame": str(Path(str(row["frame"])).resolve()),
            }
        )
    return rows


def _actual_retention_rows(
    final_rows: Sequence[Mapping],
    coarse_rows: Sequence[Mapping],
    selections: Sequence[Mapping],
) -> list[dict]:
    selection_lookup = {(row["method"], row["part"]): row for row in selections}
    outputs = []
    for method in METHODS:
        for part in PARTS:
            selected = [row for row in final_rows if row["method"] == method and row["part"] == part]
            target_response = sum(_float(row, "target_delta_sum") for row in selected)
            reference = select_matched_strength(
                coarse_rows,
                method="voting",
                part=part,
                reference_method="voting",
                retention=1.0,
            )["reference_target_response"]
            source = selection_lookup[(method, part)]
            outputs.append(
                {
                    "method": method,
                    "part": part,
                    "selected_strength": float(source["selected_strength"]),
                    "target_retention": TARGET_RETENTION,
                    "actual_retention": float(target_response / reference),
                    "reference_target_response": float(reference),
                    "selected_target_response": float(target_response),
                }
            )
    return outputs


def _subject_complete(path: Path) -> bool:
    final_metrics = path / "final_render/metrics.csv"
    return (
        (path / "COMPLETE").exists()
        and final_metrics.exists()
        and len(_read_csv(final_metrics)) == len(TEST_VIEWS) * len(METHODS) * len(PARTS)
    )


def resolve_frozen_operating_point(
    path: Path | str,
    *,
    baseline: str,
    retention: float = TARGET_RETENTION,
    expected_threshold: float | None = None,
) -> dict:
    path = Path(path)
    rows = _read_csv(path)
    matches = [
        row
        for row in rows
        if str(row.get("baseline", "")) == str(baseline)
        and abs(float(row.get("retention", -1.0)) - float(retention)) <= 1.0e-9
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {baseline}@{float(retention):.2f} row in {path}, found {len(matches)}"
        )
    row = dict(matches[0])
    if str(row.get("reference_baseline", "")) != "B1":
        raise ValueError(f"{path} {baseline}@{retention} must use reference_baseline=B1")
    strength = float(row["edit_strength"])
    if not np.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise ValueError(f"{path} contains invalid edit_strength={strength}")
    if expected_threshold is not None:
        threshold = float(row["threshold"])
        if abs(threshold - float(expected_threshold)) > 1.0e-9:
            raise ValueError(
                f"{path} {baseline}@{retention} threshold {threshold} != {float(expected_threshold)}"
            )
    row["edit_strength"] = strength
    row["retention"] = float(retention)
    row["source_csv"] = str(path.resolve())
    row["source_csv_sha256"] = _sha256(path)
    return row


def build_main_table_strength_mapping(
    saga_operating_points: Path | str,
    a5_operating_points: Path | str,
) -> tuple[dict, dict]:
    saga = resolve_frozen_operating_point(
        saga_operating_points,
        baseline="B4",
        retention=TARGET_RETENTION,
        expected_threshold=0.5,
    )
    a5 = resolve_frozen_operating_point(
        a5_operating_points,
        baseline="A5",
        retention=TARGET_RETENTION,
    )
    mapping = {
        "saga": {part: float(saga["edit_strength"]) for part in PARTS},
        "a5": {part: float(a5["edit_strength"]) for part in PARTS},
    }
    provenance = {
        "operating_point_definition": OPERATING_POINT_DEFINITION,
        "target_retention": TARGET_RETENTION,
        "reference_baseline": "B1",
        "saga": saga,
        "a5": a5,
    }
    return mapping, provenance


def run_subject(
    spec: Mapping,
    *,
    repo_root: Path,
    python_bin: Path,
    gpu: str,
) -> dict:
    subject = str(spec["subject"])
    subject_root = Path(spec["output_dir"])
    if _subject_complete(subject_root):
        return json.loads((subject_root / "summary.json").read_text(encoding="utf-8"))
    subject_root.mkdir(parents=True, exist_ok=True)
    strength_mapping, operating_point_provenance = build_main_table_strength_mapping(
        spec["saga_operating_points"],
        spec["a5_operating_points"],
    )
    strength_path = subject_root / "method_part_strengths.json"
    _write_json(strength_path, strength_mapping)
    _write_json(subject_root / "operating_point_provenance.json", operating_point_provenance)

    final_dir = subject_root / "final_render"
    final_command = build_render_command(
        spec,
        output_dir=final_dir,
        python_bin=python_bin,
        methods=METHODS,
        method_part_strengths=strength_path,
        a5_threshold=float(operating_point_provenance["a5"]["threshold"]),
    )
    _run_command(final_command, log_path=subject_root / "final_render.log", gpu=gpu, repo_root=repo_root)
    final_rows = _read_csv(final_dir / "metrics.csv")
    expected_rows = len(TEST_VIEWS) * len(METHODS) * len(PARTS)
    if len(final_rows) != expected_rows:
        raise ValueError(f"CoreView_{subject} final metrics rows {len(final_rows)} != {expected_rows}")
    ranking = rank_objective_views(final_rows)
    if not ranking:
        raise ValueError(f"CoreView_{subject} has no eligible objectively selected view")
    _write_csv(subject_root / "selection_ranking.csv", ranking)
    rgb_diagnostics = []
    for method in METHODS:
        for part in PARTS:
            selected = [row for row in final_rows if row["method"] == method and row["part"] == part]
            target = sum(_float(row, "target_delta_sum") for row in selected)
            outer = sum(_float(row, "outer_delta_sum") for row in selected)
            rgb_diagnostics.append(
                {
                    "method": method,
                    "part": part,
                    "main_table_retention": TARGET_RETENTION,
                    "main_table_edit_strength": strength_mapping[method][part],
                    "rgb_target_delta_sum": target,
                    "rgb_outer_delta_sum": outer,
                    "rgb_outer_to_target_ratio": outer / max(target, 1.0e-8),
                }
            )
    _write_csv(subject_root / "rgb_response_diagnostics.csv", rgb_diagnostics)
    manifest = _frame_manifest(subject, final_dir, final_rows)
    _write_csv(subject_root / "frame_manifest.csv", manifest)
    summary = {
        "subject": subject,
        "checkpoint": str(Path(spec["checkpoint"]).resolve()),
        "checkpoint_sha256": _sha256(spec["checkpoint"]),
        "saga_bank": str(Path(spec["saga_bank"]).resolve()),
        "saga_bank_sha256": _sha256(spec["saga_bank"]),
        "a5_bank": str(Path(spec["a5_bank"]).resolve()),
        "a5_bank_sha256": _sha256(spec["a5_bank"]),
        "target_retention": TARGET_RETENTION,
        "operating_point_definition": OPERATING_POINT_DEFINITION,
        "operating_point_provenance": operating_point_provenance,
        "strengths": strength_mapping,
        "rgb_response_diagnostics": rgb_diagnostics,
        "objectively_selected_view": ranking[0]["view"],
        "fixed_view": FIXED_VIEW,
        "test_views": list(TEST_VIEWS),
        "uses_test_parser_for_edit_selection": False,
        "uses_test_parser_for_strength_matching": False,
        "uses_test_parser_for_rgb_diagnostics_and_view_ranking": True,
        "uses_post_render_mask_composite": False,
        "final_metric_row_count": len(final_rows),
    }
    _write_json(subject_root / "summary.json", summary)
    (subject_root / "COMPLETE").write_text("complete\n", encoding="ascii")
    return summary


def _copy_selected_frames(frame_rows: Sequence[Mapping], target_dir: Path, view_by_subject: Mapping[str, str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in frame_rows:
        subject = str(row["subject"])
        if str(row["view"]) != str(view_by_subject[subject]):
            continue
        source = Path(str(row["frame"]))
        name = f"CoreView_{subject}_{row['view']}_{row['method']}_{row['part']}.png"
        Image.open(source).convert("RGB").save(target_dir / name)


def assemble_outputs(output_root: Path, subject_summaries: Sequence[Mapping]) -> dict:
    frame_rows = []
    for subject in SUBJECTS:
        frame_rows.extend(_read_csv(output_root / "subjects" / f"CoreView_{subject}" / "frame_manifest.csv"))
    fixed_views = {subject: FIXED_VIEW for subject in SUBJECTS}
    selected_views = {
        str(summary["subject"]): str(summary["objectively_selected_view"])
        for summary in subject_summaries
    }
    fixed_dir = output_root / "fixed_main"
    selected_dir = output_root / "objectively_selected"
    fixed_layout = compose_b_layout(
        frame_rows,
        fixed_dir / "saga_a5_same_view_fixed.png",
        fixed_dir / "saga_a5_same_view_fixed.pdf",
        fixed_view=FIXED_VIEW,
    )
    selected_layout = compose_b_layout(
        frame_rows,
        selected_dir / "saga_a5_same_view_objectively_selected.png",
        selected_dir / "saga_a5_same_view_objectively_selected.pdf",
        view_by_subject=selected_views,
    )
    _copy_selected_frames(frame_rows, fixed_dir / "frames", fixed_views)
    _copy_selected_frames(frame_rows, selected_dir / "frames", selected_views)
    ranking_rows = []
    for subject in SUBJECTS:
        for row in _read_csv(output_root / "subjects" / f"CoreView_{subject}" / "selection_ranking.csv"):
            ranking_rows.append({"subject": subject, **row})
    _write_csv(selected_dir / "selection_ranking.csv", ranking_rows)
    summary = {
        "subjects": list(SUBJECTS),
        "parts": list(PARTS),
        "methods": list(METHODS),
        "target_retention": TARGET_RETENTION,
        "operating_point_definition": OPERATING_POINT_DEFINITION,
        "fixed_view": FIXED_VIEW,
        "objectively_selected_views": selected_views,
        "fixed_layout": fixed_layout,
        "objectively_selected_layout": selected_layout,
        "subject_summaries": list(subject_summaries),
        "uses_test_parser_for_edit_selection": False,
        "uses_test_parser_for_strength_matching": False,
        "uses_test_parser_for_rgb_diagnostics_and_view_ranking": True,
        "uses_post_render_mask_composite": False,
    }
    _write_json(output_root / "summary.json", summary)
    readme = f"""# SAGA 与 A5 同视角论文可视化

主文固定图：`fixed_main/saga_a5_same_view_fixed.png`

客观精选图：`objectively_selected/saga_a5_same_view_objectively_selected.png`

- 对象顺序：377、386、394
- 固定视角：{FIXED_VIEW}
- 部位：hair、shoes
- 方法：SAGA-Canonical B4、A5
- 公平口径：直接复用定量主表相对于 B1 的 60% Gaussian 点激活匹配操作点
- Hair 与 Shoes 对同一对象、同一方法共用主表强度；60% 不表示每个部位的 RGB 响应为 60%
- 测试 parser 仅用于 RGB 编辑诊断和客观视角排序，不参与强度选择、Gaussian 点选择或图像合成
- 客观精选视角：{json.dumps(selected_views, sort_keys=True)}

客观精选图必须在论文图注中标注为按预先声明的低泄漏优先规则选择，不能替代固定视角主结果。
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    (output_root / "COMPLETE").write_text("complete\n", encoding="ascii")
    return summary


def verify_output(output_root: Path | str) -> dict:
    output_root = Path(output_root)
    if not (output_root / "COMPLETE").exists():
        raise ValueError(f"missing COMPLETE marker: {output_root / 'COMPLETE'}")
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"missing summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("subjects") != list(SUBJECTS):
        raise ValueError("summary subject order does not match frozen protocol")
    for subject in SUBJECTS:
        subject_root = output_root / "subjects" / f"CoreView_{subject}"
        if not _subject_complete(subject_root):
            raise ValueError(f"CoreView_{subject} output is incomplete")
        provenance = json.loads(
            (subject_root / "operating_point_provenance.json").read_text(encoding="utf-8")
        )
        if provenance.get("operating_point_definition") != OPERATING_POINT_DEFINITION:
            raise ValueError(f"CoreView_{subject} operating point definition is invalid")
        strengths = json.loads((subject_root / "method_part_strengths.json").read_text(encoding="utf-8"))
        for method in METHODS:
            expected = float(provenance[method]["edit_strength"])
            if any(abs(float(strengths[method][part]) - expected) > 1.0e-12 for part in PARTS):
                raise ValueError(f"CoreView_{subject} {method} strength does not match main table")
        ranking = _read_csv(subject_root / "selection_ranking.csv")
        if not ranking or int(ranking[0]["rank"]) != 1:
            raise ValueError(f"CoreView_{subject} objective ranking is invalid")
        if ranking[0]["view"] != summary["objectively_selected_views"][subject]:
            raise ValueError(f"CoreView_{subject} selected view does not match rank 1")
    for relative in (
        "fixed_main/saga_a5_same_view_fixed.png",
        "fixed_main/saga_a5_same_view_fixed.pdf",
        "objectively_selected/saga_a5_same_view_objectively_selected.png",
        "objectively_selected/saga_a5_same_view_objectively_selected.pdf",
    ):
        path = output_root / relative
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty paper figure: {path}")
        if path.suffix == ".png":
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"))
                if image.width <= image.height or np.std(array) < 1.0:
                    raise ValueError(f"invalid or blank paper figure: {path}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched-retention SAGA/A5 same-view paper figures.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("exp/acceptdata/saga_a5_same_view_paper_visual_20260812"),
    )
    parser.add_argument("--python-bin", type=Path, default=Path("/opt/miniconda3/envs/ictrl/bin/python"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    if args.verify_only:
        verify_output(output_root)
        print(f"verified {output_root}")
        return 0
    specs = build_subject_specs(repo_root, output_root)
    _validate_inputs(specs)
    manifest = build_dry_run_manifest(specs, python_bin=args.python_bin)
    _write_json(output_root / "run_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    summaries = []
    for spec in specs:
        print(f"CoreView_{spec['subject']} started", flush=True)
        summaries.append(
            run_subject(
                spec,
                repo_root=repo_root,
                python_bin=args.python_bin,
                gpu=str(args.gpu),
            )
        )
        print(f"CoreView_{spec['subject']} completed", flush=True)
    assemble_outputs(output_root, summaries)
    verify_output(output_root)
    print(f"wrote and verified {output_root}")
    return 0


def _float(row: Mapping, key: str) -> float:
    return float(row[key])


def select_matched_strength(
    rows: Iterable[Mapping],
    *,
    method: str,
    part: str,
    reference_method: str = "voting",
    retention: float = 0.6,
) -> dict:
    unique = {}
    for row in rows:
        if str(row["part"]) != str(part):
            continue
        key = (
            str(row["method"]),
            str(row["part"]),
            _float(row, "edit_strength"),
            str(row["view"]),
        )
        unique[key] = row
    members = list(unique.values())
    reference_rows = [row for row in members if str(row["method"]) == str(reference_method)]
    if not reference_rows:
        raise ValueError(f"missing {reference_method} reference rows for {part}")
    reference_strength = max(_float(row, "edit_strength") for row in reference_rows)
    reference_target = sum(
        _float(row, "target_delta_sum")
        for row in reference_rows
        if _float(row, "edit_strength") == reference_strength
    )
    if reference_target <= 0.0:
        raise ValueError(f"reference target response is zero for {part}")

    response_by_strength = defaultdict(float)
    for row in members:
        if str(row["method"]) == str(method):
            response_by_strength[_float(row, "edit_strength")] += _float(row, "target_delta_sum")
    if not response_by_strength:
        raise ValueError(f"missing {method} rows for {part}")
    desired = float(retention) * reference_target
    maximum = max(response_by_strength.values())
    if maximum + 1.0e-8 < desired:
        raise ValueError(
            f"{method}/{part} cannot reach target retention {float(retention):.4f}: "
            f"maximum={maximum / reference_target:.4f}"
        )
    strength, response = min(
        response_by_strength.items(),
        key=lambda item: (abs(float(item[1]) - desired), float(item[0])),
    )
    return {
        "method": str(method),
        "part": str(part),
        "reference_method": str(reference_method),
        "reference_strength": float(reference_strength),
        "reference_target_response": float(reference_target),
        "target_retention": float(retention),
        "desired_target_response": float(desired),
        "selected_strength": float(strength),
        "selected_target_response": float(response),
        "actual_retention": float(response / reference_target),
        "reachable": True,
    }


def validate_retention_value(
    value: float,
    *,
    target: float = TARGET_RETENTION,
    tolerance: float = RETENTION_TOLERANCE,
) -> None:
    actual = float(value)
    if not np.isfinite(actual) or abs(actual - float(target)) > float(tolerance) + 1.0e-12:
        raise ValueError(
            f"actual retention {actual:.6f} is outside tolerance "
            f"{float(target):.6f} +/- {float(tolerance):.6f}"
        )


def rank_objective_views(
    rows: Iterable[Mapping],
    *,
    parts: Sequence[str] = PARTS,
    methods: Sequence[str] = METHODS,
) -> list[dict]:
    required = {(str(method), str(part)) for method in methods for part in parts}
    grouped: dict[str, dict[tuple[str, str], Mapping]] = defaultdict(dict)
    for row in rows:
        key = (str(row["method"]), str(row["part"]))
        if key in required:
            grouped[str(row["view"])][key] = row

    candidates = []
    for view, cells in grouped.items():
        if set(cells) != required:
            continue
        selected = list(cells.values())
        target_responses = [_float(row, "target_delta_sum") for row in selected]
        if any(not np.isfinite(value) or value <= 0.0 for value in target_responses):
            continue
        leakages = [_float(row, "outer_to_target_delta_ratio") for row in selected]
        ious = [_float(row, "edit_response_iou") for row in selected]
        if not all(np.isfinite(value) for value in leakages + ious):
            continue
        candidates.append(
            {
                "view": view,
                "mean_normalized_leakage": float(np.mean(leakages)),
                "mean_edit_response_iou": float(np.mean(ious)),
                "cell_count": len(selected),
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["mean_normalized_leakage"]),
            -float(row["mean_edit_response_iou"]),
            str(row["view"]),
        )
    )
    for index, row in enumerate(candidates, start=1):
        row["rank"] = index
    return candidates


def foreground_crop_box(image: Image.Image, *, padding_fraction: float = 0.12) -> tuple[int, int, int, int]:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.max(array, axis=2) > 8
    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        return (0, 0, image.width, image.height)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = int(round(max(x1 - x0, y1 - y0) * float(padding_fraction)))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(image.width, x1 + padding),
        min(image.height, y1 + padding),
    )


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _fit_tile(image: Image.Image, crop_box, *, size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, (250, 250, 250))
    cropped = image.convert("RGB").crop(crop_box)
    cropped.thumbnail(size, Image.Resampling.LANCZOS)
    tile.paste(cropped, ((size[0] - cropped.width) // 2, (size[1] - cropped.height) // 2))
    return tile


def compose_b_layout(
    rows: Iterable[Mapping],
    output_png: Path | str,
    output_pdf: Path | str,
    *,
    fixed_view: str = FIXED_VIEW,
    subjects: Sequence[str] = SUBJECTS,
    view_by_subject: Mapping[str, str] | None = None,
) -> dict:
    views = {
        str(subject): str((view_by_subject or {}).get(str(subject), fixed_view))
        for subject in subjects
    }
    selected = [
        row
        for row in rows
        if str(row["subject"]) in views
        and str(row["view"]) == views[str(row["subject"])]
    ]
    lookup = {
        (str(row["subject"]), str(row["method"]), str(row["part"])): Path(str(row["frame"]))
        for row in selected
    }
    tile_size = (220, 300)
    header_font = _font(17, bold=True)
    row_font = _font(17, bold=True)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    row_label_widths = [
        measure.textbbox((0, 0), f"CoreView {subject}", font=row_font)[2]
        for subject in subjects
    ]
    max_row_label_width = max(row_label_widths, default=0)
    left = max(132, max_row_label_width + 24)
    top = 66
    gap = 10
    row_gap = 16
    width = left + len(COLUMNS) * tile_size[0] + (len(COLUMNS) - 1) * gap + 18
    height = top + len(subjects) * tile_size[1] + (len(subjects) - 1) * row_gap + 18
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for column, (label, _, _) in enumerate(COLUMNS):
        x = left + column * (tile_size[0] + gap)
        bbox = draw.textbbox((0, 0), label, font=header_font)
        draw.text((x + (tile_size[0] - (bbox[2] - bbox[0])) / 2, 23), label, fill=(25, 25, 25), font=header_font)

    paths = []
    for row_index, subject in enumerate(subjects):
        subject_view = views[str(subject)]
        input_key = (str(subject), "input", "input")
        if input_key not in lookup:
            raise ValueError(f"missing Input frame for CoreView_{subject} {subject_view}")
        input_image = Image.open(lookup[input_key]).convert("RGB")
        crop_box = foreground_crop_box(input_image)
        y = top + row_index * (tile_size[1] + row_gap)
        label = f"CoreView {subject}"
        bbox = draw.textbbox((0, 0), label, font=row_font)
        draw.text((12, y + (tile_size[1] - (bbox[3] - bbox[1])) / 2), label, fill=(30, 30, 30), font=row_font)
        for column, (_, method, part) in enumerate(COLUMNS):
            key = (str(subject), method, part)
            if key not in lookup:
                raise ValueError(f"missing frame for CoreView_{subject} {subject_view} {method}/{part}")
            path = lookup[key]
            image = Image.open(path).convert("RGB")
            if image.size != input_image.size:
                raise ValueError(f"frame dimensions do not match Input for CoreView_{subject}: {path}")
            x = left + column * (tile_size[0] + gap)
            sheet.paste(_fit_tile(image, crop_box, size=tile_size), (x, y))
            draw.rectangle((x, y, x + tile_size[0] - 1, y + tile_size[1] - 1), outline=(185, 185, 185))
            paths.append(str(path))

    output_png = Path(output_png)
    output_pdf = Path(output_pdf)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_png)
    sheet.save(output_pdf, "PDF", resolution=300.0)
    return {
        "view": str(fixed_view) if view_by_subject is None else "per_subject",
        "views": views,
        "subjects": [str(subject) for subject in subjects],
        "columns": [label for label, _, _ in COLUMNS],
        "input_frames": paths,
        "png": str(output_png),
        "pdf": str(output_pdf),
        "size": [sheet.width, sheet.height],
        "left_margin": left,
        "max_row_label_width": max_row_label_width,
    }


if __name__ == "__main__":
    raise SystemExit(main())
