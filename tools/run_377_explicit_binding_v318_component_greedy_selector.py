#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def bjt_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S BJT", time.gmtime(time.time() + 8 * 3600))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(name: str, cmd: list[str], log_path: Path, env: dict[str, str], events_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as events:
        events.write(f"{bjt_now()}\t{name}_start\t{log_path}\n")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    with events_path.open("a", encoding="utf-8") as events:
        events.write(f"{bjt_now()}\t{name}_done\tstatus={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with status {proc.returncode}; see {log_path}")


def fmt(value: float, digits: int = 8) -> str:
    return f"{float(value):.{digits}f}"


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return float(default)


def safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except Exception:
        return int(default)


def component_key(row: dict[str, str]) -> str:
    return "|".join(
        [
            str(row.get("image_name", "")),
            str(row.get("direction", "")),
            str(row.get("component_id", "")),
            str(row.get("bbox_x", "")),
            str(row.get("bbox_y", "")),
        ]
    )


def row_score(row: dict[str, str]) -> float:
    area = max(safe_float(row, "area"), 0.0)
    near = max(safe_float(row, "near_score_sum"), 0.0)
    near_count = max(safe_float(row, "near_point_count"), 0.0)
    return math.log1p(area) * (1.0 + math.log1p(near)) * (1.0 + 0.05 * math.log1p(near_count))


def read_component_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def write_component_subset(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_csv_ints(value: str) -> set[int]:
    result = set()
    for item in parse_csv_int_list(value):
        result.add(item)
    return result


def parse_csv_int_list(value: str) -> list[int]:
    result = []
    for token in str(value or "").replace("[", "").replace("]", "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(float(token)))
        except Exception:
            continue
    return result


def parse_frame_spec(value: str) -> set[int]:
    values = parse_csv_int_list(value)
    if len(values) == 3:
        start, end, step = values
        step = max(int(step), 1)
        return set(range(int(start), int(end), step))
    return set(values)


def filter_rows_for_eval(rows: list[dict[str, str]], test_views: set[int], test_frames: set[int], min_area: float) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if str(row.get("direction", "")).lower() not in {"inner", "outer"}:
            continue
        if safe_float(row, "area") < float(min_area):
            continue
        cam = safe_int(row, "cam", -1)
        frame = safe_int(row, "frame", -1)
        if test_views and cam not in test_views:
            continue
        if test_frames and frame not in test_frames:
            continue
        filtered.append(row)
    return filtered


@dataclass
class CandidateUnit:
    name: str
    rows: list[dict[str, str]]
    kind: str
    score: float

    @property
    def keys(self) -> set[str]:
        return {component_key(row) for row in self.rows}


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out = []
    for row in rows:
        key = component_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_candidate_units(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[CandidateUnit], CandidateUnit | None]:
    by_direction = {"inner": [], "outer": []}
    by_image_direction: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        direction = str(row.get("direction", "")).lower()
        if direction not in by_direction:
            continue
        by_direction[direction].append(row)
        by_image_direction.setdefault((str(row.get("image_name", "")), direction), []).append(row)

    for direction in by_direction:
        by_direction[direction].sort(key=row_score, reverse=True)
    for key in by_image_direction:
        by_image_direction[key].sort(key=row_score, reverse=True)

    candidates: list[CandidateUnit] = []

    for direction in ("inner", "outer"):
        for row in by_direction[direction][: int(args.max_individual_per_direction)]:
            image = str(row.get("image_name", "unknown"))
            cid = str(row.get("component_id", "x"))
            candidates.append(
                CandidateUnit(
                    name=f"row_{direction}_{image}_c{cid}",
                    rows=[row],
                    kind="single",
                    score=row_score(row),
                )
            )

    image_groups = []
    for (image, direction), group_rows in by_image_direction.items():
        group = group_rows[: int(args.image_group_size)]
        if not group:
            continue
        image_groups.append(
            CandidateUnit(
                name=f"image_{direction}_{image}_top{len(group)}",
                rows=group,
                kind="image_group",
                score=sum(row_score(row) for row in group),
            )
        )
    image_groups.sort(key=lambda item: item.score, reverse=True)
    candidates.extend(image_groups[: int(args.max_image_groups)])

    for direction in ("inner", "outer"):
        group = []
        for (image, group_direction), group_rows in sorted(by_image_direction.items()):
            if group_direction != direction:
                continue
            group.extend(group_rows[: int(args.global_per_image)])
        group = unique_rows(group)
        if group:
            candidates.append(
                CandidateUnit(
                    name=f"global_{direction}_top{int(args.global_per_image)}_per_image",
                    rows=group,
                    kind="global_direction",
                    score=sum(row_score(row) for row in group),
                )
            )

    full_reference = None
    all_rows = unique_rows(rows)
    if all_rows:
        full_reference = CandidateUnit(
            name="full_v307_component_set",
            rows=all_rows,
            kind="full_reference",
            score=sum(row_score(row) for row in all_rows),
        )
        if bool(args.include_full_candidate):
            candidates.append(full_reference)

    deduped = []
    seen_names = set()
    for item in sorted(candidates, key=lambda cand: cand.score, reverse=True):
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in item.name)[:140]
        if safe_name in seen_names:
            continue
        item.name = safe_name
        seen_names.add(safe_name)
        deduped.append(item)
    return deduped[: int(args.max_candidates)], full_reference


def load_metrics(render_exp: Path) -> dict[str, float]:
    contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
    }


def delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(baseline[key]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}


def strict_pass(d: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        d["inner"] < -float(args.min_inner_gain)
        and d["outer"] <= float(args.max_outer_worsen)
        and d["fg"] <= float(args.max_fg_worsen)
        and d["boundary"] <= float(args.max_boundary_worsen)
        and d["edge"] <= float(args.max_edge_worsen)
        and d["hard"] < -float(args.min_hard_gain)
    )


def probe_pass(d: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        d["hard"] < -float(args.probe_min_hard_gain)
        and d["inner"] <= float(args.probe_max_inner_worsen)
        and d["outer"] <= float(args.probe_max_outer_worsen)
        and d["fg"] <= float(args.probe_max_fg_worsen)
        and d["boundary"] <= float(args.probe_max_boundary_worsen)
        and d["edge"] <= float(args.probe_max_edge_worsen)
    )


def status_for(d: dict[str, float], args: argparse.Namespace) -> str:
    if strict_pass(d, args):
        return "strict_pass"
    if bool(args.allow_probe) and probe_pass(d, args):
        return "probe_pass"
    return "rejected"


def objective(metrics: dict[str, float], baseline: dict[str, float]) -> float:
    d = delta(metrics, baseline)
    return (
        10000.0 * d["hard"]
        + 7.0 * d["edge"]
        + 0.060 * d["inner"]
        + 0.025 * d["outer"]
        + 1500.0 * d["fg"]
        + 1200.0 * d["boundary"]
    )


def render_and_score(
    *,
    variant: str,
    component_csv: Path | None,
    args: argparse.Namespace,
    env: dict[str, str],
    events_path: Path,
    log_dir: Path,
    config_path: Path | None = None,
    ckpt: Path | None = None,
) -> dict[str, float]:
    render_exp = args.exp_root / variant
    cmd = [
        str(args.python_bin),
        "render.py",
        "--config-path",
        str((config_path or args.base_exp / ".hydra")),
        "--config-name",
        "config",
        "mode=test",
        f"load_ckpt={ckpt or args.base_ckpt}",
        f"exp_dir={render_exp}",
        f"dataset.root_dir={args.dataset_root}",
        "dataset.preload=false",
        f"dataset.train_views={args.train_views_spec}",
        f"dataset.train_frames={args.train_frames_spec}",
        f"dataset.test_views.view={args.test_views_spec}",
        f"dataset.test_frames.view={args.test_frames_spec}",
        "dataset.parsing_prior.enable=false",
        "dataset.parsing_prior.roi_enable=false",
        "export_interpretability=false",
        "export_semantic_editable_assets=false",
        "++export_opacity_maps=false",
        "++render_export_refine=false",
        f"hydra.run.dir={log_dir / 'hydra_runtime' / ('render_' + variant)}",
        "wandb_disable=true",
    ]
    if component_csv is None:
        cmd.extend(
            [
                "pipeline.compute_cov3D_python=true",
                "++pipeline.covariance_mode=default",
                "++pipeline.covariance_signed_dynamic_enable=false",
                "++pipeline.covariance_signed_screen_actuator_enable=false",
                "++pipeline.covariance_signed_center_offset_enable=false",
                "++pipeline.boundary_cov_residual_enable=false",
                "++pipeline.binding_covariance_guard_enable=false",
                "++model.deformer.rigid.geometry_fidelity_gate_enable=false",
                "++model.deformer.rigid.geometry_fidelity_component_enable=false",
                "++model.deformer.rigid.rotation_orthogonalize_enable=false",
            ]
        )
    else:
        cmd.extend(
            [
                "++explicit_binding_render_preset=v307_adopted_geometry",
                f"++explicit_binding_adopted_component_csv={component_csv}",
                f"++explicit_binding_adopted_point_csv={args.point_csv}",
                f"++explicit_binding_adopted_center_strength={args.center_strength}",
                f"++explicit_binding_adopted_outer_px={args.outer_px}",
                "++explicit_binding_adopted_component_required=true",
                "++explicit_binding_adopted_improvement_guard=true",
                f"++explicit_binding_adopted_max_points={args.max_points_per_action}",
            ]
        )
    run_command(f"render_{variant}", cmd, log_dir / f"render_{variant}.log", env, events_path)
    run_command(
        f"contours_{variant}",
        [
            str(args.python_bin),
            "tools/analyze_377_render_contours.py",
            "--render-exp",
            str(render_exp),
            "--dataset-root",
            str(args.dataset_root),
            "--subject",
            "CoreView_377",
            "--split-dir",
            "test-view",
            "--band-width",
            "7",
            "--topk",
            "12",
            "--out-dir",
            str(render_exp / "diagnostics" / "contours"),
        ],
        log_dir / f"contours_{variant}.log",
        env,
        events_path,
    )
    run_command(
        f"residuals_{variant}",
        [
            str(args.python_bin),
            "tools/analyze_377_boundary_residuals.py",
            "--render-exp",
            str(render_exp),
            "--dataset-root",
            str(args.dataset_root),
            "--subject",
            "CoreView_377",
            "--split-dir",
            "test-view",
            "--render-support-threshold",
            "0.025",
            "--close-kernel",
            "5",
            "--band-width",
            "7",
            "--search-band-width",
            "24",
            "--topk",
            "16",
            "--out-dir",
            str(render_exp / "diagnostics" / "boundary_residuals"),
        ],
        log_dir / f"boundary_residuals_{variant}.log",
        env,
        events_path,
    )
    return load_metrics(render_exp)


TRIAL_HEADER = [
    "round",
    "candidate",
    "kind",
    "row_count",
    "new_row_count",
    "render_exp",
    "fg",
    "boundary",
    "edge",
    "inner",
    "outer",
    "hard",
    "fg_delta_baseline",
    "boundary_delta_baseline",
    "edge_delta_baseline",
    "inner_delta_baseline",
    "outer_delta_baseline",
    "hard_delta_baseline",
    "fg_delta_current",
    "boundary_delta_current",
    "edge_delta_current",
    "inner_delta_current",
    "outer_delta_current",
    "hard_delta_current",
    "objective",
    "status",
    "accepted",
]


def append_trial(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_HEADER, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def metric_fields(metrics: dict[str, float], prefix: str = "") -> dict[str, str]:
    if prefix in {"delta_baseline_", "delta_current_"}:
        return {
            f"fg_{prefix.rstrip('_')}": fmt(metrics["fg"]),
            f"boundary_{prefix.rstrip('_')}": fmt(metrics["boundary"]),
            f"edge_{prefix.rstrip('_')}": fmt(metrics["edge"], 6),
            f"inner_{prefix.rstrip('_')}": fmt(metrics["inner"], 4),
            f"outer_{prefix.rstrip('_')}": fmt(metrics["outer"], 4),
            f"hard_{prefix.rstrip('_')}": fmt(metrics["hard"]),
        }
    return {
        f"{prefix}fg": fmt(metrics["fg"]),
        f"{prefix}boundary": fmt(metrics["boundary"]),
        f"{prefix}edge": fmt(metrics["edge"], 6),
        f"{prefix}inner": fmt(metrics["inner"], 4),
        f"{prefix}outer": fmt(metrics["outer"], 4),
        f"{prefix}hard": fmt(metrics["hard"]),
    }


def run_short_train(selected_csv: Path, baseline_metrics: dict[str, float], args: argparse.Namespace, env: dict[str, str], events_path: Path) -> None:
    train_exp = args.exp_root / "selected_component_color_train"
    checkpoint_steps = [int(token.strip()) for token in str(args.train_checkpoint_steps).split(",") if token.strip()]
    checkpoint_list = "[" + ",".join(str(step) for step in checkpoint_steps) + "]"
    cmd = [
        str(args.python_bin),
        "train.py",
        "--config-path",
        str(args.base_exp / ".hydra"),
        "--config-name",
        "config",
        "mode=train",
        f"dataset.root_dir={args.dataset_root}",
        "dataset.preload=false",
        f"dataset.train_views={args.train_views_spec}",
        f"dataset.val_views={args.test_views_spec}",
        f"dataset.test_views.view={args.test_views_spec}",
        f"dataset.train_frames={args.train_frames_dense_spec}",
        f"dataset.val_frames={args.test_frames_spec}",
        f"dataset.test_frames.view={args.test_frames_spec}",
        "dataset.parsing_prior.enable=false",
        "dataset.parsing_prior.roi_enable=false",
        "dataset.parsing_prior.compact_mapping_file=",
        f"start_checkpoint={args.base_ckpt}",
        f"exp_dir={train_exp}",
        f"hydra.run.dir={args.log_dir / 'hydra_runtime' / 'train_selected_component'}",
        "seed=-1",
        "wandb_disable=true",
        "++resume.allow_partial_converter_load=true",
        "++resume.restore_gaussian_optimizer_state=false",
        "++resume.restore_converter_optimizer_state=false",
        "++resume.restore_converter_scheduler_state=false",
        "++resume.disable_densify_on_resume=true",
        "++resume.disable_opacity_reset_on_resume=true",
        "++resume.require_no_densify_on_resume=true",
        "++resume.use_checkpoint_iteration_as_offset=true",
        "++resume.clear_boundary_tags_on_resume=true",
        "++resume.clear_binding_state_on_resume=false",
        "pipeline.pose_noise=0.0",
        "pipeline.compute_cov3D_python=true",
        "++pipeline.covariance_mode=default",
        "++pipeline.covariance_signed_dynamic_enable=true",
        f"++pipeline.covariance_signed_dynamic_component_csv={selected_csv}",
        f"++pipeline.covariance_signed_dynamic_point_csv={args.point_csv}",
        "++pipeline.covariance_signed_dynamic_component_signature_enable=false",
        "++pipeline.covariance_signed_dynamic_over_layer_ids=soft,free",
        "++pipeline.covariance_signed_dynamic_over_region_ids=cloth",
        "++pipeline.covariance_signed_dynamic_over_joint_ids=6,9,12,13,14,15",
        "++pipeline.covariance_signed_dynamic_under_layer_ids=soft,rigid,free",
        "++pipeline.covariance_signed_dynamic_under_region_ids=cloth,body,soft",
        "++pipeline.covariance_signed_dynamic_under_joint_ids=0,1,2,4,7,8,10",
        "++pipeline.covariance_signed_dynamic_boundary_min=0.0",
        "++pipeline.covariance_signed_dynamic_component_pad_px=10",
        "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25",
        "++pipeline.covariance_signed_dynamic_component_max_over=16",
        "++pipeline.covariance_signed_dynamic_component_max_under=16",
        "++pipeline.covariance_signed_dynamic_component_min_area=20",
        "++pipeline.covariance_signed_dynamic_component_required=true",
        f"++pipeline.covariance_signed_dynamic_max_over_points={args.max_points_per_action}",
        f"++pipeline.covariance_signed_dynamic_max_under_points={args.max_points_per_action}",
        "++pipeline.covariance_signed_screen_actuator_enable=true",
        "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940",
        "++pipeline.covariance_signed_screen_normal_grow_factor=1.025",
        "++pipeline.covariance_signed_screen_tangent_factor=1.000",
        "++pipeline.covariance_signed_center_offset_enable=true",
        f"++pipeline.covariance_signed_center_offset_outer_px={args.outer_px}",
        "++pipeline.covariance_signed_center_offset_inner_px=0.0",
        "++pipeline.covariance_signed_center_offset_outer_direction=view_center",
        "++pipeline.covariance_signed_center_offset_inner_direction=component_center",
        "++pipeline.covariance_signed_center_offset_score_weight_power=1.0",
        "++pipeline.covariance_signed_center_offset_score_weight_min=0.15",
        "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90",
        "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001",
        "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001",
        "++pipeline.covariance_signed_center_offset_max_world_step=0.0020",
        "++pipeline.boundary_cov_residual_enable=false",
        "++pipeline.binding_covariance_guard_enable=false",
        "++model.deformer.rigid.rotation_orthogonalize_enable=false",
        "++model.deformer.rigid.geometry_fidelity_gate_enable=true",
        "++model.deformer.rigid.geometry_fidelity_target=free_lbs",
        f"++model.deformer.rigid.geometry_fidelity_center_strength={args.center_strength}",
        "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0",
        "++model.deformer.rigid.geometry_fidelity_boundary_min=0.12",
        "++model.deformer.rigid.geometry_fidelity_layer_ids=soft,free",
        "++model.deformer.rigid.geometry_fidelity_region_ids=cloth,soft",
        "++model.deformer.rigid.geometry_fidelity_joint_ids=",
        "++model.deformer.rigid.geometry_fidelity_non_rigid_min=0.0",
        "++model.deformer.rigid.geometry_fidelity_power=1.2",
        "++model.deformer.rigid.geometry_fidelity_max_points=1024",
        "++model.deformer.rigid.geometry_fidelity_component_enable=true",
        f"++model.deformer.rigid.geometry_fidelity_component_csv={selected_csv}",
        "++model.deformer.rigid.geometry_fidelity_component_direction=inner",
        "++model.deformer.rigid.geometry_fidelity_component_pad_px=2",
        "++model.deformer.rigid.geometry_fidelity_component_ellipse_scale=1.05",
        "++model.deformer.rigid.geometry_fidelity_component_max=12",
        "++model.deformer.rigid.geometry_fidelity_component_min_area=40",
        "++model.deformer.rigid.geometry_fidelity_component_required=true",
        "++model.deformer.rigid.geometry_fidelity_component_improvement_enable=true",
        "model.pose_correction.delay=1",
        "++model.pose_correction.train_root_orient=false",
        "++model.pose_correction.train_pose_body=false",
        "++model.pose_correction.train_pose_hand=false",
        "++model.pose_correction.train_trans=false",
        "++model.pose_correction.train_betas=false",
        f"opt.iterations={int(args.train_iterations)}",
        "opt.position_lr_init=0.0",
        "opt.position_lr_final=0.0",
        f"opt.feature_lr={args.train_feature_lr}",
        "opt.opacity_lr=0.0",
        "opt.scaling_lr=0.0",
        "opt.rotation_lr=0.0",
        "opt.rigid_lr=0.0",
        "opt.non_rigid_lr=0.0",
        "opt.nr_latent_lr=0.0",
        "opt.pose_correction_lr=0.0",
        f"opt.texture_lr={args.train_texture_lr}",
        "opt.tex_latent_lr=0.0",
        "++opt.camera_affine_enable=false",
        "++opt.camera_affine_lr=0.0",
        "++opt.camera_geometry_enable=true",
        "++opt.camera_geometry_lr=0.0",
        "++opt.boundary_opacity_residual_lr=0.0",
        "++opt.boundary_scaling_residual_lr=0.0",
        "++opt.boundary_cov_residual_lr=0.0",
        "++opt.binding_layer_logits_lr=0.0",
        "++opt.stageB_semantic_loss_enable=false",
        "++opt.stageB_semantic_body_cloth_weight=0.0",
        "++opt.stageB_semantic_compact_weight=0.0",
        "++opt.train_sample_mode=frame_balanced_camera_weighted",
        "++opt.train_sample_camera_min_prob=0.018",
        "++opt.train_sample_camera_max_prob=0.125",
        "opt.lambda_l1=0.040",
        "opt.lambda_l1_fg=0.100",
        "opt.lambda_l1_boundary=0.040",
        "opt.lambda_dssim=0.0",
        "opt.lambda_perceptual=0.0",
        "opt.lambda_mask=0.0",
        "++opt.lambda_mask_boundary=0.0",
        "++opt.lambda_mask_boundary_hard=0.0",
        "++opt.lambda_silhouette_outer=0.0",
        "++opt.lambda_silhouette_inner=0.0",
        "opt.lambda_skinning=0.0",
        "opt.lambda_aiap_xyz=0.0",
        "opt.lambda_aiap_cov=0.0",
        "opt.percent_dense=0.0",
        "opt.densify_until_iter=0",
        "opt.densify_from_iter=1000000",
        "opt.opacity_reset_interval=1000000",
        "best_eval_split=test",
        "best_metric=l1_fg",
        "best_metric_mode=min",
        "best_metric_source=best_eval",
        "test_interval=0",
        f"test_iterations={checkpoint_list}",
        f"save_iterations={checkpoint_list}",
        f"checkpoint_iterations={checkpoint_list}",
        "++validation_image_log_limit=0",
        "opt.grad_clip=0.0015",
    ]
    run_command("train_selected_component", cmd, args.log_dir / "train_selected_component.log", env, events_path)

    summary_path = args.log_dir / "train_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "label",
                "ckpt",
                "render_exp",
                "fg",
                "boundary",
                "edge",
                "inner",
                "outer",
                "hard",
                "fg_delta_baseline",
                "boundary_delta_baseline",
                "edge_delta_baseline",
                "inner_delta_baseline",
                "outer_delta_baseline",
                "hard_delta_baseline",
                "status",
            ]
        )
        for step in checkpoint_steps:
            global_iter = int(args.base_iter) + int(step)
            ckpt = train_exp / f"ckpt{global_iter}.pth"
            if not ckpt.exists():
                continue
            label = f"train_ckpt{global_iter}"
            metrics = render_and_score(
                variant=label,
                component_csv=selected_csv,
                args=args,
                env=env,
                events_path=events_path,
                log_dir=args.log_dir,
                config_path=train_exp / ".hydra",
                ckpt=ckpt,
            )
            dbase = delta(metrics, baseline_metrics)
            writer.writerow(
                [
                    label,
                    str(ckpt),
                    str(args.exp_root / label),
                    fmt(metrics["fg"]),
                    fmt(metrics["boundary"]),
                    fmt(metrics["edge"], 6),
                    fmt(metrics["inner"], 4),
                    fmt(metrics["outer"], 4),
                    fmt(metrics["hard"]),
                    fmt(dbase["fg"]),
                    fmt(dbase["boundary"]),
                    fmt(dbase["edge"], 6),
                    fmt(dbase["inner"], 4),
                    fmt(dbase["outer"], 4),
                    fmt(dbase["hard"]),
                    status_for(dbase, args),
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="v318 component-level greedy render-in-loop selector.")
    parser.add_argument("--python-bin", type=Path, default=Path("/opt/miniconda3/envs/ictrl/bin/python"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--base-exp", type=Path, default=ROOT / "exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt")
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--base-iter", type=int, default=136410)
    parser.add_argument("--component-csv", type=Path, required=True)
    parser.add_argument("--point-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/ZJUMoCap")
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--train-views-spec", default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]")
    parser.add_argument("--train-frames-spec", default="[0,570,60]")
    parser.add_argument("--train-frames-dense-spec", default="[0,570,1]")
    parser.add_argument("--test-views-spec", default="[21,22,23]")
    parser.add_argument("--test-frames-spec", default="[0,570,60]")
    parser.add_argument("--min-component-area", type=float, default=20.0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-accept", type=int, default=4)
    parser.add_argument("--max-individual-per-direction", type=int, default=4)
    parser.add_argument("--max-image-groups", type=int, default=4)
    parser.add_argument("--image-group-size", type=int, default=2)
    parser.add_argument("--global-per-image", type=int, default=12)
    parser.add_argument("--include-full-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-full-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--center-strength", type=float, default=0.45)
    parser.add_argument("--outer-px", type=float, default=0.35)
    parser.add_argument("--max-points-per-action", type=int, default=96)
    parser.add_argument("--min-inner-gain", type=float, default=0.05)
    parser.add_argument("--min-hard-gain", type=float, default=0.000001)
    parser.add_argument("--max-outer-worsen", type=float, default=0.0)
    parser.add_argument("--max-fg-worsen", type=float, default=0.0)
    parser.add_argument("--max-boundary-worsen", type=float, default=0.0)
    parser.add_argument("--max-edge-worsen", type=float, default=0.0)
    parser.add_argument("--allow-probe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--probe-min-hard-gain", type=float, default=0.00001)
    parser.add_argument("--probe-max-inner-worsen", type=float, default=0.5)
    parser.add_argument("--probe-max-outer-worsen", type=float, default=0.5)
    parser.add_argument("--probe-max-fg-worsen", type=float, default=0.000015)
    parser.add_argument("--probe-max-boundary-worsen", type=float, default=0.000015)
    parser.add_argument("--probe-max-edge-worsen", type=float, default=0.003)
    parser.add_argument("--do-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-iterations", type=int, default=100)
    parser.add_argument("--train-checkpoint-steps", default="100")
    parser.add_argument("--train-feature-lr", type=float, default=0.00005)
    parser.add_argument("--train-texture-lr", type=float, default=0.0)
    args = parser.parse_args()

    args.base_exp = args.base_exp.resolve()
    args.base_ckpt = (args.base_ckpt or (args.base_exp / "ckpt136410.pth")).resolve()
    args.component_csv = args.component_csv.resolve()
    args.point_csv = args.point_csv.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.exp_root = args.exp_root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.exp_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "hydra_runtime").mkdir(parents=True, exist_ok=True)

    for required in (args.python_bin, args.base_exp / ".hydra" / "config.yaml", args.base_ckpt, args.component_csv, args.point_csv, args.dataset_root):
        if not Path(required).exists():
            raise FileNotFoundError(required)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "6"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "6"),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", "6"),
            "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS", "6"),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64"),
        }
    )

    events_path = args.log_dir / "events.tsv"
    events_path.write_text("time_bjt\tphase\tdetail\n", encoding="utf-8")
    start_bjt = bjt_now()
    write_json(
        args.log_dir / "run_info.json",
        {
            "run_id": args.run_id,
            "start_bjt": start_bjt,
            "gpu": args.gpu,
            "base_exp": str(args.base_exp),
            "base_ckpt": str(args.base_ckpt),
            "component_csv": str(args.component_csv),
            "point_csv": str(args.point_csv),
            "exp_root": str(args.exp_root),
            "log_dir": str(args.log_dir),
            "do_train": bool(args.do_train),
        },
    )
    status_json = args.log_dir / "status.json"
    write_json(status_json, {"run_id": args.run_id, "phase": "start", "start_bjt": start_bjt})

    fieldnames, all_rows = read_component_rows(args.component_csv)
    test_views = parse_csv_ints(args.test_views_spec)
    test_frames = parse_frame_spec(args.test_frames_spec)
    eval_rows = filter_rows_for_eval(all_rows, test_views, test_frames, args.min_component_area)
    if not eval_rows:
        raise RuntimeError(f"no component rows for test views/frames in {args.component_csv}")
    candidates, full_reference = build_candidate_units(eval_rows, args)
    if not candidates:
        raise RuntimeError("no candidate units built")

    with (args.log_dir / "candidate_units.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["name", "kind", "row_count", "score"])
        for cand in candidates:
            writer.writerow([cand.name, cand.kind, len(cand.rows), fmt(cand.score, 6)])

    print(f"{bjt_now()} render baseline", flush=True)
    baseline_metrics = render_and_score(
        variant="baseline_no_preset",
        component_csv=None,
        args=args,
        env=env,
        events_path=events_path,
        log_dir=args.log_dir,
    )
    current_metrics = dict(baseline_metrics)
    full_reference_metrics = None
    if bool(args.score_full_reference) and full_reference is not None and not bool(args.include_full_candidate):
        full_csv = args.log_dir / "component_subsets" / "full_reference.csv"
        write_component_subset(full_csv, fieldnames, full_reference.rows)
        full_reference_metrics = render_and_score(
            variant="reference_full_v307_component_set",
            component_csv=full_csv,
            args=args,
            env=env,
            events_path=events_path,
            log_dir=args.log_dir,
        )
    if full_reference_metrics is not None:
        dbase = delta(full_reference_metrics, baseline_metrics)
        append_trial(
            args.log_dir / "reference_trials.tsv",
            {
                "round": 0,
                "candidate": "full_v307_component_set",
                "kind": "full_reference",
                "row_count": len(full_reference.rows) if full_reference is not None else 0,
                "new_row_count": len(full_reference.rows) if full_reference is not None else 0,
                "render_exp": str(args.exp_root / "reference_full_v307_component_set"),
                **metric_fields(full_reference_metrics),
                **metric_fields(dbase, "delta_baseline_"),
                **metric_fields(dbase, "delta_current_"),
                "objective": fmt(objective(full_reference_metrics, baseline_metrics), 8),
                "status": status_for(dbase, args),
                "accepted": "reference",
            },
        )
    selected_rows: list[dict[str, str]] = []
    selected_keys: set[str] = set()
    remaining = list(candidates)
    trials_path = args.log_dir / "greedy_trials.tsv"

    for round_idx in range(1, int(args.max_accept) + 1):
        print(f"{bjt_now()} greedy round {round_idx} remaining={len(remaining)} selected_rows={len(selected_rows)}", flush=True)
        round_candidates = []
        for cand in remaining:
            new_rows = [row for row in cand.rows if component_key(row) not in selected_keys]
            if not new_rows:
                continue
            trial_rows = unique_rows(selected_rows + new_rows)
            trial_csv = args.log_dir / "component_subsets" / f"round{round_idx:02d}_{cand.name}.csv"
            write_component_subset(trial_csv, fieldnames, trial_rows)
            variant = f"r{round_idx:02d}_{cand.name}"
            try:
                metrics = render_and_score(
                    variant=variant,
                    component_csv=trial_csv,
                    args=args,
                    env=env,
                    events_path=events_path,
                    log_dir=args.log_dir,
                )
                dbase = delta(metrics, baseline_metrics)
                dcur = delta(metrics, current_metrics)
                status = status_for(dbase, args)
                score = objective(metrics, baseline_metrics)
                accepted = "0"
                append_trial(
                    trials_path,
                    {
                        "round": round_idx,
                        "candidate": cand.name,
                        "kind": cand.kind,
                        "row_count": len(trial_rows),
                        "new_row_count": len(new_rows),
                        "render_exp": str(args.exp_root / variant),
                        **metric_fields(metrics),
                        **metric_fields(dbase, "delta_baseline_"),
                        **metric_fields(dcur, "delta_current_"),
                        "objective": fmt(score, 8),
                        "status": status,
                        "accepted": accepted,
                    },
                )
                if status in {"strict_pass", "probe_pass"} and score < objective(current_metrics, baseline_metrics):
                    rank = (0 if status == "strict_pass" else 1, score, dbase["edge"], dbase["inner"], dbase["outer"])
                    round_candidates.append((rank, cand, new_rows, trial_rows, metrics, trial_csv, status, score))
            except Exception as exc:
                append_trial(
                    trials_path,
                    {
                        "round": round_idx,
                        "candidate": cand.name,
                        "kind": cand.kind,
                        "row_count": len(selected_rows) + len(new_rows),
                        "new_row_count": len(new_rows),
                        "render_exp": str(args.exp_root / variant),
                        "status": f"error:{exc}",
                        "accepted": "0",
                    },
                )
        if not round_candidates:
            print(f"{bjt_now()} no passing candidate in round {round_idx}", flush=True)
            break
        round_candidates.sort(key=lambda item: item[0])
        _, best, new_rows, trial_rows, metrics, trial_csv, status, score = round_candidates[0]
        selected_rows = trial_rows
        selected_keys = {component_key(row) for row in selected_rows}
        current_metrics = metrics
        remaining = [cand for cand in remaining if cand.name != best.name and not cand.keys.issubset(selected_keys)]
        with events_path.open("a", encoding="utf-8") as events:
            events.write(f"{bjt_now()}\taccept\tcandidate={best.name} status={status} rows={len(selected_rows)} score={score:.8f}\n")
        if best.kind == "full_reference":
            break

    selected_csv = args.log_dir / "selected_components.csv"
    if selected_rows:
        write_component_subset(selected_csv, fieldnames, selected_rows)
    else:
        selected_csv.write_text(",".join(fieldnames) + "\n", encoding="utf-8")

    final_delta = delta(current_metrics, baseline_metrics)
    final_status = status_for(final_delta, args) if selected_rows else "none"
    summary = {
        "run_id": args.run_id,
        "start_bjt": start_bjt,
        "end_bjt": bjt_now(),
        "selected_row_count": len(selected_rows),
        "selected_status": final_status,
        "selected_csv": str(selected_csv),
        "baseline_metrics": baseline_metrics,
        "final_metrics": current_metrics,
        "final_delta": final_delta,
    }
    write_json(args.log_dir / "selection_summary.json", summary)
    with (args.log_dir / "selection_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["label", "fg", "boundary", "edge", "inner", "outer", "hard", "status", "selected_csv"])
        writer.writerow([
            "baseline",
            fmt(baseline_metrics["fg"]),
            fmt(baseline_metrics["boundary"]),
            fmt(baseline_metrics["edge"], 6),
            fmt(baseline_metrics["inner"], 4),
            fmt(baseline_metrics["outer"], 4),
            fmt(baseline_metrics["hard"]),
            "baseline",
            "",
        ])
        writer.writerow([
            "selected",
            fmt(current_metrics["fg"]),
            fmt(current_metrics["boundary"]),
            fmt(current_metrics["edge"], 6),
            fmt(current_metrics["inner"], 4),
            fmt(current_metrics["outer"], 4),
            fmt(current_metrics["hard"]),
            final_status,
            str(selected_csv),
        ])
        writer.writerow([
            "delta",
            fmt(final_delta["fg"]),
            fmt(final_delta["boundary"]),
            fmt(final_delta["edge"], 6),
            fmt(final_delta["inner"], 4),
            fmt(final_delta["outer"], 4),
            fmt(final_delta["hard"]),
            final_status,
            str(selected_csv),
        ])

    if selected_rows and final_status in {"strict_pass", "probe_pass"} and bool(args.do_train):
        run_short_train(selected_csv, baseline_metrics, args, env, events_path)

    end_bjt = bjt_now()
    write_json(status_json, {"run_id": args.run_id, "phase": "all_done", "start_bjt": start_bjt, "end_bjt": end_bjt, "selected_status": final_status})
    with events_path.open("a", encoding="utf-8") as events:
        events.write(f"{end_bjt}\tall_done\tstatus={final_status} selected_rows={len(selected_rows)}\n")
    print(f"EXP_ROOT={args.exp_root}")
    print(f"LOG_DIR={args.log_dir}")
    print(f"SELECTED_CSV={selected_csv}")
    print(f"SUMMARY={args.log_dir / 'selection_summary.tsv'}")
    print(f"END_BJT={end_bjt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
