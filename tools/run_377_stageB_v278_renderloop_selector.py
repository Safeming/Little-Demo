#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bjt_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S BJT", time.gmtime(time.time() + 8 * 3600))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fmt(value: float, digits: int = 8) -> str:
    return f"{float(value):.{digits}f}"


def _run_command(name: str, cmd: list[str], log_path: Path, env: dict[str, str], events_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as events:
        events.write(f"{_bjt_now()}\t{name}_start\t{log_path}\n")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    with events_path.open("a", encoding="utf-8") as events:
        events.write(f"{_bjt_now()}\t{name}_done\tstatus={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with status {proc.returncode}; see {log_path}")


def _read_candidate_rows(path: Path, max_candidates: int) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if max_candidates > 0:
        rows = rows[: int(max_candidates)]
    return fieldnames, rows


def _write_subset_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_metrics(render_exp: Path) -> dict[str, float]:
    contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads(
        (render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8")
    )
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
    }


def _delta(metrics: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(base[key]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}


def _accept_pass(delta: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        delta["inner"] < -float(args.min_inner_gain)
        and delta["outer"] <= float(args.max_outer_worsen)
        and delta["fg"] <= float(args.max_fg_worsen)
        and delta["boundary"] <= float(args.max_boundary_worsen)
        and delta["edge"] <= float(args.max_edge_worsen)
        and delta["hard"] < -float(args.min_hard_gain)
    )


def _gate_pass_against_baseline(metrics: dict[str, float], baseline: dict[str, float], args: argparse.Namespace) -> bool:
    return _accept_pass(_delta(metrics, baseline), args)


def _score_delta(delta: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        -float(delta["hard"]),
        -float(delta["inner"]),
        -float(delta["outer"]),
        -float(delta["edge"]),
    )


def _append_trial_row(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    fieldnames = [
        "round",
        "candidate_idx",
        "candidate_key",
        "subset_indices",
        "ckpt",
        "render_exp",
        "fg",
        "boundary",
        "edge",
        "inner",
        "outer",
        "hard",
        "fg_delta_current",
        "boundary_delta_current",
        "edge_delta_current",
        "inner_delta_current",
        "outer_delta_current",
        "hard_delta_current",
        "fg_delta_baseline",
        "boundary_delta_baseline",
        "edge_delta_baseline",
        "inner_delta_baseline",
        "outer_delta_baseline",
        "hard_delta_baseline",
        "accept_pass",
        "accepted",
        "status",
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _candidate_key(row: dict[str, str], idx: int) -> str:
    return f"{idx}:c{row.get('birth_cam', '?')}_f{row.get('frame', '?')}_src{row.get('source_candidate_index', idx)}"


def render_and_score(
    *,
    variant: str,
    ckpt: Path,
    render_exp: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    events_path: Path,
    log_dir: Path,
) -> dict[str, float]:
    _run_command(
        f"render_{variant}",
        [
            str(args.python_bin),
            "render.py",
            "--config-path",
            str(args.base_exp / ".hydra"),
            "--config-name",
            "config",
            "mode=test",
            f"load_ckpt={ckpt}",
            f"exp_dir={render_exp}",
            f"dataset.root_dir={args.dataset_root}",
            "dataset.preload=false",
            "dataset.test_views.view=[21,22,23]",
            "dataset.test_frames.view=[0,570,60]",
            "dataset.parsing_prior.enable=false",
            "dataset.parsing_prior.roi_enable=false",
            "pipeline.compute_cov3D_python=true",
            "++render_scaling_modifier=1.0",
            "++model.deformer.rigid.rotation_orthogonalize_enable=false",
            "++opt.camera_geometry_enable=true",
            "++opt.camera_geometry_lr=0.0",
            "export_interpretability=false",
            "export_semantic_editable_assets=false",
            "++render_export_refine=false",
            f"hydra.run.dir={log_dir / 'hydra_runtime' / ('render_' + variant)}",
            "wandb_disable=true",
        ],
        log_dir / f"render_{variant}.log",
        env,
        events_path,
    )
    _run_command(
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
    _run_command(
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
    return _load_metrics(render_exp)


def append_subset_checkpoint(
    *,
    variant: str,
    subset_indices: list[int],
    fieldnames: list[str],
    candidate_rows: list[dict[str, str]],
    base_ckpt: Path,
    out_dir: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    events_path: Path,
    log_dir: Path,
) -> Path:
    subset_csv = log_dir / "candidate_subsets" / f"{variant}.csv"
    _write_subset_csv(subset_csv, fieldnames, [candidate_rows[idx] for idx in subset_indices])
    append_exp = out_dir / f"{variant}_append"
    _run_command(
        f"append_{variant}",
        [
            str(args.python_bin),
            "tools/append_377_stageB_v277_verified_support.py",
            "--config-path",
            str(args.base_exp / ".hydra" / "config.yaml"),
            "--load-ckpt",
            str(base_ckpt),
            "--candidates-csv",
            str(subset_csv),
            "--out-dir",
            str(append_exp),
            "--dataset-root",
            str(args.dataset_root),
            "--max-candidates",
            str(len(subset_indices)),
            "--checkpoint-iteration",
            str(args.append_iter),
            "--parent-screen-radius",
            str(args.parent_screen_radius),
            "--child-opacity-factor",
            str(args.child_opacity_factor),
            "--child-opacity-floor",
            str(args.child_opacity_floor),
            "--child-opacity-ceiling",
            str(args.child_opacity_ceiling),
            "--child-scale-factor",
            str(args.child_scale_factor),
            "--child-scale-max",
            str(args.child_scale_max),
        ],
        log_dir / f"append_{variant}.log",
        env,
        events_path,
    )
    return append_exp / f"ckpt{int(args.append_iter)}.pth"


def run_short_train(
    *,
    selected_ckpt: Path,
    baseline_metrics: dict[str, float],
    args: argparse.Namespace,
    env: dict[str, str],
    events_path: Path,
    log_dir: Path,
) -> list[dict[str, object]]:
    train_exp = args.exp_root / "selected_support_color_train"
    checkpoint_steps = [int(v.strip()) for v in str(args.train_checkpoint_steps).split(",") if v.strip()]
    checkpoint_list = "[" + ",".join(str(v) for v in checkpoint_steps) + "]"
    _run_command(
        "train_selected_support",
        [
            str(args.python_bin),
            "train.py",
            "--config-path",
            str(args.base_exp / ".hydra"),
            "--config-name",
            "config",
            "mode=train",
            f"dataset.root_dir={args.dataset_root}",
            "dataset.preload=false",
            "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]",
            "dataset.val_views=[21,22,23]",
            "dataset.test_views.view=[21,22,23]",
            "dataset.train_frames=[0,570,1]",
            "dataset.val_frames=[0,570,60]",
            "dataset.test_frames.view=[0,570,60]",
            "dataset.parsing_prior.enable=false",
            "dataset.parsing_prior.roi_enable=false",
            "dataset.parsing_prior.compact_mapping_file=",
            f"start_checkpoint={selected_ckpt}",
            f"exp_dir={train_exp}",
            f"hydra.run.dir={log_dir / 'hydra_runtime' / 'train_selected_support'}",
            "seed=-1",
            "wandb_disable=true",
            "++resume.allow_partial_converter_load=true",
            "++resume.restore_gaussian_optimizer_state=false",
            "++resume.restore_converter_optimizer_state=false",
            "++resume.restore_converter_scheduler_state=false",
            "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.]",
            "++resume.disable_densify_on_resume=true",
            "++resume.disable_opacity_reset_on_resume=true",
            "++resume.require_no_densify_on_resume=true",
            "++resume.use_checkpoint_iteration_as_offset=true",
            "++resume.clear_boundary_tags_on_resume=false",
            "++resume.clear_binding_state_on_resume=false",
            "pipeline.pose_noise=0.0",
            "model.pose_correction.delay=1",
            "++model.pose_correction.train_root_orient=false",
            "++model.pose_correction.train_pose_body=false",
            "++model.pose_correction.train_pose_hand=false",
            "++model.pose_correction.train_trans=false",
            "++model.pose_correction.train_betas=false",
            f"opt.iterations={int(args.train_iterations)}",
            "opt.position_lr_init=0.0",
            "opt.position_lr_final=0.0",
            "opt.feature_lr=0.00018",
            "opt.opacity_lr=0.0",
            "opt.scaling_lr=0.0",
            "opt.rotation_lr=0.0",
            "opt.rigid_lr=0.0",
            "opt.non_rigid_lr=0.0",
            "opt.nr_latent_lr=0.0",
            "opt.pose_correction_lr=0.0",
            "opt.texture_lr=0.0",
            "opt.tex_latent_lr=0.0",
            "++opt.camera_affine_enable=false",
            "++opt.camera_affine_lr=0.0",
            "++opt.camera_geometry_enable=true",
            "++opt.camera_geometry_lr=0.0",
            "++opt.boundary_support_only_grad_mask_enable=true",
            "++opt.boundary_opacity_residual_lr=0.0",
            "++opt.boundary_scaling_residual_lr=0.0",
            "++opt.stageB_semantic_loss_enable=false",
            "++opt.stageB_semantic_body_cloth_weight=0.0",
            "++opt.stageB_semantic_compact_weight=0.0",
            "++opt.lambda_binding_semantic_adapter_reg=0.0",
            "++opt.semantic_region_logits_lr=0.0",
            "++opt.semantic_compact_logits_lr=0.0",
            "++opt.train_sample_mode=frame_balanced_camera_weighted",
            "++opt.train_sample_camera_min_prob=0.018",
            "++opt.train_sample_camera_max_prob=0.125",
            "++opt.train_sample_accumulation_steps=1",
            "opt.lambda_l1=0.060",
            "opt.lambda_l1_fg=0.140",
            "opt.lambda_l1_boundary=0.080",
            "opt.lambda_perceptual=0.025",
            "opt.lambda_l1_face=0.020",
            "opt.lambda_l1_shoulder_arm=0.016",
            "opt.lambda_l1_waist=0.012",
            "opt.lambda_edge_face=0.003",
            "opt.lambda_edge_shoulder_arm=0.003",
            "opt.lambda_edge_waist=0.0015",
            "++opt.lambda_detail_face=0.0",
            "++opt.lambda_detail_shoulder_arm=0.0",
            "++opt.lambda_detail_waist=0.0",
            "++opt.lambda_perceptual_face=0.008",
            "++opt.lambda_perceptual_shoulder_arm=0.006",
            "++opt.lambda_perceptual_waist=0.003",
            "opt.lambda_mask=0.0",
            "++opt.lambda_mask_boundary=0.0",
            "++opt.lambda_mask_boundary_hard=0.0",
            "++opt.lambda_silhouette_outer=0.0",
            "++opt.lambda_silhouette_inner=0.0",
            "++opt.lambda_boundary_opacity_residual_reg=0.0",
            "++opt.lambda_boundary_scaling_residual_reg=0.0",
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
            "opt.grad_clip=0.0020",
        ],
        log_dir / "train_selected_support.log",
        env,
        events_path,
    )

    rows = []
    loaded_iter = int(args.append_iter)
    train_summary = log_dir / "train_summary.tsv"
    train_summary.write_text(
        "label\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta_baseline\tboundary_delta_baseline\tedge_delta_baseline\tinner_delta_baseline\touter_delta_baseline\thard_delta_baseline\tstrict_pass\n",
        encoding="utf-8",
    )
    for step in checkpoint_steps:
        global_iter = loaded_iter + int(step)
        ckpt = train_exp / f"ckpt{global_iter}.pth"
        if not ckpt.exists():
            continue
        label = f"train_ckpt{global_iter}"
        render_exp = args.exp_root / label
        metrics = render_and_score(
            variant=label,
            ckpt=ckpt,
            render_exp=render_exp,
            args=args,
            env=env,
            events_path=events_path,
            log_dir=log_dir,
        )
        dbase = _delta(metrics, baseline_metrics)
        strict = _accept_pass(dbase, args)
        row = {
            "label": label,
            "ckpt": str(ckpt),
            "render_exp": str(render_exp),
            "metrics": metrics,
            "delta_baseline": dbase,
            "strict_pass": strict,
        }
        rows.append(row)
        with train_summary.open("a", encoding="utf-8") as handle:
            handle.write(
                "\t".join(
                    [
                        label,
                        str(ckpt),
                        str(render_exp),
                        _fmt(metrics["fg"]),
                        _fmt(metrics["boundary"]),
                        _fmt(metrics["edge"], 6),
                        _fmt(metrics["inner"], 4),
                        _fmt(metrics["outer"], 4),
                        _fmt(metrics["hard"]),
                        _fmt(dbase["fg"]),
                        _fmt(dbase["boundary"]),
                        _fmt(dbase["edge"], 6),
                        _fmt(dbase["inner"], 4),
                        _fmt(dbase["outer"], 4),
                        _fmt(dbase["hard"]),
                        "1" if strict else "0",
                    ]
                )
                + "\n"
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="v278 greedy set-level render-in-loop support selector.")
    parser.add_argument("--python-bin", type=Path, default=Path("/opt/miniconda3/envs/ictrl/bin/python"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--base-exp", type=Path, default=ROOT / "exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt")
    parser.add_argument("--base-ckpt", type=Path, default=None)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/ZJUMoCap")
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--max-accept", type=int, default=8)
    parser.add_argument("--append-iter", type=int, default=136411)
    parser.add_argument("--parent-screen-radius", type=float, default=42.0)
    parser.add_argument("--child-opacity-factor", type=float, default=0.80)
    parser.add_argument("--child-opacity-floor", type=float, default=0.040)
    parser.add_argument("--child-opacity-ceiling", type=float, default=0.32)
    parser.add_argument("--child-scale-factor", type=float, default=0.55)
    parser.add_argument("--child-scale-max", type=float, default=0.008)
    parser.add_argument("--min-inner-gain", type=float, default=0.05)
    parser.add_argument("--min-hard-gain", type=float, default=0.000001)
    parser.add_argument("--max-outer-worsen", type=float, default=0.0)
    parser.add_argument("--max-fg-worsen", type=float, default=0.0)
    parser.add_argument("--max-boundary-worsen", type=float, default=0.0)
    parser.add_argument("--max-edge-worsen", type=float, default=0.0)
    parser.add_argument("--do-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-iterations", type=int, default=200)
    parser.add_argument("--train-checkpoint-steps", default="100,200")
    args = parser.parse_args()

    args.base_exp = args.base_exp.resolve()
    args.base_ckpt = (args.base_ckpt or (args.base_exp / "ckpt136410.pth")).resolve()
    args.candidate_csv = args.candidate_csv.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.exp_root = args.exp_root.resolve()
    args.log_dir = args.log_dir.resolve()
    args.exp_root.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "hydra_runtime").mkdir(parents=True, exist_ok=True)

    for required in (args.python_bin, args.base_exp / ".hydra" / "config.yaml", args.base_ckpt, args.candidate_csv, args.dataset_root):
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
    start_bjt = _bjt_now()
    run_info = {
        "run_id": args.run_id,
        "start_bjt": start_bjt,
        "gpu": args.gpu,
        "base_exp": str(args.base_exp),
        "base_ckpt": str(args.base_ckpt),
        "candidate_csv": str(args.candidate_csv),
        "exp_root": str(args.exp_root),
        "log_dir": str(args.log_dir),
        "gate": {
            "min_inner_gain": args.min_inner_gain,
            "min_hard_gain": args.min_hard_gain,
            "max_outer_worsen": args.max_outer_worsen,
            "max_fg_worsen": args.max_fg_worsen,
            "max_boundary_worsen": args.max_boundary_worsen,
            "max_edge_worsen": args.max_edge_worsen,
        },
    }
    _write_json(args.log_dir / "run_info.json", run_info)
    (args.log_dir / "run_info.txt").write_text("\n".join(f"{k}={v}" for k, v in run_info.items()) + "\n", encoding="utf-8")

    fieldnames, candidate_rows = _read_candidate_rows(args.candidate_csv, args.max_candidates)
    if not candidate_rows:
        raise RuntimeError(f"no candidates in {args.candidate_csv}")

    baseline_render = args.exp_root / "baseline"
    print(f"{_bjt_now()} baseline render", flush=True)
    baseline_metrics = render_and_score(
        variant="baseline",
        ckpt=args.base_ckpt,
        render_exp=baseline_render,
        args=args,
        env=env,
        events_path=events_path,
        log_dir=args.log_dir,
    )

    trials_tsv = args.log_dir / "greedy_trials.tsv"
    selected: list[int] = []
    remaining = list(range(len(candidate_rows)))
    current_ckpt = args.base_ckpt
    current_metrics = dict(baseline_metrics)
    accepted_history = []

    for round_idx in range(1, int(args.max_accept) + 1):
        print(f"{_bjt_now()} greedy round {round_idx} remaining={len(remaining)} selected={selected}", flush=True)
        round_results = []
        for cand_idx in list(remaining):
            subset_indices = selected + [cand_idx]
            variant = f"r{round_idx:02d}_cand{cand_idx:02d}"
            try:
                trial_ckpt = append_subset_checkpoint(
                    variant=variant,
                    subset_indices=subset_indices,
                    fieldnames=fieldnames,
                    candidate_rows=candidate_rows,
                    base_ckpt=args.base_ckpt,
                    out_dir=args.exp_root,
                    args=args,
                    env=env,
                    events_path=events_path,
                    log_dir=args.log_dir,
                )
                render_exp = args.exp_root / variant
                metrics = render_and_score(
                    variant=variant,
                    ckpt=trial_ckpt,
                    render_exp=render_exp,
                    args=args,
                    env=env,
                    events_path=events_path,
                    log_dir=args.log_dir,
                )
                dcur = _delta(metrics, current_metrics)
                dbase = _delta(metrics, baseline_metrics)
                accept = _accept_pass(dcur, args)
                status = "ok"
            except Exception as exc:
                trial_ckpt = Path("")
                render_exp = Path("")
                metrics = {key: float("nan") for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
                dcur = {key: float("nan") for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
                dbase = {key: float("nan") for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
                accept = False
                status = f"failed:{exc}"
            row = {
                "round": round_idx,
                "candidate_idx": cand_idx,
                "candidate_key": _candidate_key(candidate_rows[cand_idx], cand_idx),
                "subset_indices": json.dumps(subset_indices),
                "ckpt": str(trial_ckpt),
                "render_exp": str(render_exp),
                "fg": _fmt(metrics["fg"]) if metrics["fg"] == metrics["fg"] else "nan",
                "boundary": _fmt(metrics["boundary"]) if metrics["boundary"] == metrics["boundary"] else "nan",
                "edge": _fmt(metrics["edge"], 6) if metrics["edge"] == metrics["edge"] else "nan",
                "inner": _fmt(metrics["inner"], 4) if metrics["inner"] == metrics["inner"] else "nan",
                "outer": _fmt(metrics["outer"], 4) if metrics["outer"] == metrics["outer"] else "nan",
                "hard": _fmt(metrics["hard"]) if metrics["hard"] == metrics["hard"] else "nan",
                "fg_delta_current": _fmt(dcur["fg"]) if dcur["fg"] == dcur["fg"] else "nan",
                "boundary_delta_current": _fmt(dcur["boundary"]) if dcur["boundary"] == dcur["boundary"] else "nan",
                "edge_delta_current": _fmt(dcur["edge"], 6) if dcur["edge"] == dcur["edge"] else "nan",
                "inner_delta_current": _fmt(dcur["inner"], 4) if dcur["inner"] == dcur["inner"] else "nan",
                "outer_delta_current": _fmt(dcur["outer"], 4) if dcur["outer"] == dcur["outer"] else "nan",
                "hard_delta_current": _fmt(dcur["hard"]) if dcur["hard"] == dcur["hard"] else "nan",
                "fg_delta_baseline": _fmt(dbase["fg"]) if dbase["fg"] == dbase["fg"] else "nan",
                "boundary_delta_baseline": _fmt(dbase["boundary"]) if dbase["boundary"] == dbase["boundary"] else "nan",
                "edge_delta_baseline": _fmt(dbase["edge"], 6) if dbase["edge"] == dbase["edge"] else "nan",
                "inner_delta_baseline": _fmt(dbase["inner"], 4) if dbase["inner"] == dbase["inner"] else "nan",
                "outer_delta_baseline": _fmt(dbase["outer"], 4) if dbase["outer"] == dbase["outer"] else "nan",
                "hard_delta_baseline": _fmt(dbase["hard"]) if dbase["hard"] == dbase["hard"] else "nan",
                "accept_pass": "1" if accept else "0",
                "accepted": "0",
                "status": status,
            }
            _append_trial_row(trials_tsv, row)
            if accept and status == "ok":
                round_results.append(
                    {
                        "candidate_idx": cand_idx,
                        "subset_indices": subset_indices,
                        "ckpt": trial_ckpt,
                        "render_exp": render_exp,
                        "metrics": metrics,
                        "delta_current": dcur,
                        "delta_baseline": dbase,
                        "row": row,
                    }
                )
        if not round_results:
            print(f"{_bjt_now()} greedy stop: no passing candidate in round {round_idx}", flush=True)
            break
        round_results.sort(key=lambda item: _score_delta(item["delta_current"]), reverse=True)
        best = round_results[0]
        selected = list(best["subset_indices"])
        remaining = [idx for idx in remaining if idx != int(best["candidate_idx"])]
        current_ckpt = Path(best["ckpt"])
        current_metrics = dict(best["metrics"])
        accepted_history.append(
            {
                "round": round_idx,
                "candidate_idx": int(best["candidate_idx"]),
                "candidate_key": _candidate_key(candidate_rows[int(best["candidate_idx"])], int(best["candidate_idx"])),
                "subset_indices": selected,
                "ckpt": str(current_ckpt),
                "render_exp": str(best["render_exp"]),
                "metrics": current_metrics,
                "delta_current": best["delta_current"],
                "delta_baseline": best["delta_baseline"],
            }
        )
        with events_path.open("a", encoding="utf-8") as events:
            events.write(f"{_bjt_now()}\taccept_round_{round_idx}\tcandidate={best['candidate_idx']} subset={selected}\n")

    selected_csv = args.log_dir / "v278_selected_candidates.csv"
    if selected:
        _write_subset_csv(selected_csv, fieldnames, [candidate_rows[idx] for idx in selected])
    else:
        _write_subset_csv(selected_csv, fieldnames, [])

    final_pass = bool(selected) and _gate_pass_against_baseline(current_metrics, baseline_metrics, args)
    train_rows = []
    train_status = "skipped_no_selected_or_gate"
    if args.do_train and selected and final_pass:
        train_status = "started"
        train_rows = run_short_train(
            selected_ckpt=current_ckpt,
            baseline_metrics=baseline_metrics,
            args=args,
            env=env,
            events_path=events_path,
            log_dir=args.log_dir,
        )
        train_status = "done"

    end_bjt = _bjt_now()
    summary = {
        "status": "ok",
        "start_bjt": start_bjt,
        "end_bjt": end_bjt,
        "baseline_metrics": baseline_metrics,
        "selected_indices": selected,
        "accepted_count": len(selected),
        "accepted_history": accepted_history,
        "final_ckpt": str(current_ckpt),
        "final_metrics": current_metrics,
        "final_delta_baseline": _delta(current_metrics, baseline_metrics),
        "final_strict_pass": final_pass,
        "selected_candidates_csv": str(selected_csv),
        "trials_tsv": str(trials_tsv),
        "train_status": train_status,
        "train_rows": train_rows,
    }
    _write_json(args.log_dir / "selection_summary.json", summary)
    with (args.log_dir / "run_info.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"END_BJT={end_bjt}\n")
        handle.write(f"SUMMARY_JSON={args.log_dir / 'selection_summary.json'}\n")
        handle.write(f"TRIALS_TSV={trials_tsv}\n")
        handle.write(f"SELECTED_CSV={selected_csv}\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
