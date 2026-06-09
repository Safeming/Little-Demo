#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def bjt_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S BJT", time.gmtime(time.time() + 8 * 3600))


def bjt_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S_bjt", time.gmtime(time.time() + 8 * 3600))


def bjt_after(seconds: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S BJT", time.gmtime(time.time() + 8 * 3600 + seconds))


def parse_image_name(image_name: str) -> tuple[int, int]:
    match = re.fullmatch(r"c(\d+)_f(\d+)", str(image_name or ""))
    if match is None:
        raise ValueError(f"bad image_name: {image_name!r}")
    return int(match.group(1)), int(match.group(2))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_event(path: Path, phase: str, detail: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{bjt_now()}\t{phase}\t{detail}\n")


def run_command(name: str, cmd: list[str], log_path: Path, env: dict[str, str], events_path: Path) -> None:
    append_event(events_path, f"{name}_start", str(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(str(part)) for part in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    append_event(events_path, f"{name}_done", f"status={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with status {proc.returncode}; see {log_path}")


def split_asset_by_pair(asset: dict, pair_ids: set[str]) -> dict:
    groups = [g for g in asset.get("action_groups", []) if str(g.get("pair_id", "")) in pair_ids]
    children = [c for c in asset.get("children", []) if str(c.get("pair_id", "")) in pair_ids]
    actions = [a for a in asset.get("actions", []) if str(a.get("pair_id", "")) in pair_ids]
    payload = {
        **{k: v for k, v in asset.items() if k not in ("action_groups", "children", "actions")},
        "action_groups": groups,
        "children": children,
        "actions": actions,
        "group_count": len(groups),
        "child_count": len(children),
        "action_count": len(actions),
    }
    return payload


def merge_asset(base_asset: dict, closure_asset: dict, closure_pair_ids: set[str], version: str, policy: str) -> dict:
    base_ids = {str(g.get("pair_id", "")) for g in base_asset.get("action_groups", [])}
    selected_ids = set(base_ids) | set(closure_pair_ids)
    closure_group_ids = {str(g.get("pair_id", "")) for g in closure_asset.get("action_groups", [])}
    base_groups = [g for g in base_asset.get("action_groups", []) if str(g.get("pair_id", "")) in base_ids - closure_group_ids]
    base_children = [c for c in base_asset.get("children", []) if str(c.get("pair_id", "")) in base_ids - closure_group_ids]
    base_actions = [a for a in base_asset.get("actions", []) if str(a.get("pair_id", "")) in base_ids - closure_group_ids]
    closure_groups = [g for g in closure_asset.get("action_groups", []) if str(g.get("pair_id", "")) in selected_ids]
    closure_children = [c for c in closure_asset.get("children", []) if str(c.get("pair_id", "")) in selected_ids]
    closure_actions = [a for a in closure_asset.get("actions", []) if str(a.get("pair_id", "")) in selected_ids]
    groups = base_groups + closure_groups
    children = base_children + closure_children
    actions = base_actions + closure_actions
    return {
        **{k: v for k, v in closure_asset.items() if k not in ("action_groups", "children", "actions")},
        "version": version,
        "policy": policy,
        "base_group_count": len(base_ids),
        "selected_closure_group_count": len(closure_pair_ids),
        "group_count": len(groups),
        "child_count": len(children),
        "action_count": len(actions),
        "action_groups": groups,
        "children": children,
        "actions": actions,
    }


def metric_paths(render_exp: Path) -> tuple[Path, Path, Path]:
    return (
        render_exp / "diagnostics/contours/contour_summary.json",
        render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json",
        render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json",
    )


def load_metrics(render_exp: Path) -> dict[str, float]:
    contour_path, residual_path, opacity_path = metric_paths(render_exp)
    contour = load_json(contour_path)
    residual = load_json(residual_path)
    opacity = load_json(opacity_path)
    return {
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
        "opacity_inner": float(opacity["mean_primary_opacity_inner_missing_pixels"]),
        "opacity_outer": float(opacity["mean_primary_opacity_outer_leak_pixels"]),
    }


def render_and_analyze(
    *,
    label: str,
    render_exp: Path,
    image_name: str,
    asset_json: Path,
    args: argparse.Namespace,
    env: dict[str, str],
    events_path: Path,
    log_dir: Path,
) -> None:
    view, frame = parse_image_name(image_name)
    next_frame = frame + 1
    render_log = log_dir / f"render_{label}.log"
    render_cmd = [
        args.python_bin,
        "render.py",
        "--config-path",
        str(args.base_exp / ".hydra"),
        "--config-name",
        "config",
        "mode=test",
        f"load_ckpt={args.candidate_ckpt}",
        f"exp_dir={render_exp}",
        f"dataset.root_dir={args.data_root}",
        "dataset.preload=false",
        "dataset.subject=CoreView_377",
        f"dataset.train_views={args.train_views_spec}",
        f"dataset.train_frames={args.train_frames_spec}",
        f"dataset.test_views.view=[{view}]",
        f"dataset.test_frames.view=[{frame},{next_frame},1]",
        "dataset.parsing_prior.enable=false",
        "dataset.parsing_prior.roi_enable=false",
        "export_interpretability=false",
        "export_semantic_editable_assets=false",
        "++export_opacity_maps=true",
        "++render_export_refine=false",
        f"++render_export_opacity_threshold={args.render_export_opacity_threshold}",
        f"hydra.run.dir={args.hydra_run_root / label}",
        "wandb_disable=true",
        "pipeline.compute_cov3D_python=true",
        "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard",
        "++pipeline.split_child_component_enable=true",
        f"++pipeline.split_child_component_asset_json={asset_json}",
        "++pipeline.split_child_component_action_required=false",
        f"++pipeline.split_child_component_opacity={args.child_opacity}",
        "++pipeline.split_child_component_radius_scale=1.0",
        "++pipeline.split_child_component_max_children=-1",
    ]
    run_command(f"render_{label}", render_cmd, render_log, env, events_path)
    run_command(
        f"contours_{label}",
        [
            args.python_bin,
            "tools/analyze_377_render_contours.py",
            "--render-exp",
            str(render_exp),
            "--dataset-root",
            str(args.data_root),
            "--subject",
            "CoreView_377",
            "--split-dir",
            "test-view",
            "--band-width",
            "7",
            "--topk",
            "16",
            "--out-dir",
            str(render_exp / "diagnostics/contours"),
        ],
        log_dir / f"contours_{label}.log",
        env,
        events_path,
    )
    run_command(
        f"boundary_{label}",
        [
            args.python_bin,
            "tools/analyze_377_boundary_residuals.py",
            "--render-exp",
            str(render_exp),
            "--dataset-root",
            str(args.data_root),
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
            str(render_exp / "diagnostics/boundary_residuals"),
        ],
        log_dir / f"boundary_{label}.log",
        env,
        events_path,
    )
    run_command(
        f"opacity_{label}",
        [
            args.python_bin,
            "tools/analyze_377_opacity_footprint.py",
            "--render-exp",
            str(render_exp),
            "--dataset-root",
            str(args.data_root),
            "--subject",
            "CoreView_377",
            "--split-dir",
            "test-view",
            "--render-support-threshold",
            "0.025",
            "--primary-opacity-threshold",
            "0.06",
            "--opacity-thresholds",
            "0.02,0.04,0.06,0.08,0.10",
            "--rgb-close-kernel",
            "5",
            "--opacity-close-kernel",
            "3",
            "--band-width",
            "7",
            "--search-band-width",
            "24",
            "--topk",
            "16",
            "--out-dir",
            str(render_exp / "diagnostics/opacity_footprint"),
        ],
        log_dir / f"opacity_{label}.log",
        env,
        events_path,
    )


def keep_status(delta: dict[str, float], args: argparse.Namespace) -> tuple[str, str]:
    has_gain = delta["inner"] <= -float(args.min_inner_gain)
    no_harm = (
        delta["fg"] <= float(args.max_fg_regress)
        and delta["boundary"] <= float(args.max_boundary_regress)
        and delta["edge"] <= float(args.max_edge_regress)
        and delta["inner"] <= 0.0
        and delta["outer"] <= float(args.max_outer_regress)
        and delta["hard"] <= float(args.max_hard_regress)
        and delta["opacity_inner"] <= float(args.max_opacity_inner_regress)
        and delta["opacity_outer"] <= float(args.max_opacity_outer_regress)
    )
    if has_gain and no_harm:
        return "keep", "inner_gain_raw_no_harm"
    if not has_gain:
        return "drop", "no_new_inner_gain"
    return "drop", "raw_regress"


def read_raw_gate_status(summary_path: Path, variant: str) -> str:
    status = "missing"
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("variant") == variant:
                status = row.get("status", "")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "/opt/miniconda3/envs/ictrl/bin/python"))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--cpu-threads-per-job", type=int, default=int(os.environ.get("CPU_THREADS_PER_JOB", "6")))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", f"v381_closure_raw_selector_{bjt_stamp()}"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ROOT / "data/ZJUMoCap")))
    parser.add_argument("--base-exp", type=Path, default=Path(os.environ.get("BASE_EXP", ROOT / "exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt")))
    parser.add_argument("--base-asset-json", type=Path, default=Path(os.environ.get("BASE_ASSET_JSON", ROOT / "exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json")))
    parser.add_argument("--closure-asset-json", type=Path, default=Path(os.environ.get("CLOSURE_ASSET_JSON", ROOT / "exp/stageB/logs/377_explicit_binding_v380_footprint_verified_closure_v380_footprint_verified_closure_20260528_135551_bjt/assets/v380_footprint_verified_closure_asset.json")))
    parser.add_argument("--candidate-ckpt", type=Path, default=Path(os.environ.get("CANDIDATE_CKPT", ROOT / "exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth")))
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--exp-root", type=Path, default=None)
    parser.add_argument("--hydra-run-root", type=Path, default=None)
    parser.add_argument("--max-actions", type=int, default=int(os.environ.get("MAX_ACTIONS", "144")))
    parser.add_argument("--min-inner-gain", type=float, default=float(os.environ.get("MIN_INNER_GAIN", "0.5")))
    parser.add_argument("--max-fg-regress", type=float, default=float(os.environ.get("MAX_FG_REGRESS", "0.000002")))
    parser.add_argument("--max-boundary-regress", type=float, default=float(os.environ.get("MAX_BOUNDARY_REGRESS", "0.000002")))
    parser.add_argument("--max-edge-regress", type=float, default=float(os.environ.get("MAX_EDGE_REGRESS", "0.001")))
    parser.add_argument("--max-outer-regress", type=float, default=float(os.environ.get("MAX_OUTER_REGRESS", "0.0")))
    parser.add_argument("--max-hard-regress", type=float, default=float(os.environ.get("MAX_HARD_REGRESS", "0.000001")))
    parser.add_argument("--max-opacity-inner-regress", type=float, default=float(os.environ.get("MAX_OPACITY_INNER_REGRESS", "0.0")))
    parser.add_argument("--max-opacity-outer-regress", type=float, default=float(os.environ.get("MAX_OPACITY_OUTER_REGRESS", "0.0")))
    parser.add_argument("--child-opacity", default=os.environ.get("CHILD_OPACITY", "0.04"))
    parser.add_argument("--train-on-strict-pass", default=os.environ.get("TRAIN_ON_STRICT_PASS", "true"))
    parser.add_argument("--train-steps", default=os.environ.get("TRAIN_STEPS", "2000"))
    parser.add_argument("--train-views-spec", default=os.environ.get("TRAIN_VIEWS_SPEC", "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"))
    parser.add_argument("--train-frames-spec", default=os.environ.get("TRAIN_FRAMES_SPEC", "[0,570,60]"))
    parser.add_argument("--render-export-opacity-threshold", default=os.environ.get("RENDER_EXPORT_OPACITY_THRESHOLD", "0.06"))
    args = parser.parse_args()

    args.log_dir = args.log_dir or ROOT / f"exp/stageB/logs/377_explicit_binding_v381_closure_raw_selector_{args.run_id}"
    args.exp_root = args.exp_root or ROOT / f"exp/stageB/377_explicit_binding_v381_closure_raw_selector_{args.run_id}"
    args.hydra_run_root = args.hydra_run_root or args.log_dir / "hydra_runtime"
    asset_dir = args.log_dir / "assets"
    trial_dir = asset_dir / "trials"
    events_path = args.log_dir / "events.tsv"
    validation_tsv = args.log_dir / "action_validation.tsv"
    summary_tsv = args.log_dir / "summary.tsv"
    selected_asset = asset_dir / "v381_selected_closure_raw_selector_asset.json"
    final_asset = asset_dir / "v381_final_closure_raw_selector_asset.json"
    raw_gate_log = args.log_dir / "v381_raw_gate.launcher.log"
    train_log = args.log_dir / "v381_semantic_train.launcher.log"

    for required in [
        Path(args.python_bin),
        args.base_exp / ".hydra/config.yaml",
        args.data_root,
        args.base_asset_json,
        args.closure_asset_json,
        args.candidate_ckpt,
    ]:
        if not required.exists():
            print(f"missing required path: {required}", file=sys.stderr)
            return 2

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.exp_root.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)
    events_path.write_text("time_bjt\tphase\tdetail\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": str(args.cpu_threads_per_job),
            "MKL_NUM_THREADS": str(args.cpu_threads_per_job),
            "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_job),
            "NUMEXPR_NUM_THREADS": str(args.cpu_threads_per_job),
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64"),
        }
    )

    append_event(events_path, "selector_start", args.run_id)
    base_asset = load_json(args.base_asset_json)
    closure_asset = load_json(args.closure_asset_json)
    base_pair_ids = {str(g.get("pair_id", "")) for g in base_asset.get("action_groups", [])}
    closure_groups = [
        g for g in closure_asset.get("action_groups", [])
        if str(g.get("pair_id", "")) and str(g.get("pair_id", "")) not in base_pair_ids
    ]
    closure_groups.sort(
        key=lambda g: (
            str(g.get("image_name", "")),
            -float(g.get("frame_score", 0.0) or 0.0),
            -float(g.get("residual_target_score", 0.0) or 0.0),
            str(g.get("pair_id", "")),
        )
    )
    if args.max_actions > 0:
        closure_groups = closure_groups[: args.max_actions]

    control_assets: dict[str, Path] = {}
    control_exps: dict[str, Path] = {}
    rows: list[dict[str, str]] = []
    selected_ids: set[str] = set()

    fieldnames = [
        "pair_id",
        "image_name",
        "status",
        "reason",
        "fg_delta_control",
        "boundary_delta_control",
        "edge_delta_control",
        "inner_delta_control",
        "outer_delta_control",
        "hard_delta_control",
        "opacity_inner_delta_control",
        "opacity_outer_delta_control",
        "control_exp",
        "candidate_exp",
    ]
    with validation_tsv.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t").writeheader()

    for index, group in enumerate(closure_groups):
        pair_id = str(group.get("pair_id", ""))
        image_name = str(group.get("image_name", ""))
        safe_pair = re.sub(r"[^A-Za-z0-9_.-]+", "_", pair_id)[:160]
        append_event(events_path, "action_validate_start", f"{index + 1}/{len(closure_groups)} {pair_id}")

        if image_name not in control_exps:
            control_asset = asset_dir / f"control_v374_{image_name}.json"
            write_json(control_asset, merge_asset(base_asset, closure_asset, set(), "v381_control_v374_asset", "v374 control asset for marginal closure validation"))
            control_assets[image_name] = control_asset
            control_exp = args.exp_root / "action_validation" / image_name / "control_v374"
            render_and_analyze(
                label=f"control_{image_name}",
                render_exp=control_exp,
                image_name=image_name,
                asset_json=control_asset,
                args=args,
                env=env,
                events_path=events_path,
                log_dir=args.log_dir,
            )
            control_exps[image_name] = control_exp

        trial_asset = trial_dir / f"{safe_pair}.json"
        write_json(
            trial_asset,
            merge_asset(
                base_asset,
                closure_asset,
                {pair_id},
                "v381_single_closure_trial_asset",
                "v374 plus one footprint-verified closure action for raw marginal validation",
            ),
        )
        candidate_exp = args.exp_root / "action_validation" / image_name / safe_pair / "candidate"
        render_and_analyze(
            label=f"candidate_{index:03d}_{safe_pair}",
            render_exp=candidate_exp,
            image_name=image_name,
            asset_json=trial_asset,
            args=args,
            env=env,
            events_path=events_path,
            log_dir=args.log_dir,
        )
        control = load_metrics(control_exps[image_name])
        candidate = load_metrics(candidate_exp)
        delta = {key: candidate[key] - control[key] for key in control}
        status, reason = keep_status(delta, args)
        if status == "keep":
            selected_ids.add(pair_id)
        row = {
            "pair_id": pair_id,
            "image_name": image_name,
            "status": status,
            "reason": reason,
            **{f"{key}_delta_control": f"{delta[key]:.8f}" for key in delta},
            "control_exp": str(control_exps[image_name]),
            "candidate_exp": str(candidate_exp),
        }
        rows.append(row)
        with validation_tsv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writerow(row)
        append_event(events_path, "action_validate_done", f"{pair_id} status={status} reason={reason}")

    selected_payload = merge_asset(
        base_asset,
        closure_asset,
        selected_ids,
        "v381_selected_closure_raw_selector_asset",
        "v374 plus footprint-verified closure actions that pass single-action raw marginal validation.",
    )
    selected_payload["action_validation"] = {
        "validated_closure_group_count": len(rows),
        "selected_closure_group_count": len(selected_ids),
        "validation_tsv": str(validation_tsv),
    }
    write_json(selected_asset, selected_payload)
    write_json(final_asset, selected_payload)
    append_event(events_path, "asset_done", f"{final_asset} selected_closures={len(selected_ids)}")

    raw_gate_status = "not_run"
    raw_gate_summary = ""
    raw_gate_run_id = f"formal_377_v381_closure_raw_selector_raw_gate_{bjt_stamp()}"
    variant_name = "candidate_v381_closure_raw_selector"
    append_event(events_path, "raw_gate_start", raw_gate_run_id)
    raw_env = os.environ.copy()
    raw_env.update(
        {
            "GPU": str(args.gpu),
            "PYTHON_BIN": args.python_bin,
            "CPU_THREADS_PER_JOB": str(args.cpu_threads_per_job),
            "BASE_CKPT": str(args.base_exp / "ckpt136410.pth"),
            "CANDIDATE_CKPT": str(args.candidate_ckpt),
            "CANDIDATE_VARIANT_NAME": variant_name,
            "CANDIDATE_SPLIT_CHILD_COMPONENT_ENABLE": "true",
            "CANDIDATE_SPLIT_CHILD_COMPONENT_ASSET_JSON": str(final_asset),
            "CANDIDATE_SPLIT_CHILD_COMPONENT_ACTION_REQUIRED": "false",
            "CANDIDATE_SPLIT_CHILD_COMPONENT_OPACITY": str(args.child_opacity),
            "CANDIDATE_SPLIT_CHILD_COMPONENT_RADIUS_SCALE": "1.0",
            "CANDIDATE_SPLIT_CHILD_COMPONENT_MAX_CHILDREN": "-1",
            "RUN_ID": raw_gate_run_id,
        }
    )
    for key in ("LOG_DIR", "EXP_ROOT", "HYDRA_RUN_ROOT"):
        raw_env.pop(key, None)
    run_command("raw_gate", [str(ROOT / "tools/formal/run_377_v338_raw_contour_gate.sh")], raw_gate_log, raw_env, events_path)
    raw_gate_summary_path = ROOT / f"exp/formal/logs/377_v338_raw_contour_gate_{raw_gate_run_id}/summary.tsv"
    raw_gate_summary = str(raw_gate_summary_path)
    raw_gate_status = read_raw_gate_status(raw_gate_summary_path, variant_name)
    append_event(events_path, "raw_gate_done", f"status={raw_gate_status} summary={raw_gate_summary}")

    train_pid = ""
    train_exp_dir = ""
    train_est_end = ""
    if str(args.train_on_strict_pass).strip().lower() == "true" and raw_gate_status == "strict_pass":
        train_run_id = f"formal_377_v381_closure_raw_selector_semantic_train_{bjt_stamp()}"
        train_exp_dir = str(ROOT / f"exp/formal/377_v381_closure_raw_selector_semantic_train_{train_run_id}")
        train_script = args.log_dir / "v381_semantic_train.launch.sh"
        train_script.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"cd {shlex.quote(str(ROOT))}\n"
            "env -u LOG_DIR -u EXP_ROOT -u HYDRA_RUN_ROOT "
            f"GPU={shlex.quote(str(args.gpu))} "
            f"PYTHON_BIN={shlex.quote(args.python_bin)} "
            f"CPU_THREADS_PER_JOB={shlex.quote(str(args.cpu_threads_per_job))} "
            f"BASE_CKPT={shlex.quote(str(args.candidate_ckpt))} "
            f"RUN_ID={shlex.quote(train_run_id)} "
            f"EXP_DIR={shlex.quote(train_exp_dir)} "
            f"TRAIN_STEPS={shlex.quote(str(args.train_steps))} "
            f"{shlex.quote(str(ROOT / 'tools/formal/run_377_v338_semantic_train.sh'))} "
            f"++pipeline.split_child_component_enable=true "
            f"++pipeline.split_child_component_asset_json={shlex.quote(str(final_asset))} "
            f"++pipeline.split_child_component_action_required=false "
            f"++pipeline.split_child_component_opacity={shlex.quote(str(args.child_opacity))} "
            f"++pipeline.split_child_component_radius_scale=1.0 "
            f"++pipeline.split_child_component_max_children=-1\n",
            encoding="utf-8",
        )
        train_script.chmod(0o755)
        append_event(events_path, "train_start", train_exp_dir)
        proc = subprocess.Popen(
            ["setsid", "-f", str(train_script)],
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=train_log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        time.sleep(2)
        pid_proc = subprocess.run(["pgrep", "-f", str(train_script)], capture_output=True, text=True)
        train_pid = (pid_proc.stdout.strip().splitlines() or [str(proc.pid)])[-1]
        (args.log_dir / "train.pid").write_text(train_pid + "\n", encoding="utf-8")
        train_est_end = bjt_after(65 * 60)
        append_event(events_path, "train_launched", f"pid={train_pid} est_end={train_est_end}")
    else:
        append_event(events_path, "train_skip", f"raw_gate_status={raw_gate_status}")

    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "validated_closures",
                "selected_closures",
                "base_groups",
                "final_groups",
                "raw_gate_status",
                "raw_gate_summary",
                "train_pid",
                "train_exp_dir",
                "train_est_end_bjt",
                "validation_tsv",
                "final_asset_json",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "validated_closures": len(rows),
                "selected_closures": len(selected_ids),
                "base_groups": len(base_pair_ids),
                "final_groups": int(selected_payload.get("group_count", 0) or 0),
                "raw_gate_status": raw_gate_status,
                "raw_gate_summary": raw_gate_summary,
                "train_pid": train_pid,
                "train_exp_dir": train_exp_dir,
                "train_est_end_bjt": train_est_end,
                "validation_tsv": str(validation_tsv),
                "final_asset_json": str(final_asset),
            }
        )
    append_event(events_path, "finished_bjt", bjt_now())

    print(f"LOG_DIR={args.log_dir}")
    print(f"EXP_ROOT={args.exp_root}")
    print(f"SUMMARY={summary_tsv}")
    print(f"ACTION_VALIDATION_TSV={validation_tsv}")
    print(f"FINAL_ASSET_JSON={final_asset}")
    print(f"RAW_GATE_STATUS={raw_gate_status}")
    print(f"RAW_GATE_SUMMARY={raw_gate_summary}")
    print(f"TRAIN_PID={train_pid}")
    print(f"TRAIN_EXP_DIR={train_exp_dir}")
    print(f"TRAIN_EST_END_BJT={train_est_end}")
    print(f"END_BJT={bjt_now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
