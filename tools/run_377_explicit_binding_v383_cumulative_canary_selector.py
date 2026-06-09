#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from tools.run_377_explicit_binding_v381_closure_raw_selector import (
    ROOT,
    append_event,
    bjt_after,
    bjt_now,
    bjt_stamp,
    load_json,
    load_metrics,
    merge_asset,
    read_raw_gate_status,
    render_and_analyze,
    run_command,
    write_json,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160]


def read_prefilter_rows(path: Path, status: str) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if str(row.get("status", "")) == status and str(row.get("pair_id", "")):
                rows.append(row)
    return rows


def read_canary_images(path: Path, *, variant: str, top_k: int) -> list[str]:
    if not path.exists() or top_k <= 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            image = str(row.get("image", "")).strip()
            if not image:
                continue
            if variant and str(row.get("variant", "")) != variant:
                continue
            try:
                score = float(row.get("worsen_score", 0.0) or 0.0)
            except ValueError:
                score = 0.0
            rows.append((score, image))
    if not rows and variant:
        return read_canary_images(path, variant="", top_k=top_k)
    rows.sort(key=lambda item: (-item[0], item[1]))
    out = []
    seen = set()
    for _score, image in rows:
        if image in seen:
            continue
        seen.add(image)
        out.append(image)
        if len(out) >= top_k:
            break
    return out


def ordered_unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def no_harm(delta: dict[str, float], args: argparse.Namespace) -> tuple[bool, str]:
    checks = [
        ("fg", delta["fg"], float(args.max_fg_regress)),
        ("boundary", delta["boundary"], float(args.max_boundary_regress)),
        ("edge", delta["edge"], float(args.max_edge_regress)),
        ("inner", delta["inner"], float(args.max_inner_regress)),
        ("outer", delta["outer"], float(args.max_outer_regress)),
        ("hard", delta["hard"], float(args.max_hard_regress)),
        ("opacity_inner", delta["opacity_inner"], float(args.max_opacity_inner_regress)),
        ("opacity_outer", delta["opacity_outer"], float(args.max_opacity_outer_regress)),
    ]
    for name, value, limit in checks:
        if value > limit:
            return False, f"{name}_regress"
    return True, "no_harm"


def metric_delta(candidate_exp: Path, control_exp: Path) -> dict[str, float]:
    control = load_metrics(control_exp)
    candidate = load_metrics(candidate_exp)
    return {key: candidate[key] - control[key] for key in control}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "/opt/miniconda3/envs/ictrl/bin/python"))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--cpu-threads-per-job", type=int, default=int(os.environ.get("CPU_THREADS_PER_JOB", "6")))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", f"v383_cumulative_canary_selector_{bjt_stamp()}"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ROOT / "data/ZJUMoCap")))
    parser.add_argument("--base-exp", type=Path, default=Path(os.environ.get("BASE_EXP", ROOT / "exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt")))
    parser.add_argument("--base-asset-json", type=Path, default=Path(os.environ.get("BASE_ASSET_JSON", ROOT / "exp/stageB/logs/377_explicit_binding_v374_portfolio_merge_grouped_actuator_v374_v374_v376_queue_20260527_192801_bjt/assets/v374_portfolio_merge_grouped_actuator_asset.json")))
    parser.add_argument("--closure-asset-json", type=Path, default=Path(os.environ.get("CLOSURE_ASSET_JSON", ROOT / "exp/stageB/logs/377_explicit_binding_v382_post_v374_residual_bundle_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/assets/v382_post_v374_residual_bundle_candidate_asset.json")))
    parser.add_argument("--prefilter-tsv", type=Path, default=Path(os.environ.get("PREFILTER_TSV", ROOT / "exp/stageB/logs/377_explicit_binding_v381_closure_raw_selector_v382_selector_v382_post_v374_residual_bundle_selector_20260528_182940_bjt/action_validation.tsv")))
    parser.add_argument("--prefilter-status", default=os.environ.get("PREFILTER_STATUS", "keep"))
    parser.add_argument("--canary-worst-tsv", type=Path, default=Path(os.environ.get("CANARY_WORST_TSV", ROOT / "exp/formal/logs/377_v338_raw_contour_gate_formal_377_v381_closure_raw_selector_raw_gate_20260528_193641_bjt/worst_frames.tsv")))
    parser.add_argument("--canary-variant", default=os.environ.get("CANARY_VARIANT", "candidate_v381_closure_raw_selector"))
    parser.add_argument("--canary-top-k", type=int, default=int(os.environ.get("CANARY_TOP_K", "6")))
    parser.add_argument("--extra-canary-images", default=os.environ.get("EXTRA_CANARY_IMAGES", ""))
    parser.add_argument("--candidate-ckpt", type=Path, default=Path(os.environ.get("CANDIDATE_CKPT", ROOT / "exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth")))
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--exp-root", type=Path, default=None)
    parser.add_argument("--hydra-run-root", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=int(os.environ.get("MAX_CANDIDATES", "38")))
    parser.add_argument("--min-inner-gain", type=float, default=float(os.environ.get("MIN_INNER_GAIN", "0.5")))
    parser.add_argument("--max-fg-regress", type=float, default=float(os.environ.get("MAX_FG_REGRESS", "0.000002")))
    parser.add_argument("--max-boundary-regress", type=float, default=float(os.environ.get("MAX_BOUNDARY_REGRESS", "0.000002")))
    parser.add_argument("--max-edge-regress", type=float, default=float(os.environ.get("MAX_EDGE_REGRESS", "0.001")))
    parser.add_argument("--max-inner-regress", type=float, default=float(os.environ.get("MAX_INNER_REGRESS", "0.0")))
    parser.add_argument("--max-outer-regress", type=float, default=float(os.environ.get("MAX_OUTER_REGRESS", "0.0")))
    parser.add_argument("--max-hard-regress", type=float, default=float(os.environ.get("MAX_HARD_REGRESS", "0.000001")))
    parser.add_argument("--max-opacity-inner-regress", type=float, default=float(os.environ.get("MAX_OPACITY_INNER_REGRESS", "0.0")))
    parser.add_argument("--max-opacity-outer-regress", type=float, default=float(os.environ.get("MAX_OPACITY_OUTER_REGRESS", "0.0")))
    parser.add_argument("--child-opacity", default=os.environ.get("CHILD_OPACITY", "0.045"))
    parser.add_argument("--train-on-strict-pass", default=os.environ.get("TRAIN_ON_STRICT_PASS", "true"))
    parser.add_argument("--train-steps", default=os.environ.get("TRAIN_STEPS", "2000"))
    parser.add_argument("--train-views-spec", default=os.environ.get("TRAIN_VIEWS_SPEC", "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"))
    parser.add_argument("--train-frames-spec", default=os.environ.get("TRAIN_FRAMES_SPEC", "[0,570,60]"))
    parser.add_argument("--render-export-opacity-threshold", default=os.environ.get("RENDER_EXPORT_OPACITY_THRESHOLD", "0.06"))
    args = parser.parse_args()

    args.log_dir = args.log_dir or ROOT / f"exp/stageB/logs/377_explicit_binding_v383_cumulative_canary_selector_{args.run_id}"
    args.exp_root = args.exp_root or ROOT / f"exp/stageB/377_explicit_binding_v383_cumulative_canary_selector_{args.run_id}"
    args.hydra_run_root = args.hydra_run_root or args.log_dir / "hydra_runtime"
    asset_dir = args.log_dir / "assets"
    trial_dir = asset_dir / "trials"
    events_path = args.log_dir / "events.tsv"
    validation_tsv = args.log_dir / "cumulative_validation.tsv"
    summary_tsv = args.log_dir / "summary.tsv"
    final_asset = asset_dir / "v383_final_cumulative_canary_selector_asset.json"
    raw_gate_log = args.log_dir / "v383_raw_gate.launcher.log"
    train_log = args.log_dir / "v383_semantic_train.launcher.log"

    for required in [
        Path(args.python_bin),
        args.base_exp / ".hydra/config.yaml",
        args.data_root,
        args.base_asset_json,
        args.closure_asset_json,
        args.prefilter_tsv,
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
    group_by_id = {str(g.get("pair_id", "")): g for g in closure_asset.get("action_groups", [])}
    prefilter_rows = read_prefilter_rows(args.prefilter_tsv, args.prefilter_status)
    prefilter_rows = [row for row in prefilter_rows if row.get("pair_id", "") in group_by_id]
    prefilter_rows.sort(
        key=lambda row: (
            float(row.get("inner_delta_control", 0.0) or 0.0),
            float(row.get("outer_delta_control", 0.0) or 0.0),
            float(row.get("opacity_outer_delta_control", 0.0) or 0.0),
            str(row.get("pair_id", "")),
        )
    )
    if args.max_candidates > 0:
        prefilter_rows = prefilter_rows[: args.max_candidates]

    extra_canaries = [item.strip() for item in str(args.extra_canary_images).split(",") if item.strip()]
    canary_images = ordered_unique(
        read_canary_images(args.canary_worst_tsv, variant=args.canary_variant, top_k=args.canary_top_k)
        + extra_canaries
    )
    (args.log_dir / "canary_images.txt").write_text("\n".join(canary_images) + ("\n" if canary_images else ""), encoding="utf-8")
    append_event(events_path, "canary_images", ",".join(canary_images))

    selected_ids: set[str] = set()
    control_assets: dict[str, Path] = {}
    control_exps: dict[str, Path] = {}

    def ensure_control(image_name: str) -> Path:
        if image_name in control_exps:
            return control_exps[image_name]
        control_asset = asset_dir / f"control_v374_{image_name}.json"
        write_json(
            control_asset,
            merge_asset(base_asset, closure_asset, set(), "v383_control_v374_asset", "v374 control asset for cumulative canary validation"),
        )
        control_assets[image_name] = control_asset
        control_exp = args.exp_root / "cumulative_validation" / image_name / "control_v374"
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
        return control_exp

    fieldnames = [
        "pair_id",
        "image_name",
        "status",
        "reason",
        "validation_images",
        "source_inner_delta_control",
        "source_outer_delta_control",
        "source_hard_delta_control",
        "source_opacity_outer_delta_control",
        "worst_outer_delta_control",
        "worst_opacity_outer_delta_control",
        "worst_hard_delta_control",
        "selected_count_after",
        "trial_asset",
    ]
    with validation_tsv.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t").writeheader()

    for index, row in enumerate(prefilter_rows):
        pair_id = str(row.get("pair_id", ""))
        group = group_by_id[pair_id]
        image_name = str(group.get("image_name", row.get("image_name", "")) or row.get("image_name", ""))
        validation_images = ordered_unique([image_name] + canary_images)
        trial_ids = set(selected_ids) | {pair_id}
        safe_pair = safe_name(pair_id)
        append_event(events_path, "candidate_start", f"{index + 1}/{len(prefilter_rows)} {pair_id} images={','.join(validation_images)}")

        trial_asset = trial_dir / f"cumulative_{index:03d}_{safe_pair}.json"
        write_json(
            trial_asset,
            merge_asset(
                base_asset,
                closure_asset,
                trial_ids,
                "v383_cumulative_trial_asset",
                "v374 plus cumulatively selected residual bundle actions under canary validation",
            ),
        )

        deltas: dict[str, dict[str, float]] = {}
        bad_reasons: list[str] = []
        for image in validation_images:
            control_exp = ensure_control(image)
            candidate_exp = args.exp_root / "cumulative_validation" / image / safe_pair / "candidate"
            render_and_analyze(
                label=f"candidate_{index:03d}_{safe_pair}_{image}",
                render_exp=candidate_exp,
                image_name=image,
                asset_json=trial_asset,
                args=args,
                env=env,
                events_path=events_path,
                log_dir=args.log_dir,
            )
            delta = metric_delta(candidate_exp, control_exp)
            deltas[image] = delta
            ok, reason = no_harm(delta, args)
            if not ok:
                bad_reasons.append(f"{image}:{reason}")

        source_delta = deltas[image_name]
        source_has_gain = source_delta["inner"] <= -float(args.min_inner_gain)
        if source_has_gain and not bad_reasons:
            status = "keep"
            reason = "cumulative_source_gain_canary_no_harm"
            selected_ids.add(pair_id)
        elif not source_has_gain:
            status = "drop"
            reason = "no_cumulative_source_inner_gain"
        else:
            status = "drop"
            reason = ";".join(bad_reasons[:6])

        worst_outer = max(delta["outer"] for delta in deltas.values())
        worst_opacity_outer = max(delta["opacity_outer"] for delta in deltas.values())
        worst_hard = max(delta["hard"] for delta in deltas.values())
        out_row = {
            "pair_id": pair_id,
            "image_name": image_name,
            "status": status,
            "reason": reason,
            "validation_images": ",".join(validation_images),
            "source_inner_delta_control": f"{source_delta['inner']:.8f}",
            "source_outer_delta_control": f"{source_delta['outer']:.8f}",
            "source_hard_delta_control": f"{source_delta['hard']:.8f}",
            "source_opacity_outer_delta_control": f"{source_delta['opacity_outer']:.8f}",
            "worst_outer_delta_control": f"{worst_outer:.8f}",
            "worst_opacity_outer_delta_control": f"{worst_opacity_outer:.8f}",
            "worst_hard_delta_control": f"{worst_hard:.8f}",
            "selected_count_after": len(selected_ids),
            "trial_asset": str(trial_asset),
        }
        with validation_tsv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writerow(out_row)
        append_event(events_path, "candidate_done", f"{pair_id} status={status} reason={reason} selected={len(selected_ids)}")

    selected_payload = merge_asset(
        base_asset,
        closure_asset,
        selected_ids,
        "v383_cumulative_canary_selector_asset",
        "v374 plus residual bundle actions selected by cumulative greedy canary raw validation.",
    )
    selected_payload["cumulative_canary_validation"] = {
        "prefilter_tsv": str(args.prefilter_tsv),
        "canary_worst_tsv": str(args.canary_worst_tsv),
        "canary_images": canary_images,
        "validated_candidate_count": len(prefilter_rows),
        "selected_candidate_count": len(selected_ids),
        "validation_tsv": str(validation_tsv),
    }
    write_json(final_asset, selected_payload)
    append_event(events_path, "asset_done", f"{final_asset} selected={len(selected_ids)}")

    raw_gate_status = "not_run"
    raw_gate_summary = ""
    raw_gate_run_id = f"formal_377_v383_cumulative_canary_selector_raw_gate_{bjt_stamp()}"
    variant_name = "candidate_v383_cumulative_canary_selector"
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
    if (
        str(args.train_on_strict_pass).strip().lower() == "true"
        and raw_gate_status == "strict_pass"
        and len(selected_ids) > 0
    ):
        train_run_id = f"formal_377_v383_cumulative_canary_selector_semantic_train_{bjt_stamp()}"
        train_exp_dir = str(ROOT / f"exp/formal/377_v383_cumulative_canary_selector_semantic_train_{train_run_id}")
        train_script = args.log_dir / "v383_semantic_train.launch.sh"
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
        append_event(events_path, "train_skip", f"raw_gate_status={raw_gate_status} selected={len(selected_ids)}")

    base_pair_ids = {str(g.get("pair_id", "")) for g in base_asset.get("action_groups", [])}
    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "validated_candidates",
                "selected_candidates",
                "base_groups",
                "final_groups",
                "canary_images",
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
                "validated_candidates": len(prefilter_rows),
                "selected_candidates": len(selected_ids),
                "base_groups": len(base_pair_ids),
                "final_groups": int(selected_payload.get("group_count", 0) or 0),
                "canary_images": ",".join(canary_images),
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
    print(f"CUMULATIVE_VALIDATION_TSV={validation_tsv}")
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
