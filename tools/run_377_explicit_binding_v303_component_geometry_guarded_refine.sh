#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v303_component_geometry_guarded_refine_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
DO_TRAIN="${DO_TRAIN:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-200}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-100,200}"
BASE_ITER="${BASE_ITER:-136410}"
TRAIN_FEATURE_LR="${TRAIN_FEATURE_LR:-0.00010}"
TRAIN_TEXTURE_LR="${TRAIN_TEXTURE_LR:-0.00000020}"
TRAIN_TEX_LATENT_LR="${TRAIN_TEX_LATENT_LR:-0.0}"
TRAIN_TEXTURE_TRAINABLE_PATTERNS="${TRAIN_TEXTURE_TRAINABLE_PATTERNS:-[*]}"
TRAIN_TEXTURE_FROZEN_PATTERNS="${TRAIN_TEXTURE_FROZEN_PATTERNS:-}"
TRAIN_LAMBDA_L1="${TRAIN_LAMBDA_L1:-0.040}"
TRAIN_LAMBDA_L1_FG="${TRAIN_LAMBDA_L1_FG:-0.110}"
TRAIN_LAMBDA_L1_BOUNDARY="${TRAIN_LAMBDA_L1_BOUNDARY:-0.050}"
TRAIN_LAMBDA_PERCEPTUAL="${TRAIN_LAMBDA_PERCEPTUAL:-0.010}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.0015}"
TRAIN_SCREEN_COLOR_PROTECT_ENABLE="${TRAIN_SCREEN_COLOR_PROTECT_ENABLE:-false}"
TRAIN_SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH="${TRAIN_SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH:-15}"
TRAIN_SCREEN_COLOR_PROTECT_OUTER_START="${TRAIN_SCREEN_COLOR_PROTECT_OUTER_START:-1}"
TRAIN_SCREEN_COLOR_PROTECT_OUTER_END="${TRAIN_SCREEN_COLOR_PROTECT_OUTER_END:-34}"
TRAIN_SCREEN_COLOR_PROTECT_RADIUS_PAD="${TRAIN_SCREEN_COLOR_PROTECT_RADIUS_PAD:-2}"
TRAIN_SCREEN_COLOR_PROTECT_MIN_RADIUS="${TRAIN_SCREEN_COLOR_PROTECT_MIN_RADIUS:-0.0}"
TRAIN_SCREEN_COLOR_PROTECT_MAX_POINTS="${TRAIN_SCREEN_COLOR_PROTECT_MAX_POINTS:-0}"
TRAIN_BOUNDARY_COLOR_PROTECT_ENABLE="${TRAIN_BOUNDARY_COLOR_PROTECT_ENABLE:-false}"
TRAIN_BOUNDARY_COLOR_PROTECT_VERBOSE="${TRAIN_BOUNDARY_COLOR_PROTECT_VERBOSE:-false}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"
PIPELINE_COMPONENT_REQUIRED="${PIPELINE_COMPONENT_REQUIRED:-false}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,1]}"

OVER_JOINT_IDS="${OVER_JOINT_IDS:-6,9,12,13,14,15}"
UNDER_LAYER_IDS="${UNDER_LAYER_IDS:-soft,rigid,free}"
UNDER_REGION_IDS="${UNDER_REGION_IDS:-cloth,body,soft}"
UNDER_JOINT_IDS="${UNDER_JOINT_IDS:-0,1,2,4,7,8,10}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v303_component_geometry_guarded_refine_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v303_component_geometry_guarded_refine_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
PARAMS="$LOG_DIR/no_train_params.tsv"
SUMMARY="$LOG_DIR/no_train_summary.tsv"
TRAIN_SUMMARY="$LOG_DIR/train_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
SELECTED_ENV="$LOG_DIR/selected_variant.env"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

ACTUAL_BASE_ITER="$("$PYTHON_BIN" - "$BASE_CKPT" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt[-1]))
PY
)"
if [ "$ACTUAL_BASE_ITER" != "$BASE_ITER" ]; then
  echo "BASE_ITER mismatch: BASE_ITER=$BASE_ITER but checkpoint iteration=$ACTUAL_BASE_ITER ($BASE_CKPT)" >&2
  exit 2
fi

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-4200}"
EST_END_EPOCH="$((START_EPOCH + EST_SECONDS))"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$EST_END_EPOCH" '+%F %T BJT')"

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\touter_px\tgeom_enable\tcenter_strength\tboundary_min\tcomponent_enable\tcomponent_required\tcomponent_pad\tcomponent_scale\tcomponent_max\tcomponent_min_area\timprove_enable\timprove_margin_px\tlayer_ids\tregion_ids\tjoint_ids\tnr_min\tpower\tmax_points\trender_exp\n' > "$PARAMS"
printf 'label\tvariant\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta_v281\tboundary_delta_v281\tedge_delta_v281\tinner_delta_v281\touter_delta_v281\thard_delta_v281\tstrict_pass\tprobe_pass\tstatus\n' > "$TRAIN_SUMMARY"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
BASE_ITER=$BASE_ITER
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
DO_TRAIN=$DO_TRAIN
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS
PIPELINE_COMPONENT_REQUIRED=$PIPELINE_COMPONENT_REQUIRED
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TRAIN_FEATURE_LR=$TRAIN_FEATURE_LR
TRAIN_TEXTURE_LR=$TRAIN_TEXTURE_LR
TRAIN_TEX_LATENT_LR=$TRAIN_TEX_LATENT_LR
TRAIN_TEXTURE_TRAINABLE_PATTERNS=$TRAIN_TEXTURE_TRAINABLE_PATTERNS
TRAIN_TEXTURE_FROZEN_PATTERNS=$TRAIN_TEXTURE_FROZEN_PATTERNS
TRAIN_LAMBDA_L1=$TRAIN_LAMBDA_L1
TRAIN_LAMBDA_L1_FG=$TRAIN_LAMBDA_L1_FG
TRAIN_LAMBDA_L1_BOUNDARY=$TRAIN_LAMBDA_L1_BOUNDARY
TRAIN_LAMBDA_PERCEPTUAL=$TRAIN_LAMBDA_PERCEPTUAL
TRAIN_GRAD_CLIP=$TRAIN_GRAD_CLIP
TRAIN_SCREEN_COLOR_PROTECT_ENABLE=$TRAIN_SCREEN_COLOR_PROTECT_ENABLE
TRAIN_SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH=$TRAIN_SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH
TRAIN_SCREEN_COLOR_PROTECT_OUTER_START=$TRAIN_SCREEN_COLOR_PROTECT_OUTER_START
TRAIN_SCREEN_COLOR_PROTECT_OUTER_END=$TRAIN_SCREEN_COLOR_PROTECT_OUTER_END
TRAIN_SCREEN_COLOR_PROTECT_RADIUS_PAD=$TRAIN_SCREEN_COLOR_PROTECT_RADIUS_PAD
TRAIN_SCREEN_COLOR_PROTECT_MIN_RADIUS=$TRAIN_SCREEN_COLOR_PROTECT_MIN_RADIUS
TRAIN_SCREEN_COLOR_PROTECT_MAX_POINTS=$TRAIN_SCREEN_COLOR_PROTECT_MAX_POINTS
TRAIN_BOUNDARY_COLOR_PROTECT_ENABLE=$TRAIN_BOUNDARY_COLOR_PROTECT_ENABLE
TRAIN_BOUNDARY_COLOR_PROTECT_VERBOSE=$TRAIN_BOUNDARY_COLOR_PROTECT_VERBOSE

Goal:
  v303 keeps the v302b root-cause direction but adds a screen-space
  improvement guard: center-only geometry fidelity is applied only when the
  pre-binding/LBS target projects closer to the current inner residual
  component than the current explicit-bound center. This is meant to reject
  wrong local actuators instead of compensating with global parameter sweeps.

Gate:
  Compare against baseline_v281_screen_mid. Train only if no-train A/B produces
  a strict/probe pass. Training is short and freezes xyz/opacity/scale/rotation/
  boundary residuals; raw contour gate decides the result.
EOF

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" "$START_BJT" "$EST_END_BJT" <<'PY' || true
import json
import sys
import time
from pathlib import Path

path, run_id, gpu, phase, detail, start_bjt, est_end_bjt = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_bjt": start_bjt,
    "est_end_bjt": est_end_bjt,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
  write_status "$1" "$2"
}

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

render_raw() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  local outer_px="$4"
  local geom_enable="$5"
  local center_strength="$6"
  local boundary_min="$7"
  local component_enable="$8"
  local component_required="$9"
  local component_pad="${10}"
  local component_scale="${11}"
  local component_max="${12}"
  local component_min_area="${13}"
  local improve_enable="${14}"
  local improve_margin_px="${15}"
  local layer_ids="${16}"
  local region_ids="${17}"
  local joint_ids="${18}"
  local nr_min="${19}"
  local power="${20}"
  local max_points="${21}"
  local hydra_dir="${22}"

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=true" \
    "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
    "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
    "++pipeline.covariance_signed_dynamic_component_signature_enable=false" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=10" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=$component_required" \
    "++pipeline.covariance_signed_dynamic_component_top_ids_enable=false" \
    "++pipeline.covariance_signed_dynamic_component_top_ids_only=false" \
    "++pipeline.covariance_signed_dynamic_max_over_points=96" \
    "++pipeline.covariance_signed_dynamic_max_under_points=96" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=1.025" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.covariance_signed_center_offset_enable=true" \
    "++pipeline.covariance_signed_center_offset_outer_px=$outer_px" \
    "++pipeline.covariance_signed_center_offset_inner_px=0.00" \
    "++pipeline.covariance_signed_center_offset_outer_direction=view_center" \
    "++pipeline.covariance_signed_center_offset_inner_direction=component_center" \
    "++pipeline.covariance_signed_center_offset_score_weight_power=1.0" \
    "++pipeline.covariance_signed_center_offset_score_weight_min=0.15" \
    "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90" \
    "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001" \
    "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001" \
    "++pipeline.covariance_signed_center_offset_max_world_step=0.0020" \
    "++pipeline.boundary_cov_residual_enable=false" \
    "++pipeline.binding_covariance_guard_enable=false" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=$geom_enable" \
    "++model.deformer.rigid.geometry_fidelity_target=free_lbs" \
    "++model.deformer.rigid.geometry_fidelity_center_strength=$center_strength" \
    "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0" \
    "++model.deformer.rigid.geometry_fidelity_boundary_min=$boundary_min" \
    "++model.deformer.rigid.geometry_fidelity_layer_ids='$layer_ids'" \
    "++model.deformer.rigid.geometry_fidelity_region_ids='$region_ids'" \
    "++model.deformer.rigid.geometry_fidelity_joint_ids='$joint_ids'" \
    "++model.deformer.rigid.geometry_fidelity_thin_min=''" \
    "++model.deformer.rigid.geometry_fidelity_surface_min=''" \
    "++model.deformer.rigid.geometry_fidelity_surface_max=''" \
    "++model.deformer.rigid.geometry_fidelity_non_rigid_min=$nr_min" \
    "++model.deformer.rigid.geometry_fidelity_power=$power" \
    "++model.deformer.rigid.geometry_fidelity_max_points=$max_points" \
    "++model.deformer.rigid.geometry_fidelity_component_enable=$component_enable" \
    "++model.deformer.rigid.geometry_fidelity_component_csv=$COMPONENT_CSV" \
    "++model.deformer.rigid.geometry_fidelity_component_direction=inner" \
    "++model.deformer.rigid.geometry_fidelity_component_pad_px=$component_pad" \
    "++model.deformer.rigid.geometry_fidelity_component_ellipse_scale=$component_scale" \
    "++model.deformer.rigid.geometry_fidelity_component_max=$component_max" \
    "++model.deformer.rigid.geometry_fidelity_component_min_area=$component_min_area" \
    "++model.deformer.rigid.geometry_fidelity_component_required=$component_required" \
    "++model.deformer.rigid.geometry_fidelity_component_improvement_enable=$improve_enable" \
    "++model.deformer.rigid.geometry_fidelity_component_improvement_margin_px=$improve_margin_px" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true"
}

analyze_raw() {
  local variant="$1"
  local render_exp="$2"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${variant}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --close-kernel 5 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 16 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${variant}.log" 2>&1
}

run_variant() {
  local variant="$1"
  local outer_px="$2"
  local geom_enable="$3"
  local center_strength="$4"
  local boundary_min="$5"
  local component_enable="$6"
  local component_required="$7"
  local component_pad="$8"
  local component_scale="$9"
  local component_max="${10}"
  local component_min_area="${11}"
  local improve_enable="${12}"
  local improve_margin_px="${13}"
  local layer_ids="${14}"
  local region_ids="${15}"
  local joint_ids="${16}"
  local nr_min="${17}"
  local power="${18}"
  local max_points="${19}"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$outer_px" "$geom_enable" "$center_strength" "$boundary_min" "$component_enable" \
    "$component_required" "$component_pad" "$component_scale" "$component_max" "$component_min_area" \
    "$improve_enable" "$improve_margin_px" "$layer_ids" "$region_ids" "$joint_ids" "$nr_min" "$power" \
    "$max_points" "$render_exp" \
    >> "$PARAMS"

  log_event "no_train_render_start" "$variant"
  render_raw "$variant" "$BASE_CKPT" "$render_exp" "$outer_px" "$geom_enable" "$center_strength" \
    "$boundary_min" "$component_enable" "$component_required" "$component_pad" "$component_scale" \
    "$component_max" "$component_min_area" "$improve_enable" "$improve_margin_px" "$layer_ids" \
    "$region_ids" "$joint_ids" "$nr_min" "$power" "$max_points" \
    "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "no_train_analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  log_event "no_train_variant_done" "$variant"
}

summarize_no_train() {
  "$PYTHON_BIN" - "$PARAMS" "$SUMMARY" "$SELECTED_ENV" <<'PY'
import csv
import json
import sys
from pathlib import Path

params_path, summary_path, selected_env_path = [Path(arg) for arg in sys.argv[1:4]]

def read_metrics(render_exp):
    render_exp = Path(render_exp)
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

def strict(delta):
    return (
        delta["inner"] < -0.05
        and delta["outer"] <= 0.0
        and delta["fg"] <= 0.0
        and delta["boundary"] <= 0.0
        and delta["edge"] <= 0.0
        and delta["hard"] < -0.000001
    )

def probe(delta):
    return (
        delta["hard"] < -0.00001
        and delta["fg"] <= 0.000015
        and delta["boundary"] <= 0.000015
        and delta["edge"] <= 0.003
        and delta["inner"] <= 0.5
        and delta["outer"] <= 0.5
    )

def status(delta):
    if strict(delta):
        return "strict_pass"
    if probe(delta):
        return "probe_pass"
    return "rejected"

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

rows = list(csv.DictReader(params_path.open("r", encoding="utf-8"), delimiter="\t"))
metrics_by_variant = {row["variant"]: read_metrics(row["render_exp"]) for row in rows}
baseline = metrics_by_variant["baseline_v281_screen_mid"]

header = [
    "variant", "outer_px", "geom_enable", "center_strength", "boundary_min",
    "component_enable", "component_required", "component_pad", "component_scale",
    "component_max", "component_min_area", "improve_enable", "improve_margin_px",
    "layer_ids", "region_ids", "joint_ids", "nr_min", "power", "max_points", "render_exp",
    "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_v281", "boundary_delta_v281", "edge_delta_v281", "inner_delta_v281",
    "outer_delta_v281", "hard_delta_v281", "strict_pass", "probe_pass", "v281_status",
]

best = None
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for row in rows:
        metrics = metrics_by_variant[row["variant"]]
        delta = {key: metrics[key] - baseline[key] for key in metrics}
        row_status = "baseline" if row["variant"] == "baseline_v281_screen_mid" else status(delta)
        is_strict = row_status == "strict_pass"
        is_probe = row_status in ("strict_pass", "probe_pass")
        if is_probe and row["variant"] != "baseline_v281_screen_mid":
            rank = 0 if is_strict else 1
            score = (
                delta["hard"] * 10000.0
                + delta["inner"] * 0.05
                + delta["outer"] * 0.03
                + delta["edge"] * 8.0
                + delta["fg"] * 1500.0
                + delta["boundary"] * 1200.0
            )
            candidate = (rank, score, row["variant"])
            if best is None or candidate < best[0]:
                best = (candidate, row, row_status)
        writer.writerow([
            row["variant"], row["outer_px"], row["geom_enable"], row["center_strength"], row["boundary_min"],
            row["component_enable"], row["component_required"], row["component_pad"], row["component_scale"],
            row["component_max"], row["component_min_area"], row["improve_enable"], row["improve_margin_px"],
            row["layer_ids"], row["region_ids"], row["joint_ids"], row["nr_min"], row["power"], row["max_points"],
            row["render_exp"],
            fmt(metrics["fg"]), fmt(metrics["boundary"]), fmt(metrics["edge"], 6), fmt(metrics["inner"], 4),
            fmt(metrics["outer"], 4), fmt(metrics["hard"]),
            fmt(delta["fg"]), fmt(delta["boundary"]), fmt(delta["edge"], 6),
            fmt(delta["inner"], 4), fmt(delta["outer"], 4), fmt(delta["hard"]),
            "1" if is_strict else "0",
            "1" if is_probe and row["variant"] != "baseline_v281_screen_mid" else "0",
            row_status,
        ])

if best is None:
    selected_env_path.write_text("SELECTED_VARIANT=\nSELECT_REASON=no_v281_gate_pass\n", encoding="utf-8")
    print("no_v281_gate_pass")
else:
    _, row, selected_status = best
    lines = [
        f"SELECTED_VARIANT={row['variant']}",
        f"SELECT_REASON={selected_status}",
    ]
    for key, value in row.items():
        safe_value = str(value).replace("'", "'\\''")
        lines.append(f"SELECTED_{key.upper()}='{safe_value}'")
    selected_env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"selected={row['variant']} reason={selected_status}")
PY
}

append_train_summary() {
  local label="$1"
  local variant="$2"
  local ckpt="$3"
  local render_exp="$4"
  "$PYTHON_BIN" - "$TRAIN_SUMMARY" "$SUMMARY" "$label" "$variant" "$ckpt" "$render_exp" <<'PY'
import json
import sys
from pathlib import Path

train_summary, no_train_summary, label, variant, ckpt, render_exp = sys.argv[1:7]
train_summary = Path(train_summary)
no_train_summary = Path(no_train_summary)
render_exp = Path(render_exp)

contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
metrics = {
    "fg": float(contour["mean_fg_l1"]),
    "boundary": float(contour["mean_boundary_l1"]),
    "edge": float(contour["mean_edge_symmetric_dist_px"]),
    "inner": float(residual["mean_inner_missing_pixels"]),
    "outer": float(residual["mean_outer_leak_pixels"]),
    "hard": float(residual["mean_hard_residual_score"]),
}
lines = [line.rstrip("\n").split("\t") for line in no_train_summary.read_text(encoding="utf-8").splitlines()]
header = lines[0]
baseline = None
for row in lines[1:]:
    if row and row[0] == "baseline_v281_screen_mid":
        baseline = {key: float(row[header.index(key)]) for key in metrics}
        break
if baseline is None:
    baseline = dict(metrics)
delta = {key: metrics[key] - baseline[key] for key in metrics}
strict = (
    delta["inner"] < -0.05
    and delta["outer"] <= 0.0
    and delta["fg"] <= 0.0
    and delta["boundary"] <= 0.0
    and delta["edge"] <= 0.0
    and delta["hard"] < -0.000001
)
probe = (
    delta["hard"] < -0.00001
    and delta["fg"] <= 0.000015
    and delta["boundary"] <= 0.000015
    and delta["edge"] <= 0.003
    and delta["inner"] <= 0.5
    and delta["outer"] <= 0.5
)
def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"
row = [
    label,
    variant,
    ckpt,
    str(render_exp),
    fmt(metrics["fg"]),
    fmt(metrics["boundary"]),
    fmt(metrics["edge"], 6),
    fmt(metrics["inner"], 4),
    fmt(metrics["outer"], 4),
    fmt(metrics["hard"]),
    fmt(delta["fg"]),
    fmt(delta["boundary"]),
    fmt(delta["edge"], 6),
    fmt(delta["inner"], 4),
    fmt(delta["outer"], 4),
    fmt(delta["hard"]),
    "1" if strict else "0",
    "1" if probe else "0",
    "strict_pass" if strict else ("probe_pass" if probe else "rejected"),
]
with train_summary.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

train_selected_variant() {
  source "$SELECTED_ENV"
  if [ -z "${SELECTED_VARIANT:-}" ]; then
    log_event "train_skip" "${SELECT_REASON:-no_gate_pass}"
    return
  fi
  if [ "$DO_TRAIN" != "1" ]; then
    log_event "train_skip" "DO_TRAIN=$DO_TRAIN selected=$SELECTED_VARIANT"
    return
  fi

  local train_exp="$EXP_ROOT/train_${SELECTED_VARIANT}"
  local checkpoint_list="[$TRAIN_CHECKPOINT_STEPS]"
  log_event "train_start" "$SELECTED_VARIANT reason=$SELECT_REASON exp=$train_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "start_checkpoint=$BASE_CKPT" \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$train_exp" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/train_${SELECTED_VARIANT}" \
    "mode=train" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_dynamic_enable=true" \
    "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
    "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
    "++pipeline.covariance_signed_dynamic_component_signature_enable=false" \
    "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
    "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
    "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
    "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
    "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
    "++pipeline.covariance_signed_dynamic_component_pad_px=10" \
    "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25" \
    "++pipeline.covariance_signed_dynamic_component_max_over=16" \
    "++pipeline.covariance_signed_dynamic_component_max_under=16" \
    "++pipeline.covariance_signed_dynamic_component_min_area=20" \
    "++pipeline.covariance_signed_dynamic_component_required=$SELECTED_COMPONENT_REQUIRED" \
    "++pipeline.covariance_signed_dynamic_component_top_ids_enable=false" \
    "++pipeline.covariance_signed_dynamic_component_top_ids_only=false" \
    "++pipeline.covariance_signed_dynamic_max_over_points=96" \
    "++pipeline.covariance_signed_dynamic_max_under_points=96" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=1.025" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.covariance_signed_center_offset_enable=true" \
    "++pipeline.covariance_signed_center_offset_outer_px=$SELECTED_OUTER_PX" \
    "++pipeline.covariance_signed_center_offset_inner_px=0.00" \
    "++pipeline.covariance_signed_center_offset_outer_direction=view_center" \
    "++pipeline.covariance_signed_center_offset_inner_direction=component_center" \
    "++pipeline.covariance_signed_center_offset_score_weight_power=1.0" \
    "++pipeline.covariance_signed_center_offset_score_weight_min=0.15" \
    "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90" \
    "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001" \
    "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001" \
    "++pipeline.covariance_signed_center_offset_max_world_step=0.0020" \
    "++pipeline.boundary_cov_residual_enable=false" \
    "++pipeline.binding_covariance_guard_enable=false" \
    "++resume.allow_start_load_ckpt_mismatch=false" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.clear_boundary_tags_on_resume=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.]" \
    "model.pose_correction.delay=1" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=$SELECTED_GEOM_ENABLE" \
    "++model.deformer.rigid.geometry_fidelity_target=free_lbs" \
    "++model.deformer.rigid.geometry_fidelity_center_strength=$SELECTED_CENTER_STRENGTH" \
    "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0" \
    "++model.deformer.rigid.geometry_fidelity_boundary_min=$SELECTED_BOUNDARY_MIN" \
    "++model.deformer.rigid.geometry_fidelity_layer_ids='$SELECTED_LAYER_IDS'" \
    "++model.deformer.rigid.geometry_fidelity_region_ids='$SELECTED_REGION_IDS'" \
    "++model.deformer.rigid.geometry_fidelity_joint_ids='$SELECTED_JOINT_IDS'" \
    "++model.deformer.rigid.geometry_fidelity_thin_min=''" \
    "++model.deformer.rigid.geometry_fidelity_surface_min=''" \
    "++model.deformer.rigid.geometry_fidelity_surface_max=''" \
    "++model.deformer.rigid.geometry_fidelity_non_rigid_min=$SELECTED_NR_MIN" \
    "++model.deformer.rigid.geometry_fidelity_power=$SELECTED_POWER" \
    "++model.deformer.rigid.geometry_fidelity_max_points=$SELECTED_MAX_POINTS" \
    "++model.deformer.rigid.geometry_fidelity_component_enable=$SELECTED_COMPONENT_ENABLE" \
    "++model.deformer.rigid.geometry_fidelity_component_csv=$COMPONENT_CSV" \
    "++model.deformer.rigid.geometry_fidelity_component_direction=inner" \
    "++model.deformer.rigid.geometry_fidelity_component_pad_px=$SELECTED_COMPONENT_PAD" \
    "++model.deformer.rigid.geometry_fidelity_component_ellipse_scale=$SELECTED_COMPONENT_SCALE" \
    "++model.deformer.rigid.geometry_fidelity_component_max=$SELECTED_COMPONENT_MAX" \
    "++model.deformer.rigid.geometry_fidelity_component_min_area=$SELECTED_COMPONENT_MIN_AREA" \
    "++model.deformer.rigid.geometry_fidelity_component_required=$SELECTED_COMPONENT_REQUIRED" \
    "++model.deformer.rigid.geometry_fidelity_component_improvement_enable=$SELECTED_IMPROVE_ENABLE" \
    "++model.deformer.rigid.geometry_fidelity_component_improvement_margin_px=$SELECTED_IMPROVE_MARGIN_PX" \
    "opt.iterations=$TRAIN_ITERS" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.feature_lr=$TRAIN_FEATURE_LR" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.pose_correction_lr=0.0" \
    "opt.texture_lr=$TRAIN_TEXTURE_LR" \
    "opt.tex_latent_lr=$TRAIN_TEX_LATENT_LR" \
    "++opt.texture_trainable_name_patterns=$TRAIN_TEXTURE_TRAINABLE_PATTERNS" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.stageB_semantic_loss_enable=false" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0" \
    "++opt.semantic_region_logits_lr=0.0" \
    "++opt.semantic_compact_logits_lr=0.0" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.018" \
    "++opt.train_sample_camera_max_prob=0.125" \
    "++opt.train_sample_log_interval=100" \
    "++opt.train_sample_accumulation_steps=1" \
    "opt.lambda_l1=$TRAIN_LAMBDA_L1" \
    "opt.lambda_l1_fg=$TRAIN_LAMBDA_L1_FG" \
    "opt.lambda_l1_boundary=$TRAIN_LAMBDA_L1_BOUNDARY" \
    "opt.lambda_perceptual=$TRAIN_LAMBDA_PERCEPTUAL" \
    "opt.lambda_mask=0.0" \
    "++opt.lambda_mask_boundary=0.0" \
    "++opt.lambda_silhouette_outer=0.0" \
    "++opt.lambda_silhouette_inner=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "opt.percent_dense=0.0" \
    "opt.densify_until_iter=0" \
    "opt.densify_from_iter=1000000" \
    "opt.opacity_reset_interval=1000000" \
    "best_eval_split=test" \
    "best_metric=l1_fg" \
    "best_metric_mode=min" \
    "best_metric_source=best_eval" \
    "test_interval=0" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=$TRAIN_GRAD_CLIP" \
    "++opt.screen_space_color_grad_protect_enable=$TRAIN_SCREEN_COLOR_PROTECT_ENABLE" \
    "++opt.screen_space_color_grad_protect_boundary_width=$TRAIN_SCREEN_COLOR_PROTECT_BOUNDARY_WIDTH" \
    "++opt.screen_space_color_grad_protect_outer_start_width=$TRAIN_SCREEN_COLOR_PROTECT_OUTER_START" \
    "++opt.screen_space_color_grad_protect_outer_end_width=$TRAIN_SCREEN_COLOR_PROTECT_OUTER_END" \
    "++opt.screen_space_color_grad_protect_radius_pad_px=$TRAIN_SCREEN_COLOR_PROTECT_RADIUS_PAD" \
    "++opt.screen_space_color_grad_protect_min_radius_px=$TRAIN_SCREEN_COLOR_PROTECT_MIN_RADIUS" \
    "++opt.screen_space_color_grad_protect_max_points=$TRAIN_SCREEN_COLOR_PROTECT_MAX_POINTS" \
    "++opt.boundary_color_grad_protect_enable=$TRAIN_BOUNDARY_COLOR_PROTECT_ENABLE" \
    "++opt.boundary_color_grad_protect_verbose=$TRAIN_BOUNDARY_COLOR_PROTECT_VERBOSE" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "wandb_disable=true" \
    > "$LOG_DIR/train_${SELECTED_VARIANT}.log" 2>&1
  log_event "train_done" "$train_exp"

  IFS=',' read -ra steps <<< "$TRAIN_CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    local global_iter=$((BASE_ITER + step))
    local ckpt="$train_exp/ckpt${global_iter}.pth"
    local label="ckpt${global_iter}"
    local render_exp="${train_exp}_raw_render_${label}"
    if [ ! -f "$ckpt" ]; then
      log_event "train_render_skip" "missing=$ckpt"
      continue
    fi
    log_event "train_render_start" "$label"
    render_raw "${SELECTED_VARIANT}_${label}" "$ckpt" "$render_exp" "$SELECTED_OUTER_PX" \
      "$SELECTED_GEOM_ENABLE" "$SELECTED_CENTER_STRENGTH" "$SELECTED_BOUNDARY_MIN" \
      "$SELECTED_COMPONENT_ENABLE" "$SELECTED_COMPONENT_REQUIRED" "$SELECTED_COMPONENT_PAD" \
      "$SELECTED_COMPONENT_SCALE" "$SELECTED_COMPONENT_MAX" "$SELECTED_COMPONENT_MIN_AREA" \
      "$SELECTED_IMPROVE_ENABLE" "$SELECTED_IMPROVE_MARGIN_PX" "$SELECTED_LAYER_IDS" \
      "$SELECTED_REGION_IDS" "$SELECTED_JOINT_IDS" "$SELECTED_NR_MIN" "$SELECTED_POWER" \
      "$SELECTED_MAX_POINTS" "$HYDRA_RUN_ROOT/render_${SELECTED_VARIANT}_${label}" \
      > "$LOG_DIR/render_${SELECTED_VARIANT}_${label}.log" 2>&1
    analyze_raw "${SELECTED_VARIANT}_${label}" "$render_exp"
    append_train_summary "$label" "$SELECTED_VARIANT" "$ckpt" "$render_exp"
    log_event "train_render_done" "$label"
  done
}

run_variant baseline_v281_screen_mid 0.00 false 0.00 0.08 false false 8 1.20 16 20 false 0.0 soft,free cloth,soft "" 0.000 1.0 -1
run_variant v302b_reference_025_outer035 0.35 true 0.25 0.12 true true 2 1.05 12 40 false 0.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_025_outer035_m0 0.35 true 0.25 0.12 true true 2 1.05 12 40 true 0.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_025_outer035_m1 0.35 true 0.25 0.12 true true 2 1.05 12 40 true 1.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_030_outer035_m0 0.35 true 0.30 0.12 true true 2 1.05 12 40 true 0.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_030_outer035_m1 0.35 true 0.30 0.12 true true 2 1.05 12 40 true 1.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_035_outer035_m0 0.35 true 0.35 0.12 true true 2 1.05 12 40 true 0.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_035_outer035_m1 0.35 true 0.35 0.12 true true 2 1.05 12 40 true 1.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_045_outer035_m0 0.35 true 0.45 0.12 true true 2 1.05 12 40 true 0.0 soft,free cloth,soft "" 0.000 1.2 1024
run_variant guard_035_outer025_m0 0.25 true 0.35 0.12 true true 2 1.05 12 40 true 0.0 soft,free cloth,soft "" 0.000 1.2 1024

summarize_no_train
train_selected_variant

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
  echo "SELECTED_ENV=$SELECTED_ENV"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
echo "SELECTED_ENV=$SELECTED_ENV"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
