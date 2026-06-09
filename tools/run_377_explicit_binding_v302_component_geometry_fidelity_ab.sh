#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v302_component_geometry_fidelity_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"

OVER_JOINT_IDS="${OVER_JOINT_IDS:-6,9,12,13,14,15}"
UNDER_LAYER_IDS="${UNDER_LAYER_IDS:-soft,rigid,free}"
UNDER_REGION_IDS="${UNDER_REGION_IDS:-cloth,body,soft}"
UNDER_JOINT_IDS="${UNDER_JOINT_IDS:-0,1,2,4,7,8,10}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v302_component_geometry_fidelity_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v302_component_geometry_fidelity_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
PARAMS="$LOG_DIR/variant_params.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
SELECTED_ENV="$LOG_DIR/selected_variant.env"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\touter_px\tgeom_enable\tcenter_strength\tboundary_min\tcomponent_enable\tcomponent_required\tcomponent_pad\tcomponent_scale\tcomponent_max\tcomponent_min_area\tlayer_ids\tregion_ids\tjoint_ids\tnr_min\tpower\tmax_points\trender_exp\n' > "$PARAMS"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT

Goal:
  v302 narrows the v301 positive signal. v301 showed center-only geometry
  fidelity reduces inner/hard but slightly increases outer; rotation fidelity
  is destructive. This script applies center-only fidelity only when the
  candidate target projects into current-view inner residual components, and
  combines it with the existing outer center offset suppression.

Gate:
  Compare every variant against baseline_v281_screen_mid. No training is run
  unless a future guarded script consumes a strict/probe pass.
EOF

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
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
  local layer_ids="${12}"
  local region_ids="${13}"
  local joint_ids="${14}"
  local nr_min="${15}"
  local power="${16}"
  local max_points="${17}"
  local render_exp="${18}"
  local hydra_dir="${19}"

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
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
    "++pipeline.covariance_signed_dynamic_component_required=false" \
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
  local layer_ids="${12}"
  local region_ids="${13}"
  local joint_ids="${14}"
  local nr_min="${15}"
  local power="${16}"
  local max_points="${17}"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$outer_px" "$geom_enable" "$center_strength" "$boundary_min" "$component_enable" \
    "$component_required" "$component_pad" "$component_scale" "$component_max" "$component_min_area" \
    "$layer_ids" "$region_ids" "$joint_ids" "$nr_min" "$power" "$max_points" "$render_exp" \
    >> "$PARAMS"

  log_event "render_start" "$variant"
  render_raw "$variant" "$outer_px" "$geom_enable" "$center_strength" "$boundary_min" "$component_enable" \
    "$component_required" "$component_pad" "$component_scale" "$component_max" "$component_min_area" \
    "$layer_ids" "$region_ids" "$joint_ids" "$nr_min" "$power" "$max_points" \
    "$render_exp" "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  log_event "variant_done" "$variant"
}

run_variant baseline_v281_screen_mid 0.00 false 0.00 0.08 false false 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 -1
run_variant outer_center_025 0.25 false 0.00 0.08 false false 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 -1
run_variant outer_center_035 0.35 false 0.00 0.08 false false 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 -1
run_variant comp_inner_025_outer025 0.25 true 0.25 0.08 true true 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 2048
run_variant comp_inner_035_outer025 0.25 true 0.35 0.08 true true 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 2048
run_variant comp_inner_050_outer025 0.25 true 0.50 0.08 true true 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 2048
run_variant comp_inner_035_outer035 0.35 true 0.35 0.08 true true 8 1.20 16 20 soft,free cloth,soft "" 0.000 1.0 2048
run_variant comp_inner_tight_035_outer025 0.25 true 0.35 0.12 true true 2 1.05 12 40 soft,free cloth,soft "" 0.000 1.2 1024
run_variant comp_inner_tight_025_outer035 0.35 true 0.25 0.12 true true 2 1.05 12 40 soft,free cloth,soft "" 0.000 1.2 1024
run_variant comp_inner_tight_030_outer035 0.35 true 0.30 0.12 true true 2 1.05 12 40 soft,free cloth,soft "" 0.000 1.2 1024
run_variant comp_inner_tight_035_outer035 0.35 true 0.35 0.12 true true 2 1.05 12 40 soft,free cloth,soft "" 0.000 1.2 1024
run_variant comp_inner_tight_035_outer045 0.45 true 0.35 0.12 true true 2 1.05 12 40 soft,free cloth,soft "" 0.000 1.2 1024
run_variant comp_inner_tight_045_outer035 0.35 true 0.45 0.12 true true 2 1.05 12 40 soft,free cloth,soft "" 0.000 1.2 1024
run_variant comp_inner_nr_035_outer025 0.25 true 0.35 0.08 true true 8 1.20 16 20 soft,free cloth,soft "" 0.002 1.2 1536
run_variant comp_inner_underjoints_035_outer025 0.25 true 0.35 0.08 true true 8 1.20 16 20 soft,rigid,free cloth,body,soft "$UNDER_JOINT_IDS" 0.000 1.2 1536

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
        and delta["fg"] <= 0.00002
        and delta["boundary"] <= 0.00002
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
stable = metrics_by_variant["baseline_v281_screen_mid"]

header = [
    "variant", "outer_px", "geom_enable", "center_strength", "boundary_min",
    "component_enable", "component_required", "component_pad", "component_scale",
    "component_max", "component_min_area", "layer_ids", "region_ids", "joint_ids",
    "nr_min", "power", "max_points", "render_exp",
    "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_v281", "boundary_delta_v281", "edge_delta_v281", "inner_delta_v281",
    "outer_delta_v281", "hard_delta_v281", "v281_status",
]

best = None
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for row in rows:
        m = metrics_by_variant[row["variant"]]
        delta = {key: m[key] - stable[key] for key in m}
        vstatus = "baseline" if row["variant"] == "baseline_v281_screen_mid" else status(delta)
        if vstatus in ("strict_pass", "probe_pass"):
            score = (
                delta["hard"] * 10000.0
                + delta["inner"] * 0.03
                + delta["outer"] * 0.02
                + delta["edge"] * 10.0
                + delta["fg"] * 1000.0
                + delta["boundary"] * 1000.0
            )
            rank = 0 if vstatus == "strict_pass" else 1
            candidate = (rank, score, row["variant"])
            if best is None or candidate < best[0]:
                best = (candidate, row, vstatus)
        writer.writerow([
            row["variant"], row["outer_px"], row["geom_enable"], row["center_strength"], row["boundary_min"],
            row["component_enable"], row["component_required"], row["component_pad"], row["component_scale"],
            row["component_max"], row["component_min_area"], row["layer_ids"], row["region_ids"], row["joint_ids"],
            row["nr_min"], row["power"], row["max_points"], row["render_exp"],
            fmt(m["fg"]), fmt(m["boundary"]), fmt(m["edge"], 6), fmt(m["inner"], 4),
            fmt(m["outer"], 4), fmt(m["hard"]),
            fmt(delta["fg"]), fmt(delta["boundary"]), fmt(delta["edge"], 6),
            fmt(delta["inner"], 4), fmt(delta["outer"], 4), fmt(delta["hard"]),
            vstatus,
        ])

if best is None:
    selected_env_path.write_text("SELECTED_VARIANT=\nSELECT_REASON=no_v281_gate_pass\n", encoding="utf-8")
else:
    _, row, selected_status = best
    selected_env_path.write_text(
        f"SELECTED_VARIANT={row['variant']}\n"
        f"SELECT_REASON={selected_status}\n"
        f"SELECTED_RENDER_EXP={row['render_exp']}\n",
        encoding="utf-8",
    )

print(summary_path)
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "SELECTED_ENV=$SELECTED_ENV"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "SUMMARY=$SUMMARY"
echo "SELECTED_ENV=$SELECTED_ENV"
echo "END_BJT=$END_BJT"
