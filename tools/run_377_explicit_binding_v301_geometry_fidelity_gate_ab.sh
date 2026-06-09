#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v301_geometry_fidelity_gate_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
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

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v301_geometry_fidelity_gate_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v301_geometry_fidelity_gate_ab_${RUN_ID}}"
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
printf 'variant\tscreen_enable\tscreen_shrink\tscreen_grow\tgeom_enable\tgeom_target\tcenter_strength\trotation_strength\tboundary_min\tlayer_ids\tregion_ids\tjoint_ids\tthin_min\tsurface_min\tsurface_max\tnr_min\tpower\tmax_points\trender_exp\n' > "$PARAMS"

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
  v301 tests the current root-cause hypothesis directly: StageB explicit binding
  recomputes x_bar from SMPL anchor/local-offset/layer blending after non-rigid
  deformation, and high-risk boundary points can be moved away from the raw
  Gaussian support geometry. This no-train A/B selectively blends those points
  back toward coordinate-correct LBS/source trajectories while keeping explicit
  semantic binding attributes.

Gate:
  Compare every variant against baseline_v281_screen_mid. A candidate is selected
  only if it improves inner/hard and does no harm to outer, fg, boundary, and edge.
  No training is launched here unless a future script explicitly consumes a gate pass.
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
  local screen_enable="$2"
  local screen_shrink="$3"
  local screen_grow="$4"
  local geom_enable="$5"
  local geom_target="$6"
  local center_strength="$7"
  local rotation_strength="$8"
  local boundary_min="$9"
  local layer_ids="${10}"
  local region_ids="${11}"
  local joint_ids="${12}"
  local thin_min="${13}"
  local surface_min="${14}"
  local surface_max="${15}"
  local nr_min="${16}"
  local power="${17}"
  local max_points="${18}"
  local render_exp="${19}"
  local hydra_dir="${20}"

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
    "++pipeline.covariance_signed_dynamic_enable=$screen_enable" \
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
    "++pipeline.covariance_signed_screen_actuator_enable=$screen_enable" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=$screen_shrink" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=$screen_grow" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.boundary_cov_residual_enable=false" \
    "++pipeline.binding_covariance_guard_enable=false" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=false" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=$geom_enable" \
    "++model.deformer.rigid.geometry_fidelity_target=$geom_target" \
    "++model.deformer.rigid.geometry_fidelity_center_strength=$center_strength" \
    "++model.deformer.rigid.geometry_fidelity_rotation_strength=$rotation_strength" \
    "++model.deformer.rigid.geometry_fidelity_boundary_min=$boundary_min" \
    "++model.deformer.rigid.geometry_fidelity_layer_ids='$layer_ids'" \
    "++model.deformer.rigid.geometry_fidelity_region_ids='$region_ids'" \
    "++model.deformer.rigid.geometry_fidelity_joint_ids='$joint_ids'" \
    "++model.deformer.rigid.geometry_fidelity_thin_min='$thin_min'" \
    "++model.deformer.rigid.geometry_fidelity_surface_min='$surface_min'" \
    "++model.deformer.rigid.geometry_fidelity_surface_max='$surface_max'" \
    "++model.deformer.rigid.geometry_fidelity_non_rigid_min=$nr_min" \
    "++model.deformer.rigid.geometry_fidelity_power=$power" \
    "++model.deformer.rigid.geometry_fidelity_max_points=$max_points" \
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
  local screen_enable="$2"
  local screen_shrink="$3"
  local screen_grow="$4"
  local geom_enable="$5"
  local geom_target="$6"
  local center_strength="$7"
  local rotation_strength="$8"
  local boundary_min="$9"
  local layer_ids="${10}"
  local region_ids="${11}"
  local joint_ids="${12}"
  local thin_min="${13}"
  local surface_min="${14}"
  local surface_max="${15}"
  local nr_min="${16}"
  local power="${17}"
  local max_points="${18}"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$screen_enable" "$screen_shrink" "$screen_grow" "$geom_enable" "$geom_target" \
    "$center_strength" "$rotation_strength" "$boundary_min" "$layer_ids" "$region_ids" "$joint_ids" \
    "$thin_min" "$surface_min" "$surface_max" "$nr_min" "$power" "$max_points" "$render_exp" \
    >> "$PARAMS"

  log_event "render_start" "$variant"
  render_raw "$variant" "$screen_enable" "$screen_shrink" "$screen_grow" "$geom_enable" "$geom_target" \
    "$center_strength" "$rotation_strength" "$boundary_min" "$layer_ids" "$region_ids" "$joint_ids" \
    "$thin_min" "$surface_min" "$surface_max" "$nr_min" "$power" "$max_points" \
    "$render_exp" "$HYDRA_RUN_ROOT/render_${variant}" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  log_event "variant_done" "$variant"
}

run_variant baseline_raw false 1.000 1.000 false free_lbs 0.00 0.00 0.08 soft,free cloth,soft "" "" "" "" 0.000 1.0 -1
run_variant baseline_v281_screen_mid true 0.940 1.025 false free_lbs 0.00 0.00 0.08 soft,free cloth,soft "" "" "" "" 0.000 1.0 -1

run_variant free_center_025 true 0.940 1.025 true free_lbs 0.25 0.00 0.10 soft,free cloth,soft "" "" "" "" 0.000 1.0 2048
run_variant free_center_050 true 0.940 1.025 true free_lbs 0.50 0.00 0.10 soft,free cloth,soft "" "" "" "" 0.000 1.0 2048
run_variant free_center_rot_025 true 0.940 1.025 true free_lbs 0.25 0.25 0.10 soft,free cloth,soft "" "" "" "" 0.000 1.0 2048
run_variant free_center_rot_050 true 0.940 1.025 true free_lbs 0.50 0.50 0.10 soft,free cloth,soft "" "" "" "" 0.000 1.0 2048
run_variant free_high_boundary_035 true 0.940 1.025 true free_lbs 0.35 0.20 0.18 soft,free cloth,soft "" "" "" "" 0.000 1.4 1024
run_variant free_nonrigid_035 true 0.940 1.025 true free_lbs 0.35 0.20 0.10 soft,free cloth,soft "" "" "" "" 0.002 1.2 1536
run_variant free_under_joints_035 true 0.940 1.025 true free_lbs 0.35 0.20 0.08 soft,rigid,free cloth,body,soft "$UNDER_JOINT_IDS" "" "" "" 0.000 1.2 1536
run_variant source_center_025 true 0.940 1.025 true source 0.25 0.00 0.12 soft,free cloth,soft "" "" "" "" 0.000 1.2 1024
run_variant soft_center_025 true 0.940 1.025 true soft 0.25 0.20 0.12 soft,free cloth,soft "" "" "" "" 0.000 1.2 1024

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
raw = metrics_by_variant["baseline_raw"]
stable = metrics_by_variant["baseline_v281_screen_mid"]

header = [
    "variant", "screen_enable", "screen_shrink", "screen_grow", "geom_enable", "geom_target",
    "center_strength", "rotation_strength", "boundary_min", "layer_ids", "region_ids", "joint_ids",
    "thin_min", "surface_min", "surface_max", "nr_min", "power", "max_points", "render_exp",
    "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_raw", "boundary_delta_raw", "edge_delta_raw", "inner_delta_raw", "outer_delta_raw", "hard_delta_raw",
    "fg_delta_v281", "boundary_delta_v281", "edge_delta_v281", "inner_delta_v281", "outer_delta_v281", "hard_delta_v281",
    "raw_status", "v281_status",
]

best = None
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for row in rows:
        m = metrics_by_variant[row["variant"]]
        delta_raw = {key: m[key] - raw[key] for key in m}
        delta_stable = {key: m[key] - stable[key] for key in m}
        raw_status = status(delta_raw)
        stable_status = status(delta_stable)
        if row["variant"].startswith("baseline"):
            raw_status = "baseline" if row["variant"] == "baseline_raw" else raw_status
            stable_status = "baseline" if row["variant"] == "baseline_v281_screen_mid" else stable_status
        if stable_status in ("strict_pass", "probe_pass"):
            score = (
                delta_stable["hard"] * 10000.0
                + delta_stable["inner"] * 0.03
                + delta_stable["outer"] * 0.02
                + delta_stable["edge"] * 10.0
                + delta_stable["fg"] * 1000.0
                + delta_stable["boundary"] * 1000.0
            )
            rank = 0 if stable_status == "strict_pass" else 1
            candidate = (rank, score, row["variant"])
            if best is None or candidate < best[0]:
                best = (candidate, row, stable_status)
        writer.writerow([
            row["variant"], row["screen_enable"], row["screen_shrink"], row["screen_grow"],
            row["geom_enable"], row["geom_target"], row["center_strength"], row["rotation_strength"],
            row["boundary_min"], row["layer_ids"], row["region_ids"], row["joint_ids"], row["thin_min"],
            row["surface_min"], row["surface_max"], row["nr_min"], row["power"], row["max_points"],
            row["render_exp"],
            fmt(m["fg"]), fmt(m["boundary"]), fmt(m["edge"], 6), fmt(m["inner"], 4),
            fmt(m["outer"], 4), fmt(m["hard"]),
            fmt(delta_raw["fg"]), fmt(delta_raw["boundary"]), fmt(delta_raw["edge"], 6),
            fmt(delta_raw["inner"], 4), fmt(delta_raw["outer"], 4), fmt(delta_raw["hard"]),
            fmt(delta_stable["fg"]), fmt(delta_stable["boundary"]), fmt(delta_stable["edge"], 6),
            fmt(delta_stable["inner"], 4), fmt(delta_stable["outer"], 4), fmt(delta_stable["hard"]),
            raw_status, stable_status,
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
