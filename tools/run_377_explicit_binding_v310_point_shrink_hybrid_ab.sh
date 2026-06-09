#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v310_point_shrink_hybrid_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

MAX_SHRINK="${MAX_SHRINK:-96}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v310_point_shrink_hybrid_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v310_point_shrink_hybrid_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
POINT_PRIOR_JSON="$LOG_DIR/signed_point_shrink_prior.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-1500}"
EST_END_EPOCH="$((START_EPOCH + EST_SECONDS))"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$EST_END_EPOCH" '+%F %T BJT')"

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
}

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
EST_END_BJT=$EST_END_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
DATA_ROOT=$DATA_ROOT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
POINT_PRIOR_JSON=$POINT_PRIOR_JSON
MAX_SHRINK=$MAX_SHRINK
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v310 tests the asymmetric conclusion from v309: use checkpoint-level signed
  point prior only for outer shrink, while retaining component-level center
  fidelity for inner geometry. No training.
EOF

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=$GPU"
  "OMP_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "MKL_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "OPENBLAS_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "NUMEXPR_NUM_THREADS=$CPU_THREADS_PER_JOB"
  "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
)

log_event "make_prior" "$POINT_PRIOR_JSON"
"$PYTHON_BIN" tools/make_377_signed_point_prior.py \
  --point-csv "$POINT_CSV" \
  --out-json "$POINT_PRIOR_JSON" \
  --max-shrink "$MAX_SHRINK" \
  --max-grow 0 \
  --min-abs-signed 0.0 \
  --min-hit-gap 0 \
  > "$LOG_DIR/make_prior.log" 2>&1

render_variant() {
  local variant="$1"
  local render_exp="$2"
  local mode="$3"
  local hydra_dir="$4"

  local extra_args=()
  if [ "$mode" = "adopted" ]; then
    extra_args=(
      "++explicit_binding_render_preset=v307_adopted_geometry"
      "++explicit_binding_adopted_component_csv=$COMPONENT_CSV"
      "++explicit_binding_adopted_point_csv=$POINT_CSV"
      "++explicit_binding_adopted_center_strength=0.45"
      "++explicit_binding_adopted_outer_px=0.35"
      "++explicit_binding_adopted_component_required=true"
      "++explicit_binding_adopted_improvement_guard=true"
      "++explicit_binding_adopted_max_points=96"
    )
  elif [ "$mode" = "point_shrink_hybrid" ]; then
    extra_args=(
      "pipeline.compute_cov3D_python=true"
      "++pipeline.covariance_mode=default"
      "++pipeline.covariance_signed_point_json=$POINT_PRIOR_JSON"
      "++pipeline.covariance_signed_point_screen_actuator_enable=true"
      "++pipeline.covariance_signed_shrink_factor=1.0"
      "++pipeline.covariance_signed_grow_factor=1.0"
      "++pipeline.covariance_signed_max_shrink_points=$MAX_SHRINK"
      "++pipeline.covariance_signed_max_grow_points=0"
      "++pipeline.covariance_signed_dynamic_enable=false"
      "++pipeline.covariance_signed_screen_actuator_enable=true"
      "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940"
      "++pipeline.covariance_signed_screen_normal_grow_factor=1.000"
      "++pipeline.covariance_signed_screen_tangent_factor=1.000"
      "++pipeline.covariance_signed_center_offset_enable=true"
      "++pipeline.covariance_signed_center_offset_outer_px=0.35"
      "++pipeline.covariance_signed_center_offset_inner_px=0.00"
      "++pipeline.covariance_signed_center_offset_outer_direction=view_center"
      "++pipeline.covariance_signed_center_offset_inner_direction=component_center"
      "++pipeline.covariance_signed_center_offset_score_weight_power=1.0"
      "++pipeline.covariance_signed_center_offset_score_weight_min=0.15"
      "++pipeline.covariance_signed_center_offset_score_weight_quantile=0.90"
      "++pipeline.covariance_signed_center_offset_jacobian_eps=0.001"
      "++pipeline.covariance_signed_center_offset_jacobian_damping=0.00001"
      "++pipeline.covariance_signed_center_offset_max_world_step=0.0020"
      "++model.deformer.rigid.rotation_orthogonalize_enable=false"
      "++model.deformer.rigid.geometry_fidelity_gate_enable=true"
      "++model.deformer.rigid.geometry_fidelity_target=free_lbs"
      "++model.deformer.rigid.geometry_fidelity_center_strength=0.45"
      "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0"
      "++model.deformer.rigid.geometry_fidelity_boundary_min=0.12"
      "++model.deformer.rigid.geometry_fidelity_layer_ids='soft,free'"
      "++model.deformer.rigid.geometry_fidelity_region_ids='cloth,soft'"
      "++model.deformer.rigid.geometry_fidelity_component_enable=true"
      "++model.deformer.rigid.geometry_fidelity_component_csv=$COMPONENT_CSV"
      "++model.deformer.rigid.geometry_fidelity_component_direction=inner"
      "++model.deformer.rigid.geometry_fidelity_component_pad_px=2"
      "++model.deformer.rigid.geometry_fidelity_component_ellipse_scale=1.05"
      "++model.deformer.rigid.geometry_fidelity_component_max=12"
      "++model.deformer.rigid.geometry_fidelity_component_min_area=40"
      "++model.deformer.rigid.geometry_fidelity_component_required=true"
      "++model.deformer.rigid.geometry_fidelity_component_improvement_enable=true"
      "++model.deformer.rigid.geometry_fidelity_component_improvement_margin_px=0.0"
      "++model.deformer.rigid.geometry_fidelity_power=1.2"
      "++model.deformer.rigid.geometry_fidelity_max_points=1024"
    )
  fi

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$BASE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$TEST_VIEWS_SPEC" \
    "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "${extra_args[@]}" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
}

analyze_variant() {
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

BASELINE_RENDER_EXP="$EXP_ROOT/baseline_no_preset"
ADOPTED_RENDER_EXP="$EXP_ROOT/v307_adopted_preset"
HYBRID_RENDER_EXP="$EXP_ROOT/v310_point_shrink_hybrid"

for spec in \
  "baseline_no_preset|$BASELINE_RENDER_EXP|none" \
  "v307_adopted_preset|$ADOPTED_RENDER_EXP|adopted" \
  "v310_point_shrink_hybrid|$HYBRID_RENDER_EXP|point_shrink_hybrid"; do
  IFS='|' read -r variant render_exp mode <<< "$spec"
  log_event "render_start" "$variant"
  render_variant "$variant" "$render_exp" "$mode" "$HYDRA_RUN_ROOT/$variant"
  analyze_variant "$variant" "$render_exp"
  log_event "render_done" "$variant"
done

"$PYTHON_BIN" - "$SUMMARY" "$BASELINE_RENDER_EXP" "$ADOPTED_RENDER_EXP" "$HYBRID_RENDER_EXP" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
variant_paths = {
    "baseline_no_preset": Path(sys.argv[2]),
    "v307_adopted_preset": Path(sys.argv[3]),
    "v310_point_shrink_hybrid": Path(sys.argv[4]),
}

def metrics_from_render(render_exp):
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

metrics = {name: metrics_from_render(path) for name, path in variant_paths.items()}
baseline = metrics["baseline_no_preset"]

def status_for(delta):
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
    return strict, probe, "strict_pass" if strict else ("probe_pass" if probe else "rejected")

header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "strict_pass", "probe_pass", "status",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, render_exp in variant_paths.items():
        values = metrics[name]
        delta = {key: values[key] - baseline[key] for key in values}
        strict, probe, status = status_for(delta)
        if name == "baseline_no_preset":
            strict = probe = False
            status = "baseline"
        writer.writerow([
            name,
            str(render_exp),
            f"{values['fg']:.8f}",
            f"{values['boundary']:.8f}",
            f"{values['edge']:.6f}",
            f"{values['inner']:.4f}",
            f"{values['outer']:.4f}",
            f"{values['hard']:.8f}",
            f"{delta['fg']:.8f}",
            f"{delta['boundary']:.8f}",
            f"{delta['edge']:.6f}",
            f"{delta['inner']:.4f}",
            f"{delta['outer']:.4f}",
            f"{delta['hard']:.8f}",
            "1" if strict else "0",
            "1" if probe else "0",
            status,
        ])
print(json.dumps({"summary": str(summary_path), "metrics": metrics}, indent=2))
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "BASELINE_RENDER_EXP=$BASELINE_RENDER_EXP"
  echo "ADOPTED_RENDER_EXP=$ADOPTED_RENDER_EXP"
  echo "HYBRID_RENDER_EXP=$HYBRID_RENDER_EXP"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
