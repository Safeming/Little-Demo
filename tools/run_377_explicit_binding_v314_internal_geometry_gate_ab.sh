#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v314_internal_geometry_gate_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
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

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v314_internal_geometry_gate_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v314_internal_geometry_gate_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-2400}"
EST_END_EPOCH="$((START_EPOCH + EST_SECONDS))"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
EST_END_BJT="$(TZ=Asia/Shanghai date -d "@$EST_END_EPOCH" '+%F %T BJT')"

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

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
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v314 tests a structure-level hypothesis before any training: whether explicit
  binding's own uncertainty signals can localize the boundary x_bar/free_lbs
  mismatch that v307 fixes with external residual components.

Principles:
  no training, no support append, no StageC/export refinement, no held-out
  residual component CSV for the v314 internal variants.
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

render_variant() {
  local variant="$1"
  local render_exp="$2"
  local mode="$3"
  local center_strength="$4"
  local boundary_min="$5"
  local entropy_min="$6"
  local softfree_min="$7"
  local delta_min="$8"
  local delta_max="$9"
  local max_points="${10}"
  local hydra_dir="${11}"

  local extra_args=()
  if [ "$mode" = "v307_reference" ]; then
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
  elif [ "$mode" = "internal" ]; then
    extra_args=(
      "pipeline.compute_cov3D_python=true"
      "++pipeline.covariance_mode=default"
      "++pipeline.covariance_signed_dynamic_enable=false"
      "++pipeline.covariance_signed_screen_actuator_enable=false"
      "++pipeline.covariance_signed_center_offset_enable=false"
      "++pipeline.boundary_cov_residual_enable=false"
      "++pipeline.binding_covariance_guard_enable=false"
      "++model.deformer.rigid.rotation_orthogonalize_enable=false"
      "++model.deformer.rigid.geometry_fidelity_gate_enable=true"
      "++model.deformer.rigid.geometry_fidelity_target=free_lbs"
      "++model.deformer.rigid.geometry_fidelity_center_strength=$center_strength"
      "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0"
      "++model.deformer.rigid.geometry_fidelity_boundary_min=$boundary_min"
      "++model.deformer.rigid.geometry_fidelity_layer_ids='soft,free'"
      "++model.deformer.rigid.geometry_fidelity_region_ids='cloth,soft'"
      "++model.deformer.rigid.geometry_fidelity_joint_ids="
      "++model.deformer.rigid.geometry_fidelity_thin_min="
      "++model.deformer.rigid.geometry_fidelity_surface_min="
      "++model.deformer.rigid.geometry_fidelity_surface_max="
      "++model.deformer.rigid.geometry_fidelity_non_rigid_min=0.0"
      "++model.deformer.rigid.geometry_fidelity_power=1.2"
      "++model.deformer.rigid.geometry_fidelity_max_points=$max_points"
      "++model.deformer.rigid.geometry_fidelity_component_enable=false"
      "++model.deformer.rigid.geometry_fidelity_component_required=false"
      "++model.deformer.rigid.geometry_fidelity_internal_gate_enable=true"
      "++model.deformer.rigid.geometry_fidelity_internal_entropy_min=$entropy_min"
      "++model.deformer.rigid.geometry_fidelity_internal_softfree_min=$softfree_min"
      "++model.deformer.rigid.geometry_fidelity_internal_screen_delta_min_px=$delta_min"
      "++model.deformer.rigid.geometry_fidelity_internal_screen_delta_max_px=$delta_max"
      "++model.deformer.rigid.geometry_fidelity_internal_score_mode=entropy_delta"
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

run_variant() {
  local variant="$1"
  local mode="$2"
  local center_strength="$3"
  local boundary_min="$4"
  local entropy_min="$5"
  local softfree_min="$6"
  local delta_min="$7"
  local delta_max="$8"
  local max_points="$9"
  local render_exp="$EXP_ROOT/$variant"

  log_event "render_start" "$variant"
  render_variant "$variant" "$render_exp" "$mode" "$center_strength" "$boundary_min" "$entropy_min" "$softfree_min" "$delta_min" "$delta_max" "$max_points" "$HYDRA_RUN_ROOT/$variant"
  log_event "analyze_start" "$variant"
  analyze_variant "$variant" "$render_exp"
  log_event "variant_done" "$variant"
}

run_variant baseline_no_preset none 0.00 0.00 "" "" "" "" -1
run_variant v307_external_component_reference v307_reference 0.45 0.12 "" "" "" "" 1024
run_variant v314_internal_sparse_ambiguous internal 0.35 0.12 0.58 0.055 0.20 3.00 384
run_variant v314_internal_highdelta internal 0.45 0.10 0.42 0.035 0.55 4.00 512
run_variant v314_internal_tight_safe internal 0.25 0.16 0.66 0.080 0.25 2.25 256
run_variant v314_internal_softfree_mid internal 0.35 0.12 0.50 0.090 0.15 3.50 384

"$PYTHON_BIN" - "$SUMMARY" "$EXP_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
variants = [
    "baseline_no_preset",
    "v307_external_component_reference",
    "v314_internal_sparse_ambiguous",
    "v314_internal_highdelta",
    "v314_internal_tight_safe",
    "v314_internal_softfree_mid",
]

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

metrics = {name: metrics_from_render(exp_root / name) for name in variants}
baseline = metrics["baseline_no_preset"]
reference = metrics["v307_external_component_reference"]

def status_for(name, delta_base, values):
    if name == "baseline_no_preset":
        return False, False, "baseline"
    if name == "v307_external_component_reference":
        return False, False, "external_reference"
    strict = (
        delta_base["inner"] < -0.05
        and delta_base["outer"] <= 0.0
        and delta_base["fg"] <= 0.0
        and delta_base["boundary"] <= 0.0
        and delta_base["edge"] <= 0.0
        and delta_base["hard"] < -0.000001
    )
    probe = (
        delta_base["hard"] < -0.000010
        and delta_base["inner"] <= 0.50
        and delta_base["outer"] <= 1.50
        and delta_base["fg"] <= 0.000080
        and delta_base["boundary"] <= 0.000080
        and delta_base["edge"] <= 0.008000
    )
    catches_reference_inner = values["inner"] <= reference["inner"] + 0.50
    if strict and catches_reference_inner:
        return True, True, "strict_pass_reference_close"
    return strict, probe, "strict_pass" if strict else ("probe_pass" if probe else "rejected")

header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "fg_delta_v307", "boundary_delta_v307", "edge_delta_v307",
    "inner_delta_v307", "outer_delta_v307", "hard_delta_v307",
    "strict_pass", "probe_pass", "status",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name in variants:
        values = metrics[name]
        delta_base = {key: values[key] - baseline[key] for key in values}
        delta_ref = {key: values[key] - reference[key] for key in values}
        strict, probe, status = status_for(name, delta_base, values)
        writer.writerow([
            name,
            str(exp_root / name),
            f"{values['fg']:.8f}",
            f"{values['boundary']:.8f}",
            f"{values['edge']:.6f}",
            f"{values['inner']:.4f}",
            f"{values['outer']:.4f}",
            f"{values['hard']:.8f}",
            f"{delta_base['fg']:.8f}",
            f"{delta_base['boundary']:.8f}",
            f"{delta_base['edge']:.6f}",
            f"{delta_base['inner']:.4f}",
            f"{delta_base['outer']:.4f}",
            f"{delta_base['hard']:.8f}",
            f"{delta_ref['fg']:.8f}",
            f"{delta_ref['boundary']:.8f}",
            f"{delta_ref['edge']:.6f}",
            f"{delta_ref['inner']:.4f}",
            f"{delta_ref['outer']:.4f}",
            f"{delta_ref['hard']:.8f}",
            str(bool(strict)).lower(),
            str(bool(probe)).lower(),
            status,
        ])
print(summary_path)
PY

log_event "summary_done" "$SUMMARY"
END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
log_event "done" "END_BJT=$END_BJT"
echo "START_BJT=$START_BJT"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
echo "SUMMARY=$SUMMARY"
