#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v311_projected_component_ab_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
SOURCE_COMPONENT_CSV="${SOURCE_COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"

SOURCE_VIEWS_SPEC="${SOURCE_VIEWS_SPEC:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"
PREDICTOR_TRAIN_VIEWS_SPEC="${PREDICTOR_TRAIN_VIEWS_SPEC:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
PREDICTOR_TRAIN_FRAMES_SPEC="${PREDICTOR_TRAIN_FRAMES_SPEC:-0,570,60}"
PREDICTOR_TARGET_VIEWS_SPEC="${PREDICTOR_TARGET_VIEWS_SPEC:-21,22,23}"
PREDICTOR_TARGET_FRAMES_SPEC="${PREDICTOR_TARGET_FRAMES_SPEC:-0,570,60}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v311_projected_component_ab_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v311_projected_component_ab_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
PREDICTOR_DIR="$LOG_DIR/predictor"
PRED_COMPONENT_CSV="$PREDICTOR_DIR/component_contributors_predicted.csv"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$PREDICTOR_DIR"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$SOURCE_COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
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
SOURCE_COMPONENT_CSV=$SOURCE_COMPONENT_CSV
POINT_CSV=$POINT_CSV
SOURCE_VIEWS_SPEC=$SOURCE_VIEWS_SPEC
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
PRED_COMPONENT_CSV=$PRED_COMPONENT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v311 tests whether the v307 residual-component localization can be replaced
  by a non-leaking projected component prior. The predicted CSV is built from
  train-view component clusters only, then projected into held-out test views.
  No checkpoint edit and no training are used.
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

log_event "predict_start" "project train-view component clusters to held-out views"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v311_projected_component_prior.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --source-component-csv "$SOURCE_COMPONENT_CSV" \
  --dataset-root "$DATA_ROOT" \
  --out-dir "$PREDICTOR_DIR" \
  --output-csv "$PRED_COMPONENT_CSV" \
  --source-views "$SOURCE_VIEWS_SPEC" \
  --train-views "$PREDICTOR_TRAIN_VIEWS_SPEC" \
  --train-frames "$PREDICTOR_TRAIN_FRAMES_SPEC" \
  --target-views "$PREDICTOR_TARGET_VIEWS_SPEC" \
  --target-frames "$PREDICTOR_TARGET_FRAMES_SPEC" \
  > "$LOG_DIR/predictor.log" 2>&1
log_event "predict_done" "$PRED_COMPONENT_CSV"

render_variant() {
  local variant="$1"
  local render_exp="$2"
  local preset="$3"
  local component_csv="$4"
  local hydra_dir="$5"

  local preset_args=()
  if [ "$preset" != "none" ]; then
    preset_args=(
      "++explicit_binding_render_preset=$preset"
      "++explicit_binding_adopted_component_csv=$component_csv"
      "++explicit_binding_adopted_point_csv=$POINT_CSV"
      "++explicit_binding_adopted_center_strength=0.45"
      "++explicit_binding_adopted_outer_px=0.35"
      "++explicit_binding_adopted_component_required=true"
      "++explicit_binding_adopted_improvement_guard=true"
      "++explicit_binding_adopted_max_points=96"
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
    "${preset_args[@]}" \
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
EXACT_RENDER_EXP="$EXP_ROOT/v307_exact_component"
PRED_RENDER_EXP="$EXP_ROOT/v311_projected_component"

for spec in \
  "baseline_no_preset|$BASELINE_RENDER_EXP|none|$SOURCE_COMPONENT_CSV" \
  "v307_exact_component|$EXACT_RENDER_EXP|v307_adopted_geometry|$SOURCE_COMPONENT_CSV" \
  "v311_projected_component|$PRED_RENDER_EXP|v307_adopted_geometry|$PRED_COMPONENT_CSV"; do
  IFS='|' read -r variant render_exp preset component_csv <<< "$spec"
  log_event "render_start" "$variant"
  render_variant "$variant" "$render_exp" "$preset" "$component_csv" "$HYDRA_RUN_ROOT/$variant"
  analyze_variant "$variant" "$render_exp"
  log_event "render_done" "$variant"
done

"$PYTHON_BIN" - "$SUMMARY" "$BASELINE_RENDER_EXP" "$EXACT_RENDER_EXP" "$PRED_RENDER_EXP" "$PRED_COMPONENT_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
variant_paths = {
    "baseline_no_preset": Path(sys.argv[2]),
    "v307_exact_component": Path(sys.argv[3]),
    "v311_projected_component": Path(sys.argv[4]),
}
pred_csv = Path(sys.argv[5])

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

metrics = {name: metrics_from_render(path) for name, path in variant_paths.items()}
baseline = metrics["baseline_no_preset"]
exact = metrics["v307_exact_component"]

header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "fg_delta_exact", "boundary_delta_exact", "edge_delta_exact",
    "inner_delta_exact", "outer_delta_exact", "hard_delta_exact",
    "strict_pass", "probe_pass", "status",
]
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, path in variant_paths.items():
        item = metrics[name]
        delta_base = {key: item[key] - baseline[key] for key in item}
        delta_exact = {key: item[key] - exact[key] for key in item}
        strict, probe, status = status_for(delta_base)
        writer.writerow([
            name,
            str(path),
            f"{item['fg']:.8f}",
            f"{item['boundary']:.8f}",
            f"{item['edge']:.6f}",
            f"{item['inner']:.4f}",
            f"{item['outer']:.4f}",
            f"{item['hard']:.8f}",
            f"{delta_base['fg']:.8f}",
            f"{delta_base['boundary']:.8f}",
            f"{delta_base['edge']:.6f}",
            f"{delta_base['inner']:.4f}",
            f"{delta_base['outer']:.4f}",
            f"{delta_base['hard']:.8f}",
            f"{delta_exact['fg']:.8f}",
            f"{delta_exact['boundary']:.8f}",
            f"{delta_exact['edge']:.6f}",
            f"{delta_exact['inner']:.4f}",
            f"{delta_exact['outer']:.4f}",
            f"{delta_exact['hard']:.8f}",
            "1" if strict else "0",
            "1" if probe else "0",
            status,
        ])

print(json.dumps({
    "summary": str(summary_path),
    "pred_component_csv": str(pred_csv),
    "metrics": metrics,
}, indent=2), flush=True)
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "PRED_COMPONENT_CSV=$PRED_COMPONENT_CSV"
  echo "BASELINE_RENDER_EXP=$BASELINE_RENDER_EXP"
  echo "EXACT_RENDER_EXP=$EXACT_RENDER_EXP"
  echo "PRED_RENDER_EXP=$PRED_RENDER_EXP"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "PRED_COMPONENT_CSV=$PRED_COMPONENT_CSV"
echo "BASELINE_RENDER_EXP=$BASELINE_RENDER_EXP"
echo "EXACT_RENDER_EXP=$EXACT_RENDER_EXP"
echo "PRED_RENDER_EXP=$PRED_RENDER_EXP"
echo "SUMMARY=$SUMMARY"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
