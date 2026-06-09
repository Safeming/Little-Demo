#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v339_full_frame_component_field_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"
V337_EXP_ROOT="${V337_EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v337_temporal_propagated_group_field_v337_temporal_propagated_group_field_20260521_143930_bjt}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,1]}"
TRAIN_VIEWS_CSV="${TRAIN_VIEWS_CSV:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
TRAIN_FRAMES_CSV="${TRAIN_FRAMES_CSV:-0,570,60}"
TARGET_VIEWS_CSV="${TARGET_VIEWS_CSV:-21,22,23}"
TARGET_FRAMES_CSV="${TARGET_FRAMES_CSV:-0,570,1}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v339_full_frame_component_field_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v339_full_frame_component_field_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
COMPONENT_DIR="$LOG_DIR/component"
FIELD_DIR="$LOG_DIR/field"
TARGET_COMPONENT_CSV="$COMPONENT_DIR/full_frame_raw_components.csv"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
WORST_SUMMARY="$LOG_DIR/worst_frames.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$COMPONENT_DIR" "$FIELD_DIR"
for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$POINT_CSV" "$DATA_ROOT" \
  "$V337_EXP_ROOT/baseline_no_preset" \
  "$V337_EXP_ROOT/formal_v320_v307_signed_geometry"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$GPU" "$1" "$2" "$START_BJT" <<'PY' || true
import json, sys, time
from pathlib import Path
path, run_id, gpu, phase, detail, start_bjt = sys.argv[1:]
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_bjt": start_bjt,
    "now_epoch": int(time.time()),
}, indent=2), encoding="utf-8")
PY
}

log_event() {
  printf '%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$1" "$2" | tee -a "$EVENTS"
  write_status "$1" "$2"
}

if [ -n "$WAIT_FOR_PID" ]; then
  log_event "wait_for_pid_start" "$WAIT_FOR_PID"
  while ps -p "$WAIT_FOR_PID" >/dev/null 2>&1; do
    sleep 60
  done
  log_event "wait_for_pid_done" "$WAIT_FOR_PID"
fi

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
WAIT_FOR_PID=$WAIT_FOR_PID
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
POINT_CSV=$POINT_CSV
V337_EXP_ROOT=$V337_EXP_ROOT
TARGET_COMPONENT_CSV=$TARGET_COMPONENT_CSV
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v339 tests the root-cause direction after v337/v338: remove temporal
  keyframe propagation as a source of local signed mismatch. It extracts
  inner/outer raw residual components on every target frame, builds paired
  signed point fields from those full-frame components, and evaluates only
  no-train render geometry on top of the formal v320/v307 preset.
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

make_field() {
  local label="$1"
  shift
  local out_json="$FIELD_DIR/${label}.json"
  log_event "field_${label}_start" "$out_json"
  "$PYTHON_BIN" tools/make_377_stageB_v336_group_paired_signed_boundary_field.py \
    --component-csv "$TARGET_COMPONENT_CSV" \
    --point-csv "$POINT_CSV" \
    --out-json "$out_json" \
    "$@" \
    > "$LOG_DIR/field_${label}.log" 2>&1
  log_event "field_${label}_done" "$out_json"
}

render_variant() {
  local variant="$1"
  local render_exp="$2"
  shift 2
  log_event "render_${variant}_start" "$render_exp"
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
    "$@" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$variant" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "render_${variant}_done" "status=0"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 30 \
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
    --topk 30 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${variant}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_opacity_footprint.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --render-support-threshold 0.025 \
    --primary-opacity-threshold 0.06 \
    --opacity-thresholds 0.02,0.04,0.06,0.08,0.10 \
    --rgb-close-kernel 5 \
    --opacity-close-kernel 3 \
    --band-width 7 \
    --search-band-width 24 \
    --topk 30 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${variant}.log" 2>&1
  log_event "analyze_${variant}_done" "status=0"
}

render_formal_field() {
  local variant="$1"
  local render_exp="$2"
  local field_json="$3"
  render_variant "$variant" "$render_exp" \
    "++explicit_binding_adopted_outer_px=0.18" \
    "++explicit_binding_render_preset=v320_v307_signed_geometry" \
    "++pipeline.covariance_signed_point_json=$field_json" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=true"
}

log_event "component_full_frame_start" "$TARGET_COMPONENT_CSV"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" tools/make_377_stageB_v330_target_raw_component_csv.py \
  --config-path "$BASE_EXP/.hydra/config.yaml" \
  --load-ckpt "$BASE_CKPT" \
  --dataset-root "$DATA_ROOT" \
  --out-dir "$COMPONENT_DIR" \
  --out-csv "$TARGET_COMPONENT_CSV" \
  --train-views "$TRAIN_VIEWS_CSV" \
  --train-frames "$TRAIN_FRAMES_CSV" \
  --target-views "$TARGET_VIEWS_CSV" \
  --target-frames "$TARGET_FRAMES_CSV" \
  --render-support-threshold 0.025 \
  --close-kernel 5 \
  --search-band-width 24 \
  --min-component-area 24 \
  --max-components-per-direction 12 \
  --top-points 8 \
  --point-pad-px 10 \
  --require-top-points \
  > "$LOG_DIR/component_full_frame.log" 2>&1
log_event "component_full_frame_done" "$TARGET_COMPONENT_CSV"

make_field full_paired_balanced \
  --max-shrink-per-image 40 --max-grow-per-image 64 --max-components-per-direction 8 \
  --top-points-per-component 5 --min-component-area 30 --require-paired-image \
  --protect-grow-from-shrink --protect-inner-points 40 --direction-score-margin -1.0 \
  --grow-min-boundary 0.10 --grow-min-visible-hits 15 --grow-allowed-regions 1,2 \
  --shrink-min-boundary 0.10 --shrink-min-visible-hits 20 --shrink-allowed-regions 2
make_field full_inner_guard \
  --max-shrink-per-image 24 --max-grow-per-image 80 --max-components-per-direction 10 \
  --top-points-per-component 6 --min-component-area 24 --require-paired-image \
  --protect-grow-from-shrink --protect-inner-points 64 --direction-score-margin -1.0 \
  --grow-min-boundary 0.08 --grow-min-visible-hits 12 --grow-allowed-regions 1,2 \
  --shrink-min-boundary 0.14 --shrink-min-visible-hits 30 --shrink-allowed-regions 2
make_field full_component_open \
  --max-shrink-per-image 32 --max-grow-per-image 64 --max-components-per-direction 8 \
  --top-points-per-component 5 --min-component-area 36 \
  --protect-grow-from-shrink --protect-inner-points 48 --direction-score-margin -1.0 \
  --grow-min-boundary 0.10 --grow-min-visible-hits 15 --grow-allowed-regions 1,2 \
  --shrink-min-boundary 0.12 --shrink-min-visible-hits 25 --shrink-allowed-regions 2

BASELINE_EXP="$V337_EXP_ROOT/baseline_no_preset"
FORMAL_EXP="$V337_EXP_ROOT/formal_v320_v307_signed_geometry"
BALANCED_EXP="$EXP_ROOT/formal_plus_full_paired_balanced"
INNER_EXP="$EXP_ROOT/formal_plus_full_inner_guard"
OPEN_EXP="$EXP_ROOT/formal_plus_full_component_open"

log_event "reuse_baseline_no_preset" "$BASELINE_EXP"
log_event "reuse_formal_v320_v307_signed_geometry" "$FORMAL_EXP"
render_formal_field formal_plus_full_paired_balanced "$BALANCED_EXP" "$FIELD_DIR/full_paired_balanced.json"
render_formal_field formal_plus_full_inner_guard "$INNER_EXP" "$FIELD_DIR/full_inner_guard.json"
render_formal_field formal_plus_full_component_open "$OPEN_EXP" "$FIELD_DIR/full_component_open.json"

"$PYTHON_BIN" - "$SUMMARY" "$WORST_SUMMARY" \
  baseline_no_preset "$BASELINE_EXP" \
  formal_v320_v307_signed_geometry "$FORMAL_EXP" \
  formal_plus_full_paired_balanced "$BALANCED_EXP" \
  formal_plus_full_inner_guard "$INNER_EXP" \
  formal_plus_full_component_open "$OPEN_EXP" <<'PY'
import csv, json, sys
from pathlib import Path

summary_path = Path(sys.argv[1])
worst_path = Path(sys.argv[2])
pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))

def load_metrics(render_exp):
    render_exp = Path(render_exp)
    contour = json.loads((render_exp / "diagnostics/contours/contour_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((render_exp / "diagnostics/boundary_residuals/boundary_residual_summary.json").read_text(encoding="utf-8"))
    opacity = json.loads((render_exp / "diagnostics/opacity_footprint/opacity_footprint_summary.json").read_text(encoding="utf-8"))
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

def load_records(render_exp):
    records = {}
    render_exp = Path(render_exp)
    for path in [
        render_exp / "diagnostics/contours/contour_samples.csv",
        render_exp / "diagnostics/boundary_residuals/boundary_residual_samples.csv",
        render_exp / "diagnostics/opacity_footprint/opacity_footprint_samples.csv",
    ]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = f"c{int(float(row.get('cam', 0))):02d}_f{int(float(row.get('frame', 0))):06d}"
                rec = records.setdefault(key, {})
                for k, v in row.items():
                    if v is None or v == "":
                        continue
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
    return records

def status_for(delta):
    strict = (
        delta["fg"] <= 0.0 and delta["boundary"] <= 0.0 and delta["edge"] <= 0.0
        and delta["inner"] <= 0.0 and delta["outer"] <= 0.0 and delta["hard"] <= -0.000001
        and delta["opacity_inner"] <= 0.0 and delta["opacity_outer"] <= 0.0
    )
    probe = (
        delta["hard"] < -0.00001 and delta["fg"] <= 0.000015 and delta["boundary"] <= 0.000015
        and delta["edge"] <= 0.003 and delta["inner"] <= 0.5 and delta["outer"] <= 0.5
        and delta["opacity_inner"] <= 0.5 and delta["opacity_outer"] <= 0.5
    )
    return "strict_pass" if strict else ("probe_pass" if probe else "rejected")

metrics = {name: load_metrics(path) for name, path in pairs}
base = metrics["baseline_no_preset"]
formal = metrics["formal_v320_v307_signed_geometry"]
metric_names = ["fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer"]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    header = ["variant", "render_exp", *metric_names]
    header += [f"{name}_delta_base" for name in metric_names]
    header += [f"{name}_delta_formal" for name in metric_names]
    header += ["status"]
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, render_exp in pairs:
        m = metrics[name]
        db = {key: m[key] - base[key] for key in metric_names}
        df = {key: m[key] - formal[key] for key in metric_names}
        status = "baseline" if name == "baseline_no_preset" else status_for(db)
        writer.writerow([
            name, render_exp,
            f'{m["fg"]:.8f}', f'{m["boundary"]:.8f}', f'{m["edge"]:.6f}', f'{m["inner"]:.4f}',
            f'{m["outer"]:.4f}', f'{m["hard"]:.8f}', f'{m["opacity_inner"]:.4f}', f'{m["opacity_outer"]:.4f}',
            f'{db["fg"]:.8f}', f'{db["boundary"]:.8f}', f'{db["edge"]:.6f}', f'{db["inner"]:.4f}',
            f'{db["outer"]:.4f}', f'{db["hard"]:.8f}', f'{db["opacity_inner"]:.4f}', f'{db["opacity_outer"]:.4f}',
            f'{df["fg"]:.8f}', f'{df["boundary"]:.8f}', f'{df["edge"]:.6f}', f'{df["inner"]:.4f}',
            f'{df["outer"]:.4f}', f'{df["hard"]:.8f}', f'{df["opacity_inner"]:.4f}', f'{df["opacity_outer"]:.4f}',
            status,
        ])

base_records = load_records(dict(pairs)["baseline_no_preset"])
aliases = {
    "fg": "fg_l1",
    "boundary": "boundary_l1",
    "edge": "edge_symmetric_dist_px",
    "inner": "inner_missing_pixels",
    "outer": "outer_leak_pixels",
    "hard": "hard_residual_score",
    "opacity_inner": "primary_opacity_inner_missing_pixels",
    "opacity_outer": "primary_opacity_outer_leak_pixels",
}
with worst_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = ["variant", "image", "worsen_score"] + [f"{name}_delta" for name in aliases]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    rows = []
    for name, render_exp in pairs:
        if name == "baseline_no_preset":
            continue
        for image, rec in load_records(render_exp).items():
            base_rec = base_records.get(image)
            if not base_rec:
                continue
            row = {"variant": name, "image": image, "worsen_score": 0.0}
            for metric, column in aliases.items():
                if column not in rec or column not in base_rec:
                    continue
                delta = float(rec[column]) - float(base_rec[column])
                row[f"{metric}_delta"] = delta
                row["worsen_score"] += max(delta, 0.0)
            rows.append(row)
    rows.sort(key=lambda item: item["worsen_score"], reverse=True)
    for row in rows[:100]:
        writer.writerow({key: (f"{row[key]:.8f}" if isinstance(row.get(key), float) else row.get(key, "")) for key in fieldnames})
print(summary_path)
PY

END_BJT="$(TZ=Asia/Shanghai date '+%F %T BJT')"
echo "END_BJT=$END_BJT" >> "$LOG_DIR/run_info.txt"
log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "WORST_SUMMARY=$WORST_SUMMARY"
echo "END_BJT=$END_BJT"
