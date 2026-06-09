#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v333_full_formal_signed_geometry_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,1]}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v333_full_formal_signed_geometry_gate_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v333_full_formal_signed_geometry_gate_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
WORST_SUMMARY="$LOG_DIR/worst_frames.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" \
  "$ROOT/assets/adopted_geometry/377/v320_selected_components.csv" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv" \
  "$ROOT/assets/adopted_geometry/377/manifest.json"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-3600}"
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

{
  echo "RUN_ID=$RUN_ID"
  echo "START_BJT=$START_BJT"
  echo "EST_END_BJT=$EST_END_BJT"
  echo "GPU=$GPU"
  echo "BASE_EXP=$BASE_EXP"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC"
  echo "TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC"
  echo "TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC"
  echo "TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC"
  echo "ADOPTED_ASSET_DIR=$ROOT/assets/adopted_geometry/377"
  echo "EXP_ROOT=$EXP_ROOT"
  echo "LOG_DIR=$LOG_DIR"
  echo
  echo "Goal:"
  echo "  v333 runs a wider no-train gate for the formal signed geometry preset."
  echo "  It verifies the render.py preset after promoted adopted CSV assets are used"
  echo "  instead of experiment-log CSV paths."
} > "$LOG_DIR/run_info.txt"

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
    "++export_opacity_maps=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$variant" \
    "wandb_disable=true" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "render_${variant}_done" "status=0"

  log_event "contours_${variant}_start" "$render_exp"
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 30 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${variant}.log" 2>&1
  log_event "contours_${variant}_done" "status=0"

  log_event "residuals_${variant}_start" "$render_exp"
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
  log_event "residuals_${variant}_done" "status=0"
}

BASELINE_EXP="$EXP_ROOT/baseline_no_preset"
FORMAL_EXP="$EXP_ROOT/formal_v320_v307_signed_geometry"

render_variant baseline_no_preset "$BASELINE_EXP" \
  "pipeline.compute_cov3D_python=true" \
  "++pipeline.covariance_mode=default" \
  "++pipeline.covariance_signed_dynamic_enable=false" \
  "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
  "++pipeline.covariance_signed_center_offset_enable=false" \
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

render_variant formal_v320_v307_signed_geometry "$FORMAL_EXP" \
  "++explicit_binding_render_preset=v320_v307_signed_geometry"

"$PYTHON_BIN" - "$SUMMARY" "$WORST_SUMMARY" \
  baseline_no_preset "$BASELINE_EXP" \
  formal_v320_v307_signed_geometry "$FORMAL_EXP" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
worst_path = Path(sys.argv[2])
pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))

def load_metrics(render_exp):
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

def load_records(render_exp):
    render_exp = Path(render_exp)
    paths = [
        render_exp / "diagnostics" / "contours" / "contour_samples.csv",
        render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_samples.csv",
    ]
    records = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = f"c{row.get('cam', '')}_f{int(float(row.get('frame', 0))):06d}"
                rec = records.setdefault(key, {})
                for k, v in row.items():
                    if v is None or v == "":
                        continue
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        rec[k] = v
    return records

def gate(delta):
    strict = (
        delta["fg"] <= 0.0
        and delta["boundary"] <= 0.0
        and delta["edge"] <= 0.0
        and delta["inner"] <= 0.0
        and delta["outer"] <= 0.0
        and delta["hard"] <= -0.000001
    )
    probe = (
        delta["hard"] < -0.00001
        and delta["fg"] <= 0.000015
        and delta["boundary"] <= 0.000015
        and delta["edge"] <= 0.003
        and delta["inner"] <= 0.5
        and delta["outer"] <= 0.5
    )
    return "strict_pass" if strict else ("probe_pass" if probe else "rejected")

metrics = {name: load_metrics(path) for name, path in pairs}
base = metrics["baseline_no_preset"]
header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base", "status",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, render_exp in pairs:
        m = metrics[name]
        d = {key: m[key] - base[key] for key in m}
        status = "baseline" if name == "baseline_no_preset" else gate(d)
        writer.writerow([
            name,
            render_exp,
            f"{m['fg']:.8f}",
            f"{m['boundary']:.8f}",
            f"{m['edge']:.6f}",
            f"{m['inner']:.4f}",
            f"{m['outer']:.4f}",
            f"{m['hard']:.8f}",
            f"{d['fg']:.8f}",
            f"{d['boundary']:.8f}",
            f"{d['edge']:.6f}",
            f"{d['inner']:.4f}",
            f"{d['outer']:.4f}",
            f"{d['hard']:.8f}",
            status,
        ])

base_records = load_records(dict(pairs)["baseline_no_preset"])
formal_records = load_records(dict(pairs)["formal_v320_v307_signed_geometry"])
metric_aliases = {
    "fg": ("fg_l1", "l1_fg", "mean_fg_l1"),
    "boundary": ("boundary_l1", "l1_boundary", "mean_boundary_l1"),
    "edge": ("edge_symmetric_dist_px", "edge", "mean_edge_symmetric_dist_px"),
    "inner": ("inner_missing_pixels", "mean_inner_missing_pixels"),
    "outer": ("outer_leak_pixels", "mean_outer_leak_pixels"),
    "hard": ("hard_residual_score", "mean_hard_residual_score"),
}
worst = []
for key, formal in formal_records.items():
    base_rec = base_records.get(key)
    if not base_rec:
        continue
    row = {"image": key}
    score = 0.0
    for metric, aliases in metric_aliases.items():
        f_val = next((formal[a] for a in aliases if a in formal and isinstance(formal[a], float)), None)
        b_val = next((base_rec[a] for a in aliases if a in base_rec and isinstance(base_rec[a], float)), None)
        if f_val is None or b_val is None:
            continue
        delta = f_val - b_val
        row[f"{metric}_delta"] = delta
        if metric in ("fg", "boundary", "edge", "inner", "outer"):
            score += max(delta, 0.0)
        elif metric == "hard":
            score += max(delta + 0.000001, 0.0)
    row["worsen_score"] = score
    worst.append(row)
worst.sort(key=lambda r: r.get("worsen_score", 0.0), reverse=True)
with worst_path.open("w", encoding="utf-8", newline="") as handle:
    keys = ["image", "worsen_score", "fg_delta", "boundary_delta", "edge_delta", "inner_delta", "outer_delta", "hard_delta"]
    writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in worst[:30]:
        writer.writerow({k: (f"{row[k]:.8f}" if isinstance(row.get(k), float) else row.get(k, "")) for k in keys})

print(json.dumps({
    "summary": str(summary_path),
    "worst": str(worst_path),
    "formal_status": list(csv.DictReader(summary_path.open(encoding="utf-8"), delimiter="\t"))[-1]["status"],
}, indent=2))
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "BASELINE_EXP=$BASELINE_EXP"
  echo "FORMAL_EXP=$FORMAL_EXP"
  echo "SUMMARY=$SUMMARY"
  echo "WORST_SUMMARY=$WORST_SUMMARY"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "BASELINE_EXP=$BASELINE_EXP"
echo "FORMAL_EXP=$FORMAL_EXP"
echo "SUMMARY=$SUMMARY"
echo "WORST_SUMMARY=$WORST_SUMMARY"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
