#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-}"
RUN_ID="${RUN_ID:-formal_377_v338_raw_contour_gate_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/formal/377_v338_raw_contour_gate_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/formal/logs/377_v338_raw_contour_gate_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"

SUMMARY="$LOG_DIR/summary.tsv"
WORST_SUMMARY="$LOG_DIR/worst_frames.tsv"
EVENTS="$LOG_DIR/events.tsv"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT" \
  "$ROOT/assets/adopted_geometry/377/manifest.json" \
  "$ROOT/assets/adopted_geometry/377/v320_selected_components.csv" \
  "$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv" \
  "$ROOT/assets/adopted_geometry/377/v338_temporal_selector_grow_only_guard.json"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done
if [ -n "$CANDIDATE_CKPT" ] && [ ! -e "$CANDIDATE_CKPT" ]; then
  echo "missing CANDIDATE_CKPT: $CANDIDATE_CKPT" >&2
  exit 2
fi

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"

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

render_variant() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  shift 3

  log_event "render_${variant}_start" "$render_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.subject=CoreView_377" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$TEST_VIEWS_SPEC" \
    "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$variant" \
    "wandb_disable=true" \
    "$@" \
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

BASELINE_EXP="$EXP_ROOT/baseline_no_preset"
FORMAL_EXP="$EXP_ROOT/formal_v320_v307_signed_geometry"
V338_EXP="$EXP_ROOT/formal_v338_temporal_selector_grow_only_guard"

render_variant baseline_no_preset "$BASE_CKPT" "$BASELINE_EXP" \
  "pipeline.compute_cov3D_python=true" \
  "++pipeline.covariance_mode=default" \
  "++pipeline.covariance_signed_dynamic_enable=false" \
  "++pipeline.covariance_signed_point_json=" \
  "++pipeline.covariance_signed_point_screen_actuator_enable=false" \
  "++pipeline.covariance_signed_center_offset_enable=false" \
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"

render_variant formal_v320_v307_signed_geometry "$BASE_CKPT" "$FORMAL_EXP" \
  "++explicit_binding_render_preset=v320_v307_signed_geometry"

render_variant formal_v338_temporal_selector_grow_only_guard "$BASE_CKPT" "$V338_EXP" \
  "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"

if [ -n "$CANDIDATE_CKPT" ]; then
  render_variant candidate_v338_temporal_selector_grow_only_guard "$CANDIDATE_CKPT" "$EXP_ROOT/candidate_v338_temporal_selector_grow_only_guard" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"
fi

SUMMARY_PAIRS=(
  baseline_no_preset "$BASELINE_EXP"
  formal_v320_v307_signed_geometry "$FORMAL_EXP"
  formal_v338_temporal_selector_grow_only_guard "$V338_EXP"
)
if [ -n "$CANDIDATE_CKPT" ]; then
  SUMMARY_PAIRS+=(
    candidate_v338_temporal_selector_grow_only_guard "$EXP_ROOT/candidate_v338_temporal_selector_grow_only_guard"
  )
fi

"$PYTHON_BIN" - "$SUMMARY" "$WORST_SUMMARY" "${SUMMARY_PAIRS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
worst_path = Path(sys.argv[2])
pairs = list(zip(sys.argv[3::2], sys.argv[4::2]))

metric_aliases = {
    "fg": ("fg_l1",),
    "boundary": ("boundary_l1",),
    "edge": ("edge_symmetric_dist_px",),
    "inner": ("inner_missing_pixels",),
    "outer": ("outer_leak_pixels",),
    "hard": ("hard_residual_score",),
    "opacity_inner": ("primary_opacity_inner_missing_pixels",),
    "opacity_outer": ("primary_opacity_outer_leak_pixels",),
}


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
    render_exp = Path(render_exp)
    records = {}
    for path in [
        render_exp / "diagnostics/contours/contour_samples.csv",
        render_exp / "diagnostics/boundary_residuals/boundary_residual_samples.csv",
        render_exp / "diagnostics/opacity_footprint/opacity_footprint_samples.csv",
    ]:
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
header = [
    "variant", "render_exp",
    "fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base", "inner_delta_base",
    "outer_delta_base", "hard_delta_base", "opacity_inner_delta_base", "opacity_outer_delta_base",
    "status",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    for name, render_exp in pairs:
        m = metrics[name]
        d = {key: m[key] - base[key] for key in m}
        status = "baseline" if name == "baseline_no_preset" else status_for(d)
        writer.writerow([
            name, render_exp,
            f'{m["fg"]:.8f}', f'{m["boundary"]:.8f}', f'{m["edge"]:.6f}',
            f'{m["inner"]:.4f}', f'{m["outer"]:.4f}', f'{m["hard"]:.8f}',
            f'{m["opacity_inner"]:.4f}', f'{m["opacity_outer"]:.4f}',
            f'{d["fg"]:.8f}', f'{d["boundary"]:.8f}', f'{d["edge"]:.6f}',
            f'{d["inner"]:.4f}', f'{d["outer"]:.4f}', f'{d["hard"]:.8f}',
            f'{d["opacity_inner"]:.4f}', f'{d["opacity_outer"]:.4f}', status,
        ])

base_records = load_records(dict(pairs)["baseline_no_preset"])
rows = []
for name, render_exp in pairs:
    if name == "baseline_no_preset":
        continue
    for image, rec in load_records(render_exp).items():
        base_rec = base_records.get(image)
        if not base_rec:
            continue
        row = {"variant": name, "image": image, "worsen_score": 0.0}
        for metric, aliases in metric_aliases.items():
            val = next((rec[a] for a in aliases if isinstance(rec.get(a), float)), None)
            base_val = next((base_rec[a] for a in aliases if isinstance(base_rec.get(a), float)), None)
            if val is None or base_val is None:
                continue
            delta = val - base_val
            row[f"{metric}_delta"] = delta
            row["worsen_score"] += max(delta, 0.0)
        rows.append(row)
rows.sort(key=lambda item: item["worsen_score"], reverse=True)
with worst_path.open("w", encoding="utf-8", newline="") as handle:
    keys = ["variant", "image", "worsen_score"] + [f"{metric}_delta" for metric in metric_aliases]
    writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    for row in rows[:80]:
        writer.writerow({k: (f"{row[k]:.8f}" if isinstance(row.get(k), float) else row.get(k, "")) for k in keys})
PY

echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "WORST_SUMMARY=$WORST_SUMMARY"
