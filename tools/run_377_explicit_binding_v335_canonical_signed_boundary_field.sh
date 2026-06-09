#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v335_canonical_signed_boundary_field_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/assets/adopted_geometry/377/v320_selected_components.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/assets/adopted_geometry/377/v304_point_contributors_all.csv}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,1]}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v335_canonical_signed_boundary_field_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v335_canonical_signed_boundary_field_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
FIELD_DIR="$LOG_DIR/field"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
WORST_SUMMARY="$LOG_DIR/worst_frames.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$FIELD_DIR"
for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
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

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v335 converts selected residual components into canonical point-level signed
  boundary fields, then runs no-train full-frame A/B. It tests whether the
  fix can move away from per-image component ellipses toward stable signed
  point roles.
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
  "$PYTHON_BIN" tools/make_377_stageB_v335_canonical_signed_boundary_field.py \
    --component-csv "$COMPONENT_CSV" \
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

make_field canonical_128 \
  --max-shrink 128 --max-grow 128 --top-points-per-component 8 --min-component-area 20 \
  --grow-min-boundary 0.12 --grow-min-visible-hits 20 --grow-allowed-regions 1,2 \
  --shrink-min-boundary 0.08
make_field canonical_256 \
  --max-shrink 256 --max-grow 256 --top-points-per-component 8 --min-component-area 20 \
  --grow-min-boundary 0.12 --grow-min-visible-hits 20 --grow-allowed-regions 1,2 \
  --shrink-min-boundary 0.08
make_field canonical_384_balanced \
  --max-shrink 384 --max-grow 384 --top-points-per-component 8 --min-component-area 20 --balance min \
  --grow-min-boundary 0.12 --grow-min-visible-hits 20 --grow-allowed-regions 1,2 \
  --shrink-min-boundary 0.08

BASELINE_EXP="$EXP_ROOT/baseline_no_preset"
FORMAL_EXP="$EXP_ROOT/formal_v320_v307_signed_geometry"
P128_EXP="$EXP_ROOT/canonical_point_128"
P256_EXP="$EXP_ROOT/canonical_point_256"
P384_EXP="$EXP_ROOT/canonical_point_384_balanced"
HYBRID_EXP="$EXP_ROOT/formal_plus_canonical_256"

COMMON_NO_PRESET_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_dynamic_enable=false"
  "++pipeline.covariance_signed_point_screen_actuator_enable=false"
  "++pipeline.covariance_signed_center_offset_enable=false"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
)

point_args() {
  local prior="$1"
  local grow_factor="$2"
  local outer_px="$3"
  printf '%s\n' \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_point_json=$prior" \
    "++pipeline.covariance_signed_point_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_dynamic_enable=false" \
    "++pipeline.covariance_signed_screen_actuator_enable=true" \
    "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
    "++pipeline.covariance_signed_screen_normal_grow_factor=$grow_factor" \
    "++pipeline.covariance_signed_screen_tangent_factor=1.000" \
    "++pipeline.covariance_signed_center_offset_enable=true" \
    "++pipeline.covariance_signed_center_offset_outer_px=$outer_px" \
    "++pipeline.covariance_signed_center_offset_inner_px=0.0" \
    "++pipeline.covariance_signed_center_offset_outer_direction=view_center" \
    "++pipeline.covariance_signed_center_offset_inner_direction=component_center" \
    "++pipeline.covariance_signed_center_offset_max_world_step=0.0020" \
    "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
}

render_variant baseline_no_preset "$BASELINE_EXP" "${COMMON_NO_PRESET_ARGS[@]}"
render_variant formal_v320_v307_signed_geometry "$FORMAL_EXP" \
  "++explicit_binding_render_preset=v320_v307_signed_geometry"

mapfile -t P128_ARGS < <(point_args "$FIELD_DIR/canonical_128.json" 1.015 0.30)
mapfile -t P256_ARGS < <(point_args "$FIELD_DIR/canonical_256.json" 1.020 0.35)
mapfile -t P384_ARGS < <(point_args "$FIELD_DIR/canonical_384_balanced.json" 1.025 0.35)
render_variant canonical_point_128 "$P128_EXP" "${P128_ARGS[@]}"
render_variant canonical_point_256 "$P256_EXP" "${P256_ARGS[@]}"
render_variant canonical_point_384_balanced "$P384_EXP" "${P384_ARGS[@]}"
render_variant formal_plus_canonical_256 "$HYBRID_EXP" \
  "++explicit_binding_render_preset=v320_v307_signed_geometry" \
  "++pipeline.covariance_signed_point_json=$FIELD_DIR/canonical_256.json" \
  "++pipeline.covariance_signed_point_screen_actuator_enable=true" \
  "++pipeline.covariance_signed_screen_normal_grow_factor=1.020"

"$PYTHON_BIN" - "$SUMMARY" "$WORST_SUMMARY" \
  baseline_no_preset "$BASELINE_EXP" \
  formal_v320_v307_signed_geometry "$FORMAL_EXP" \
  canonical_point_128 "$P128_EXP" \
  canonical_point_256 "$P256_EXP" \
  canonical_point_384_balanced "$P384_EXP" \
  formal_plus_canonical_256 "$HYBRID_EXP" <<'PY'
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

def status_for(d):
    strict = (
        d["fg"] <= 0.0 and d["boundary"] <= 0.0 and d["edge"] <= 0.0
        and d["inner"] <= 0.0 and d["outer"] <= 0.0 and d["hard"] <= -0.000001
        and d["opacity_inner"] <= 0.0 and d["opacity_outer"] <= 0.0
    )
    probe = (
        d["hard"] < -0.00001 and d["fg"] <= 0.000015 and d["boundary"] <= 0.000015
        and d["edge"] <= 0.003 and d["inner"] <= 0.5 and d["outer"] <= 0.5
        and d["opacity_inner"] <= 0.5 and d["opacity_outer"] <= 0.5
    )
    return "strict_pass" if strict else ("probe_pass" if probe else "rejected")

metrics = {name: load_metrics(path) for name, path in pairs}
base = metrics["baseline_no_preset"]
header = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base", "inner_delta_base", "outer_delta_base",
    "hard_delta_base", "opacity_inner_delta_base", "opacity_outer_delta_base", "status",
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
with worst_path.open("w", encoding="utf-8", newline="") as handle:
    keys = ["variant", "image", "worsen_score"] + [f"{k}_delta" for k in metric_aliases]
    writer = csv.DictWriter(handle, fieldnames=keys, delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    rows = []
    for name, render_exp in pairs:
        if name == "baseline_no_preset":
            continue
        for key, rec in load_records(render_exp).items():
            base_rec = base_records.get(key)
            if not base_rec:
                continue
            out = {"variant": name, "image": key, "worsen_score": 0.0}
            for metric, aliases in metric_aliases.items():
                f_val = next((rec[a] for a in aliases if isinstance(rec.get(a), float)), None)
                b_val = next((base_rec[a] for a in aliases if isinstance(base_rec.get(a), float)), None)
                if f_val is None or b_val is None:
                    continue
                delta = f_val - b_val
                out[f"{metric}_delta"] = delta
                out["worsen_score"] += max(delta, 0.0)
            rows.append(out)
    rows.sort(key=lambda row: row["worsen_score"], reverse=True)
    for row in rows[:80]:
        writer.writerow({k: (f"{row[k]:.8f}" if isinstance(row.get(k), float) else row.get(k, "")) for k in keys})
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
