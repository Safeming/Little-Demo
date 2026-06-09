#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-4}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
CANDIDATE_CKPT="${CANDIDATE_CKPT:-$ROOT/exp/formal/377_v338_semantic_train_formal_377_v338_mainline_20260522_141739_bjt/ckpt138410.pth}"
V369_LOG_DIR="${V369_LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v369_residual_component_multi_micro_grouped_actuator_v369_multi_target_full_retry2_20260526_141742_bjt}"
V369_SEED_ASSET_JSON="${V369_SEED_ASSET_JSON:-$V369_LOG_DIR/assets/v369_seed_residual_component_multi_micro_grouped_actuator_asset.json}"
V369_GROUP_VALIDATION_TSV="${V369_GROUP_VALIDATION_TSV:-$V369_LOG_DIR/group_validation.tsv}"

RUN_ID="${RUN_ID:-v369_silent_action_diagnosis_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v369_silent_action_diagnosis_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v369_silent_action_diagnosis_${RUN_ID}}"
HYDRA_RUN_ROOT="${HYDRA_RUN_ROOT:-$LOG_DIR/hydra_runtime}"
ASSET_DIR="$LOG_DIR/assets"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,60]}"
RENDER_EXPORT_OPACITY_THRESHOLD="${RENDER_EXPORT_OPACITY_THRESHOLD:-0.06}"
CHILD_OPACITY="${CHILD_OPACITY:-0.04}"
PAIR_LIMIT="${PAIR_LIMIT:-6}"
PAIR_IDS="${PAIR_IDS:-}"
STRONG_OPACITY_MULT="${STRONG_OPACITY_MULT:-6.0}"
STRONG_RADIUS_MULT="${STRONG_RADIUS_MULT:-2.0}"
SPLIT_DEBUG="${SPLIT_DEBUG:-1}"

PAIRS_TSV="$LOG_DIR/diagnostic_pairs.tsv"
RESULTS_TSV="$LOG_DIR/diagnostic_results.tsv"
EVENTS="$LOG_DIR/events.tsv"

for required in \
  "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$CANDIDATE_CKPT" "$DATA_ROOT" \
  "$V369_SEED_ASSET_JSON" "$V369_GROUP_VALIDATION_TSV"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT" "$ASSET_DIR"
printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'pair_id\tsource_component_key\timage_name\tview\tframe\tbaseline_status\tbaseline_gain\tvariant\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\topacity_inner_delta\topacity_outer_delta\tdebug_input_children\tdebug_pair_actions\tdebug_active\tdebug_appended\tdebug_activation_fail\tdebug_pair_fail\tdebug_self_protect_drop\tdebug_log\n' > "$RESULTS_TSV"

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

log_event "select_pairs_start" "$V369_GROUP_VALIDATION_TSV"
"$PYTHON_BIN" - "$V369_GROUP_VALIDATION_TSV" "$V369_SEED_ASSET_JSON" "$PAIRS_TSV" "$PAIR_LIMIT" "$PAIR_IDS" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

validation_path = Path(sys.argv[1])
asset_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
limit = int(float(sys.argv[4]))
explicit = [item.strip() for item in sys.argv[5].split(",") if item.strip()]

rows = list(csv.DictReader(validation_path.open("r", encoding="utf-8"), delimiter="\t"))
asset = json.loads(asset_path.read_text(encoding="utf-8"))
groups = {str(g.get("pair_id", "")): g for g in asset.get("action_groups", [])}

def parse_image(name):
    m = re.match(r"c(\d+)_f(\d+)$", name or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def enrich(row):
    pair_id = str(row.get("pair_id", ""))
    group = groups.get(pair_id, {})
    image = str(row.get("image_name", "") or group.get("image_name", "") or group.get("source_image_name", ""))
    view, frame = parse_image(image)
    if view is None:
        return None
    out = dict(row)
    out["image_name"] = image
    out["view"] = view
    out["frame"] = frame
    out["target_rank"] = group.get("residual_target_rank", "")
    out["group_score"] = group.get("frame_score", "")
    return out

selected = []
if explicit:
    wanted = set(explicit)
    for row in rows:
        if row.get("pair_id") in wanted:
            item = enrich(row)
            if item is not None:
                selected.append(item)
else:
    keep = [enrich(row) for row in rows if row.get("status") == "keep"]
    drop = [enrich(row) for row in rows if row.get("status") != "keep"]
    keep = [row for row in keep if row is not None]
    drop = [row for row in drop if row is not None]
    keep.sort(key=lambda r: float(r.get("target_gain", 0.0) or 0.0), reverse=True)
    drop.sort(key=lambda r: (
        str(r.get("image_name", "")),
        str(r.get("source_component_key", "")),
        str(r.get("pair_id", "")),
    ))
    if keep:
        selected.append(keep[0])
    buckets = {}
    for row in drop:
        image = row["image_name"]
        key = image if image in ("c21_f000300", "c21_f000360", "c21_f000420", "c21_f000480") else "other"
        buckets.setdefault(key, []).append(row)
    for key in ("c21_f000300", "c21_f000360", "c21_f000420", "c21_f000480", "other"):
        for row in buckets.get(key, [])[:2]:
            selected.append(row)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    selected = selected[:limit]

with out_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
        "pair_id", "source_component_key", "image_name", "view", "frame",
        "status", "target_gain", "inner_delta_control", "outer_delta_control",
        "opacity_inner_delta_control", "opacity_outer_delta_control",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in selected:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
print(f"wrote {out_path} pairs={len(selected)}")
PY
log_event "select_pairs_done" "$PAIRS_TSV"

make_variant_asset() {
  local pair_id="$1"
  local variant="$2"
  local out_json="$3"
  "$PYTHON_BIN" - "$V369_SEED_ASSET_JSON" "$pair_id" "$variant" "$out_json" "$STRONG_OPACITY_MULT" "$STRONG_RADIUS_MULT" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
pair_id = sys.argv[2]
variant = sys.argv[3]
out = Path(sys.argv[4])
opacity_mult = float(sys.argv[5])
radius_mult = float(sys.argv[6])
data = json.loads(src.read_text(encoding="utf-8"))
children = [dict(c) for c in data.get("children", []) if str(c.get("pair_id", "")) == pair_id]
actions = [dict(a) for a in data.get("actions", []) if str(a.get("pair_id", "")) == pair_id]
groups = [dict(g) for g in data.get("action_groups", []) if str(g.get("pair_id", "")) == pair_id]

for child in children:
    if variant in ("no_protect", "no_activation", "strong"):
        child["child_self_protect_enable"] = False
    if variant == "no_activation":
        child["activation_required"] = False
        child["activation_owner_gate"] = False
        child["pair_activation_required"] = False
    if variant == "strong":
        child["child_opacity"] = float(child.get("child_opacity", 0.04) or 0.04) * opacity_mult
        child["child_radius_scale"] = float(child.get("child_radius_scale", 1.0) or 1.0) * radius_mult
        cov = child.get("canonical_covariance", None)
        if isinstance(cov, list):
            try:
                child["canonical_covariance"] = [
                    [float(v) * (radius_mult ** 2) for v in row] for row in cov
                ]
            except Exception:
                pass
        cov6 = child.get("canonical_covariance_6", None)
        if isinstance(cov6, list):
            try:
                child["canonical_covariance_6"] = [float(v) * (radius_mult ** 2) for v in cov6]
            except Exception:
                pass

for action in actions:
    if variant == "no_activation":
        action["activation_required"] = False
        action["activation_owner_gate"] = False
        action["pair_activation_required"] = False

payload = {
    **{k: v for k, v in data.items() if k not in ("children", "actions", "action_groups")},
    "version": f"v369_silent_diagnosis_{variant}",
    "children": children,
    "actions": actions,
    "action_groups": groups,
    "child_count": len(children),
    "action_count": len(actions),
    "group_count": len(groups),
    "diagnostic_variant": variant,
    "diagnostic_pair_id": pair_id,
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"wrote {out} variant={variant} children={len(children)} actions={len(actions)} groups={len(groups)}")
PY
}

render_and_analyze() {
  local label="$1"
  local views_spec="$2"
  local frames_spec="$3"
  local render_exp="$4"
  local log_file="$5"
  shift 5
  log_event "render_start" "$label -> $render_exp"
  env "${COMMON_ENV[@]}" "STAGEB_SPLIT_CHILD_DEBUG=$SPLIT_DEBUG" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$CANDIDATE_CKPT" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.subject=CoreView_377" \
    "dataset.train_views=$TRAIN_VIEWS_SPEC" \
    "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
    "dataset.test_views.view=$views_spec" \
    "dataset.test_frames.view=$frames_spec" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++export_opacity_maps=true" \
    "++render_export_refine=false" \
    "++render_export_opacity_threshold=$RENDER_EXPORT_OPACITY_THRESHOLD" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/$label" \
    "wandb_disable=true" \
    "$@" \
    > "$log_file" 2>&1
  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
    --split-dir test-view --band-width 7 --topk 16 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_boundary_residuals.py \
    --render-exp "$render_exp" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
    --split-dir test-view --render-support-threshold 0.025 --close-kernel 5 \
    --band-width 7 --search-band-width 24 --topk 16 \
    --out-dir "$render_exp/diagnostics/boundary_residuals" \
    > "$LOG_DIR/boundary_residuals_${label}.log" 2>&1
  "$PYTHON_BIN" tools/analyze_377_opacity_footprint.py \
    --render-exp "$render_exp" --dataset-root "$DATA_ROOT" --subject CoreView_377 \
    --split-dir test-view --render-support-threshold 0.025 --primary-opacity-threshold 0.06 \
    --opacity-thresholds 0.02,0.04,0.06,0.08,0.10 --rgb-close-kernel 5 \
    --opacity-close-kernel 3 --band-width 7 --search-band-width 24 --topk 16 \
    --out-dir "$render_exp/diagnostics/opacity_footprint" \
    > "$LOG_DIR/opacity_footprint_${label}.log" 2>&1
  log_event "render_done" "$label"
}

append_result() {
  local pair_id="$1"
  local source_component_key="$2"
  local image_name="$3"
  local view="$4"
  local frame="$5"
  local baseline_status="$6"
  local baseline_gain="$7"
  local variant="$8"
  local control_exp="$9"
  local candidate_exp="${10}"
  local debug_log="${11}"
  "$PYTHON_BIN" - "$RESULTS_TSV" "$pair_id" "$source_component_key" "$image_name" "$view" "$frame" \
    "$baseline_status" "$baseline_gain" "$variant" "$control_exp" "$candidate_exp" "$debug_log" <<'PY'
import ast
import csv
import json
import re
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
pair_id, source_key, image_name, view, frame = sys.argv[2:7]
baseline_status, baseline_gain, variant = sys.argv[7:10]
control_exp = Path(sys.argv[10])
candidate_exp = Path(sys.argv[11])
debug_log = Path(sys.argv[12])

def load_metrics(render_exp):
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

control = load_metrics(control_exp)
candidate = load_metrics(candidate_exp)
delta = {key: candidate[key] - control[key] for key in control}

debug = {
    "input_children": "",
    "input_pair_actions": "",
    "active": "",
    "appended": "",
    "activation_fail": "",
    "pair_fail": "",
    "self_protect_drop": "",
}
text = debug_log.read_text(encoding="utf-8", errors="replace") if debug_log.exists() else ""
for line in reversed(text.splitlines()):
    if "[split_child_debug]" not in line or f"filter=\"pair_id={pair_id}\"" not in line and f"filter='pair_id={pair_id}'" not in line:
        continue
    active_match = re.search(r"active=(\d+)", line)
    appended_match = re.search(r"appended=(\d+)", line)
    counters_match = re.search(r"counters=(\{.*?\})(?: rows=|$)", line)
    if active_match:
        debug["active"] = active_match.group(1)
    if appended_match:
        debug["appended"] = appended_match.group(1)
    if counters_match:
        try:
            counters = ast.literal_eval(counters_match.group(1))
            for key in ("input_children", "input_pair_actions", "activation_fail", "pair_fail", "self_protect_drop", "appended"):
                debug[key] = str(counters.get(key, debug.get(key, "")))
        except Exception:
            pass
    break

row = {
    "pair_id": pair_id,
    "source_component_key": source_key,
    "image_name": image_name,
    "view": view,
    "frame": frame,
    "baseline_status": baseline_status,
    "baseline_gain": baseline_gain,
    "variant": variant,
    "debug_log": str(debug_log),
    **{f"{key}_delta": delta[key] for key in ("fg", "boundary", "edge", "inner", "outer", "hard", "opacity_inner", "opacity_outer")},
    "debug_input_children": debug["input_children"],
    "debug_pair_actions": debug["input_pair_actions"],
    "debug_active": debug["active"],
    "debug_appended": debug["appended"],
    "debug_activation_fail": debug["activation_fail"],
    "debug_pair_fail": debug["pair_fail"],
    "debug_self_protect_drop": debug["self_protect_drop"],
}
fieldnames = [
    "pair_id", "source_component_key", "image_name", "view", "frame",
    "baseline_status", "baseline_gain", "variant",
    "fg_delta", "boundary_delta", "edge_delta", "inner_delta", "outer_delta",
    "hard_delta", "opacity_inner_delta", "opacity_outer_delta",
    "debug_input_children", "debug_pair_actions", "debug_active", "debug_appended",
    "debug_activation_fail", "debug_pair_fail", "debug_self_protect_drop", "debug_log",
]
with out_path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
    writer.writerow(row)
print(f"{pair_id} {variant} inner={delta['inner']:.6f} outer={delta['outer']:.6f} op_inner={delta['opacity_inner']:.6f} op_outer={delta['opacity_outer']:.6f} debug={debug}")
PY
}

while IFS=$'\t' read -r pair_id source_component_key image_name view frame baseline_status baseline_gain _rest; do
  if [ "$pair_id" = "pair_id" ]; then
    continue
  fi
  safe_pair="$(printf '%s' "$pair_id" | tr -c 'A-Za-z0-9_' '_')"
  next_frame=$((frame + 1))
  views_spec="[$view]"
  frames_spec="[$frame,$next_frame,1]"
  pair_root="$EXP_ROOT/$safe_pair"
  control_exp="$pair_root/control_v338"
  control_log="$LOG_DIR/render_${safe_pair}_control.log"
  render_and_analyze "${safe_pair}_control" "$views_spec" "$frames_spec" "$control_exp" "$control_log" \
    "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard"

  for variant in normal no_protect no_activation strong; do
    asset_json="$ASSET_DIR/${safe_pair}_${variant}.json"
    make_variant_asset "$pair_id" "$variant" "$asset_json" > "$LOG_DIR/make_${safe_pair}_${variant}.log" 2>&1
    candidate_exp="$pair_root/$variant"
    candidate_log="$LOG_DIR/render_${safe_pair}_${variant}.log"
    render_and_analyze "${safe_pair}_${variant}" "$views_spec" "$frames_spec" "$candidate_exp" "$candidate_log" \
      "pipeline.compute_cov3D_python=true" \
      "++explicit_binding_render_preset=v338_temporal_selector_grow_only_guard" \
      "++pipeline.split_child_component_enable=true" \
      "++pipeline.split_child_component_asset_json=$asset_json" \
      "++pipeline.split_child_component_action_filter='pair_id=$pair_id'" \
      "++pipeline.split_child_component_action_required=true" \
      "++pipeline.split_child_component_opacity=$CHILD_OPACITY" \
      "++pipeline.split_child_component_radius_scale=1.0" \
      "++pipeline.split_child_component_max_children=-1"
    append_result "$pair_id" "$source_component_key" "$image_name" "$view" "$frame" \
      "$baseline_status" "$baseline_gain" "$variant" "$control_exp" "$candidate_exp" "$candidate_log"
  done
done < "$PAIRS_TSV"

log_event "finished" "$RESULTS_TSV"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "PAIRS_TSV=$PAIRS_TSV"
echo "RESULTS_TSV=$RESULTS_TSV"
