#!/usr/bin/env bash
set -euo pipefail
shopt -s inherit_errexit

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
SUBJECT="${SUBJECT:-CoreView_377}"
SEED="${SEED:-20260710}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_TAG="${RUN_TAG:-coreview377_tether_quality_full_pipeline_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S')_bjt}"
ZERO_EXP_ROOT="$ROOT/exp/zero_train_to_v395"
RUN="${RUN:-$ZERO_EXP_ROOT/$RUN_TAG}"
LOG_DIR="$RUN/logs"
SMOKE="${SMOKE:-0}"
STOP_AFTER_CANARY="${STOP_AFTER_CANARY:-0}"

CANARY_TETHER_CKPT="${CANARY_TETHER_CKPT:-$ZERO_EXP_ROOT/coreview377_surface_coherent_anchor_tether_20260711_bjt/run_20260711_222829_bjt/neutral_longhorizon_fromzero/ckpt64000.pth}"
TETHER_LAUNCHER="$ROOT/exp/zero_train_to_v395/coreview377_surface_coherent_anchor_tether_20260711_bjt/launch_surface_coherent_anchor_tether.sh"
SELECTOR="$ROOT/tools/select_tether_quality_candidate.py"

CONTINUATION_ITERS="${CONTINUATION_ITERS:-32000}"
LATE_CLEAN_ITERS="${LATE_CLEAN_ITERS:-4000}"
RESIDUAL_ITERS="${RESIDUAL_ITERS:-3000}"
CONTINUATION_TESTS="${CONTINUATION_TESTS:-[2000,4000,6000,8000,10000,12000,14000,16000,18000,20000,22000,24000,26000,28000,30000,32000]}"
CONTINUATION_SAVES="${CONTINUATION_SAVES:-[8000,16000,24000,32000]}"
LATE_TESTS="${LATE_TESTS:-[500,1000,1500,2000,2500,3000,3500,4000]}"
LATE_SAVES="${LATE_SAVES:-[1000,2000,3000,4000]}"
RESIDUAL_TESTS="${RESIDUAL_TESTS:-[500,1000,1500,2000,2500,3000]}"
RESIDUAL_SAVES="${RESIDUAL_SAVES:-[500,1000,1500,2000,2500,3000]}"

if [[ "$SMOKE" == "1" ]]; then
  CONTINUATION_ITERS=2
  LATE_CLEAN_ITERS=2
  RESIDUAL_ITERS=2
  CONTINUATION_TESTS="[2]"
  CONTINUATION_SAVES="[2]"
  LATE_TESTS="[2]"
  LATE_SAVES="[2]"
  RESIDUAL_TESTS="[2]"
  RESIDUAL_SAVES="[2]"
fi

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" > "$LOG_DIR/pipeline.pid"
printf '%s\n' "$RUN" > "$ZERO_EXP_ROOT/coreview377_tether_quality_full_pipeline_latest.txt"
exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

COMMON_OPTIONS=(
  --option stageA_377_multiview_explicit_hq_v4
  --option stageA_377_multiview_explicit_hq_v4_fast
  --option stageA_377_multiview_explicit_hq_v89a_fresh_clarity_mainline_safe_v1
  --option stageA_377_multiview_explicit_hq_v89e_fresh_clarity_mainline_decay0005_v1
  --option stageA_377_multiview_explicit_hq_v90f_fresh_decay0002_noise165_v1
  --option stageA_377_multiview_explicit_hq_fromzero_stable_base_v1
)

log_event() {
  printf '[tether-quality] %s BJT=%s\n' "$*" "$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')"
}

write_state() {
  local phase="$1"
  "$PY" - "$RUN/pipeline_state.json" "$phase" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {"phase": sys.argv[2], "updated_utc": datetime.now(timezone.utc).isoformat()}
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_diagnostic() {
  local checkpoint="$1"
  local out_dir="$2"
  local frames="$3"
  local metrics_path="$4"
  mkdir -p "$out_dir"
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PY" tools/run_377_stageA2_multiview_explicit.py \
      --exp-dir "$out_dir" \
      --python "$PY" \
      --dataset zjumocap_377_multiview_hq \
      --dataset-root "$DATA_ROOT" \
      --non-rigid hashgrid \
      --texture shallow_mlp_lownoise \
      "${COMMON_OPTIONS[@]}" \
      --extra-override dataset.subject="$SUBJECT" \
      --extra-override seed="$SEED" \
      --extra-override texture_name=shallow_mlp_lownoise \
      --extra-override start_checkpoint="$checkpoint" \
      --extra-override dataset.test_frames.view="$frames" \
      --extra-override resume.use_checkpoint_iteration_as_offset=true \
      --extra-override ++resume.restore_gaussian_optimizer_state=false \
      --extra-override ++resume.restore_converter_optimizer_state=false \
      --extra-override ++resume.restore_converter_scheduler_state=false \
      --extra-override '++resume.partial_converter_missing_keys_allow_patterns=[texture.structured_trunk_,deformer.rigid.forward_trunk_mlp.,texture.shadow_handoff_approved]' \
      --extra-override ++opt.camera_geometry_enable=false \
      --extra-override ++opt.camera_affine_enable=false \
      --extra-override ++opt.diagnostic_validate_at_start=true \
      --extra-override ++opt.diagnostic_validate_interval_iteration=global \
      --extra-override ++opt.diagnostic_validation_metrics_path="$metrics_path" \
      --extra-override ++opt.diagnostic_exit_after_start_validation=true \
      --extra-override ++validation_image_log_limit=0
}

run_render_contour() {
  local checkpoint="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PY" render.py \
      mode=test \
      dataset=zjumocap_377_multiview_hq \
      dataset.root_dir="$DATA_ROOT" \
      dataset.preload=false \
      dataset.parsing_prior.enable=false \
      dataset.person_crop.enable=false \
      rigid=explicit_binding \
      non_rigid=hashgrid \
      pose_correction=direct \
      texture=shallow_mlp_lownoise \
      'option=[stageA_377_multiview_explicit_hq_v1,stageA_377_multiview_explicit_hq_v4,stageA_377_multiview_explicit_hq_v4_fast,stageA_377_multiview_explicit_hq_v89a_fresh_clarity_mainline_safe_v1,stageA_377_multiview_explicit_hq_v89e_fresh_clarity_mainline_decay0005_v1,stageA_377_multiview_explicit_hq_v90f_fresh_decay0002_noise165_v1,stageA_377_multiview_explicit_hq_fromzero_stable_base_v1]' \
      +exp_dir="$out_dir/render" \
      wandb_disable=true \
      dataset.subject="$SUBJECT" \
      'dataset.test_views.view=[21,22,23]' \
      'dataset.test_frames.view=[0,570,60]' \
      load_ckpt="$checkpoint" \
      texture_name=shallow_mlp_lownoise \
      evaluate=true \
      '++resume.partial_converter_missing_keys_allow_patterns=[texture.structured_trunk_,deformer.rigid.forward_trunk_mlp.,texture.shadow_handoff_approved]' \
      ++opt.camera_geometry_enable=false \
      ++opt.camera_affine_enable=false \
      ++validation_image_log_limit=0 \
      ++export_opacity_maps=false
  "$PY" tools/analyze_render_quality_edges.py \
    --render-dir "$out_dir/render/test-view/renders" \
    --gt-template "$DATA_ROOT/$SUBJECT/{cam}/{frame:06d}.jpg" \
    --mask-template "$DATA_ROOT/$SUBJECT/{cam}/{frame:06d}.png" \
    --out-dir "$out_dir/contour" \
    --topk 12
}

run_continuation() {
  local start_ckpt="$1"
  local phase_root="$2"
  local out_dir="$phase_root/continuation"
  mkdir -p "$out_dir"
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PY" tools/run_377_stageA2_multiview_explicit.py \
      --exp-dir "$out_dir" \
      --python "$PY" \
      --dataset zjumocap_377_multiview_hq \
      --dataset-root "$DATA_ROOT" \
      --non-rigid hashgrid \
      --texture shallow_mlp_lownoise \
      "${COMMON_OPTIONS[@]}" \
      --option stageA_377_multiview_explicit_hq_fromzero_optimizer_preserving_continuation_v1 \
      --extra-override dataset.subject="$SUBJECT" \
      --extra-override seed="$SEED" \
      --extra-override texture_name=shallow_mlp_lownoise \
      --extra-override start_checkpoint="$start_ckpt" \
      --extra-override resume.use_checkpoint_iteration_as_offset=true \
      --extra-override resume.restore_gaussian_optimizer_state=true \
      --extra-override resume.require_gaussian_optimizer_state_restore=true \
      --extra-override resume.restore_converter_optimizer_state=true \
      --extra-override resume.restore_converter_scheduler_state=true \
      --extra-override opt.iterations="$CONTINUATION_ITERS" \
      --extra-override test_iterations="$CONTINUATION_TESTS" \
      --extra-override save_iterations="$CONTINUATION_SAVES" \
      --extra-override checkpoint_iterations="$CONTINUATION_SAVES" \
      --extra-override opt.densify_until_iter=0 \
      --extra-override opt.densify_from_iter=1000000 \
      --extra-override opt.densification_interval=1000000 \
      --extra-override opt.opacity_reset_interval=1000000 \
      --extra-override opt.percent_dense=0.0 \
      --extra-override ++validation_image_log_limit=0
}

select_continuation_checkpoint() {
  local start_ckpt="$1"
  local phase_root="$2"
  local audit_root="$phase_root/continuation_audit"
  local tsv="$audit_root/candidates.tsv"
  local manifest="$audit_root/candidates.json"
  local selection="$audit_root/selection.json"
  mkdir -p "$audit_root"
  printf 'label\tcheckpoint\tmetrics\tcontour\n' > "$tsv"

  local candidates=("$start_ckpt")
  [[ -f "$phase_root/continuation/best_ckpt.pth" ]] && candidates+=("$phase_root/continuation/best_ckpt.pth")
  local checkpoint
  while IFS= read -r checkpoint; do candidates+=("$checkpoint"); done < <(
    find "$phase_root/continuation" -maxdepth 1 -type f -name 'ckpt*.pth' | sort -V
  )

  declare -A seen=()
  local index=0
  for checkpoint in "${candidates[@]}"; do
    checkpoint="$(readlink -f "$checkpoint")"
    [[ -n "${seen[$checkpoint]:-}" ]] && continue
    seen[$checkpoint]=1
    local label="candidate_${index}"
    local candidate_root="$audit_root/$label"
    run_diagnostic "$checkpoint" "$candidate_root/metrics_eval" '[0,570,60]' "$candidate_root/metrics.json"
    run_render_contour "$checkpoint" "$candidate_root"
    printf '%s\t%s\t%s\t%s\n' "$label" "$checkpoint" "$candidate_root/metrics.json" "$candidate_root/contour/render_quality_summary.json" >> "$tsv"
    index=$((index + 1))
  done

  "$PY" - "$tsv" "$manifest" <<'PY'
import csv
import json
import pathlib
import sys

rows = []
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        metrics_payload = json.loads(pathlib.Path(row["metrics"]).read_text())
        metrics = metrics_payload.get("best_eval") or metrics_payload.get("test") or metrics_payload
        contour = json.loads(pathlib.Path(row["contour"]).read_text())
        rows.append({
            "label": row["label"],
            "checkpoint": row["checkpoint"],
            "lpips_fg": float(metrics["lpips_fg"]),
            "psnr_fg": float(metrics["psnr_fg"]),
            "edge_px": float(contour["mean_edge_symmetric_dist_px"]),
            "boundary_l1": float(contour["mean_boundary_l1"]),
        })
pathlib.Path(sys.argv[2]).write_text(json.dumps({"candidates": rows}, indent=2) + "\n")
PY
  "$PY" "$SELECTOR" select-continuation --manifest "$manifest" --output "$selection"
  "$PY" - "$selection" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])
PY
}

run_late_clean() {
  local start_ckpt="$1"
  local phase_root="$2"
  local out_dir="$phase_root/late_clean"
  mkdir -p "$out_dir"
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PY" tools/run_377_stageA2_multiview_explicit.py \
      --exp-dir "$out_dir" --python "$PY" \
      --dataset zjumocap_377_multiview_hq --dataset-root "$DATA_ROOT" \
      --non-rigid hashgrid --texture shallow_mlp_lownoise \
      "${COMMON_OPTIONS[@]}" \
      --option stageA_377_multiview_explicit_hq_fromzero_late_clean_refine_v1 \
      --extra-override dataset.subject="$SUBJECT" \
      --extra-override seed="$SEED" \
      --extra-override start_checkpoint="$start_ckpt" \
      --extra-override 'dataset.test_frames.view=[0,570,60]' \
      --extra-override resume.use_checkpoint_iteration_as_offset=true \
      --extra-override resume.restore_gaussian_optimizer_state=false \
      --extra-override resume.restore_converter_optimizer_state=false \
      --extra-override resume.restore_converter_scheduler_state=false \
      --extra-override opt.iterations="$LATE_CLEAN_ITERS" \
      --extra-override test_iterations="$LATE_TESTS" \
      --extra-override save_iterations="$LATE_SAVES" \
      --extra-override checkpoint_iterations="$LATE_SAVES" \
      --extra-override ++validation_image_log_limit=0
}

run_residual() {
  local start_ckpt="$1"
  local phase_root="$2"
  local out_dir="$phase_root/residual_balanced"
  mkdir -p "$out_dir"
  HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONUNBUFFERED=1 \
    "$PY" tools/run_377_stageA2_multiview_explicit.py \
      --exp-dir "$out_dir" --python "$PY" \
      --dataset zjumocap_377_multiview_hq --dataset-root "$DATA_ROOT" \
      --non-rigid hashgrid --texture shallow_mlp_lownoise \
      "${COMMON_OPTIONS[@]}" \
      --option stageA_fromzero_residual_balanced_late_refine_v1 \
      --extra-override dataset.subject="$SUBJECT" \
      --extra-override seed="$SEED" \
      --extra-override start_checkpoint="$start_ckpt" \
      --extra-override 'dataset.test_frames.view=[0,570,60]' \
      --extra-override resume.use_checkpoint_iteration_as_offset=true \
      --extra-override resume.restore_gaussian_optimizer_state=false \
      --extra-override resume.restore_converter_optimizer_state=false \
      --extra-override resume.restore_converter_scheduler_state=false \
      --extra-override opt.photometric_correction_enable=false \
      --extra-override opt.iterations="$RESIDUAL_ITERS" \
      --extra-override test_iterations="$RESIDUAL_TESTS" \
      --extra-override save_iterations="$RESIDUAL_SAVES" \
      --extra-override checkpoint_iterations="$RESIDUAL_SAVES" \
      --extra-override ++validation_image_log_limit=0
}

run_final_gate() {
  local checkpoint="$1"
  local phase_root="$2"
  local result_path="$3"
  local final_root="$phase_root/final_audit"
  run_diagnostic "$checkpoint" "$final_root/same30_eval" '[0,570,60]' "$final_root/same30.json"
  run_diagnostic "$checkpoint" "$final_root/original57_eval" '[0,570,30]' "$final_root/original57.json"
  run_render_contour "$checkpoint" "$final_root"
  "$PY" "$SELECTOR" evaluate-final \
    --same30 "$final_root/same30.json" \
    --original57 "$final_root/original57.json" \
    --contour "$final_root/contour/render_quality_summary.json" \
    --output "$result_path"
  printf '%s\n' "$(readlink -f "$checkpoint")" > "$phase_root/FINAL_BEST_CKPT.txt"
  "$PY" - "$result_path" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["accepted"] else 1)
PY
}

run_continuation_tail() {
  local phase="$1"
  local start_ckpt="$2"
  local phase_root="$3"
  local result_path="$4"
  mkdir -p "$phase_root"
  log_event "${phase}_CONTINUATION_START checkpoint=$start_ckpt"
  run_continuation "$start_ckpt" "$phase_root"
  local selected_ckpt
  selected_ckpt="$(select_continuation_checkpoint "$start_ckpt" "$phase_root" | tail -n 1)"
  log_event "${phase}_CONTINUATION_SELECTED checkpoint=$selected_ckpt"
  run_late_clean "$selected_ckpt" "$phase_root"
  local late_ckpt="$phase_root/late_clean/best_ckpt.pth"
  [[ -f "$late_ckpt" ]] || late_ckpt="$selected_ckpt"
  run_residual "$late_ckpt" "$phase_root"
  local residual_ckpt="$phase_root/residual_balanced/best_ckpt.pth"
  [[ -f "$residual_ckpt" ]] || residual_ckpt="$late_ckpt"
  run_final_gate "$residual_ckpt" "$phase_root" "$result_path"
}

log_event "PIPELINE_START RUN=$RUN"
write_state "canary_continuation"
if [[ ! -f "$CANARY_TETHER_CKPT" ]]; then
  echo "missing CANARY_TETHER_CKPT: $CANARY_TETHER_CKPT" >&2
  exit 2
fi

CANARY_RUN="$RUN/canary"
if run_continuation_tail "CANARY" "$CANARY_TETHER_CKPT" "$CANARY_RUN" "$RUN/CANARY_RESULT.json"; then
  log_event "CANARY_PASSED"
else
  log_event "CANARY_REJECTED"
  write_state "canary_rejected"
  log_event "PIPELINE_DONE_BJT canary_rejected"
  exit 0
fi

if [[ "$STOP_AFTER_CANARY" == "1" ]]; then
  log_event "PIPELINE_DONE_BJT stop_after_canary"
  exit 0
fi

write_state "full_fromzero"
log_event "FULL_FROMZERO_START"
FULL_RUN="$RUN/full"
mkdir -p "$FULL_RUN/stage1"
BASE_RUN="$FULL_RUN/stage1" RUN="$FULL_RUN/stage1" SMOKE="$SMOKE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$TETHER_LAUNCHER"
FULL_STAGE1_CKPT="$FULL_RUN/stage1/neutral_longhorizon_fromzero/ckpt64000.pth"
if [[ "$SMOKE" == "1" ]]; then
  FULL_STAGE1_CKPT="$FULL_RUN/stage1/neutral_longhorizon_fromzero/ckpt20.pth"
fi
if [[ ! -f "$FULL_STAGE1_CKPT" ]]; then
  echo "missing full from-zero stage1 checkpoint: $FULL_STAGE1_CKPT" >&2
  exit 3
fi

if run_continuation_tail "full" "$FULL_STAGE1_CKPT" "$FULL_RUN" "$RUN/FULL_RESULT.json"; then
  log_event "FULL_ACCEPTED"
  cp "$FULL_RUN/FINAL_BEST_CKPT.txt" "$RUN/FINAL_BEST_CKPT.txt"
else
  log_event "FULL_REJECTED"
  cp "$FULL_RUN/FINAL_BEST_CKPT.txt" "$RUN/FINAL_BEST_CKPT.txt"
fi
write_state "complete"
log_event "PIPELINE_DONE_BJT"
