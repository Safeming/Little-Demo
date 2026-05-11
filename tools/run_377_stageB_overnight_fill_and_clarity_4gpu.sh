#!/usr/bin/env bash
set -u
set -o pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script directly with bash, not via source" >&2
  return 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/3dgs-avatar/bin/python}"
RUN_ID="${RUN_ID:-stageB_overnight_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
SEED="${SEED:--1}"
GPUS="${GPUS:-0,1,2,3}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-32400}"
START_EPOCH="${START_EPOCH:-$(date +%s)}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$((START_EPOCH + TIME_BUDGET_SECONDS))}"
MIN_START_SECONDS="${MIN_START_SECONDS:-1200}"
SMOKE="${SMOKE:-0}"
DO_RENDER="${DO_RENDER:-1}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-8}"
QUEUE_LAUNCH_STAGGER_SECONDS="${QUEUE_LAUNCH_STAGGER_SECONDS:-45}"

export OMP_NUM_THREADS="$CPU_THREADS_PER_JOB"
export MKL_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS_PER_JOB"
export BLIS_NUM_THREADS="$CPU_THREADS_PER_JOB"
export OPENCV_FOR_THREADS_NUM="$CPU_THREADS_PER_JOB"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
PARSER_ROOT="${PARSER_ROOT:-$ROOT/data/parsers_from_hulk_multiview}"
COMPACT_MAPPING="${COMPACT_MAPPING:-$ROOT/configs/semantic/hulk_cihp_compact_6.json}"
V215C_EXP="${V215C_EXP:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt}"
V215C_CKPT="${V215C_CKPT:-$V215C_EXP/ckpt109410.pth}"
V215C_RENDER="${V215C_RENDER:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_resume_v215c_v214v350_gate_only_boundary_20260510_095626_bjt_render_quick_ckpt109410}"
V222A_EXP="${V222A_EXP:-$ROOT/exp/stageB/377_hulk_light_v222a_stageB_hulk_light_all20_compact_stageB_hulk_light_20260510_221812_bjt}"
V222A_CKPT="${V222A_CKPT:-$V222A_EXP/best_ckpt.pth}"
V222B_EXP="${V222B_EXP:-$ROOT/exp/stageB/377_hulk_light_v222b_stageB_hulk_light_core12_parent_guard_stageB_hulk_light_20260510_221812_bjt}"
V222B_CKPT="${V222B_CKPT:-$V222B_EXP/best_ckpt.pth}"

LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_stageB_overnight_fill_clarity_4gpu_$RUN_ID}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/summary.tsv"
ASSET_SUMMARY="$LOG_DIR/asset_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
PIDS="$LOG_DIR/pids.tsv"
STATUS_JSON="$LOG_DIR/status.json"
mkdir -p "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$V215C_EXP/.hydra/config.yaml" "$V215C_CKPT" "$V222A_EXP/.hydra/config.yaml" "$V222A_CKPT" "$V222B_EXP/.hydra/config.yaml" "$V222B_CKPT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done
for cam in $(seq 1 20); do
  parser_dir="$PARSER_ROOT/CoreView_377/mask_cihp/Camera_B${cam}"
  if [ ! -d "$parser_dir" ]; then
    echo "missing parser dir: $parser_dir" >&2
    exit 3
  fi
done

START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"
DEADLINE_BJT="$(TZ=Asia/Shanghai date -d "@$DEADLINE_EPOCH" '+%F %T BJT')"
ALL20="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]"
CORE12="[2,3,6,7,8,9,11,13,15,18,19,20]"
RELIABLE8="[2,3,6,8,11,13,19,20]"
RELIABLE5="[2,8,11,19,20]"
PARSER_ASSET_VIEWS="[2,8,11,19,20]"
MINI_VIEWS="[2,11]"
MINI_FRAMES="[240,421,60]"
HP_SELECT=(render_c21_f000240.png render_c21_f000300.png render_c22_f000240.png render_c23_f000300.png render_c23_f000420.png)
HP_SELECT_MINI=(render_c2_f000240.png render_c2_f000300.png render_c2_f000360.png render_c2_f000420.png render_c11_f000240.png render_c11_f000300.png render_c11_f000360.png render_c11_f000420.png)
HP_CROP=(150 35 650 430)
BINDING_MAPS="[layer,region,compact_semantic,body_prob,soft_prob,cloth_prob,semantic,temporal,thin]"

printf 'RUN_ID=%s\nSTART_BJT=%s\nDEADLINE_BJT=%s\nTIME_BUDGET_SECONDS=%s\nGPUS=%s\nCPU_THREADS_PER_JOB=%s\nQUEUE_LAUNCH_STAGGER_SECONDS=%s\nV215C_EXP=%s\nV215C_CKPT=%s\nV222A_EXP=%s\nV222A_CKPT=%s\nV222B_EXP=%s\nV222B_CKPT=%s\nDATA_ROOT=%s\nPARSER_ROOT=%s\nCOMPACT_MAPPING=%s\nSMOKE=%s\nDO_RENDER=%s\n' \
  "$RUN_ID" "$START_BJT" "$DEADLINE_BJT" "$TIME_BUDGET_SECONDS" "$GPUS" \
  "$CPU_THREADS_PER_JOB" "$QUEUE_LAUNCH_STAGGER_SECONDS" \
  "$V215C_EXP" "$V215C_CKPT" "$V222A_EXP" "$V222A_CKPT" "$V222B_EXP" "$V222B_CKPT" \
  "$DATA_ROOT" "$PARSER_ROOT" "$COMPACT_MAPPING" "$SMOKE" "$DO_RENDER" | tee "$LOG_DIR/run_info.txt"

cat >> "$LOG_DIR/run_info.txt" <<'EOF'

Purpose:
  1. Fill missing StageB plan artifacts: baseline interpretability, direct-parser fine assets,
     binding summaries, highpass/contour diagnostics, and comparison montages.
  2. Use the remaining overnight window for StageB-safe clarity probes. Geometry/pose/camera
     stay frozen in most branches; only semantic adapters plus small texture/high-frequency
     paths are opened. The footprint probe is the only branch that opens tiny opacity/scale.
EOF

printf 'time_bjt\tgpu\tname\tphase\tdetail\n' > "$EVENTS"
printf 'name\tgpu\tpid\n' > "$PIDS"
printf 'name\tkind\texp_dir\trender_exp\tpsnr\tssim\tlpips\tfg_l1\tboundary_l1\tedge_px\tfg_hp_ratio\tcrop_hp_ratio\tfg_hp_l1\tcrop_hp_l1\tasset_mode\traw_masks\tparser_preview\tgrouped_preview\tcompact_masks\tbinding_summary\tstatus\n' > "$SUMMARY"
printf 'name\tasset_root\tasset_mode\tnum_views\traw_masks\tparser_preview\tgrouped_preview\tpreview\tcompact_masks\tcoarse_masks\tmotions\tstatus\n' > "$ASSET_SUMMARY"

log_event() {
  local gpu="$1"
  local name="$2"
  local phase="$3"
  local detail="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "$(TZ=Asia/Shanghai date '+%F %T BJT')" "$gpu" "$name" "$phase" "$detail" | tee -a "$EVENTS"
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  echo $((DEADLINE_EPOCH - now))
}

write_status() {
  "$PYTHON_BIN" - "$STATUS_JSON" "$RUN_ID" "$START_EPOCH" "$DEADLINE_EPOCH" "$1" "$2" "$3" <<'PY'
import json, sys, time
from pathlib import Path
path, run_id, start, deadline, gpu, phase, detail = sys.argv[1:]
start = int(start); deadline = int(deadline); now = int(time.time())
Path(path).write_text(json.dumps({
    "run_id": run_id,
    "gpu": gpu,
    "phase": phase,
    "detail": detail,
    "start_epoch": start,
    "deadline_epoch": deadline,
    "now_epoch": now,
    "elapsed_seconds": now - start,
    "remaining_seconds": max(0, deadline - now),
}, indent=2), encoding="utf-8")
PY
}

split_csv() {
  local value="$1"
  local IFS=','
  read -ra _items <<< "$value"
  printf '%s\n' "${_items[@]}"
}

run_highpass_diag() {
  local name="$1"
  local gpu="$2"
  local render_exp="$3"
  local label="$4"
  shift 4
  local select_files=("$@")
  if [ "${#select_files[@]}" -eq 0 ]; then
    select_files=("${HP_SELECT[@]}")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_highpass_energy.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --select "${select_files[@]}" \
    --crop "${HP_CROP[@]}" > "$LOG_DIR/${name}_highpass_${label}.log" 2>&1
}

run_contour_diag() {
  local name="$1"
  local gpu="$2"
  local render_exp="$3"
  local label="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --band-width 7 \
    --topk 12 > "$LOG_DIR/${name}_contour_${label}.log" 2>&1
}

summarize_asset_root() {
  local name="$1"
  local asset_root="$2"
  local status="$3"
  "$PYTHON_BIN" - "$name" "$asset_root" "$status" "$ASSET_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
name, asset_root, status, out = sys.argv[1:5]
root = Path(asset_root)
def count(rel):
    p = root / rel
    return sum(1 for x in p.rglob("*") if x.is_file()) if p.exists() else 0
mode = ""
views = 0
if (root / "meta.json").exists():
    meta = json.loads((root / "meta.json").read_text())
    mode = str(meta.get("mask_export_mode", ""))
    views = int(meta.get("num_views", 0))
row = [
    name, str(root), mode, str(views),
    str(count("raw_masks")),
    str(count("parser_preview")),
    str(count("grouped_preview")),
    str(count("preview")),
    str(count("compact_head_masks")),
    str(count("coarse_masks")),
    str(count("motions")),
    status,
]
with open(out, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY
}

write_summary_row() {
  local name="$1"
  local kind="$2"
  local exp_dir="$3"
  local render_exp="$4"
  local status="$5"
  "$PYTHON_BIN" - "$name" "$kind" "$exp_dir" "$render_exp" "$status" "$SUMMARY" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
name, kind, exp_dir, render_exp, status, summary = sys.argv[1:7]
render = Path(render_exp) if render_exp else None
def fmt(v, digits=8):
    if v is None:
        return "nan"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "nan"
metrics = {}
if render and (render / "test-view" / "results.npz").exists():
    data = np.load(render / "test-view" / "results.npz")
    for key in ("psnr", "ssim", "lpips"):
        if key in data.files:
            metrics[key] = float(np.asarray(data[key]).mean())
contour = {}
highpass = {}
if render:
    if (render / "diagnostics" / "contour_summary.json").exists():
        contour = json.loads((render / "diagnostics" / "contour_summary.json").read_text())
    if (render / "diagnostics" / "highpass_summary.json").exists():
        highpass = json.loads((render / "diagnostics" / "highpass_summary.json").read_text())
asset_mode = ""
raw = parser = grouped = compact = "0"
if render:
    asset = render / "test-view" / "semantic_editable_assets"
    if (asset / "meta.json").exists():
        meta = json.loads((asset / "meta.json").read_text())
        asset_mode = str(meta.get("mask_export_mode", ""))
    def count(rel):
        p = asset / rel
        return str(sum(1 for x in p.rglob("*") if x.is_file())) if p.exists() else "0"
    raw = count("raw_masks")
    parser = count("parser_preview")
    grouped = count("grouped_preview")
    compact = count("compact_head_masks")
binding_summary = "0"
if render and (render / "test-view" / "binding_maps" / "summary.json").exists():
    binding_summary = "1"
row = [
    name, kind, exp_dir, str(render) if render else "",
    fmt(metrics.get("psnr"), 6), fmt(metrics.get("ssim")), fmt(metrics.get("lpips")),
    fmt(contour.get("mean_fg_l1"), 6),
    fmt(contour.get("mean_boundary_l1"), 6),
    fmt(contour.get("mean_edge_symmetric_dist_px"), 4),
    fmt(highpass.get("fg_hp_ratio_mean"), 4),
    fmt(highpass.get("crop_hp_ratio_mean"), 4),
    fmt(highpass.get("fg_hp_l1_mean"), 5),
    fmt(highpass.get("crop_hp_l1_mean"), 5),
    asset_mode, raw, parser, grouped, compact, binding_summary, status,
]
with open(summary, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY
}

render_export() {
  local name="$1"
  local gpu="$2"
  local config_exp="$3"
  local ckpt="$4"
  local out_exp="$5"
  local extra_kind="$6"
  shift 6
  local extra_overrides=("$@")
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_${extra_kind}"
  log_event "$gpu" "$name" "render_export_start" "$out_exp"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" render.py \
    --config-path "$config_exp/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$out_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "export_interpretability=true" \
    "++binding_map_names=$BINDING_MAPS" \
    "export_semantic_editable_assets=true" \
    "semantic_editable_parser_root=$PARSER_ROOT" \
    "semantic_editable_parser_layout=cihp_subject" \
    "semantic_editable_direct_parser_mode=true" \
    "semantic_editable_export_compact_head=true" \
    "semantic_editable_include_binding_summary=true" \
    "+semantic_editable_preview_min_area=18" \
    "hydra.run.dir=$hydra_run_dir" \
    wandb_disable=true \
    "${extra_overrides[@]}" > "$LOG_DIR/${name}_${extra_kind}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    write_summary_row "$name" "$extra_kind" "$config_exp" "$out_exp" "render_failed"
    log_event "$gpu" "$name" "render_export_failed" "status=$status"
    return "$status"
  fi
  run_contour_diag "$name" "$gpu" "$out_exp" "$extra_kind" || true
  run_highpass_diag "$name" "$gpu" "$out_exp" "$extra_kind" || true
  "$PYTHON_BIN" tools/summarize_binding_interpretability.py \
    --exp-dir "$out_exp" \
    --split test-view \
    --copy-assets > "$LOG_DIR/${name}_binding_summary_${extra_kind}.log" 2>&1 || true
  "$PYTHON_BIN" tools/make_binding_paper_montage.py \
    --exp-dir "$out_exp" \
    --gt-root "$DATA_ROOT/CoreView_377" \
    --split test-view \
    --panels gt render layer region body_prob cloth_prob compact_semantic thin semantic \
    --select "${HP_SELECT[@]}" \
    --output-dir "$out_exp/test-view/paper_montages_selected" > "$LOG_DIR/${name}_binding_montage_${extra_kind}.log" 2>&1 || true
  summarize_asset_root "$name" "$out_exp/test-view/semantic_editable_assets" "ok"
  write_summary_row "$name" "$extra_kind" "$config_exp" "$out_exp" "ok"
  log_event "$gpu" "$name" "render_export_done" "$out_exp"
}

train_stageB_semantic() {
  local name="$1"
  local gpu="$2"
  local train_views="$3"
  local iterations="$4"
  local checkpoint_list="$5"
  local exp_dir="$ROOT/exp/stageB/377_hulk_light_${name}_${RUN_ID}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  shift 5
  local overrides=("$@")
  if [ "$SMOKE" = "1" ]; then
    iterations=2
    checkpoint_list="[2]"
  fi
  mkdir -p "$exp_dir"
  log_event "$gpu" "$name" "train_start" "iterations=$iterations views=$train_views"
  write_status "$gpu" "train_start" "$name"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" train.py \
    --config-path "$V215C_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$train_views" \
    "dataset.val_views=[21,22,23]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.val_frames=[0,570,60]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "dataset.parsing_prior.skip_empty_samples=false" \
    "dataset.parsing_prior.skip_empty_min_pixels=96" \
    "start_checkpoint=$V215C_CKPT" \
    "exp_dir=$exp_dir" \
    "seed=$SEED" \
    "wandb_disable=true" \
    "hydra.run.dir=$hydra_run_dir" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.pose_body_train_mask]" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "pipeline.pose_noise=0.0" \
    "model.gaussian.delay=0" \
    "++model.gaussian.semantic_logits_adapter_enable=true" \
    "++model.gaussian.semantic_logits_adapter_compact_classes=6" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$iterations" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.feature_lr=0.0" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.pose_correction_lr=0.0" \
    "opt.tex_latent_lr=0.0" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.semantic_region_logits_lr=0.006" \
    "++opt.semantic_compact_logits_lr=0.006" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0005" \
    "++opt.stageB_semantic_loss_enable=true" \
    "++opt.stageB_semantic_ignore_uncertain=true" \
    "++opt.stageB_semantic_ignore_boundary_width=7" \
    "++opt.stageB_semantic_use_opacity_support=true" \
    "++opt.stageB_semantic_opacity_threshold=0.04" \
    "++opt.stageB_semantic_min_valid_pixels=96" \
    "++opt.stageB_semantic_body_cloth_bce_weight=1.0" \
    "++opt.stageB_semantic_body_cloth_dice_weight=0.75" \
    "++opt.stageB_semantic_compact_bce_weight=1.0" \
    "++opt.stageB_semantic_compact_dice_weight=0.75" \
    "++opt.stageB_semantic_parent_consistency_enable=true" \
    "++opt.stageB_semantic_adapter_smooth_weight=0.012" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.015" \
    "++opt.train_sample_camera_max_prob=0.105" \
    "++opt.train_sample_log_interval=100" \
    "opt.lambda_mask=0.0" \
    "++opt.lambda_mask_boundary=0.0" \
    "++opt.lambda_mask_boundary_hard=0.0" \
    "++opt.lambda_silhouette_outer=0.0" \
    "++opt.lambda_silhouette_inner=0.0" \
    "++opt.lambda_boundary_opacity_residual_reg=0.0" \
    "++opt.lambda_boundary_scaling_residual_reg=0.0" \
    "++opt.lambda_boundary_opacity_residual_smooth=0.0" \
    "++opt.lambda_boundary_scaling_residual_smooth=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "opt.percent_dense=0.0" \
    "opt.densify_until_iter=0" \
    "opt.densify_from_iter=1000000" \
    "opt.opacity_reset_interval=1000000" \
    "test_interval=400" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "${overrides[@]}" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    write_summary_row "$name" "train" "$exp_dir" "" "train_failed"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi
  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"
  if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
    render_export "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "${exp_dir}_render_full_best" "best"
  fi
}

train_clarity_probe() {
  local name="$1"
  local gpu="$2"
  local train_views="$3"
  local iterations="$4"
  local texture_lr="$5"
  local checkpoint_list="$6"
  local exp_dir="$ROOT/exp/stageB/377_stageB_clarity_${name}_${RUN_ID}"
  local hydra_run_dir="$HYDRA_RUN_ROOT/${name}_train"
  shift 6
  local overrides=("$@")
  if [ "$SMOKE" = "1" ]; then
    iterations=2
    checkpoint_list="[2]"
  fi
  mkdir -p "$exp_dir"
  log_event "$gpu" "$name" "train_start" "iterations=$iterations texture_lr=$texture_lr views=$train_views"
  write_status "$gpu" "train_start" "$name"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" train.py \
    --config-path "$V215C_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=$train_views" \
    "dataset.val_views=[21,22,23]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.val_frames=[0,570,60]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=true" \
    "dataset.parsing_prior.roi_enable=true" \
    "dataset.parsing_prior.parser_root=$PARSER_ROOT" \
    "dataset.parsing_prior.parser_layout=cihp_subject" \
    "dataset.parsing_prior.use_direct_parser_labels=true" \
    "dataset.parsing_prior.compact_mapping_file=$COMPACT_MAPPING" \
    "dataset.parsing_prior.skip_empty_samples=false" \
    "dataset.parsing_prior.skip_empty_min_pixels=96" \
    "start_checkpoint=$V215C_CKPT" \
    "exp_dir=$exp_dir" \
    "seed=$SEED" \
    "wandb_disable=true" \
    "hydra.run.dir=$hydra_run_dir" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.,pose_correction.pose_body_train_mask]" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "pipeline.pose_noise=0.0" \
    "model.gaussian.delay=0" \
    "++model.gaussian.semantic_logits_adapter_enable=true" \
    "++model.gaussian.semantic_logits_adapter_compact_classes=6" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$iterations" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.feature_lr=0.0" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.pose_correction_lr=0.0" \
    "opt.tex_latent_lr=0.0" \
    "opt.texture_lr=$texture_lr" \
    "++opt.texture_trainable_name_patterns=[detail_high_freq_context_proj.*,detail_high_freq_carrier_proj.*,detail_high_freq_structure_proj.*,detail_high_freq_mlp.*,detail_high_freq_gate_mlp.*,detail_high_freq_luma_mlp.*,detail_high_freq_face_mlp.*,detail_high_freq_face_gate_mlp.*,detail_high_freq_face_local_proj.*,detail_high_freq_face_extra_local_projs.*,structured_trunk_output_head_hf_head_mlp.*,structured_trunk_output_head_hf_head_gate_mlp.*,structured_trunk_output_head_local_color_mlps.*,structured_trunk_output_head_local_color_owner_head_mlps.*,structured_trunk_output_head_local_color_owner_head_gate_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_mlps.*,structured_trunk_output_head_local_color_owner_head_boundary_gate_mlps.*,structured_trunk_output_head_local_fusion_projs.*,structured_trunk_output_head_local_geometry_fusion_projs.*]" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.semantic_region_logits_lr=0.0035" \
    "++opt.semantic_compact_logits_lr=0.0035" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0008" \
    "++opt.stageB_semantic_loss_enable=true" \
    "++opt.stageB_semantic_ignore_uncertain=true" \
    "++opt.stageB_semantic_ignore_boundary_width=9" \
    "++opt.stageB_semantic_use_opacity_support=true" \
    "++opt.stageB_semantic_opacity_threshold=0.04" \
    "++opt.stageB_semantic_min_valid_pixels=96" \
    "++opt.stageB_semantic_body_cloth_weight=0.42" \
    "++opt.stageB_semantic_compact_weight=0.45" \
    "++opt.stageB_semantic_parent_consistency_weight=0.24" \
    "++opt.stageB_semantic_exclusive_weight=0.10" \
    "++opt.stageB_semantic_body_cloth_bce_weight=1.0" \
    "++opt.stageB_semantic_body_cloth_dice_weight=0.75" \
    "++opt.stageB_semantic_compact_bce_weight=1.0" \
    "++opt.stageB_semantic_compact_dice_weight=0.75" \
    "++opt.stageB_semantic_parent_consistency_enable=true" \
    "++opt.stageB_semantic_adapter_smooth_weight=0.010" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.015" \
    "++opt.train_sample_camera_max_prob=0.105" \
    "++opt.train_sample_log_interval=100" \
    "++opt.clarity_debug_enable=true" \
    "++opt.clarity_debug_interval=150" \
    "++opt.clarity_debug_warmup_iters=0" \
    "++opt.face_region_source=parser_only" \
    "++opt.face_region_parser_dilate=1" \
    "++opt.shoulder_arm_region_source=parser_only" \
    "++opt.shoulder_arm_region_parser_dilate=1" \
    "++opt.upper_torso_region_source=parser_only" \
    "++opt.upper_torso_region_parser_dilate=1" \
    "++opt.waist_region_source=parser_only" \
    "++opt.waist_region_mode=lower_cloth" \
    "++opt.waist_region_parser_dilate=1" \
    "++opt.detail_interior_erode_kernel_size=1" \
    "++opt.detail_interior_exclude_boundary_width=18" \
    "++opt.face_detail_interior_exclude_boundary_width=10" \
    "++opt.shoulder_arm_detail_interior_exclude_boundary_width=18" \
    "++opt.upper_torso_detail_interior_exclude_boundary_width=18" \
    "++opt.waist_detail_interior_exclude_boundary_width=18" \
    "++opt.owner_local_detail_boost_enable=true" \
    "++opt.owner_local_detail_boost_warmup_iters=0" \
    "++opt.owner_local_detail_boost_takeover_floor=0.14" \
    "++opt.owner_local_detail_boost_takeover_gain=1.22" \
    "++opt.owner_local_detail_boost_takeover_power=1.12" \
    "++opt.owner_local_detail_boost_min_signal=0.001" \
    "++opt.owner_local_detail_boost_detail_max_extra=0.34" \
    "++opt.owner_local_detail_boost_luma_max_extra=0.46" \
    "++opt.owner_local_detail_boost_patch_max_extra=0.34" \
    "++opt.owner_local_detail_boost_edge_max_extra=0.14" \
    "++model.texture.detail_residual.high_frequency.enable=true" \
    "++model.texture.detail_residual.high_frequency.tiny_repair_scale=0.020" \
    "++model.texture.detail_residual.high_frequency.scale=0.055" \
    "++model.texture.detail_residual.high_frequency.gate_bias=-0.05" \
    "++model.texture.detail_residual.high_frequency.min_gate=0.04" \
    "++model.texture.detail_residual.high_frequency.face_branch.enable=true" \
    "++model.texture.detail_residual.high_frequency.face_branch.tiny_repair_scale=0.012" \
    "++model.texture.detail_residual.high_frequency.face_branch.scale=0.180" \
    "++model.texture.detail_residual.high_frequency.face_branch.gate_bias=-0.10" \
    "++model.texture.detail_residual.high_frequency.face_branch.min_gate=0.07" \
    "++model.texture.structured_trunk.output_head.local_color.owner.enable=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.enable=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.scale=0.55" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_local_color_input=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_local_color_output=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_local_geometry_raw=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.use_support_feature=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.init_from_local_color=true" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.init_scale=0.55" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.gate_gain=1.15" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.gate_bias=-0.04" \
    "++model.texture.structured_trunk.output_head.local_color.owner.head.min_gate=0.07" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=0.38" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.gate_gain=1.12" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.min_gate=0.05" \
    "++model.texture.structured_trunk.output_head.dual_head.hf_head.max_residual=0.16" \
    "++opt.lambda_l1=0.014" \
    "opt.lambda_l1_fg=0.095" \
    "opt.lambda_l1_boundary=0.210" \
    "opt.lambda_l1_face=0.040" \
    "opt.lambda_l1_shoulder_arm=0.032" \
    "opt.lambda_l1_waist=0.030" \
    "opt.lambda_perceptual=0.044" \
    "opt.lambda_edge_face=0.012" \
    "opt.lambda_edge_shoulder_arm=0.014" \
    "opt.lambda_edge_waist=0.006" \
    "++opt.lambda_detail_face_luma_dog=0.018" \
    "++opt.lambda_detail_shoulder_arm_luma_dog=0.016" \
    "++opt.lambda_detail_upper_torso_luma_dog=0.018" \
    "++opt.lambda_detail_upper_torso_core_luma_dog=0.024" \
    "++opt.lambda_detail_waist_luma_dog=0.010" \
    "++opt.lambda_perceptual_face_patch=0.010" \
    "++opt.lambda_perceptual_shoulder_arm_patch=0.008" \
    "++opt.lambda_perceptual_upper_torso_patch=0.008" \
    "++opt.lambda_perceptual_upper_torso_core_patch=0.010" \
    "++opt.lambda_perceptual_waist_patch=0.005" \
    "opt.lambda_mask=0.0" \
    "opt.lambda_skinning=0.0" \
    "opt.lambda_aiap_xyz=0.0" \
    "opt.lambda_aiap_cov=0.0" \
    "opt.percent_dense=0.0" \
    "opt.densify_until_iter=0" \
    "opt.densify_from_iter=1000000" \
    "opt.opacity_reset_interval=1000000" \
    "test_interval=600" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=0.0040" \
    "${overrides[@]}" > "$LOG_DIR/${name}.log" 2>&1
  local status=$?
  if [ "$status" -ne 0 ]; then
    write_summary_row "$name" "train" "$exp_dir" "" "train_failed"
    log_event "$gpu" "$name" "train_failed" "status=$status"
    return "$status"
  fi
  log_event "$gpu" "$name" "train_done" "$LOG_DIR/${name}.log"
  if [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
    render_export "$name" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "${exp_dir}_render_full_best" "best"
  fi
}

run_named_job() {
  local gpu="$1"
  local job="$2"
  case "$job" in
    export_v215c_baseline_full)
      render_export "$job" "$gpu" "$V215C_EXP" "$V215C_CKPT" "$ROOT/exp/stageA2/377_v215c_ckpt109410_interp_semantic_full_${RUN_ID}" "baseline"
      ;;
    export_v215c_parser_assets)
      render_export "$job" "$gpu" "$V215C_EXP" "$V215C_CKPT" "$ROOT/exp/stageA2/377_v215c_ckpt109410_parser_assets_${RUN_ID}" "parser_assets" \
        "dataset.test_views.view=$PARSER_ASSET_VIEWS" \
        "dataset.test_frames.view=[0,570,60]"
      ;;
    export_v222a_full_assets)
      render_export "$job" "$gpu" "$V222A_EXP" "$V222A_CKPT" "${V222A_EXP}_render_full_assets_${RUN_ID}" "full_assets"
      ;;
    export_v222a_parser_assets)
      render_export "$job" "$gpu" "$V222A_EXP" "$V222A_CKPT" "${V222A_EXP}_render_parser_assets_${RUN_ID}" "parser_assets" \
        "dataset.test_views.view=$PARSER_ASSET_VIEWS" \
        "dataset.test_frames.view=[0,570,60]"
      ;;
    export_v222b_full_assets)
      render_export "$job" "$gpu" "$V222B_EXP" "$V222B_CKPT" "${V222B_EXP}_render_full_assets_${RUN_ID}" "full_assets"
      ;;
    export_v222b_parser_assets)
      render_export "$job" "$gpu" "$V222B_EXP" "$V222B_CKPT" "${V222B_EXP}_render_parser_assets_${RUN_ID}" "parser_assets" \
        "dataset.test_views.view=$PARSER_ASSET_VIEWS" \
        "dataset.test_frames.view=[0,570,60]"
      ;;
    v223a_semantic_all20_long)
      train_stageB_semantic "$job" "$gpu" "$ALL20" 15000 "[3750,7500,11250,15000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.35" \
        "++opt.stageB_semantic_body_cloth_weight=0.58" \
        "++opt.stageB_semantic_compact_weight=0.82" \
        "++opt.stageB_semantic_parent_consistency_weight=0.34" \
        "++opt.stageB_semantic_exclusive_weight=0.12" \
        "opt.lambda_l1=0.0" \
        "opt.lambda_l1_fg=0.0" \
        "opt.lambda_l1_boundary=0.0" \
        "opt.lambda_l1_face=0.0" \
        "opt.lambda_l1_shoulder_arm=0.0" \
        "opt.lambda_l1_waist=0.0" \
        "opt.lambda_perceptual=0.0" \
        "opt.texture_lr=0.0" \
        "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
        "opt.grad_clip=0.01"
      ;;
    v223b_semantic_core12_parent)
      train_stageB_semantic "$job" "$gpu" "$CORE12" 15000 "[3750,7500,11250,15000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.55" \
        "++opt.stageB_semantic_ignore_boundary_width=9" \
        "++opt.stageB_semantic_body_cloth_weight=0.68" \
        "++opt.stageB_semantic_compact_weight=0.76" \
        "++opt.stageB_semantic_parent_consistency_weight=0.68" \
        "++opt.stageB_semantic_exclusive_weight=0.24" \
        "opt.lambda_l1=0.0" \
        "opt.lambda_l1_fg=0.0" \
        "opt.lambda_l1_boundary=0.0" \
        "opt.lambda_l1_face=0.0" \
        "opt.lambda_l1_shoulder_arm=0.0" \
        "opt.lambda_l1_waist=0.0" \
        "opt.lambda_perceptual=0.0" \
        "opt.texture_lr=0.0" \
        "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
        "opt.grad_clip=0.01"
      ;;
    v223c_clarity_reliable8_semantic_guard)
      train_clarity_probe "$job" "$gpu" "$RELIABLE8" 12000 7.5e-07 "[3000,6000,9000,12000]" \
        "++opt.stageB_semantic_compact_weight=0.52" \
        "++opt.stageB_semantic_parent_consistency_weight=0.30" \
        "opt.lambda_l1_boundary=0.220" \
        "opt.lambda_perceptual=0.048" \
        "opt.grad_clip=0.0042"
      ;;
    v223d_clarity_all20_semantic_guard)
      train_clarity_probe "$job" "$gpu" "$ALL20" 12000 6.0e-07 "[3000,6000,9000,12000]" \
        "++opt.stageB_semantic_body_cloth_weight=0.46" \
        "++opt.stageB_semantic_compact_weight=0.55" \
        "++opt.stageB_semantic_parent_consistency_weight=0.32" \
        "opt.lambda_l1_boundary=0.225" \
        "opt.lambda_perceptual=0.050" \
        "opt.grad_clip=0.0038"
      ;;
    v223e_clarity_mini_overfit_capacity)
      train_clarity_probe "$job" "$gpu" "$MINI_VIEWS" 9000 1.6e-06 "[2250,4500,6750,9000]" \
        "++opt.train_sample_mode=random" \
        "++opt.stageB_semantic_body_cloth_weight=0.35" \
        "++opt.stageB_semantic_compact_weight=0.38" \
        "opt.lambda_l1_boundary=0.235" \
        "opt.lambda_perceptual=0.040" \
        "++opt.lambda_detail_face_luma_dog=0.026" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.034" \
        "opt.grad_clip=0.0060"
      local status=$?
      if [ "$status" -eq 0 ] && [ "$DO_RENDER" = "1" ] && [ "$SMOKE" != "1" ]; then
        local exp_dir="$ROOT/exp/stageB/377_stageB_clarity_${job}_${RUN_ID}"
        local render_exp="${exp_dir}_render_trainprobe_best"
        render_export "$job" "$gpu" "$exp_dir" "$exp_dir/best_ckpt.pth" "$render_exp" "trainprobe" \
          "dataset.test_views.view=$MINI_VIEWS" \
          "dataset.test_frames.view=$MINI_FRAMES"
        run_highpass_diag "$job" "$gpu" "$render_exp" "trainprobe" "${HP_SELECT_MINI[@]}" || true
      fi
      return "$status"
      ;;
    v223f_footprint_tiny_scale_probe)
      train_clarity_probe "$job" "$gpu" "$RELIABLE5" 9000 5.5e-07 "[2250,4500,6750,9000]" \
        "opt.opacity_lr=0.00018" \
        "opt.scaling_lr=0.000020" \
        "++opt.boundary_opacity_residual_lr=0.000015" \
        "++opt.boundary_scaling_residual_lr=0.000006" \
        "opt.lambda_l1_boundary=0.245" \
        "opt.lambda_perceptual=0.040" \
        "opt.grad_clip=0.0030"
      ;;
    v223g_clarity_reliable8_highpass_push)
      train_clarity_probe "$job" "$gpu" "$RELIABLE8" 12000 1.05e-06 "[3000,6000,9000,12000]" \
        "++opt.stageB_semantic_body_cloth_weight=0.36" \
        "++opt.stageB_semantic_compact_weight=0.42" \
        "++opt.stageB_semantic_parent_consistency_weight=0.22" \
        "opt.lambda_l1_boundary=0.235" \
        "opt.lambda_perceptual=0.040" \
        "++opt.lambda_detail_face_luma_dog=0.030" \
        "++opt.lambda_detail_shoulder_arm_luma_dog=0.026" \
        "++opt.lambda_detail_upper_torso_luma_dog=0.030" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.040" \
        "++opt.lambda_perceptual_face_patch=0.016" \
        "++opt.lambda_perceptual_upper_torso_core_patch=0.016" \
        "++model.texture.detail_residual.high_frequency.scale=0.072" \
        "++model.texture.detail_residual.high_frequency.face_branch.scale=0.220" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=0.48" \
        "opt.grad_clip=0.0048"
      ;;
    v223h_clarity_all20_lowlr_rgb_guard)
      train_clarity_probe "$job" "$gpu" "$ALL20" 12000 3.8e-07 "[3000,6000,9000,12000]" \
        "++opt.stageB_semantic_body_cloth_weight=0.52" \
        "++opt.stageB_semantic_compact_weight=0.58" \
        "++opt.stageB_semantic_parent_consistency_weight=0.36" \
        "opt.lambda_l1_fg=0.112" \
        "opt.lambda_l1_boundary=0.240" \
        "opt.lambda_perceptual=0.055" \
        "++opt.lambda_detail_face_luma_dog=0.014" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.018" \
        "++model.texture.detail_residual.high_frequency.scale=0.040" \
        "++model.texture.detail_residual.high_frequency.face_branch.scale=0.145" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=0.30" \
        "opt.grad_clip=0.0032"
      ;;
    v223i_semantic_all20_lowdelta_smooth)
      train_stageB_semantic "$job" "$gpu" "$ALL20" 15000 "[3750,7500,11250,15000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=0.95" \
        "++opt.semantic_region_logits_lr=0.0045" \
        "++opt.semantic_compact_logits_lr=0.0045" \
        "++opt.lambda_binding_semantic_adapter_reg=0.0012" \
        "++opt.stageB_semantic_body_cloth_weight=0.60" \
        "++opt.stageB_semantic_compact_weight=0.88" \
        "++opt.stageB_semantic_parent_consistency_weight=0.40" \
        "++opt.stageB_semantic_exclusive_weight=0.12" \
        "++opt.stageB_semantic_adapter_smooth_weight=0.020" \
        "opt.lambda_l1=0.0" \
        "opt.lambda_l1_fg=0.0" \
        "opt.lambda_l1_boundary=0.0" \
        "opt.lambda_l1_face=0.0" \
        "opt.lambda_l1_shoulder_arm=0.0" \
        "opt.lambda_l1_waist=0.0" \
        "opt.lambda_perceptual=0.0" \
        "opt.texture_lr=0.0" \
        "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
        "opt.grad_clip=0.008"
      ;;
    v223j_semantic_reliable5_parent_guard)
      train_stageB_semantic "$job" "$gpu" "$RELIABLE5" 15000 "[3750,7500,11250,15000]" \
        "++model.gaussian.semantic_logits_adapter_max_delta=1.60" \
        "++opt.stageB_semantic_ignore_boundary_width=11" \
        "++opt.stageB_semantic_body_cloth_weight=0.72" \
        "++opt.stageB_semantic_compact_weight=0.70" \
        "++opt.stageB_semantic_parent_consistency_weight=0.76" \
        "++opt.stageB_semantic_exclusive_weight=0.28" \
        "opt.lambda_l1=0.0" \
        "opt.lambda_l1_fg=0.0" \
        "opt.lambda_l1_boundary=0.0" \
        "opt.lambda_l1_face=0.0" \
        "opt.lambda_l1_shoulder_arm=0.0" \
        "opt.lambda_l1_waist=0.0" \
        "opt.lambda_perceptual=0.0" \
        "opt.texture_lr=0.0" \
        "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
        "opt.grad_clip=0.010"
      ;;
    v223k_clarity_reliable5_texture_capacity)
      train_clarity_probe "$job" "$gpu" "$RELIABLE5" 9000 1.25e-06 "[2250,4500,6750,9000]" \
        "++opt.stageB_semantic_body_cloth_weight=0.32" \
        "++opt.stageB_semantic_compact_weight=0.36" \
        "++opt.stageB_semantic_parent_consistency_weight=0.18" \
        "opt.lambda_l1_boundary=0.245" \
        "opt.lambda_perceptual=0.038" \
        "++opt.lambda_detail_face_luma_dog=0.034" \
        "++opt.lambda_detail_upper_torso_core_luma_dog=0.046" \
        "++model.texture.detail_residual.high_frequency.scale=0.080" \
        "++model.texture.detail_residual.high_frequency.face_branch.scale=0.250" \
        "++model.texture.structured_trunk.output_head.dual_head.hf_head.scale=0.55" \
        "opt.grad_clip=0.0050"
      ;;
    *)
      log_event "$gpu" "$job" "unknown_job" "skipped"
      return 2
      ;;
  esac
}

queue_gpu() {
  local gpu="$1"
  shift
  local jobs=("$@")
  local launched=0
  for job in "${jobs[@]}"; do
    local remain
    remain="$(remaining_seconds)"
    if [ "$SMOKE" != "1" ] && [ "$remain" -le "$MIN_START_SECONDS" ]; then
      log_event "$gpu" "$job" "deadline_skip" "remaining=${remain}s"
      continue
    fi
    run_named_job "$gpu" "$job"
    local status=$?
    launched=$((launched + 1))
    if [ "$status" -eq 0 ]; then
      log_event "$gpu" "$job" "job_done" "remaining=$(remaining_seconds)s"
    else
      log_event "$gpu" "$job" "job_failed" "status=$status remaining=$(remaining_seconds)s"
    fi
  done
  write_status "$gpu" "queue_done" "launched=$launched"
  log_event "$gpu" "queue" "done" "launched=$launched remaining=$(remaining_seconds)s"
}

launch_queue() {
  local gpu="$1"
  local delay="${2:-0}"
  shift 2
  local jobs=("$@")
  (
    if [ "$delay" -gt 0 ]; then
      log_event "$gpu" "queue" "launch_delay" "${delay}s"
      sleep "$delay"
    fi
    queue_gpu "$gpu" "${jobs[@]}"
  ) &
  local pid=$!
  printf 'gpu%s_queue\t%s\t%s\n' "$gpu" "$gpu" "$pid" >> "$PIDS"
}

build_compare_panels() {
  if [ "$DO_RENDER" != "1" ] || [ "$SMOKE" = "1" ]; then
    return 0
  fi
  "$PYTHON_BIN" - "$LOG_DIR" "$SUMMARY" "$DATA_ROOT" <<'PY' > "$LOG_DIR/build_compare_panels.log" 2>&1
import csv, subprocess, sys
from pathlib import Path
log_dir = Path(sys.argv[1])
summary = Path(sys.argv[2])
data_root = Path(sys.argv[3])
rows = list(csv.DictReader(summary.open(), delimiter="\t"))
render_rows = [r for r in rows if r.get("status") == "ok" and r.get("render_exp") and (Path(r["render_exp"]) / "test-view" / "renders").exists()]
priority = []
for key in ("export_v215c_baseline_full", "export_v222a_full_assets", "v223a_semantic_all20_long", "v223c_clarity_reliable8_semantic_guard", "v223d_clarity_all20_semantic_guard", "v223f_footprint_tiny_scale_probe"):
    match = next((r for r in render_rows if r["name"] == key), None)
    if match:
        priority.append(match)
if len(priority) < 2:
    priority = render_rows[:6]
if len(priority) < 2:
    raise SystemExit(0)
render_exps = [r["render_exp"] for r in priority]
labels = [r["name"].replace("export_", "").replace("_stageB_", "_").replace("_semantic_", "_sem_").replace("_clarity_", "_clr_")[:22] for r in priority]
out = log_dir / "compare_panels" / "main"
cmd = [
    sys.executable, "tools/make_377_render_comparison_montage.py",
    "--render-exp", *render_exps,
    "--labels", *labels,
    "--gt-root", str(data_root / "CoreView_377"),
    "--output-dir", str(out),
    "--select", "render_c21_f000240.png", "render_c21_f000300.png", "render_c22_f000240.png", "render_c23_f000300.png", "render_c23_f000420.png",
    "--crop", "150", "35", "650", "430",
    "--panel-width", "210",
    "--stack",
]
subprocess.run(cmd, check=False)
PY
}

mapfile -t SELECTED_GPUS < <(split_csv "$GPUS")
if [ "${#SELECTED_GPUS[@]}" -lt 4 ]; then
  echo "expected four GPUs in GPUS, got: $GPUS" >&2
  exit 4
fi

GPU0="${SELECTED_GPUS[0]}"
GPU1="${SELECTED_GPUS[1]}"
GPU2="${SELECTED_GPUS[2]}"
GPU3="${SELECTED_GPUS[3]}"

launch_queue "$GPU0" 0 \
  v223a_semantic_all20_long \
  v223e_clarity_mini_overfit_capacity \
  v223i_semantic_all20_lowdelta_smooth \
  export_v215c_baseline_full \
  export_v215c_parser_assets
launch_queue "$GPU1" "$QUEUE_LAUNCH_STAGGER_SECONDS" \
  v223c_clarity_reliable8_semantic_guard \
  v223g_clarity_reliable8_highpass_push \
  export_v222a_full_assets \
  export_v222a_parser_assets
launch_queue "$GPU2" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 2))" \
  v223b_semantic_core12_parent \
  v223j_semantic_reliable5_parent_guard \
  export_v222b_full_assets \
  export_v222b_parser_assets
launch_queue "$GPU3" "$((QUEUE_LAUNCH_STAGGER_SECONDS * 3))" \
  v223d_clarity_all20_semantic_guard \
  v223f_footprint_tiny_scale_probe \
  v223h_clarity_all20_lowlr_rgb_guard \
  v223k_clarity_reliable5_texture_capacity

echo "RUN_ID=$RUN_ID"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "ASSET_SUMMARY=$ASSET_SUMMARY"
echo "DEADLINE_BJT=$DEADLINE_BJT"
cat "$PIDS"

wait
build_compare_panels
log_event "all" "queue" "all_done" "summary=$SUMMARY"
echo "SUMMARY=$SUMMARY"
