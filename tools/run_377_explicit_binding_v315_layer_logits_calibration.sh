#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v315_layer_logits_calibration_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v304_consistent_component_audit_v304_consistent_component_geometry_20260519_100431_bjt_audit_all_views_sparse/point_contributors_all.csv}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,1]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

ITERATIONS="${ITERATIONS:-300}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-100,200,300}"
LAYER_LOGITS_LR="${LAYER_LOGITS_LR:-0.0030}"
LAYER_LOGITS_MAX_DELTA="${LAYER_LOGITS_MAX_DELTA:-0.55}"
LAYER_LOGITS_BOUNDARY_MIN="${LAYER_LOGITS_BOUNDARY_MIN:-0.12}"
LAYER_LOGITS_REG="${LAYER_LOGITS_REG:-0.0030}"
DO_RENDER="${DO_RENDER:-1}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v315_layer_logits_calibration_${RUN_ID}}"
TRAIN_EXP="$EXP_ROOT/train_layer_logits"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v315_layer_logits_calibration_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$TRAIN_EXP" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

BASE_ITER="$("$PYTHON_BIN" - "$BASE_CKPT" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt[-1]))
PY
)"

START_EPOCH="$(date +%s)"
EST_SECONDS="${EST_SECONDS:-3000}"
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
BASE_ITER=$BASE_ITER
DATA_ROOT=$DATA_ROOT
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
ITERATIONS=$ITERATIONS
CHECKPOINT_STEPS=$CHECKPOINT_STEPS
LAYER_LOGITS_LR=$LAYER_LOGITS_LR
LAYER_LOGITS_MAX_DELTA=$LAYER_LOGITS_MAX_DELTA
LAYER_LOGITS_BOUNDARY_MIN=$LAYER_LOGITS_BOUNDARY_MIN
LAYER_LOGITS_REG=$LAYER_LOGITS_REG
EXP_ROOT=$EXP_ROOT
TRAIN_EXP=$TRAIN_EXP
LOG_DIR=$LOG_DIR

Goal:
  v315 calibrates the explicit-binding layer mixture itself. It freezes all
  existing Gaussian geometry/appearance, deformer, pose, camera, texture, and
  boundary residual parameters, then trains only a per-Gaussian [rigid, soft,
  free] layer-logit residual under RGB/fg/boundary losses. No support append,
  no mask-only silhouette loss, no component CSV supervision, no export refine.
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

COMMON_RENDER_ARGS=(
  "dataset.root_dir=$DATA_ROOT"
  "dataset.preload=false"
  "dataset.train_views=$TRAIN_VIEWS_SPEC"
  "dataset.train_frames=[0,570,60]"
  "dataset.test_views.view=$TEST_VIEWS_SPEC"
  "dataset.test_frames.view=$TEST_FRAMES_SPEC"
  "dataset.parsing_prior.enable=false"
  "dataset.parsing_prior.roi_enable=false"
  "export_interpretability=false"
  "export_semantic_editable_assets=false"
  "++export_opacity_maps=false"
  "++render_export_refine=false"
  "wandb_disable=true"
)

ADAPTER_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_dynamic_enable=false"
  "++pipeline.covariance_signed_screen_actuator_enable=false"
  "++pipeline.covariance_signed_center_offset_enable=false"
  "++pipeline.boundary_cov_residual_enable=false"
  "++pipeline.binding_covariance_guard_enable=false"
  "++model.deformer.rigid.rotation_orthogonalize_enable=false"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
  "++model.deformer.rigid.geometry_fidelity_center_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_component_enable=false"
  "++model.gaussian.binding_layer_logits_adapter_enable=true"
  "++model.gaussian.binding_layer_logits_adapter_max_delta=$LAYER_LOGITS_MAX_DELTA"
  "++model.gaussian.binding_layer_logits_adapter_boundary_min=$LAYER_LOGITS_BOUNDARY_MIN"
)

render_variant() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  local mode="$4"
  local config_dir="$5"
  local hydra_dir="$6"

  local extra_args=()
  if [ "$mode" = "v307_reference" ]; then
    if [ ! -e "$COMPONENT_CSV" ] || [ ! -e "$POINT_CSV" ]; then
      log_event "render_skip" "$variant missing v307 csv"
      return 0
    fi
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
  elif [ "$mode" = "adapter" ]; then
    extra_args=("${ADAPTER_ARGS[@]}")
  fi

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$config_dir" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "${COMMON_RENDER_ARGS[@]}" \
    "${extra_args[@]}" \
    "hydra.run.dir=$hydra_dir" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
}

analyze_variant() {
  local variant="$1"
  local render_exp="$2"
  if [ ! -d "$render_exp/test-view" ]; then
    return 0
  fi
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

run_render_eval() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  local mode="$4"
  local config_dir="$5"

  if [ ! -f "$ckpt" ]; then
    log_event "render_skip" "$variant missing=$ckpt"
    return 0
  fi
  log_event "render_start" "$variant"
  render_variant "$variant" "$ckpt" "$render_exp" "$mode" "$config_dir" "$HYDRA_RUN_ROOT/render_${variant}"
  log_event "analyze_start" "$variant"
  analyze_variant "$variant" "$render_exp"
  log_event "render_done" "$variant"
}

if [ "$DO_RENDER" = "1" ]; then
  run_render_eval "baseline_no_adapter" "$BASE_CKPT" "$EXP_ROOT/baseline_no_adapter" "none" "$BASE_EXP/.hydra"
  run_render_eval "v307_external_component_reference" "$BASE_CKPT" "$EXP_ROOT/v307_external_component_reference" "v307_reference" "$BASE_EXP/.hydra"
  run_render_eval "v315_init_zero_adapter" "$BASE_CKPT" "$EXP_ROOT/v315_init_zero_adapter" "adapter" "$BASE_EXP/.hydra"
fi

checkpoint_list="[$CHECKPOINT_STEPS]"
write_status "train_start" "$TRAIN_EXP"
log_event "train_start" "$TRAIN_EXP"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  mode=train \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.train_views=$TRAIN_VIEWS_SPEC" \
  "dataset.val_views=$TEST_VIEWS_SPEC" \
  "dataset.test_views.view=$TEST_VIEWS_SPEC" \
  "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
  "dataset.val_frames=$TEST_FRAMES_SPEC" \
  "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=false" \
  "dataset.parsing_prior.compact_mapping_file=" \
  "start_checkpoint=$BASE_CKPT" \
  "exp_dir=$TRAIN_EXP" \
  "hydra.run.dir=$HYDRA_RUN_ROOT/train" \
  "seed=-1" \
  "wandb_disable=true" \
  "++resume.allow_partial_converter_load=true" \
  "++resume.restore_gaussian_optimizer_state=false" \
  "++resume.restore_converter_optimizer_state=false" \
  "++resume.restore_converter_scheduler_state=false" \
  "++resume.partial_converter_missing_keys_allow_patterns=[deformer.rigid.forward_trunk_mlp.,camera_affine.,pose_correction.]" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.clear_boundary_tags_on_resume=true" \
  "++resume.clear_binding_state_on_resume=false" \
  "pipeline.pose_noise=0.0" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "${ADAPTER_ARGS[@]}" \
  "opt.iterations=$ITERATIONS" \
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
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "++opt.texture_trainable_name_patterns=[__freeze_texture_no_match__]" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.boundary_cov_residual_lr=0.0" \
  "++opt.binding_layer_logits_lr=$LAYER_LOGITS_LR" \
  "++opt.lambda_binding_layer_logits_adapter_reg=$LAYER_LOGITS_REG" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.stageB_semantic_body_cloth_weight=0.0" \
  "++opt.stageB_semantic_compact_weight=0.0" \
  "++opt.lambda_binding_semantic_adapter_reg=0.0" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.018" \
  "++opt.train_sample_camera_max_prob=0.125" \
  "++opt.train_sample_log_interval=100" \
  "++opt.train_sample_accumulation_steps=1" \
  "opt.lambda_l1=0.050" \
  "opt.lambda_l1_fg=0.140" \
  "opt.lambda_l1_boundary=0.120" \
  "opt.lambda_dssim=0.0" \
  "opt.lambda_perceptual=0.0" \
  "opt.lambda_l1_face=0.0" \
  "opt.lambda_l1_shoulder_arm=0.0" \
  "opt.lambda_l1_waist=0.0" \
  "opt.lambda_edge_face=0.0" \
  "opt.lambda_edge_shoulder_arm=0.0" \
  "opt.lambda_edge_waist=0.0" \
  "++opt.lambda_detail_face=0.0" \
  "++opt.lambda_detail_shoulder_arm=0.0" \
  "++opt.lambda_detail_waist=0.0" \
  "++opt.lambda_detail_face_luma_dog=0.0" \
  "++opt.lambda_detail_shoulder_arm_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_luma_dog=0.0" \
  "++opt.lambda_detail_upper_torso_core_luma_dog=0.0" \
  "++opt.lambda_detail_waist_luma_dog=0.0" \
  "++opt.lambda_perceptual_face=0.0" \
  "++opt.lambda_perceptual_shoulder_arm=0.0" \
  "++opt.lambda_perceptual_waist=0.0" \
  "++opt.lambda_perceptual_face_patch=0.0" \
  "++opt.lambda_perceptual_shoulder_arm_patch=0.0" \
  "++opt.lambda_perceptual_upper_torso_patch=0.0" \
  "++opt.lambda_perceptual_upper_torso_core_patch=0.0" \
  "++opt.lambda_perceptual_waist_patch=0.0" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "++opt.lambda_silhouette_outer=0.0" \
  "++opt.lambda_silhouette_inner=0.0" \
  "++opt.lambda_silhouette_shoulder_arm_outer_shell=0.0" \
  "++opt.lambda_silhouette_upper_torso_outer_shell=0.0" \
  "++opt.lambda_boundary_opacity_residual_reg=0.0" \
  "++opt.lambda_boundary_scaling_residual_reg=0.0" \
  "++opt.lambda_boundary_cov_residual_reg=0.0" \
  "++opt.lambda_boundary_opacity_residual_smooth=0.0" \
  "++opt.lambda_boundary_scaling_residual_smooth=0.0" \
  "++opt.lambda_boundary_cov_residual_smooth=0.0" \
  "opt.lambda_skinning=0.0" \
  "opt.lambda_aiap_xyz=0.0" \
  "opt.lambda_aiap_cov=0.0" \
  "opt.percent_dense=0.0" \
  "opt.densify_until_iter=0" \
  "opt.densify_from_iter=1000000" \
  "opt.opacity_reset_interval=1000000" \
  "best_eval_split=test" \
  "best_metric=l1_fg" \
  "best_metric_mode=min" \
  "best_metric_source=best_eval" \
  "test_interval=0" \
  "test_iterations=$checkpoint_list" \
  "save_iterations=$checkpoint_list" \
  "checkpoint_iterations=$checkpoint_list" \
  "++validation_image_log_limit=0" \
  "opt.grad_clip=0.0015" \
  > "$LOG_DIR/train.log" 2>&1
write_status "train_done" "$TRAIN_EXP"
log_event "train_done" "$TRAIN_EXP"

if [ "$DO_RENDER" = "1" ]; then
  IFS=',' read -ra steps <<< "$CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    global_iter=$((BASE_ITER + step))
    ckpt="$TRAIN_EXP/ckpt${global_iter}.pth"
    variant="v315_ckpt${global_iter}"
    run_render_eval "$variant" "$ckpt" "$EXP_ROOT/$variant" "none" "$TRAIN_EXP/.hydra"
  done
fi

"$PYTHON_BIN" - "$SUMMARY" "$EXP_ROOT" "$CHECKPOINT_STEPS" "$BASE_ITER" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
steps = [int(item) for item in sys.argv[3].split(",") if item.strip()]
base_iter = int(sys.argv[4])
variants = [
    "baseline_no_adapter",
    "v307_external_component_reference",
    "v315_init_zero_adapter",
]
variants.extend([f"v315_ckpt{base_iter + step}" for step in steps])

def load_metrics(name):
    render_exp = exp_root / name
    contour_path = render_exp / "diagnostics" / "contours" / "contour_summary.json"
    residual_path = render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json"
    if not contour_path.exists() or not residual_path.exists():
        return None
    contour = json.loads(contour_path.read_text(encoding="utf-8"))
    residual = json.loads(residual_path.read_text(encoding="utf-8"))
    return {
        "render_exp": str(render_exp),
        "fg": float(contour["mean_fg_l1"]),
        "boundary": float(contour["mean_boundary_l1"]),
        "edge": float(contour["mean_edge_symmetric_dist_px"]),
        "inner": float(residual["mean_inner_missing_pixels"]),
        "outer": float(residual["mean_outer_leak_pixels"]),
        "hard": float(residual["mean_hard_residual_score"]),
    }

metrics = {name: load_metrics(name) for name in variants}
baseline = metrics.get("baseline_no_adapter")
reference = metrics.get("v307_external_component_reference")

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
        values = metrics.get(name)
        if values is None:
            writer.writerow([name, str(exp_root / name)] + ["nan"] * (len(header) - 3) + ["missing"])
            continue
        delta_base = {key: 0.0 for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
        delta_ref = {key: 0.0 for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
        if baseline is not None:
            delta_base = {key: values[key] - baseline[key] for key in delta_base}
        if reference is not None:
            delta_ref = {key: values[key] - reference[key] for key in delta_ref}
        if name == "baseline_no_adapter":
            strict, probe, status = False, False, "baseline"
        elif name == "v307_external_component_reference":
            strict, probe, status = False, False, "external_reference"
        else:
            strict = (
                delta_base["inner"] < -0.05
                and delta_base["outer"] <= 0.0
                and delta_base["fg"] <= 0.0
                and delta_base["boundary"] <= 0.0
                and delta_base["edge"] <= 0.0
                and delta_base["hard"] < -0.000001
            )
            probe = (
                delta_base["inner"] < 0.0
                and delta_base["hard"] < 0.0
                and delta_base["outer"] <= 1.0
                and delta_base["fg"] <= 0.000050
                and delta_base["boundary"] <= 0.000050
                and delta_base["edge"] <= 0.005000
            )
            status = "strict_pass" if strict else ("probe_pass" if probe else "rejected")
        writer.writerow([
            name,
            values["render_exp"],
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
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
} >> "$LOG_DIR/run_info.txt"

write_status "all_done" "$END_BJT"
log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "TRAIN_EXP=$TRAIN_EXP"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "END_BJT=$END_BJT"
