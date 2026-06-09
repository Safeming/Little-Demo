#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v313_view_conditioned_xbar_teacher_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
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

TRAIN_ITERS="${TRAIN_ITERS:-300}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-100,200,300}"
TRAIN_RIGID_LR="${TRAIN_RIGID_LR:-0.00025}"
TRAIN_LAMBDA_TEACHER="${TRAIN_LAMBDA_TEACHER:-80.0}"
TRAIN_LAMBDA_XBAR_REG="${TRAIN_LAMBDA_XBAR_REG:-0.010}"
TRAIN_GRAD_CLIP="${TRAIN_GRAD_CLIP:-0.010}"
TEACHER_CENTER_STRENGTH="${TEACHER_CENTER_STRENGTH:-0.45}"
TEACHER_COMPONENT_PAD="${TEACHER_COMPONENT_PAD:-2}"
TEACHER_COMPONENT_SCALE="${TEACHER_COMPONENT_SCALE:-1.05}"
TEACHER_COMPONENT_MAX="${TEACHER_COMPONENT_MAX:-12}"
TEACHER_COMPONENT_MIN_AREA="${TEACHER_COMPONENT_MIN_AREA:-40}"
TEACHER_MAX_POINTS="${TEACHER_MAX_POINTS:-1024}"
FORWARD_XBAR_SCALE="${FORWARD_XBAR_SCALE:-0.0035}"
FORWARD_XBAR_MAX="${FORWARD_XBAR_MAX:-0.010}"
FORWARD_XBAR_NORMAL_SCALE="${FORWARD_XBAR_NORMAL_SCALE:-0.70}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v313_view_conditioned_xbar_teacher_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v313_view_conditioned_xbar_teacher_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
STATUS_JSON="$LOG_DIR/status.json"
TRAIN_EXP="$EXP_ROOT/train_view_conditioned_xbar"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
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
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS
TRAIN_RIGID_LR=$TRAIN_RIGID_LR
TRAIN_LAMBDA_TEACHER=$TRAIN_LAMBDA_TEACHER
TRAIN_LAMBDA_XBAR_REG=$TRAIN_LAMBDA_XBAR_REG
TEACHER_CENTER_STRENGTH=$TEACHER_CENTER_STRENGTH
FORWARD_XBAR_SCALE=$FORWARD_XBAR_SCALE
FORWARD_XBAR_MAX=$FORWARD_XBAR_MAX
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v313 internalizes the v307 component-local center correction into the
  explicit-binding model. Training uses train-view residual components only as a
  teacher loss for the view-conditioned forward_trunk_xbar head. Evaluation
  disables component CSV, geometry_fidelity, and screen-space signed actuators.
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
  "dataset.train_frames=$TRAIN_FRAMES_SPEC"
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

LEARNED_XBAR_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_dynamic_enable=false"
  "++pipeline.covariance_signed_screen_actuator_enable=false"
  "++pipeline.covariance_signed_center_offset_enable=false"
  "++pipeline.boundary_cov_residual_enable=false"
  "++pipeline.binding_covariance_guard_enable=false"
  "resume.allow_partial_converter_load=true"
  "resume.restore_gaussian_optimizer_state=false"
  "resume.restore_converter_optimizer_state=false"
  "resume.restore_converter_scheduler_state=false"
  "resume.partial_converter_missing_keys_allow_patterns=[deformer.rigid.forward_trunk_mlp.,camera_affine.,pose_correction.]"
  "++model.deformer.rigid.rotation_orthogonalize_enable=false"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
  "++model.deformer.rigid.geometry_fidelity_target=free_lbs"
  "++model.deformer.rigid.geometry_fidelity_center_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_component_enable=false"
  "++model.deformer.rigid.geometry_fidelity_component_csv=''"
  "++model.deformer.rigid.forward_trunk_enable=true"
  "++model.deformer.rigid.forward_trunk_anchor_enable=false"
  "++model.deformer.rigid.forward_trunk_support_enable=false"
  "++model.deformer.rigid.forward_trunk_layer_enable=false"
  "++model.deformer.rigid.forward_trunk_xbar_enable=true"
  "++model.deformer.rigid.forward_trunk_viewdir_enable=true"
  "++model.deformer.rigid.forward_trunk_viewdir_detach=true"
  "++model.deformer.rigid.forward_trunk_feature_detach=true"
  "++model.deformer.rigid.forward_trunk_blend_alpha=1.0"
  "++model.deformer.rigid.forward_trunk_xbar_alpha_scale=1.0"
  "++model.deformer.rigid.forward_trunk_output_clamp=4.0"
  "++model.deformer.rigid.forward_trunk_xbar_scale=$FORWARD_XBAR_SCALE"
  "++model.deformer.rigid.forward_trunk_xbar_max=$FORWARD_XBAR_MAX"
  "++model.deformer.rigid.forward_trunk_xbar_tangent_scale=1.0"
  "++model.deformer.rigid.forward_trunk_xbar_normal_scale=$FORWARD_XBAR_NORMAL_SCALE"
  "++model.deformer.rigid.forward_trunk_mlp.n_neurons=96"
  "++model.deformer.rigid.forward_trunk_mlp.n_hidden_layers=3"
  "++model.deformer.rigid.forward_trunk_mlp.skip_in=[]"
  "++model.deformer.rigid.forward_trunk_mlp.cond_in=[]"
  "++model.deformer.rigid.forward_trunk_mlp.multires=0"
  "++model.deformer.rigid.forward_trunk_mlp.last_layer_init=true"
)

TEACHER_TRAIN_ARGS=(
  "${LEARNED_XBAR_ARGS[@]}"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_enable=true"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_center_strength=$TEACHER_CENTER_STRENGTH"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_boundary_min=0.12"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_layer_ids='soft,free'"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_region_ids='cloth,soft'"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_joint_ids=''"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_non_rigid_min=0.0"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_power=1.2"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_max_points=$TEACHER_MAX_POINTS"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_enable=true"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_csv=$COMPONENT_CSV"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_direction=inner"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_pad_px=$TEACHER_COMPONENT_PAD"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_ellipse_scale=$TEACHER_COMPONENT_SCALE"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_max=$TEACHER_COMPONENT_MAX"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_min_area=$TEACHER_COMPONENT_MIN_AREA"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_required=true"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_improvement_enable=true"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_component_improvement_margin_px=0.0"
  "++model.deformer.rigid.forward_trunk_xbar_teacher_loss=l1"
)

render_variant() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  local mode="$4"
  local hydra_dir="$5"

  local extra_args=()
  if [ "$mode" = "v307" ]; then
    extra_args=(
      "++explicit_binding_render_preset=v307_adopted_geometry"
      "++explicit_binding_adopted_component_csv=$COMPONENT_CSV"
      "++explicit_binding_adopted_point_csv=$POINT_CSV"
      "++explicit_binding_adopted_center_strength=$TEACHER_CENTER_STRENGTH"
      "++explicit_binding_adopted_outer_px=0.35"
      "++explicit_binding_adopted_component_required=true"
      "++explicit_binding_adopted_improvement_guard=true"
      "++explicit_binding_adopted_max_points=96"
    )
  elif [ "$mode" = "learned" ]; then
    extra_args=("${LEARNED_XBAR_ARGS[@]}")
  fi

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
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

render_and_analyze() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  local mode="$4"
  log_event "render_start" "$variant"
  render_variant "$variant" "$ckpt" "$render_exp" "$mode" "$HYDRA_RUN_ROOT/render_${variant}"
  analyze_variant "$variant" "$render_exp"
  log_event "render_done" "$variant"
}

BASELINE_RENDER_EXP="$EXP_ROOT/baseline_no_preset"
V307_RENDER_EXP="$EXP_ROOT/v307_external_component_reference"
INIT_RENDER_EXP="$EXP_ROOT/v313_init_learned_no_external"

render_and_analyze "baseline_no_preset" "$BASE_CKPT" "$BASELINE_RENDER_EXP" "none"
render_and_analyze "v307_external_component_reference" "$BASE_CKPT" "$V307_RENDER_EXP" "v307"
render_and_analyze "v313_init_learned_no_external" "$BASE_CKPT" "$INIT_RENDER_EXP" "learned"

checkpoint_list="[$TRAIN_CHECKPOINT_STEPS]"
log_event "train_start" "$TRAIN_EXP"
env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
  --config-path "$BASE_EXP/.hydra" \
  --config-name config \
  "mode=train" \
  "start_checkpoint=$BASE_CKPT" \
  "load_ckpt=$BASE_CKPT" \
  "exp_dir=$TRAIN_EXP" \
  "hydra.run.dir=$HYDRA_RUN_ROOT/train" \
  "dataset.root_dir=$DATA_ROOT" \
  "dataset.preload=false" \
  "dataset.train_views=$TRAIN_VIEWS_SPEC" \
  "dataset.train_frames=$TRAIN_FRAMES_SPEC" \
  "dataset.test_views.view=$TEST_VIEWS_SPEC" \
  "dataset.test_frames.view=$TEST_FRAMES_SPEC" \
  "dataset.parsing_prior.enable=false" \
  "dataset.parsing_prior.roi_enable=false" \
  "${TEACHER_TRAIN_ARGS[@]}" \
  "++resume.allow_start_load_ckpt_mismatch=false" \
  "++resume.disable_densify_on_resume=true" \
  "++resume.disable_opacity_reset_on_resume=true" \
  "++resume.require_no_densify_on_resume=true" \
  "++resume.use_checkpoint_iteration_as_offset=true" \
  "++resume.clear_boundary_tags_on_resume=false" \
  "model.pose_correction.delay=1" \
  "++model.pose_correction.train_root_orient=false" \
  "++model.pose_correction.train_pose_body=false" \
  "++model.pose_correction.train_pose_hand=false" \
  "++model.pose_correction.train_trans=false" \
  "++model.pose_correction.train_betas=false" \
  "opt.iterations=$TRAIN_ITERS" \
  "opt.position_lr_init=0.0" \
  "opt.position_lr_final=0.0" \
  "opt.feature_lr=0.0" \
  "opt.opacity_lr=0.0" \
  "opt.scaling_lr=0.0" \
  "opt.rotation_lr=0.0" \
  "opt.rigid_lr=$TRAIN_RIGID_LR" \
  "++opt.rigid_trainable_name_patterns=[forward_trunk_mlp.*]" \
  "opt.non_rigid_lr=0.0" \
  "opt.nr_latent_lr=0.0" \
  "opt.pose_correction_lr=0.0" \
  "opt.texture_lr=0.0" \
  "opt.tex_latent_lr=0.0" \
  "++opt.camera_affine_enable=false" \
  "++opt.camera_affine_lr=0.0" \
  "++opt.camera_geometry_enable=true" \
  "++opt.camera_geometry_lr=0.0" \
  "++opt.boundary_opacity_residual_lr=0.0" \
  "++opt.boundary_scaling_residual_lr=0.0" \
  "++opt.boundary_cov_residual_lr=0.0" \
  "++opt.stageB_semantic_loss_enable=false" \
  "++opt.semantic_region_logits_lr=0.0" \
  "++opt.semantic_compact_logits_lr=0.0" \
  "opt.lambda_l1=0.0" \
  "opt.lambda_l1_fg=0.0" \
  "++opt.lambda_l1_boundary=0.0" \
  "opt.lambda_perceptual=0.0" \
  "opt.lambda_mask=0.0" \
  "++opt.lambda_mask_boundary=0.0" \
  "++opt.lambda_mask_boundary_hard=0.0" \
  "++opt.lambda_silhouette_outer=0.0" \
  "++opt.lambda_silhouette_inner=0.0" \
  "opt.lambda_skinning=0.0" \
  "opt.lambda_aiap_xyz=0.0" \
  "opt.lambda_aiap_cov=0.0" \
  "++opt.lambda_binding_rigid=0.0" \
  "++opt.lambda_binding_soft=0.0" \
  "++opt.lambda_binding_canonical=0.0" \
  "++opt.lambda_binding_surface=0.0" \
  "++opt.lambda_binding_entropy=0.0" \
  "++opt.lambda_binding_temporal=0.0" \
  "++opt.lambda_binding_semantic=0.0" \
  "++opt.lambda_binding_forward_trunk_xbar_teacher=$TRAIN_LAMBDA_TEACHER" \
  "++opt.lambda_binding_forward_trunk_xbar_reg=$TRAIN_LAMBDA_XBAR_REG" \
  "++opt.lambda_binding_forward_trunk_xbar_abs_mean=0.0" \
  "++opt.binding_teacher_only_loss_enable=true" \
  "++opt.binding_teacher_only_loss_names=[binding_forward_trunk_xbar_teacher]" \
  "++opt.binding_teacher_only_include_xbar_reg=true" \
  "++opt.train_sample_mode=frame_balanced_camera_weighted" \
  "++opt.train_sample_camera_min_prob=0.018" \
  "++opt.train_sample_camera_max_prob=0.125" \
  "++opt.train_sample_log_interval=100" \
  "++opt.train_sample_accumulation_steps=1" \
  "opt.percent_dense=0.0" \
  "opt.densify_until_iter=0" \
  "opt.densify_from_iter=1000000" \
  "opt.opacity_reset_interval=1000000" \
  "test_interval=0" \
  "test_iterations=$checkpoint_list" \
  "save_iterations=$checkpoint_list" \
  "checkpoint_iterations=$checkpoint_list" \
  "++validation_image_log_limit=0" \
  "opt.grad_clip=$TRAIN_GRAD_CLIP" \
  "export_interpretability=false" \
  "export_semantic_editable_assets=false" \
  "++render_export_refine=false" \
  "wandb_disable=true" \
  > "$LOG_DIR/train.log" 2>&1
log_event "train_done" "$TRAIN_EXP"

IFS=',' read -ra steps <<< "$TRAIN_CHECKPOINT_STEPS"
for step in "${steps[@]}"; do
  global_iter=$((BASE_ITER + step))
  ckpt="$TRAIN_EXP/ckpt${global_iter}.pth"
  if [ ! -f "$ckpt" ]; then
    log_event "render_skip" "missing=$ckpt"
    continue
  fi
  render_and_analyze "v313_ckpt${global_iter}_learned_no_external" \
    "$ckpt" \
    "$EXP_ROOT/v313_ckpt${global_iter}_learned_no_external" \
    "learned"
done

"$PYTHON_BIN" - "$SUMMARY" "$BASELINE_RENDER_EXP" "$V307_RENDER_EXP" "$INIT_RENDER_EXP" "$EXP_ROOT" "$TRAIN_CHECKPOINT_STEPS" "$BASE_ITER" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
baseline_render = Path(sys.argv[2])
v307_render = Path(sys.argv[3])
init_render = Path(sys.argv[4])
exp_root = Path(sys.argv[5])
steps = [int(x) for x in sys.argv[6].split(",") if x.strip()]
base_iter = int(sys.argv[7])

variant_paths = {
    "baseline_no_preset": baseline_render,
    "v307_external_component_reference": v307_render,
    "v313_init_learned_no_external": init_render,
}
for step in steps:
    global_iter = base_iter + step
    render_exp = exp_root / f"v313_ckpt{global_iter}_learned_no_external"
    if (render_exp / "diagnostics" / "contours" / "contour_summary.json").exists():
        variant_paths[f"v313_ckpt{global_iter}_learned_no_external"] = render_exp

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

metrics = {name: metrics_from_render(path) for name, path in variant_paths.items()}
baseline = metrics["baseline_no_preset"]
v307 = metrics["v307_external_component_reference"]

def gate(delta):
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

fields = [
    "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "fg_delta_v307", "boundary_delta_v307", "edge_delta_v307",
    "inner_delta_v307", "outer_delta_v307", "hard_delta_v307",
    "strict_pass", "probe_pass", "status",
]
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for name, render_exp in variant_paths.items():
        values = metrics[name]
        delta_base = {key: values[key] - baseline[key] for key in values}
        delta_v307 = {key: values[key] - v307[key] for key in values}
        strict, probe, status = gate(delta_base)
        if name == "baseline_no_preset":
            strict = probe = False
            status = "baseline"
        elif name == "v307_external_component_reference":
            status = "external_reference"
        writer.writerow({
            "variant": name,
            "render_exp": str(render_exp),
            "fg": f"{values['fg']:.8f}",
            "boundary": f"{values['boundary']:.8f}",
            "edge": f"{values['edge']:.6f}",
            "inner": f"{values['inner']:.4f}",
            "outer": f"{values['outer']:.4f}",
            "hard": f"{values['hard']:.8f}",
            "fg_delta_base": f"{delta_base['fg']:.8f}",
            "boundary_delta_base": f"{delta_base['boundary']:.8f}",
            "edge_delta_base": f"{delta_base['edge']:.6f}",
            "inner_delta_base": f"{delta_base['inner']:.4f}",
            "outer_delta_base": f"{delta_base['outer']:.4f}",
            "hard_delta_base": f"{delta_base['hard']:.8f}",
            "fg_delta_v307": f"{delta_v307['fg']:.8f}",
            "boundary_delta_v307": f"{delta_v307['boundary']:.8f}",
            "edge_delta_v307": f"{delta_v307['edge']:.6f}",
            "inner_delta_v307": f"{delta_v307['inner']:.4f}",
            "outer_delta_v307": f"{delta_v307['outer']:.4f}",
            "hard_delta_v307": f"{delta_v307['hard']:.8f}",
            "strict_pass": "1" if strict else "0",
            "probe_pass": "1" if probe else "0",
            "status": status,
        })
print(json.dumps({"summary": str(summary_path), "variants": list(variant_paths)}, indent=2), flush=True)
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "TRAIN_EXP=$TRAIN_EXP"
  echo "BASELINE_RENDER_EXP=$BASELINE_RENDER_EXP"
  echo "V307_RENDER_EXP=$V307_RENDER_EXP"
  echo "INIT_RENDER_EXP=$INIT_RENDER_EXP"
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "TRAIN_EXP=$TRAIN_EXP"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
