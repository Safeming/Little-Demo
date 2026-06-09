#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v287_deformer_factor_probe_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"
DO_TRAIN="${DO_TRAIN:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-80}"
TRAIN_CHECKPOINT_STEPS="${TRAIN_CHECKPOINT_STEPS:-40,80}"
BASE_ITER="${BASE_ITER:-136410}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"
COMPONENT_CSV="${COMPONENT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/component_contributors.csv}"
POINT_CSV="${POINT_CSV:-$ROOT/exp/stageB/logs/377_stageB_v281_allframe_contributor_audit_20260518_100947_bjt/point_contributors_all.csv}"

OVER_JOINT_IDS="${OVER_JOINT_IDS:-6,9,12,13,14,15}"
UNDER_LAYER_IDS="${UNDER_LAYER_IDS:-soft,rigid,free}"
UNDER_REGION_IDS="${UNDER_REGION_IDS:-cloth,body,soft}"
UNDER_JOINT_IDS="${UNDER_JOINT_IDS:-0,1,2,4,7,8,10}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v287_deformer_factor_probe_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v287_deformer_factor_probe_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
SUMMARY="$LOG_DIR/no_train_summary.tsv"
TRAIN_SUMMARY="$LOG_DIR/train_summary.tsv"
EVENTS="$LOG_DIR/events.tsv"
SELECTED_ENV="$LOG_DIR/selected_variant.env"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

for required in "$PYTHON_BIN" "$BASE_EXP/.hydra/config.yaml" "$BASE_CKPT" "$COMPONENT_CSV" "$POINT_CSV" "$DATA_ROOT"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 2
  fi
done

START_EPOCH="$(date +%s)"
START_BJT="$(TZ=Asia/Shanghai date -d "@$START_EPOCH" '+%F %T BJT')"

cat > "$LOG_DIR/run_info.txt" <<EOF
RUN_ID=$RUN_ID
START_BJT=$START_BJT
GPU=$GPU
BASE_EXP=$BASE_EXP
BASE_CKPT=$BASE_CKPT
BASE_ITER=$BASE_ITER
COMPONENT_CSV=$COMPONENT_CSV
POINT_CSV=$POINT_CSV
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR
DATA_ROOT=$DATA_ROOT
DO_TRAIN=$DO_TRAIN
TRAIN_ITERS=$TRAIN_ITERS
TRAIN_CHECKPOINT_STEPS=$TRAIN_CHECKPOINT_STEPS

Goal:
  v287 probes explicit_binding deformer factors before training.
  It keeps the same checkpoint and tests whether boundary footprint damage is driven by
  non-rigid geometry blend, non-rigid carry, soft-rigid blend, soft-normal blend, or rotation orthogonalization.
  Short train runs only if a no-train factor passes raw contour gate.
EOF

printf 'time_bjt\tphase\tdetail\n' > "$EVENTS"
printf 'variant\tdynamic_screen\tgeom_blend\trigid_carry\tsoft_carry\tsoft_rigid_blend\tsoft_normal_blend\torth\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$SUMMARY"
printf 'label\tvariant\tckpt\trender_exp\tfg\tboundary\tedge\tinner\touter\thard\tfg_delta\tboundary_delta\tedge_delta\tinner_delta\touter_delta\thard_delta\tstrict_pass\tprobe_pass\tstatus\n' > "$TRAIN_SUMMARY"

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

dynamic_args() {
  local dynamic_screen="$1"
  if [ "$dynamic_screen" = "true" ]; then
    printf '%s\n' \
      "++pipeline.covariance_signed_dynamic_enable=true" \
      "++pipeline.covariance_signed_dynamic_component_csv=$COMPONENT_CSV" \
      "++pipeline.covariance_signed_dynamic_point_csv=$POINT_CSV" \
      "++pipeline.covariance_signed_dynamic_component_signature_enable=false" \
      "++pipeline.covariance_signed_dynamic_over_layer_ids='soft,free'" \
      "++pipeline.covariance_signed_dynamic_over_region_ids='cloth'" \
      "++pipeline.covariance_signed_dynamic_over_joint_ids='$OVER_JOINT_IDS'" \
      "++pipeline.covariance_signed_dynamic_under_layer_ids='$UNDER_LAYER_IDS'" \
      "++pipeline.covariance_signed_dynamic_under_region_ids='$UNDER_REGION_IDS'" \
      "++pipeline.covariance_signed_dynamic_under_joint_ids='$UNDER_JOINT_IDS'" \
      "++pipeline.covariance_signed_dynamic_boundary_min=0.0" \
      "++pipeline.covariance_signed_dynamic_component_pad_px=10" \
      "++pipeline.covariance_signed_dynamic_component_ellipse_scale=1.25" \
      "++pipeline.covariance_signed_dynamic_component_max_over=16" \
      "++pipeline.covariance_signed_dynamic_component_max_under=16" \
      "++pipeline.covariance_signed_dynamic_component_min_area=20" \
      "++pipeline.covariance_signed_dynamic_component_required=false" \
      "++pipeline.covariance_signed_dynamic_max_over_points=96" \
      "++pipeline.covariance_signed_dynamic_max_under_points=96" \
      "++pipeline.covariance_signed_screen_actuator_enable=true" \
      "++pipeline.covariance_signed_screen_normal_shrink_factor=0.940" \
      "++pipeline.covariance_signed_screen_normal_grow_factor=1.025" \
      "++pipeline.covariance_signed_screen_tangent_factor=1.000"
  else
    printf '%s\n' \
      "++pipeline.covariance_signed_dynamic_enable=false" \
      "++pipeline.covariance_signed_screen_actuator_enable=false"
  fi
}

render_raw() {
  local ckpt="$1"
  local render_exp="$2"
  local hydra_dir="$3"
  local dynamic_screen="$4"
  local geom_blend="$5"
  local rigid_carry="$6"
  local soft_carry="$7"
  local soft_rigid="$8"
  local soft_normal="$9"
  local orth="${10}"
  mapfile -t cov_args < <(dynamic_args "$dynamic_screen")

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_shrink_factor=1.000" \
    "++pipeline.covariance_signed_grow_factor=1.000" \
    "++pipeline.covariance_signed_anisotropic_axis=all" \
    "${cov_args[@]}" \
    "++model.deformer.rigid.non_rigid_geometry_blend=$geom_blend" \
    "++model.deformer.rigid.non_rigid_delta_rigid_preserve=$rigid_carry" \
    "++model.deformer.rigid.non_rigid_delta_soft_preserve=$soft_carry" \
    "++model.deformer.rigid.soft_rigid_blend=$soft_rigid" \
    "++model.deformer.rigid.soft_normal_blend=$soft_normal" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=$orth" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "export_interpretability=false" \
    "export_semantic_editable_assets=false" \
    "++render_export_refine=false" \
    "hydra.run.dir=$hydra_dir" \
    "wandb_disable=true"
}

analyze_raw() {
  local label="$1"
  local render_exp="$2"

  "$PYTHON_BIN" tools/analyze_377_render_contours.py \
    --render-exp "$render_exp" \
    --dataset-root "$DATA_ROOT" \
    --subject CoreView_377 \
    --split-dir test-view \
    --band-width 7 \
    --topk 12 \
    --out-dir "$render_exp/diagnostics/contours" \
    > "$LOG_DIR/contours_${label}.log" 2>&1

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
    > "$LOG_DIR/boundary_residuals_${label}.log" 2>&1
}

append_summary() {
  local summary_path="$1"
  local label="$2"
  local variant="$3"
  local ckpt="$4"
  local render_exp="$5"
  local dynamic_screen="$6"
  local geom_blend="$7"
  local rigid_carry="$8"
  local soft_carry="$9"
  local soft_rigid="${10}"
  local soft_normal="${11}"
  local orth="${12}"

  "$PYTHON_BIN" - "$summary_path" "$label" "$variant" "$ckpt" "$render_exp" "$dynamic_screen" "$geom_blend" "$rigid_carry" "$soft_carry" "$soft_rigid" "$soft_normal" "$orth" <<'PY'
import json
import sys
from pathlib import Path

summary_path, label, variant, ckpt, render_exp, dynamic_screen, geom_blend, rigid_carry, soft_carry, soft_rigid, soft_normal, orth = sys.argv[1:13]
summary_path = Path(summary_path)
render_exp = Path(render_exp)
baseline_summary_path = summary_path.with_name("no_train_summary.tsv")
contour = json.loads((render_exp / "diagnostics" / "contours" / "contour_summary.json").read_text(encoding="utf-8"))
residual = json.loads((render_exp / "diagnostics" / "boundary_residuals" / "boundary_residual_summary.json").read_text(encoding="utf-8"))
metrics = {
    "fg": float(contour["mean_fg_l1"]),
    "boundary": float(contour["mean_boundary_l1"]),
    "edge": float(contour["mean_edge_symmetric_dist_px"]),
    "inner": float(residual["mean_inner_missing_pixels"]),
    "outer": float(residual["mean_outer_leak_pixels"]),
    "hard": float(residual["mean_hard_residual_score"]),
}
baseline_source = baseline_summary_path if summary_path.name.startswith("train_") and baseline_summary_path.exists() else summary_path
rows = [line.rstrip("\n").split("\t") for line in baseline_source.read_text(encoding="utf-8").splitlines()]
header = rows[0]
baseline = None
baseline_label = "baseline_dynamic_screen_mid"
for row in rows[1:]:
    if row and row[0] == baseline_label:
        baseline = {key: float(row[header.index(key)]) for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
        break
if baseline is None or label == baseline_label:
    baseline = dict(metrics)
delta = {key: metrics[key] - baseline[key] for key in metrics}
strict = (
    delta["inner"] < -0.05
    and delta["outer"] <= 0.0
    and delta["fg"] <= 0.0
    and delta["boundary"] <= 0.0
    and delta["edge"] <= 0.0
    and delta["hard"] < -0.000001
)
probe = (
    delta["hard"] < -0.000010
    and delta["inner"] <= 0.50
    and delta["outer"] <= 1.50
    and delta["fg"] <= 0.000080
    and delta["boundary"] <= 0.000080
    and delta["edge"] <= 0.008000
)
status = "strict_pass" if strict else ("probe_pass" if probe else "rejected")

def fmt(value, digits=8):
    return f"{float(value):.{digits}f}"

if summary_path.name.startswith("train_"):
    row = [
        label,
        variant,
        ckpt,
        str(render_exp),
        fmt(metrics["fg"]),
        fmt(metrics["boundary"]),
        fmt(metrics["edge"], 6),
        fmt(metrics["inner"], 4),
        fmt(metrics["outer"], 4),
        fmt(metrics["hard"]),
        fmt(delta["fg"]),
        fmt(delta["boundary"]),
        fmt(delta["edge"], 6),
        fmt(delta["inner"], 4),
        fmt(delta["outer"], 4),
        fmt(delta["hard"]),
        "1" if strict else "0",
        "1" if probe else "0",
        status,
    ]
else:
    row = [
        label,
        dynamic_screen,
        geom_blend,
        rigid_carry,
        soft_carry,
        soft_rigid,
        soft_normal,
        orth,
        str(render_exp),
        fmt(metrics["fg"]),
        fmt(metrics["boundary"]),
        fmt(metrics["edge"], 6),
        fmt(metrics["inner"], 4),
        fmt(metrics["outer"], 4),
        fmt(metrics["hard"]),
        fmt(delta["fg"]),
        fmt(delta["boundary"]),
        fmt(delta["edge"], 6),
        fmt(delta["inner"], 4),
        fmt(delta["outer"], 4),
        fmt(delta["hard"]),
        "1" if strict else "0",
        "1" if probe else "0",
        status,
    ]
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

render_variant() {
  local variant="$1"
  local dynamic_screen="$2"
  local geom_blend="$3"
  local rigid_carry="$4"
  local soft_carry="$5"
  local soft_rigid="$6"
  local soft_normal="$7"
  local orth="$8"
  local render_exp="$EXP_ROOT/no_train_${variant}"

  log_event "no_train_render_start" "$variant dynamic=$dynamic_screen geom=$geom_blend rigid_carry=$rigid_carry soft_carry=$soft_carry soft_rigid=$soft_rigid soft_normal=$soft_normal orth=$orth"
  render_raw "$BASE_CKPT" "$render_exp" "$HYDRA_RUN_ROOT/render_${variant}" "$dynamic_screen" "$geom_blend" "$rigid_carry" "$soft_carry" "$soft_rigid" "$soft_normal" "$orth" \
    > "$LOG_DIR/render_${variant}.log" 2>&1
  log_event "no_train_analyze_start" "$variant"
  analyze_raw "$variant" "$render_exp"
  append_summary "$SUMMARY" "$variant" "$variant" "$BASE_CKPT" "$render_exp" "$dynamic_screen" "$geom_blend" "$rigid_carry" "$soft_carry" "$soft_rigid" "$soft_normal" "$orth"
  log_event "no_train_variant_done" "$variant"
}

select_variant() {
  "$PYTHON_BIN" - "$SUMMARY" "$SELECTED_ENV" <<'PY'
import csv
import shlex
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
selected_env = Path(sys.argv[2])
rows = list(csv.DictReader(summary_path.open(encoding="utf-8"), delimiter="\t"))
rows = [row for row in rows if row["variant"] != "baseline_dynamic_screen_mid"]
candidates = [row for row in rows if row["strict_pass"] == "1"]
reason = "strict"
if not candidates:
    candidates = [row for row in rows if row["probe_pass"] == "1"]
    reason = "probe"
if not candidates:
    selected_env.write_text("TRAIN_SELECTED=0\nSELECT_REASON=no_gate_pass\n", encoding="utf-8")
    raise SystemExit(0)

def score(row):
    return (
        float(row["hard_delta"]),
        float(row["edge_delta"]),
        float(row["inner_delta"]),
        float(row["outer_delta"]),
        float(row["fg_delta"]),
    )

best = sorted(candidates, key=score)[0]
values = {
    "TRAIN_SELECTED": "1",
    "SELECT_REASON": reason,
    "SELECTED_VARIANT": best["variant"],
    "SELECTED_DYNAMIC_SCREEN": best["dynamic_screen"],
    "SELECTED_GEOM_BLEND": best["geom_blend"],
    "SELECTED_RIGID_CARRY": best["rigid_carry"],
    "SELECTED_SOFT_CARRY": best["soft_carry"],
    "SELECTED_SOFT_RIGID_BLEND": best["soft_rigid_blend"],
    "SELECTED_SOFT_NORMAL_BLEND": best["soft_normal_blend"],
    "SELECTED_ORTH": best["orth"],
}
selected_env.write_text(
    "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()),
    encoding="utf-8",
)
PY
}

train_selected_variant() {
  # shellcheck disable=SC1090
  source "$SELECTED_ENV"
  if [ "${TRAIN_SELECTED:-0}" != "1" ]; then
    log_event "train_skip" "${SELECT_REASON:-no_gate_pass}"
    return 0
  fi
  if [ "$DO_TRAIN" != "1" ]; then
    log_event "train_skip" "DO_TRAIN=$DO_TRAIN selected=$SELECTED_VARIANT"
    return 0
  fi
  mapfile -t cov_args < <(dynamic_args "$SELECTED_DYNAMIC_SCREEN")
  local train_exp="$EXP_ROOT/train_${SELECTED_VARIANT}"
  local checkpoint_list="[$TRAIN_CHECKPOINT_STEPS]"

  log_event "train_start" "$SELECTED_VARIANT reason=$SELECT_REASON exp=$train_exp"
  env "${COMMON_ENV[@]}" "$PYTHON_BIN" train.py \
    --config-path "$BASE_EXP/.hydra" \
    --config-name config \
    mode=train \
    "dataset.root_dir=$DATA_ROOT" \
    "dataset.preload=false" \
    "dataset.train_views=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]" \
    "dataset.val_views=[21,22,23]" \
    "dataset.test_views.view=[21,22,23]" \
    "dataset.train_frames=[0,570,1]" \
    "dataset.val_frames=[0,570,60]" \
    "dataset.test_frames.view=[0,570,60]" \
    "dataset.parsing_prior.enable=false" \
    "dataset.parsing_prior.roi_enable=false" \
    "dataset.parsing_prior.compact_mapping_file=" \
    "start_checkpoint=$BASE_CKPT" \
    "exp_dir=$train_exp" \
    "hydra.run.dir=$HYDRA_RUN_ROOT/train_${SELECTED_VARIANT}" \
    "seed=-1" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
    "++resume.partial_converter_missing_keys_allow_patterns=[texture.detail_high_freq_view_conflict_,camera_affine.]" \
    "++resume.disable_densify_on_resume=true" \
    "++resume.disable_opacity_reset_on_resume=true" \
    "++resume.require_no_densify_on_resume=true" \
    "++resume.use_checkpoint_iteration_as_offset=true" \
    "++resume.clear_boundary_tags_on_resume=true" \
    "++resume.clear_binding_state_on_resume=false" \
    "pipeline.pose_noise=0.0" \
    "pipeline.compute_cov3D_python=true" \
    "++pipeline.covariance_mode=default" \
    "++pipeline.covariance_signed_point_json=" \
    "++pipeline.covariance_signed_shrink_factor=1.000" \
    "++pipeline.covariance_signed_grow_factor=1.000" \
    "++pipeline.covariance_signed_anisotropic_axis=all" \
    "${cov_args[@]}" \
    "++model.deformer.rigid.non_rigid_geometry_blend=$SELECTED_GEOM_BLEND" \
    "++model.deformer.rigid.non_rigid_delta_rigid_preserve=$SELECTED_RIGID_CARRY" \
    "++model.deformer.rigid.non_rigid_delta_soft_preserve=$SELECTED_SOFT_CARRY" \
    "++model.deformer.rigid.soft_rigid_blend=$SELECTED_SOFT_RIGID_BLEND" \
    "++model.deformer.rigid.soft_normal_blend=$SELECTED_SOFT_NORMAL_BLEND" \
    "++model.deformer.rigid.rotation_orthogonalize_enable=$SELECTED_ORTH" \
    "model.pose_correction.delay=1" \
    "++model.pose_correction.train_root_orient=false" \
    "++model.pose_correction.train_pose_body=false" \
    "++model.pose_correction.train_pose_hand=false" \
    "++model.pose_correction.train_trans=false" \
    "++model.pose_correction.train_betas=false" \
    "opt.iterations=$TRAIN_ITERS" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.feature_lr=0.00008" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.pose_correction_lr=0.0" \
    "opt.texture_lr=0.00000010" \
    "opt.tex_latent_lr=0.0" \
    "++opt.texture_trainable_name_patterns=[*]" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.stageB_semantic_loss_enable=false" \
    "++opt.stageB_semantic_body_cloth_weight=0.0" \
    "++opt.stageB_semantic_compact_weight=0.0" \
    "++opt.lambda_binding_semantic_adapter_reg=0.0" \
    "++opt.semantic_region_logits_lr=0.0" \
    "++opt.semantic_compact_logits_lr=0.0" \
    "++opt.train_sample_mode=frame_balanced_camera_weighted" \
    "++opt.train_sample_camera_min_prob=0.018" \
    "++opt.train_sample_camera_max_prob=0.125" \
    "++opt.foreground_mask_source=hard" \
    "++opt.global_mask_source=hard" \
    "++opt.boundary_target_mask_source=hard" \
    "++opt.boundary_region_source=binary" \
    "++opt.boundary_band_width=8" \
    "opt.lambda_l1=0.002" \
    "opt.lambda_l1_fg=0.010" \
    "opt.lambda_l1_boundary=0.004" \
    "opt.lambda_perceptual=0.0" \
    "++opt.lambda_opacity_gated_rgb_inner=0.0" \
    "++opt.lambda_opacity_gated_rgb_outer=0.0" \
    "++opt.lambda_raw_support_hinge_inner=0.0" \
    "++opt.lambda_raw_support_hinge_outer=0.0" \
    "++opt.contour_uncertainty_enable=true" \
    "++opt.contour_uncertainty_band_width=13" \
    "++opt.contour_uncertainty_outer_width=24" \
    "++opt.contour_uncertainty_min_weight=0.22" \
    "++opt.teacher_render_distill_enable=false" \
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
    "++opt.lambda_silhouette_outer_shell=0.0" \
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
    "best_eval_split=test" \
    "best_metric=l1_fg" \
    "best_metric_mode=min" \
    "best_metric_source=best_eval" \
    "test_interval=0" \
    "test_iterations=$checkpoint_list" \
    "save_iterations=$checkpoint_list" \
    "checkpoint_iterations=$checkpoint_list" \
    "++validation_image_log_limit=0" \
    "opt.grad_clip=0.0010" \
    > "$LOG_DIR/train_${SELECTED_VARIANT}.log" 2>&1
  log_event "train_done" "$SELECTED_VARIANT exp=$train_exp"

  IFS=',' read -ra steps <<< "$TRAIN_CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    local global_iter=$((BASE_ITER + step))
    local ckpt="$train_exp/ckpt${global_iter}.pth"
    local label="${SELECTED_VARIANT}_ckpt${global_iter}"
    local render_exp="${train_exp}_raw_render_${label}"
    if [ ! -f "$ckpt" ]; then
      log_event "train_render_skip" "missing=$ckpt"
      continue
    fi
    log_event "train_render_start" "$label"
    render_raw "$ckpt" "$render_exp" "$HYDRA_RUN_ROOT/render_${label}" "$SELECTED_DYNAMIC_SCREEN" "$SELECTED_GEOM_BLEND" "$SELECTED_RIGID_CARRY" "$SELECTED_SOFT_CARRY" "$SELECTED_SOFT_RIGID_BLEND" "$SELECTED_SOFT_NORMAL_BLEND" "$SELECTED_ORTH" \
      > "$LOG_DIR/render_${label}.log" 2>&1
    log_event "train_analyze_start" "$label"
    analyze_raw "$label" "$render_exp"
    append_summary "$TRAIN_SUMMARY" "$label" "$SELECTED_VARIANT" "$ckpt" "$render_exp" "$SELECTED_DYNAMIC_SCREEN" "$SELECTED_GEOM_BLEND" "$SELECTED_RIGID_CARRY" "$SELECTED_SOFT_CARRY" "$SELECTED_SOFT_RIGID_BLEND" "$SELECTED_SOFT_NORMAL_BLEND" "$SELECTED_ORTH"
    log_event "train_gate_done" "$label"
  done
}

render_variant baseline_dynamic_screen_mid true 0.35 0.4 0.7 0.6 0.8 false
render_variant baseline_raw false 0.35 0.4 0.7 0.6 0.8 false
render_variant geom0_dynamic true 0.0 0.4 0.7 0.6 0.8 false
render_variant geom015_dynamic true 0.15 0.4 0.7 0.6 0.8 false
render_variant geom060_dynamic true 0.60 0.4 0.7 0.6 0.8 false
render_variant carry0_dynamic true 0.35 0.0 0.0 0.6 0.8 false
render_variant rigidcarry0_dynamic true 0.35 0.0 0.7 0.6 0.8 false
render_variant softcarry0_dynamic true 0.35 0.4 0.0 0.6 0.8 false
render_variant soft_rigid050_dynamic true 0.35 0.4 0.7 0.5 0.8 false
render_variant soft_rigid075_dynamic true 0.35 0.4 0.7 0.75 0.8 false
render_variant soft_normal050_dynamic true 0.35 0.4 0.7 0.6 0.5 false
render_variant soft_normal100_dynamic true 0.35 0.4 0.7 0.6 1.0 false
render_variant orth_dynamic true 0.35 0.4 0.7 0.6 0.8 true

select_variant
train_selected_variant

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "SUMMARY=$SUMMARY"
  echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
  if [ -f "$SELECTED_ENV" ]; then
    cat "$SELECTED_ENV"
  fi
} >> "$LOG_DIR/run_info.txt"

log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "SUMMARY=$SUMMARY"
echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
echo "END_BJT=$END_BJT"
