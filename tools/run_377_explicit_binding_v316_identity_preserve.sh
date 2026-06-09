#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ictrl/bin/python}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-v316_identity_preserve_$(TZ=Asia/Shanghai date '+%Y%m%d_%H%M%S_bjt')}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-6}"

DATA_ROOT="${DATA_ROOT:-$ROOT/data/ZJUMoCap}"
BASE_EXP="${BASE_EXP:-$ROOT/exp/stageB/377_explicit_binding_v271_color_texture_only_v271_color_texture_only_20260517_150215_bjt}"
BASE_CKPT="${BASE_CKPT:-$BASE_EXP/ckpt136410.pth}"

TRAIN_VIEWS_SPEC="${TRAIN_VIEWS_SPEC:-[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}"
TRAIN_FRAMES_SPEC="${TRAIN_FRAMES_SPEC:-[0,570,1]}"
TEST_VIEWS_SPEC="${TEST_VIEWS_SPEC:-[21,22,23]}"
TEST_FRAMES_SPEC="${TEST_FRAMES_SPEC:-[0,570,60]}"

RUN_TRAIN="${RUN_TRAIN:-1}"
ITERATIONS="${ITERATIONS:-100}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-100}"
FEATURE_LR="${FEATURE_LR:-0.00005}"
TEXTURE_LR="${TEXTURE_LR:-0.00000010}"
DO_RENDER="${DO_RENDER:-1}"

EXP_ROOT="${EXP_ROOT:-$ROOT/exp/stageB/377_explicit_binding_v316_identity_preserve_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT/exp/stageB/logs/377_explicit_binding_v316_identity_preserve_${RUN_ID}}"
HYDRA_RUN_ROOT="$LOG_DIR/hydra_runtime"
EVENTS="$LOG_DIR/events.tsv"
NO_TRAIN_SUMMARY="$LOG_DIR/no_train_summary.tsv"
TRAIN_SUMMARY="$LOG_DIR/train_summary.tsv"
SELECTED_ENV="$LOG_DIR/selected_variant.env"
STATUS_JSON="$LOG_DIR/status.json"

mkdir -p "$EXP_ROOT" "$LOG_DIR" "$HYDRA_RUN_ROOT"

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
TRAIN_VIEWS_SPEC=$TRAIN_VIEWS_SPEC
TRAIN_FRAMES_SPEC=$TRAIN_FRAMES_SPEC
TEST_VIEWS_SPEC=$TEST_VIEWS_SPEC
TEST_FRAMES_SPEC=$TEST_FRAMES_SPEC
RUN_TRAIN=$RUN_TRAIN
ITERATIONS=$ITERATIONS
CHECKPOINT_STEPS=$CHECKPOINT_STEPS
FEATURE_LR=$FEATURE_LR
TEXTURE_LR=$TEXTURE_LR
EXP_ROOT=$EXP_ROOT
LOG_DIR=$LOG_DIR

Goal:
  v316 tests the root-cause fix directly: explicit binding must attach
  interpretable binding attributes without changing the no-edit Gaussian render.
  The no-train A/B keeps binding maps/semantic attributes, but preserves visible
  xyz/rotation. It then tests which fwd_transform mode keeps texture stable.
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

COMMON_IDENTITY_ARGS=(
  "pipeline.compute_cov3D_python=true"
  "++pipeline.covariance_mode=default"
  "++pipeline.covariance_signed_dynamic_enable=false"
  "++pipeline.covariance_signed_screen_actuator_enable=false"
  "++pipeline.covariance_signed_center_offset_enable=false"
  "++pipeline.boundary_cov_residual_enable=false"
  "++pipeline.binding_covariance_guard_enable=false"
  "++model.deformer.rigid.geometry_fidelity_gate_enable=false"
  "++model.deformer.rigid.geometry_fidelity_center_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_rotation_strength=0.0"
  "++model.deformer.rigid.geometry_fidelity_component_enable=false"
)

render_variant() {
  local variant="$1"
  local ckpt="$2"
  local render_exp="$3"
  local identity_enable="$4"
  local fwd_mode="$5"
  local rotation_mode="$6"
  local config_dir="$7"
  local hydra_dir="$8"

  env "${COMMON_ENV[@]}" "$PYTHON_BIN" render.py \
    --config-path "$config_dir" \
    --config-name config \
    mode=test \
    "load_ckpt=$ckpt" \
    "exp_dir=$render_exp" \
    "${COMMON_RENDER_ARGS[@]}" \
    "${COMMON_IDENTITY_ARGS[@]}" \
    "++model.deformer.rigid.identity_render_enable=$identity_enable" \
    "++model.deformer.rigid.identity_render_fwd_transform_mode=$fwd_mode" \
    "++model.deformer.rigid.identity_render_rotation_mode=$rotation_mode" \
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

run_render_eval() {
  local variant="$1"
  local ckpt="$2"
  local identity_enable="$3"
  local fwd_mode="$4"
  local rotation_mode="$5"
  local config_dir="$6"
  local render_exp="$EXP_ROOT/$variant"

  log_event "render_start" "$variant"
  render_variant "$variant" "$ckpt" "$render_exp" "$identity_enable" "$fwd_mode" "$rotation_mode" "$config_dir" "$HYDRA_RUN_ROOT/render_${variant}"
  log_event "analyze_start" "$variant"
  analyze_variant "$variant" "$render_exp"
  log_event "render_done" "$variant"
}

if [ "$DO_RENDER" = "1" ]; then
  run_render_eval "baseline_no_identity" "$BASE_CKPT" false binding preserve "$BASE_EXP/.hydra"
  run_render_eval "identity_fwd_binding_rot_preserve" "$BASE_CKPT" true binding preserve "$BASE_EXP/.hydra"
  run_render_eval "identity_fwd_identity_rot_preserve" "$BASE_CKPT" true identity preserve "$BASE_EXP/.hydra"
  run_render_eval "identity_fwd_translation_rot_preserve" "$BASE_CKPT" true translation preserve "$BASE_EXP/.hydra"
  run_render_eval "identity_fwd_binding_rot_binding" "$BASE_CKPT" true binding binding "$BASE_EXP/.hydra"
fi

"$PYTHON_BIN" - "$NO_TRAIN_SUMMARY" "$EXP_ROOT" "$SELECTED_ENV" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
selected_env = Path(sys.argv[3])
variants = [
    ("baseline_no_identity", "false", "binding", "preserve"),
    ("identity_fwd_binding_rot_preserve", "true", "binding", "preserve"),
    ("identity_fwd_identity_rot_preserve", "true", "identity", "preserve"),
    ("identity_fwd_translation_rot_preserve", "true", "translation", "preserve"),
    ("identity_fwd_binding_rot_binding", "true", "binding", "binding"),
]

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

metrics = {name: load_metrics(name) for name, _, _, _ in variants}
baseline = metrics.get("baseline_no_identity")
if baseline is None:
    raise RuntimeError("baseline_no_identity metrics missing")

header = [
    "variant", "identity_enable", "fwd_mode", "rotation_mode", "render_exp",
    "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "strict_pass", "probe_pass", "status",
]
rows = []
selected = None
for name, identity_enable, fwd_mode, rotation_mode in variants:
    values = metrics.get(name)
    if values is None:
        row = [name, identity_enable, fwd_mode, rotation_mode, str(exp_root / name)] + ["nan"] * 12 + ["false", "false", "missing"]
        rows.append(row)
        continue
    delta = {key: values[key] - baseline[key] for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
    if name == "baseline_no_identity":
        strict = False
        probe = False
        status = "baseline"
    else:
        strict = (
            delta["inner"] < -0.05
            and delta["outer"] <= 0.0
            and delta["fg"] <= 0.0
            and delta["boundary"] <= 0.0
            and delta["edge"] <= 0.0
            and delta["hard"] < -0.000001
        )
        probe = (
            delta["inner"] <= 0.0
            and delta["hard"] < -0.000001
            and delta["outer"] <= 1.0
            and delta["fg"] <= 0.000050
            and delta["boundary"] <= 0.000050
            and delta["edge"] <= 0.005000
        )
        status = "strict_pass" if strict else ("probe_pass" if probe else "rejected")
    row = [
        name, identity_enable, fwd_mode, rotation_mode, values["render_exp"],
        f"{values['fg']:.8f}", f"{values['boundary']:.8f}", f"{values['edge']:.6f}",
        f"{values['inner']:.4f}", f"{values['outer']:.4f}", f"{values['hard']:.8f}",
        f"{delta['fg']:.8f}", f"{delta['boundary']:.8f}", f"{delta['edge']:.6f}",
        f"{delta['inner']:.4f}", f"{delta['outer']:.4f}", f"{delta['hard']:.8f}",
        str(bool(strict)).lower(), str(bool(probe)).lower(), status,
    ]
    rows.append(row)
    if status in ("strict_pass", "probe_pass"):
        rank = (0 if status == "strict_pass" else 1, values["hard"], values["edge"], values["fg"])
        candidate = (rank, name, identity_enable, fwd_mode, rotation_mode)
        if selected is None or candidate[0] < selected[0]:
            selected = candidate

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerows(rows)

if selected is None:
    selected_env.write_text("SELECTED_VARIANT=\nSELECTED_IDENTITY_ENABLE=\nSELECTED_FWD_MODE=\nSELECTED_ROTATION_MODE=\nSELECTED_STATUS=none\n", encoding="utf-8")
else:
    _, name, identity_enable, fwd_mode, rotation_mode = selected
    status = next(row[-1] for row in rows if row[0] == name)
    selected_env.write_text(
        "\n".join([
            f"SELECTED_VARIANT={name}",
            f"SELECTED_IDENTITY_ENABLE={identity_enable}",
            f"SELECTED_FWD_MODE={fwd_mode}",
            f"SELECTED_ROTATION_MODE={rotation_mode}",
            f"SELECTED_STATUS={status}",
            "",
        ]),
        encoding="utf-8",
    )
PY

source "$SELECTED_ENV"
log_event "no_train_done" "selected=${SELECTED_VARIANT:-none} status=${SELECTED_STATUS:-none}"

if [ "$RUN_TRAIN" = "1" ] && [ -n "${SELECTED_VARIANT:-}" ] && [ "${SELECTED_STATUS:-none}" != "none" ]; then
  TRAIN_EXP="$EXP_ROOT/train_${SELECTED_VARIANT}"
  mkdir -p "$TRAIN_EXP"
  checkpoint_list="[$CHECKPOINT_STEPS]"

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
    "hydra.run.dir=$HYDRA_RUN_ROOT/train_${SELECTED_VARIANT}" \
    "seed=-1" \
    "wandb_disable=true" \
    "++resume.allow_partial_converter_load=true" \
    "++resume.restore_gaussian_optimizer_state=false" \
    "++resume.restore_converter_optimizer_state=false" \
    "++resume.restore_converter_scheduler_state=false" \
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
    "${COMMON_IDENTITY_ARGS[@]}" \
    "++model.deformer.rigid.identity_render_enable=$SELECTED_IDENTITY_ENABLE" \
    "++model.deformer.rigid.identity_render_fwd_transform_mode=$SELECTED_FWD_MODE" \
    "++model.deformer.rigid.identity_render_rotation_mode=$SELECTED_ROTATION_MODE" \
    "opt.iterations=$ITERATIONS" \
    "opt.position_lr_init=0.0" \
    "opt.position_lr_final=0.0" \
    "opt.feature_lr=$FEATURE_LR" \
    "opt.opacity_lr=0.0" \
    "opt.scaling_lr=0.0" \
    "opt.rotation_lr=0.0" \
    "opt.rigid_lr=0.0" \
    "opt.non_rigid_lr=0.0" \
    "opt.nr_latent_lr=0.0" \
    "opt.pose_correction_lr=0.0" \
    "opt.texture_lr=$TEXTURE_LR" \
    "opt.tex_latent_lr=0.0" \
    "++opt.camera_affine_enable=false" \
    "++opt.camera_affine_lr=0.0" \
    "++opt.camera_geometry_enable=true" \
    "++opt.camera_geometry_lr=0.0" \
    "++opt.boundary_opacity_residual_lr=0.0" \
    "++opt.boundary_scaling_residual_lr=0.0" \
    "++opt.boundary_cov_residual_lr=0.0" \
    "++opt.binding_layer_logits_lr=0.0" \
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
    "opt.lambda_l1_fg=0.120" \
    "opt.lambda_l1_boundary=0.060" \
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
    > "$LOG_DIR/train_${SELECTED_VARIANT}.log" 2>&1
  log_event "train_done" "$TRAIN_EXP"

  IFS=',' read -ra steps <<< "$CHECKPOINT_STEPS"
  for step in "${steps[@]}"; do
    global_iter=$((BASE_ITER + step))
    ckpt="$TRAIN_EXP/ckpt${global_iter}.pth"
    variant="train_${SELECTED_VARIANT}_ckpt${global_iter}"
    if [ -f "$ckpt" ]; then
      run_render_eval "$variant" "$ckpt" "$SELECTED_IDENTITY_ENABLE" "$SELECTED_FWD_MODE" "$SELECTED_ROTATION_MODE" "$TRAIN_EXP/.hydra"
    else
      log_event "train_render_skip" "missing=$ckpt"
    fi
  done
fi

"$PYTHON_BIN" - "$TRAIN_SUMMARY" "$EXP_ROOT" "$SELECTED_ENV" "$BASE_ITER" "$CHECKPOINT_STEPS" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
exp_root = Path(sys.argv[2])
selected_env = Path(sys.argv[3])
base_iter = int(sys.argv[4])
steps = [int(x) for x in sys.argv[5].split(",") if x.strip()]
env = {}
if selected_env.exists():
    for line in selected_env.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
selected = env.get("SELECTED_VARIANT", "")
baseline_path = exp_root / "baseline_no_identity"

def load_metrics(render_exp):
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

baseline = load_metrics(baseline_path)
header = [
    "label", "variant", "render_exp", "fg", "boundary", "edge", "inner", "outer", "hard",
    "fg_delta_base", "boundary_delta_base", "edge_delta_base",
    "inner_delta_base", "outer_delta_base", "hard_delta_base",
    "strict_pass", "probe_pass", "status",
]
rows = []
if selected and baseline is not None:
    for step in steps:
        global_iter = base_iter + step
        name = f"train_{selected}_ckpt{global_iter}"
        metrics = load_metrics(exp_root / name)
        if metrics is None:
            continue
        delta = {key: metrics[key] - baseline[key] for key in ("fg", "boundary", "edge", "inner", "outer", "hard")}
        strict = (
            delta["inner"] < -0.05
            and delta["outer"] <= 0.0
            and delta["fg"] <= 0.0
            and delta["boundary"] <= 0.0
            and delta["edge"] <= 0.0
            and delta["hard"] < -0.000001
        )
        probe = (
            delta["inner"] <= 0.0
            and delta["hard"] < -0.000001
            and delta["outer"] <= 1.0
            and delta["fg"] <= 0.000050
            and delta["boundary"] <= 0.000050
            and delta["edge"] <= 0.005000
        )
        status = "strict_pass" if strict else ("probe_pass" if probe else "rejected")
        rows.append([
            f"ckpt{global_iter}", selected, metrics["render_exp"],
            f"{metrics['fg']:.8f}", f"{metrics['boundary']:.8f}", f"{metrics['edge']:.6f}",
            f"{metrics['inner']:.4f}", f"{metrics['outer']:.4f}", f"{metrics['hard']:.8f}",
            f"{delta['fg']:.8f}", f"{delta['boundary']:.8f}", f"{delta['edge']:.6f}",
            f"{delta['inner']:.4f}", f"{delta['outer']:.4f}", f"{delta['hard']:.8f}",
            str(bool(strict)).lower(), str(bool(probe)).lower(), status,
        ])

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(header)
    writer.writerows(rows)
PY

END_EPOCH="$(date +%s)"
END_BJT="$(TZ=Asia/Shanghai date -d "@$END_EPOCH" '+%F %T BJT')"
{
  echo "END_BJT=$END_BJT"
  echo "NO_TRAIN_SUMMARY=$NO_TRAIN_SUMMARY"
  echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
  echo "SELECTED_ENV=$SELECTED_ENV"
} >> "$LOG_DIR/run_info.txt"

write_status "all_done" "$END_BJT"
log_event "all_done" "$END_BJT"
echo "EXP_ROOT=$EXP_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "NO_TRAIN_SUMMARY=$NO_TRAIN_SUMMARY"
echo "TRAIN_SUMMARY=$TRAIN_SUMMARY"
echo "SELECTED_ENV=$SELECTED_ENV"
echo "EST_END_BJT=$EST_END_BJT"
echo "END_BJT=$END_BJT"
