#!/usr/bin/env bash
set -euo pipefail

ROOT="/remote-home/ming/3dgs-avatar-release-main"
PYTHON_BIN="/opt/miniconda3/envs/anim/bin/python"
EXP_DIR="${1:-$ROOT/exp/stageA2/377_multiview_explicit_hq_rootfix_fresh_v159a_v157a_aplus_supportrelax_screen4k}"
ITERATIONS="${2:-4000}"
if [ "$#" -ge 2 ]; then
  shift 2
else
  shift "$#"
fi

mkdir -p "$EXP_DIR"
cd "$ROOT"
exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$ROOT/tools/run_377_stageA2_multiview_explicit.py" \
  --exp-dir "$EXP_DIR" \
  --non-rigid hashgrid \
  --texture shallow_mlp_lownoise \
  --option stageA_377_multiview_explicit_hq_v4 \
  --option stageA_377_multiview_explicit_hq_v4_fast \
  --option stageA_377_multiview_explicit_hq_v89a_fresh_clarity_mainline_safe_v1 \
  --option stageA_377_multiview_explicit_hq_v89e_fresh_clarity_mainline_decay0005_v1 \
  --option stageA_377_multiview_explicit_hq_v90f_fresh_decay0002_noise165_v1 \
  --option stageA_377_multiview_explicit_hq_v98b_v90f_detailinterior_safe_v1 \
  --option stageA_377_multiview_explicit_hq_v99a_v98b_faceboost_safe_v1 \
  --option stageA_377_multiview_explicit_hq_v100a_v99a_hfcarrier_safe_v1 \
  --option stageA_377_multiview_explicit_hq_v101a_v100a_hflumachroma_safe_v1 \
  --option stageA_377_multiview_explicit_hq_v103b_v101a_lumadog_patchlpips_v1 \
  --option stageA_377_multiview_explicit_hq_v1060_v103b_facebranch_warminit_base_v1 \
  --option stageA_377_multiview_explicit_hq_v107b_v103b_facewarmrgb_nopoint_balanced_v1 \
  --option stageA_377_multiview_explicit_hq_v1090_v108b_facelocal_base_v1 \
  --option stageA_377_multiview_explicit_hq_v109b_v108b_facelocal_rgb_push_v1 \
  --option stageA_377_multiview_explicit_hq_v1100_v109b_regionlocal_base_v1 \
  --option stageA_377_multiview_explicit_hq_v110b_v109b_regionlocal_torso_v1 \
  --option stageA_377_multiview_explicit_hq_v1130_v110b_nonrigidcarry_base_v1 \
  --option stageA_377_multiview_explicit_hq_v113a_v110b_nonrigidcarry_soft_v1 \
  --option stageA_377_multiview_explicit_hq_v1140_v113a_structureaware_base_v1 \
  --option stageA_377_multiview_explicit_hq_v114a_v113a_geomaware_safe_v1 \
  --option stageA_377_multiview_explicit_hq_v1150_v114a_structurebridge_base_v1 \
  --option stageA_377_multiview_explicit_hq_v115b_v114a_structurebridge_richer_v1 \
  --option stageA_377_multiview_explicit_hq_v1160_v115b_trunkmain_base_v1 \
  --option stageA_377_multiview_explicit_hq_v116b_v115b_trunkmain_richer_v1 \
  --option stageA_377_multiview_explicit_hq_v1170_v116b_trunkmlp_base_v1 \
  --option stageA_377_multiview_explicit_hq_v1220_v121a_trunkcolor_full_tinyrepair_v1 \
  --option stageA_377_multiview_explicit_hq_v123b_v122a_trunkregioncolor_gateopen_push_v1 \
  --option stageA_377_multiview_explicit_hq_v1240_v123b_trunklocalctx_v1 \
  --option stageA_377_multiview_explicit_hq_v124b_v124a_trunksoftgeom_v1 \
  --option stageA_377_multiview_explicit_hq_v1250_v124b_trunkdualhead_maincolor_v1 \
  --option stageA_377_multiview_explicit_hq_v126a_v125a_hfdecouple_geom_v1 \
  --option stageA_377_multiview_explicit_hq_v127a_v126a_trunkowner_full_strong_v1 \
  --option stageA_377_multiview_explicit_hq_v1310_v127a_trunkowner_supportstabilize_v1 \
  --option stageA_377_multiview_explicit_hq_v1320_v1310_trunkowner_localcolorowner_full_v1 \
  --option stageA_377_multiview_explicit_hq_v1330_v1320_ownerhf_complete_v1 \
  --option stageA_377_multiview_explicit_hq_v1340_v1330_ownertakeover_v1 \
  --option stageA_377_multiview_explicit_hq_v135a_v1340_ownerseedcut_v1 \
  --option stageA_377_multiview_explicit_hq_v136a_v135a_ownerdetailboost_v1 \
  --option stageA_377_multiview_explicit_hq_v151_v150a_priorguided_forwardtrunk_mainpath_xbar_v1 \
  --option stageA_377_multiview_explicit_hq_v152a_v151_geomlearn_probe_v1 \
  --option stageA_377_multiview_explicit_hq_v159a_v157a_aplus_supportrelax_v1 \
  --extra-override "opt.iterations=$ITERATIONS" \
  --extra-override "++validation_image_log_limit=0" \
  "$@"
