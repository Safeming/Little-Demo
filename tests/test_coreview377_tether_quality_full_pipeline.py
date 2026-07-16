from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/run_coreview377_tether_quality_full_pipeline.sh"
TETHER_LAUNCHER = ROOT / (
    "exp/zero_train_to_v395/coreview377_surface_coherent_anchor_tether_20260711_bjt/"
    "launch_surface_coherent_anchor_tether.sh"
)


def test_launcher_chains_canary_and_conditional_full_fromzero():
    source = LAUNCHER.read_text()

    assert "CANARY_TETHER_CKPT" in source
    assert "ckpt64000.pth" in source
    assert "run_continuation_tail" in source
    assert "select_tether_quality_candidate.py" in source
    assert "CANARY_REJECTED" in source
    assert "FULL_FROMZERO_START" in source
    assert "STOP_AFTER_CANARY" in source
    assert "launch_surface_coherent_anchor_tether.sh" in source
    assert "PIPELINE_DONE_BJT" in source


def test_raw_validation_disables_heldout_camera_corrections():
    source = LAUNCHER.read_text()

    assert "shopt -s inherit_errexit" in source
    assert "opt.camera_geometry_enable=false" in source
    assert "opt.camera_affine_enable=false" in source
    assert "++resume.restore_gaussian_optimizer_state=false" in source
    assert "++resume.restore_converter_optimizer_state=false" in source
    assert "++resume.restore_converter_scheduler_state=false" in source
    assert source.count(
        "++resume.partial_converter_missing_keys_allow_patterns="
        "[texture.structured_trunk_,deformer.rigid.forward_trunk_mlp.,"
        "texture.shadow_handoff_approved]"
    ) >= 2
    assert "dataset.test_views.view=[21,22,23]" in source
    assert "dataset.train_views=[21,22,23]" not in source


def test_full_tail_uses_only_checkpoints_from_the_new_full_run():
    source = LAUNCHER.read_text()

    assert 'FULL_STAGE1_CKPT="$FULL_RUN/stage1/neutral_longhorizon_fromzero/ckpt64000.pth"' in source
    assert 'run_continuation_tail "full" "$FULL_STAGE1_CKPT" "$FULL_RUN"' in source
    checkpoint_assignments = [
        line for line in source.splitlines() if "_CKPT=" in line and not line.lstrip().startswith("#")
    ]
    assert checkpoint_assignments
    assert all("v395" not in line.lower() for line in checkpoint_assignments)


def test_launcher_has_smoke_step_overrides_and_result_artifacts():
    source = LAUNCHER.read_text()

    assert 'SMOKE="${SMOKE:-0}"' in source
    assert "CONTINUATION_ITERS" in source
    assert "LATE_CLEAN_ITERS" in source
    assert "RESIDUAL_ITERS" in source
    assert "CANARY_RESULT.json" in source
    assert "FULL_RESULT.json" in source
    assert "FINAL_BEST_CKPT.txt" in source


def test_existing_tether_launcher_accepts_run_directory_override():
    source = TETHER_LAUNCHER.read_text()

    assert 'BASE_RUN="${BASE_RUN:-' in source
