from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OPTION = ROOT / (
    "configs/option/"
    "stageA_377_multiview_explicit_hq_fromzero_surface_responsibility_v2.yaml"
)
LAUNCHER = ROOT / "tools/run_coreview377_surface_responsibility_v2.sh"


def test_option_uses_passive_all_joint_surface_shell():
    cfg = yaml.safe_load(OPTION.read_text())
    opt = cfg["opt"]

    assert opt["lambda_surface_carrier_competition"] == 0.0
    assert opt["residual_surface_reallocation_enable"] is False
    assert opt["lambda_local_anchor_tether"] == [
        0.0,
        200,
        0.001,
        1000,
        0.004,
        12000,
        0.004,
    ]
    assert opt["local_anchor_tether_joint_ids"] == list(range(24))
    assert opt["local_anchor_tether_normal_limit"] == 0.055
    assert opt["local_anchor_tether_tangent_limit"] == 0.15
    assert opt["local_anchor_tether_normal_weight"] == 1.0
    assert opt["local_anchor_tether_tangent_weight"] == 0.02
    assert opt["local_anchor_tether_metric_gate_enable"] is False


def test_launcher_is_12k_fromzero_surface_area_canary():
    source = LAUNCHER.read_text()

    assert 'TRAIN_ITERS="${TRAIN_ITERS:-12000}"' in source
    assert 'CONVERTER_LR_MAX_STEPS="${CONVERTER_LR_MAX_STEPS:-64000}"' in source
    assert "stageA_377_multiview_explicit_hq_fromzero_surface_responsibility_v2" in source
    assert "dataset.init_point_count=80000" in source
    assert "dataset.init_sampling_mode=surface_carrier_v1" in source
    assert "dataset.init_surface_seed=$SEED" in source
    assert "dataset.init_surface_head_fraction=0.10" in source
    assert "dataset.init_surface_shoulder_fraction=0.22" in source
    assert "dataset.init_surface_hand_fraction=0.08" in source
    assert "dataset.init_surface_tangent_scale_factor=1.8" in source
    assert "dataset.init_surface_normal_scale_ratio=0.25" in source
    assert 'TESTS="[1000,2000,3000,4000,5000,6500,8000,10000,12000]"' in source
    assert 'SAVES="[5000,8000,12000]"' in source
    assert 'DIAGNOSTIC_ITERS="5000 8000 12000"' in source
    assert 'SMOKE="${SMOKE:-0}"' in source
    assert 'RUN="${RUN:-' in source

    assert "start_checkpoint" not in source
    assert "load_ckpt" not in source
    assert "focus_head_hands" not in source
    assert "activation_start" not in source
    assert "activation_end" not in source


def test_launcher_keeps_topology_frozen_and_overrides_base_focus_last():
    source = LAUNCHER.read_text()

    assert "opt.densify_until_iter=0" in source
    assert "opt.densify_from_iter=1000000" in source
    assert "opt.densification_interval=1000000" in source
    assert "opt.opacity_reset_interval=1000000" in source
    assert "opt.percent_dense=0.0" in source
    assert "opt.densify_prune_min_points=80000" in source
    assert "EXTRA_OVERRIDES=" in source
    assert "export EXTRA_OVERRIDES" in source
    assert "exec \"$BASE_LAUNCHER\"" in source
