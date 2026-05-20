import csv
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


DISABLED_PRESETS = {"", "none", "null", "false", "off", "0"}
SUPPORTED_PRESETS = {
    "adopted_geometry",
    "v306_adopted_geometry",
    "v307_adopted_geometry",
    "v308_binding_internal",
    "binding_internal",
    "v313_learned_xbar",
    "learned_xbar",
    "stageb_adopted_geometry",
    "v320_v307_signed_geometry",
    "v320_selected_geometry",
    "stageb_signed_geometry",
}
V320_PRESETS = {
    "v320_v307_signed_geometry",
    "v320_selected_geometry",
    "stageb_signed_geometry",
}
FORMAL_SUBJECT = "CoreView_377"
FORMAL_ASSET_SUBDIR = ("assets", "adopted_geometry", "377")
COMPONENT_COLUMNS = {
    "image_name",
    "direction",
    "area",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "near_score_sum",
    "top_point_ids",
}
POINT_COLUMNS = {
    "point_idx",
    "layer_id",
    "region_id",
    "dominant_joint",
    "surface_distance",
    "thin_score",
    "boundary_score",
}


def apply_explicit_binding_render_preset(config, repo_root=None):
    preset = str(config.get("explicit_binding_render_preset", "") or "").strip().lower()
    if preset in DISABLED_PRESETS:
        return None
    if preset not in SUPPORTED_PRESETS:
        raise ValueError(f"Unsupported explicit_binding_render_preset: {preset}")

    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    binding_internal = preset in ("v308_binding_internal", "binding_internal")
    learned_xbar = preset in ("v313_learned_xbar", "learned_xbar")
    v320_selected_geometry = preset in V320_PRESETS

    defaults = _formal_asset_defaults(repo_root) if v320_selected_geometry else {}
    pipeline = config.get("pipeline", OmegaConf.create({}))
    rigid = (
        config.get("model", OmegaConf.create({}))
        .get("deformer", OmegaConf.create({}))
        .get("rigid", OmegaConf.create({}))
    )

    component_csv = _config_path(
        config,
        "explicit_binding_adopted_component_csv",
        default=str(
            pipeline.get("covariance_signed_dynamic_component_csv", "")
            or rigid.get("geometry_fidelity_component_csv", "")
            or defaults.get("component_csv", "")
        ),
    )
    point_csv = _config_path(
        config,
        "explicit_binding_adopted_point_csv",
        default=str(pipeline.get("covariance_signed_dynamic_point_csv", "") or defaults.get("point_csv", "")),
    )

    require_csv = _as_bool(config.get("explicit_binding_adopted_require_component_csv", True))
    if require_csv and not component_csv:
        raise ValueError("explicit_binding_render_preset requires explicit_binding_adopted_component_csv")
    if require_csv and component_csv and not Path(component_csv).exists():
        raise FileNotFoundError(f"explicit_binding_adopted_component_csv not found: {component_csv}")
    if point_csv and not Path(point_csv).exists():
        raise FileNotFoundError(f"explicit_binding_adopted_point_csv not found: {point_csv}")

    validation = {}
    uses_formal_defaults = (
        v320_selected_geometry
        and _same_path(component_csv, defaults.get("component_csv", ""))
        and _same_path(point_csv, defaults.get("point_csv", ""))
    )
    if uses_formal_defaults:
        validation = _validate_formal_377_asset(config, repo_root, component_csv, point_csv)

    center_strength = float(config.get("explicit_binding_adopted_center_strength", 0.45))
    outer_px = float(config.get("explicit_binding_adopted_outer_px", 0.35))
    component_required = _as_bool(config.get("explicit_binding_adopted_component_required", not binding_internal))
    improvement_guard = _as_bool(config.get("explicit_binding_adopted_improvement_guard", True))
    max_points = int(config.get("explicit_binding_adopted_max_points", 96))
    component_csv_value = "" if (binding_internal or learned_xbar) else component_csv
    point_csv_value = "" if (binding_internal or learned_xbar) else point_csv

    overrides = _formal_overrides(
        preset=preset,
        learned_xbar=learned_xbar,
        binding_internal=binding_internal,
        component_csv_value=component_csv_value,
        point_csv_value=point_csv_value,
        component_required=component_required,
        improvement_guard=improvement_guard,
        max_points=max_points,
        center_strength=center_strength,
        outer_px=outer_px,
    )
    overrides.update({
        "explicit_binding_adopted_asset_validation": validation,
    })
    for key, value in overrides.items():
        _set_nested_config(config, key, value)
    return validation


def _formal_asset_defaults(repo_root):
    asset_dir = repo_root.joinpath(*FORMAL_ASSET_SUBDIR)
    return {
        "manifest": str(asset_dir / "manifest.json"),
        "component_csv": str(asset_dir / "v320_selected_components.csv"),
        "point_csv": str(asset_dir / "v304_point_contributors_all.csv"),
    }


def _validate_formal_377_asset(config, repo_root, component_csv, point_csv):
    dataset = config.get("dataset", OmegaConf.create({}))
    subject = str(dataset.get("subject", "") or "")
    allow_mismatch = _as_bool(config.get("explicit_binding_adopted_allow_subject_mismatch", False))
    if subject and subject != FORMAL_SUBJECT and not allow_mismatch:
        raise ValueError(
            f"v320_v307_signed_geometry formal asset is validated only for {FORMAL_SUBJECT}; "
            f"got dataset.subject={subject}. Pass explicit asset CSVs or set "
            "explicit_binding_adopted_allow_subject_mismatch=true for an intentional override."
        )

    defaults = _formal_asset_defaults(repo_root)
    manifest_path = Path(defaults["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"formal adopted geometry manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("subject", "")) != FORMAL_SUBJECT:
        raise ValueError(f"formal adopted geometry manifest subject mismatch: {manifest.get('subject')}")

    _validate_csv_schema(component_csv, COMPONENT_COLUMNS, "component_csv")
    _validate_csv_schema(point_csv, POINT_COLUMNS, "point_csv")
    component_rows = _csv_row_count(component_csv)
    point_rows = _csv_row_count(point_csv)
    expected_component_rows = int(manifest.get("component_rows", -1))
    expected_point_rows = int(manifest.get("point_rows", -1))
    if component_rows != expected_component_rows:
        raise ValueError(f"component_csv row count mismatch: got {component_rows}, expected {expected_component_rows}")
    if point_rows != expected_point_rows:
        raise ValueError(f"point_csv row count mismatch: got {point_rows}, expected {expected_point_rows}")

    expected_hash = manifest.get("sha256", {})
    component_hash = _sha256(component_csv)
    point_hash = _sha256(point_csv)
    if component_hash != expected_hash.get("component_csv"):
        raise ValueError("component_csv sha256 mismatch for formal adopted geometry asset")
    if point_hash != expected_hash.get("point_csv"):
        raise ValueError("point_csv sha256 mismatch for formal adopted geometry asset")

    return {
        "subject": FORMAL_SUBJECT,
        "manifest": str(manifest_path),
        "component_csv_sha256": component_hash,
        "point_csv_sha256": point_hash,
        "component_rows": component_rows,
        "point_rows": point_rows,
        "status": "validated",
    }


def _formal_overrides(
    *,
    preset,
    learned_xbar,
    binding_internal,
    component_csv_value,
    point_csv_value,
    component_required,
    improvement_guard,
    max_points,
    center_strength,
    outer_px,
):
    return {
        "pipeline.compute_cov3D_python": True,
        "pipeline.covariance_mode": "default",
        "pipeline.covariance_signed_dynamic_enable": not learned_xbar,
        "pipeline.covariance_signed_dynamic_component_csv": component_csv_value,
        "pipeline.covariance_signed_dynamic_point_csv": point_csv_value,
        "pipeline.covariance_signed_dynamic_component_signature_enable": False,
        "pipeline.covariance_signed_dynamic_over_layer_ids": "soft,free",
        "pipeline.covariance_signed_dynamic_over_region_ids": "cloth",
        "pipeline.covariance_signed_dynamic_over_joint_ids": "6,9,12,13,14,15",
        "pipeline.covariance_signed_dynamic_under_layer_ids": "soft,rigid,free",
        "pipeline.covariance_signed_dynamic_under_region_ids": "cloth,body,soft",
        "pipeline.covariance_signed_dynamic_under_joint_ids": "0,1,2,4,7,8,10",
        "pipeline.covariance_signed_dynamic_boundary_min": 0.0,
        "pipeline.covariance_signed_dynamic_component_pad_px": 10,
        "pipeline.covariance_signed_dynamic_component_ellipse_scale": 1.25,
        "pipeline.covariance_signed_dynamic_component_max_over": 16,
        "pipeline.covariance_signed_dynamic_component_max_under": 16,
        "pipeline.covariance_signed_dynamic_component_min_area": 20,
        "pipeline.covariance_signed_dynamic_component_required": component_required,
        "pipeline.covariance_signed_dynamic_component_top_ids_enable": False,
        "pipeline.covariance_signed_dynamic_component_top_ids_only": False,
        "pipeline.covariance_signed_dynamic_max_over_points": max_points,
        "pipeline.covariance_signed_dynamic_max_under_points": max_points,
        "pipeline.covariance_signed_screen_actuator_enable": not learned_xbar,
        "pipeline.covariance_signed_screen_normal_shrink_factor": 0.940,
        "pipeline.covariance_signed_screen_normal_grow_factor": 1.025,
        "pipeline.covariance_signed_screen_tangent_factor": 1.000,
        "pipeline.covariance_signed_center_offset_enable": not learned_xbar,
        "pipeline.covariance_signed_center_offset_outer_px": outer_px,
        "pipeline.covariance_signed_center_offset_inner_px": 0.0,
        "pipeline.covariance_signed_center_offset_outer_direction": "view_center",
        "pipeline.covariance_signed_center_offset_inner_direction": "component_center",
        "pipeline.covariance_signed_center_offset_score_weight_power": 1.0,
        "pipeline.covariance_signed_center_offset_score_weight_min": 0.15,
        "pipeline.covariance_signed_center_offset_score_weight_quantile": 0.90,
        "pipeline.covariance_signed_center_offset_jacobian_eps": 0.001,
        "pipeline.covariance_signed_center_offset_jacobian_damping": 0.00001,
        "pipeline.covariance_signed_center_offset_max_world_step": 0.0020,
        "pipeline.boundary_cov_residual_enable": False,
        "pipeline.binding_covariance_guard_enable": False,
        "model.deformer.rigid.rotation_orthogonalize_enable": False,
        "model.deformer.rigid.geometry_fidelity_gate_enable": not learned_xbar,
        "model.deformer.rigid.geometry_fidelity_target": "free_lbs",
        "model.deformer.rigid.geometry_fidelity_center_strength": center_strength,
        "model.deformer.rigid.geometry_fidelity_rotation_strength": 0.0,
        "model.deformer.rigid.geometry_fidelity_boundary_min": 0.12,
        "model.deformer.rigid.geometry_fidelity_layer_ids": "soft,free",
        "model.deformer.rigid.geometry_fidelity_region_ids": "cloth,soft",
        "model.deformer.rigid.geometry_fidelity_joint_ids": "",
        "model.deformer.rigid.geometry_fidelity_thin_min": "",
        "model.deformer.rigid.geometry_fidelity_surface_min": "",
        "model.deformer.rigid.geometry_fidelity_surface_max": "",
        "model.deformer.rigid.geometry_fidelity_non_rigid_min": 0.0,
        "model.deformer.rigid.geometry_fidelity_power": 1.2,
        "model.deformer.rigid.geometry_fidelity_max_points": 1024,
        "model.deformer.rigid.geometry_fidelity_component_enable": (not binding_internal and not learned_xbar),
        "model.deformer.rigid.geometry_fidelity_component_csv": component_csv_value,
        "model.deformer.rigid.geometry_fidelity_component_direction": "inner",
        "model.deformer.rigid.geometry_fidelity_component_pad_px": 2,
        "model.deformer.rigid.geometry_fidelity_component_ellipse_scale": 1.05,
        "model.deformer.rigid.geometry_fidelity_component_max": 12,
        "model.deformer.rigid.geometry_fidelity_component_min_area": 40,
        "model.deformer.rigid.geometry_fidelity_component_required": component_required,
        "model.deformer.rigid.geometry_fidelity_component_improvement_enable": improvement_guard,
        "model.deformer.rigid.geometry_fidelity_component_improvement_margin_px": 0.0,
        "opt.camera_geometry_enable": True,
        "opt.camera_geometry_lr": 0.0,
        "render_export_refine": False,
        "explicit_binding_render_preset_applied": preset,
        "explicit_binding_adopted_component_csv_resolved": component_csv_value,
        "explicit_binding_adopted_point_csv_resolved": point_csv_value,
    }


def _validate_csv_schema(path, required_columns, label):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _csv_row_count(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_nested_config(config, dotted_key, value):
    node = config
    parts = str(dotted_key).split(".")
    for part in parts[:-1]:
        child = node.get(part, None)
        if child is None:
            child = OmegaConf.create({})
            node[part] = child
        node = child
    node[parts[-1]] = value


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _config_path(config, *keys, default=""):
    for key in keys:
        value = config.get(key, None)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _same_path(left, right):
    if not left or not right:
        return False
    return Path(left).resolve() == Path(right).resolve()
