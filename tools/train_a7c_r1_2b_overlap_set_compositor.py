#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_a7c_r1_2a_quotient_compositor import (
    _build_streams,
    _load_probe,
    _load_teacher_manifest,
    _torch_streams,
    sample_block_ids,
    verify_source_file,
)
from utils.a7c_overlap_set_compositor import DenseOverlapSetCompositor
from utils.a7c_quotient_compositor import (
    contiguous_training_segments,
    project_joint_target_budget,
    renderer_sequence_objective,
    runtime_target_mass,
)
from utils.a7c_ray_context_probe import select_feature_group
from utils.a7c_renderer_compositor import (
    build_canary_splits,
    fit_feature_normalization,
)
from utils.part_label_bank import PART_NAMES, load_part_label_bank


def train_one(
    *,
    name: str,
    train_mask,
    features,
    projected_xy,
    log_depth,
    visibility,
    runtime_mass,
    a5_weight,
    objective_streams,
    guard_streams,
    camera_index,
    frame_index,
    block_ids,
    contract,
    output_dir: Path,
    device: str,
) -> dict:
    values = np.asarray(features, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if values.ndim != 3 or mask.shape != (values.shape[0],):
        raise ValueError("features and train mask are not aligned")
    samples, carriers = values.shape[:2]
    xy = np.asarray(projected_xy, dtype=np.float32)
    depth = np.asarray(log_depth, dtype=np.float32)
    visible = np.asarray(visibility, dtype=np.float32)
    if xy.shape != (samples, carriers, 2):
        raise ValueError("projected_xy is not aligned")
    if depth.shape != (samples, carriers) or visible.shape != depth.shape:
        raise ValueError("depth and visibility are not aligned")

    stats = fit_feature_normalization(values, sample_mask=mask)
    normalized = (values - stats["mean"]) / stats["scale"]
    x = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    xy_tensor = torch.as_tensor(xy, dtype=torch.float32, device=device)
    depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=device)
    visibility_tensor = torch.as_tensor(
        visible, dtype=torch.float32, device=device
    )
    mass = torch.as_tensor(
        np.asarray(runtime_mass, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    weights = torch.as_tensor(
        np.asarray(a5_weight, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    streams_objective = _torch_streams(objective_streams, device)
    streams_guard = _torch_streams(guard_streams, device)
    segments = contiguous_training_segments(
        train_mask=mask,
        camera_index=camera_index,
        frame_index=frame_index,
        frame_stride=int(contract["frame_stride"]),
        block_ids=block_ids,
    )

    torch.manual_seed(int(contract["random_seed"]))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(contract["random_seed"]))
    model = DenseOverlapSetCompositor(
        values.shape[-1],
        int(contract["node_hidden_dimension"]),
        int(contract["gate_hidden_dimension"]),
        minimum_gate=float(contract["minimum_gate"]),
        maximum_gate=float(contract["maximum_gate"]),
        initial_gate=float(contract["initial_minimum_gate"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
    )

    def forward():
        raw = model(
            x,
            xy_tensor,
            depth_tensor,
            visibility_tensor,
            spatial_scale=float(contract["spatial_scale"]),
            depth_scale=float(contract["depth_scale"]),
            edge_log_weight_minimum=float(
                contract["edge_log_weight_minimum"]
            ),
        )
        projected = project_joint_target_budget(
            raw_gates=raw,
            runtime_mass=mass,
            a5_weight=weights,
            proxy_target_response=float(contract["proxy_target_response"]),
            selection_threshold=float(contract["selection_threshold"]),
            minimum_gate=float(contract["minimum_gate"]),
        )
        components = renderer_sequence_objective(
            gates=projected,
            segments=segments,
            objective_streams=streams_objective,
            guard_streams=streams_guard,
            contract=contract,
        )
        return raw, projected, components

    losses = []
    first_components = None
    for _ in range(int(contract["training_epochs"])):
        optimizer.zero_grad(set_to_none=True)
        _, _, components = forward()
        if first_components is None:
            first_components = {
                key: float(value.detach().cpu())
                for key, value in components.items()
            }
        components["loss"].backward()
        optimizer.step()
        losses.append(float(components["loss"].detach().cpu()))
    with torch.no_grad():
        raw, projected, final_tensor = forward()
    final_components = {
        key: float(value.detach().cpu()) for key, value in final_tensor.items()
    }

    root = Path(output_dir) / name
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": stats["mean"],
            "feature_scale": stats["scale"],
            "contract": dict(contract),
            "predictor": "dense_overlap_set",
            "paper_test_eligible": False,
        },
        root / "model.pt",
    )
    np.savez_compressed(
        root / "predictions.npz",
        raw_gates=raw.cpu().numpy().astype(np.float32),
        projected_gates=projected.cpu().numpy().astype(np.float32),
        train_mask=mask,
        paper_test_eligible=np.array(0, dtype=np.uint8),
    )
    summary = {
        "name": name,
        "predictor": "dense_overlap_set",
        "training_sample_count": int(np.count_nonzero(mask)),
        "segment_count": len(segments),
        "initial_loss": float(first_components["loss"]),
        "final_loss": float(final_components["loss"]),
        "initial_components": first_components,
        "final_components": final_components,
        "raw_minimum_gate": float(raw.min().cpu()),
        "raw_maximum_gate": float(raw.max().cpu()),
        "projected_minimum_gate": float(projected.min().cpu()),
        "projected_maximum_gate": float(projected.max().cpu()),
        "teacher_gate_loss_weight": float(contract["teacher_gate_loss_weight"]),
        "paper_test_eligible": False,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the frozen A7c R1.2-B overlap-set compositor."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--a5-bank", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    source_specs = (
        (args.probe, contract["source_probe_sha256"], "probe"),
        (args.evidence, contract["source_evidence_sha256"], "evidence"),
        (args.a5_bank, contract["source_a5_bank_sha256"], "A5 bank"),
        (args.teacher, contract["source_teacher_sha256"], "teacher"),
        (
            REPO_ROOT / contract["source_r1_2b_design"],
            contract["source_r1_2b_design_sha256"],
            "R1.2-B design",
        ),
        (
            REPO_ROOT / contract["source_r1_2a_contract"],
            contract["source_r1_2a_contract_sha256"],
            "R1.2-A contract",
        ),
    )
    source_fingerprints = {
        name: verify_source_file(path, expected, name)
        for path, expected, name in source_specs
    }
    r1_contract_path = REPO_ROOT / contract["source_r1_1_contract"]
    source_fingerprints["R1.1 contract"] = verify_source_file(
        r1_contract_path,
        contract["source_r1_1_contract_sha256"],
        "R1.1 contract",
    )
    r1_contract = json.loads(r1_contract_path.read_text(encoding="utf-8"))
    probe = _load_probe(args.probe)
    teacher = _load_teacher_manifest(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"probe and teacher {key} differ")
    if str(probe["source_teacher_fingerprint"]) != str(
        teacher["output_fingerprint"]
    ):
        raise ValueError("probe source teacher fingerprint differs")
    with np.load(args.evidence, allow_pickle=False) as source:
        evidence = {key: source[key] for key in source.files}
    if not np.array_equal(
        evidence["renderer_sequence_camera_index"], teacher["camera_index"]
    ) or not np.array_equal(
        evidence["renderer_sequence_frame_index"], teacher["frame_index"]
    ):
        raise ValueError("evidence and teacher sample manifest differ")

    bank = load_part_label_bank(args.a5_bank)
    part_index = PART_NAMES.index(str(contract["part"]))
    all_weights = np.asarray(
        bank["soft_edit_weights"], dtype=np.float32
    )[:, part_index]
    carrier_ids = np.asarray(teacher["carrier_ids"], dtype=np.int64)
    a5_weight = all_weights[carrier_ids]
    all_features = np.asarray(probe["features"], dtype=np.float32)
    feature_names = list(map(str, probe["feature_names"]))
    features = select_feature_group(
        all_features,
        feature_names,
        r1_contract["feature_groups"][contract["score_feature_group"]],
    )
    field = {
        name: all_features[:, :, feature_names.index(name)]
        for name in (
            "visibility",
            "camera_x_over_z",
            "camera_y_over_z",
            "log_depth",
            "alpha_transmittance_mass",
            "semantic_support_mean",
            "alpha_mean",
        )
    }
    projected_xy = np.stack(
        (field["camera_x_over_z"], field["camera_y_over_z"]), axis=-1
    )
    runtime_mass_values = runtime_target_mass(
        alpha_transmittance_mass=torch.from_numpy(
            field["alpha_transmittance_mass"]
        ),
        a5_weight=torch.from_numpy(a5_weight),
        semantic_support_mean=torch.from_numpy(field["semantic_support_mean"]),
        alpha_mean=torch.from_numpy(field["alpha_mean"]),
    ).numpy()
    streams = _build_streams(evidence, all_weights, carrier_ids, part_index)
    camera_index = np.asarray(teacher["camera_index"])
    frame_index = np.asarray(teacher["frame_index"])
    blocks = sample_block_ids(
        camera_index, frame_index, int(contract["temporal_block_count"])
    )
    split = build_canary_splits(
        camera_index=camera_index,
        frame_index=frame_index,
        fit_camera_indices=(0, 1, 2, 3),
        audit_camera_indices=(4, 5, 6, 7),
        block_count=int(contract["temporal_block_count"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "features": features,
        "projected_xy": projected_xy,
        "log_depth": field["log_depth"],
        "visibility": field["visibility"],
        "runtime_mass": runtime_mass_values,
        "a5_weight": a5_weight,
        "objective_streams": streams["objective"],
        "guard_streams": streams["guard"],
        "camera_index": camera_index,
        "frame_index": frame_index,
        "block_ids": blocks,
        "contract": contract,
        "output_dir": args.output_dir,
        "device": args.device,
    }
    summaries = []
    for fold, held in enumerate(split["held_block_masks"]):
        summaries.append(
            train_one(
                name=f"fold_{fold}",
                train_mask=split["fit_mask"] & ~held,
                **common,
            )
        )
    summaries.append(
        train_one(
            name="final", train_mask=split["fit_mask"], **common
        )
    )
    payload = {
        "experiment_id": contract["experiment_id"],
        "predictor": "dense_overlap_set",
        "models": summaries,
        "source_fingerprints": source_fingerprints,
        "teacher_gate_values_accessed": False,
        "paper_test_eligible": False,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
