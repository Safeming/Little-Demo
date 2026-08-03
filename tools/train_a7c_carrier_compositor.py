#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.a7c_oracle_capacity import _artifact_fingerprint, load_teacher_artifact
from utils.a7c_renderer_compositor import (
    BoundedCarrierMLP,
    build_canary_splits,
    fit_feature_normalization,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the frozen A7c carrier MLP canary.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _load_probe(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if _artifact_fingerprint(arrays) != str(arrays["output_fingerprint"]):
        raise ValueError("probe artifact fingerprint mismatch")
    return arrays


def _train_one(name, train_mask, *, features, teacher, camera_index, frame_index, contract, output, device):
    stats = fit_feature_normalization(features, sample_mask=train_mask)
    normalized = (features.astype(np.float32) - stats["mean"]) / stats["scale"]
    x = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(teacher.astype(np.float32)).to(device)
    sample_mask = torch.from_numpy(train_mask).to(device)
    adjacent_mask_np = (
        train_mask[1:]
        & train_mask[:-1]
        & (np.asarray(camera_index)[1:] == np.asarray(camera_index)[:-1])
        & (np.asarray(frame_index)[1:] - np.asarray(frame_index)[:-1] == int(contract["frame_stride"]))
    )
    adjacent_mask = torch.from_numpy(adjacent_mask_np).to(device)
    torch.manual_seed(int(contract["random_seed"]))
    model = BoundedCarrierMLP(
        x.shape[-1], contract["hidden_dimensions"],
        minimum_gate=contract["minimum_gate"],
        maximum_gate=contract["maximum_gate"],
        initial_gate=contract["initial_minimum_gate"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=contract["learning_rate"], weight_decay=contract["weight_decay"]
    )
    losses = []
    for epoch in range(int(contract["training_epochs"])):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x.reshape(-1, x.shape[-1])).reshape(y.shape)
        teacher_loss = F.smooth_l1_loss(
            prediction[sample_mask], y[sample_mask], beta=float(contract["huber_delta"])
        )
        perturb = x[sample_mask] + 0.01 * torch.randn_like(x[sample_mask])
        perturb_loss = torch.mean(torch.square(model(perturb.reshape(-1, x.shape[-1])).reshape(perturb.shape[:-1]) - prediction[sample_mask]))
        adjacent = torch.mean(torch.abs(prediction[1:][adjacent_mask] - prediction[:-1][adjacent_mask]))
        loss = teacher_loss + float(contract["adjacent_gate_penalty"]) * adjacent + float(contract["feature_perturbation_penalty"]) * perturb_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        prediction = model(x.reshape(-1, x.shape[-1])).reshape(y.shape).cpu().numpy().astype(np.float32)
    root = output / name
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_mean": stats["mean"],
            "feature_scale": stats["scale"],
            "contract": contract,
            "paper_test_eligible": False,
        },
        root / "model.pt",
    )
    np.savez_compressed(root / "predictions.npz", gates=prediction, train_mask=train_mask)
    summary = {
        "name": name,
        "training_sample_count": int(np.count_nonzero(train_mask)),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_gate": float(prediction.min()),
        "maximum_gate": float(prediction.max()),
        "paper_test_eligible": False,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)


def main(argv=None):
    args = parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    probe = _load_probe(args.probe)
    teacher = load_teacher_artifact(args.teacher)
    for key in ("carrier_ids", "camera_index", "frame_index"):
        if not np.array_equal(probe[key], teacher[key]):
            raise ValueError(f"probe and teacher {key} differ")
    if str(probe["source_teacher_fingerprint"]) != str(teacher["output_fingerprint"]):
        raise ValueError("probe source teacher fingerprint differs")
    features = np.asarray(probe["features"], dtype=np.float32)
    gates = np.asarray(teacher["gates"], dtype=np.float32)
    split = build_canary_splits(
        camera_index=probe["camera_index"], frame_index=probe["frame_index"],
        fit_camera_indices=(0, 1, 2, 3), audit_camera_indices=(4, 5, 6, 7),
        block_count=contract["temporal_block_count"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold, held in enumerate(split["held_block_masks"]):
        _train_one(
            f"fold_{fold}", split["fit_mask"] & ~held,
            features=features, teacher=gates, contract=contract,
            camera_index=probe["camera_index"], frame_index=probe["frame_index"],
            output=args.output_dir, device=args.device,
        )
    _train_one(
        "final", split["fit_mask"], features=features, teacher=gates,
        camera_index=probe["camera_index"], frame_index=probe["frame_index"],
        contract=contract, output=args.output_dir, device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
