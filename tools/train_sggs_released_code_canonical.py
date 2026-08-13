#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gaussian_grouping_canonical import balanced_pixel_indices
from utils.part_label_bank import PART_NAMES, save_part_label_bank, summarize_part_label_bank
from utils.sggs_released_code_canonical import (
    Compact6Readout,
    compact6_predictions,
    topology_consistency_loss,
)

SGGS_RELEASE_HEAD = "27b9ed9c9e4c5663deb169247c2339ccafe1c254"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a compact-6 readout on frozen SG-GS topology-geometric features."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--samples-per-class", type=int, default=512)
    parser.add_argument("--topology-lambda", type=float, default=0.1)
    parser.add_argument("--topology-interval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--resume", default="none", help="none, auto, or an explicit checkpoint path")
    return parser


def validate_training_inputs(input_dir: Path, prior_dir: Path):
    input_dir = Path(input_dir)
    prior_dir = Path(prior_dir)
    source = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((prior_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(source.get("view_count", -1)) != 80 or len(source.get("views", [])) != 80:
        raise ValueError("expected exactly 80 frozen training views")
    if list(source.get("part_names", [])) != list(PART_NAMES):
        raise ValueError("frozen manifest part order does not match compact-6 protocol")
    if source.get("subject") != prior.get("subject"):
        raise ValueError("source and SG-GS prior subject mismatch")
    if int(prior.get("view_count", -1)) != 80:
        raise ValueError("SG-GS prior must record exactly 80 frozen views")
    if int(prior.get("feature_dim", -1)) != 32:
        raise ValueError("SG-GS prior feature dimension must be 32")
    if prior.get("sggs_head") != SGGS_RELEASE_HEAD:
        raise ValueError("SG-GS prior commit does not match the audited release")
    expected_input = str(input_dir.resolve())
    if str(Path(prior.get("source_frozen_views", "")).resolve()) != expected_input:
        raise ValueError("SG-GS prior frozen-view source mismatch")
    if prior.get("source_checkpoint_sha256") != source.get("source_checkpoint_sha256"):
        raise ValueError("source checkpoint hash mismatch")

    canonical_xyz = torch.load(input_dir / "canonical_xyz.pt", map_location="cpu")
    features = torch.load(prior_dir / "topology_features.pt", map_location="cpu")
    graph = torch.load(prior_dir / "topology_knn.pt", map_location="cpu")
    point_count = int(source.get("point_count", -1))
    if canonical_xyz.shape != (point_count, 3):
        raise ValueError("canonical point count does not match source manifest")
    if int(prior.get("point_count", -1)) != point_count or features.shape != (point_count, 32):
        raise ValueError("SG-GS prior point count or feature shape mismatch")
    indices = graph.get("indices")
    weights = graph.get("weights")
    if not torch.is_tensor(indices) or not torch.is_tensor(weights):
        raise ValueError("SG-GS topology graph must contain tensor indices and weights")
    if indices.ndim != 2 or indices.shape != weights.shape or indices.shape[0] != point_count:
        raise ValueError("SG-GS topology graph point count mismatch")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= point_count):
        raise ValueError("SG-GS topology graph contains invalid indices")
    if not torch.isfinite(features).all() or not torch.isfinite(weights).all():
        raise ValueError("SG-GS topology prior must be finite")
    view_paths = [input_dir / "views" / str(name) for name in source["views"]]
    missing = [str(path) for path in view_paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing frozen training views: {missing[:3]}")
    return source, prior, features.float(), graph, view_paths


def find_resume_checkpoint(output_dir: Path, *, iterations: int) -> Path | None:
    output_dir = Path(output_dir)
    if (output_dir / "COMPLETE").is_file():
        return None
    for path in reversed(sorted(output_dir.glob("checkpoint_*.pt"))):
        payload = torch.load(path, map_location="cpu")
        if int(payload.get("iteration", 0)) < int(iterations):
            return path
    return None


def _settings(view: dict, background: torch.Tensor):
    from diff_gaussian_rasterization_contrastive_f import GaussianRasterizationSettings

    return GaussianRasterizationSettings(
        image_height=int(view["height"]),
        image_width=int(view["width"]),
        tanfovx=math.tan(float(view["fovx"]) * 0.5),
        tanfovy=math.tan(float(view["fovy"]) * 0.5),
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view["viewmatrix"],
        projmatrix=view["projmatrix"],
        sh_degree=0,
        campos=view["camera_center"],
        prefiltered=False,
        debug=False,
    )


def _to_cuda(view: dict) -> dict:
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in view.items()}


def _render_logits(view: dict, point_logits: torch.Tensor) -> torch.Tensor:
    from diff_gaussian_rasterization_contrastive_f import GaussianRasterizer

    channels = torch.cat(
        (point_logits, torch.zeros((point_logits.shape[0], 26), dtype=point_logits.dtype, device="cuda")), dim=1
    )
    background = torch.zeros((32,), dtype=torch.float32, device="cuda")
    screenspace = torch.zeros_like(view["xyz"], requires_grad=True)
    rasterizer = GaussianRasterizer(raster_settings=_settings(view, background))
    rendered, _ = rasterizer(
        means3D=view["xyz"],
        means2D=screenspace,
        shs=None,
        colors_precomp=channels.contiguous(),
        opacities=view["opacity"],
        scales=view["scaling"],
        rotations=view["rotation"],
        cov3D_precomp=None,
    )
    return rendered[: len(PART_NAMES)]


def _save_checkpoint(path: Path, *, iteration: int, model, optimizer) -> None:
    torch.save(
        {
            "iteration": int(iteration),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations <= 0 or args.topology_interval <= 0:
        raise ValueError("iterations and topology_interval must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    source, prior, features_cpu, graph, view_paths = validate_training_inputs(args.input, args.prior)
    features = features_cpu.cuda()
    graph_indices = graph["indices"].long().cuda()
    graph_weights = graph["weights"].float().cuda()
    model = Compact6Readout(input_dim=32, hidden_dim=args.hidden_dim, class_count=len(PART_NAMES)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    first_iteration = 0
    resume_path = None
    if args.resume == "auto":
        resume_path = find_resume_checkpoint(args.output, iterations=args.iterations)
    elif args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        first_iteration = int(checkpoint["iteration"])
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    initial_parameters = torch.cat([value.detach().flatten() for value in model.parameters()]).clone()
    metrics_path = args.output / "metrics.jsonl"
    started = time.time()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    consecutive_invalid = 0
    for iteration in range(first_iteration + 1, args.iterations + 1):
        view_path = random.choice(view_paths)
        view = _to_cuda(torch.load(view_path, map_location="cpu"))
        if int(view["xyz"].shape[0]) != int(features.shape[0]):
            raise ValueError(f"view point count mismatch: {view_path}")
        point_logits = model(features)
        rendered = _render_logits(view, point_logits)
        flat_indices = balanced_pixel_indices(
            view["labels"], samples_per_class=args.samples_per_class, seed=args.seed + iteration
        )
        sampled_labels = view["labels"].reshape(-1)[flat_indices].long()
        if sampled_labels.numel() == 0 or torch.unique(sampled_labels).numel() < 2:
            consecutive_invalid += 1
            if consecutive_invalid >= len(view_paths):
                raise RuntimeError("unable to sample a frozen view with at least two semantic classes")
            continue
        consecutive_invalid = 0
        sampled_logits = rendered.reshape(len(PART_NAMES), -1)[:, flat_indices].transpose(0, 1)
        loss_2d = F.cross_entropy(sampled_logits, sampled_labels) / math.log(len(PART_NAMES))
        probabilities = torch.softmax(point_logits, dim=1)
        if iteration % args.topology_interval == 0:
            loss_topology = topology_consistency_loss(probabilities, graph_indices, graph_weights)
        else:
            loss_topology = probabilities.sum() * 0.0
        loss = loss_2d + args.topology_lambda * loss_topology
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters())).item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at iteration {iteration}")
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if iteration % args.log_interval == 0 or iteration == first_iteration + 1:
            elapsed = time.time() - started
            current_parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
            row = {
                "iteration": iteration,
                "loss": float(loss.detach().item()),
                "loss_2d": float(loss_2d.detach().item()),
                "loss_topology": float(loss_topology.detach().item()),
                "gradient_norm": grad_norm,
                "parameter_delta": float((current_parameters - initial_parameters).norm().item()),
                "sampled_pixels": int(sampled_labels.numel()),
                "sampled_classes": int(torch.unique(sampled_labels).numel()),
                "elapsed_seconds": elapsed,
                "seconds_per_iteration": elapsed / max(1, iteration - first_iteration),
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
                "view": view_path.stem,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if iteration % args.save_interval == 0 or iteration == args.iterations:
            _save_checkpoint(
                args.output / f"checkpoint_{iteration:06d}.pt",
                iteration=iteration,
                model=model,
                optimizer=optimizer,
            )

    predictions = compact6_predictions(model(features).detach())
    point_count = int(features.shape[0])
    zeros = np.zeros((point_count,), dtype=np.int16)
    save_part_label_bank(
        args.output / "part_label_bank.npz",
        part_label=predictions["part_label"],
        confidence=predictions["confidence"],
        vote_count=zeros,
        per_part_votes=np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
        visible_vote_count=zeros,
        conflict_count=zeros,
        source_checkpoint=source["source_checkpoint"],
        source_asset_root=str(args.input.resolve()),
        source_iteration=int(source.get("loaded_iteration", 0)),
        semantic_probs=predictions["semantic_probs"],
        semantic_margin=predictions["semantic_margin"],
        editable_label=predictions["editable_label"],
        soft_edit_weights=predictions["semantic_probs"],
        source_type="sggs_released_code_canonical_controlled_input_topology_readout",
    )
    torch.save(model.state_dict(), args.output / "compact6_readout.pt")
    bank = dict(np.load(args.output / "part_label_bank.npz", allow_pickle=False))
    summary = {
        **summarize_part_label_bank(bank),
        "method": "SG-GS-Released-Code-Canonical (controlled-input adaptation)",
        "subject": source["subject"],
        "iterations": args.iterations,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "topology_lambda": args.topology_lambda,
        "topology_interval": args.topology_interval,
        "view_count": len(view_paths),
        "point_count": point_count,
        "source_checkpoint": source["source_checkpoint"],
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "sggs_head": prior["sggs_head"],
        "prior_manifest": str((args.prior / "manifest.json").resolve()),
        "elapsed_seconds": time.time() - started,
        "mean_last_100_loss": float(np.mean(losses[-100:])) if losses else None,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.output / "COMPLETE").write_text("complete\n", encoding="ascii")
    print(json.dumps({"complete": True, "output": str(args.output), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
