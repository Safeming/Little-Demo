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

from utils.gaussian_grouping_canonical import (
    balanced_pixel_indices,
    grouping_3d_consistency_loss,
    identity_predictions,
)
from utils.part_label_bank import PART_NAMES, save_part_label_bank, summarize_part_label_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Gaussian Grouping identities on frozen avatar Gaussians.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--identity-dim", type=int, default=16)
    parser.add_argument("--identity-lr", type=float, default=0.0025)
    parser.add_argument("--classifier-lr", type=float, default=5.0e-4)
    parser.add_argument("--samples-per-class", type=int, default=512)
    parser.add_argument("--reg3d-interval", type=int, default=2)
    parser.add_argument("--reg3d-k", type=int, default=5)
    parser.add_argument("--reg3d-lambda", type=float, default=2.0)
    parser.add_argument("--reg3d-max-points", type=int, default=300000)
    parser.add_argument("--reg3d-sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--resume", default="none", help="none, auto, or an explicit checkpoint path")
    return parser


def validate_input_manifest(input_dir: Path) -> tuple[dict, torch.Tensor, list[Path]]:
    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("view_count", -1)) != 80 or len(manifest.get("views", [])) != 80:
        raise ValueError("expected exactly 80 frozen training views")
    if list(manifest.get("part_names", [])) != list(PART_NAMES):
        raise ValueError("frozen manifest part order does not match compact-6 protocol")
    canonical_xyz = torch.load(input_dir / "canonical_xyz.pt", map_location="cpu")
    if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3:
        raise ValueError("canonical_xyz must have shape [N, 3]")
    if int(manifest.get("point_count", -1)) != int(canonical_xyz.shape[0]):
        raise ValueError("manifest canonical point count does not match canonical_xyz")
    view_paths = [input_dir / "views" / str(name) for name in manifest["views"]]
    missing = [str(path) for path in view_paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing frozen training views: {missing[:3]}")
    return manifest, canonical_xyz.float(), view_paths


def find_resume_checkpoint(output_dir: Path, *, iterations: int) -> Path | None:
    output_dir = Path(output_dir)
    if (output_dir / "COMPLETE").is_file():
        return None
    candidates = sorted(output_dir.glob("checkpoint_*.pt"))
    for path in reversed(candidates):
        payload = torch.load(path, map_location="cpu")
        if int(payload.get("iteration", 0)) < int(iterations):
            return path
    return None


def _settings(view, background):
    from diff_gaussian_rasterization import GaussianRasterizationSettings

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


def _render_identities(view: dict, encodings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from diff_gaussian_rasterization import GaussianRasterizer

    screenspace = torch.zeros_like(view["xyz"], requires_grad=True)
    background = torch.zeros((3,), dtype=torch.float32, device="cuda")
    rasterizer = GaussianRasterizer(raster_settings=_settings(view, background))
    colors = torch.zeros((view["xyz"].shape[0], 3), dtype=torch.float32, device="cuda")
    _, radii, rendered_identities = rasterizer(
        means3D=view["xyz"],
        means2D=screenspace,
        shs=None,
        sh_objs=encodings[:, None, :].contiguous(),
        colors_precomp=colors,
        opacities=view["opacity"],
        scales=view["scaling"],
        rotations=view["rotation"],
        cov3D_precomp=None,
    )
    return rendered_identities, radii


def _save_checkpoint(path, *, iteration, encodings, classifier, identity_optimizer, classifier_optimizer):
    torch.save(
        {
            "iteration": int(iteration),
            "encodings": encodings.detach().cpu(),
            "classifier": classifier.state_dict(),
            "identity_optimizer": identity_optimizer.state_dict(),
            "classifier_optimizer": classifier_optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.identity_dim != 16:
        raise ValueError("Gaussian Grouping rasterizer is fixed to 16 identity channels")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, canonical_xyz_cpu, view_paths = validate_input_manifest(args.input)
    canonical_xyz = canonical_xyz_cpu.cuda()
    encodings = torch.nn.Parameter(torch.randn((canonical_xyz.shape[0], 16), device="cuda") * 1.0e-2)
    classifier = torch.nn.Conv2d(16, len(PART_NAMES), kernel_size=1).cuda()
    identity_optimizer = torch.optim.Adam([encodings], lr=args.identity_lr)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr)
    first_iteration = 0
    resume_path = None
    if args.resume == "auto":
        resume_path = find_resume_checkpoint(args.output, iterations=args.iterations)
    elif args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        first_iteration = int(checkpoint["iteration"])
        encodings.data.copy_(checkpoint["encodings"].cuda())
        classifier.load_state_dict(checkpoint["classifier"])
        identity_optimizer.load_state_dict(checkpoint["identity_optimizer"])
        classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    initial_encodings = encodings.detach().clone()
    initial_classifier = classifier.weight.detach().clone()
    metrics_path = args.output / "metrics.jsonl"
    started = time.time()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    consecutive_invalid = 0
    for iteration in range(first_iteration + 1, args.iterations + 1):
        view_path = random.choice(view_paths)
        view = _to_cuda(torch.load(view_path, map_location="cpu"))
        if int(view["xyz"].shape[0]) != int(encodings.shape[0]):
            raise ValueError(f"view point count mismatch: {view_path}")
        rendered, _ = _render_identities(view, encodings)
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
        rendered_flat = rendered.reshape(16, -1)[:, flat_indices]
        logits = classifier(rendered_flat[:, :, None]).squeeze(-1).transpose(0, 1)
        loss_2d = F.cross_entropy(logits, sampled_labels) / math.log(len(PART_NAMES))
        point_logits = classifier(encodings.transpose(0, 1)[:, :, None]).squeeze(-1).transpose(0, 1)
        probabilities = torch.softmax(point_logits, dim=-1)
        if iteration % args.reg3d_interval == 0:
            loss_3d = grouping_3d_consistency_loss(
                canonical_xyz.detach(),
                probabilities,
                k=args.reg3d_k,
                lambda_val=args.reg3d_lambda,
                max_points=args.reg3d_max_points,
                sample_size=args.reg3d_sample_size,
                seed=args.seed + iteration,
            )
        else:
            loss_3d = probabilities.sum() * 0.0
        loss = loss_2d + loss_3d
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {iteration}")
        identity_optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        identity_grad = float(encodings.grad.norm().item())
        classifier_grad = float(classifier.weight.grad.norm().item())
        if not math.isfinite(identity_grad) or not math.isfinite(classifier_grad):
            raise FloatingPointError(f"non-finite gradient at iteration {iteration}")
        identity_optimizer.step()
        classifier_optimizer.step()
        losses.append(float(loss.detach().item()))
        if iteration % args.log_interval == 0 or iteration == first_iteration + 1:
            elapsed = time.time() - started
            row = {
                "iteration": iteration,
                "loss": float(loss.detach().item()),
                "loss_2d": float(loss_2d.detach().item()),
                "loss_3d": float(loss_3d.detach().item()),
                "identity_grad_norm": identity_grad,
                "classifier_grad_norm": classifier_grad,
                "identity_parameter_delta": float((encodings.detach() - initial_encodings).norm().item()),
                "classifier_parameter_delta": float((classifier.weight.detach() - initial_classifier).norm().item()),
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
                encodings=encodings,
                classifier=classifier,
                identity_optimizer=identity_optimizer,
                classifier_optimizer=classifier_optimizer,
            )

    predictions = identity_predictions(
        encodings.detach(), classifier.weight.detach().reshape(len(PART_NAMES), 16), classifier.bias.detach()
    )
    point_count = int(encodings.shape[0])
    zeros = np.zeros((point_count,), dtype=np.int16)
    save_part_label_bank(
        args.output / "part_label_bank.npz",
        part_label=predictions["part_label"],
        confidence=predictions["confidence"],
        vote_count=zeros,
        per_part_votes=np.zeros((point_count, len(PART_NAMES)), dtype=np.int16),
        visible_vote_count=zeros,
        conflict_count=zeros,
        source_checkpoint=manifest["source_checkpoint"],
        source_asset_root=str(args.input.resolve()),
        source_iteration=int(manifest["loaded_iteration"]),
        semantic_probs=predictions["semantic_probs"],
        semantic_margin=predictions["semantic_margin"],
        editable_label=predictions["editable_label"],
        soft_edit_weights=predictions["semantic_probs"],
        source_type="gaussian_grouping_canonical_controlled_input_identity",
    )
    torch.save(encodings.detach().cpu(), args.output / "identity_encodings.pt")
    torch.save(classifier.state_dict(), args.output / "classifier.pt")
    bank = dict(np.load(args.output / "part_label_bank.npz", allow_pickle=False))
    summary = {
        **summarize_part_label_bank(bank),
        "method": "Gaussian Grouping-Canonical (controlled-input adaptation)",
        "subject": manifest["subject"],
        "iterations": args.iterations,
        "identity_dim": 16,
        "view_count": len(view_paths),
        "point_count": point_count,
        "source_checkpoint": manifest["source_checkpoint"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
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
