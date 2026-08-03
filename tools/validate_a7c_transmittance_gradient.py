#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_renderer_aligned_temporal_evidence import _build_config
from utils.a7c_oracle_capacity import load_teacher_artifact
from utils.a7c_ray_context_probe import exact_alpha_transmittance_mass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--subject", default="377")
    parser.add_argument("--explicit-binding-render-preset", default="none")
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text())
    teacher = load_teacher_artifact(args.teacher)
    from gaussian_renderer import rasterize_gaussians
    from scene import GaussianModel, Scene
    from tools.semantic_viewer.build_part_label_bank import _find_dataset_index
    cfg = _build_config(args, {"cameras":["c01"],"frame_start":0,"frame_end":5,"frame_stride":5,"frames":[0],"parts":["lower"],"formal_protocol":True})
    background = torch.zeros(3, device="cuda")
    with torch.no_grad():
        gaussians=GaussianModel(cfg.model.gaussian); scene=Scene(cfg,gaussians,str(args.output.parent.resolve())); scene.eval(); iteration=int(scene.load_checkpoint(str(args.checkpoint.resolve())))
    view=scene.test_dataset[_find_dataset_index(scene.test_dataset,"c01_f000000")]
    with torch.no_grad(): deformed,_,_=scene.convert_gaussians(view,iteration,compute_loss=False)
    count=int(scene.gaussians.get_xyz.shape[0]); colors=torch.ones((count,3),device="cuda",requires_grad=True)
    pkg=rasterize_gaussians(view,deformed,cfg.pipeline,background,colors_precomp=colors,return_opacity=False)
    mass=exact_alpha_transmittance_mass(pkg["render"],colors).detach()
    ids=torch.as_tensor(teacher["carrier_ids"],device="cuda",dtype=torch.long)
    carrier=int(ids[torch.argmax(mass[ids])].item()); analytical=float(mass[carrier].item())
    epsilon=float(contract["gradient_finite_difference_epsilon"])
    values=[]
    for sign in (-1.0,1.0):
        perturbed=torch.ones((count,3),device="cuda"); perturbed[carrier]+=sign*epsilon
        with torch.no_grad(): out=rasterize_gaussians(view,deformed,cfg.pipeline,background,colors_precomp=perturbed,return_opacity=False)["render"]
        values.append(float(out.sum().item()))
    finite=(values[1]-values[0])/(2*epsilon*3.0)
    relative=abs(finite-analytical)/max(abs(analytical),1e-8)
    invisible=~pkg["visibility_filter"][:count]
    invisible_max=float(mass[invisible].abs().max().item()) if bool(invisible.any()) else 0.0
    result={"carrier_id":carrier,"analytical_mass":analytical,"finite_difference_mass":finite,"relative_error":relative,"invisible_maximum_mass":invisible_max,"passed":bool(relative<=contract["gradient_maximum_relative_error"] and invisible_max<=1e-8),"paper_test_eligible":False}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__": raise SystemExit(main())
