#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.build_renderer_aligned_temporal_evidence import _build_config, _file_sha256
from utils.a7c_oracle_capacity import _artifact_fingerprint, load_teacher_artifact
from utils.a7c_ray_context_probe import exact_alpha_transmittance_mass, ray_depth_moments, sample_footprint_context
from utils.a7c_renderer_compositor import extract_runtime_probe_features
from utils.part_label_bank import PART_NAMES, load_part_label_bank

def args_parser(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,required=True); p.add_argument('--config',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--a5-bank',type=Path,required=True); p.add_argument('--teacher',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--max-samples',type=int); p.add_argument('--dataset-root',default=''); p.add_argument('--subject',default='377'); p.add_argument('--explicit-binding-render-preset',default='none'); p.add_argument('--dry-run',action='store_true'); return p.parse_args(argv)

def save(path,arrays):
 fp=_artifact_fingerprint(arrays); arrays['output_fingerprint']=np.array(fp); path.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(path)+'.tmp.npz'); np.savez_compressed(tmp,**arrays); tmp.replace(path); return fp

def main(argv=None):
 args=args_parser(argv); contract=json.loads(args.contract.read_text()); teacher=load_teacher_artifact(args.teacher); bank=load_part_label_bank(args.a5_bank); ids=np.asarray(teacher['carrier_ids'],np.int64)
 cameras=contract['fit_cameras']+contract['audit_cameras']; frames=list(range(contract['frame_start'],contract['frame_end'],contract['frame_stride'])); expected=[(c,f) for c in cameras for f in frames]
 if args.max_samples: expected=expected[:args.max_samples]
 if args.dry_run: print(json.dumps({'samples':len(expected),'carriers':len(ids),'F3_features':len(contract['feature_groups']['F3'])})); return 0
 from gaussian_renderer import rasterize_gaussians
 from scene import GaussianModel, Scene
 from tools.semantic_viewer.build_part_label_bank import _find_dataset_index, _project_points
 cfg=_build_config(args,{'cameras':cameras,'frame_start':contract['frame_start'],'frame_end':contract['frame_end'],'frame_stride':contract['frame_stride'],'frames':frames,'parts':['lower'],'formal_protocol':True}); bg=torch.zeros(3,device='cuda')
 with torch.no_grad(): gaussians=GaussianModel(cfg.model.gaussian); scene=Scene(cfg,gaussians,str(args.output.parent.resolve())); scene.eval(); iteration=int(scene.load_checkpoint(str(args.checkpoint.resolve())))
 count=int(scene.gaussians.get_xyz.shape[0]); lower=np.asarray(bank['soft_edit_weights'],np.float32)[:,PART_NAMES.index('lower')]; sem=np.asarray(bank['semantic_probs'],np.float32); margin=np.asarray(bank['semantic_margin'],np.float32)
 if count!=len(lower): raise ValueError('checkpoint and A5 bank differ')
 canonical=scene.gaussians.get_xyz.detach().float().cpu().numpy(); center=np.median(canonical,axis=0); scale=max(np.quantile(np.linalg.norm(canonical-center,axis=1),.95),1e-6); canonical=(canonical-center)/scale
 scaling=np.log(np.maximum(scene.gaussians.get_scaling.detach().float().cpu().numpy(),1e-8)); rotation=np.abs(scene.gaussians.get_rotation.detach().float().cpu().numpy())
 static=np.concatenate([canonical,scaling,rotation,sem,margin[:,None],np.zeros((count,1),np.float32)],axis=1).astype(np.float32)
 rows=[]; cams=[]; frame_rows=[]
 for n,(camera,frame) in enumerate(expected):
  view=scene.test_dataset[_find_dataset_index(scene.test_dataset,f'{camera}_f{frame:06d}')]
  with torch.no_grad(): deformed,_,_=scene.convert_gaussians(view,iteration,compute_loss=False)
  xy,valid,depth=_project_points(deformed.get_xyz,view); colors=torch.ones((count,3),device='cuda',requires_grad=True)
  pkg=rasterize_gaussians(view,deformed,cfg.pipeline,bg,colors_precomp=colors,return_opacity=False); mass=exact_alpha_transmittance_mass(pkg['render'],colors).detach()
  depth_safe=depth.detach().float().clamp_min(0); lower_t=torch.as_tensor(lower,device='cuda'); sem_t=torch.as_tensor(sem[:,PART_NAMES.index('lower')],device='cuda')
  with torch.no_grad():
   dcolors=torch.stack([depth_safe,depth_safe.square(),lower_t],1); dbuf=rasterize_gaussians(view,deformed,cfg.pipeline,bg,colors_precomp=dcolors,return_opacity=False)['render']
   scolors=torch.stack([sem_t,torch.ones_like(sem_t),torch.ones_like(sem_t)],1); sbuf=rasterize_gaussians(view,deformed,cfg.pipeline,bg,colors_precomp=scolors,return_opacity=False)['render']
  means=deformed.get_xyz.detach().float().cpu().numpy(); radii=pkg['radii'][:count].detach().float().cpu().numpy(); opacity=deformed.get_opacity[:count].detach().float().cpu().numpy().reshape(-1)
  f0=extract_runtime_probe_features(means3d=means,world_view_transform=view.world_view_transform.detach().cpu().numpy(),camera_center=view.camera_center.detach().cpu().numpy(),visibility=pkg['visibility_filter'][:count].detach().cpu().numpy(),radii=radii,opacity=opacity,a5_lower_weight=lower,selected_lower=lower>=contract['selection_threshold'])
  alpha=pkg['render'][0].detach().float().cpu().numpy(); dnp=dbuf.detach().float().cpu().numpy(); snp=sbuf.detach().float().cpu().numpy(); dmean,dvar,avail=ray_depth_moments(alpha,dnp[0],dnp[1]); buffers=np.stack([alpha,dmean,dvar,dnp[2],snp[0]])
  context=sample_footprint_context(buffers,projected_xy=xy.detach().float().cpu().numpy(),radii=radii); context_order=np.stack([context[:,0],context[:,5],context[:,10],context[:,1],context[:,6],context[:,11],context[:,2],context[:,7],context[:,12],context[:,3],context[:,8],context[:,13],context[:,4],context[:,9],context[:,14]],1)
  residual=depth_safe.detach().cpu().numpy()-context[:,1]; dynamic=np.concatenate([mass.cpu().numpy()[:,None],np.log1p(mass.cpu().numpy())[:,None],context_order,residual[:,None],(context[:,0]>1e-8)[:,None]],1).astype(np.float32)
  feature=np.concatenate([f0,static,dynamic],1)
  if feature.shape[1]!=len(contract['feature_groups']['F3']) or not np.isfinite(feature).all(): raise ValueError('F3 feature schema mismatch')
  rows.append(feature[ids]); cams.append(cameras.index(camera)); frame_rows.append(frame); print(f'[R1.1 probe] {n+1}/{len(expected)} {camera}_f{frame:06d}',flush=True)
 arrays={'schema_version':np.array(1,np.int32),'features':np.stack(rows).astype(np.float32),'feature_names':np.asarray(contract['feature_groups']['F3'],dtype='U48'),'carrier_ids':ids,'camera_index':np.asarray(cams,np.int16),'frame_index':np.asarray(frame_rows,np.int32),'source_checkpoint_sha256':np.array(_file_sha256(args.checkpoint)),'source_a5_bank_sha256':np.array(_file_sha256(args.a5_bank)),'source_teacher_fingerprint':np.array(str(teacher['output_fingerprint'])),'paper_test_eligible':np.array(0,np.uint8)}
 if args.max_samples is None:
  if not np.array_equal(arrays['camera_index'],teacher['camera_index']) or not np.array_equal(arrays['frame_index'],teacher['frame_index']): raise ValueError('sample order differs from teacher')
 fp=save(args.output,arrays); print(json.dumps({'output_fingerprint':fp,'shape':arrays['features'].shape},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
