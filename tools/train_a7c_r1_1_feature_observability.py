#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.a7c_oracle_capacity import _artifact_fingerprint,load_teacher_artifact
from utils.a7c_ray_context_probe import select_feature_group
from utils.a7c_renderer_compositor import BoundedCarrierMLP,build_canary_splits,fit_feature_normalization

def load_probe(path):
 with np.load(path,allow_pickle=False) as z: p={k:z[k] for k in z.files}
 if _artifact_fingerprint(p)!=str(p['output_fingerprint']): raise ValueError('probe fingerprint mismatch')
 return p

def train(name,features,target,train_mask,contract,out,device):
 stats=fit_feature_normalization(features,sample_mask=train_mask); x=torch.from_numpy(((features-stats['mean'])/stats['scale']).astype(np.float32)).to(device); y=torch.from_numpy(target.astype(np.float32)).to(device); mask=torch.from_numpy(train_mask).to(device)
 torch.manual_seed(contract['random_seed']); model=BoundedCarrierMLP(x.shape[-1],contract['hidden_dimensions'],minimum_gate=contract['minimum_gate'],maximum_gate=contract['maximum_gate'],initial_gate=contract['initial_minimum_gate']).to(device); opt=torch.optim.AdamW(model.parameters(),lr=contract['learning_rate'],weight_decay=contract['weight_decay']); losses=[]
 for _ in range(contract['training_epochs']):
  opt.zero_grad(set_to_none=True); pred=model(x.reshape(-1,x.shape[-1])).reshape(y.shape); loss=F.smooth_l1_loss(pred[mask],y[mask],beta=contract['huber_delta']); loss.backward(); opt.step(); losses.append(float(loss.detach()))
 with torch.no_grad(): pred=model(x.reshape(-1,x.shape[-1])).reshape(y.shape).cpu().numpy().astype(np.float32)
 root=out/name; root.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'mean':stats['mean'],'scale':stats['scale'],'paper_test_eligible':False},root/'model.pt'); np.savez_compressed(root/'predictions.npz',gates=pred,train_mask=train_mask)
 summary={'name':name,'initial_loss':losses[0],'final_loss':losses[-1],'minimum_gate':float(pred.min()),'maximum_gate':float(pred.max()),'paper_test_eligible':False}; (root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)); print(json.dumps(summary),flush=True)

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,required=True); p.add_argument('--probe',type=Path,required=True); p.add_argument('--teacher',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',default='cuda'); a=p.parse_args(argv); c=json.loads(a.contract.read_text()); probe=load_probe(a.probe); teacher=load_teacher_artifact(a.teacher)
 for k in ('carrier_ids','camera_index','frame_index'):
  if not np.array_equal(probe[k],teacher[k]): raise ValueError(f'{k} mismatch')
 split=build_canary_splits(camera_index=probe['camera_index'],frame_index=probe['frame_index'],fit_camera_indices=(0,1,2,3),audit_camera_indices=(4,5,6,7),block_count=c['temporal_block_count']); full=probe['features'].astype(np.float32); names=list(map(str,probe['feature_names'])); target=teacher['gates'].astype(np.float32)
 for group,requested in c['feature_groups'].items():
  features=select_feature_group(full,names,requested)
  for fold,held in enumerate(split['held_block_masks']): train(f'{group}/fold_{fold}',features,target,split['fit_mask']&~held,c,a.output_dir,a.device)
  train(f'{group}/final',features,target,split['fit_mask'],c,a.output_dir,a.device)
 return 0
if __name__=='__main__': raise SystemExit(main())
