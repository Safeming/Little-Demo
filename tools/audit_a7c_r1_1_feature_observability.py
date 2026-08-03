#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from utils.a7c_oracle_capacity import load_teacher_artifact
from utils.a7c_renderer_compositor import build_canary_splits,evaluate_contribution_predictions
from utils.part_label_bank import PART_NAMES,load_part_label_bank

def summarize(rows,c):
 s={'record_count':len(rows),'outer_gain':float(np.mean([r['outer_gain'] for r in rows])),'boundary_gain':float(np.mean([r['boundary_gain'] for r in rows])),'minimum_target_response':float(min(r['minimum_target_response'] for r in rows)),'maximum_soft_iou_drop':float(max(r['maximum_soft_iou_drop'] for r in rows)),'maximum_adjacent_gate_change':float(max(r['maximum_adjacent_gate_change'] for r in rows)),'paper_test_eligible':False}
 for signal in ('outer','boundary'):
  g=np.array([r[f'{signal}_gain'] for r in rows]); s[f'{signal}_positive_block_fraction']=float(np.mean(g>0)); s[f'{signal}_block_gain_quantile']=float(np.quantile(g,c['block_gain_quantile'])); s[f'{signal}_worst_block_gain']=float(g.min())
 s['passed']=bool(s['outer_gain']>=c['minimum_outer_gain'] and s['boundary_gain']>=c['minimum_boundary_gain'] and s['minimum_target_response']>=c['minimum_target_response']-1e-7 and s['maximum_soft_iou_drop']<=c['maximum_selection_soft_iou_drop']+1e-7 and s['maximum_adjacent_gate_change']<=c['maximum_adjacent_gate_change']+1e-7 and all(s[f'{x}_positive_block_fraction']>=c['minimum_positive_block_fraction'] and s[f'{x}_block_gain_quantile']>=c['minimum_block_gain_quantile']-1e-9 and s[f'{x}_worst_block_gain']>=-c['maximum_worst_block_regression']-1e-9 for x in ('outer','boundary'))); return s

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,required=True); p.add_argument('--evidence',type=Path,required=True); p.add_argument('--a5-bank',type=Path,required=True); p.add_argument('--teacher',type=Path,required=True); p.add_argument('--training-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(argv); c=json.loads(a.contract.read_text()); t=load_teacher_artifact(a.teacher)
 with np.load(a.evidence,allow_pickle=False) as z: ev={k:z[k] for k in z.files}
 bank=load_part_label_bank(a.a5_bank); part=PART_NAMES.index('lower'); w=np.asarray(bank['soft_edit_weights'],float)[:,part]; ids=t['carrier_ids']; cam=t['camera_index']; frame=t['frame_index']; split=build_canary_splits(camera_index=cam,frame_index=frame,fit_camera_indices=(0,1,2,3),audit_camera_indices=(4,5,6,7),block_count=c['temporal_block_count'])
 streams={}
 for role,prefix in (('objective','renderer'),('guard','renderer_selection')):
  streams[role]={}
  for sig in ('target','outer','boundary'):
   v=np.asarray(ev[f'{prefix}_{sig}_contribution_sequence'],float)[:,:,part]; streams[role][sig]=(v@w,v[:,ids]*w[ids][None,:])
 aggregate={}
 for group in ('F0','F1','F2','F3'):
  rows=[]
  for fold,held in enumerate(split['held_block_masks']):
   pred=np.load(a.training_dir/group/f'fold_{fold}'/'predictions.npz')['gates']
   for camera in range(4):
    m=held&(cam==camera); outputs={}
    for role in ('objective','guard'):
     kw={}
     for sig in ('target','outer','boundary'): kw[sig]=streams[role][sig][0][m]; kw[f'point_{sig}']=streams[role][sig][1][m]
     outputs[role]=evaluate_contribution_predictions(**kw,gates=pred[m])
    rows.append({'fold':fold,'camera_index':camera,'outer_gain':outputs['objective']['outer_gain'],'boundary_gain':outputs['objective']['boundary_gain'],'minimum_target_response':outputs['guard']['minimum_target_response'],'maximum_soft_iou_drop':outputs['guard']['maximum_soft_iou_drop'],'maximum_adjacent_gate_change':float(np.abs(np.diff(pred[m],axis=0)).max())})
  aggregate[group]={'summary':summarize(rows,c),'records':rows}
 aggregate['F3']['summary']['improves_over_F0']=bool(aggregate['F3']['summary']['outer_gain']>aggregate['F0']['summary']['outer_gain'] and aggregate['F3']['summary']['boundary_gain']>aggregate['F0']['summary']['boundary_gain']); passed=bool(aggregate['F3']['summary']['passed'] and aggregate['F3']['summary']['improves_over_F0']); payload={'groups':aggregate,'held_block_passed':passed,'paper_test_eligible':False}; a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/'held_block_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)); (a.output_dir/('.held_block_passed' if passed else '.rejected')).touch(); print(json.dumps(payload,indent=2,sort_keys=True)); return 0 if passed else 2
if __name__=='__main__': raise SystemExit(main())
