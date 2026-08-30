#!/usr/bin/env python3
"""Gate 06F: frozen zero-shot HIBA external evaluation across seven backbone pairs."""

from __future__ import annotations

import argparse, csv, hashlib, json, sys
from pathlib import Path
from typing import Any, Mapping
from collections import Counter

import numpy as np
import torch, yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import build_eval_transform
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.hierarchical_evaluator import FINAL_CLASS_NAMES, build_hierarchical_routing
from src.evaluation.phase04_comparative_harness import collect_single_task_predictions, collect_shared_isic_predictions
from src.models.classification_backbone import build_classification_model
from src.models.shared_three_task import build_shared_three_task_model
from src.utils.reproducibility import make_generator, seed_worker

EXPECTED_BACKBONES=("efficientnet_b0","densenet121","densenet169","resnet50","mobilenet_v3_large","efficientnet_b2","efficientnet_b3")
CLASS_NAMES=("non_malignant","melanoma","bcc","scc")

def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def write_json(p:Path,x:object):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n',encoding='utf-8')

class HIBADataset(Dataset):
    def __init__(self, manifest:Path, root:Path, verify_hashes:bool=True):
        with manifest.open(newline='',encoding='utf-8-sig') as f: self.rows=list(csv.DictReader(f))
        self.root=root; self.tfm=build_eval_transform(); self.verify_hashes=verify_hashes
        if verify_hashes:
            for r in self.rows:
                p=root/r['image_path']
                if not p.is_file(): raise FileNotFoundError(p)
                if sha256_file(p).lower()!=r['image_sha256'].lower(): raise ValueError(f"Image hash mismatch: {p}")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; p=self.root/r['image_path']
        with Image.open(p) as im: image=self.tfm(im.convert('RGB'))
        return {'image':image,'target':torch.tensor(int(r['target_index']),dtype=torch.long),'image_id':r['isic_id'],'patient_id':r['patient_id']}


def resolve_checkpoint(project_root:Path, phase02_root:Path, item:Mapping[str,object])->Path:
    base=project_root if item['root']=='project' else phase02_root
    p=(base/str(item['path'])).resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    if sha256_file(p).lower()!=str(item['sha256']).lower(): raise ValueError(f"Checkpoint hash mismatch: {p}")
    return p

def load_payload(p:Path):
    x=torch.load(p,map_location='cpu',weights_only=False)
    if int(x.get('epoch',-1))<0 or 'model_state_dict' not in x: raise ValueError(f"Bad checkpoint: {p}")
    return x

def load_flat(a,p,e):
    x=load_payload(p)
    if int(x['epoch'])!=e: raise ValueError(f"Epoch mismatch: {p}")
    m=build_classification_model(a,4,pretrained='none',dropout_probability=0.2); m.load_state_dict(x['model_state_dict'],strict=True); return m.eval()
def load_shared(a,p,e):
    x=load_payload(p)
    if int(x['epoch'])!=e: raise ValueError(f"Epoch mismatch: {p}")
    m=build_shared_three_task_model(a,pretrained='none',dropout_probability=0.2); m.load_state_dict(x['model_state_dict'],strict=True); return m.eval()

def patient_cluster_ci(target,flat,hier,patient_ids,reps,seed):
    patients=np.array(sorted(set(patient_ids)),dtype=object); by={p:np.flatnonzero(np.array(patient_ids,dtype=object)==p) for p in patients}; rng=np.random.default_rng(seed)
    deltas=[]
    for _ in range(reps):
        sampled=rng.choice(patients,size=len(patients),replace=True); idx=np.concatenate([by[p] for p in sampled])
        fm=compute_classification_metrics(target[idx],flat[idx],FINAL_CLASS_NAMES); hm=compute_classification_metrics(target[idx],hier[idx],FINAL_CLASS_NAMES)
        deltas.append(float(hm['macro_f1'])-float(fm['macro_f1']))
    lo,hi=np.quantile(np.asarray(deltas),[.025,.975]); return [float(lo),float(hi)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',type=Path,default=ROOT); ap.add_argument('--phase02-root',type=Path,default=Path.home()/ 'projects/Skin-Cancer-Hierarchical-Classification-phase02'); ap.add_argument('--config',type=Path,default=Path('configs/evaluation/phase06f_hiba_frozen_external.yaml')); ap.add_argument('--device',choices=('cpu','cuda'),default='cuda'); ap.add_argument('--preflight-only',action='store_true'); args=ap.parse_args()
    root=args.project_root.resolve(); phase02=args.phase02_root.expanduser().resolve(); cfgp=args.config if args.config.is_absolute() else root/args.config; cfg=yaml.safe_load(cfgp.read_text()); device=torch.device(args.device)
    if tuple(cfg['backbones'])!=EXPECTED_BACKBONES: raise ValueError('Backbone set/order mismatch')
    if cfg['protocol']['external_training_allowed'] or cfg['protocol']['external_finetuning_allowed'] or cfg['protocol']['external_threshold_tuning_allowed'] or cfg['protocol']['external_checkpoint_selection_allowed']: raise ValueError('External protocol is not frozen zero-shot')
    manifest=(root/cfg['hiba']['manifest_path']).resolve()
    if sha256_file(manifest).lower()!=cfg['hiba']['manifest_sha256'].lower(): raise ValueError('HIBA manifest hash mismatch')
    with manifest.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if len(rows)!=int(cfg['hiba']['expected_rows']): raise ValueError('HIBA row count mismatch')
    counts=Counter(r['target_label'] for r in rows)
    if dict(counts)!=dict(cfg['hiba']['expected_class_counts']): raise ValueError(f'HIBA class-count mismatch: {counts}')
    if len({r['patient_id'] for r in rows})!=int(cfg['hiba']['expected_unique_patients']): raise ValueError('HIBA patient-count mismatch')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    pre={'status':'PASS','dataset_constructed':False,'device':str(device),'backbones':{}}
    for a in EXPECTED_BACKBONES:
        pair=cfg['backbones'][a]; fp=resolve_checkpoint(root,phase02,pair['flat']); sp=resolve_checkpoint(root,phase02,pair['shared']); f=load_flat(a,fp,int(pair['flat']['expected_epoch'])); s=load_shared(a,sp,int(pair['shared']['expected_epoch'])); pre['backbones'][a]={'flat':str(fp),'shared':str(sp)}; del f,s
    if args.preflight_only: print(json.dumps(pre,indent=2,sort_keys=True)); return
    ds=HIBADataset(manifest,root,verify_hashes=bool(cfg['hiba']['verify_image_sha256']))
    lc=cfg['loader']; loader=DataLoader(ds,batch_size=int(lc['batch_size']),shuffle=False,num_workers=int(lc['num_workers']),pin_memory=bool(lc['pin_memory']),persistent_workers=bool(lc['persistent_workers']),prefetch_factor=int(lc['prefetch_factor']),worker_init_fn=seed_worker,generator=make_generator(int(cfg['seed'])))
    patient_by_id={r['isic_id']:r['patient_id'] for r in rows}; out=root/cfg['outputs']['directory']; out.mkdir(parents=True,exist_ok=True); results={}
    for a in EXPECTED_BACKBONES:
        print(f'=== {a}: HIBA zero-shot ===',flush=True); pair=cfg['backbones'][a]; fp=resolve_checkpoint(root,phase02,pair['flat']); sp=resolve_checkpoint(root,phase02,pair['shared'])
        fm=load_flat(a,fp,int(pair['flat']['expected_epoch'])); fc=collect_single_task_predictions(fm,loader,class_names=CLASS_NAMES,device=device); fm.to('cpu'); del fm; torch.cuda.empty_cache() if device.type=='cuda' else None
        sm=load_shared(a,sp,int(pair['shared']['expected_epoch'])); sc=collect_shared_isic_predictions(sm,loader,device=device); sm.to('cpu'); del sm; torch.cuda.empty_cache() if device.type=='cuda' else None
        if fc.sample_ids!=sc.sample_ids: raise ValueError('Sample order mismatch')
        routing=build_hierarchical_routing(sc.stage1_targets,sc.stage1_predictions,sc.stage2_targets,sc.stage2_predictions)
        flatm=compute_classification_metrics(fc.targets,fc.predictions,FINAL_CLASS_NAMES); hardm=compute_classification_metrics(routing.final_targets,routing.predicted_gate_predictions,FINAL_CLASS_NAMES); oraclem=compute_classification_metrics(routing.final_targets,routing.oracle_gate_predictions,FINAL_CLASS_NAMES)
        pids=[patient_by_id[x] for x in fc.sample_ids]; ci=patient_cluster_ci(fc.targets,fc.predictions,routing.predicted_gate_predictions,pids,int(cfg['uncertainty']['bootstrap_replicates']),int(cfg['uncertainty']['seed']))
        rows_out=[]
        for i,sid in enumerate(fc.sample_ids): rows_out.append({'isic_id':sid,'patient_id':pids[i],'target':int(fc.targets[i]),'flat_prediction':int(fc.predictions[i]),'shared_hard_prediction':int(routing.predicted_gate_predictions[i]),'shared_oracle_prediction':int(routing.oracle_gate_predictions[i])})
        pp=out/a/'paired_hiba_predictions.csv'; pp.parent.mkdir(parents=True,exist_ok=True)
        with pp.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
        r={'architecture':a,'sample_count':len(rows_out),'flat':flatm,'shared_predicted_gate':hardm,'shared_oracle_gate':oraclem,'routing_loss_macro_f1':float(oraclem['macro_f1'])-float(hardm['macro_f1']),'delta_hierarchy_minus_flat_macro_f1':float(hardm['macro_f1'])-float(flatm['macro_f1']),'patient_cluster_bootstrap_95ci_macro_f1_delta':ci,'paired_predictions_sha256':sha256_file(pp)}; write_json(out/a/'metrics_and_statistics.json',r); results[a]=r
        print(f"{a}: flat={flatm['macro_f1']:.6f} shared={hardm['macro_f1']:.6f} oracle={oraclem['macro_f1']:.6f}",flush=True)
    summary={'gate':'06F','status':'PASS','dataset':'HIBA','zero_shot':True,'manifest_sha256':sha256_file(manifest),'config_sha256':sha256_file(cfgp),'results':results}; write_json(out/'gate06f_hiba_external_summary.json',summary); print('PASS: Gate 06F HIBA external evaluation complete.')
if __name__=='__main__': main()
