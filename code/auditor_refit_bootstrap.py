"""Exploratory training/validation/test bootstrap; external labels never fit."""
import csv,gzip,json,time,argparse
import numpy as np
from scale_study import ROOT,OUT,QV,TEMP,MODELS,calibrate,calibrated,softmax,write_csv,write_json

def read(path):
 op=gzip.open if str(path).endswith('.gz')else open
 with op(path,'rt')as f:return list(csv.DictReader(f))

def main(replicates=500):
 protocol=json.loads((OUT/'protocol.json').read_text());aa=read(OUT/'auditor_predictions.csv')
 tr=[r for r in aa if r['split']=='train'];va=[r for r in aa if r['split']=='validation']
 xt=np.array([float(r['raw_mean'])for r in tr]);yt=np.array([float(r['p'])for r in tr]);xv=np.array([float(r['raw_mean'])for r in va]);yv=np.array([float(r['p'])for r in va]);plans=[]
 for directory in [OUT,ROOT/'results/external_math']:
  data=read(directory/'per_prompt_metrics.csv.gz');audit=read(directory/'auditor_predictions.csv');pp=json.loads((directory/'protocol.json').read_text())
  for condition in pp['conditions']:
   variant='new_evidence'if condition=='new_evidence'else'base';arr={r['case_id']:r for r in audit if r['split']=='test'and r['condition']==variant};ids=sorted(arr)
   for key in MODELS:
    base={k:[]for k in ids}
    for r in data:
     if r['model']==key and r['condition']==condition and r['method']=='LM-base':base[r['case_id']].append(float(r['report_regret']))
    assert all(len(v)==2 for v in base.values())
    plans.append(dict(dataset=directory.name,model=key,condition=condition,raw=np.array([float(arr[k]['raw_mean'])for k in ids]),truth=np.array([float(arr[k]['p'])for k in ids]),base=np.array([np.mean(base[k])for k in ids]),contexts=len(ids)))
 rng=np.random.default_rng(2026110253);draws=[];audit_draws=[];start=time.time()
 for b in range(replicates):
  ti=rng.integers(0,len(tr),len(tr));vi=rng.integers(0,len(va),len(va));candidates=[]
  for reg in protocol['calibration_regularization_grid']:
   w=calibrate(xt[ti],yt[ti],reg);candidates.append((float(np.mean((calibrated(xv[vi],w)-yv[vi])**2)),w))
  weight=min(candidates,key=lambda x:x[0])[1]
  for p in plans:
   phat=calibrated(p['raw'],weight);tau=softmax(-(QV[None,:]-phat[:,None])**2/TEMP);full=np.sum(tau*(QV[None,:]-p['truth'][:,None])**2,axis=1)
   ix=rng.integers(0,p['contexts'],p['contexts']);delta=float(np.mean((full-p['base'])[ix]))
   if p['model']=='qwen15':
    center=float(yt[ti].mean());constant=softmax(-(QV-center)**2/TEMP);cr=np.sum(constant[None,:]*(QV[None,:]-p['truth'][:,None])**2,axis=1)
    audit_draws.append(dict(repetition=b,dataset=p['dataset'],condition=p['condition'],full_minus_training_mean=float(np.mean((full-cr)[ix])),point_minus_training_mean=float(np.mean(((phat-p['truth'])**2-(center-p['truth'])**2)[ix]))))
   for fraction in [.5,1.]:draws.append(dict(repetition=b,dataset=p['dataset'],model=p['model'],condition=p['condition'],fraction=fraction,paired_regret_difference=fraction*delta))
  if (b+1)%100==0:print(f'Auditor refit {b+1}/{replicates}',flush=True)
 write_csv(OUT/'auditor_refit_bootstrap_draws.csv.gz',draws);summary=[];audit_summary=[]
 for p in plans:
  for fraction in [.5,1.]:
   v=[r['paired_regret_difference']for r in draws if all(r[k]==p[k]for k in ['dataset','model','condition'])and r['fraction']==fraction];lo,hi=np.quantile(v,[.025,.975])
   summary.append(dict(dataset=p['dataset'],model=p['model'],condition=p['condition'],fraction=fraction,contexts=p['contexts'],replicates=replicates,ci95_low=float(lo),ci95_high=float(hi)))
  if p['model']=='qwen15':
   for metric in ['full_minus_training_mean','point_minus_training_mean']:
    v=[r[metric]for r in audit_draws if r['dataset']==p['dataset']and r['condition']==p['condition']];lo,hi=np.quantile(v,[.025,.975]);audit_summary.append(dict(dataset=p['dataset'],condition=p['condition'],metric=metric,replicates=replicates,ci95_low=float(lo),ci95_high=float(hi)))
 write_csv(OUT/'auditor_refit_intervals.csv',summary);write_csv(OUT/'auditor_vs_constant_draws.csv.gz',audit_draws);write_csv(OUT/'auditor_vs_constant_intervals.csv',audit_summary)
 write_json(OUT/'auditor_refit_bootstrap_status.json',dict(completed=True,replicates=replicates,training_cases_resampled=256,validation_cases_resampled=64,test_contexts_resampled=True,regularization_reselected_on_resampled_validation=True,external_labels_used_for_fitting=False,seconds=time.time()-start))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=500);main(ap.parse_args().replicates)
