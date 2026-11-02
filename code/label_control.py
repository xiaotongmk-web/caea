"""Held-out answer-bank permutation audit. Does not retrain or select models."""
import json,time,argparse
import numpy as np
import torch
from experiment import ROOT,softmax,write_csv
from neural_probe import encode,QV,STYLE,MODELS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh-cache',action='store_true');args=ap.parse_args()
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    original=[json.loads(x) for x in (ROOT/'results/neural_prompts.jsonl').read_text().splitlines()]
    test=[r for r in original if r['split']=='test']
    rng=np.random.default_rng(2217);records=[];qrows=[];styrows=[];crows=[]
    for r in test:
        perm=rng.permutation(10);q=QV[perm];sty=STYLE[perm]
        options='\n'.join(f'{chr(65+i)}: {int(100*v)}%, {"concise" if z==0 else "warm"}.' for i,(v,z) in enumerate(zip(q,sty)))
        rr=r.copy();rr['prompt']=r['prompt'].split('A: ')[0]+options+'\nAnswer:'
        rr['option_permutation']=perm.tolist();records.append(rr)
        qrows.append(q);styrows.append(sty);crows.append(perm//2)
    (ROOT/'results/neural_permuted_prompts.jsonl').write_text('\n'.join(json.dumps(r)for r in records)+'\n')
    q=np.array(qrows);sty=np.array(styrows);c=np.array(crows)
    p=np.array([r['p']for r in records]);pref=np.array([r['preferred_style']for r in records])
    tau=softmax(-(np.linspace(0,1,5)[None,:]-p[:,None])**2/.03)
    rows=[];metas=[]
    for model in MODELS:
        h,base,mass,meta=encode(model,records,tag='_permuted',refresh=args.refresh_cache);metas.append(meta)
        short=model.split('/')[-1];trans=np.load(ROOT/'results'/f'neural_transform_{short}.npz')
        f=np.column_stack([np.ones(len(h)),(h-trans['mean'])@trans['basis']/trans['scale']])
        fits=json.loads((ROOT/'results'/f'neural_fit_{short}.json').read_text())['fits']
        for fit in fits:
            dpo=softmax(base+f@np.array(fit['weight']).T);ref=softmax(base)
            m=np.stack([np.sum(dpo*(c==j),axis=-1)for j in range(5)],axis=-1)
            full=dpo*np.take_along_axis(tau,c,axis=1)/np.take_along_axis(m,c,axis=1)
            for name,pi in [('LM-base',ref),('Head-DPO',dpo),('CAEA-1',full)]:
                rows.append(dict(model=model,seed=fit['seed'],method=name,test_prompts=len(records),
                                 risk=float(np.mean(np.sum(pi*(q-p[:,None])**2,axis=-1))),
                                 style_match=float(np.mean(np.sum(pi*(sty==pref[:,None]),axis=-1))),
                                 option_mass=float(mass.mean())))
    write_csv(ROOT/'results/neural_label_control.csv',rows)
    (ROOT/'results/neural_label_control_metadata.json').write_text(json.dumps(metas,indent=2))
    print('Answer-bank permutation control complete',flush=True)

if __name__=='__main__':main()
