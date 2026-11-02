"""Offline replay, paired context-cluster uncertainty, and mathematical checks.

This reuses archived frozen-model arrays; it does not perform fresh model
inference. Intervals are exploratory crossed seed/context bootstraps, conditional
on the recorded training/validation split, evidence grid and pretrained model.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp
from repair import repair_class_marginal

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['SmolLM2-135M-Instruct', 'Qwen2.5-0.5B-Instruct']


def softmax(x):
    return np.exp(x - logsumexp(x, axis=-1, keepdims=True))


def csv_rows(name):
    with open(ROOT / 'results' / name, newline='') as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    with open(ROOT / 'results' / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def replay():
    points = []; comparisons = []; largest = 0.
    original = [json.loads(t) for t in (ROOT/'results/neural_prompts.jsonl').read_text().splitlines()]
    permuted = [json.loads(t) for t in (ROOT/'results/neural_permuted_prompts.jsonl').read_text().splitlines()]
    original_test = [i for i,r in enumerate(original) if r['split']=='test']
    q0 = np.repeat(np.linspace(0,1,5),2); style0 = np.tile([0,1],5)
    rng = np.random.default_rng(20261102)
    for model in MODELS:
        tr = np.load(ROOT/'results'/f'neural_transform_{model}.npz')
        fits = json.loads((ROOT/'results'/f'neural_fit_{model}.json').read_text())['fits']
        for condition,records,tag in [('original',[original[i] for i in original_test],''),
                                      ('permuted',permuted,'_permuted')]:
            archive = np.load(ROOT/'results'/f'neural_{model}{tag}.npz')
            ix = original_test if condition=='original' else list(range(len(records)))
            hidden = archive['hidden']; base = archive['base_logits'][ix]
            # Preserve the original full-array PCA projection before selection.
            feat = np.column_stack([np.ones(len(hidden)),(hidden-tr['mean'])@tr['basis']/tr['scale']])[ix]
            perm = np.array([r.get('option_permutation',list(range(10))) for r in records])
            q = q0[perm]; style = style0[perm]; classes = perm//2
            p = np.array([r['p'] for r in records]); requested = np.array([r['preferred_style'] for r in records])
            tau = softmax(-(np.linspace(0,1,5)[None,:]-p[:,None])**2/.03)
            evidence = [(round(r['prior'],12),round(r['l1'],12),round(r['l0'],12)) for r in records]
            keys = list(dict.fromkeys(evidence)); clusters = [np.where(np.array([e==k for e in evidence]))[0] for k in keys]
            assert len(keys)==24 and all(len(c)==2 for c in clusters)
            seed_metrics=[]
            for fit in fits:
                seed=fit['seed']; dpo=softmax(base+feat@np.array(fit['weight']).T)
                # Canonicalize response order, repair with independent helper,
                # then restore each prompt's letter ordering.
                canonical=np.take_along_axis(dpo,np.argsort(perm,axis=1),axis=1)
                repaired=repair_class_marginal(canonical.reshape(-1,5,2),tau).reshape(-1,10)
                full=np.take_along_axis(repaired,perm,axis=1)
                audit=np.take_along_axis(np.repeat(tau/2,2,axis=1),perm,axis=1)
                methods={'LM-base':softmax(base),'Head-DPO':dpo,'Audit-only':audit,
                         'CAEA-0.5':.5*(dpo+full),'CAEA-1':full}
                sm={}
                for method,pi in methods.items():
                    risk=np.sum(pi*(q-p[:,None])**2,axis=-1)
                    match=np.sum(pi*(style==requested[:,None]),axis=-1)
                    sm[method]={'risk':risk,'style_match':match}
                    for i,r in enumerate(records):
                        points.append(dict(model=model,condition=condition,seed=seed,
                            evidence_context=keys.index(evidence[i]),stance=r['s'],method=method,
                            risk=float(risk[i]),style_match=float(match[i])))
                    recorded=csv_rows('neural_per_seed.csv' if condition=='original' else 'neural_label_control.csv')
                    rr=[r for r in recorded if r['model'].split('/')[-1]==model and int(r['seed'])==seed and r['method']==method]
                    if rr:
                        assert len(rr)==1
                        for metric in ['risk','style_match']:
                            diff=abs(np.mean(sm[method][metric])-float(rr[0][metric]))
                            largest=max(largest,float(diff))
                            # Archived PCA arithmetic uses float32; BLAS replay
                            # can vary in its final few digits across machines.
                            assert diff<1e-7,(model,condition,seed,method,metric,diff)
                # Report-level exact matching, risk and TV mixture identities.
                marginal=np.stack([np.sum(full*(classes==j),axis=-1) for j in range(5)],axis=-1)
                assert np.max(np.abs(marginal-tau))<1e-12
                assert np.max(np.abs(sm['CAEA-1']['risk']-sm['Audit-only']['risk']))<1e-12
                seed_metrics.append(sm)
            for metric,a,b in [('risk','Head-DPO','LM-base'),('style_match','CAEA-1','Audit-only')]:
                delta=np.array([[np.mean(s[a][metric][c]-s[b][metric][c]) for c in clusters] for s in seed_metrics])
                boot=np.empty(5000)
                for j in range(len(boot)):
                    si=rng.integers(0,len(fits),len(fits)); ci=rng.integers(0,len(clusters),len(clusters))
                    boot[j]=delta[np.ix_(si,ci)].mean()
                lo,hi=np.quantile(boot,[.025,.975])
                comparisons.append(dict(model=model,condition=condition,metric=metric,
                    difference=a+' minus '+b,mean=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi),
                    evidence_clusters=24,label_seeds=5,bootstrap_replicates=5000))
    save_csv('review_prompt_metrics.csv',points);save_csv('review_cluster_intervals.csv',comparisons)
    return largest,comparisons


def mathematical_checks():
    # Test the actual empty-support bug and zero-mass feasible edge case.
    sparse=np.array([[.6,.4],[0.,0.]])
    try: repair_class_marginal(sparse,np.array([.5,.5]))
    except ValueError: pass
    else: raise AssertionError('Empty class was not rejected')
    assert np.allclose(repair_class_marginal(sparse,np.array([1.,0.])),sparse)
    for lam in [0.,.1,.5,1.]:
        p=np.array([[.1,.2],[.4,.3]]);tau=np.array([.6,.4])
        out=repair_class_marginal(p,tau,lam);m=p.sum(-1);b=(1-lam)*m+lam*tau
        assert np.allclose(out.sum(-1),b) and np.isclose(out.sum(),1)
        assert np.allclose(out/b[:,None],p/m[:,None])
        assert np.isclose(.5*np.abs(b-tau).sum(),(1-lam)*.5*np.abs(m-tau).sum())
    # Tight h/2 curvature remainder: exact numerical derivative plus secants.
    for a in [.2,1.,4.,10.]:
        H=2*a*a/(3*np.sqrt(3));x=np.linspace(0,1,9);mu=expit(a*(2*x-1));h=x[1]-x[0]
        for j in range(len(x)-1):
            dense=np.linspace(x[j],x[j+1],1001);z=expit(a*(2*dense-1))
            assert np.max(np.abs(2*a*z*(1-z)-(mu[j+1]-mu[j])/h))<=H*h/2+1e-12
    # Low-regret extension: p=.2, reports .19/.21, normalized pressure c=2.
    p0=.2;d=.01;beta=.5;c=2.;k=c*beta/(2*d*d)
    u=np.array([p0-.0006,p0+.0006])
    for _ in range(400):
        mu=expit(4*k*d*(u-p0)/beta)
        assert abs(float(mu.mean())-.5)<1e-11
        u=.75*u+.25*(p0+d*(2*mu-1))
    assert u[1]-u[0]>.019
    (ROOT/'results/near_calibrated_counterexample.json').write_text(json.dumps(dict(
        p=p0,d=d,k=k,beta=beta,c=c,eta=.25,rounds=400,
        pooled_Brier=p0*(1-p0)+d*d,pooled_Brier_regret=d*d,
        central_multiplier=.75+.25*c,final_beliefs=u.tolist(),
        interpretation='Analytic construction with rescaled agreement pressure, not a human estimate'),indent=2)+'\n')
    # Exact cost and risk identities on the reconstructed finite-action fits.
    from experiment import policy,target,project,Q,kl
    rng=np.random.default_rng(508);p=rng.uniform(.05,.95,80);s=rng.uniform(.05,.95,80)
    theta=np.array(json.loads((ROOT/'results/fits.json').read_text())[0]['theta'])
    pi=policy(theta,p,s);tau=target(p);full=project(pi,tau)
    maxres=0.
    for lam in np.linspace(0,1,11):
        out=project(pi,tau,lam)
        r=lambda v:np.sum(v*(Q-p[:,None])**2,axis=-1)
        maxres=max(maxres,float(np.max(np.abs(r(out)-((1-lam)*r(pi)+lam*r(full))))))
        m=pi.reshape(-1,21,3).sum(-1);b=out.reshape(-1,21,3).sum(-1)
        assert np.max(np.abs(kl(out,pi)-kl(b,m)))<1e-12
    return maxres


def main():
    identity=mathematical_checks();replay_error,comparisons=replay()
    summary=dict(completed=True,fresh_neural_inference=False,
        archived_neural_replay_max_abs_error=replay_error,linear_risk_identity_max_abs_error=identity,
        support_error_check='passed',curvature_remainder='H*h/2',
        bootstrap='5000 paired crossed seed/context resamples; 24 contexts; 5 label seeds; exploratory',
        uncertainty_limits='Conditions on fixed training/validation split, finite evidence grid and backbone; excludes fresh inference and training-context variation',
        comparisons=comparisons)
    (ROOT/'results/review_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
