"""Additional mechanism and failure-case audits (not preregistered).
Outputs are distinct from the main experiment. No LLM inference is performed.
"""
import json,time
import numpy as np
from scipy.special import expit
from experiment import (ROOT,Q,generate_pairs,fit_dpo,policy,target,project,
                        ref_log,write_csv,phase_experiments)

def sensitivity():
    rows=[]
    for k in [0.,1.,4.,8.]:
        for seed in range(5):
            p,s,df,y,_=generate_pairs(30000+int(k)*100+seed,16000,k=k)
            # Fitting beta=1 learns reward coefficients. Scaling by 1/beta
            # yields the corresponding DPO policy without refitting the same
            # logistic likelihood; this equivalence is exact in this model.
            rewardtheta,fit=fit_dpo(df,y,beta=1.)
            rng=np.random.default_rng(40000+seed)
            p=.05+.9*rng.beta(.8,.8,1000)
            for beta in [.25,.5,2.]:
                for group in ['aligned','opposed','random']:
                    s={'aligned':p,'opposed':1-p,'random':rng.uniform(.05,.95,len(p))}[group]
                    ref=np.exp(ref_log(p));dpo=policy(rewardtheta/beta,p,s)
                    repaired=project(dpo,target(p))
                    for name,pi in [('Reference',ref),('DPO',dpo),('CAEA-1',repaired)]:
                        rows.append(dict(k=k,seed=seed,beta=beta,group=group,method=name,
                            risk=float(np.mean(np.sum(pi*(Q-p[:,None])**2,axis=-1))),
                            style=float(np.mean(np.sum(pi*.8*(2*s[:,None]-1)*np.tile([-1.,0.,1.],21),axis=-1)))))
        print(f'sensitivity k={k} completed',flush=True)
    write_csv(ROOT/'results/sensitivity.csv',rows)

def stochastic():
    rows=[];N=5000;T=400
    for eta in [.05,.25]:
        for seed in range(10):
            for lam in [0.,.25,.5,.75,1.]:
                rng=np.random.default_rng(50000+seed)
                u=np.tile([.47,.53],N//2).astype(float)
                for _ in range(T):
                    prob=(1-lam)*expit(4*(2*u-1))+.5*lam
                    report=(rng.random(N)<prob).astype(float)
                    u=(1-eta)*u+eta*report
                oracle_var=(1-eta)**(2*T)*.0009+eta/(2-eta)*.25*(1-(1-eta)**(2*T))
                rows.append(dict(seed=seed,N=N,T=T,a=4.,eta=eta,lam=lam,
                                 final_variance=float(u.var()),oracle_variance=oracle_var,
                                 BPD=float(u.var()-oracle_var),
                                 raw_variance_drift=float(u.var()-.0009),
                                 final_mean=float(u.mean()),user_MSE=float(np.mean((u-.5)**2))))
        print(f'stochastic eta={eta} completed',flush=True)
    write_csv(ROOT/'results/stochastic.csv',rows)

def semantic_failure():
    rows=[]
    # Literal forecasts are always .5. An unmodeled semantic/style signal is
    # perceived as an additional directional cue. Report-class invariance alone
    # cannot control this user mechanism.
    for delta in [0.,.02,.05,.1]:
        u=np.array([.47,.53]);p=.5
        for _ in range(100):
            interpreted=.5+delta*np.sign(u-.5)
            u=.75*u+.25*interpreted
        rows.append(dict(delta=delta,literal_forecast=.5,literal_report_regret=0.,
                         final_user_MSE=float(np.mean((u-p)**2)),
                         final_variance=float(u.var())))
    write_csv(ROOT/'results/semantic_failure.csv',rows)

if __name__=='__main__':
    t=time.time();sensitivity();stochastic();semantic_failure();phase_experiments()
    (ROOT/'results/stress_status.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-t),indent=2))
    print(f'all stress experiments completed in {time.time()-t:.1f}s',flush=True)
