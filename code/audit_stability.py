"""Finite binary-response audit certificates, conditional on known curvature.
The curvature bound is supplied by the constructed logistic model, not learned
from these finite audits. Also constructs an indistinguishable smooth failure.
"""
import json
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,write_csv

def bump(s,center,width):
    z=(np.asarray(s)-center)/width
    out=np.zeros_like(z,dtype=float);inside=np.abs(z)<1
    out[inside]=np.exp(1-1/(1-z[inside]**2))
    return out

def main():
    a=4.;H=2*a*a/(3*np.sqrt(3));Ltrue=a/2;delta=.05;gamma=.1
    rows=[];rng=np.random.default_rng(9217)
    for budget in [1024,4096,16384,65536,262144]:
        planners=[]
        for J in [5,9,17,33,65]:
            n=budget//J;h=1/(J-1)
            eps=np.sqrt(np.log(2*J/delta)/(2*n))
            planners.append((2*eps/h+H*h,J,n,h,eps))
        _,J,n,h,eps=min(planners) # planned without using sampled audit data
        x=np.linspace(0,1,J);mu=expit(a*(2*x-1))
        for rep in range(500):
            empirical=rng.binomial(n,mu)/n
            cert=float(np.max(np.abs(np.diff(empirical)))/h+2*eps/h+H*h)
            lam=float(max(0,1-(1-gamma)/cert))
            rows.append(dict(total_budget=budget,J=J,n_per_point=n,used_samples=J*n,
                repetition=rep,delta=delta,gamma=gamma,H=H,true_L=Ltrue,
                audit_error_bound=eps,certified_L=cert,repair_fraction=lam,
                true_contraction_coefficient=.75+.25*(1-lam)*Ltrue,
                coverage=int(cert>=Ltrue)))
    write_csv(ROOT/'results/audit_stability.csv',rows)
    summary=[]
    for N in dict.fromkeys(r['total_budget']for r in rows):
        rr=[r for r in rows if r['total_budget']==N]
        summary.append(dict(total_budget=N,J=rr[0]['J'],n_per_point=rr[0]['n_per_point'],
            mean_certified_L=float(np.mean([r['certified_L']for r in rr])),
            mean_repair_fraction=float(np.mean([r['repair_fraction']for r in rr])),
            empirical_coverage=float(np.mean([r['coverage']for r in rr]))))
    write_csv(ROOT/'results/audit_stability_summary.csv',summary)
    # All sampled audit points have exactly the invariant Bernoulli(.5) law,
    # but there are stable off-grid fixed points at .13 and .87.
    grid=np.linspace(0,1,9);dense=np.linspace(0,1,10001)
    # Support avoids the nearest audited stances .125 and .875.
    width=.002
    m=.5+.37*(bump(dense,.87,width)-bump(dense,.13,width))
    audit_m=.5+.37*(bump(grid,.87,width)-bump(grid,.13,width))
    fixed=np.array([.13,.87])
    fixed_m=.5+.37*(bump(fixed,.87,width)-bump(fixed,.13,width))
    assert np.max(np.abs(audit_m-.5))==0
    assert np.max(np.abs(fixed_m-fixed))<1e-12
    failure=dict(audit_points=grid.tolist(),audit_means=audit_m.tolist(),
                 off_grid_fixed_points=fixed.tolist(),bump_width=width,
                 stable_fixed_point_variance=float(fixed.var()),
                 assumptions='No valid supplied global curvature/Lipschitz bound')
    (ROOT/'results/audit_indistinguishability.json').write_text(json.dumps(failure,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.3))
    axs[0].semilogx([r['total_budget']for r in summary],[r['mean_repair_fraction']for r in summary],'o-',label='Finite-audit certificate')
    axs[0].axhline(1-(1-gamma)/Ltrue,color='gray',ls='--',label='Known-L budget (same margin)')
    axs[0].set(xlabel='Binary report audit budget',ylabel='Sufficient repair fraction',ylim=(.5,.9))
    axs[0].legend(fontsize=7,frameon=False)
    axs[1].plot(dense,m,label='Smooth unaudited kernel',lw=1.1)
    axs[1].scatter(grid,audit_m,s=20,color='black',label='Finite audit locations',zorder=4)
    axs[1].scatter(fixed,fixed_m,s=25,color='red',label='Stable off-grid beliefs',zorder=5)
    axs[1].plot([0,1],[0,1],ls=':',color='gray');axs[1].set(xlabel='User stance',ylabel='Expected binary report')
    axs[1].legend(fontsize=7,frameon=False,loc='upper left')
    fig.tight_layout();fig.savefig(ROOT/'figures/audit_stability.pdf');fig.savefig(ROOT/'figures/audit_stability.png',dpi=180)
    (ROOT/'results/audit_stability_status.json').write_text(json.dumps(dict(completed=True,summary=summary,failure=failure),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
