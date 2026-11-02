"""CAEA controlled experiments. No language models or human subjects are used.

Run from the project root: OPENBLAS_NUM_THREADS=1 python code/experiment.py
Every learned-policy number is produced by fitting a pairwise logistic DPO loss.
Exact-policy calculations and mean-response simulations are labeled separately.
"""
from pathlib import Path
import argparse, csv, json, platform, time
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t
from repair import repair_class_marginal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
Q = np.repeat(np.linspace(0, 1, 21), 3)
Z = np.tile(np.array([-1., 0., 1.]), 21)
NU = np.tile(np.array([.25, .5, .25]), 21)
BETA = .5
K = 4.
TARGET_TEMP = .01

def softmax(a):
    return np.exp(a-logsumexp(a, axis=-1, keepdims=True))

def features(p, s):
    p, s = np.broadcast_arrays(np.asarray(p), np.asarray(s))
    q = np.broadcast_to(Q, p.shape+(len(Q),))
    z = np.broadcast_to(Z, q.shape)
    t = (2*s-1)[..., None]
    return np.stack([q, q*q, q*p[..., None], q*s[..., None],
                     z*t, z*t*(2*q-1)], axis=-1)

def ref_log(p):
    p = np.asarray(p)
    pref = .5+.65*(p-.5)
    a = -(Q-pref[..., None])**2/.06+np.log(NU)
    return a-logsumexp(a, axis=-1, keepdims=True)

def target(p, temp=TARGET_TEMP):
    p = np.asarray(p)
    return softmax(-(np.linspace(0,1,21)-p[..., None])**2/temp)

def reward(p, s, k=K):
    return (-(Q-np.asarray(p)[..., None])**2
            -k*(Q-np.asarray(s)[..., None])**2
            +.8*(2*np.asarray(s)[..., None]-1)*Z
            +.25*(2*Q-1)*(2*np.asarray(s)[..., None]-1)*Z)

def policy(theta, p, s):
    return softmax(ref_log(p)+np.einsum('...af,f->...a', features(p,s),theta))

def project(pi, tau, lam=1.):
    shape = pi.shape
    a = pi.reshape(shape[:-1]+(21,3))
    return repair_class_marginal(a, tau, lam).reshape(shape)

def kl(p,q):
    return np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(q,1e-300))),axis=-1)

def sample_rows(rng, pi):
    return (rng.random(pi.shape[0])[:,None]>np.cumsum(pi,axis=1)).sum(1).clip(max=pi.shape[1]-1)

def generate_pairs(seed,n,k=K):
    rng = np.random.default_rng(seed)
    p = .05+.9*rng.beta(.8,.8,n)
    s = rng.uniform(.05,.95,n)
    ref = np.exp(ref_log(p))
    proposal = .5*ref+.5/len(Q)
    a,b=sample_rows(rng,proposal),sample_rows(rng,proposal)
    f=features(p,s)
    df=f[np.arange(n),a]-f[np.arange(n),b]
    r=reward(p,s,k)
    prob=expit(r[np.arange(n),a]-r[np.arange(n),b])
    label=(rng.random(n)<prob).astype(float)
    chosen=np.where(label==1,a,b)
    return p,s,df,label,chosen

def fit_dpo(df,y,beta=BETA):
    def fun(theta):
        margin=beta*(df@theta)
        val=np.mean(np.logaddexp(0,margin)-y*margin)+.5e-6*np.sum(theta**2)
        grad=beta*(df.T@(expit(margin)-y))/len(y)+1e-6*theta
        return val,grad
    opt=minimize(fun,np.zeros(df.shape[1]),jac=True,method='L-BFGS-B',
                 options={'gtol':1e-8,'ftol':1e-12,'maxiter':500})
    if not opt.success:
        raise RuntimeError(opt.message)
    return opt.x,dict(success=bool(opt.success),loss=float(opt.fun),iterations=int(opt.nit))

def fit_sft(p,s,chosen):
    f=features(p,s); lr=ref_log(p)
    fc=f[np.arange(len(p)),chosen]
    def fun(theta):
        logits=lr+np.einsum('naf,f->na',f,theta)
        pp=softmax(logits)
        val=np.mean(logsumexp(logits,axis=-1)-logits[np.arange(len(p)),chosen])+.5e-6*np.sum(theta**2)
        grad=np.mean(np.einsum('na,naf->nf',pp,f)-fc,axis=0)+1e-6*theta
        return val,grad
    opt=minimize(fun,np.zeros(f.shape[-1]),jac=True,method='L-BFGS-B',
                 options={'gtol':1e-7,'ftol':1e-11,'maxiter':500})
    if not opt.success: raise RuntimeError(opt.message)
    return opt.x,dict(success=bool(opt.success),loss=float(opt.fun),iterations=int(opt.nit))

def metrics(pi,p,s,ref,tau):
    risk=np.sum(pi*(Q-p[...,None])**2,axis=-1)
    riskref=np.sum(ref*(Q-p[...,None])**2,axis=-1)
    m=pi.reshape(pi.shape[:-1]+(21,3)).sum(-1)
    mr=ref.reshape(ref.shape[:-1]+(21,3)).sum(-1)
    tv=.5*np.abs(m-tau).sum(-1)
    tvref=.5*np.abs(mr-tau).sum(-1)
    sty=.8*(2*s[...,None]-1)*Z+.25*(2*Q-1)*(2*s[...,None]-1)*Z
    return dict(risk=float(risk.mean()),ECR=float((risk-riskref).mean()),
                target_TV=float(tv.mean()),PCD=float((tv-tvref).mean()),
                reward=float(np.sum(pi*reward(p,s),axis=-1).mean()),
                style=float(np.sum(pi*sty,axis=-1).mean()),
                mean_report_bias=float(np.mean(np.sum(pi*Q,axis=-1)-p)))

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def learned_experiments(seeds,ntrain,ntest):
    rows=[];fits=[]
    for seed in range(seeds):
        start=time.time()
        p,s,df,y,chosen=generate_pairs(1000+seed,ntrain)
        th,fit=fit_dpo(df,y)
        sh,sfit=fit_sft(p,s,chosen)
        pv,sv,dv,yv,_=generate_pairs(9000+seed,4000)
        margin=BETA*(dv@th)
        fit['heldout_pair_NLL']=float(np.mean(np.logaddexp(0,margin)-yv*margin))
        rng=np.random.default_rng(2000+seed)
        px=.05+.9*rng.beta(.8,.8,ntest)
        sg=np.linspace(.05,.95,9)
        p=np.repeat(px,len(sg));s=np.tile(sg,ntest)
        ref=np.exp(ref_log(p));tau=target(p)
        dpo=policy(th,p,s)
        audit=(tau[...,None]*np.array([.25,.5,.25])).reshape(-1,len(Q))
        full=project(dpo,tau)
        # The same noisy audit is shared across counterfactual stances for each x.
        phat=(rng.binomial(32,px)+.5)/33
        noisytau=target(np.repeat(phat,len(sg)))
        noisy_audit=(noisytau[...,None]*np.array([.25,.5,.25])).reshape(-1,len(Q))
        nearest=np.zeros_like(tau)
        nearest[np.arange(len(p)),np.argmin((np.linspace(0,1,21)-p[:,None])**2,axis=1)]=1.
        dpo_m=dpo.reshape(ntest,len(sg),21,3).sum(-1)
        pooled=softmax(np.log(np.maximum(dpo_m,1e-300)).mean(axis=1))
        methods={'Reference':ref,'DPO':dpo,'SFT-chosen':policy(sh,p,s),
                 'Audit-only':audit,'CAEA-0.5':project(dpo,tau,.5),'CAEA-1':full,
                 'CAEA-noisy32':project(dpo,noisytau),
                 'Audit-only-noisy32':noisy_audit,
                 'Oracle-nearest':project(dpo,nearest),
                 'Pooled-marginal':project(dpo,np.repeat(pooled,len(sg),axis=0)),
                 'Uniform-coverage':project(dpo,np.full_like(tau,1/21))}
        for penalty in [.5,2.,8.]:
            methods[f'Soft-score-{penalty:g}']=softmax(np.log(np.maximum(dpo,1e-300))-penalty*(Q-p[:,None])**2/BETA)
        for name,pi in methods.items():
            item=dict(seed=seed,method=name,**metrics(pi,p,s,ref,tau))
            item['KL_to_DPO']=float(kl(pi,dpo).mean())
            rows.append(item)
        # Diagnostics are not self-fulfilling performance tests: verify identities
        # and compare a completely independent optimizer below.
        marginal=full.reshape(-1,21,3).sum(-1)
        fit['max_marginal_residual']=float(np.max(np.abs(marginal-tau)))
        fit['max_conditional_residual']=float(np.max(np.abs(
            full.reshape(-1,21,3)/tau[...,None]
            -dpo.reshape(-1,21,3)/dpo.reshape(-1,21,3).sum(-1)[...,None])))
        fits.append(dict(seed=seed,theta=th.tolist(),sft_theta=sh.tolist(),dpo_fit=fit,sft_fit=sfit))
        print(f'seed {seed}: DPO heldout NLL={fit["heldout_pair_NLL"]:.5f}; {time.time()-start:.1f}s',flush=True)
    write_csv(ROOT/'results/learned_per_seed.csv',rows)
    (ROOT/'results/fits.json').write_text(json.dumps(fits,indent=2))
    metrics_names=[k for k in rows[0] if k not in ('seed','method')]
    summary=[]
    for name in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==name]
        item=dict(method=name,n=seeds)
        for key in metrics_names:
            a=np.array([r[key] for r in rr]);mean=a.mean()
            hw=float(student_t.ppf(.975,seeds-1)*a.std(ddof=1)/np.sqrt(seeds)) if seeds>1 else 0.
            item[key]=float(mean);item[key+'_ci95']=hw
        summary.append(item)
    write_csv(ROOT/'results/learned_summary.csv',summary)
    return summary

def phase_experiments():
    rows=[]
    # Mean-response dynamics, not sampled dialogues or observed human behavior.
    u0=np.array([.47,.53]);eta=.25;T=400
    for a in [0.,1.,1.9,2.,2.1,3.,4.,6.]:
        for lam in [0.,.25,.5,.75,1.]:
            u=u0.copy()
            for t in range(T):
                m=(1-lam)*expit(a*(2*u-1))+lam*.5
                u=(1-eta)*u+eta*m
            # Binary report risk at p=.5 is exactly .25 for any mixture.
            pi=expit(a*(2*u0-1));m=(1-lam)*pi+lam*.5
            cost=np.mean(m*np.log(m/pi)+(1-m)*np.log((1-m)/(1-pi)))
            oracle_var=float(u0.var()*(1-eta)**(2*T))
            rows.append(dict(a=a,lam=lam,steps=T,eta=eta,initial_variance=float(u0.var()),
                             final_variance=float(u.var()),BPD=float(u.var()-oracle_var),
                             raw_variance_drift=float(u.var()-u0.var()),
                             user_MSE=float(np.mean((u-.5)**2)),
                             one_step_Brier_regret=.25,aggregate_report_p1=float(m.mean()),
                             center_multiplier=1-eta+eta*(1-lam)*a/2,
                             repair_KL=float(cost),u_low=float(u[0]),u_high=float(u[1])))
    write_csv(ROOT/'results/phase.csv',rows)
    # Dense phase surface, all settings retained, no selection by outcome.
    aa=np.linspace(0,6,121);ll=np.linspace(0,1,81)
    A,L=np.meshgrid(aa,ll)
    u=np.stack([np.full(A.shape,.47),np.full(A.shape,.53)],-1)
    for _ in range(400):
        u=.75*u+.25*((1-L[...,None])*expit(A[...,None]*(2*u-1))+.5*L[...,None])
    var=u.var(axis=-1)
    np.savez_compressed(ROOT/'results/phase_surface.npz',a=aa,lam=ll,variance=var)
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.4))
    im=axs[0].pcolormesh(aa,ll,var,shading='auto',cmap='magma',rasterized=True,linewidth=0,antialiased=False)
    av=np.linspace(2.001,6,300)
    axs[0].plot(av,1-2/av,'c--',lw=1.8,label='Local threshold')
    axs[0].set(xlabel='Agreement pressure a = k / beta',ylabel='Repair fraction lambda')
    axs[0].legend(frameon=False,loc='upper left');fig.colorbar(im,ax=axs[0],label='Final user variance')
    for lam in [0,.25,.5,1.]:
        u=np.array([.47,.53]);hist=[u.copy()]
        for _ in range(60):
            u=.75*u+.25*((1-lam)*expit(4*(2*u-1))+.5*lam);hist.append(u.copy())
        h=np.array(hist)
        axs[1].plot(h[:,1],label=f'lambda={lam:g}')
        axs[1].plot(h[:,0],color=axs[1].lines[-1].get_color(),alpha=.65)
    axs[1].set(xlabel='Interaction round',ylabel='Mean-response user belief',ylim=(0,1))
    axs[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures/phase.pdf');fig.savefig(ROOT/'figures/phase.png',dpi=180);plt.close(fig)
    return rows

def audit_experiments():
    rng=np.random.default_rng(781)
    tau=target(np.array(.73));rows=[]
    for n in [8,16,32,64,128,256,512,1024]:
        # Dirichlet smoothing makes all classes possible, keeping KL finite.
        samples=rng.multinomial(n,tau,size=2000)
        alpha=.1;hat=(samples+alpha)/(n+21*alpha)
        tv=.5*np.abs(hat-tau).sum(-1)
        bound=min(1.,.5*np.sqrt(20/n)+21*alpha/(n+21*alpha))
        rows.append(dict(n=n,repetitions=2000,mean_TV=float(tv.mean()),
                         TV_MC_se=float(tv.std(ddof=1)/np.sqrt(len(tv))),expected_TV_bound=bound,
                         mean_forecast_error=float(np.abs(hat@np.linspace(0,1,21)-tau@np.linspace(0,1,21)).mean())))
    write_csv(ROOT/'results/audit.csv',rows)
    # Wrong auditors can be invariant and stable while epistemically wrong.
    bad=[]
    for ptrue in [.2,.5,.8]:
        for paudit in [.2,.5,.8]:
            mu=float(target(np.array(paudit))@np.linspace(0,1,21))
            u=np.array([.2,.8])
            for _ in range(100):u=.75*u+.25*mu
            bad.append(dict(p_true=ptrue,p_audit=paudit,mean_report=mu,
                            final_variance=float(u.var()),final_MSE=float(np.mean((u-ptrue)**2))))
    write_csv(ROOT/'results/wrong_auditor.csv',bad)
    return rows

def independent_verification():
    rng=np.random.default_rng(42);errs=[]
    # Independent constrained SLSQP solves the full response-level problem.
    for _ in range(20):
        p=rng.dirichlet(np.ones(12));tau=rng.dirichlet(np.ones(4))
        marginal=p.reshape(4,3).sum(-1)
        closed=(tau[:,None]*p.reshape(4,3)/marginal[:,None]).ravel()
        A=np.kron(np.eye(4),np.ones((1,3)))
        def fun(z):return np.sum(z*np.log(z/p)),np.log(z/p)+1
        opt=minimize(fun,closed*.99+.01*np.repeat(tau/3,3),jac=True,method='SLSQP',
                     bounds=[(1e-12,1)]*12,
                     constraints=[dict(type='eq',fun=lambda z:A@z-tau,jac=lambda z:A)],
                     options={'ftol':1e-12,'maxiter':1000})
        if not opt.success:raise RuntimeError(opt.message)
        errs.append(abs(float(opt.fun)-float(kl(closed,p))))
        assert np.max(np.abs(A@closed-tau))<1e-12
        assert abs(float(kl(closed,p))-float(kl(tau,marginal)))<1e-12
    # Negative controls against the original manuscript's universal claims.
    corrected=dict(projection_max_objective_gap=max(errs),independent_solves=len(errs),
        gaussian_counterexample=dict(s=1.,reference_mean=0.,reference_variance=1.,
            k=1.,beta=2.,optimal_mean=.5,optimal_variance=.5,
            evidence_target_mean=.5,evidence_target_variance=.5,KL=0.),
        original_BPD_moderation_example=abs(.2-.8)/.8,
        exact_full_repair_contraction_per_round=.75,
        unchanged_assistant_parameters_during_feedback=True)
    (ROOT/'results/verification.json').write_text(json.dumps(corrected,indent=2))
    return corrected

def plots(summary,audit):
    ss={r['method']:r for r in summary}
    names=['Reference','DPO','SFT-chosen','Audit-only','CAEA-0.5','CAEA-1','CAEA-noisy32','Pooled-marginal','Uniform-coverage']
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.6))
    for ax,key,title in zip(axs,['risk','style','target_TV'],['Proper-score regret','Style preference reward','Conditional target mismatch']):
        vals=[ss[n][key] for n in names];err=[ss[n][key+'_ci95'] for n in names]
        ax.bar(np.arange(len(names)),vals,yerr=err,color=['#8397a6' if 'CAEA' not in n else '#218c85' for n in names],capsize=2)
        ax.set_xticks(np.arange(len(names)),names,rotation=65,ha='right',fontsize=7);ax.set_title(title,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'figures/learned.pdf');fig.savefig(ROOT/'figures/learned.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.4))
    for path,label,color in [(['DPO','CAEA-0.5','CAEA-1'],'CAEA mixture path','#218c85'),
                             (['DPO','Soft-score-0.5','Soft-score-2','Soft-score-8'],'Soft-score settings','#c66d29')]:
        axs[0].plot([ss[n]['KL_to_DPO']for n in path],[ss[n]['risk']for n in path],
                    'o-',lw=1.1,ms=4,label=label,color=color)
    for name,short in [('Reference','Reference'),('Audit-only','Audit-only'),('SFT-chosen','SFT'),('Uniform-coverage','Uniform')]:
        r=ss[name];axs[0].scatter(r['KL_to_DPO'],r['risk'],s=25,color='#657d94')
        offset=(-38,-10) if name=='Uniform-coverage'else(3,4)
        axs[0].annotate(short,(r['KL_to_DPO'],r['risk']),fontsize=7,xytext=offset,textcoords='offset points')
    axs[0].legend(frameon=False,fontsize=7,loc='upper left')
    axs[0].set(xlabel='KL distance from learned DPO',ylabel='Proper-score regret')
    axs[1].loglog([r['n'] for r in audit],[r['mean_TV'] for r in audit],'o-',label='Measured mean target TV')
    axs[1].loglog([r['n'] for r in audit],[r['expected_TV_bound'] for r in audit],'--',label='Conservative expectation bound')
    axs[1].set(xlabel='Audited class samples per context',ylabel='Target estimation error')
    axs[1].legend(frameon=False,fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'figures/tradeoff.pdf');fig.savefig(ROOT/'figures/tradeoff.png',dpi=180);plt.close(fig)

def latex_tables(summary):
    keys=['risk','style','target_TV','KL_to_DPO']
    lines=[]
    for r in summary:
        cells=[r['method'].replace('-','\\mbox{-}')]
        cells += [f"{r[k]:.4f} $\\pm$ {r[k+'_ci95']:.4f}" for k in keys]
        lines.append(' & '.join(cells)+r' \\')
    (ROOT/'paper/table_results.tex').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=10)
    ap.add_argument('--ntrain',type=int,default=24000);ap.add_argument('--ntest',type=int,default=600)
    args=ap.parse_args();start=time.time()
    for d in ['results','figures','logs']:(ROOT/d).mkdir(exist_ok=True)
    config=dict(seeds=args.seeds,ntrain=args.ntrain,ntest=args.ntest,holdout_pairs=4000,
                beta=BETA,k=K,q_classes=21,styles=3,target_temp=TARGET_TEMP,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                experiment_type='finite-action log-linear preference learning and synthetic interaction',
                language_model_training=False,human_subjects=False)
    (ROOT/'results/config.json').write_text(json.dumps(config,indent=2))
    verification=independent_verification()
    summary=learned_experiments(args.seeds,args.ntrain,args.ntest)
    phase_experiments();audit=audit_experiments();plots(summary,audit);latex_tables(summary)
    (ROOT/'results/run_status.json').write_text(json.dumps(dict(completed=True,seconds=time.time()-start,verification=verification),indent=2))
    print(json.dumps(dict(completed=True,seconds=time.time()-start),indent=2))

if __name__=='__main__':main()
