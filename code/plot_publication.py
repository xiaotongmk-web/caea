"""Redraw Figures 1, 3 and 4 from saved outputs; no training or data mutation.
Wide PDFs replace manuscript assets; *_singlecolumn.pdf are 8.8 cm wide.
"""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
def rows(n):
    with (ROOT/'results'/n).open() as f:return list(csv.DictReader(f))
def bump(s,c,w):
    z=(s-c)/w;v=np.zeros_like(z);ok=np.abs(z)<1;v[ok]=np.exp(1-1/(1-z[ok]**2));return v

def draw(name,compact):
    w=8.8/2.54 if compact else 6.8
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(2 if compact else 1,1 if compact else 2,figsize=(w,5.8 if compact else 2.8),layout='constrained')
    a,b=axes.flatten()
    if name=='phase':
        d=np.load(ROOT/'results/phase_surface.npz')
        im=a.pcolormesh(d['a'],d['lam'],d['variance'],shading='auto',cmap='magma',rasterized=True,linewidth=0,antialiased=False)
        av=np.linspace(2.001,6,300);a.plot(av,1-2/av,'c--',lw=1.3,label='Contraction boundary')
        a.set(xlabel='Agreement pressure a = k / beta',ylabel='Repair fraction lambda');a.legend(frameon=False,loc='upper left',labelcolor='white')
        fig.colorbar(im,ax=a,label='Final user variance',pad=.02)
        for lam in [0,.25,.5,1.]:
            u=np.array([.47,.53]);hist=[u.copy()]
            for _ in range(60):u=.75*u+.25*((1-lam)*expit(4*(2*u-1))+.5*lam);hist.append(u.copy())
            h=np.array(hist);line,=b.plot(h[:,1],label=f'lambda={lam:g}');b.plot(h[:,0],color=line.get_color(),alpha=.65)
        b.set(xlabel='Interaction round',ylabel='Mean-response user belief',ylim=(0,1));b.legend(frameon=False)
    elif name=='audit_stability':
        d=rows('audit_stability_summary.csv');raw=rows('audit_stability.csv')[0]
        a.semilogx([float(r['total_budget']) for r in d],[float(r['mean_repair_fraction']) for r in d],'o-',label='Finite-audit certificate')
        a.axhline(1-(1-float(raw['gamma']))/float(raw['true_L']),color='gray',ls='--',label='Known-L budget (same margin)')
        a.set(xlabel='Binary report audit budget',ylabel='Sufficient repair fraction',ylim=(.5,.9));a.legend(frameon=False)
        fail=json.loads((ROOT/'results/audit_indistinguishability.json').read_text());s=np.linspace(0,1,10001)
        m=.5+.37*(bump(s,.87,fail['bump_width'])-bump(s,.13,fail['bump_width']))
        b.plot(s,m,label='Smooth unaudited kernel',lw=1.1);b.scatter(fail['audit_points'],fail['audit_means'],s=15,c='black',label='Finite audit locations',zorder=4)
        fp=np.array(fail['off_grid_fixed_points']);b.scatter(fp,fp,s=20,c='red',label='Stable off-grid beliefs',zorder=5)
        b.plot([0,1],[0,1],ls=':',c='gray');b.set(xlabel='User stance',ylabel='Expected binary report');b.legend(frameon=False,loc='upper left')
    else:
        d={r['method']:r for r in rows('learned_summary.csv')}
        for ns,label,color in [(['DPO','CAEA-0.5','CAEA-1'],'CAEA mixture path','#218c85'),(['DPO','Soft-score-0.5','Soft-score-2','Soft-score-8'],'Soft-score settings','#c66d29')]:
            a.plot([float(d[n]['KL_to_DPO']) for n in ns],[float(d[n]['risk']) for n in ns],'o-',lw=1.1,ms=4,label=label,c=color)
        for n,label in [('Reference','Reference'),('Audit-only','Audit-only'),('SFT-chosen','SFT'),('Uniform-coverage','Uniform')]:
            r=d[n];a.scatter(float(r['KL_to_DPO']),float(r['risk']),s=20,c='#657d94');a.annotate(label,(float(r['KL_to_DPO']),float(r['risk'])),fontsize=7,xytext=(-32,-10) if n=='Uniform-coverage' else (3,4),textcoords='offset points')
        a.set(xlabel='KL distance from learned DPO',ylabel='Proper-score regret');a.legend(frameon=False,loc='upper left')
        rr=rows('audit.csv');b.loglog([float(r['n']) for r in rr],[float(r['mean_TV']) for r in rr],'o-',label='Measured mean target TV');b.loglog([float(r['n']) for r in rr],[float(r['expected_TV_bound']) for r in rr],'--',label='Conservative expectation bound')
        b.set(xlabel='Audited class samples per context',ylabel='Target estimation error');b.legend(frameon=False)
    out=ROOT/'figures'/(name+('_singlecolumn' if compact else ''))
    fig.savefig(out.with_suffix('.pdf'));fig.savefig(out.with_suffix('.png'),dpi=180);plt.close(fig)
if __name__=='__main__':
    for name in ['phase','audit_stability','tradeoff']:
        for compact in [False,True]:draw(name,compact)
