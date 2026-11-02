"""Publication tables/figure from raw-score analysis; no model inference."""
import csv,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scale_study import ROOT,OUT,MODELS,load_rows,write_csv
NAMES={'qwen15':'Qwen 1.5B','smol17':'SmolLM2 1.7B','qwen7':'Qwen 7B'}
def read(p):
 op=gzip.open if str(p).endswith('.gz')else open
 with op(p,'rt')as f:return list(csv.DictReader(f))
def main():
 summaries={d:read(ROOT/'results'/d/'summary.csv')for d in ['scale','external_math']};refit=read(OUT/'auditor_refit_intervals.csv');mass=[]
 for d in summaries:
  for k in MODELS:
   rows=load_rows(ROOT/'results'/d/f'predictions_{k}.jsonl.gz');a=np.array([r['candidate_mass']for r in rows]);mass.append(dict(dataset=d,model=k,requests=len(rows),minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),valid_greedy_letter_fraction=float(np.mean([r['selected_token']in r['choice_token_ids']for r in rows]))))
 write_csv(OUT/'candidate_mass.csv',mass)
 lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Task & Backbone & Base $\mathcal R$ & Half $\mathcal R$ & $\Delta\mathcal R$ [95\%] & Stance gap (base/half)\\',r'\midrule']
 for d,task in [('scale','Bayes'),('external_math','Math probe')]:
  for k in MODELS:
   get=lambda m:next(r for r in summaries[d]if r['model']==k and r['condition']=='base'and r['method']==m)
   b=get('LM-base');h=get('CAEA-0.5-calibrated');iv=next(r for r in refit if r['dataset']==d and r['model']==k and r['condition']=='base'and float(r['fraction'])==.5)
   lines.append(f"{task} & {NAMES[k]} & {float(b['report_regret']):.4f} & {float(h['report_regret']):.4f} & {float(h['paired_regret_difference']):+.4f} [{float(iv['ci95_low']):+.4f}, {float(iv['ci95_high']):+.4f}] & {float(b['mean_stance_gap']):.3f}/{float(h['mean_stance_gap']):.3f}\\\\")
  lines.append(r'\midrule')
 lines[-1]=r'\bottomrule';lines.append(r'\end{tabular}');(ROOT/'paper/table_scale.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Task & Constant & Raw audit & Calibrated & Nearest & Oracle\\',r'\midrule']
 for d,task in [('scale','Bayes'),('external_math','Math probe')]:
  methods=['Training-mean-audit','CAEA-full-raw-audit','CAEA-1-calibrated','CAEA-nearest-calibrated','Oracle-full'];a=[float(next(r for r in summaries[d]if r['model']=='qwen15'and r['condition']=='base'and r['method']==m)['report_regret'])for m in methods];lines.append(task+' & '+' & '.join(f'{v:.4f}'for v in a)+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}']);(ROOT/'paper/table_auditor.tex').write_text('\n'.join(lines)+'\n')
 fig,axs=plt.subplots(2,2,figsize=(9.2,5.8),sharey='row');rng=np.random.default_rng(2026110267)
 methods=[('LM-base','Base','#244668',-.18),('CAEA-0.5-calibrated','Half repair','#d47a27',0),('CAEA-1-calibrated','Full / audit-only','#33877c',.18)]
 for row,d in enumerate(summaries):
  data=read(ROOT/'results'/d/'per_prompt_metrics.csv.gz')
  for col,c in enumerate(['base','permuted']):
   ax=axs[row,col]
   for m,label,color,offset in methods:
    means=[];lows=[];highs=[]
    for k in MODELS:
     groups={}
     for r in data:
      if r['model']==k and r['condition']==c and r['method']==m:groups.setdefault(r['case_id'],[]).append(float(r['report_regret']))
     a=np.array([np.mean(v)for v in groups.values()]);ix=rng.integers(0,len(a),(5000,len(a)));lo,hi=np.quantile(a[ix].mean(1),[.025,.975]);means.append(a.mean());lows.append(a.mean()-lo);highs.append(hi-a.mean())
    ax.errorbar(np.arange(3)+offset,means,yerr=[lows,highs],fmt='o',ms=4,capsize=2,color=color,lw=1,label=label)
   const=float(next(r for r in summaries[d]if r['model']=='qwen15'and r['condition']==c and r['method']=='Training-mean-audit')['report_regret']);ax.axhline(const,color='#777',linestyle='--',lw=.9,label='Constant audit')
   ax.set_xticks(range(3),['Qwen 1.5B','Smol 1.7B','Qwen 7B']);ax.set_title(('Bayesian test (128 contexts)'if d=='scale'else'Math probes (32 questions)')+' / '+c,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
   if col==0:ax.set_ylabel('Expected report regret')
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc='lower center',frameon=False,fontsize=9);fig.tight_layout(rect=(0,.06,1,1));fig.savefig(ROOT/'figures/scale_audit.pdf');fig.savefig(ROOT/'figures/scale_audit.png',dpi=170);plt.close(fig);print('Publication tables and figure completed')
if __name__=='__main__':main()
