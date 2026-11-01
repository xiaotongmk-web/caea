"""Build manuscript tables and provenance from completed outputs only."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from scipy.stats import t
from experiment import ROOT

def rows(name):return list(csv.DictReader(open(ROOT/'results'/name)))
def stat(rr,k):
    a=np.array([float(r[k])for r in rr]);m=a.mean()
    hw=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0.
    return m,hw

def main():
    body=[]
    for r in rows('learned_summary.csv'):
        cell=[r['method']]
        for k in ['risk','style','target_TV','KL_to_DPO']:
            value=float(r[k])
            if abs(value)<.00005:value=0.
            cell.append(f"{value:.4f} $\\pm$ {float(r[k+'_ci95']):.4f}")
        body.append(' & '.join(cell)+r' \\')
    (ROOT/'paper/table_results.tex').write_text(
        r'\begin{tabular}{lrrrr}\toprule'+'\n'+
        r'Method & Proper-score regret $\downarrow$ & Style reward $\uparrow$ & Target TV $\downarrow$ & KL to DPO\\\midrule'+'\n'+
        '\n'.join(body)+'\n'+r'\bottomrule\end{tabular}'+'\n')
    neural=rows('neural_per_seed.csv');table=[];flat=[]
    for model in dict.fromkeys(r['model']for r in neural):
        short=model.split('/')[-1].replace('-Instruct','')
        for name in ['LM-base','Head-DPO','Audit-only','CAEA-0.5','CAEA-1']:
            rr=[r for r in neural if r['model']==model and r['method']==name]
            cell=[short if name=='LM-base' else '',name]
            item=dict(model=model,method=name)
            for k in ['risk','style_match','target_TV','KL_to_head_DPO']:
                m,hw=stat(rr,k);item[k]=float(m);item[k+'_ci95']=float(hw)
                cell.append(f'{m:.4f} $\\pm$ {hw:.4f}')
            flat.append(item);table.append(' & '.join(cell)+r' \\')
    (ROOT/'paper/table_neural.tex').write_text(
        r'\begin{tabular}{llrrrr}\toprule'+'\n'+
        r'Frozen backbone & Selector & Report regret & Style match & Target TV & KL to head-DPO\\\midrule'+'\n'+
        '\n'.join(table)+'\n'+r'\bottomrule\end{tabular}'+'\n')
    (ROOT/'results/neural_aggregate.json').write_text(json.dumps(flat,indent=2))
    # Manuscript prose is maintained in paper/neural_section.tex.
    # A machine-readable inventory supports review; it is not a fabricated
    # experiment registry or a preregistration.
    inventory=[]
    for p in sorted((ROOT/'results').glob('*')):
        if p.is_file():inventory.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (ROOT/'results_inventory.json').write_text(json.dumps(inventory,indent=2))
    print('Completed-output manuscript tables and inventory built')

if __name__=='__main__':main()
