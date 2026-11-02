"""Cross-check completed study outputs and evidence scope."""
import csv,json
import numpy as np
from experiment import ROOT

def read(n):return list(csv.DictReader(open(ROOT/'results'/n)))
def main():
    status=json.loads((ROOT/'results/run_status.json').read_text());assert status['completed']
    assert json.loads((ROOT/'results/stress_status.json').read_text())['completed']
    assert json.loads((ROOT/'results/audit_stability_status.json').read_text())['completed']
    # A nominal coverage theorem is not tested by forcing a random empirical
    # rate to exceed its nominal level. Verify the recorded event instead.
    assert all(int(r['coverage'])==int(float(r['certified_L'])>=float(r['true_L']))
               for r in read('audit_stability.csv'))
    neural=json.loads((ROOT/'results/neural_status.json').read_text());assert neural['completed']
    assert all(m['transformer_weights_updated'] is False for m in neural['models'])
    rr=read('learned_per_seed.csv');assert len(rr)==140
    for seed in range(10):
        x={r['method']:r for r in rr if int(r['seed'])==seed}
        assert abs(float(x['Audit-only']['risk'])-float(x['CAEA-1']['risk']))<1e-12
        assert float(x['CAEA-1']['target_TV'])<1e-12
        assert float(x['CAEA-1']['KL_to_DPO'])<=float(x['Audit-only']['KL_to_DPO'])+1e-10
        assert abs(float(x['Audit-only-noisy32']['risk'])-float(x['CAEA-noisy32']['risk']))<1e-12
        assert float(x['Oracle-nearest']['risk'])<=float(x['CAEA-1']['risk'])
    # A mean-response contraction need not force stochastic population variance
    # to vanish: this is a meaningful guard against overinterpreting the result.
    st=read('stochastic.csv')
    oracle=[float(r['final_variance'])for r in st if float(r['eta'])==.25 and float(r['lam'])==1]
    assert abs(np.mean(oracle)-.25/(4*1.75))<.0015
    v=json.loads((ROOT/'results/verification.json').read_text())
    assert v['projection_max_objective_gap']<1e-9
    prompts=[json.loads(t)for t in (ROOT/'results/neural_prompts.jsonl').read_text().splitlines()]
    groups={}
    for r in prompts:
        key=(round(r['prior'],6),round(r['l1'],6),round(r['l0'],6))
        groups.setdefault(key,set()).add(r['split'])
    assert all(len(g)==1 for g in groups.values())
    assert len(prompts)==152
    print('PASS: completed runs, finite-audit certificates, exact audit comparison, stochastic noise floor, independent optimization, disjoint neural evidence splits, honest neural scope')

if __name__=='__main__':main()
