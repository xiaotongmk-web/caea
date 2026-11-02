"""Independent consistency and leakage checks for the expanded study.

No model downloads or inference are needed for this verifier.
"""
import copy,json,gzip
import numpy as np
from scale_study import (OUT,MODELS,QV,load_rows,digest,fit_auditor,calibrated,softmax)


def verify_protocol():
    p=json.loads((OUT/'protocol.json').read_text());cases=load_rows(OUT/'cases.jsonl.gz');req=load_rows(OUT/'requests.jsonl.gz')
    assert digest(cases)==p['cases_sha256'] and digest(req)==p['requests_sha256']
    assert len({c['id']for c in cases})==len(cases)
    evidence={}
    for c in cases:
        k=c['prior'],c['l1'],c['l0'];assert k not in evidence;evidence[k]=c['split']
        truth=c['prior']*c['l1']/(c['prior']*c['l1']+(100-c['prior'])*c['l0'])
        assert abs(c['p']-truth)<1e-14
        assert abs(c['p_new']-truth*c['f1']/(truth*c['f1']+(1-truth)*c['f0']))<1e-14
    assert sum(c['split']=='test'for c in cases)==128
    assert sum(r['kind']=='audit'for r in req)==1152
    assert sum(r['kind']=='select'for r in req)==1024
    assert len({r['prompt_sha256']for r in req})==len(req)
    for r in req:
        assert digest(r['prompt'])==r['prompt_sha256']
        assert sorted(r['permutation'])==list(range(5 if r['kind']=='audit'else 10))
    return p,cases,req


def verify_completed():
    p,cases,req=verify_protocol()
    # Reconstruct the evidence-only auditor inputs from raw native scores.
    audit={r['id']:r for r in load_rows(OUT/'predictions_qwen15.jsonl.gz')};raw={}
    for c in cases:
        scores=[]
        for r in req:
            if r['kind']=='audit'and r['case_id']==c['id']and r['condition']=='base':
                scores.append(float(softmax(np.array(audit[r['id']]['logits']))@QV[r['permutation']]))
        assert len(scores)==2;raw[c['id'],'base']=float(np.mean(scores))
    fit,_,_=fit_auditor(cases,raw,p['calibration_regularization_grid'])
    altered=copy.deepcopy(cases)
    for c in altered:
        if c['split']=='test':c['p']=1-c['p'];c['p_new']=1-c['p_new']
    altered_fit,_,_=fit_auditor(altered,raw,p['calibration_regularization_grid'])
    assert fit==altered_fit,'Changing test labels changed auditor fitting'
    for key in MODELS:
        rows=load_rows(OUT/f'predictions_{key}.jsonl.gz');ids={r['id']for r in rows}
        expected={r['id']for r in req if key=='qwen15'or r['kind']=='select'}
        assert ids==expected and len(rows)==len(ids)
        meta=json.loads((OUT/f'model_{key}.json').read_text())
        assert meta['completed']and meta['fresh_inference']
        for r in rows:
            assert np.isfinite(r['logits']).all() and 0<=r['candidate_mass']<=1+1e-10
        # The sampling/score API must agree on the argmax whenever that argmax
        # is one of the scored choices (no sampling or penalties were used).
        assert all(r['selected_token']==r['choice_token_ids'][int(np.argmax(r['logits']))]
            for r in rows if r['selected_token']in r['choice_token_ids'])
    import csv
    with gzip.open(OUT/'per_prompt_metrics.csv.gz','rt') as f:metrics=list(csv.DictReader(f))
    groups={}
    for r in metrics:
        groups.setdefault((r['model'],r['case_id'],r['condition'],r['stance']),{})[r['method']]=r
        assert abs(float(r['report_regret'])-float(r['report_variance'])-float(r['mean_report_squared_error']))<1e-12
    for g in groups.values():
        assert abs(float(g['CAEA-1-calibrated']['report_regret'])-float(g['Audit-only-calibrated']['report_regret']))<1e-12
        assert float(g['CAEA-1-calibrated']['target_TV'])<1e-12
        for fraction in [.25,.5,.75,1.]:
            a=float(g[f'CAEA-{fraction:g}-calibrated']['report_regret'])
            b=(1-fraction)*float(g['LM-base']['report_regret'])+fraction*float(g['CAEA-1-calibrated']['report_regret'])
            assert abs(a-b)<1e-12
    # Reconstruct endpoint risks independently from the archived native logits.
    case_map={c['id']:c for c in cases}
    for c in cases:
        if c['split']!='test':continue
        values=[]
        for order in [0,1]:
            r=next(r for r in req if r['id']==f"audit:{c['id']}:new_evidence:{order}")
            values.append(float(softmax(np.array(audit[r['id']]['logits']))@QV[r['permutation']]))
        raw[c['id'],'new_evidence']=float(np.mean(values))
    for key in MODELS:
        predictions={r['id']:r for r in load_rows(OUT/f'predictions_{key}.jsonl.gz')}
        for r in req:
            if r['kind']!='select':continue
            c=case_map[r['case_id']];variant='new_evidence'if r['condition']=='new_evidence'else'base'
            truth=c['p_new']if variant=='new_evidence'else c['p']
            phat=float(calibrated(np.array(raw[c['id'],variant]),np.array(fit['weight'])))
            tau=softmax(-(QV-phat)**2/.03)
            base=softmax(np.array(predictions[r['id']]['logits']));q=QV[np.array(r['permutation'])//2]
            g=groups[key,c['id'],r['condition'],str(r['stance'])]
            assert abs(float(g['LM-base']['report_regret'])-base@((q-truth)**2))<1e-12
            assert abs(float(g['CAEA-1-calibrated']['report_regret'])-tau@((QV-truth)**2))<1e-12
    record=dict(completed=True,independent_evidence_cases=len(cases),test_contexts=128,
        fresh_inference_requests=sum(len(load_rows(OUT/f'predictions_{key}.jsonl.gz'))for key in MODELS),
        test_label_mutation_leakage_check='passed',posterior_formula_check='passed',
        raw_score_argmax_check='passed',risk_decomposition_check='passed',repair_identities_check='passed')
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':verify_completed()
