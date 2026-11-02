"""Offline verification of external source claims, native scores and calibration."""
import csv,gzip,json
import numpy as np
from collections import Counter
from external_study import OUT,prompt
from scale_study import ROOT,MODELS,digest,load_rows

def main():
 p=json.loads((OUT/'protocol.json').read_text());cases=load_rows(OUT/'cases.jsonl.gz');req=load_rows(OUT/'requests.jsonl.gz');src=load_rows(OUT/'source_selected.jsonl.gz')
 assert len(cases)==32 and len({c['source_test_index']for c in cases})==32
 assert Counter(c['p']for c in cases)=={0.:16,1.:16}
 assert digest(cases)==p['cases_sha256']and digest(req)==p['requests_sha256']and len(req)==192
 for i,(c,s)in enumerate(zip(cases,src)):
  assert s['question']==c['question']and s['source_test_index']==c['source_test_index']
  gold=int(s['answer'].rsplit('####',1)[-1].strip().replace(',',''));assert gold==c['gold_answer']
  assert c['p']==float(i%2==0)==float(c['proposed_answer']==gold)
  assert c['proposed_answer']==(gold if c['p']else gold+max(1,abs(gold)//10+1))
 byid={c['id']:c for c in cases}
 for r in req:
  text,perm=prompt(byid[r['case_id']],kind=r['kind'],order=r.get('order',0),condition=r['condition'],stance=r.get('stance',.1))
  assert text==r['prompt']and perm==r['permutation']and digest(text)==r['prompt_sha256']
 a=json.loads((OUT/'auditor_calibration.json').read_text());b=json.loads((ROOT/'results/scale/auditor_calibration.json').read_text())
 for k in ['selected','all_validation_results','training_mean']:assert a[k]==b[k]
 assert not a['test_truth_used_for_fit']
 for key in MODELS:
  rows=load_rows(OUT/f'predictions_{key}.jsonl.gz');expected={r['id']:r for r in req if key=='qwen15'or r['kind']=='select'}
  assert len(rows)==len(expected)and{r['id']for r in rows}==set(expected)
  for r in rows:
   assert r['prompt_sha256']==expected[r['id']]['prompt_sha256']and np.isfinite(r['logits']).all()and 0<=r['candidate_mass']<=1+1e-10
   if r['selected_token']in r['choice_token_ids']:assert r['selected_token']==r['choice_token_ids'][int(np.argmax(r['logits']))]
 with gzip.open(OUT/'per_prompt_metrics.csv.gz','rt')as f:metrics=list(csv.DictReader(f))
 groups={}
 for r in metrics:groups.setdefault((r['model'],r['case_id'],r['condition'],r['stance']),{})[r['method']]=r
 assert len(groups)==384
 for g in groups.values():
  full=float(g['CAEA-1-calibrated']['report_regret']);base=float(g['LM-base']['report_regret']);assert abs(full-float(g['Audit-only-calibrated']['report_regret']))<1e-12
  for fraction in [.25,.5,.75,1.]:assert abs(float(g[f'CAEA-{fraction:g}-calibrated']['report_regret'])-((1-fraction)*base+fraction*full))<1e-12
 record=dict(completed=True,independent_source_questions=32,fresh_inference_requests=448,inherited_calibrator_unchanged=True,external_labels_used_for_fitting=False,source_solution_and_claim_check='passed',prompt_fingerprints='passed',repair_identities='passed')
 (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':main()
