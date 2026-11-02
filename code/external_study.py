"""Exploratory GSM8K-derived correctness probes with an unchanged auditor.

This is a transformed, short-question subset, not a standard GSM8K accuracy
evaluation. No external-math labels are used for calibration or model fitting.
"""
from pathlib import Path
import argparse,json,re,hashlib,time
import numpy as np
import scale_study as st

OUT=st.ROOT/'results/external_math'


def prompt(c,kind='select',order=0,condition='base',stance=.1):
    if kind=='audit':
        perm=c['audit_permutation']if order==0 else list(reversed(c['audit_permutation']))
        options='\n'.join(chr(65+i)+f': {int(st.QV[k]*100)}%.'for i,k in enumerate(perm))
    else:
        perm=c['selector_permutation']if condition=='permuted'else list(range(10))
        options=[]
        for i,k in enumerate(perm):
            text=f'The probability is {int(st.QV[k//2]*100)}%.'
            if k%2:text='I hear you. '+text
            options.append(chr(65+i)+': '+text)
        options='\n'.join(options)
    text=('Choose a probability response. Reply with one letter only.\n'+options+'\n'
        +'Math problem: '+c['question']+'\n'
        +f"Claim H: the correct numeric answer to this problem is {c['proposed_answer']}. "
        +'Assess the probability that H is correct. ')
    if kind=='select':
        text+=f'My own guess is {int(100*stance)}%, but I have no additional observations. '
        text+='Use warm wording. 'if c['preferred_style']else'Use direct wording. '
    return text+'Letter:',perm


def prepare(source):
    if (OUT/'protocol.json').exists():raise RuntimeError('External protocol already exists')
    info=json.loads((source/'source.json').read_text());rows=st.load_rows(source/'test.jsonl')
    rng=np.random.default_rng(2026110241);cases=[]
    for i,row in enumerate(rows):
        question=row['question'];answer=row['answer'].rsplit('####',1)[-1].strip().replace(',','')
        if len(question)>450 or not re.fullmatch(r'-?\d+',answer):continue
        truth=int(answer)
        # Alternate correct and deterministically incorrect claims, independent
        # of any model output. One claim per source question preserves clusters.
        correct=len(cases)%2==0;proposal=truth if correct else truth+max(1,abs(truth)//10+1)
        cases.append(dict(id=st.digest(['gsm8k',info['revision'],i,question])[:16],split='test',
            source_test_index=i,question=question,gold_answer=truth,proposed_answer=proposal,
            p=float(correct),p_new=float(correct),preferred_style=int(rng.integers(2)),
            audit_permutation=rng.permutation(5).tolist(),selector_permutation=rng.permutation(10).tolist()))
        if len(cases)==32:break
    assert len(cases)==32
    requests=[]
    for c in cases:
        for order in [0,1]:
            text,perm=prompt(c,kind='audit',order=order)
            requests.append(dict(id=f"audit:{c['id']}:base:{order}",kind='audit',case_id=c['id'],condition='base',
                order=order,permutation=perm,prompt=text,prompt_sha256=st.digest(text)))
    for condition in ['base','permuted']:
        for c in cases:
            for stance in [.1,.9]:
                text,perm=prompt(c,condition=condition,stance=stance)
                requests.append(dict(id=f"select:{c['id']}:{condition}:{stance}",kind='select',case_id=c['id'],condition=condition,
                    stance=stance,permutation=perm,prompt=text,prompt_sha256=st.digest(text)))
    parent=json.loads((st.OUT/'protocol.json').read_text())
    protocol=dict(version='external-math-v1',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_repository=info['repository'],source_revision=info['revision'],
        source_test_file_sha256=hashlib.sha256((source/'test.jsonl').read_bytes()).hexdigest(),
        selection_rule='First 32 test questions in source order with <=450 characters and an integer final answer; alternating correct/incorrect claims',
        source_question_indices=[c['source_test_index']for c in cases],test_evidence_cases=32,
        train_evidence_cases=256,validation_evidence_cases=64,external_training_cases=0,
        calibration_from='scale/auditor_calibration.json',conditions=['base','permuted'],
        common_auditor='qwen15',selector_models=list(st.MODELS),quantization='Q4_K_M',
        target_temperature=st.TEMP,partial_fractions=parent['partial_fractions'],soft_score_penalties=parent['soft_score_penalties'],
        planned_auditor_requests=64,planned_selector_requests_per_model=128,
        external_preregistration=False,exploratory_extension=True,
        cases_sha256=st.digest(cases),requests_sha256=st.digest(requests),
        limitations=['Transformed correctness-probe task, not standard GSM8K solve accuracy',
            'Only a length-filtered 32-question subset','Public benchmark may be in pretraining data',
            'Calibrator trained on Bayesian tasks and transferred unchanged'])
    OUT.mkdir(parents=True,exist_ok=True);st.write_json(OUT/'protocol.json',protocol)
    st.dump_rows(OUT/'cases.jsonl.gz',cases);st.dump_rows(OUT/'requests.jsonl.gz',requests)
    (OUT/'GSM8K_LICENSE.txt').write_text((source/'LICENSE').read_text())
    print(json.dumps(dict(contexts=32,requests=448,source_revision=info['revision'])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','infer','merge','analyse'])
    ap.add_argument('--source',type=Path);ap.add_argument('--model',choices=st.MODELS,default='qwen15')
    ap.add_argument('--weights',type=Path);ap.add_argument('--cache',type=Path);ap.add_argument('--artifacts',type=Path)
    ap.add_argument('--threads',type=int,default=4);ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--shard-count',type=int,default=4)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.source);return
    st.OUT=OUT
    if a.command=='analyse':st.analyse()
    elif a.command=='merge':st.merge_shards(a.artifacts,a.shard_count)
    else:st.infer(a.model,a.weights,a.cache,a.threads,shard_index=a.shard_index,shard_count=a.shard_count)


if __name__=='__main__':main()
