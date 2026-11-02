"""Pending BF16 CUDA replication of the same prompts; never rents hardware.

prepare/help run without Torch. Inference requires an actual CUDA GPU.
Outputs are separate from the completed Q4_K_M study. This is a numerical/
implementation robustness study, not additional independent test evidence.
"""
from pathlib import Path
import argparse,json,time,hashlib,shutil,platform,os
import scale_study as st
from gpu_io import read_validated_cache, validate_rows
SPECS={
'qwen15':('Qwen/Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
'smol17':('HuggingFaceTB/SmolLM2-1.7B-Instruct','31b70e2e869a7173562077fd711b654946d38674'),
'qwen7':('Qwen/Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28')}

def prepare(out):
 for dataset in ['scale','external_math']:
  src=st.ROOT/'results'/dataset;dest=out/dataset;dest.mkdir(parents=True,exist_ok=True)
  parent=json.loads((src/'protocol.json').read_text());p=dict(parent)
  p.update(version='bf16-replication-v1',parent_protocol_sha256=st.digest(parent),quantization=None,
   weight_dtype='bfloat16',runtime='transformers/PyTorch CUDA',model_revisions=SPECS,
   preparation_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),additional_independent_test_cases=0,
   precision_interpretation='HF/BF16 versus GGUF/Q4 pipeline robustness; runtime/template/base-revision differences can also contribute')
  if (dest/'protocol.json').exists():
   old=json.loads((dest/'protocol.json').read_text());p['preparation_utc']=old['preparation_utc'];assert p==old,'Refusing to replace a changed GPU protocol'
  else:st.write_json(dest/'protocol.json',p)
  for name in ['cases.jsonl.gz','requests.jsonl.gz','GSM8K_LICENSE.txt','source_selected.jsonl.gz']:
   if (src/name).exists():shutil.copyfile(src/name,dest/name)
 print('BF16 plan prepared; no GPU rented, no GPU inference performed.')

def preflight(weights,minimum_free_gb):
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; no CPU fallback')
 if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable CUDA GPU required')
 weights.mkdir(parents=True,exist_ok=True)
 free=shutil.disk_usage(weights).free/1024**3
 if free<minimum_free_gb:raise RuntimeError(f'Only {free:.1f} GiB free; require {minimum_free_gb:g} GiB before model downloads')
 with torch.inference_mode():
  x=torch.ones((64,64),device='cuda',dtype=torch.bfloat16)
  if not torch.isfinite(x@x).all():raise RuntimeError('BF16 CUDA matrix multiplication failed')
 torch.cuda.synchronize()
 print(json.dumps(dict(cuda_ready=True,gpu=torch.cuda.get_device_name(0),
  vram_gib=torch.cuda.get_device_properties(0).total_memory/1024**3,
  free_disk_gib=free,torch=torch.__version__,cuda=torch.version.cuda)))

def verify(out,keys):
 records=[]
 for dataset in ['scale','external_math']:
  dest=out/dataset;p=json.loads((dest/'protocol.json').read_text())
  requests=st.load_rows(dest/'requests.jsonl.gz');cases=st.load_rows(dest/'cases.jsonl.gz')
  if st.digest(requests)!=p['requests_sha256']or st.digest(cases)!=p['cases_sha256']:
   raise ValueError('GPU input archive no longer matches the protocol')
  for key in keys:
   expected=[r for r in requests if key=='qwen15'or r['kind']=='select']
   meta=json.loads((dest/f'model_{key}.json').read_text())
   rows=st.load_rows(dest/f'predictions_{key}.jsonl.gz')
   validate_rows(rows,expected)
   if not meta.get('completed')or meta.get('smoke_test')or meta['protocol_sha256']!=st.digest(p):
    raise ValueError('Incomplete or incompatible GPU model metadata')
   if meta['revision']!=SPECS[key][1]or meta['model_key']!=key or meta['total_requests']!=len(rows):
    raise ValueError('GPU model provenance mismatch')
   records.append(dict(dataset=dataset,model=key,requests=len(rows),gpu=meta['gpu']))
 result=dict(completed=True,records=records,scope='Archive consistency, not independent authentication of GPU execution')
 st.write_json(out/'verification.json',result);print(json.dumps(result))

def infer(key,out,weights,limit=None):
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from huggingface_hub import snapshot_download,HfApi
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; CPU fallback is intentionally disabled')
 if not torch.cuda.is_bf16_supported():raise RuntimeError('The selected GPU does not support BF16')
 repo,rev=SPECS[key];weights.mkdir(parents=True,exist_ok=True)
 path=Path(snapshot_download(repo,revision=rev,local_dir=weights/key,
  allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer.model','vocab.*','merges.txt']))
 info=HfApi().model_info(repo,revision=rev,files_metadata=True);hashes={}
 for f in info.siblings:
  if not f.rfilename.endswith('.safetensors'):continue
  h=hashlib.sha256()
  with (path/f.rfilename).open('rb')as stream:
   for b in iter(lambda:stream.read(16*1024*1024),b''):h.update(b)
  actual=h.hexdigest();expected=f.lfs.sha256 if f.lfs is not None else None
  if expected is None or actual!=expected:raise RuntimeError('Unverified upstream tensor file: '+f.rfilename)
  hashes[f.rfilename]=actual
 torch.manual_seed(1739);torch.backends.cuda.matmul.allow_tf32=False
 tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,device_map={'':0},
  trust_remote_code=False,attn_implementation='eager').eval()
 for dataset in ['scale','external_math']:
  dest=out/dataset;p=json.loads((dest/'protocol.json').read_text());requests=st.load_rows(dest/'requests.jsonl.gz')
  if st.digest(requests)!=p['requests_sha256']:raise ValueError('Requests differ from the fixed protocol')
  requests=[r for r in requests if key=='qwen15'or r['kind']=='select']
  if limit is not None:
   # Cover both query paths in the auditor model's smoke test.
   audits=[r for r in requests if r['kind']=='audit']
   selects=[r for r in requests if r['kind']=='select']
   if audits and selects:
    n_audit=min(len(audits),(limit+1)//2)
    requests=audits[:n_audit]+selects[:limit-n_audit]
   else:requests=requests[:limit]
  meta=dict(model_key=key,repo=repo,revision=rev,weight_dtype='bfloat16',quantization=None,runtime='transformers/PyTorch CUDA',
   torch_version=torch.__version__,transformers_version=transformers.__version__,cuda_version=torch.version.cuda,
   gpu=torch.cuda.get_device_name(0),python=platform.python_version(),fresh_inference=True,transformer_weights_updated=False,
   protocol_sha256=st.digest(p),chat_template_sha256=st.digest(tokenizer.chat_template),weight_sha256=hashes,
   engine_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   validation_engine_sha256=hashlib.sha256(Path(__file__).with_name('gpu_io.py').read_bytes()).hexdigest(),
   batch_size=1,attention_implementation='eager',smoke_test=limit is not None,request_limit=limit)
  raw=dest/f'cache_{key}.jsonl';meta_path=dest/f'cache_{key}_metadata.json'
  if meta_path.exists():assert json.loads(meta_path.read_text())==meta,'Cache provenance changed; use a fresh output directory'
  else:st.write_json(meta_path,meta)
  rows=read_validated_cache(raw,requests);done={r['id']:r for r in rows}
  started=time.time()
  with raw.open('a',encoding='utf-8')as f:
   for i,r in enumerate(requests):
    if r['id']in done:continue
    choices=[tokenizer.encode(chr(65+j),add_special_tokens=False)for j in range(5 if r['kind']=='audit'else 10)]
    assert all(len(x)==1 for x in choices);ids=[x[0]for x in choices]
    messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',content=r['prompt'])]
    tokens=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
    if tokens.shape[1]>=512:raise RuntimeError('Prompt exceeds fixed context limit')
    torch.cuda.synchronize();t=time.perf_counter()
    with torch.inference_mode():logits=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=False).logits[0,-1,:].float()
    scores=logits[ids];row=dict(id=r['id'],prompt_sha256=r['prompt_sha256'],prompt_token_ids_sha256=st.digest(tokens[0].tolist()),
     logits=scores.cpu().tolist(),candidate_mass=float(torch.exp(torch.logsumexp(scores,0)-torch.logsumexp(logits,0)).item()),
     selected_token=int(logits.argmax().item()),choice_token_ids=ids,prompt_tokens=int(tokens.shape[1]),seconds=time.perf_counter()-t)
    validate_rows([row],[r]);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
    if (i+1)%32==0:print(dataset,key,i+1,len(requests),flush=True)
  rows=read_validated_cache(raw,requests);validate_rows(rows,requests)
  st.dump_rows(dest/f'predictions_{key}.jsonl.gz',rows);st.write_json(dest/f'model_{key}.json',dict(**meta,completed=limit is None,total_requests=len(rows),seconds_this_invocation=time.time()-started))
 print('Completed BF16 scoring for',key,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','preflight','infer','verify','analyse']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--weights',type=Path);ap.add_argument('--model',choices=SPECS,default='qwen7');ap.add_argument('--limit',type=int);ap.add_argument('--minimum-free-gb',type=float,default=35);a=ap.parse_args()
 if a.limit is not None and a.limit<1:ap.error('--limit must be positive')
 if a.command=='prepare':prepare(a.out)
 elif a.command=='preflight':
  if a.weights is None:ap.error('--weights is required')
  preflight(a.weights,a.minimum_free_gb)
 elif a.command=='infer':
  if a.weights is None:ap.error('--weights is required')
  infer(a.model,a.out,a.weights,a.limit)
 elif a.command=='verify':verify(a.out,list(SPECS))
 else:
  verify(a.out,list(SPECS))
  for d in ['scale','external_math']:
   st.OUT=a.out/d;st.analyse()
if __name__=='__main__':main()
