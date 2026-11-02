"""Download pinned public GGUF weights and verify upstream SHA-256 hashes.

Model weights are external dependencies, never redistributed with this study.
"""
from pathlib import Path
import argparse,hashlib,json,os,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor

MODELS={
 'qwen15':dict(repo='Qwen/Qwen2.5-1.5B-Instruct-GGUF',revision='91cad51170dc346986eccefdc2dd33a9da36ead9',
     files=['qwen2.5-1.5b-instruct-q4_k_m.gguf'],parameters_billion=1.54),
 'smol17':dict(repo='HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF',revision='2d4a76a30b4af41ecd395c35725ac11688d4cfe4',
     files=['smollm2-1.7b-instruct-q4_k_m.gguf'],parameters_billion=1.71),
 'qwen7':dict(repo='Qwen/Qwen2.5-7B-Instruct-GGUF',revision='bb5d59e06d9551d752d08b292a50eb208b07ab1f',
     files=['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf','qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf'],parameters_billion=7.61),
}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def download_file(task):
    spec,name,info,dest=task;path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
    expected=info['lfs']['oid'];size=info['size']
    if path.exists() and path.stat().st_size==size and sha256(path)==expected:
        return dict(file=name,bytes=size,sha256=expected,verified=True)
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(5):
        offset=part.stat().st_size if part.exists() else 0
        if offset==size:break
        url=f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{name}?download=true"
        headers={'Range':f'bytes={offset}-'} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                mode='ab' if offset and response.status==206 else 'wb'
                done=offset if mode=='ab' else 0;mark=done//(256*1024*1024)
                with part.open(mode) as f:
                    while True:
                        b=response.read(8*1024*1024)
                        if not b:break
                        f.write(b);done+=len(b)
                        if done//(256*1024*1024)>mark:
                            mark=done//(256*1024*1024);print(f'{name}: {done}/{size} bytes',flush=True)
            if part.stat().st_size==size:break
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            print(f'{name}: download attempt {attempt+1} failed ({type(exc).__name__})',flush=True)
    if not part.exists() or part.stat().st_size!=size:raise RuntimeError('Incomplete public model download: '+name)
    actual=sha256(part)
    if actual!=expected:raise RuntimeError('Model checksum mismatch: '+name)
    part.replace(path);print(name+': upstream SHA-256 verified',flush=True)
    return dict(file=name,bytes=size,sha256=actual,verified=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    ap.add_argument('--directory',type=Path,required=True);args=ap.parse_args()
    tasks=[];registry={}
    for key in args.models:
        spec=MODELS[key]
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{spec['repo']}/tree/{spec['revision']}?recursive=false",timeout=30) as response:
            metadata={r['path']:r for r in json.load(response)}
        registry[key]=dict(**spec,quantization='Q4_K_M',upstream_files=[metadata[n] for n in spec['files']])
        for name in spec['files']:tasks.append((spec,name,metadata[name],args.directory))
    args.directory.mkdir(parents=True,exist_ok=True)
    (args.directory/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=2) as pool:receipts=list(pool.map(download_file,tasks))
    (args.directory/'download_verification.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(dict(completed=True,files=len(receipts),total_bytes=sum(x['bytes']for x in receipts))))


if __name__=='__main__':main()
