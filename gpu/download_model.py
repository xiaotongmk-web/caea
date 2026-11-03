"""Download pinned research weights without CUDA or Hugging Face HEAD requests."""
import argparse, fnmatch, hashlib, json, subprocess
from pathlib import Path

SPECS = {
    'qwen15': ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'),
    'smol17': ('HuggingFaceTB/SmolLM2-1.7B-Instruct', '31b70e2e869a7173562077fd711b654946d38674'),
    'qwen7': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
}
PATTERNS = ['*.json', '*.safetensors', '*.jinja', 'tokenizer.model', 'vocab.*', 'merges.txt']


def check(path, entry):
    if not path.is_file():
        return False
    lfs = entry.get('lfs')
    if lfs:
        if path.stat().st_size != lfs['size']:
            return False
        h = hashlib.sha256()
    else:
        h = hashlib.sha1()
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest() == (lfs['sha256'] if lfs else entry['blobId'])


def curl(url, output, large=False):
    args = ['curl', '--fail', '--location', '--retry', '2', '--retry-delay', '2',
            '--connect-timeout', '10', '--max-time', '1800' if large else '60',
            '--speed-limit', '1024', '--speed-time', '60', '--output', str(output)]
    if large:
        args += ['--continue-at', '-']
    subprocess.run(args + [url], check=True)


def download(key, root):
    repo, rev = SPECS[key]
    dest = root/key
    dest.mkdir(parents=True, exist_ok=True)
    pending = dest/'upstream_manifest.pending.json'
    curl(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true', pending)
    info = json.loads(pending.read_text())
    if info.get('sha') != rev:
        raise ValueError('Upstream revision differs from pinned research revision')
    entries = [e for e in info['siblings'] if any(fnmatch.fnmatch(e['rfilename'], p) for p in PATTERNS)]
    if not any(e['rfilename'].endswith('.safetensors') for e in entries):
        raise ValueError('No tensor files in upstream manifest')
    for entry in entries:
        name = entry['rfilename']
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe upstream path')
        target = dest/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if check(target, entry):
            print('VERIFIED existing:', name, flush=True)
            continue
        large = bool(entry.get('lfs'))
        part = target.with_name(target.name+'.download-part')
        route = 'resolve' if large else 'raw'
        print('Downloading:', name, flush=True)
        curl(f'https://huggingface.co/{repo}/{route}/{rev}/{name}', part, large)
        if not check(part, entry):
            raise ValueError('Upstream hash verification failed: '+name)
        part.replace(target)
        print('VERIFIED:', name, flush=True)
    manifest = dict(model_key=key, repo=repo, revision=rev, entries=entries)
    (dest/'verified_download.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('DOWNLOAD_VERIFIED', key, dest, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=SPECS, default='qwen15')
    p.add_argument('--weights', type=Path, default=Path('/root/autodl-tmp/caea-bf16/weights'))
    a = p.parse_args()
    download(a.model, a.weights)
