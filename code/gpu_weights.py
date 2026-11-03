"""Validate predownloaded pinned weights before offline CUDA inference."""
import hashlib, json
from pathlib import Path


def verify_local_weights(path, key, repo, revision):
    path = Path(path)
    manifest = json.loads((path/'verified_download.json').read_text())
    if (manifest.get('model_key'), manifest.get('repo'), manifest.get('revision')) != (key, repo, revision):
        raise ValueError('Local weight manifest differs from pinned model')
    names = set()
    hashes = {}
    for entry in manifest['entries']:
        name = entry['rfilename']
        if Path(name).is_absolute() or '..' in Path(name).parts or name in names:
            raise ValueError('Unsafe or duplicate local weight entry')
        names.add(name)
        fpath = path/name
        lfs = entry.get('lfs')
        if lfs:
            if fpath.stat().st_size != lfs['size']:
                raise ValueError('Local weight file size changed: '+name)
            h = hashlib.sha256()
            expected = lfs['sha256']
        else:
            h = hashlib.sha1()
            h.update(f'blob {fpath.stat().st_size}\0'.encode())
            expected = entry['blobId']
        with fpath.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('Local weight hash mismatch: '+name)
        if name.endswith('.safetensors'):
            if not lfs:
                raise ValueError('Tensor file has no upstream SHA256')
            hashes[name] = h.hexdigest()
    if not hashes or not {'config.json','tokenizer_config.json'}.issubset(names):
        raise ValueError('Incomplete local model manifest')
    print('Offline weight hashes verified:', key, flush=True)
    return path, hashes
