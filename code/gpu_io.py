"""Strict, Torch-free validation of incremental GPU score archives."""
import json
import math


def read_validated_cache(path, requests):
    expected = {r['id']: r for r in requests}
    if len(expected) != len(requests):
        raise ValueError('Duplicate request IDs')
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f'Empty cache line {line_number}')
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid cache line {line_number}; preserve the file before repairing a partial last write') from exc
    validate_rows(rows, requests, complete=False)
    return rows


def validate_rows(rows, requests, complete=True):
    expected = {r['id']: r for r in requests}
    seen = set()
    for row in rows:
        rid = row['id']
        if rid not in expected or rid in seen:
            raise ValueError('Unexpected or duplicate GPU cache request: '+rid)
        seen.add(rid)
        req = expected[rid]
        if row['prompt_sha256'] != req['prompt_sha256']:
            raise ValueError('GPU cache prompt mismatch: '+rid)
        n = 5 if req['kind'] == 'audit' else 10
        scores, ids = row['logits'], row['choice_token_ids']
        if len(scores) != n or not all(math.isfinite(x) for x in scores):
            raise ValueError('Invalid candidate logits: '+rid)
        if len(ids) != n or len(set(ids)) != n or not all(isinstance(x, int) and x >= 0 for x in ids):
            raise ValueError('Invalid choice token IDs: '+rid)
        if not math.isfinite(row['candidate_mass']) or not 0 <= row['candidate_mass'] <= 1+1e-6:
            raise ValueError('Invalid candidate probability mass: '+rid)
        if not isinstance(row['prompt_tokens'], int) or not 0 < row['prompt_tokens'] < 512:
            raise ValueError('Invalid prompt token count: '+rid)
        if not math.isfinite(row['seconds']) or row['seconds'] < 0:
            raise ValueError('Invalid inference duration: '+rid)
        if not isinstance(row['selected_token'], int) or row['selected_token'] < 0:
            raise ValueError('Invalid selected token: '+rid)
        if row['selected_token'] in ids and scores[ids.index(row['selected_token'])] != max(scores):
            raise ValueError('Native argmax does not agree with candidate logits: '+rid)
        token_hash = row.get('prompt_token_ids_sha256', '')
        if len(token_hash) != 64 or any(c not in '0123456789abcdef' for c in token_hash):
            raise ValueError('Missing token fingerprint: '+rid)
    if complete and seen != set(expected):
        raise ValueError(f'Incomplete GPU archive: {len(seen)}/{len(expected)} requests')
    return len(seen)
