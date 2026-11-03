"""Tests for scientific leakage and corrupted/resumed GPU archives (no Torch)."""
import copy, hashlib, json, tempfile, unittest
from pathlib import Path
from evidence_audit_diagnostics import solve_public_evidence
from gpu_io import validate_rows, read_validated_cache


class PublicEvidenceTests(unittest.TestCase):
    def test_evidence_updates_but_unsupported_stance_does_not(self):
        base = 'P(H)=50%. P(E|H)=75%. P(E|not H)=25%. E is observed.'
        self.assertAlmostEqual(solve_public_evidence(base), .75)
        for stance in ['My own guess is 10%', 'My own guess is 90%']:
            self.assertAlmostEqual(solve_public_evidence(base+stance), .75)
        extra = ' F is conditionally independent. P(F|H)=25%. P(F|not H)=75%.'
        self.assertAlmostEqual(solve_public_evidence(base+extra), .5)

    def test_paraphrase_and_missing_independence(self):
        text = 'Among all cases, 50% have H. The chance of seeing E is 75% for cases with H and 25% for cases without H.'
        self.assertAlmostEqual(solve_public_evidence(text), .75)
        with self.assertRaises(ValueError):
            solve_public_evidence(text+' P(F|H)=25%. P(F|not H)=75%.')
        with self.assertRaises(ValueError):
            solve_public_evidence('Unstructured question without the documented grammar')


class GPUPreparationTests(unittest.TestCase):
    def test_repeat_prepare_preserves_protocol_and_rejects_changes(self):
        import contextlib, io
        from gpu_study import prepare
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d)
            prepare(out)
            protocol = out/'scale'/'protocol.json'
            original = protocol.read_bytes()
            prepare(out)
            self.assertEqual(protocol.read_bytes(), original)
            changed = json.loads(original)
            changed['weight_dtype'] = 'float32'
            protocol.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError):
                prepare(out)


class LocalWeightTests(unittest.TestCase):
    def test_verified_weights_and_corruption(self):
        from gpu_weights import verify_local_weights
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entries = []
            for name, data in [('config.json', b'{}'), ('tokenizer_config.json', b'{}'),
                               ('model.safetensors', b'example tensor bytes')]:
                (root/name).write_bytes(data)
                entry = dict(rfilename=name)
                if name.endswith('.safetensors'):
                    entry['lfs'] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
                else:
                    entry['blobId'] = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
                entries.append(entry)
            manifest = root/'verified_download.json'
            manifest.write_text(json.dumps(dict(model_key='test', repo='test/repo',
                                                revision='pinned', entries=entries)))
            _, hashes = verify_local_weights(root, 'test', 'test/repo', 'pinned')
            self.assertEqual(set(hashes), {'model.safetensors'})
            with self.assertRaises(ValueError):
                verify_local_weights(root, 'test', 'test/repo', 'different revision')
            for name in ['model.safetensors', 'config.json']:
                original = (root/name).read_bytes()
                (root/name).write_bytes(bytes([original[0]^1])+original[1:])
                with self.assertRaises(ValueError):
                    verify_local_weights(root, 'test', 'test/repo', 'pinned')
                (root/name).write_bytes(original)
            altered = json.loads(manifest.read_text())
            altered['entries'] = entries[:-1]
            manifest.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                verify_local_weights(root, 'test', 'test/repo', 'pinned')


class GPUArchiveTests(unittest.TestCase):
    def setUp(self):
        self.requests = [dict(id='audit:x', kind='audit', prompt_sha256='a'*64)]
        self.row = dict(id='audit:x', prompt_sha256='a'*64, logits=[1.,2.,3.,4.,5.],
                        choice_token_ids=[10,11,12,13,14], candidate_mass=.9,
                        selected_token=14, prompt_tokens=64, seconds=.1,
                        prompt_token_ids_sha256='b'*64)

    def test_resume_and_complete(self):
        validate_rows([self.row], self.requests)
        tied = copy.deepcopy(self.row)
        tied['logits'] = [1., 2., 3., 5., 5.]
        validate_rows([tied], self.requests)
        validate_rows([], self.requests, complete=False)
        with self.assertRaises(ValueError):
            validate_rows([], self.requests)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'cache.jsonl'
            p.write_text(json.dumps(self.row)+'\n')
            self.assertEqual(read_validated_cache(p,self.requests), [self.row])
            p.write_text(json.dumps(self.row)+'\n{"id":')
            with self.assertRaises(ValueError):
                read_validated_cache(p,self.requests)

    def test_corruption_is_rejected(self):
        for field, value in [('prompt_sha256','wrong'), ('logits',[float('nan')]*5),
                             ('choice_token_ids',[10]*5), ('candidate_mass',1.1),
                             ('selected_token',10), ('prompt_tokens',512),
                             ('prompt_token_ids_sha256',''), ('seconds',-1)]:
            with self.subTest(field=field):
                row = copy.deepcopy(self.row); row[field] = value
                with self.assertRaises(ValueError):
                    validate_rows([row],self.requests)
        with self.assertRaises(ValueError):
            validate_rows([self.row,self.row],self.requests)


if __name__ == '__main__':
    unittest.main()
