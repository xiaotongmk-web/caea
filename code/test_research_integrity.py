"""Tests for scientific leakage and corrupted/resumed GPU archives (no Torch)."""
import copy, json, tempfile, unittest
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
