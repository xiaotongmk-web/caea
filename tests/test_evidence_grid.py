import hashlib
import unittest
from collections import defaultdict

from protocols.build_evidence_grid import build


class EvidenceGridTests(unittest.TestCase):
    def test_deterministic_factorial_and_scoring_separation(self):
        requests, labels = build(development=2, confirmation=3)
        self.assertEqual((requests, labels), build(development=2, confirmation=3))
        self.assertEqual(len(requests), 60)
        self.assertEqual(len({r['id'] for r in requests}), 60)
        groups = defaultdict(list)
        targets = {r['id']: r['target'] for r in labels}
        for r in requests:
            groups[r['source_id']].append(r)
            self.assertNotIn('target', r)
            self.assertEqual(r['prompt_sha256'], hashlib.sha256(r['prompt'].encode()).hexdigest())
        for rows in groups.values():
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({r['split'] for r in rows}), 1)
            for evidence in ('original', 'counterfactual'):
                subset = [r for r in rows if r['evidence'] == evidence]
                self.assertEqual(len({targets[r['id']] for r in subset}), 1)
            before = next(targets[r['id']] for r in rows if r['evidence'] == 'original')
            after = next(targets[r['id']] for r in rows if r['evidence'] == 'counterfactual')
            self.assertNotEqual(before, after)

    def test_evidence_source_disjointness(self):
        _, labels = build()
        splits = defaultdict(set)
        for row in labels:
            splits[(row['prior'], row['l1'], row['l0'])].add(row['split'])
        # Counterfactual swaps are the same source too. They may coincide with
        # another source's original triple, so canonicalize likelihood order.
        canonical = defaultdict(set)
        for row in labels:
            canonical[(row['prior'], min(row['l1'],row['l0']), max(row['l1'],row['l0']))].add(row['split'])
        self.assertTrue(all(len(x) == 1 for x in canonical.values()))

    def test_invalid_sizes(self):
        for a, b in [(0, 3), (3, 0), (5000, 1)]:
            with self.assertRaises(ValueError):
                build(development=a, confirmation=b)


if __name__ == '__main__':
    unittest.main()
