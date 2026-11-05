import json
import tempfile
import unittest
from pathlib import Path

from experiments.manage import evaluate, inventory, write_new


class ExperimentTests(unittest.TestCase):
    def test_inventory_and_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "assets"
            root.mkdir()
            (root / "log.json").write_text("{}")
            result = inventory(dict(root=str(root), assets=[dict(path="log.json")]))
            self.assertEqual(len(result["assets"][0]["sha256"]), 64)
            (root / "escape").symlink_to(Path(d))
            with self.assertRaises(ValueError):
                inventory(dict(root=str(root), assets=[dict(path="escape")]))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "metrics.json"
            write_new(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_new(path, {"replacement": True})
            self.assertEqual(json.loads(path.read_text()), {"original": True})

    def test_flip_failure_and_duplicate(self):
        row = dict(source_id="s1", method="evidence_dpo", seed="42", condition="flip",
                   target_a=0.2, target_b=0.8, p_a=0.3, p_b=0.7,
                   status_a="ok", status_b="ok")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            failed = dict(row, source_id="s2", p_b=None, status_b="parse_failure")
            path.write_text("\n".join(json.dumps(r) for r in [row, failed]))
            group = evaluate(path)["groups"][0]
            self.assertEqual(group["source_pairs"], 2)
            self.assertEqual(group["valid_pairs"], 1)
            self.assertEqual(group["failed_responses"], 1)
            self.assertAlmostEqual(group["conditional_update_error"], 0.2)
            self.assertAlmostEqual(group["failure_penalized_squared_error"], 0.2575)
            self.assertEqual(group["conditional_direction_accuracy"], 1)
            path.write_text(json.dumps(row) + "\n" + json.dumps(row))
            with self.assertRaises(ValueError):
                evaluate(path)

    def test_invalid_probabilities_and_interventions(self):
        row = dict(source_id="s", method="base", seed="base", condition="stance",
                   target_a=0.5, target_b=0.5, p_a=0.3, p_b=0.7,
                   status_a="ok", status_b="ok")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            for change in [dict(p_a=float("nan")), dict(p_a=True), dict(target_b=0.8),
                           dict(status_a="refusal"), dict(condition="flip")]:
                path.write_text(json.dumps(dict(row, **change)))
                with self.assertRaises(ValueError):
                    evaluate(path)


if __name__ == "__main__":
    unittest.main()
